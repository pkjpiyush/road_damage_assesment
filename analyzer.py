"""
analyzer.py
Classifies a road photo, finds the damage with Grad-CAM and estimates its severity.
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import keras
import numpy as np
import tensorflow as tf

import config

COLOURS = {"Low": (46, 139, 87), "Medium": (230, 145, 0), "High": (200, 40, 30)}


def pretty_name(name):
    return name.replace("_", " ").title()


def get_severity(area):
    if area < config.SEVERITY_LOW_BELOW:
        return "Low"
    if area < config.SEVERITY_MEDIUM_BELOW:
        return "Medium"
    return "High"


def angle_between(a, b):
    """0 = parallel, 90 = crossing like a +."""
    diff = abs(a - b) % 180
    return min(diff, 180 - diff)


def find_lines(mask, min_length, min_ratio):
    """Find long, thin shapes in a mask. Returns (angle, length, centre, direction) for each."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    lines = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < 15 or np.hypot(w, h) < min_length:
            continue
        ys, xs = np.nonzero(labels[y:y + h, x:x + w] == i)
        points = np.column_stack([xs + x, ys + y]).astype(float)
        values, vectors = np.linalg.eigh(np.cov(points, rowvar=False))
        length = np.sqrt(12 * max(values[1], 1e-6))
        width = np.sqrt(12 * max(values[0], 1e-6))
        if length >= min_length and length / width >= min_ratio:
            direction = vectors[:, 1]
            angle = np.degrees(np.arctan2(direction[1], direction[0])) % 180
            lines.append((angle, length, points.mean(axis=0), direction))
    return lines


def find_centre_line(image):
    """Find a painted yellow (or else white) line. Returns (angle, segment, mask) or None."""
    diagonal = np.hypot(*image.shape[:2])
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    colour_ranges = [((15, 80, 120), (40, 255, 255)),  # yellow
                     ((0, 0, 200), (179, 45, 255))]    # white

    for low, high in colour_ranges:
        mask = cv2.morphologyEx(cv2.inRange(hsv, low, high), cv2.MORPH_OPEN,
                                np.ones((3, 3), np.uint8))
        lines = find_lines(mask, config.LANE_MIN_LENGTH * diagonal, config.LANE_MIN_ELONGATION)
        if lines:
            angle, _, centre, direction = max(lines, key=lambda line: line[1])
            start, end = centre - direction * diagonal, centre + direction * diagonal
            segment = tuple(int(v) for v in (*start, *end))
            return angle, segment, mask
    return None


def find_crack_angle(image, box, paint_mask=None):
    """Main direction of the crack inside the box, or None if it's unclear."""
    x, y, w, h = box
    x0, y0 = max(0, x - w // 7), max(0, y - h // 7)
    x1, y1 = x + w + w // 7, y + h + h // 7

    gray = cv2.medianBlur(cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY), 3)
    size = max(9, (min(gray.shape) // 15) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    dark_lines = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)  # cracks show up here
    _, mask = cv2.threshold(dark_lines, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if paint_mask is not None:  # ignore the edges of the painted line
        mask[cv2.dilate(paint_mask[y0:y1, x0:x1], np.ones((9, 9), np.uint8)) > 0] = 0

    lines = find_lines(mask, config.CRACK_MIN_LENGTH * np.hypot(*gray.shape), 3)
    if not lines:
        return None

    # Length-weighted average direction (angles doubled so 1° and 179° count as the same)
    angles = np.radians([2 * line[0] for line in lines])
    lengths = np.array([line[1] for line in lines])
    c, s = np.sum(lengths * np.cos(angles)), np.sum(lengths * np.sin(angles))
    if np.hypot(c, s) / lengths.sum() < config.CRACK_MIN_CONSISTENCY:
        return None
    return np.degrees(np.arctan2(s, c)) / 2 % 180


class RoadDamageAnalyzer:
    def __init__(self):
        self.model = keras.models.load_model(config.MODEL_PATH)
        self.class_names = json.loads(config.CLASS_NAMES_PATH.read_text())
        last_layer = self.model.get_layer(config.LAST_CONV_LAYER)
        self.grad_model = keras.Model(self.model.inputs,
                                      [last_layer.output, self.model.outputs[0]])

    def predict(self, image):
        """Predicted class index and a 0-1 Grad-CAM heatmap."""
        resized = cv2.resize(image, (config.IMG_SIZE, config.IMG_SIZE)).astype(np.float32)
        batch = tf.convert_to_tensor(keras.applications.mobilenet_v2.preprocess_input(resized)[None])

        with tf.GradientTape() as tape:
            features, predictions = self.grad_model(batch, training=False)
            index = int(tf.argmax(predictions[0]))
            score = predictions[:, index]

        weights = tf.reduce_mean(tape.gradient(score, features), axis=(0, 1, 2))
        heatmap = tf.nn.relu(tf.reduce_sum(features[0] * weights, axis=-1)).numpy()
        return index, heatmap / (heatmap.max() + 1e-8)

    @staticmethod
    def find_box(heatmap, width, height):
        """Box around the largest hot area, plus the share of the photo it covers."""
        heat = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
        mask = (heat >= config.HEATMAP_THRESHOLD).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        if count <= 1:
            return (0, 0, width, height), 1.0
        x, y, w, h, area = stats[1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])]
        return (int(x), int(y), int(w), int(h)), area / (width * height)

    def check_direction(self, image, box, predicted):
        """Centre-line rule: along the line = longitudinal, across it = transverse."""
        longitudinal = next((n for n in self.class_names if "longitudinal" in n.lower()), None)
        transverse = next((n for n in self.class_names if "transverse" in n.lower()), None)
        if None in (longitudinal, transverse) or predicted not in (longitudinal, transverse):
            return None

        line = find_centre_line(image)
        if line:
            road_angle, segment, paint_mask = line
        elif config.ASSUME_ROAD_RUNS_UP_DOWN:
            road_angle, segment, paint_mask = 90, None, None
        else:
            return {"status": "no_line", "segment": None}

        crack_angle = find_crack_angle(image, box, paint_mask)
        if crack_angle is None:
            return {"status": "unclear", "segment": segment}

        diff = angle_between(crack_angle, road_angle)
        if diff <= config.PARALLEL_MAX_ANGLE:
            decision = longitudinal
        elif diff >= config.CROSSING_MIN_ANGLE:
            decision = transverse
        else:
            return {"status": "unclear", "segment": segment}

        status = "confirmed" if decision == predicted else "changed"
        return {"status": status, "decision": decision, "segment": segment}

    def analyze(self, image):
        """image: RGB NumPy array. Returns the damage type, severity and box."""
        height, width = image.shape[:2]
        index, heatmap = self.predict(image)
        box, area = self.find_box(heatmap, width, height)
        predicted = self.class_names[index]

        check = self.check_direction(image, box, predicted) if config.USE_DIRECTION_RULE else None
        final = (check or {}).get("decision") or predicted

        return {
            "class_name": final,
            "label": pretty_name(final),
            "model_class_name": predicted,
            "severity": get_severity(area),
            "box": box,
            "direction_check": check,
        }


def draw_result(image, result):
    """Draw the damage box, its label and the centre line (if one was used)."""
    out = image.copy()
    height, width = out.shape[:2]
    colour = COLOURS[result["severity"]]
    thickness = max(2, min(height, width) // 150)
    x, y, w, h = result["box"]
    cv2.rectangle(out, (x, y), (x + w, y + h), colour, thickness)

    segment = (result["direction_check"] or {}).get("segment")
    if segment:
        cv2.line(out, segment[:2], segment[2:], (30, 110, 230), thickness)

    text = f"{result['label']} | {result['severity']}"
    scale = max(0.5, min(height, width) / 700)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    ty = y - 8 if y > th + 12 else y + th + 12  # above the box, or inside if there's no room
    tx = min(x, max(0, width - tw - 12))
    cv2.rectangle(out, (tx, ty - th - 8), (tx + tw + 12, ty + 6), colour, cv2.FILLED)
    cv2.putText(out, text, (tx + 6, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out