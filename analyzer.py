"""
analyzer.py
Everything that happens after training: classify a photo, find where the damage is with
Grad-CAM, draw a box around it and estimate severity. The web app (app.py) uses this.

How the box is found (Grad-CAM, in plain words):
  The model's last feature layer is a 7x7 grid of "what I noticed here" signals.
  Grad-CAM asks: which grid cells pushed the model towards its answer the most?
  Those cells form a heatmap. The hottest area becomes the bounding box, and the
  share of the photo it covers becomes the severity estimate.
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import tensorflow as tf
import keras

import config

preprocess_input = keras.applications.mobilenet_v2.preprocess_input

# Box colours for each severity level (RGB)
SEVERITY_COLOURS = {
    "Low": (46, 139, 87),     # green
    "Medium": (230, 145, 0),  # amber
    "High": (200, 40, 30),    # red
}


def pretty_name(class_name):
    """'alligator_crack' -> 'Alligator Crack'"""
    return class_name.replace("_", " ").replace("-", " ").strip().title()


def severity_from_area(area_ratio):
    if area_ratio < config.SEVERITY_LOW_BELOW:
        return "Low"
    if area_ratio < config.SEVERITY_MEDIUM_BELOW:
        return "Medium"
    return "High"


class RoadDamageAnalyzer:
    def __init__(self, model_path=config.MODEL_PATH, class_names_path=config.CLASS_NAMES_PATH):
        self.model = keras.models.load_model(model_path)
        self.class_names = json.loads(class_names_path.read_text())

        # A second view of the same model that also returns the last feature layer
        last_conv = self.model.get_layer(config.LAST_CONV_LAYER)
        self.grad_model = keras.Model(self.model.inputs, [last_conv.output, self.model.outputs[0]])

    # ------------------------------------------------------------ steps
    def _to_batch(self, image_rgb):
        resized = cv2.resize(image_rgb, (config.IMG_SIZE, config.IMG_SIZE))
        batch = preprocess_input(resized.astype(np.float32))
        return tf.convert_to_tensor(batch[np.newaxis, ...])

    def _gradcam(self, batch):
        with tf.GradientTape() as tape:
            features, predictions = self.grad_model(batch, training=False)
            class_index = int(tf.argmax(predictions[0]))
            score = predictions[:, class_index]

        grads = tape.gradient(score, features)                 # how much each feature mattered
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))        # one importance value per feature
        heatmap = tf.reduce_sum(features[0] * weights, axis=-1)
        heatmap = tf.nn.relu(heatmap)                          # keep only positive evidence
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)    # scale to 0..1
        return predictions[0].numpy(), class_index, heatmap.numpy()

    @staticmethod
    def _box_from_heatmap(heatmap, width, height):
        """Stretch the 7x7 heatmap to photo size and box the largest hot region."""
        heat = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
        heat = np.clip(heat, 0, 1)
        mask = (heat >= config.HEATMAP_THRESHOLD).astype(np.uint8)

        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:  # nothing above threshold: fall back to the whole image
            return heat, (0, 0, width, height), 1.0

        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))  # label 0 is background
        x, y, w, h, area = stats[biggest]
        return heat, (int(x), int(y), int(w), int(h)), area / (width * height)

    # ------------------------------------------------------------ main entry point
    def analyze(self, image_rgb):
        """image_rgb: a NumPy array (height, width, 3) in RGB order."""
        height, width = image_rgb.shape[:2]
        probabilities, class_index, heatmap = self._gradcam(self._to_batch(image_rgb))
        heat, box, area_ratio = self._box_from_heatmap(heatmap, width, height)

        return {
            "class_name": self.class_names[class_index],
            "label": pretty_name(self.class_names[class_index]),
            "confidence": float(probabilities[class_index]),
            "probabilities": {pretty_name(n): float(p)
                              for n, p in zip(self.class_names, probabilities)},
            "box": box,                        # (x, y, width, height) in pixels
            "area_ratio": float(area_ratio),   # share of the photo covered by the damage
            "severity": severity_from_area(area_ratio),
            "heatmap": heat,                   # 0..1 values, same size as the photo
        }


# ---------------------------------------------------------------- drawing helpers
def draw_result(image_rgb, result):
    """Return a copy of the photo with the box and a label drawn on it."""
    out = image_rgb.copy()
    height, width = out.shape[:2]
    colour = SEVERITY_COLOURS[result["severity"]]
    thickness = max(2, round(min(height, width) / 150))
    x, y, w, h = result["box"]
    cv2.rectangle(out, (x, y), (x + w, y + h), colour, thickness)

    text = f"{result['label']} | {result['severity']} | {result['confidence']:.0%}"
    font_scale = max(0.5, min(height, width) / 700)
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    # Put the label just above the box; if the box touches the top, put it inside instead
    label_y = y - 8 if y - th - 12 > 0 else y + th + 12
    label_x = min(x, max(0, width - tw - 12))
    cv2.rectangle(out, (label_x, label_y - th - 8), (label_x + tw + 12, label_y + baseline),
                  colour, cv2.FILLED)
    cv2.putText(out, text, (label_x + 6, label_y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def overlay_heatmap(image_rgb, heatmap, strength=0.45):
    """Blend the Grad-CAM heatmap over the photo (red = where the model looked most)."""
    coloured = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 1 - strength, coloured, strength, 0)
