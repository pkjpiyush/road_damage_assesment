"""
prepare_data.py
Step 1 of the project. Run it with:  python prepare_data.py

What it does:
  1. Reads your images from data/raw/<class_name>/
  2. Splits each class into train (70%), validation (15%) and test (15%)
  3. Creates augmented copies of the TRAINING images only
     (tilt, rotate, flip, brightness 50-120%)

Validation and test images are never augmented, so your accuracy numbers stay honest.
"""
import shutil

import cv2
import numpy as np

import config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------- reading and saving
def read_image(path):
    """Read an image safely (also works with spaces or non-English characters in the path)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)  # returns None if the file is not an image


def save_image(path, image):
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if ok:
        buffer.tofile(str(path))


def shrink(image, max_side=config.MAX_SIDE):
    """Make big photos smaller (keeps the shape) so everything runs faster."""
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image


# ---------------------------------------------------------------- augmentations
def flip(image, rng):
    if config.HORIZONTAL_FLIP and rng.random() < 0.5:
        image = cv2.flip(image, 1)  # left <-> right
    if config.VERTICAL_FLIP and rng.random() < 0.5:
        image = cv2.flip(image, 0)  # top <-> bottom
    return image


def rotate(image, rng):
    h, w = image.shape[:2]
    angle = rng.uniform(-config.ROTATION_DEGREES, config.ROTATION_DEGREES)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    # BORDER_REFLECT fills the corners with mirrored road instead of black triangles
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def tilt(image, rng):
    """Pretend the camera was tilted: pull each corner inward by a random amount."""
    h, w = image.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    inward = np.float32([[1, 1], [-1, 1], [-1, -1], [1, -1]])  # direction towards the centre
    shift = rng.uniform(0, config.TILT_STRENGTH, size=(4, 2)) * np.float32([w, h])
    dst = (src + inward * shift).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def change_brightness(image, rng):
    low, high = config.BRIGHTNESS_RANGE
    factor = rng.uniform(low, high)  # 0.5 = half as bright, 1.2 = 20% brighter
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def augment(image, rng):
    """Apply a random mix of the four augmentations. Brightness always changes,
    so every copy is different from the original."""
    image = flip(image, rng)
    if rng.random() < 0.7:
        image = rotate(image, rng)
    if rng.random() < 0.7:
        image = tilt(image, rng)
    return change_brightness(image, rng)


# ---------------------------------------------------------------- main
def split_files(files, rng):
    files = list(files)
    rng.shuffle(files)
    n = len(files)
    n_val = max(1, round(n * config.SPLIT_RATIOS[1]))
    n_test = max(1, round(n * config.SPLIT_RATIOS[2]))
    return {
        "test": files[:n_test],
        "val": files[n_test:n_test + n_val],
        "train": files[n_test + n_val:],
    }


def main():
    class_dirs = sorted(d for d in config.RAW_DIR.iterdir() if d.is_dir())
    if len(class_dirs) < 2:
        raise SystemExit(
            f"Found {len(class_dirs)} class folder(s) in {config.RAW_DIR}.\n"
            "Put your images in one folder per class, for example data/raw/pothole/"
        )

    # Start fresh each time so re-running never creates duplicates
    if config.PROCESSED_DIR.exists():
        shutil.rmtree(config.PROCESSED_DIR)

    rng = np.random.default_rng(config.SEED)
    print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}\n")

    for class_dir in class_dirs:
        files = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        splits = split_files(files, rng)
        counts = {}

        for split_name, split_files_list in splits.items():
            out_dir = config.PROCESSED_DIR / split_name / class_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            saved = 0

            for path in split_files_list:
                image = read_image(path)
                if image is None:
                    print(f"  Skipped (not a readable image): {path.name}")
                    continue
                image = shrink(image)
                save_image(out_dir / f"{path.stem}.jpg", image)
                saved += 1

                if split_name == "train":  # augment training images only
                    for i in range(1, config.AUG_COPIES_PER_IMAGE + 1):
                        save_image(out_dir / f"{path.stem}_aug{i}.jpg", augment(image, rng))
                        saved += 1

            counts[split_name] = saved

        print(f"{class_dir.name:>20}:  train {counts['train']:>4}  |  "
              f"val {counts['val']:>3}  |  test {counts['test']:>3}")

    print(f"\nDone! Processed images are in {config.PROCESSED_DIR}")
    print("Next step:  python train.py")


if __name__ == "__main__":
    main()
