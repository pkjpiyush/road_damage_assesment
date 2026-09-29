"""
config.py
All project settings live here, so you only ever change values in one place.
"""
from pathlib import Path

# ---------------------------------------------------------------- folders
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"              # your original images, one folder per class
PROCESSED_DIR = ROOT / "data" / "processed"  # train / val / test splits (created for you)
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_PATH = MODEL_DIR / "road_damage_mobilenetv2.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

# ---------------------------------------------------------------- data split
SEED = 42                          # fixed seed = same split every time you run
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train, validation, test
MAX_SIDE = 512                     # images are shrunk so the longest side is 512 px

# ---------------------------------------------------------------- augmentation
AUG_COPIES_PER_IMAGE = 4           # 70 training images x (1 + 4) = 350 per class
ROTATION_DEGREES = 15              # rotate between -15 and +15 degrees
TILT_STRENGTH = 0.12               # how far corners move for the "camera tilt" effect
BRIGHTNESS_RANGE = (0.8, 1.2)      # 80% to 120% of the original brightness
HORIZONTAL_FLIP = True
VERTICAL_FLIP = False              # turn on only if your photos are taken from straight above

# ---------------------------------------------------------------- training
IMG_SIZE = 224                     # MobileNetV2's native input size
BATCH_SIZE = 16
EPOCHS_HEAD = 100                  # stage 1: train only the new classifier layer (maximum)
EPOCHS_FINE_TUNE = 100             # stage 2: gently retrain the top of MobileNetV2 (maximum)
EARLY_STOP_PATIENCE = 10           # stop a stage if validation loss hasn't improved for this many epochs
LR_HEAD = 1e-3
LR_FINE_TUNE = 1e-5
FINE_TUNE_LAYERS = 30              # how many of MobileNetV2's last layers to unfreeze

# ---------------------------------------------------------------- Grad-CAM, box and severity
LAST_CONV_LAYER = "out_relu"       # MobileNetV2's final feature layer
HEATMAP_THRESHOLD = 0.5            # heatmap values above this count as "damage"

# Severity is based on how much of the photo the damaged area covers.
# These are starting points: test on your own photos and adjust.
SEVERITY_LOW_BELOW = 0.15          # covers less than 15% of the image  -> Low
SEVERITY_MEDIUM_BELOW = 0.35       # covers 15% to 35%                   -> Medium
                                   # covers more than 35%                -> High

LOW_CONFIDENCE = 0.50              # below this, the app warns that the model is unsure

# ---------------------------------------------------------------- direction check (centre-line rule)
# When the model says "longitudinal" or "transverse", the app looks for a painted white/yellow
# centre line and compares the crack's direction with it:
#   crack roughly parallel to the line  -> longitudinal
#   crack crossing the line (like a +)  -> transverse
USE_DIRECTION_RULE = True
PARALLEL_MAX_ANGLE = 35            # crack within 35 degrees of the line       -> longitudinal
CROSSING_MIN_ANGLE = 55            # crack more than 55 degrees from the line  -> transverse
                                   # in between: too unclear, keep the model's answer
LANE_MIN_LENGTH = 0.08             # a painted line piece must be at least 8% of the photo's diagonal
LANE_MIN_ELONGATION = 4            # ...and at least 4 times longer than it is wide
CRACK_MIN_LENGTH = 0.15            # a crack piece must be at least 15% of the damage box's diagonal
CRACK_MIN_CONSISTENCY = 0.5        # 0 to 1: how much the crack pieces must agree on one direction
ASSUME_ROAD_RUNS_UP_DOWN = False   # if no centre line is found, treat "up the photo" as the road
                                   # direction (useful only if all photos face along the road)