"""
config.py
All project settings in one place.
"""
from pathlib import Path

# Folders and files
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODEL_PATH = MODEL_DIR / "road_damage_mobilenetv2.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

# Data split
SEED = 42                          # same split every run
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train, validation, test
MAX_SIDE = 512                     # longest side of saved images, in pixels

# Augmentation (training images only)
AUG_COPIES_PER_IMAGE = 4
ROTATION_DEGREES = 15
TILT_STRENGTH = 0.12
BRIGHTNESS_RANGE = (0.8, 1.2)
HORIZONTAL_FLIP = True
VERTICAL_FLIP = False

# Training
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS_HEAD = 100                  # stage 1: new classifier layer only
EPOCHS_FINE_TUNE = 100             # stage 2: fine-tune the top of MobileNetV2
EARLY_STOP_PATIENCE = 10           # stop after this many epochs without improvement
LR_HEAD = 1e-3
LR_FINE_TUNE = 1e-5
FINE_TUNE_LAYERS = 30

# Grad-CAM box and severity
LAST_CONV_LAYER = "out_relu"
HEATMAP_THRESHOLD = 0.5
SEVERITY_LOW_BELOW = 0.15          # damage covers < 15% of photo -> Low
SEVERITY_MEDIUM_BELOW = 0.35       # 15-35% -> Medium, above -> High

# Direction check: crack along the centre line = longitudinal, across it = transverse
USE_DIRECTION_RULE = True
PARALLEL_MAX_ANGLE = 35
CROSSING_MIN_ANGLE = 55
LANE_MIN_LENGTH = 0.08             # share of the photo's diagonal
LANE_MIN_ELONGATION = 4            # length-to-width ratio of a painted line
CRACK_MIN_LENGTH = 0.15            # share of the damage box's diagonal
CRACK_MIN_CONSISTENCY = 0.5        # how much crack pieces must agree on direction (0-1)
ASSUME_ROAD_RUNS_UP_DOWN = False   # used when no centre line is found