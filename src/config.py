"""
Central configuration for the crop disease detection project.
Change these values to scale from the demo subset up to the full dataset.
"""
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# Where training images live, organized as data/raw/<ClassName>/*.jpg
DATA_DIR = os.environ.get("CROP_DATA_DIR", os.path.join(PROJECT_ROOT, "data", "raw"))

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_PATH = os.path.join(MODEL_DIR, "crop_disease_model.keras")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint_latest.keras")
TRAIN_STATE_PATH = os.path.join(MODEL_DIR, "train_state.json")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

# Image / training hyperparameters
IMG_SIZE = 128          # px, square. Smaller = faster on CPU, larger = more detail.
BATCH_SIZE = 32
EPOCHS = 38
VAL_SPLIT = 0.2
SEED = 42
LEARNING_RATE = 4e-4

# Set True once you've scaled to the full 38-class dataset and want to squeeze
# out extra accuracy by unfreezing the top of the backbone for a few epochs.
FINE_TUNE = False
FINE_TUNE_AT_LAYER = 100   # unfreeze from this layer index onward
FINE_TUNE_EPOCHS = 5
FINE_TUNE_LR = 1e-5
