
# ============================================================
# config.py
# Project configuration
# ============================================================

import os

# ------------------------------------------------------------
# Dataset configuration
# ------------------------------------------------------------

BASE_PATH = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_PATH = os.path.join(
    BASE_PATH,
    "Dataset"
)

# ------------------------------------------------------------
# Audio / MFCC configuration
# ------------------------------------------------------------

N_MFCC = 40
MAX_LEN = 174

# ------------------------------------------------------------
# Target classes
# ------------------------------------------------------------

TARGET_CLASSES = [
    "dog_bark",
    "car_horn",
    "air_conditioner",
    "siren",
    "engine_idling"
]

NUM_CLASSES = len(TARGET_CLASSES)

# ------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------

TEST_SIZE = 0.20
RANDOM_STATE = 42

BATCH_SIZE = 32
EPOCHS = 30

# ------------------------------------------------------------
# Model / output directories
# ------------------------------------------------------------

MODEL_DIR = os.path.join(
    BASE_PATH,
    "models"
)

RESULTS_DIR = os.path.join(
    BASE_PATH,
    "results"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_model.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "final_model.keras"
)
