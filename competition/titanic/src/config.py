"""
Configuration file for Titanic Machine Learning project.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"

TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "gender_submission.csv"

# Global Constants
RANDOM_SEED = 42
N_SPLITS = 5
TARGET_COL = "Survived"
ID_COL = "PassengerId"
