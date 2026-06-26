from pathlib import Path


# ============================
# Project Root
# ============================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================
# Data Paths
# ============================

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "customer_churn.csv"

PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "data.pkl"


# ============================
# Model Paths
# ============================

MODEL_DIR = PROJECT_ROOT / "models"

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

MODEL_PATH = MODEL_DIR / "model.pkl"


# ============================
# MLflow
# ============================

EXPERIMENT_NAME = "customer-churn"


# ============================
# Random Seed
# ============================

RANDOM_STATE = 42


# ============================
# Train/Test Split
# ============================

TEST_SIZE = 0.2


# ============================
# ML FLOW
# ============================

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

EXPERIMENT_NAME = "customer-churn"

REGISTERED_MODEL_NAME = "customer-churn-model"