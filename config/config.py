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


# ============================
# DRIFT DETECTION
# ============================

REFERENCE_DATA_PATH = "data/raw/customer_churn.csv"

PRODUCTION_DATA_PATH = "monitoring/production_logs/predictions.csv"

DRIFT_REPORT_PATH = "monitoring/reports/drift_report.html"

DRIFT_RESULT_PATH = "monitoring/reports/drift_result.json"


# ============================
# GCS STORAGE
# ============================

GCS_BUCKET_NAME = "your-bucket-name"

PREDICTIONS_BLOB = "production_logs/predictions.csv"

DRIFT_REPORT_BLOB = "reports/drift_report.html"

DRIFT_RESULT_BLOB = "reports/drift_result.json"

# ------------------------
# BigQuery Configuration
# ------------------------

BQ_PROJECT_ID = "women-safety-by-pioneers"
BQ_DATASET_ID = "customer_churn_mlops"
BQ_TABLE_ID = "prediction_logs"
BQ_REFERENCE_TABLE_ID = "reference_data"
BQ_PREDICTION_TABLE_ID = "prediction_logs"

# ------------------------
# GCS Configuration
# ------------------------

GCS_BUCKET_NAME = "tejas-mlops-storage"
DRIFT_REPORT_BLOB = "reports/drift_report.html"
DRIFT_RESULT_BLOB = "reports/drift_result.json"

MODEL_VERSION = "v1"