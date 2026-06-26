import mlflow
import mlflow.sklearn
from config.config import REGISTERED_MODEL_NAME

from config.config import (
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)

def configure_mlflow():
    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

def start_run():
    return mlflow.start_run()

def log_parameters(params: dict):
    mlflow.log_params(flatten_params(params))

def log_metrics(metrics: dict):
    mlflow.log_metrics(metrics)

def log_model(model):
    mlflow.xgboost.log_model(
        xgb_model=model,
        name="model"
    )

def register_model(run_id):
    model_uri = f"runs:/{run_id}/model"

    mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME
    )

def flatten_params(params):
    flat = {}
    for section, value in params.items():
        if isinstance(value, dict):
            for key, val in value.items():
                flat[f"{section}.{key}"] = val
        else:
            flat[section] = value

    return flat

def end_run():
    mlflow.end_run()