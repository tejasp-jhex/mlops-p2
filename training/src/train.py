import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from training.src.evaluate import evaluate_model
from utils import load_params
from training.src.model_promotion import should_register_model
from training.src.logger import get_logger

from config.config import (
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    MODEL_DIR,
)

from training.src.mlflow_utils import (
    configure_mlflow,
    start_run,
    log_parameters,
    log_metrics,
    log_model,
    register_model
)

logger = get_logger(__name__)

configure_mlflow()

def load_processed_data():
    return joblib.load(PROCESSED_DATA_PATH)

def train_model(X_train, y_train, params):
    # model = XGBClassifier(
    # random_state=params["random_state"],
    # n_estimators=params["model"]["n_estimators"],
    # learning_rate=params["model"]["learning_rate"],
    # max_depth=params["model"]["max_depth"],
    # eval_metric=params["model"]["eval_metric"],
    # )

    model = RandomForestClassifier(
        random_state=params["random_state"],
        n_estimators=params["model"]["n_estimators"],
        max_depth=params["model"]["max_depth"],
    )
     
    model.fit(X_train, y_train)

    return model

def save_model(model):

    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(
        model,
        MODEL_PATH,
    )


def main():

    params = load_params()

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = load_processed_data()

    configure_mlflow()

    with start_run() as run:
        model = train_model(
            X_train,
            y_train,
            params
        )

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        

        save_model(model)
        log_parameters(params)
        log_metrics(metrics)
        log_model(model)

        if should_register_model(metrics, params):
            register_model(run.info.run_id)
            logger.info("Model registered successfully.")
        else:
            logger.info("Model did not meet the promotion criteria.")

    logger.info("Training completed successfully.")

    logger.info(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()