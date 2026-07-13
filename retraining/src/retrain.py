from datetime import datetime, UTC

from retraining.src.candidate_model import CandidateModel
from retraining.src.data_loader import load_retraining_data
from retraining.src.evaluate import evaluate_model
from mlflow_utils import (
    configure_mlflow,
    get_run_id,
    log_candidate,
    log_metrics,
    log_parameters,
    start_run,
)
from retraining.src.preprocessing import preprocess
from retraining.src.registry import (
    get_production_metrics,
    register_candidate,
    should_promote,
)
from retraining.src.train import train_model
from utils import load_params
from retraining.storage.bigquery import insert_retraining_history

from config.config import REGISTERED_MODEL_NAME
from logger import get_logger

logger = get_logger(__name__)


def main():

    try:

        logger.info("Starting retraining pipeline.")

        # -----------------------------------------
        # Load Data
        # -----------------------------------------

        df = load_retraining_data()

        # -----------------------------------------
        # Preprocess
        # -----------------------------------------

        (
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor,
        ) = preprocess(df)

        # -----------------------------------------
        # Train
        # -----------------------------------------

        model = train_model(
            X_train,
            y_train,
        )

        # -----------------------------------------
        # Evaluate
        # -----------------------------------------

        metrics, predictions = evaluate_model(
            model,
            X_test,
            y_test,
        )

        # -----------------------------------------
        # Candidate
        # -----------------------------------------

        candidate = CandidateModel(
            model=model,
            preprocessor=preprocessor,
            metrics=metrics,
            predictions=predictions,
            trained_at=datetime.now(UTC),
            training_rows=len(X_train),
        )

        # -----------------------------------------
        # MLflow
        # -----------------------------------------

        configure_mlflow()

        params = load_params()

        with start_run():

            log_parameters(
                {
                    "algorithm": "XGBoost",
                    "n_estimators": params["model"]["n_estimators"],
                    "learning_rate": params["model"]["learning_rate"],
                    "max_depth": params["model"]["max_depth"],
                    "eval_metric": params["model"]["eval_metric"],
                    "training_rows": candidate.training_rows,
                }
            )

            log_metrics(candidate.metrics)

            log_candidate(candidate)

            candidate.run_id = get_run_id()

            # -----------------------------------------
            # Compare
            # -----------------------------------------

            production_metrics = get_production_metrics(
                REGISTERED_MODEL_NAME,
            )

            if should_promote(
                candidate.metrics,
                production_metrics,
            ):

                version = register_candidate(
                    candidate,
                    REGISTERED_MODEL_NAME,
                )

                candidate.promoted = True
                candidate.model_version = str(version)

                logger.info(
                    f"Candidate promoted to Production (Version {version})."
                )

            else:

                candidate.promoted = False

                logger.info(
                    "Candidate was not promoted."
                )

        # -----------------------------------------
        # Save History
        # -----------------------------------------

        insert_retraining_history(candidate)

        logger.info(
            "Retraining pipeline completed successfully."
        )

    except Exception:

        logger.exception(
            "Retraining pipeline failed."
        )

        raise


if __name__ == "__main__":
    main()