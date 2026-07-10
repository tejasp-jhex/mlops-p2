from datetime import datetime

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

MODEL_NAME = "customer-churn-model"

configure_mlflow()


def main():

    df = load_retraining_data()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    ) = preprocess(df)

    model = train_model(
        X_train,
        y_train,
    )

    metrics, predictions = evaluate_model(
        model,
        X_test,
        y_test,
    )

    candidate = CandidateModel(
        model=model,
        preprocessor=preprocessor,
        metrics=metrics,
        predictions=predictions,
        trained_at=datetime.utcnow(),
        training_rows=len(X_train),
    )

    with start_run():

        log_parameters(
            {
                "algorithm": "XGBoost",
                "training_rows": candidate.training_rows,
            }
        )

        log_metrics(candidate.metrics)

        log_candidate(candidate)

        candidate.run_id = get_run_id()

        production_metrics = get_production_metrics(
            MODEL_NAME,
        )

        if should_promote(
            candidate.metrics,
            production_metrics,
        ):
            register_candidate(
                candidate,
                MODEL_NAME,
            )
        else:
            print("Candidate not promoted.")


if __name__ == "__main__":
    main()