from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from logger import get_logger

logger = get_logger(__name__)


def evaluate_model(
    model,
    X_test,
    y_test,
):
    """
    Evaluate the trained model.
    """

    logger.info("Evaluating candidate model.")

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
        ),
        "recall": recall_score(
            y_test,
            predictions,
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
        ),
    }

    logger.info(
        "Evaluation completed."
    )

    logger.info(
        f"Metrics: {metrics}"
    )

    return metrics, predictions