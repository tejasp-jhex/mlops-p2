from uuid import uuid4
from datetime import datetime, timezone

from google.cloud import bigquery

from config.config import (
    BQ_PROJECT_ID,
    BQ_DATASET_ID,
    BQ_RETRAINING_HISTORY_TABLE,
)

from logger import get_logger

logger = get_logger(__name__)

client = bigquery.Client(
    project=BQ_PROJECT_ID,
)


def get_table_id():

    return (
        f"{BQ_PROJECT_ID}."
        f"{BQ_DATASET_ID}."
        f"{BQ_RETRAINING_HISTORY_TABLE}"
    )


def insert_retraining_history(
    candidate,
    trigger="drift_detection",
    status="SUCCESS",
):

    logger.info(
        "Saving retraining history to BigQuery."
    )

    row = {
        "retraining_id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": candidate.run_id,
        "model_version": candidate.model_version,
        "training_rows": candidate.training_rows,
        "accuracy": candidate.metrics["accuracy"],
        "precision": candidate.metrics["precision"],
        "recall": candidate.metrics["recall"],
        "f1_score": candidate.metrics["f1_score"],
        "promoted": candidate.promoted,
        "trigger": trigger,
        "status": status,
    }

    errors = client.insert_rows_json(
        get_table_id(),
        [row],
    )

    if errors:
        logger.error(errors)
        raise RuntimeError(
            "Failed to insert retraining history."
        )

    logger.info(
        "Retraining history saved successfully."
    )