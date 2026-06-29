from datetime import datetime

import pandas as pd
from google.cloud import bigquery

from config.config import (
    BQ_PROJECT_ID,
    BQ_DATASET_ID,
    BQ_REFERENCE_TABLE_ID,
    BQ_PREDICTION_TABLE_ID,
)
from logger import get_logger

logger = get_logger(__name__)


def get_bigquery_client() -> bigquery.Client:
    """
    Creates and returns a BigQuery client.
    """
    return bigquery.Client(project=BQ_PROJECT_ID)


def get_production_table_id() -> str:
    """
    Returns the fully qualified BigQuery table ID.
    """
    return f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_PREDICTION_TABLE_ID}"


def get_reference_table_id() -> str:
    """
    Returns the fully qualified BigQuery table ID.
    """
    return f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_REFERENCE_TABLE_ID}"


def insert_prediction(
    input_data: dict,
    prediction: bool,
    probability: float,
    model_version: str,
) -> None:
    """
    Inserts a single prediction into BigQuery.
    """

    try:
        logger.info("Inserting prediction into BigQuery.")

        client = get_bigquery_client()
        table_id = get_production_table_id()

        row = input_data.copy()

        row["timestamp"] = datetime.utcnow().isoformat()
        row["prediction"] = prediction
        row["probability"] = probability
        row["model_version"] = model_version

        errors = client.insert_rows_json(
            table_id,
            [row],
        )

        if errors:
            logger.error(f"BigQuery insertion failed: {errors}")
            raise RuntimeError(errors)

        logger.info("Prediction inserted successfully.")

    except Exception:
        logger.exception("Failed to insert prediction into BigQuery.")
        raise


def fetch_predictions(limit: int = 10000) -> pd.DataFrame:
    """
    Fetches recent prediction logs from BigQuery.
    """

    try:
        logger.info("Fetching production predictions from BigQuery.")

        client = get_bigquery_client()
        table_id = get_production_table_id()

        query = f"""
        SELECT *
        FROM `{table_id}`
        ORDER BY timestamp DESC
        LIMIT {limit}
        """

        dataframe = client.query(query).to_dataframe()

        logger.info(
            f"Fetched {len(dataframe)} prediction records from BigQuery."
        )

        return dataframe

    except Exception:
        logger.exception("Failed to fetch prediction logs.")
        raise


def fetch_reference_data() -> pd.DataFrame:
    """
    Fetches the reference dataset from BigQuery.
    """

    try:
        logger.info("Fetching reference dataset from BigQuery.")

        client = get_bigquery_client()

        table_id = get_reference_table_id()

        query = f"""
        SELECT *
        FROM `{table_id}`
        """

        dataframe = client.query(query).to_dataframe()

        logger.info(
            f"Fetched {len(dataframe)} reference records from BigQuery."
        )

        return dataframe

    except Exception:
        logger.exception(
            "Failed to fetch reference dataset."
        )
        raise
