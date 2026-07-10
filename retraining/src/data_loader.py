from google.cloud import bigquery
import pandas as pd

from config.config import (
    BQ_PROJECT_ID,
    BQ_DATASET_ID,
    BQ_RETRAINING_TABLE_ID,
)
from logger import get_logger

logger = get_logger(__name__)


def get_bigquery_client() -> bigquery.Client:
    """
    Returns an authenticated BigQuery client.
    """

    return bigquery.Client(
        project=BQ_PROJECT_ID,
    )


def get_table_id(table_id: str) -> str:
    """
    Builds a fully-qualified BigQuery table ID.
    """

    return (
        f"{BQ_PROJECT_ID}."
        f"{BQ_DATASET_ID}."
        f"{table_id}"
    )


def load_retraining_data() -> pd.DataFrame:
    """
    Loads the retraining dataset from BigQuery.
    """

    try:

        logger.info(
            "Loading retraining dataset from BigQuery."
        )

        client = get_bigquery_client()

        table_id = get_table_id(
            BQ_RETRAINING_TABLE_ID,
        )

        query = f"""
        SELECT *
        FROM `{table_id}`
        """

        dataframe = client.query(
            query
        ).to_dataframe()

        if dataframe.empty:
            raise ValueError(
                "Retraining dataset is empty."
            )

        logger.info(
            f"Loaded {len(dataframe)} rows."
        )

        logger.info(
            f"Columns: {list(dataframe.columns)}"
        )

        return dataframe

    except Exception:

        logger.exception(
            "Failed to load retraining dataset."
        )

        raise