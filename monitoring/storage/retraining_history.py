from datetime import datetime, timezone

from monitoring.storage.bigquery import get_bigquery_client, get_history_table_id

from config.config import (
    RETRAINING_COOLDOWN_HOURS,
)

from logger import get_logger

logger = get_logger(__name__)

client = get_bigquery_client()


def get_last_retraining_timestamp():

    logger.info(
        "Fetching latest successful retraining timestamp."
    )

    table_id = get_history_table_id()

    query = f"""
        SELECT timestamp
        FROM `{table_id}`
        WHERE status = 'SUCCESS'
        ORDER BY timestamp DESC
        LIMIT 1
    """

    result = client.query(query).result()

    rows = list(result)

    if not rows:

        logger.info(
            "No previous retraining history found."
        )

        return None

    timestamp = rows[0]["timestamp"]

    logger.info(
        f"Last retraining timestamp: {timestamp}"
    )

    return timestamp


def can_trigger_retraining():

    last_timestamp = get_last_retraining_timestamp()

    if last_timestamp is None:

        logger.info(
            "Retraining is allowed."
        )

        return True

    now = datetime.now(
        timezone.utc,
    )

    hours_since_last_training = (
        now - last_timestamp
    ).total_seconds() / 3600

    logger.info(
        f"Hours since last retraining: {hours_since_last_training:.2f}"
    )

    if hours_since_last_training >= RETRAINING_COOLDOWN_HOURS:

        logger.info(
            "Cooldown satisfied."
        )

        return True

    logger.info(
        "Cooldown not satisfied."
    )

    return False