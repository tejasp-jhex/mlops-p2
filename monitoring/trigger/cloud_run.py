from google.cloud import run_v2

from config.config import (
    PROJECT_ID,
    REGION,
    RETRAINING_JOB_NAME,
)

from logger import get_logger

logger = get_logger(__name__)

client = run_v2.JobsClient()


def trigger_retraining_job():

    logger.info(
        "Triggering retraining Cloud Run Job."
    )

    job_name = (
        f"projects/{PROJECT_ID}"
        f"/locations/{REGION}"
        f"/jobs/{RETRAINING_JOB_NAME}"
    )

    operation = client.run_job(
        name=job_name,
    )

    logger.info(
        f"Retraining job started: {operation.operation.name}"
    )