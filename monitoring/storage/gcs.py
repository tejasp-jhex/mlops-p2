from datetime import datetime
from pathlib import Path

from google.cloud import storage

from config.config import (
    GCS_BUCKET_NAME,
)
from logger import get_logger

logger = get_logger(__name__)


def get_gcs_client() -> storage.Client:
    """
    Creates and returns a Google Cloud Storage client.
    """

    return storage.Client()


def get_bucket():
    """
    Returns the configured GCS bucket.
    """

    client = get_gcs_client()

    return client.bucket(GCS_BUCKET_NAME)


from config.config import MODEL_VERSION

def get_report_folder() -> str:
    """
    Returns today's report folder.

    Example:
    reports/v1/2026/07/01
    """

    today = datetime.utcnow()

    return (
        f"reports/"
        f"{MODEL_VERSION}/"
        f"{today.year}/"
        f"{today.month:02d}/"
        f"{today.day:02d}"
    )

def upload_file(
    local_path: str,
    blob_name: str,
):
    """
    Uploads any local file to GCS.
    """

    try:

        logger.info(
            f"Uploading {local_path} to GCS."
        )

        bucket = get_bucket()

        blob = bucket.blob(blob_name)

        blob.upload_from_filename(local_path)

        logger.info(
            f"Successfully uploaded {blob_name}."
        )

    except Exception:

        logger.exception(
            f"Failed to upload {local_path}."
        )

        raise

def upload_report(
    report_path: str,
):
    """
    Uploads the HTML drift report.
    """

    report_folder = get_report_folder()

    blob_name = (
        f"{report_folder}/"
        f"{Path(report_path).name}"
    )

    upload_file(
        report_path,
        blob_name,
    )


def upload_json(
    json_path: str,
):
    """
    Uploads the JSON drift summary.
    """

    report_folder = get_report_folder()

    blob_name = (
        f"{report_folder}/"
        f"{Path(json_path).name}"
    )

    upload_file(
        json_path,
        blob_name,
    )


def download_file(
    blob_name: str,
    destination_path: str,
):
    """
    Downloads a file from GCS.
    """

    try:

        logger.info(
            f"Downloading {blob_name}."
        )

        bucket = get_bucket()

        blob = bucket.blob(blob_name)

        blob.download_to_filename(
            destination_path,
        )

        logger.info(
            "Download completed."
        )

    except Exception:

        logger.exception(
            f"Failed to download {blob_name}."
        )

        raise
