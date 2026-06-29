from google.cloud import storage

from config.config import (
    GCS_BUCKET_NAME,
)

client = storage.Client()

bucket = client.bucket(GCS_BUCKET_NAME)