#!/bin/bash

set -e

echo "Starting MLflow..."

mlflow server \
    --backend-store-uri "$BACKEND_STORE_URI" \
    --default-artifact-root "$ARTIFACT_ROOT" \
    --host 0.0.0.0 \
    --allowed-hosts "*" \
    --cors-allowed-origins "*" \
    --port 5000