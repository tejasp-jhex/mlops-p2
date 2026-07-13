FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY retraining/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY retraining/ retraining/
COPY config/ config/
COPY logger.py .
COPY mlflow_utils.py . 
COPY utils.py . 
COPY params.yaml .

ENV PYTHONPATH=/app

CMD ["python", "-m", "retraining.src.retrain"]