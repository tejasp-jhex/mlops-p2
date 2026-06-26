FROM python:3.13-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Print logs immediately
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies
COPY requirements/base.txt .
COPY requirements/serving.txt .

RUN pip install --no-cache-dir -r serving.txt

# Copy source code
COPY serving ./serving
COPY config ./config
COPY models ./models

EXPOSE 8000

CMD ["uvicorn",  "serving.app.main:app", "--host", "0.0.0.0",  "--port", "8000"]