FROM python:3.13-slim

WORKDIR /app

# Install Python dependencies
COPY monitoring/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy required source code
COPY monitoring ./monitoring
COPY config ./config
COPY logger.py .

# Allow absolute imports
ENV PYTHONPATH=/app

CMD ["python", "-m", "monitoring.drift"]