from fastapi import FastAPI
from serving.app.predictor import predict
from serving.app.schemas import (
    CustomerData,
    PredictionResponse,
)
from contextlib import asynccontextmanager
from serving.app.model_loader import load_models
from serving.logger import get_logger

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    logger.info("👋 API shutting down.")

app = FastAPI(
    title="Customer Churn Prediction API",
    lifespan=lifespan,
)

@app.get("/")
def home():
    return {"message": "API is running"}

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_customer(data: CustomerData):
    return predict(data.model_dump())