from fastapi import FastAPI
from app.predictor import predict
from app.schemas import (
    CustomerData,
    PredictionResponse,
)
from contextlib import asynccontextmanager
from app.model_loader import load_models

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    print("👋 API shutting down.")

app = FastAPI(
    title="Customer Churn Prediction API",
    lifespan=lifespan,
)

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict_customer(data: CustomerData):
    return predict(data.model_dump())