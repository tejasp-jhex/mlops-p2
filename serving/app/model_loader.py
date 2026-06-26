import joblib
from serving.logger import get_logger

logger = get_logger(__name__)

from config.config import (
    MODEL_PATH,
    PREPROCESSOR_PATH,
)

import serving.app.state as state


def load_models():
    """
    Load model and preprocessor into application state.
    """

    state.model = joblib.load(MODEL_PATH)
    state.preprocessor = joblib.load(PREPROCESSOR_PATH)

    logger.info("Model and preprocessor loaded.")