from xgboost import XGBClassifier

from utils import load_params
from logger import get_logger

logger = get_logger(__name__)


def train_model(
    X_train,
    y_train,
):
    """
    Train the production XGBoost model.
    """

    logger.info("Starting model training.")

    params = load_params()

    model = XGBClassifier(
        random_state=params["random_state"],
        n_estimators=params["model"]["n_estimators"],
        learning_rate=params["model"]["learning_rate"],
        max_depth=params["model"]["max_depth"],
        eval_metric=params["model"]["eval_metric"],
    )

    model.fit(
        X_train,
        y_train,
    )

    logger.info("Model training completed successfully.")

    return model