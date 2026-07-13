from dataclasses import dataclass
from datetime import datetime

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer


@dataclass
class CandidateModel:
    """
    Represents a candidate model produced during retraining.
    """

    model: BaseEstimator

    preprocessor: ColumnTransformer

    metrics: dict

    predictions: object

    trained_at: datetime

    training_rows: int

    run_id: str | None = None

    promoted: bool = False

    model_version: str | None = None