import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from utils import load_params
from logger import get_logger

logger = get_logger(__name__)


def preprocess(
    df: pd.DataFrame,
    test_size: float | None = None,
    random_state: int | None = None,
):
    """
    Cleans the retraining dataset, splits it into train/test sets,
    builds a preprocessing pipeline, fits it, and transforms the data.
    """

    logger.info("Starting preprocessing pipeline.")

    df = df.copy()

    # --------------------------------------------------
    # Load parameters
    # --------------------------------------------------

    params = load_params()

    test_size = (
        test_size
        if test_size is not None
        else params["data"]["test_size"]
    )

    random_state = (
        random_state
        if random_state is not None
        else params["random_state"]
    )

    # --------------------------------------------------
    # Data Cleaning
    # --------------------------------------------------

    df.drop(
        columns=[
            "customerID",
            "model_version",
        ],
        errors="ignore",
        inplace=True,
    )

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    logger.info(
        f"Dataset shape after cleaning: {df.shape}"
    )

    # --------------------------------------------------
    # Features & Target
    # --------------------------------------------------

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    if df["Churn"].dtype == 'boolean':
        y = df["Churn"].astype(int)
    else:
        y = df["Churn"].map(
            {
                "No": 0,
                "Yes": 1,
            }
        )

    # --------------------------------------------------
    # Train Test Split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        f"Training samples: {len(X_train)} | "
        f"Testing samples: {len(X_test)}"
    )

    # --------------------------------------------------
    # Feature Groups
    # --------------------------------------------------

    numerical_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_features = X_train.select_dtypes(
        include=["object"]
    ).columns

    logger.info(
        f"Numerical Features: {len(numerical_features)}"
    )

    logger.info(
        f"Categorical Features: {len(categorical_features)}"
    )

    # --------------------------------------------------
    # Pipelines
    # --------------------------------------------------

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    logger.info(
        "Fitting preprocessing pipeline."
    )

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    logger.info(
        f"Processed training shape: {X_train_processed.shape}"
    )

    logger.info(
        f"Processed testing shape: {X_test_processed.shape}"
    )

    logger.info(
        "Preprocessing completed successfully."
    )

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
    )