from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config.config import (
    RAW_DATA_PATH,
    MODEL_DIR,
    PREPROCESSOR_PATH,
    PROCESSED_DATA_PATH
)
from utils import load_params
from training.src.logger import get_logger

logger = get_logger(__name__)


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load the dataset from the given path.
    """
    return pd.read_csv(data_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data cleaning.
    """

    df = df.copy()

    # Remove customer ID
    df.drop(columns=["customerID"], inplace=True)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    return df

def split_features_target(df: pd.DataFrame):

    X = df.drop(columns=["Churn"])

    y = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    return X, y

def split_data(X, y):
    params = load_params()
    
    return train_test_split(
        X,
        y,
        test_size=params["data"]["test_size"],
        random_state=params["random_state"],
        stratify=y
    )


def build_preprocessor(X_train):

    numerical_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_features = X_train.select_dtypes(
        include=["object"]
    ).columns

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor


def preprocess_data(
    preprocessor,
    X_train,
    X_test
):

    X_train_processed = preprocessor.fit_transform(X_train)

    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed


def save_artifacts(
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test
):

    Path("models").mkdir(exist_ok=True)

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH
    )

    joblib.dump(
        (X_train, X_test, y_train, y_test),
        PROCESSED_DATA_PATH
    )


def main():

    data_path = RAW_DATA_PATH

    df = load_data(data_path)

    df = clean_data(df)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    preprocessor = build_preprocessor(X_train)

    X_train, X_test = preprocess_data(
        preprocessor,
        X_train,
        X_test
    )

    save_artifacts(
        preprocessor,
        X_train,
        X_test,
        y_train,
        y_test
    )

    logger.info("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()