import joblib
from pathlib import Path
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from config.config import (
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    MODEL_DIR,
    RANDOM_STATE,
)


def load_processed_data():

    return joblib.load(PROCESSED_DATA_PATH)

def train_model(X_train, y_train):

    model = XGBClassifier(
        random_state=RANDOM_STATE,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        eval_metric="logloss",
    )

    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
    }

    return metrics

def save_model(model):

    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(
        model,
        MODEL_PATH,
    )


def main():

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = load_processed_data()

    model = train_model(
        X_train,
        y_train,
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    save_model(model)

    print("\nTraining Complete\n")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()