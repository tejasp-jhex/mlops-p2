import pandas as pd

from monitoring.logger import log_prediction
import serving.app.state as state

def predict(data):

    df = pd.DataFrame([data])

    processed = state.preprocessor.transform(df)

    prediction = state.model.predict(processed)[0]

    probability = state.model.predict_proba(processed)[0][1]

    log_prediction(
        data,
        bool(prediction),
        round(float(probability), 4),
    )

    return {
        "will_churn": bool(prediction),
        "probability": round(float(probability), 4)
    }