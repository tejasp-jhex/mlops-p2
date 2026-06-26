import pandas as pd

import app.state as state

def predict(data):

    df = pd.DataFrame([data])

    processed = state.preprocessor.transform(df)

    prediction = state.model.predict(processed)[0]

    probability = state.model.predict_proba(processed)[0][1]

    return {
        "will_churn": bool(prediction),
        "probability": round(float(probability), 4)
    }