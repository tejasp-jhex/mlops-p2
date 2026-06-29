from datetime import datetime
import pandas as pd
from pathlib import Path

from config.config import PRODUCTION_DATA_PATH

LOG_FILE = Path(PRODUCTION_DATA_PATH)


def log_prediction(input_data, prediction, probability):

    row = input_data.copy()

    row["timestamp"] = datetime.now().isoformat()

    row["prediction"] = prediction

    row["probability"] = probability

    df = pd.DataFrame([row])

    if LOG_FILE.exists():
        df.to_csv(
            LOG_FILE,
            mode="a",
            header=False,
            index=False,
        )
    else:
        df.to_csv(
            LOG_FILE,
            index=False,
        )