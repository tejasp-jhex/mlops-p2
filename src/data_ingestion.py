from pathlib import Path
import pandas as pd
from config.config import RAW_DATA_PATH


def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset from CSV.
    """
    return pd.read_csv(path)


if __name__ == "__main__":

    data_path = RAW_DATA_PATH

    df = load_data(data_path)

    print(df.head())

    print(f"\nDataset Shape: {df.shape}")