from pathlib import Path
import yaml

def load_params(path: str = "params.yaml"):
    with open(path, "r") as file:
        params = yaml.safe_load(file)

    return params