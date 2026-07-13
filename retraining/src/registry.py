import mlflow
from mlflow import MlflowClient

from logger import get_logger

logger = get_logger(__name__)

client = MlflowClient()


def get_production_metrics(model_name: str):
    """
    Fetch metrics of the current production model.

    Returns None if no production model exists.
    """

    try:
        model = client.get_model_version_by_alias(
            name=model_name,
            alias="Production",
        )

        run = client.get_run(
            model.run_id,
        )

        logger.info(
            f"Production model found (version={model.version})."
        )

        return run.data.metrics

    except Exception:
        logger.info(
            "No production model found."
        )

        return None


def should_promote(
    candidate_metrics: dict,
    production_metrics: dict | None,
):
    """
    Decide whether the candidate model should be promoted.
    """

    if production_metrics is None:
        logger.info(
            "No production model exists. Candidate will be promoted."
        )
        return True

    candidate_f1 = candidate_metrics["f1_score"]
    production_f1 = production_metrics["f1_score"]

    logger.info(
        f"Candidate F1: {candidate_f1:.4f} | "
        f"Production F1: {production_f1:.4f}"
    )

    return candidate_f1 > production_f1


def register_candidate(
    candidate,
    model_name: str,
):
    """
    Register the candidate model.
    """

    logger.info(
        "Registering candidate model."
    )

    model_uri = f"runs:/{candidate.run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    client.set_registered_model_alias(
        name=model_name,
        alias="Production",
        version=registered_model.version,
    )

    logger.info(
        f"Candidate promoted to Production (version={registered_model.version})."
    )

    return registered_model.version