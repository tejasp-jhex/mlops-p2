import json

from evidently import Report
from evidently.presets import DataDriftPreset

from config.config import (
    DRIFT_REPORT_PATH,
    DRIFT_RESULT_PATH,
)
from logger import get_logger
from monitoring.storage.bigquery import (
    fetch_reference_data,
    fetch_predictions,
)

logger = get_logger(__name__)


def load_data():
    logger.info("Loading reference and production datasets from BigQuery.")

    reference = fetch_reference_data()
    production = fetch_predictions()

    logger.info(
        f"Loaded {len(reference)} reference records and "
        f"{len(production)} production records."
    )

    return reference, production


def prepare_data(reference, production):
    logger.info("Preparing datasets for drift detection.")

    reference = reference.drop(
        columns=[
            "customerID",
            "Churn",
            "model_version",
        ],
        errors="ignore",
    )

    production = production.drop(
        columns=[
            "timestamp",
            "prediction",
            "probability",
            "model_version",
        ],
        errors="ignore",
    )

    # Keep only the columns present in the reference dataset
    production = production[reference.columns]

    logger.info(
        f"Prepared datasets with {len(reference.columns)} features."
    )

    logger.info(
        f"Reference Shape: {reference.shape}, "
        f"Production Shape: {production.shape}"
    )

    return reference, production


def generate_report(reference, production):
    logger.info("Running Evidently data drift analysis.")

    report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )

    result = report.run(
        reference_data=reference,
        current_data=production,
    )

    logger.info("Data drift analysis completed.")

    return result


def save_html_report(report):
    logger.info(f"Saving HTML report to {DRIFT_REPORT_PATH}")

    report.save_html(DRIFT_REPORT_PATH)

def save_json_summary(eval_result) -> None:
    """Save a JSON summary of drift results. Compatible with Evidently v0.7+."""
    logger.info(f"Saving JSON summary to {DRIFT_RESULT_PATH}")

    result = eval_result.dict()  # ✅ was: report.as_dict()

    # v0.7 Snapshot dict structure is different — extract defensively
    summary = _extract_drift_summary(result)

    with open(DRIFT_RESULT_PATH, "w") as file:
        json.dump(summary, file, indent=4)

    logger.info(
        f"Drift Summary -> "
        f"Dataset Drift: {summary['dataset_drift']}, "
        f"Drifted Columns: {summary['drifted_columns']}, "
        f"Share: {summary['share_drifted_columns']:.2f}"
    )


def _extract_drift_summary(result: dict) -> dict:
    """
    Extract drift metrics from Evidently's dict output.
    Handles both v0.6 (metrics[0]["result"]) and v0.7 (metrics[*]["metric_id"])
    structures defensively.
    """
    # Walk all metric entries and find the DatasetDriftMetric result
    for metric_entry in result.get("metrics", []):
        metric_result = metric_entry.get("result", {})

        # v0.6 key names
        if "dataset_drift" in metric_result:
            return {
                "dataset_drift": metric_result["dataset_drift"],
                "drifted_columns": metric_result.get("number_of_drifted_columns", 0),
                "share_drifted_columns": metric_result.get("share_of_drifted_columns", 0.0),
            }

        # v0.7 key names (snake_case, slightly different names)
        if "share_drifted" in metric_result or "num_drifted" in metric_result:
            share = metric_result.get("share_drifted", 0.0)
            num = metric_result.get("num_drifted", 0)
            return {
                "dataset_drift": share > 0.5,   # Evidently's default threshold
                "drifted_columns": num,
                "share_drifted_columns": share,
            }

    # Last resort: dump the raw result so you can inspect it and update the paths
    logger.warning("Could not extract drift summary from known key paths. Raw result saved.")
    return {"dataset_drift": None, "drifted_columns": None, "share_drifted_columns": 0.0, "_raw": result}

def detect_drift():
    try:
        logger.info("Starting drift detection.")

        reference, production = load_data()

        reference, production = prepare_data(
            reference,
            production,
        )

        report = generate_report(
            reference,
            production,
        )

        save_html_report(report)

        save_json_summary(report)

        logger.info("Drift detection completed successfully.")

    except Exception:
        logger.exception("Drift detection failed.")
        raise


def main():
    detect_drift()


if __name__ == "__main__":
    main()