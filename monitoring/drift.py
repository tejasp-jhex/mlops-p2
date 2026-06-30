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

def _extract_drift_summary(result: dict) -> dict:
    metrics = result.get("metrics", [])

    # --- Dataset-level drift (metrics[0]) ---
    drift_entry = metrics[0]
    count = drift_entry["value"]["count"]
    share = drift_entry["value"]["share"]

    drift_share_threshold = drift_entry["config"].get("drift_share", 0.5)
    dataset_drift = bool(share > drift_share_threshold)  # ✅ explicit cast

    # --- Per-column drift scores (metrics[1..N]) ---
    column_drifts = {}
    for m in metrics[1:]:
        col = m["config"].get("column")
        score = float(m["value"])                          # ✅ explicit cast
        threshold = float(m["config"].get("threshold", 0.1))
        column_drifts[col] = {
            "drift_score": score,
            "drifted": bool(score > threshold),            # ✅ explicit cast
            "method": m["config"].get("method"),
        }

    return {
        "dataset_drift": bool(dataset_drift),
        "drifted_columns": int(count),
        "share_drifted_columns": float(share),             # ✅ explicit cast
        "column_drifts": column_drifts,
    }


def save_json_summary(eval_result) -> None:
    logger.info(f"Saving JSON summary to {DRIFT_RESULT_PATH}")

    result = eval_result.dict()
    summary = _extract_drift_summary(result)

    with open(DRIFT_RESULT_PATH, "w") as file:
        json.dump(summary, file, indent=4, default=_json_serializer)  # ✅ safety net

    logger.info(
        f"Drift Summary -> "
        f"Dataset Drift: {summary['dataset_drift']}, "
        f"Drifted Columns: {summary['drifted_columns']}, "
        f"Share: {summary['share_drifted_columns']:.2f}"
    )


def _json_serializer(obj):
    """Fallback serializer for types json.dump can't handle natively."""
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, (int, float)):  # catches numpy scalar types too
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

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