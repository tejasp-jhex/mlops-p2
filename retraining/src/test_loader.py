from datetime import datetime

from retraining.src.candidate_model import CandidateModel
from retraining.storage.bigquery import (
    insert_retraining_history,
)


candidate = CandidateModel(
    model=None,
    preprocessor=None,
    metrics={
        "accuracy": 0.81,
        "precision": 0.72,
        "recall": 0.60,
        "f1_score": 0.65,
    },
    predictions=None,
    trained_at=datetime.utcnow(),
    training_rows=7043,
    run_id="test_run",
    promoted=True,
    model_version="5",
)

insert_retraining_history(candidate)