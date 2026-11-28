"""Fixed model settings used for the final test evaluation."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}
THRESHOLD = 0.5


def make_model() -> RandomForestClassifier:
    """Create the fixed v5 Random Forest classifier."""
    return RandomForestClassifier(**RANDOM_FOREST_PARAMS)
