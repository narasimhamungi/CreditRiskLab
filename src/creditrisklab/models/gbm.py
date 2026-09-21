"""Gradient boosting challenger.

Reported as a challenger, never promoted to primary on this sample size. With roughly ten
positive observations, a boosted ensemble will almost always post a higher in-sample and a
noisier out-of-fold AUC than logistic regression, and the difference will not be
statistically distinguishable. Where the out-of-fold gap is inside the bootstrap confidence
interval, the validation report says so rather than declaring a winner.

Missing values are handled natively by HistGradientBoostingClassifier, so no imputer is
applied — the model treats missingness as information, which is defensible here because
missing debt tags in distressed filings are not missing at random.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from creditrisklab.config import load_model_config


def build_model(model_cfg: dict | None = None) -> HistGradientBoostingClassifier:
    cfg = (model_cfg or load_model_config()).get("gbm", {})
    return HistGradientBoostingClassifier(
        max_depth=int(cfg.get("max_depth", 3)),
        max_iter=int(cfg.get("max_iter", 120)),
        learning_rate=float(cfg.get("learning_rate", 0.05)),
        min_samples_leaf=int(cfg.get("min_samples_leaf", 3)),
        l2_regularization=float(cfg.get("l2_regularization", 1.0)),
        random_state=int((model_cfg or load_model_config())["validation"].get("random_state", 42)),
    )


def enabled(model_cfg: dict | None = None) -> bool:
    return bool((model_cfg or load_model_config()).get("gbm", {}).get("enabled", False))


def class_weights(y: np.ndarray) -> np.ndarray:
    """Balanced sample weights — HistGradientBoosting has no class_weight parameter."""
    y = np.asarray(y)
    positives = max(int(y.sum()), 1)
    negatives = max(int(len(y) - y.sum()), 1)
    weight_pos = len(y) / (2.0 * positives)
    weight_neg = len(y) / (2.0 * negatives)
    return np.where(y == 1, weight_pos, weight_neg)
