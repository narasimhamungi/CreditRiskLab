"""Discrimination: AUC, Gini, KS, with confidence intervals.

Point estimates on ten positives are close to meaningless on their own. Every metric here
is reported with a bootstrap interval, and the bootstrap resamples **issuers**, not rows.
Resampling rows would treat the five annual observations of one issuer as five independent
facts and would produce intervals roughly half as wide as the truth.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, np.asarray(scores, dtype="float64")))


def gini(y_true: np.ndarray, scores: np.ndarray) -> float:
    value = auc(y_true, scores)
    return float("nan") if np.isnan(value) else 2.0 * value - 1.0


def ks_statistic(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Maximum separation between the cumulative default and non-default score distributions."""
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, np.asarray(scores, dtype="float64"))
    return float(np.max(np.abs(tpr - fpr)))


def bootstrap_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    iterations: int = 2000,
    random_state: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Cluster bootstrap over issuers."""
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype="float64")
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in unique_groups}

    draws: list[float] = []
    for _ in range(iterations):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([index_by_group[g] for g in sampled])
        if len(np.unique(y_true[idx])) < 2:
            continue
        draws.append(roc_auc_score(y_true[idx], scores[idx]))

    point = auc(y_true, scores)
    if not draws:
        return {"auc": point, "lower": float("nan"), "upper": float("nan"), "n_draws": 0}
    arr = np.asarray(draws)
    return {
        "auc": point,
        "lower": float(np.quantile(arr, alpha / 2)),
        "upper": float(np.quantile(arr, 1 - alpha / 2)),
        "n_draws": int(len(arr)),
    }


def paired_auc_difference(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    groups: np.ndarray,
    iterations: int = 2000,
    random_state: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Model A minus model B, same resample each draw.

    A paired bootstrap is what makes 'the model beats Altman' a testable claim rather than
    a comparison of two point estimates. If the interval straddles zero, the report says the
    difference is not distinguishable — that is a finding, not a failure.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true, dtype=int)
    a = np.asarray(scores_a, dtype="float64")
    b = np.asarray(scores_b, dtype="float64")
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in unique_groups}

    draws: list[float] = []
    for _ in range(iterations):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([index_by_group[g] for g in sampled])
        if len(np.unique(y_true[idx])) < 2:
            continue
        draws.append(roc_auc_score(y_true[idx], a[idx]) - roc_auc_score(y_true[idx], b[idx]))

    point = auc(y_true, a) - auc(y_true, b)
    if not draws:
        return {"difference": point, "lower": float("nan"), "upper": float("nan"), "significant": False}
    arr = np.asarray(draws)
    lower = float(np.quantile(arr, alpha / 2))
    upper = float(np.quantile(arr, 1 - alpha / 2))
    return {
        "difference": point,
        "lower": lower,
        "upper": upper,
        "significant": bool(lower > 0 or upper < 0),
        "n_draws": int(len(arr)),
    }
