"""Calibration testing.

Discrimination answers 'does the model rank correctly'. Calibration answers 'is a 2% PD
actually a 2% PD'. Only the second one feeds expected loss, capital or provisioning, and
it is the half that portfolio credit models usually skip.

Tests implemented:
  * Brier score, decomposed into reliability, resolution and uncertainty (Murphy, 1973).
    The decomposition matters because a low Brier on an imbalanced sample can be achieved
    by predicting the base rate for everything — resolution near zero exposes that.
  * Calibration slope and intercept (Cox, 1958): regress the outcome on the predicted
    logit. Slope 1 / intercept 0 is perfect; slope below 1 means predictions are too
    extreme, which is the usual small-sample failure.
  * Hosmer-Lemeshow. Reported with an explicit caveat: with ten positives the test has
    almost no power and its p-value should not be read as evidence of good calibration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from creditrisklab.models._sklearn_compat import unpenalised_logistic_regression

EPS = 1e-9


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype="float64")
    p = np.asarray(p, dtype="float64")
    return float(np.mean((p - y_true) ** 2))


def brier_decomposition(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype="float64")
    p = np.asarray(p, dtype="float64")
    base = float(np.mean(y_true))
    bins = np.clip(np.digitize(p, np.linspace(0, 1, n_bins + 1)[1:-1]), 0, n_bins - 1)
    reliability = 0.0
    resolution = 0.0
    for b in range(n_bins):
        mask = bins == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        p_bar = float(np.mean(p[mask]))
        o_bar = float(np.mean(y_true[mask]))
        reliability += n_b * (p_bar - o_bar) ** 2
        resolution += n_b * (o_bar - base) ** 2
    n = len(y_true)
    return {
        "brier": brier_score(y_true, p),
        "reliability": reliability / n,   # lower is better
        "resolution": resolution / n,     # higher is better
        "uncertainty": base * (1 - base),
    }


def calibration_slope_intercept(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(p, dtype="float64"), EPS, 1 - EPS)
    if len(np.unique(y_true)) < 2:
        return {"slope": float("nan"), "intercept": float("nan")}
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    model = unpenalised_logistic_regression(max_iter=1000).fit(logit, y_true)
    return {"slope": float(model.coef_[0][0]), "intercept": float(model.intercept_[0])}


def hosmer_lemeshow(y_true: np.ndarray, p: np.ndarray, n_groups: int = 10) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype="float64")
    p = np.asarray(p, dtype="float64")
    order = np.argsort(p)
    y_true, p = y_true[order], p[order]
    groups = np.array_split(np.arange(len(p)), min(n_groups, max(len(p), 1)))
    statistic = 0.0
    used = 0
    for idx in groups:
        if len(idx) == 0:
            continue
        observed = float(y_true[idx].sum())
        expected = float(p[idx].sum())
        denom = expected * (1 - expected / len(idx))
        if denom <= EPS:
            continue
        statistic += (observed - expected) ** 2 / denom
        used += 1
    dof = max(used - 2, 1)
    p_value = float(1 - stats.chi2.cdf(statistic, dof))
    return {
        "statistic": float(statistic),
        "dof": int(dof),
        "p_value": p_value,
        "caveat": "low power at this sample size; a non-rejection is not evidence of calibration",
    }


def reliability_table(y_true: np.ndarray, p: np.ndarray, n_bins: int = 5) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype="float64")
    p = np.asarray(p, dtype="float64")
    order = np.argsort(p)
    chunks = np.array_split(order, min(n_bins, max(len(p), 1)))
    rows = []
    for i, idx in enumerate(chunks, start=1):
        if len(idx) == 0:
            continue
        rows.append(
            {
                "bucket": i,
                "n": int(len(idx)),
                "mean_predicted_pd": float(np.mean(p[idx])),
                "observed_default_rate": float(np.mean(y_true[idx])),
                "observed_defaults": int(np.sum(y_true[idx])),
            }
        )
    return pd.DataFrame(rows)
