"""Population stability.

PSI is included because model validation functions ask for it, and because the panel here
spans 2015-2024 — a period containing a pandemic-era default cluster. If feature
distributions in the late window differ materially from the early window, the model's
out-of-time behaviour is not the same as its out-of-fold behaviour, and the report should
say so.

Conventional reading: PSI < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 material shift.
Those thresholds are industry convention, not a statistical test, and are labelled as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    expected = np.asarray(expected, dtype="float64")
    actual = np.asarray(actual, dtype="float64")
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    quantiles = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))
    if len(quantiles) < 3:
        return 0.0
    edges = quantiles.copy()
    edges[0], edges[-1] = -np.inf, np.inf
    exp_counts = np.histogram(expected, bins=edges)[0] / len(expected)
    act_counts = np.histogram(actual, bins=edges)[0] / len(actual)
    exp_counts = np.clip(exp_counts, EPS, None)
    act_counts = np.clip(act_counts, EPS, None)
    return float(np.sum((act_counts - exp_counts) * np.log(act_counts / exp_counts)))


def interpret(value: float) -> str:
    if np.isnan(value):
        return "insufficient data"
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "moderate shift"
    return "material shift"


def psi_by_feature(panel: pd.DataFrame, features: list[str], split_year: int) -> pd.DataFrame:
    """Early window vs. late window, split on the observation year."""
    years = pd.Series([d.year if hasattr(d, "year") else int(str(d)[:4]) for d in panel["as_of"]])
    early = panel.loc[(years < split_year).to_numpy()]
    late = panel.loc[(years >= split_year).to_numpy()]
    rows = []
    for feature in features:
        value = psi(early[feature].to_numpy(dtype="float64"), late[feature].to_numpy(dtype="float64"))
        rows.append({"feature": feature, "psi": value, "reading": interpret(value)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
