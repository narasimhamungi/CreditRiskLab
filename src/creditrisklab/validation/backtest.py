"""Out-of-fold prediction.

Three resampling schemes, each answering a different question:

  * `grouped_oof` — repeated stratified group k-fold. Groups are issuers, so no issuer
    appears in both train and test. This is the headline out-of-fold estimate.
  * `leave_one_issuer_out` — the most conservative estimate available on this sample size,
    and the one a model validator will ask for when told there are only ten defaults.
  * `out_of_time` — train on the early window, test on the late window. This is the honest
    test of whether the model generalises forward, and it is expected to be weaker than the
    grouped estimate. Reporting only the stronger number would be the dishonest choice.

A single 80/20 split is deliberately not offered. With ten positives, one split is a
coin toss dressed as a result.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def grouped_oof(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    build_model: Callable[[], object],
    n_splits: int = 5,
    repeats: int = 20,
    random_state: int = 42,
    sample_weight_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Mean out-of-fold probability per observation across repeated grouped CV."""
    y = np.asarray(y, dtype=int)
    n = len(y)
    effective_splits = min(n_splits, max(int(np.unique(groups[y == 1]).size), 2))
    totals = np.zeros(n, dtype="float64")
    counts = np.zeros(n, dtype="float64")

    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(n_splits=effective_splits, shuffle=True, random_state=random_state + repeat)
        try:
            folds = list(splitter.split(X, y, groups))
        except ValueError:
            continue
        for train_idx, test_idx in folds:
            if len(np.unique(y[train_idx])) < 2:
                continue
            model = build_model()
            if sample_weight_fn is not None:
                model.fit(X[train_idx], y[train_idx], sample_weight=sample_weight_fn(y[train_idx]))
            else:
                model.fit(X[train_idx], y[train_idx])
            totals[test_idx] += model.predict_proba(X[test_idx])[:, 1]
            counts[test_idx] += 1.0

    out = np.full(n, np.nan)
    scored = counts > 0
    out[scored] = totals[scored] / counts[scored]
    return out


def leave_one_issuer_out(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    build_model: Callable[[], object],
    sample_weight_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    out = np.full(len(y), np.nan)
    for issuer in np.unique(groups):
        test_idx = np.flatnonzero(groups == issuer)
        train_idx = np.flatnonzero(groups != issuer)
        if len(np.unique(y[train_idx])) < 2:
            continue
        model = build_model()
        if sample_weight_fn is not None:
            model.fit(X[train_idx], y[train_idx], sample_weight=sample_weight_fn(y[train_idx]))
        else:
            model.fit(X[train_idx], y[train_idx])
        out[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return out


def out_of_time(
    panel: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    build_model: Callable[[], object],
    split_year: int,
    sample_weight_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Train pre-split_year, predict split_year onward. Returns (predictions, test_mask)."""
    years = np.array([d.year if hasattr(d, "year") else int(str(d)[:4]) for d in panel["as_of"]])
    train_mask = years < split_year
    test_mask = ~train_mask
    out = np.full(len(y), np.nan)
    if train_mask.sum() == 0 or test_mask.sum() == 0 or len(np.unique(y[train_mask])) < 2:
        return out, test_mask
    model = build_model()
    if sample_weight_fn is not None:
        model.fit(X[train_mask], y[train_mask], sample_weight=sample_weight_fn(y[train_mask]))
    else:
        model.fit(X[train_mask], y[train_mask])
    out[test_mask] = model.predict_proba(X[test_mask])[:, 1]
    return out, test_mask


def annual_default_backtest(predictions: pd.DataFrame, pd_column: str = "calibrated_pd") -> pd.DataFrame:
    """Expected vs. realised defaults per observation year.

    Uses SAMPLE-scale PDs (calibrated, before prior correction), because the realised
    default counts are sample counts. Comparing prior-corrected PDs against an oversampled
    default count would show a large, meaningless under-prediction.

    z = (realised - expected) / sqrt(sum p(1-p)) is the normal approximation to the
    Poisson-binomial. With one or two defaults a year it is indicative only.
    """
    frame = predictions.copy()
    frame["year"] = [d.year if hasattr(d, "year") else int(str(d)[:4]) for d in frame["as_of"]]
    rows = []
    for year, group in frame.groupby("year"):
        p = group[pd_column].to_numpy(dtype="float64")
        expected = float(p.sum())
        variance = float((p * (1 - p)).sum())
        realised = int(group["label"].sum())
        z = (realised - expected) / np.sqrt(variance) if variance > 0 else float("nan")
        rows.append(
            {
                "year": int(year),
                "issuers": int(len(group)),
                "expected_defaults": round(expected, 2),
                "realised_defaults": realised,
                "z": round(float(z), 2) if np.isfinite(z) else float("nan"),
            }
        )
    return pd.DataFrame(rows)
