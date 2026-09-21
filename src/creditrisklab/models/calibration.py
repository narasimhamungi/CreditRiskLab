"""Calibration and prior correction.

This module exists because of a specific, deliberate design choice upstream: the sample is
built by deliberately over-representing defaults (roughly 10% of observations, against a
population corporate default rate near 1-3%). That is the right way to build the sample —
it is the only way to get usable positives out of SEC filings — but it makes every raw
predicted probability wrong by a large multiple.

The fix is a prior correction on the odds (King & Zeng, 2001, "Logistic Regression in Rare
Events Data", Political Analysis 9(2); the same adjustment appears in Anderson, 1972, as
the choice-based-sampling intercept correction):

    odds_corrected = odds_sample * (tau / (1 - tau)) * ((1 - ybar) / ybar)

where tau is the population default rate and ybar the sample default rate. Ranking, and
therefore AUC, is unchanged — the correction is monotone. What changes is the absolute PD,
and therefore expected loss and every risk grade. An uncorrected portfolio PD model reports
expected losses roughly an order of magnitude too high.

Because tau is itself an assumption, the run reports PDs across a sensitivity grid.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from creditrisklab.models._sklearn_compat import logistic_regression

EPS = 1e-9


def prior_correct(p_sample: np.ndarray | float, population_rate: float, sample_rate: float) -> np.ndarray:
    """Shift probabilities from the sampled default rate onto the population rate."""
    if not 0.0 < population_rate < 1.0:
        raise ValueError("population_rate must be in (0, 1)")
    if not 0.0 < sample_rate < 1.0:
        raise ValueError("sample_rate must be in (0, 1)")
    p = np.clip(np.asarray(p_sample, dtype="float64"), EPS, 1 - EPS)
    odds = p / (1.0 - p)
    factor = (population_rate / (1.0 - population_rate)) * ((1.0 - sample_rate) / sample_rate)
    adjusted = odds * factor
    return adjusted / (1.0 + adjusted)


def sensitivity(p_sample: np.ndarray, sample_rate: float, grid: list[float]) -> dict[float, np.ndarray]:
    return {tau: prior_correct(p_sample, tau, sample_rate) for tau in grid}


class OutOfFoldCalibrator:
    """Isotonic where there is enough data, Platt otherwise.

    Isotonic regression with ten positives will interpolate through noise and produce a
    step function that looks perfectly calibrated in-sample and generalises badly. The
    fallback threshold is explicit rather than left to chance.
    """

    def __init__(self, method: str = "isotonic", min_positives_for_isotonic: int = 25):
        self.method = method
        self.min_positives_for_isotonic = min_positives_for_isotonic
        self.fitted_method_: str | None = None
        self._model = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> "OutOfFoldCalibrator":
        p = np.clip(np.asarray(p, dtype="float64"), EPS, 1 - EPS)
        y = np.asarray(y, dtype=int)
        positives = int(y.sum())
        use_isotonic = self.method == "isotonic" and positives >= self.min_positives_for_isotonic
        if use_isotonic:
            self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
            self.fitted_method_ = "isotonic"
        else:
            logit = np.log(p / (1 - p)).reshape(-1, 1)
            self._model = logistic_regression(max_iter=1000).fit(logit, y)
            self.fitted_method_ = "platt"
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("calibrator is not fitted")
        p = np.clip(np.asarray(p, dtype="float64"), EPS, 1 - EPS)
        if self.fitted_method_ == "isotonic":
            return np.clip(self._model.predict(p), EPS, 1 - EPS)
        logit = np.log(p / (1 - p)).reshape(-1, 1)
        return np.clip(self._model.predict_proba(logit)[:, 1], EPS, 1 - EPS)


def cross_fitted_calibration(
    p: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    method: str = "isotonic",
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, str]:
    """Calibrate out-of-fold predictions without evaluating the calibrator on its own fit.

    Fitting a calibrator on the out-of-fold predictions and then scoring Brier / slope on
    those same predictions reports a fitted number as a validated one. Here each issuer's
    predictions are calibrated by a map learned only from *other* issuers, so the calibration
    metrics in the report are genuinely out-of-sample.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    p = np.clip(np.asarray(p, dtype="float64"), EPS, 1 - EPS)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    out = p.copy()
    positive_groups = int(np.unique(groups[y == 1]).size)
    splits = min(n_splits, max(positive_groups, 2))
    methods: list[str] = []
    try:
        folds = list(StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=random_state).split(p, y, groups))
    except ValueError:
        folds = []
    for train_idx, test_idx in folds:
        if len(np.unique(y[train_idx])) < 2:
            continue  # leave these predictions uncalibrated rather than fit on one class
        cal = OutOfFoldCalibrator(method=method).fit(p[train_idx], y[train_idx])
        out[test_idx] = cal.transform(p[test_idx])
        methods.append(str(cal.fitted_method_))
    used = max(set(methods), key=methods.count) if methods else "none"
    return out, used
