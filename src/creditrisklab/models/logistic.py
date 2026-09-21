"""Primary PD model: penalised logistic regression.

Chosen over a tree ensemble as the primary, not because it scores better, but because a
credit model has to be explainable to a credit committee and a model validator. Coefficient
signs are checkable against economic priors (see `features.ratios.EXPECTED_SIGN`); a
gradient boosting challenger is reported alongside in `models.gbm`.

Every preprocessing step lives inside the sklearn Pipeline so that winsorisation bounds,
imputation medians and scaling parameters are all fitted on the training fold only. Fitting
any of them on the full panel is a leak that inflates cross-validated AUC, and it is the
most common defect in portfolio credit models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from creditrisklab.config import feature_names, load_model_config
from creditrisklab.models._sklearn_compat import logistic_regression


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column at quantiles learned on the training fold only."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):  # noqa: N803
        frame = pd.DataFrame(X)
        self.lower_bounds_ = np.array(frame.quantile(self.lower).to_numpy(dtype="float64"), copy=True)
        self.upper_bounds_ = np.array(frame.quantile(self.upper).to_numpy(dtype="float64"), copy=True)
        # A degenerate column (all-identical) would otherwise clip everything to one value.
        flat = self.lower_bounds_ >= self.upper_bounds_
        self.lower_bounds_[flat] = -np.inf
        self.upper_bounds_[flat] = np.inf
        return self

    def transform(self, X):  # noqa: N803
        values = pd.DataFrame(X).to_numpy(dtype="float64")
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)


def build_pipeline(model_cfg: dict | None = None) -> Pipeline:
    cfg = model_cfg or load_model_config()
    pre = cfg.get("preprocessing", {})
    lo, hi = pre.get("winsorize_quantiles", [0.01, 0.99])
    log_cfg = cfg["logistic"]
    return Pipeline(
        steps=[
            ("winsorize", Winsorizer(lower=float(lo), upper=float(hi))),
            ("impute", SimpleImputer(strategy=str(pre.get("impute", "median")))),
            ("scale", StandardScaler()),
            (
                "clf",
                logistic_regression(
                    penalty=log_cfg.get("penalty", "l2"),
                    C=float(log_cfg.get("C", 1.0)),
                    class_weight=log_cfg.get("class_weight", "balanced"),
                    max_iter=int(log_cfg.get("max_iter", 2000)),
                    solver=log_cfg.get("solver", "lbfgs"),
                ),
            ),
        ]
    )


def design_matrix(panel: pd.DataFrame, model_cfg: dict | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, groups). Groups are tickers — the same issuer must never straddle folds."""
    cfg = model_cfg or load_model_config()
    features = feature_names(cfg)
    missing = [f for f in features if f not in panel.columns]
    if missing:
        raise KeyError(f"panel is missing configured features: {missing}")
    X = panel[features].astype("float64").to_numpy()
    y = panel["label"].astype(int).to_numpy()
    groups = panel["ticker"].astype(str).to_numpy()
    return X, y, groups


def fit(panel: pd.DataFrame, model_cfg: dict | None = None) -> Pipeline:
    cfg = model_cfg or load_model_config()
    X, y, _ = design_matrix(panel, cfg)
    pipe = build_pipeline(cfg)
    pipe.fit(X, y)
    return pipe


def coefficients(pipe: Pipeline, model_cfg: dict | None = None) -> pd.DataFrame:
    """Standardised coefficients with the economic sign check attached."""
    from creditrisklab.features.ratios import EXPECTED_SIGN

    cfg = model_cfg or load_model_config()
    features = feature_names(cfg)
    coefs = pipe.named_steps["clf"].coef_[0]
    rows = []
    for name, value in zip(features, coefs):
        expected = EXPECTED_SIGN.get(name, 0)
        actual = int(np.sign(value)) if abs(value) > 1e-8 else 0
        rows.append(
            {
                "feature": name,
                "coefficient": float(value),
                "expected_sign": expected,
                "actual_sign": actual,
                "sign_consistent": bool(expected == 0 or actual == 0 or expected == actual),
            }
        )
    return pd.DataFrame(rows).sort_values("coefficient", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def sign_violations(pipe: Pipeline, model_cfg: dict | None = None, min_magnitude: float = 0.05) -> list[str]:
    """Material coefficients that contradict credit priors. Reported, never auto-corrected."""
    frame = coefficients(pipe, model_cfg)
    bad = frame[(~frame["sign_consistent"]) & (frame["coefficient"].abs() >= min_magnitude)]
    return [
        f"{r.feature}: coefficient {r.coefficient:+.3f} contradicts the expected sign "
        f"({'higher value => higher PD' if r.expected_sign > 0 else 'higher value => lower PD'})"
        for r in bad.itertuples()
    ]
