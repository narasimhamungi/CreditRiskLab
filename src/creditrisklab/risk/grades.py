"""Internal risk-grade masterscale.

A 7-grade scale mapped from calibrated, prior-corrected PD. Two things make this more than
a lookup table:

  * Grades are assigned on the **corrected** PD. Assigning them on raw model output would
    put almost the entire portfolio in the worst two grades, because the sample default
    rate is ~10% by construction.
  * Monotonicity is tested, not assumed. A masterscale whose realised default rate is not
    non-decreasing across grades is broken, and the test reports it rather than smoothing it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditrisklab.config import load_model_config


def scale(model_cfg: dict | None = None) -> list[dict]:
    return (model_cfg or load_model_config())["grades"]["scale"]


def assign_grade(pd_value: float, model_cfg: dict | None = None) -> str:
    if pd_value is None or (isinstance(pd_value, float) and np.isnan(pd_value)):
        return "unrated"
    for band in scale(model_cfg):
        if float(pd_value) <= float(band["pd_upper"]):
            return str(band["grade"])
    return str(scale(model_cfg)[-1]["grade"])


def assign_grades(pd_values: np.ndarray, model_cfg: dict | None = None) -> np.ndarray:
    return np.array([assign_grade(p, model_cfg) for p in np.asarray(pd_values, dtype="float64")])


def grade_table(pd_values: np.ndarray, y_true: np.ndarray, model_cfg: dict | None = None) -> pd.DataFrame:
    cfg = model_cfg or load_model_config()
    grades = assign_grades(pd_values, cfg)
    frame = pd.DataFrame({"grade": grades, "pd": np.asarray(pd_values, dtype="float64"), "label": np.asarray(y_true, dtype=int)})
    labels = {str(b["grade"]): str(b["label"]) for b in scale(cfg)}
    out = (
        frame.groupby("grade")
        .agg(
            n=("label", "size"),
            observed_defaults=("label", "sum"),
            mean_pd=("pd", "mean"),
        )
        .reset_index()
    )
    out["observed_default_rate"] = out["observed_defaults"] / out["n"]
    out["label"] = out["grade"].map(labels)
    order = [str(b["grade"]) for b in scale(cfg)]
    out["__order"] = out["grade"].apply(lambda g: order.index(g) if g in order else len(order))
    return out.sort_values("__order").drop(columns="__order").reset_index(drop=True)


def monotonicity_check(table: pd.DataFrame, min_bucket: int = 3) -> dict:
    """Realised default rate must not fall as grades worsen.

    Buckets below `min_bucket` observations are excluded from the test — with five
    observations a single default swings the rate by 20 points and the violation would be
    noise, not a model defect. Excluded buckets are listed, not silently dropped.
    """
    usable = table[table["n"] >= min_bucket]
    excluded = table.loc[table["n"] < min_bucket, "grade"].tolist()
    rates = usable["observed_default_rate"].to_numpy(dtype="float64")
    violations = []
    for i in range(1, len(rates)):
        if rates[i] < rates[i - 1] - 1e-12:
            violations.append(
                {
                    "from_grade": str(usable["grade"].iloc[i - 1]),
                    "to_grade": str(usable["grade"].iloc[i]),
                    "from_rate": float(rates[i - 1]),
                    "to_rate": float(rates[i]),
                }
            )
    return {
        "monotonic": len(violations) == 0,
        "violations": violations,
        "grades_tested": usable["grade"].tolist(),
        "grades_excluded_small_n": excluded,
        "min_bucket": min_bucket,
    }
