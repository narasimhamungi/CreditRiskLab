"""Ohlson O-score (1980) — second benchmark.

Reference: Ohlson, "Financial Ratios and the Probabilistic Prediction of Bankruptcy",
Journal of Accounting Research 18(1), 1980, Model 1 (one-year horizon).

Two honest deviations from the published specification, both flagged rather than hidden:

  1. SIZE is log(total assets / GNP price-level index). The original index base is a 1968-era
     series that is not maintained in its published form. A configurable deflator is used,
     defaulting to a 2015=100 base. The effect is a level shift in O, which moves the
     absolute PD but leaves the ranking — and therefore AUC — unchanged.
  2. FFO (funds from operations) is proxied by cash flow from operations. Ohlson's FFO is
     pre-working-capital; CFO is post. This is the standard modern substitute and is the
     reason the O-score PD here should be read as a ranking, not a calibrated probability.
"""

from __future__ import annotations

from math import exp, log
from typing import Any

import numpy as np
import pandas as pd

COEFFICIENTS = {
    "const": -1.32,
    "size": -0.407,
    "tl_ta": 6.03,
    "wc_ta": -1.43,
    "cl_ca": 0.0757,
    "oeneg": -1.72,
    "ni_ta": -2.37,
    "ffo_tl": -1.83,
    "intwo": 0.285,
    "chin": -0.521,
}

DEFAULT_PRICE_INDEX = 100.0


def o_score(snap: dict[str, Any], price_index: float = DEFAULT_PRICE_INDEX) -> float | None:
    ta = _f(snap.get("total_assets"))
    tl = _f(snap.get("total_liabilities"))
    ca = _f(snap.get("current_assets"))
    cl = _f(snap.get("current_liabilities"))
    ni = _f(snap.get("net_income"))
    cfo = _f(snap.get("cfo"))
    prior_ni = _f(snap.get("prior_net_income"))

    if None in (ta, tl, ca, cl, ni) or ta <= 0 or tl <= 0 or cl <= 0:
        return None

    size = log(ta / price_index)
    tl_ta = tl / ta
    wc_ta = (ca - cl) / ta
    cl_ca = cl / ca if ca > 0 else 0.0
    oeneg = 1.0 if tl > ta else 0.0
    ni_ta = ni / ta
    ffo_tl = (cfo / tl) if cfo is not None else 0.0
    intwo = 1.0 if (prior_ni is not None and ni < 0 and prior_ni < 0) else 0.0
    if prior_ni is None:
        chin = 0.0
    else:
        denom = abs(ni) + abs(prior_ni)
        chin = (ni - prior_ni) / denom if denom > 0 else 0.0

    c = COEFFICIENTS
    return (
        c["const"]
        + c["size"] * size
        + c["tl_ta"] * tl_ta
        + c["wc_ta"] * wc_ta
        + c["cl_ca"] * cl_ca
        + c["oeneg"] * oeneg
        + c["ni_ta"] * ni_ta
        + c["ffo_tl"] * ffo_tl
        + c["intwo"] * intwo
        + c["chin"] * chin
    )


def o_score_pd(snap: dict[str, Any], price_index: float = DEFAULT_PRICE_INDEX) -> float | None:
    score = o_score(snap, price_index=price_index)
    if score is None:
        return None
    return 1.0 / (1.0 + exp(-max(-40.0, min(40.0, score))))


def score_frame(snapshots: pd.DataFrame, price_index: float = DEFAULT_PRICE_INDEX) -> np.ndarray:
    """Risk-oriented O-scores for a frame of raw snapshots (higher = riskier)."""
    values = pd.Series([o_score(row.to_dict(), price_index=price_index) for _, row in snapshots.iterrows()], dtype="float64")
    return values.fillna(values.median()).to_numpy()


def _f(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(out) else out
