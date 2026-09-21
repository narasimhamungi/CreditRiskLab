"""Altman Z-score family — benchmark, not the primary model.

Variant choice is a substantive decision, not a detail.

  * Original Z (1968) requires market value of equity and is calibrated on listed
    **manufacturers**. The default cohort here is retail, pharmacy retail, telecom, media
    and E&P. Applying manufacturer coefficients to retailers is the single most common
    misuse of this model, and market cap is not available from XBRL filings anyway.
  * Z'' / "EM score" (Altman, Hartzell & Peck, 1995) drops the asset-turnover term
    precisely because it is industry-sensitive, and uses book equity rather than market
    value. It is the correct variant for a mixed non-manufacturing cohort and is the one
    reported as the benchmark.
  * Z' (1983, private manufacturers) is implemented for completeness and reported, not
    used for the headline comparison.

Reference: Altman (1968), Journal of Finance 23(4); Altman, Hartzell & Peck (1995),
"Emerging Market Corporate Bonds: A Scoring System", Salomon Brothers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

ZPP_COEFFICIENTS = {"wc_ta": 6.56, "re_ta": 3.26, "ebit_ta": 6.72, "equity_tl": 1.05}
ZPP_CONSTANT = 3.25  # EM-score constant, standardising the distress threshold onto a rating scale
ZPP_ZONES = {"safe": 2.60, "distress": 1.10}  # applied to the score excluding the constant

ZPRIME_COEFFICIENTS = {
    "wc_ta": 0.717,
    "re_ta": 0.847,
    "ebit_ta": 3.107,
    "equity_tl": 0.420,
    "sales_ta": 0.998,
}
ZPRIME_ZONES = {"safe": 2.90, "distress": 1.23}


def z_double_prime(row: dict[str, Any] | pd.Series, include_constant: bool = False) -> float | None:
    """Z'' for non-manufacturers. Returns None if any input ratio is missing."""
    total = 0.0
    for feature, weight in ZPP_COEFFICIENTS.items():
        value = row.get(feature) if isinstance(row, dict) else row.get(feature, None)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        total += weight * float(value)
    return total + (ZPP_CONSTANT if include_constant else 0.0)


def z_prime(row: dict[str, Any] | pd.Series) -> float | None:
    """Z' for private manufacturers. Needs sales/assets, which the core feature set omits."""
    total = 0.0
    for feature, weight in ZPRIME_COEFFICIENTS.items():
        value = row.get(feature) if isinstance(row, dict) else row.get(feature, None)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        total += weight * float(value)
    return total


def zone(score: float | None, zones: dict[str, float] | None = None) -> str:
    zones = zones or ZPP_ZONES
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return "unscored"
    if score > zones["safe"]:
        return "safe"
    if score < zones["distress"]:
        return "distress"
    return "grey"


def score_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach Z'' and its zone to a modelling panel."""
    out = panel.copy()
    out["z_double_prime"] = [z_double_prime(row) for _, row in out.iterrows()]
    out["z_zone"] = [zone(value) for value in out["z_double_prime"]]
    return out


def as_risk_score(panel: pd.DataFrame) -> np.ndarray:
    """Higher = riskier, so Z'' can be compared against model PDs on the same AUC orientation.

    Unscored rows fall back to the cohort median rather than being dropped, so the benchmark
    is evaluated on the identical sample as the fitted model. Dropping them would flatter
    whichever model happened to score the harder cases.
    """
    scores = pd.Series([z_double_prime(row) for _, row in panel.iterrows()], dtype="float64")
    median = scores.median()
    return (-scores.fillna(median)).to_numpy()
