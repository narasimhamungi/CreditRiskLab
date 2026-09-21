"""Control-group hardness.

The single largest threat to this project's credibility is not a coding error; it is an easy
discrimination task. Ten famous bankruptcies against five-to-thirteen mega-cap
investment-grade names will produce a high AUC for almost any reasonable score, including
Altman's 1968-era ratios. A reviewer who knows credit will discount a near-perfect AUC on
that design immediately.

This module measures the problem instead of hiding it: what share of non-default
observations actually look like speculative-grade credits? If the answer is near zero, the
report states that the AUC measures an easy task. The screen also defines, by rule rather
than by hand-picking, which surviving issuers belong in a harder control cohort — selecting
controls by name after the fact would be selection on the outcome.

Leverage bands follow the S&P Corporate Methodology core-ratio table, where debt/EBITDA
above 4x corresponds to an "aggressive" or "highly leveraged" financial risk profile.
Interest cover below 2x is a conventional speculative-grade marker and is labelled as a
convention, not a sourced threshold.
"""

from __future__ import annotations

import pandas as pd

AGGRESSIVE_LEVERAGE = 4.0     # S&P Corporate Methodology: >4x = aggressive or worse
WEAK_COVERAGE = 2.0           # convention


def speculative_mask(panel: pd.DataFrame, leverage: float = AGGRESSIVE_LEVERAGE, coverage: float = WEAK_COVERAGE) -> pd.Series:
    lev = panel.get("debt_ebitda")
    cov = panel.get("interest_cover")
    lev_flag = (lev > leverage) if lev is not None else False
    cov_flag = (cov < coverage) if cov is not None else False
    return (lev_flag | cov_flag).fillna(False).astype(bool)


def control_group_hardness(panel: pd.DataFrame) -> dict:
    controls = panel.loc[panel["label"] == 0]
    if controls.empty:
        return {"control_observations": 0, "speculative_like_share": float("nan"), "speculative_like_issuers": []}
    mask = speculative_mask(controls)
    never_defaulted = controls["is_default_issuer"] == 0 if "is_default_issuer" in controls else pd.Series(True, index=controls.index)
    survivors = controls.loc[mask & never_defaulted, "ticker"].unique().tolist()
    return {
        "control_observations": int(len(controls)),
        "speculative_like_share": float(mask.mean()),
        "speculative_like_share_true_survivors": float(mask[never_defaulted].mean()) if never_defaulted.any() else float("nan"),
        "speculative_like_issuers": sorted(survivors),
    }


def hardness_verdict(stats: dict) -> str:
    share = stats.get("speculative_like_share_true_survivors", float("nan"))
    if share != share:  # NaN
        return "not computable"
    if share < 0.10:
        return (
            "EASY TASK — under 10% of surviving-issuer observations look speculative-grade. A high AUC "
            "on this design measures the gap between mega-cap IG and bankruptcy, not model skill."
        )
    if share < 0.30:
        return "partially hard — some surviving controls carry speculative-grade leverage or coverage"
    return "hard task — a material share of surviving controls look speculative-grade"
