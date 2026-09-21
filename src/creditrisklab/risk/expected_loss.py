"""Expected loss and regulatory capital.

EL = PD x LGD x EAD.

Unexpected loss is included via the Basel IRB corporate risk-weight function because EL and
UL answer different questions and a credit risk deliverable that stops at EL is incomplete:
EL is provisioned through the P&L, UL is what capital is held against. The formula below is
the supervisory one, not an invention — see BCBS, "Basel II: International Convergence of
Capital Measurement and Capital Standards" (2006), paragraphs 272-273, carried into the
consolidated Basel Framework at CRE31.

Correlation:   R = 0.12 * w + 0.24 * (1 - w),  w = (1 - e^(-50*PD)) / (1 - e^(-50))
Maturity adj:  b = (0.11852 - 0.05478 * ln(PD))^2
Capital req:   K = [LGD * N( (1-R)^-0.5 * G(PD) + (R/(1-R))^0.5 * G(0.999) ) - PD * LGD]
                   * (1 + (M - 2.5) * b) / (1 - 1.5 * b)
RWA = K * 12.5 * EAD
"""

from __future__ import annotations

from math import exp, log, sqrt

import numpy as np
import pandas as pd
from scipy.stats import norm

PD_FLOOR = 0.0003  # Basel floors corporate PD at 3bp; applied so capital is never zero


def expected_loss(pd_value: float, lgd: float, ead: float) -> float:
    _check(pd_value, lgd, ead)
    return float(pd_value) * float(lgd) * float(ead)


def asset_correlation(pd_value: float) -> float:
    pd_value = max(float(pd_value), PD_FLOOR)
    w = (1 - exp(-50 * pd_value)) / (1 - exp(-50))
    return 0.12 * w + 0.24 * (1 - w)


def firm_size_adjustment(correlation: float, sales_eur_millions: float | None) -> float:
    """SME adjustment: -0.04 * (1 - (S-5)/45) for turnover between EUR 5m and EUR 50m."""
    if sales_eur_millions is None:
        return correlation
    s = min(max(float(sales_eur_millions), 5.0), 50.0)
    return correlation - 0.04 * (1 - (s - 5.0) / 45.0)


def maturity_adjustment(pd_value: float) -> float:
    pd_value = max(float(pd_value), PD_FLOOR)
    return (0.11852 - 0.05478 * log(pd_value)) ** 2


def irb_capital_requirement(
    pd_value: float,
    lgd: float,
    maturity_years: float = 2.5,
    sales_eur_millions: float | None = None,
) -> float:
    """K — capital requirement per unit of EAD, corporate IRB."""
    _check(pd_value, lgd, 1.0)
    pd_value = max(float(pd_value), PD_FLOOR)
    r = firm_size_adjustment(asset_correlation(pd_value), sales_eur_millions)
    b = maturity_adjustment(pd_value)
    conditional = norm.cdf(norm.ppf(pd_value) / sqrt(1 - r) + sqrt(r / (1 - r)) * norm.ppf(0.999))
    k = (lgd * conditional - pd_value * lgd) * (1 + (maturity_years - 2.5) * b) / (1 - 1.5 * b)
    return float(max(k, 0.0))


def risk_weighted_assets(pd_value: float, lgd: float, ead: float, maturity_years: float = 2.5) -> float:
    return irb_capital_requirement(pd_value, lgd, maturity_years) * 12.5 * float(ead)


def portfolio_expected_loss(frame: pd.DataFrame) -> dict[str, float]:
    """Aggregate EL, UL and RWA over a frame with columns pd, lgd, ead[, maturity_years]."""
    required = {"pd", "lgd", "ead"}
    if not required.issubset(frame.columns):
        raise KeyError(f"frame must contain {sorted(required)}")
    maturity = frame["maturity_years"] if "maturity_years" in frame.columns else pd.Series(2.5, index=frame.index)
    el = np.array([expected_loss(p, l, e) for p, l, e in zip(frame["pd"], frame["lgd"], frame["ead"])])
    k = np.array([irb_capital_requirement(p, l, m) for p, l, m in zip(frame["pd"], frame["lgd"], maturity)])
    ead_values = frame["ead"].to_numpy(dtype="float64")
    total_ead = float(ead_values.sum())
    return {
        "total_ead": total_ead,
        "expected_loss": float(el.sum()),
        "expected_loss_rate": float(el.sum() / total_ead) if total_ead else float("nan"),
        "capital_requirement": float((k * ead_values).sum()),
        "rwa": float((k * ead_values).sum() * 12.5),
        "exposure_weighted_pd": float((frame["pd"].to_numpy() * ead_values).sum() / total_ead) if total_ead else float("nan"),
        "exposure_weighted_lgd": float((frame["lgd"].to_numpy() * ead_values).sum() / total_ead) if total_ead else float("nan"),
    }


def _check(pd_value: float, lgd: float, ead: float) -> None:
    if not 0.0 <= float(pd_value) <= 1.0:
        raise ValueError(f"PD out of range: {pd_value}")
    if not 0.0 <= float(lgd) <= 1.0:
        raise ValueError(f"LGD out of range: {lgd}")
    if float(ead) < 0:
        raise ValueError(f"EAD must be non-negative: {ead}")
