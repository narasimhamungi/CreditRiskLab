"""Exposure at default.

EAD = drawn + CCF x undrawn.

The undrawn term is where corporate EAD is actually decided. Borrowers approaching default
draw down committed revolvers aggressively, so treating a revolver as zero exposure because
it is currently undrawn understates EAD materially. The CCFs used here come from published
empirical work on credit-line usage, cited in config/recovery_rates.yaml, and sit within
the Basel standardised range.

Where the run has no instrument-level data — which is the case when exposures are inferred
from balance-sheet debt rather than a loan tape — `ead_from_balance_sheet` is used and the
report states that the exposure is a balance-sheet proxy, not a real facility schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from creditrisklab.config import load_recovery_config


@dataclass(frozen=True)
class Facility:
    instrument: str          # revolving_credit_facility | term_loan | bond
    drawn: float
    undrawn: float = 0.0


def ccf(instrument: str, recovery_cfg: dict | None = None) -> float:
    cfg = recovery_cfg or load_recovery_config()
    factors = cfg.get("ead", {}).get("credit_conversion_factors", {})
    block = factors.get(instrument)
    if block is None:
        raise ValueError(f"no CCF configured for instrument '{instrument}'")
    return float(block["value"])


def facility_ead(facility: Facility, recovery_cfg: dict | None = None) -> float:
    if facility.drawn < 0 or facility.undrawn < 0:
        raise ValueError("drawn and undrawn amounts must be non-negative")
    return facility.drawn + ccf(facility.instrument, recovery_cfg) * facility.undrawn


def portfolio_ead(facilities: Iterable[Facility], recovery_cfg: dict | None = None) -> float:
    return sum(facility_ead(f, recovery_cfg) for f in facilities)


def ead_from_balance_sheet(long_term_debt: float | None, current_debt: float | None) -> float:
    """Fallback exposure proxy: total reported debt, fully drawn.

    This is a PROXY and is labelled as one everywhere it is used. It has no undrawn
    component, so it is a lower bound on true EAD for any issuer with committed lines.
    """
    return float(long_term_debt or 0.0) + float(current_debt or 0.0)
