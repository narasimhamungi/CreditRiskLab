"""Credit ratios.

Design rules:
  * Every ratio returns None rather than a guessed value when its denominator is missing
    or effectively zero. Imputation is a modelling decision made inside the training fold,
    not something smuggled into feature construction.
  * Ratios are capped at economically meaningful bounds before winsorisation, because a
    single issuer with near-zero EBITDA can otherwise produce a leverage figure that
    dominates the scaler.
  * Signs are stated so coefficient signs can be sanity-checked in validation: a credit
    model whose leverage coefficient comes out protective is broken, not clever.
"""

from __future__ import annotations

from math import log
from typing import Any

# Expected sign of the coefficient on default (higher value => higher PD?)
EXPECTED_SIGN: dict[str, int] = {
    "wc_ta": -1,
    "re_ta": -1,
    "ebit_ta": -1,
    "equity_tl": -1,
    "current_ratio": -1,
    "debt_ebitda": +1,
    "interest_cover": -1,
    "cfo_total_debt": -1,
    "net_margin": -1,
    "log_assets": -1,
    "negative_equity": +1,
    "two_year_loss": +1,
}

CAPS: dict[str, tuple[float, float]] = {
    "debt_ebitda": (-50.0, 50.0),
    "interest_cover": (-50.0, 50.0),
    "cfo_total_debt": (-5.0, 5.0),
    "current_ratio": (0.0, 20.0),
    "equity_tl": (-10.0, 20.0),
    "net_margin": (-5.0, 5.0),
    "wc_ta": (-5.0, 5.0),
    "re_ta": (-20.0, 5.0),
    "ebit_ta": (-5.0, 5.0),
}


def _div(num: Any, den: Any, min_den: float = 1e-6) -> float | None:
    if num is None or den is None:
        return None
    try:
        num_f, den_f = float(num), float(den)
    except (TypeError, ValueError):
        return None
    if abs(den_f) < min_den:
        return None
    return num_f / den_f


def total_debt(snap: dict[str, Any]) -> float | None:
    """Long-term plus current debt, without double counting current maturities.

    `LongTermDebt` already includes current maturities. When that inclusive tag is the one
    reported, a current-maturity tag (LongTermDebtCurrent / DebtCurrent) is not added again;
    only genuinely separate short-term borrowings are. Operating lease liabilities are not
    included — see README limitations on lease-adjusted leverage for retailers.
    """
    from creditrisklab.ingest.schema import CURRENT_MATURITY_TAGS, LONG_TERM_DEBT_INCLUSIVE_TAGS

    lt, cur = snap.get("long_term_debt"), snap.get("current_debt")
    if lt is None and cur is None:
        reported = snap.get("total_debt_reported")
        return None if reported is None else float(reported)
    tags = snap.get("_tags") or {}
    if lt is not None and cur is not None:
        if tags.get("long_term_debt") in LONG_TERM_DEBT_INCLUSIVE_TAGS and tags.get("current_debt") in CURRENT_MATURITY_TAGS:
            return float(lt)
    return float(lt or 0.0) + float(cur or 0.0)


def ebitda(snap: dict[str, Any]) -> float | None:
    ebit = snap.get("ebit")
    if ebit is None:
        return None
    return float(ebit) + float(snap.get("depreciation_amortisation") or 0.0)


def coverage_interest(snap: dict[str, Any]) -> float | None:
    """Interest burden for the coverage ratio.

    Gross interest expense when tagged. Otherwise net interest, but only when it is a net
    EXPENSE (negative): a net expense understates gross expense by the interest income, so
    coverage is slightly flattered, which is the smaller error. A net interest INCOME says
    nothing about the debt-service burden and is not used — the reason net interest was
    excluded from the gross tag list in the first place.
    """
    gross = snap.get("interest_expense")
    if gross is not None:
        return float(gross)
    net = snap.get("net_interest")
    if net is not None and float(net) < 0:
        return abs(float(net))
    return None


def working_capital(snap: dict[str, Any]) -> float | None:
    ca, cl = snap.get("current_assets"), snap.get("current_liabilities")
    if ca is None or cl is None:
        return None
    return float(ca) - float(cl)


def compute_ratios(snap: dict[str, Any]) -> dict[str, float | None]:
    """Canonical feature vector for one point-in-time snapshot."""
    ta = snap.get("total_assets")
    tl = snap.get("total_liabilities")
    equity = snap.get("equity")
    if equity is None and ta is not None and tl is not None:
        equity = float(ta) - float(tl)

    td = total_debt(snap)
    eb = ebitda(snap)
    interest = coverage_interest(snap)

    out: dict[str, float | None] = {
        "wc_ta": _div(working_capital(snap), ta),
        "re_ta": _div(snap.get("retained_earnings"), ta),
        "ebit_ta": _div(snap.get("ebit"), ta),
        "equity_tl": _div(equity, tl),
        "current_ratio": _div(snap.get("current_assets"), snap.get("current_liabilities")),
        "debt_ebitda": _div(td, eb) if (td is not None and eb is not None and eb > 0) else None,
        "interest_cover": _div(snap.get("ebit"), abs(float(interest))) if interest is not None and float(interest) != 0 else None,
        "cfo_total_debt": _div(snap.get("cfo"), td),
        "net_margin": _div(snap.get("net_income"), snap.get("revenue")),
        "log_assets": log(float(ta)) if ta not in (None, 0) and float(ta) > 0 else None,
        "negative_equity": _negative_equity(equity),
        "two_year_loss": _two_year_loss(snap),
    }

    # Negative EBITDA is a distress signal, not a missing value: leverage is undefined but
    # the firm is worse off than any positive-EBITDA peer. Map it to the cap, not to None.
    if out["debt_ebitda"] is None and td is not None and eb is not None and eb <= 0:
        out["debt_ebitda"] = CAPS["debt_ebitda"][1]
    # Only a REPORTED zero interest expense earns the top coverage value. A missing interest
    # tag is a data gap, not evidence of no debt service — treating it as strong coverage would
    # hand a distressed issuer with an untagged interest line the best score in the sample.
    if out["interest_cover"] is None and snap.get("ebit") is not None and interest is not None and float(interest) == 0:
        out["interest_cover"] = CAPS["interest_cover"][1]

    return {k: _apply_cap(k, v) for k, v in out.items()}


def _negative_equity(equity: Any) -> float | None:
    if equity is None:
        return None
    return 1.0 if float(equity) < 0 else 0.0


def _two_year_loss(snap: dict[str, Any]) -> float | None:
    ni, prior = snap.get("net_income"), snap.get("prior_net_income")
    if ni is None or prior is None:
        return None
    return 1.0 if (float(ni) < 0 and float(prior) < 0) else 0.0


def _apply_cap(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    lo, hi = CAPS.get(name, (-1e12, 1e12))
    return max(lo, min(hi, float(value)))
