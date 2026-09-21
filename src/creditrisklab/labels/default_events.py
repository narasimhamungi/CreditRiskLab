"""Default labelling.

Framing: discrete-time hazard, not a single cross-section. Each issuer contributes one
observation per year it is alive and filing. The label is 1 if the issuer defaults within
`horizon_days` after that observation date, 0 otherwise. Observations after the default
date do not exist — the issuer has exited.

Why this rather than one row per company: a one-row-per-company cross-section throws away
every pre-distress year of a defaulted issuer, which is exactly the period a PD model is
supposed to learn from, and leaves a sample of ~20 rows. The panel construction is standard
in the literature (Shumway, 2001, "Forecasting Bankruptcy More Accurately: A Simple Hazard
Model", Journal of Business 74(1)) and is the reason the same issuer must never appear in
both the training and test fold — hence grouped cross-validation downstream.
"""

from __future__ import annotations

from datetime import date, timedelta

from creditrisklab.universe import Issuer

DEFAULT_OBSERVATION_MONTH_DAY = (6, 30)


def label_observation(issuer: Issuer, as_of: date, horizon_days: int = 365) -> int | None:
    """1 = defaults within the horizon, 0 = survives it, None = observation is invalid.

    None is returned when the observation sits at or after the default date. Those rows are
    dropped rather than labelled 0, which would teach the model that a bankrupt firm is safe.
    """
    if not issuer.is_default:
        return 0
    assert issuer.default_date is not None
    if as_of >= issuer.default_date:
        return None
    if issuer.default_date <= as_of + timedelta(days=horizon_days):
        return 1
    return 0


def observation_dates(
    issuer: Issuer,
    window: tuple[date, date],
    month_day: tuple[int, int] = DEFAULT_OBSERVATION_MONTH_DAY,
) -> list[date]:
    """Annual observation grid, identical for defaulters and non-defaulters.

    The same grid for both cohorts is what stops the model learning a calendar effect: if
    defaulters were only observed in 2019-2023 and survivors across 2015-2024, 'year' would
    do the discrimination.
    """
    start, end = window
    month, day = month_day
    dates: list[date] = []
    for year in range(start.year, end.year + 1):
        candidate = date(year, month, day)
        if start <= candidate <= end:
            if issuer.is_default and issuer.default_date is not None and candidate >= issuer.default_date:
                continue
            dates.append(candidate)
    return dates


def time_to_default_days(issuer: Issuer, as_of: date) -> int | None:
    if not issuer.is_default or issuer.default_date is None:
        return None
    return (issuer.default_date - as_of).days
