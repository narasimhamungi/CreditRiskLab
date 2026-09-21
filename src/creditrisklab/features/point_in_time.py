"""Point-in-time collapse of long fundamentals into an as-of snapshot.

Two filters, applied in order, and the order matters:

1. Drop every fact with `filed > as_of`. This removes restatements and later filings that
   an analyst standing at `as_of` could not have seen.
2. Within what remains, for each (field, period_end) keep the latest `filed` version —
   the value as it was most recently reported *at that time*, restatements included up to
   the as-of date but not beyond.

Then take the most recent fiscal period whose data is actually visible. Doing step 2 before
step 1 is the classic look-ahead bug: it picks the final restated value and then filters,
leaving post-hoc numbers in a "point-in-time" panel.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from creditrisklab.ingest.schema import APPROXIMATE_TAGS, CANONICAL_FIELDS


def visible_facts(long_frame: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Step 1: everything filed on or before `as_of`."""
    if long_frame.empty:
        return long_frame
    filed = pd.to_datetime(long_frame["filed"]).dt.date
    return long_frame.loc[filed <= as_of].copy()


def latest_vintage(visible: pd.DataFrame) -> pd.DataFrame:
    """Step 2: latest filed version of each (field, period_end)."""
    if visible.empty:
        return visible
    ordered = visible.sort_values(["field", "period_end", "filed"])
    return ordered.groupby(["field", "period_end"], as_index=False).last()


def as_of_snapshot(
    long_frame: pd.DataFrame,
    as_of: date,
    min_period_end: date | None = None,
    max_staleness_days: int = 550,
) -> dict[str, object] | None:
    """Wide, single-period snapshot of one issuer as known on `as_of`.

    `max_staleness_days` guards against scoring an issuer off filings that are years old —
    a real problem for distressed names, which often stop filing before they file for
    bankruptcy. Where that happens the observation is dropped rather than carried forward,
    because carrying it forward would quietly turn "stopped filing" into a feature.
    """
    visible = visible_facts(long_frame, as_of)
    if visible.empty:
        return None
    vintage = latest_vintage(visible)
    if min_period_end is not None:
        vintage = vintage.loc[pd.to_datetime(vintage["period_end"]).dt.date >= min_period_end]
    if vintage.empty:
        return None

    period_end = anchor_period(vintage)
    if (as_of - period_end).days > max_staleness_days:
        return None

    current = vintage.loc[pd.to_datetime(vintage["period_end"]).dt.date == period_end]
    snapshot: dict[str, object] = {
        "as_of": as_of,
        "period_end": period_end,
        "filed": max(pd.to_datetime(current["filed"]).dt.date),
        "report_lag_days": (max(pd.to_datetime(current["filed"]).dt.date) - period_end).days,
        "cik": current["cik"].iloc[0],
        "ticker": current["ticker"].iloc[0],
        "source": current["source"].iloc[0],
    }
    values = dict(zip(current["field"], current["value"]))
    for field in CANONICAL_FIELDS:
        snapshot[field] = _to_float(values.get(field))
    snapshot["_tags"] = dict(zip(current["field"], current["tag"])) if "tag" in current.columns else {}

    # Many filers never tag the total-liabilities subtotal. Derive it from the accounting
    # identity rather than leave it missing, and record that it was derived.
    derived: list[str] = []
    derived.extend(APPROXIMATE_TAGS[t] for t in snapshot["_tags"].values() if t in APPROXIMATE_TAGS)
    derived.extend(_derive_ebit(snapshot))
    if snapshot.get("total_liabilities") is None and snapshot.get("total_assets") is not None and snapshot.get("equity") is not None:
        snapshot["total_liabilities"] = float(snapshot["total_assets"]) - float(snapshot["equity"])
        derived.append("total_liabilities")
    snapshot["derived_fields"] = derived

    prior_end = _prior_period_end(vintage, period_end)
    if prior_end is not None:
        prior = vintage.loc[pd.to_datetime(vintage["period_end"]).dt.date == prior_end]
        prior_values = dict(zip(prior["field"], prior["value"]))
        snapshot["prior_period_end"] = prior_end
        snapshot["prior_net_income"] = _to_float(prior_values.get("net_income"))
        snapshot["prior_revenue"] = _to_float(prior_values.get("revenue"))
        snapshot["prior_total_assets"] = _to_float(prior_values.get("total_assets"))
    else:
        snapshot["prior_period_end"] = None
        snapshot["prior_net_income"] = None
        snapshot["prior_revenue"] = None
        snapshot["prior_total_assets"] = None
    return snapshot


def anchor_period(vintage: pd.DataFrame) -> date:
    """The fiscal period the snapshot describes.

    Anchored on total assets, not on the latest date of any fact. A 10-K can carry a stray
    fact dated after the fiscal year end (a subsequent event, a post-year-end debt amount);
    taking the maximum date across all fields would select that date and return a snapshot
    with one or two fields and everything else missing. Falls back to the most-populated
    period when total assets is absent.
    """
    ends = pd.to_datetime(vintage["period_end"]).dt.date
    assets = vintage.loc[(vintage["field"] == "total_assets").to_numpy()]
    if not assets.empty:
        return max(pd.to_datetime(assets["period_end"]).dt.date)
    counts = pd.Series(1, index=ends).groupby(level=0).sum()
    top = counts.max()
    return max(d for d, n in counts.items() if n == top)


def _derive_ebit(snapshot: dict) -> list[str]:
    """EBIT = pre-tax income + gross interest expense, or pre-tax income minus net interest.

    Used only when OperatingIncomeLoss is untagged. With no interest information at all,
    EBIT stays missing rather than being set to pre-tax income, which would understate it
    by the full interest burden and bias every such issuer toward distress.
    """
    if snapshot.get("ebit") is not None or snapshot.get("pretax_income") is None:
        return []
    pretax = float(snapshot["pretax_income"])
    if snapshot.get("interest_expense") is not None:
        snapshot["ebit"] = pretax + abs(float(snapshot["interest_expense"]))
        return ["ebit"]
    if snapshot.get("net_interest") is not None:
        snapshot["ebit"] = pretax - float(snapshot["net_interest"])
        return ["ebit"]
    return []


def _prior_period_end(vintage: pd.DataFrame, period_end: date) -> date | None:
    ends = sorted({d for d in pd.to_datetime(vintage["period_end"]).dt.date if d < period_end})
    if not ends:
        return None
    candidate = ends[-1]
    # Require the prior period to be roughly one year earlier, not an arbitrary stub period.
    if timedelta(days=270) <= (period_end - candidate) <= timedelta(days=460):
        return candidate
    return None


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(out) else out
