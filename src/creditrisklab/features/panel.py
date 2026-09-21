"""Assembly of the modelling panel: issuers x observation dates -> features + label."""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, Mapping

import pandas as pd

from creditrisklab.config import feature_names, load_model_config, load_universe
from creditrisklab.features.point_in_time import as_of_snapshot
from creditrisklab.features.ratios import compute_ratios, total_debt as _total_debt
from creditrisklab.labels.default_events import label_observation, observation_dates, time_to_default_days
from creditrisklab.universe import Issuer

META_COLUMNS = [
    "ticker",
    "cik",
    "name",
    "sector",
    "as_of",
    "period_end",
    "filed",
    "report_lag_days",
    "source",
    "is_default_issuer",
    "days_to_default",
    "label",
    "total_assets",
    "total_debt",
]


def _parse_window(raw: Iterable) -> tuple[date, date]:
    values = list(raw)
    out = []
    for value in values[:2]:
        if isinstance(value, datetime):
            out.append(value.date())
        elif isinstance(value, date):
            out.append(value)
        else:
            out.append(datetime.strptime(str(value), "%Y-%m-%d").date())
    return out[0], out[1]


def build_panel(
    issuers: Iterable[Issuer],
    fundamentals: Mapping[str, pd.DataFrame],
    universe_cfg: dict | None = None,
    model_cfg: dict | None = None,
) -> pd.DataFrame:
    """One row per issuer-year. `fundamentals` maps CIK -> long fundamentals frame."""
    universe_cfg = universe_cfg or load_universe()
    model_cfg = model_cfg or load_model_config()
    meta = universe_cfg["meta"]
    window = _parse_window(meta["observation_window"])
    horizon = int(meta.get("horizon_days", 365))
    features = feature_names(model_cfg)

    rows: list[dict] = []
    for issuer in issuers:
        long_frame = fundamentals.get(str(issuer.cik))
        if long_frame is None or len(long_frame) == 0:
            continue
        for as_of in observation_dates(issuer, window):
            label = label_observation(issuer, as_of, horizon_days=horizon)
            if label is None:
                continue
            snap = as_of_snapshot(long_frame, as_of)
            if snap is None:
                continue
            ratios = compute_ratios(snap)
            row = {
                "ticker": issuer.ticker,
                "cik": issuer.cik,
                "name": issuer.name,
                "sector": issuer.sector,
                "as_of": as_of,
                "period_end": snap["period_end"],
                "filed": snap["filed"],
                "report_lag_days": snap["report_lag_days"],
                "source": snap["source"],
                "is_default_issuer": int(issuer.is_default),
                "days_to_default": time_to_default_days(issuer, as_of),
                "label": int(label),
                "total_assets": snap.get("total_assets"),
                "total_debt": _total_debt(snap),
            }
            row.update({name: ratios.get(name) for name in features})
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=META_COLUMNS + features)
    panel = pd.DataFrame(rows)
    return panel[META_COLUMNS + features].sort_values(["as_of", "ticker"]).reset_index(drop=True)


def panel_diagnostics(panel: pd.DataFrame) -> dict:
    """Facts a reviewer will ask for before believing any AUC."""
    if panel.empty:
        return {"observations": 0}
    features = [c for c in panel.columns if c not in META_COLUMNS]
    missing = panel[features].isna().mean().sort_values(ascending=False)
    return {
        "observations": int(len(panel)),
        "issuers": int(panel["ticker"].nunique()),
        "positive_labels": int(panel["label"].sum()),
        "sample_default_rate": float(panel["label"].mean()),
        "observation_years": sorted({d.year if hasattr(d, "year") else d for d in panel["as_of"]}),
        "defaulted_issuers_represented": int(panel.loc[panel["label"] == 1, "ticker"].nunique()),
        "median_report_lag_days": float(panel["report_lag_days"].median()),
        "worst_missingness": {k: round(float(v), 3) for k, v in missing.head(5).items()},
        "source_mix": panel["source"].value_counts().to_dict(),
    }


def coverage_warnings(panel: pd.DataFrame, min_positive: int = 8, max_missing: float = 0.35) -> list[str]:
    """Explicit statistical-power warnings, surfaced in the report rather than left implicit."""
    warnings: list[str] = []
    if panel.empty:
        return ["panel is empty"]
    positives = int(panel["label"].sum())
    if positives < min_positive:
        warnings.append(
            f"only {positives} positive observations — discrimination estimates will have very wide "
            "confidence intervals and should not be read as a validated model"
        )
    features = [c for c in panel.columns if c not in META_COLUMNS]
    for feature in features:
        rate = float(panel[feature].isna().mean())
        if rate > max_missing:
            warnings.append(f"feature '{feature}' is {rate:.0%} missing — imputation is doing most of the work")
    if panel.loc[panel["label"] == 1, "ticker"].nunique() < 5:
        warnings.append("fewer than 5 distinct defaulted issuers contribute a positive label")
    return warnings
