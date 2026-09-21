"""Trellis adapter.

Trellis (Project 1) is consumed as an installed package, exactly as ValuationLab and
DealLab do. Nothing in Trellis is imported by name at module scope and nothing in Trellis
is modified — if the installed version's API differs, this adapter degrades to the direct
EDGAR client rather than failing the run.

Why a fallback exists at all: Trellis's validated universe is investment-grade large caps.
The default cohort here is delisted, post-bankruptcy issuers that Trellis was never
exercised against. The adapter tries Trellis first and records which path each issuer took,
so the provenance of every row is visible in the panel.
"""

from __future__ import annotations

import importlib
from typing import Any

import pandas as pd

from creditrisklab.ingest import edgar_client


def trellis_available() -> bool:
    try:
        importlib.import_module("trellis")
        return True
    except Exception:
        return False


def trellis_universe_tickers() -> list[str]:
    """Best-effort read of Trellis's own validated universe.

    Trellis's public API is not pinned here; several plausible accessors are tried and an
    empty list is returned if none exist. Used only by scripts/sync_trellis_universe.py,
    which asks for confirmation before writing anything into universe.yaml.
    """
    try:
        mod = importlib.import_module("trellis")
    except Exception:
        return []
    for attr in ("VALIDATED_UNIVERSE", "UNIVERSE", "validated_universe", "universe"):
        obj = getattr(mod, attr, None)
        if obj is None:
            continue
        try:
            values = obj() if callable(obj) else obj
            return [str(v).upper() for v in values]
        except Exception:
            continue
    return []


def _normalise_trellis_frame(raw: Any, cik: str, ticker: str) -> pd.DataFrame | None:
    """Map a Trellis fundamentals object into the canonical long schema.

    Returns None when the shape is not recognised, which sends the issuer down the EDGAR
    path rather than producing a silently mismapped row.
    """
    if raw is None:
        return None
    frame = raw if isinstance(raw, pd.DataFrame) else getattr(raw, "to_frame", lambda: None)()
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    required = {"field", "period_end", "value"}
    if not required.issubset({str(c).lower() for c in frame.columns}):
        return None
    frame = frame.rename(columns={c: str(c).lower() for c in frame.columns})
    if "filed" not in frame.columns:
        # No filing date means no point-in-time guarantee. Refuse rather than assume.
        return None
    frame["cik"] = cik
    frame["ticker"] = ticker
    frame["source"] = "trellis"
    for col in ("form", "fiscal_year", "tag"):
        if col not in frame.columns:
            frame[col] = None
    frame["period_end"] = pd.to_datetime(frame["period_end"]).dt.date
    frame["filed"] = pd.to_datetime(frame["filed"]).dt.date
    return frame[["cik", "ticker", "field", "period_end", "filed", "form", "fiscal_year", "value", "tag", "source"]]


def configured_loader():
    """Explicit Trellis entry point, set once: CREDITRISKLAB_TRELLIS_LOADER="module:function".

    Guessing a function name is a stopgap. Point this at Trellis's real public loader and the
    heuristic below is never consulted.
    """
    import os

    spec = os.environ.get("CREDITRISKLAB_TRELLIS_LOADER", "").strip()
    if not spec or ":" not in spec:
        return None
    module_name, func_name = spec.split(":", 1)
    try:
        return getattr(importlib.import_module(module_name), func_name)
    except Exception:
        return None


def fetch_fundamentals(cik: str, ticker: str = "", prefer_trellis: bool = True, use_cache: bool = True) -> pd.DataFrame:
    """Canonical long fundamentals for one issuer, Trellis first, EDGAR second.

    Trellis output is only accepted if it carries SEC filing dates. Without them no
    point-in-time guarantee is possible, and the issuer is routed to the EDGAR client instead.
    The `source` column on every row records which path was used.
    """
    if prefer_trellis and trellis_available():
        try:
            mod = importlib.import_module("trellis")
            loader = configured_loader()
            if loader is None:
                for attr in ("load_fundamentals", "fundamentals", "get_fundamentals", "ingest"):
                    loader = getattr(mod, attr, None)
                    if callable(loader):
                        break
            if callable(loader):
                normalised = _normalise_trellis_frame(loader(cik), cik=cik, ticker=ticker)
                if normalised is not None and not normalised.empty:
                    return normalised
        except Exception:
            pass  # fall through to EDGAR; provenance column records which path was used
    return edgar_client.company_fundamentals(cik, ticker=ticker, use_cache=use_cache)
