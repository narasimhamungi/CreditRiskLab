"""Direct SEC XBRL ingestion, point-in-time by construction.

The `filed` date on every XBRL fact is carried through to the modelling panel. That single
column is what makes the point-in-time claim in the README real rather than aspirational:
a restatement filed after the as-of date cannot be selected, because the filter is applied
to the data, not to the analyst's memory.

Network access is required. Nothing here is cached in the repo — `data/raw/` is gitignored.
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from creditrisklab.config import project_root
from creditrisklab.ingest.schema import (
    ANNUAL_MAX_DAYS,
    ANNUAL_MIN_DAYS,
    DURATION_CONCEPTS,
    INSTANT_CONCEPTS,
)

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

# SEC requires a descriptive User-Agent with contact details. Set CREDITRISKLAB_SEC_UA.
DEFAULT_UA = "CreditRiskLab/0.1 (portfolio project; set CREDITRISKLAB_SEC_UA with your email)"
REQUEST_PAUSE_SECONDS = 0.15  # SEC fair-access guidance is <=10 requests/second


class EdgarError(RuntimeError):
    pass


def _headers() -> dict[str, str]:
    import os

    try:  # .env support, matching the rest of the portfolio; override beats stale setx values
        from dotenv import load_dotenv

        load_dotenv(override=True)
    except ImportError:
        pass
    return {"User-Agent": os.environ.get("CREDITRISKLAB_SEC_UA", DEFAULT_UA), "Accept-Encoding": "gzip, deflate"}


def _get_json(url: str) -> dict[str, Any]:
    import requests  # imported lazily so offline tests never touch the network stack

    time.sleep(REQUEST_PAUSE_SECONDS)
    resp = requests.get(url, headers=_headers(), timeout=30)
    if resp.status_code == 404:
        raise EdgarError(f"404 from SEC for {url}")
    resp.raise_for_status()
    return resp.json()


def raw_cache_dir() -> Path:
    path = project_root() / "data" / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch_company_facts(cik: str, use_cache: bool = True) -> dict[str, Any]:
    cik = str(cik).lstrip("0").zfill(10)
    cache_path = raw_cache_dir() / f"companyfacts_{cik}.json"
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    payload = _get_json(COMPANYFACTS_URL.format(cik=cik))
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def resolve_cik_by_ticker(ticker: str, use_cache: bool = True) -> str | None:
    """Resolve via SEC's ticker map. Only covers current registrants — delisted issuers
    (which is most of the default cohort) will miss here and fall through to name search."""
    cache_path = raw_cache_dir() / "company_tickers.json"
    if use_cache and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        payload = _get_json(TICKERS_URL)
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
    for row in payload.values():
        if str(row.get("ticker", "")).upper() == ticker.upper():
            return str(row["cik_str"]).zfill(10)
    return None


def verify_cik(cik: str, expected_name_fragment: str) -> tuple[bool, str]:
    """Confirm a CIK actually belongs to the intended issuer before it is used as truth."""
    payload = _get_json(SUBMISSIONS_URL.format(cik=str(cik).lstrip("0").zfill(10)))
    name = str(payload.get("name", ""))
    former = " ".join(str(f.get("name", "")) for f in payload.get("formerNames", []) or [])
    haystack = f"{name} {former}".upper()
    needle = expected_name_fragment.upper().replace('"', "").strip()
    return (needle in haystack, name)


def list_filings(cik: str, form: str = "8-K", around=None, window_days: int = 60) -> pd.DataFrame:
    """Filing index for a CIK, including SEC's paginated history.

    The `recent` block in the submissions payload holds only the latest ~1,000 filings. For
    an issuer that kept filing heavily after its bankruptcy (iHeartMedia, Chesapeake,
    Hertz), the 8-K announcing the petition sits in an older page. With `around`, only the
    history pages whose date range covers the window are fetched.
    """
    from datetime import timedelta

    payload = _get_json(SUBMISSIONS_URL.format(cik=str(cik).lstrip("0").zfill(10)))
    filings = payload.get("filings", {})
    frames = [pd.DataFrame(filings.get("recent", {}))]
    for page in filings.get("files", []) or []:
        name = page.get("name")
        if not name:
            continue
        if around is not None:
            lo, hi = around - timedelta(days=window_days), around + timedelta(days=window_days)
            start = datetime.strptime(page.get("filingFrom", "1900-01-01"), "%Y-%m-%d").date()
            end = datetime.strptime(page.get("filingTo", "2999-12-31"), "%Y-%m-%d").date()
            if end < lo or start > hi:
                continue
        frames.append(pd.DataFrame(_get_json(f"https://data.sec.gov/submissions/{name}")))
    frame = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    if frame.empty:
        return pd.DataFrame(columns=["form", "filingDate", "items", "accessionNumber"])
    keep = [c for c in ("form", "filingDate", "items", "primaryDocDescription", "accessionNumber") if c in frame]
    frame = frame[keep]
    if form:
        frame = frame[frame["form"].astype(str).str.upper() == form.upper()]
    return frame.drop_duplicates().reset_index(drop=True)


def _facts_to_rows(facts: dict[str, Any], tags: list[str], duration: bool) -> list[dict[str, Any]]:
    gaap = facts.get("facts", {}).get("us-gaap", {})
    rows: list[dict[str, Any]] = []
    for tag in tags:
        block = gaap.get(tag)
        if not block:
            continue
        for unit, entries in block.get("units", {}).items():
            if not unit.startswith("USD"):
                continue
            for entry in entries:
                if str(entry.get("form", "")).upper() not in {"10-K", "10-K/A"}:
                    continue
                end = entry.get("end")
                filed = entry.get("filed")
                if not end or not filed:
                    continue
                if duration:
                    start = entry.get("start")
                    if not start:
                        continue
                    span = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days
                    if not ANNUAL_MIN_DAYS <= span <= ANNUAL_MAX_DAYS:
                        continue
                rows.append(
                    {
                        "period_end": end,
                        "filed": filed,
                        "value": entry.get("val"),
                        "form": entry.get("form"),
                        "fiscal_year": entry.get("fy"),
                        "tag": tag,
                        "tag_rank": tags.index(tag),
                    }
                )
    # No early exit across tags. Issuers migrate tags over time — ASC 606 moved most revenue
    # from SalesRevenueNet/Revenues to RevenueFromContractWithCustomer... in 2018 — so
    # stopping at the first tag with any data silently deletes the pre-migration history.
    # Preference is resolved per (period_end, filed) in `company_fundamentals`.
    return rows


def company_fundamentals(cik: str, ticker: str = "", use_cache: bool = True) -> pd.DataFrame:
    """Long-form fundamentals: one row per (period_end, filed, field).

    Deliberately long, not wide. Collapsing to wide requires an as-of date, and doing that
    at ingestion would bake in hindsight. The collapse happens in `features.point_in_time`.
    """
    facts = fetch_company_facts(cik, use_cache=use_cache)
    frames: list[pd.DataFrame] = []
    for field, tags in list(INSTANT_CONCEPTS.items()):
        rows = _facts_to_rows(facts, tags, duration=False)
        if rows:
            frame = pd.DataFrame(rows)
            frame["field"] = field
            frames.append(frame)
    for field, tags in list(DURATION_CONCEPTS.items()):
        rows = _facts_to_rows(facts, tags, duration=True)
        if rows:
            frame = pd.DataFrame(rows)
            frame["field"] = field
            frames.append(frame)
    if not frames:
        raise EdgarError(f"no usable 10-K XBRL facts for CIK {cik}")
    out = pd.concat(frames, ignore_index=True)
    out = resolve_tag_preference(out)
    out["cik"] = str(cik).lstrip("0").zfill(10)
    out["ticker"] = ticker
    out["source"] = "edgar"
    out["period_end"] = pd.to_datetime(out["period_end"]).dt.date
    out["filed"] = pd.to_datetime(out["filed"]).dt.date
    return out[["cik", "ticker", "field", "period_end", "filed", "form", "fiscal_year", "value", "tag", "source"]]


def resolve_tag_preference(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one value per (field, period_end, filed): the highest-preference tag reported."""
    if frame.empty:
        return frame
    ordered = frame.sort_values(["field", "period_end", "filed", "tag_rank"])
    return ordered.drop_duplicates(subset=["field", "period_end", "filed"], keep="first").reset_index(drop=True)


def as_of_today() -> date:
    return datetime.now(timezone.utc).date()
