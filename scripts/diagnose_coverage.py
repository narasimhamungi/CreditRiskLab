"""Coverage diagnostic for the first real run.

Answers three questions before any result is interpreted:
  1. Which issuers drive the missing features, and is missingness worse for defaulters?
     Median imputation on a defaulter makes it look like an average credit, which
     depresses discrimination; imputation on survivors does the opposite.
  2. For issuers with missing debt or interest, which us-gaap tags do they actually report?
     Those are candidates to add to ingest/schema.py.
  3. What do the Altman Z'' inputs look like per issuer? Explains a weak benchmark AUC.

Read-only. Uses data/outputs/panel.csv and the cached data/raw/companyfacts_*.json.

    python scripts/diagnose_coverage.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from creditrisklab.config import feature_names  # noqa: E402
from creditrisklab.ingest.schema import DURATION_CONCEPTS, INSTANT_CONCEPTS  # noqa: E402
from creditrisklab.models.altman import z_double_prime  # noqa: E402

TAG_PATTERN = re.compile(r"(Debt|Borrowing|NotesPayable|SeniorNotes|LineOfCredit|CommercialPaper|FinanceLease|Interest)", re.I)
IN_SCHEMA = {t for tags in list(INSTANT_CONCEPTS.values()) + list(DURATION_CONCEPTS.values()) for t in tags}


def main() -> int:
    panel_path = ROOT / "data" / "outputs" / "panel.csv"
    if not panel_path.exists():
        print("data/outputs/panel.csv not found — run `python -m creditrisklab.cli run` first")
        return 1
    panel = pd.read_csv(panel_path, dtype={"cik": str})
    panel["cik"] = panel["cik"].str.zfill(10)
    features = feature_names()
    out: list[str] = []

    # 1. missingness by issuer --------------------------------------------------------
    miss = panel.groupby("ticker")[features + ["total_debt"]].apply(lambda g: g.isna().mean()).round(2)
    miss = miss.loc[:, (miss > 0).any(axis=0)]
    miss.insert(0, "defaulter", panel.groupby("ticker")["is_default_issuer"].max())
    miss.insert(1, "obs", panel.groupby("ticker").size())
    out.append("=== 1. MISSING SHARE BY ISSUER (0 = complete) ===")
    out.append(miss.sort_values(["defaulter", "ticker"]).to_string())
    by_class = panel.groupby("is_default_issuer")[features].apply(lambda g: g.isna().mean()).T.round(2)
    by_class.columns = ["survivors", "defaulters"][: len(by_class.columns)]
    out.append("\nmissing share by cohort (features with any gap):")
    out.append(by_class.loc[(by_class > 0).any(axis=1)].to_string())

    # positive rows specifically: is the one observation that carries the label imputed?
    pos = panel.loc[panel["label"] == 1, ["ticker", "as_of"] + features]
    pos_gaps = pos.set_index(["ticker", "as_of"]).isna()
    pos_gaps = pos_gaps.loc[:, pos_gaps.any()].astype(int)
    out.append("\npositive (pre-default) observations with gaps (1 = missing):")
    out.append(pos_gaps.loc[pos_gaps.any(axis=1)].to_string() if not pos_gaps.empty else "none")

    # 2. debt / interest tags actually reported ---------------------------------------
    out.append("\n=== 2. DEBT / INTEREST TAGS REPORTED BY ISSUERS WITH GAPS ===")
    out.append("(* = already in schema; count = distinct annual period ends in 10-K facts)")
    gap_tickers = miss.index[(miss.drop(columns=["defaulter", "obs"]) > 0).any(axis=1)]
    for ticker in gap_tickers:
        cik = panel.loc[panel["ticker"] == ticker, "cik"].iloc[0]
        raw = ROOT / "data" / "raw" / f"companyfacts_{cik}.json"
        if not raw.exists():
            out.append(f"\n{ticker}: no cached companyfacts at {raw.name}")
            continue
        gaap = json.loads(raw.read_text(encoding="utf-8")).get("facts", {}).get("us-gaap", {})
        rows = []
        for tag, block in gaap.items():
            if not TAG_PATTERN.search(tag):
                continue
            ends = set()
            for unit, entries in block.get("units", {}).items():
                if unit.startswith("USD"):
                    ends |= {e.get("end") for e in entries if str(e.get("form", "")).startswith("10-K")}
            if ends:
                rows.append((tag, len(ends), min(ends)[:4], max(ends)[:4]))
        rows.sort(key=lambda r: -r[1])
        out.append(f"\n{ticker}")
        for tag, n, lo, hi in rows[:18]:
            out.append(f"  {'*' if tag in IN_SCHEMA else ' '} {tag:<70s} {n:3d}  {lo}-{hi}")

    # 3. Z'' inputs -------------------------------------------------------------------
    out.append("\n=== 3. ALTMAN Z'' INPUTS, MEDIAN BY ISSUER ===")
    panel["z_dp"] = [z_double_prime(r) for _, r in panel.iterrows()]
    cols = ["wc_ta", "re_ta", "ebit_ta", "equity_tl", "z_dp"]
    z = panel.groupby("ticker")[cols].median().round(3)
    z.insert(0, "defaulter", panel.groupby("ticker")["is_default_issuer"].max())
    out.append(z.sort_values(["defaulter", "z_dp"]).to_string())
    out.append("Z'' zones: > 2.60 safe, 1.10-2.60 grey, < 1.10 distress")

    # 4. raw fields behind each gap -------------------------------------------------------
    facts_path = ROOT / "data" / "interim" / "fundamentals.csv"
    if facts_path.exists():
        from creditrisklab.features.point_in_time import as_of_snapshot
        from creditrisklab.ingest.schema import CANONICAL_FIELDS

        facts = pd.read_csv(facts_path, dtype={"cik": str})
        facts["cik"] = facts["cik"].str.zfill(10)
        facts["period_end"] = pd.to_datetime(facts["period_end"]).dt.date
        facts["filed"] = pd.to_datetime(facts["filed"]).dt.date
        out.append("\n=== 4. RAW CANONICAL FIELDS MISSING BEHIND EACH GAP ROW ===")
        gap_rows = panel.loc[panel[features].isna().any(axis=1)]
        for _, row in gap_rows.iterrows():
            as_of = pd.to_datetime(row["as_of"]).date()
            snap = as_of_snapshot(facts.loc[facts["cik"] == row["cik"]], as_of)
            if snap is None:
                continue
            missing = [f for f in CANONICAL_FIELDS if snap.get(f) is None]
            flag = "  <- POSITIVE" if row["label"] == 1 else ""
            out.append(f"{row['ticker']:5s} {as_of}  period={snap['period_end']}  derived={snap.get('derived_fields')}  missing={missing}{flag}")

        # 5. which tags exist for the still-missing fields, at the snapshot's own period --
        probes = {
            "revenue": r"Revenue|Sales",
            "net_income": r"NetIncome|ProfitLoss|NetEarnings",
            "interest_expense": r"InterestExpense|InterestAndDebtExpense",
        }
        out.append("\n=== 5. TAGS PRESENT AT THE SNAPSHOT PERIOD FOR STILL-MISSING FIELDS ===")
        out.append("(* = already in schema; span = days covered; only 10-K USD facts ending within 7 days of the period)")
        seen: set[tuple[str, str]] = set()
        for _, row in gap_rows.iterrows():
            as_of = pd.to_datetime(row["as_of"]).date()
            snap = as_of_snapshot(facts.loc[facts["cik"] == row["cik"]], as_of)
            if snap is None:
                continue
            raw = ROOT / "data" / "raw" / f"companyfacts_{row['cik']}.json"
            if not raw.exists():
                continue
            gaap = None
            for field, pattern in probes.items():
                if snap.get(field) is not None or (row["ticker"], field) in seen:
                    continue
                seen.add((row["ticker"], field))
                gaap = gaap or json.loads(raw.read_text(encoding="utf-8")).get("facts", {}).get("us-gaap", {})
                hits = []
                for tag, block in gaap.items():
                    if not re.search(pattern, tag):
                        continue
                    for unit, entries in block.get("units", {}).items():
                        if not unit.startswith("USD"):
                            continue
                        for e in entries:
                            if not str(e.get("form", "")).startswith("10-K") or not e.get("start") or not e.get("end"):
                                continue
                            end = pd.to_datetime(e["end"]).date()
                            if abs((end - snap["period_end"]).days) > 7:
                                continue
                            span = (end - pd.to_datetime(e["start"]).date()).days
                            hits.append((tag, span, e.get("val"), e.get("filed")))
                hits = sorted(set(hits), key=lambda h: (-(300 <= h[1] <= 430), h[0]))[:10]
                out.append(f"\n{row['ticker']} {field} @ {snap['period_end']}")
                if not hits:
                    out.append("    no matching tag at this period")
                for tag, span, val, filed in hits:
                    out.append(f"  {'*' if tag in IN_SCHEMA else ' '} {tag:<75s} span={span:4d} val={val} filed={filed}")

    text = "\n".join(out)
    print(text)
    target = ROOT / "data" / "interim" / "coverage_diagnostic.txt"
    target.write_text(text, encoding="utf-8")
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
