import pandas as pd

from creditrisklab.ingest import schema
from creditrisklab.ingest.edgar_client import _facts_to_rows, resolve_tag_preference


def _fact(end, filed, val, start=None, form="10-K"):
    entry = {"end": end, "filed": filed, "val": val, "form": form, "fy": int(filed[:4])}
    if start:
        entry["start"] = start
    return entry


def test_tag_migration_history_is_kept():
    """ASC 606 moved revenue tags in 2018. Both vintages must survive ingestion."""
    facts = {"facts": {"us-gaap": {
        "SalesRevenueNet": {"units": {"USD": [_fact("2016-12-31", "2017-03-01", 900, start="2016-01-01")]}},
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [_fact("2019-12-31", "2020-03-01", 1000, start="2019-01-01")]}},
    }}}
    rows = _facts_to_rows(facts, schema.DURATION_CONCEPTS["revenue"], duration=True)
    assert {r["period_end"] for r in rows} == {"2016-12-31", "2019-12-31"}


def test_quarterly_spans_are_excluded():
    facts = {"facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [
        _fact("2019-12-31", "2020-03-01", 50, start="2019-10-01"),
        _fact("2019-12-31", "2020-03-01", 200, start="2019-01-01"),
    ]}}}}}
    rows = _facts_to_rows(facts, ["NetIncomeLoss"], duration=True)
    assert [r["value"] for r in rows] == [200]


def test_non_annual_forms_are_excluded():
    facts = {"facts": {"us-gaap": {"Assets": {"units": {"USD": [
        _fact("2019-09-30", "2019-11-01", 1, form="10-Q"),
        _fact("2019-12-31", "2020-03-01", 2),
    ]}}}}}
    rows = _facts_to_rows(facts, ["Assets"], duration=False)
    assert [r["value"] for r in rows] == [2]


def test_tag_preference_resolved_per_period():
    frame = pd.DataFrame([
        {"field": "revenue", "period_end": "2019-12-31", "filed": "2020-03-01", "value": 1100, "tag": "Revenues", "tag_rank": 1},
        {"field": "revenue", "period_end": "2019-12-31", "filed": "2020-03-01", "value": 1000, "tag": "RevenueFromContract", "tag_rank": 0},
    ])
    out = resolve_tag_preference(frame)
    assert len(out) == 1 and out["value"].iloc[0] == 1000


def test_known_wrong_fallback_tags_are_not_candidates():
    assert "LiabilitiesAndStockholdersEquity" not in schema.INSTANT_CONCEPTS["total_liabilities"]
    assert "InterestIncomeExpenseNet" not in schema.DURATION_CONCEPTS["interest_expense"]
