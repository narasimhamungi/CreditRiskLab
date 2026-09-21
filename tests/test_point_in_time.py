from datetime import date

from creditrisklab.features.point_in_time import as_of_snapshot


def test_filing_after_as_of_is_invisible(make_facts):
    facts = make_facts([
        ("total_assets", date(2018, 12, 31), date(2019, 3, 1), 100.0),
        ("total_assets", date(2019, 12, 31), date(2020, 3, 1), 200.0),
    ])
    snap = as_of_snapshot(facts, date(2019, 6, 30))
    assert snap["period_end"] == date(2018, 12, 31)
    assert snap["total_assets"] == 100.0


def test_restatement_filed_after_as_of_does_not_leak(make_facts):
    # FY2018 originally 100, restated to 60 in a filing made AFTER the observation date.
    facts = make_facts([
        ("total_assets", date(2018, 12, 31), date(2019, 3, 1), 100.0),
        ("total_assets", date(2018, 12, 31), date(2019, 11, 1), 60.0),
    ])
    assert as_of_snapshot(facts, date(2019, 6, 30))["total_assets"] == 100.0
    assert as_of_snapshot(facts, date(2019, 12, 31))["total_assets"] == 60.0


def test_stale_filings_are_dropped_not_carried_forward(make_facts):
    facts = make_facts([("total_assets", date(2016, 12, 31), date(2017, 3, 1), 100.0)])
    assert as_of_snapshot(facts, date(2019, 6, 30), max_staleness_days=550) is None


def test_total_liabilities_derived_when_untagged(make_facts):
    facts = make_facts([
        ("total_assets", date(2018, 12, 31), date(2019, 3, 1), 100.0),
        ("equity", date(2018, 12, 31), date(2019, 3, 1), 30.0),
    ])
    snap = as_of_snapshot(facts, date(2019, 6, 30))
    assert snap["total_liabilities"] == 70.0
    assert "total_liabilities" in snap["derived_fields"]


def test_prior_period_attached_when_roughly_one_year_earlier(make_facts):
    facts = make_facts([
        ("net_income", date(2017, 12, 31), date(2018, 3, 1), -5.0),
        ("net_income", date(2018, 12, 31), date(2019, 3, 1), -7.0),
    ])
    snap = as_of_snapshot(facts, date(2019, 6, 30))
    assert snap["prior_net_income"] == -5.0


def test_empty_history_returns_none(make_facts):
    facts = make_facts([("total_assets", date(2020, 12, 31), date(2021, 3, 1), 1.0)])
    assert as_of_snapshot(facts, date(2019, 6, 30)) is None
