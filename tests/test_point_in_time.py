from datetime import date

import pytest
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


def test_ebit_prefers_gross_profit_derivation_over_pretax_plus_interest(make_facts):
    """The actual bug this guards against: J&J has carried no OperatingIncomeLoss
    tag since FY2015 Q1, and the two derivations disagree by 31% (~$8B) on its
    real FY2025 figures. gross_profit - sga_expense - rnd_expense matches J&J's
    externally reported operating income ($25.596B) to the dollar; pretax_income
    + interest_expense does not ($33.552B). Both are present in this fixture on
    purpose, to prove the gross-profit path is the one actually chosen, not just
    available."""
    facts = make_facts([
        ("gross_profit", date(2025, 12, 28), date(2026, 1, 21), 63_937_000_000.0),
        ("sga_expense", date(2025, 12, 28), date(2026, 1, 21), 23_676_000_000.0),
        ("rnd_expense", date(2025, 12, 28), date(2026, 1, 21), 14_665_000_000.0),
        ("pretax_income", date(2025, 12, 28), date(2026, 1, 21), 32_581_000_000.0),
        ("interest_expense", date(2025, 12, 28), date(2026, 1, 21), 971_000_000.0),
    ])
    snap = as_of_snapshot(facts, date(2026, 6, 30))
    assert snap["ebit"] == pytest.approx(25_596_000_000.0)
    assert "ebit" in snap["derived_fields"]


def test_ebit_falls_back_to_pretax_plus_interest_when_gross_profit_untagged(make_facts):
    """Not removed, just demoted: some issuers won't tag gross_profit/sga/rnd
    either, and this is still better than leaving EBIT missing for them."""
    facts = make_facts([
        ("pretax_income", date(2018, 12, 31), date(2019, 3, 1), 80.0),
        ("interest_expense", date(2018, 12, 31), date(2019, 3, 1), 20.0),
    ])
    snap = as_of_snapshot(facts, date(2019, 6, 30))
    assert snap["ebit"] == 100.0
    assert "ebit" in snap["derived_fields"]


def test_gross_profit_derived_from_revenue_minus_cost_of_revenue_when_untagged(make_facts):
    """Mirrors Trellis's identical fallback. Chained: this must fire before
    _derive_ebit runs, or EBIT falls through to the weaker pretax-based
    formula even though gross_profit was recoverable."""
    facts = make_facts([
        ("revenue", date(2018, 12, 31), date(2019, 3, 1), 100.0),
        ("cost_of_revenue", date(2018, 12, 31), date(2019, 3, 1), 60.0),
        ("sga_expense", date(2018, 12, 31), date(2019, 3, 1), 20.0),
        ("rnd_expense", date(2018, 12, 31), date(2019, 3, 1), 5.0),
    ])
    snap = as_of_snapshot(facts, date(2019, 6, 30))
    assert snap["gross_profit"] == 40.0
    assert "gross_profit" in snap["derived_fields"]
    assert snap["ebit"] == 15.0  # 40 - 20 - 5, the gross-profit path, not pretax-based
    assert "ebit" in snap["derived_fields"]


def test_ebit_falls_back_when_sga_expense_is_missing(make_facts):
    """gross_profit and sga_expense are still strictly required -- unlike
    rnd_expense (below), their absence has no economically meaningful
    default, so a partial set here must not be treated as sufficient."""
    facts = make_facts([
        ("gross_profit", date(2018, 12, 31), date(2019, 3, 1), 100.0),
        ("pretax_income", date(2018, 12, 31), date(2019, 3, 1), 80.0),
        ("interest_expense", date(2018, 12, 31), date(2019, 3, 1), 20.0),
    ])
    snap = as_of_snapshot(facts, date(2019, 6, 30))
    assert snap["ebit"] == 100.0  # the pretax+interest path, not gross_profit alone
    assert "ebit" in snap["derived_fields"]


def test_ebit_treats_missing_rnd_expense_as_zero_not_as_blocking(make_facts):
    """Regression test for a real bug: the first version of this fix required
    rnd_expense strictly non-None, one guard stricter than Trellis's own
    validated derivation (`data.get("rnd_expense", 0.0)`). On live data this
    made RAD (a pharmacy retailer, no distinct R&D line) fall through to the
    weaker pretax-based path and disagree with Trellis by -410%, and NKE by
    +1.4% -- both companies simply don't tag rnd_expense, which is a real,
    common, economically meaningful zero (no R&D spend), not missing data."""
    facts = make_facts([
        ("gross_profit", date(2018, 12, 31), date(2019, 3, 1), 100.0),
        ("sga_expense", date(2018, 12, 31), date(2019, 3, 1), 30.0),
        # deliberately no rnd_expense fact at all
        ("pretax_income", date(2018, 12, 31), date(2019, 3, 1), 999.0),  # would give a
        ("interest_expense", date(2018, 12, 31), date(2019, 3, 1), 999.0),  # very different
    ])                                                                      # wrong answer if used
    snap = as_of_snapshot(facts, date(2019, 6, 30))
    assert snap["ebit"] == 70.0  # 100 - 30 - 0, the gross-profit path with rnd=0
    assert "ebit" in snap["derived_fields"]


def test_ebit_not_derived_when_operating_income_already_tagged(make_facts):
    facts = make_facts([
        ("ebit", date(2018, 12, 31), date(2019, 3, 1), 55.0),
        ("gross_profit", date(2018, 12, 31), date(2019, 3, 1), 100.0),
        ("sga_expense", date(2018, 12, 31), date(2019, 3, 1), 20.0),
        ("rnd_expense", date(2018, 12, 31), date(2019, 3, 1), 10.0),
    ])
    snap = as_of_snapshot(facts, date(2019, 6, 30))
    assert snap["ebit"] == 55.0  # the tagged value, not the derived 70.0
    assert "ebit" not in snap["derived_fields"]


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
