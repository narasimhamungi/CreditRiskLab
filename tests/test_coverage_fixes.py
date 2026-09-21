"""Regression tests for the gaps found on the first real run (coverage diagnostic)."""

from datetime import date

from creditrisklab.features.point_in_time import as_of_snapshot
from creditrisklab.features.ratios import compute_ratios, coverage_interest, total_debt
from creditrisklab.ingest import schema

FY = date(2018, 12, 31)
FILED = date(2019, 3, 1)


def test_stray_later_dated_fact_does_not_hijack_the_period(make_facts):
    facts = make_facts([
        ("total_assets", FY, FILED, 100.0),
        ("net_income", FY, FILED, 5.0),
        ("long_term_debt", date(2019, 2, 15), FILED, 40.0),  # subsequent-event style fact
    ])
    snap = as_of_snapshot(facts, date(2019, 6, 30))
    assert snap["period_end"] == FY and snap["net_income"] == 5.0


def test_ebit_derived_from_pretax_plus_gross_interest(make_facts):
    facts = make_facts([
        ("total_assets", FY, FILED, 100.0),
        ("pretax_income", FY, FILED, 8.0),
        ("interest_expense", FY, FILED, 2.0),
    ])
    snap = as_of_snapshot(facts, date(2019, 6, 30))
    assert snap["ebit"] == 10.0 and "ebit" in snap["derived_fields"]


def test_ebit_derived_from_pretax_minus_net_interest(make_facts):
    facts = make_facts([
        ("total_assets", FY, FILED, 100.0),
        ("pretax_income", FY, FILED, 8.0),
        ("net_interest", FY, FILED, -3.0),
    ])
    assert as_of_snapshot(facts, date(2019, 6, 30))["ebit"] == 11.0


def test_ebit_not_invented_without_interest_information(make_facts):
    facts = make_facts([("total_assets", FY, FILED, 100.0), ("pretax_income", FY, FILED, 8.0)])
    assert as_of_snapshot(facts, date(2019, 6, 30))["ebit"] is None


def test_reported_operating_income_is_never_overwritten(make_facts):
    facts = make_facts([
        ("total_assets", FY, FILED, 100.0),
        ("ebit", FY, FILED, 7.0),
        ("pretax_income", FY, FILED, 8.0),
        ("interest_expense", FY, FILED, 2.0),
    ])
    snap = as_of_snapshot(facts, date(2019, 6, 30))
    assert snap["ebit"] == 7.0 and "ebit" not in snap["derived_fields"]


def test_net_interest_used_for_coverage_only_when_it_is_an_expense():
    assert coverage_interest({"net_interest": -4.0}) == 4.0
    assert coverage_interest({"net_interest": 6.0}) is None
    assert coverage_interest({"interest_expense": 5.0, "net_interest": -4.0}) == 5.0


def test_coverage_from_net_interest_expense_flows_into_ratio():
    snap = {"total_assets": 100.0, "ebit": 12.0, "net_interest": -3.0}
    assert compute_ratios(snap)["interest_cover"] == 4.0


def test_reported_debt_total_used_only_when_components_absent():
    assert total_debt({"total_debt_reported": 50.0}) == 50.0
    assert total_debt({"long_term_debt": 30.0, "total_debt_reported": 50.0}) == 30.0


def test_schema_covers_tags_seen_on_real_filers():
    assert "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations" in schema.DURATION_CONCEPTS["cfo"]
    assert "LongTermDebtAndCapitalLeaseObligations" in schema.INSTANT_CONCEPTS["long_term_debt"]
    assert "LongTermDebtAndCapitalLeaseObligationsCurrent" in schema.INSTANT_CONCEPTS["current_debt"]
    assert "DebtAndCapitalLeaseObligations" in schema.INSTANT_CONCEPTS["total_debt_reported"]
    # net interest must never be treated as a gross expense tag
    assert not set(schema.DURATION_CONCEPTS["net_interest"]) & set(schema.DURATION_CONCEPTS["interest_expense"])


def test_approximate_tags_are_flagged_in_derived_fields(make_facts):
    facts = make_facts([
        ("total_assets", FY, FILED, 100.0),
        ("net_income", FY, FILED, -5.0, "NetIncomeLossAvailableToCommonStockholdersBasic"),
        ("interest_expense", FY, FILED, 2.0, "InterestExpenseDebtExcludingAmortization"),
    ])
    snap = as_of_snapshot(facts, date(2019, 6, 30))
    assert snap["net_income"] == -5.0
    assert "net_income~available_to_common" in snap["derived_fields"]
    assert "interest_expense~excl_amortisation" in snap["derived_fields"]


def test_later_restated_fact_stays_invisible_even_when_it_is_the_better_tag(make_facts):
    """Frontier FY2015: NetIncomeLoss was tagged only in a 2018 filing. At a 2016 observation
    date the model must use what was visible then, not the later, cleaner figure."""
    facts = make_facts([
        ("total_assets", date(2015, 12, 31), date(2016, 2, 25), 100.0),
        ("net_income", date(2015, 12, 31), date(2016, 2, 25), -316.0, "NetIncomeLossAvailableToCommonStockholdersBasic"),
        ("net_income", date(2015, 12, 31), date(2018, 3, 1), -196.0, "NetIncomeLoss"),
    ])
    assert as_of_snapshot(facts, date(2016, 6, 30))["net_income"] == -316.0
    assert as_of_snapshot(facts, date(2018, 6, 30), max_staleness_days=2000)["net_income"] == -196.0
