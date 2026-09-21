from datetime import date

import pandas as pd

from creditrisklab.features.controls import control_group_hardness, hardness_verdict, speculative_mask
from creditrisklab.features.panel import coverage_warnings
from creditrisklab.features.ratios import CAPS, compute_ratios, total_debt
from creditrisklab.labels.default_events import label_observation, observation_dates
from creditrisklab.universe import Issuer

BASE = {
    "total_assets": 1000.0, "total_liabilities": 600.0, "current_assets": 300.0, "current_liabilities": 200.0,
    "retained_earnings": 150.0, "equity": 400.0, "ebit": 100.0, "depreciation_amortisation": 20.0,
    "revenue": 900.0, "net_income": 60.0, "cfo": 110.0, "interest_expense": 25.0,
    "long_term_debt": 300.0, "current_debt": 50.0, "prior_net_income": 55.0,
}


def test_core_ratios():
    r = compute_ratios(dict(BASE))
    assert r["wc_ta"] == 0.1
    assert r["equity_tl"] == 400 / 600
    assert r["debt_ebitda"] == 350 / 120
    assert r["interest_cover"] == 4.0


def test_negative_ebitda_maps_to_worst_leverage_not_missing():
    r = compute_ratios({**BASE, "ebit": -50.0, "depreciation_amortisation": 10.0})
    assert r["debt_ebitda"] == CAPS["debt_ebitda"][1]


def test_missing_interest_is_missing_not_strong_coverage():
    snap = {k: v for k, v in BASE.items() if k != "interest_expense"}
    assert compute_ratios(snap)["interest_cover"] is None


def test_reported_zero_interest_is_strong_coverage():
    assert compute_ratios({**BASE, "interest_expense": 0.0})["interest_cover"] == CAPS["interest_cover"][1]


def test_inclusive_long_term_debt_tag_not_double_counted():
    snap = {**BASE, "_tags": {"long_term_debt": "LongTermDebt", "current_debt": "LongTermDebtCurrent"}}
    assert total_debt(snap) == 300.0
    snap["_tags"]["current_debt"] = "ShortTermBorrowings"
    assert total_debt(snap) == 350.0


def test_zero_denominator_returns_none():
    assert compute_ratios({**BASE, "current_liabilities": 0.0})["current_ratio"] is None


def test_two_year_loss_flag():
    assert compute_ratios({**BASE, "net_income": -1.0, "prior_net_income": -2.0})["two_year_loss"] == 1.0
    assert compute_ratios(dict(BASE))["two_year_loss"] == 0.0


def _defaulter():
    return Issuer("D", "D", "s", cik="1", default_date=date(2020, 5, 15), verified=True, exposure={"senior_secured": 1.0})


def test_label_inside_and_outside_horizon():
    d = _defaulter()
    assert label_observation(d, date(2019, 6, 30)) == 1
    assert label_observation(d, date(2018, 6, 30)) == 0


def test_observation_after_default_is_dropped_not_labelled_safe():
    assert label_observation(_defaulter(), date(2020, 6, 30)) is None


def test_observation_grid_stops_at_default():
    grid = observation_dates(_defaulter(), (date(2015, 1, 1), date(2024, 12, 31)))
    assert max(grid) == date(2019, 6, 30)


def test_non_default_issuer_always_zero():
    assert label_observation(Issuer("N", "N", "s", cik="2"), date(2019, 6, 30)) == 0


def test_panel_has_at_most_one_positive_per_defaulter(panel):
    counts = panel.loc[panel["label"] == 1].groupby("ticker").size()
    assert (counts <= 1).all() and len(counts) > 0


def test_panel_never_uses_a_filing_after_its_observation_date(panel):
    """The point-in-time invariant, checked across every row of the panel."""
    assert (pd.to_datetime(panel["filed"]) <= pd.to_datetime(panel["as_of"])).all()


def test_coverage_warning_fires_on_thin_positives(panel):
    thin = panel.loc[(panel["label"] == 0) | (panel["ticker"].isin(panel.loc[panel["label"] == 1, "ticker"].head(3)))]
    assert any("positive observations" in w for w in coverage_warnings(thin))


def test_speculative_mask_uses_leverage_or_coverage():
    frame = pd.DataFrame({"debt_ebitda": [2.0, 5.0, 2.0], "interest_cover": [8.0, 8.0, 1.5]})
    assert speculative_mask(frame).tolist() == [False, True, True]


def test_easy_control_group_is_called_out(panel):
    stats = control_group_hardness(panel)
    assert "speculative_like_share_true_survivors" in stats
    assert hardness_verdict({"speculative_like_share_true_survivors": 0.0}).startswith("EASY TASK")
