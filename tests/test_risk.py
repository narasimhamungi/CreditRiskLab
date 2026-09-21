import numpy as np
import pandas as pd
import pytest

from creditrisklab.risk import ead, grades, lgd
from creditrisklab.risk.expected_loss import (
    expected_loss,
    irb_capital_requirement,
    portfolio_expected_loss,
    risk_weighted_assets,
)


def test_class_lgd_is_one_minus_sourced_recovery(recovery_cfg):
    assert lgd.class_lgd("senior_secured", recovery_cfg) == pytest.approx(1 - 0.58)


def test_blended_lgd_is_exposure_weighted(recovery_cfg):
    mix = {"senior_secured": 0.5, "senior_unsecured": 0.5}
    assert lgd.blended_lgd(mix, recovery_cfg) == pytest.approx(0.5 * 0.42 + 0.5 * 0.52)


def test_blending_different_recovery_bases_is_refused(recovery_cfg):
    with pytest.raises(lgd.LGDError, match="bases"):
        lgd.blended_lgd({"bank_loan_secured": 0.5, "senior_unsecured": 0.5}, recovery_cfg)


def test_unknown_seniority_is_rejected(recovery_cfg):
    with pytest.raises(lgd.LGDError):
        lgd.class_lgd("mezzanine_equity", recovery_cfg)


def test_blended_lgd_requires_full_mix(recovery_cfg):
    with pytest.raises(lgd.LGDError):
        lgd.blended_lgd({"senior_secured": 0.6}, recovery_cfg)


def test_downturn_lgd_exceeds_through_the_cycle(recovery_cfg):
    mix = {"senior_secured": 0.5, "senior_unsecured": 0.5}
    assert lgd.downturn_lgd(mix, recovery_cfg) > lgd.blended_lgd(mix, recovery_cfg)


def test_every_recovery_input_has_a_citation(recovery_cfg):
    cites = lgd.citations(recovery_cfg)
    assert cites and all(len(c) > 20 for c in cites.values())


def test_revolver_ead_applies_ccf_to_undrawn(recovery_cfg):
    f = ead.Facility("revolving_credit_facility", drawn=40.0, undrawn=60.0)
    assert ead.facility_ead(f, recovery_cfg) == pytest.approx(40 + 0.75 * 60)


def test_expected_loss_is_the_product():
    assert expected_loss(0.02, 0.45, 1_000_000) == pytest.approx(9_000)


def test_expected_loss_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        expected_loss(1.2, 0.45, 100)


def test_irb_matches_published_basel_risk_weight():
    """BCBS illustrative corporate risk weight: PD 1%, LGD 45%, M 2.5y -> 92.32%."""
    assert irb_capital_requirement(0.01, 0.45) * 12.5 == pytest.approx(0.9232, abs=0.001)


def test_irb_capital_rises_with_pd_and_lgd():
    assert irb_capital_requirement(0.03, 0.45) > irb_capital_requirement(0.01, 0.45)
    assert irb_capital_requirement(0.01, 0.60) > irb_capital_requirement(0.01, 0.45)


def test_rwa_is_k_times_12_5_times_ead():
    assert risk_weighted_assets(0.01, 0.45, 100.0) == pytest.approx(irb_capital_requirement(0.01, 0.45) * 12.5 * 100)


def test_portfolio_aggregation():
    frame = pd.DataFrame({"pd": [0.01, 0.05], "lgd": [0.4, 0.6], "ead": [100.0, 300.0]})
    out = portfolio_expected_loss(frame)
    assert out["expected_loss"] == pytest.approx(0.01 * 0.4 * 100 + 0.05 * 0.6 * 300)
    assert out["exposure_weighted_pd"] == pytest.approx((0.01 * 100 + 0.05 * 300) / 400)


def test_grade_boundaries(model_cfg):
    assert grades.assign_grade(0.0005, model_cfg) == "1"
    assert grades.assign_grade(0.0010, model_cfg) == "1"
    assert grades.assign_grade(0.0011, model_cfg) == "2"
    assert grades.assign_grade(0.5, model_cfg) == "7"
    assert grades.assign_grade(float("nan"), model_cfg) == "unrated"


def test_monotonicity_check_detects_inversion():
    table = pd.DataFrame({"grade": ["3", "4", "5"], "n": [10, 10, 10], "observed_default_rate": [0.0, 0.3, 0.1]})
    out = grades.monotonicity_check(table)
    assert not out["monotonic"] and out["violations"][0]["from_grade"] == "4"


def test_monotonicity_check_excludes_tiny_buckets():
    table = pd.DataFrame({"grade": ["3", "4", "5"], "n": [10, 1, 10], "observed_default_rate": [0.0, 1.0, 0.2]})
    out = grades.monotonicity_check(table)
    assert out["monotonic"] and out["grades_excluded_small_n"] == ["4"]
