import numpy as np
import pandas as pd
import pytest

from creditrisklab.validation import backtest, calibration_tests, discrimination, stability


def test_auc_perfect_and_ks():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert discrimination.auc(y, s) == 1.0
    assert discrimination.gini(y, s) == 1.0
    assert discrimination.ks_statistic(y, s) == 1.0


def test_auc_undefined_with_one_class():
    assert np.isnan(discrimination.auc(np.zeros(4), np.arange(4)))


def test_cluster_bootstrap_interval_brackets_point():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(30), 4)
    y = (rng.random(120) < 0.2).astype(int)
    s = y * 0.5 + rng.random(120)
    out = discrimination.bootstrap_auc(y, s, groups, iterations=300)
    assert 0 <= out["lower"] <= out["auc"] <= out["upper"] <= 1


def test_paired_difference_of_identical_scores_is_not_significant():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(20), 3)
    y = (rng.random(60) < 0.3).astype(int)
    s = rng.random(60) + y
    out = discrimination.paired_auc_difference(y, s, s, groups, iterations=200)
    assert out["difference"] == 0 and not out["significant"]


def test_brier_decomposition_identity_holds_approximately():
    rng = np.random.default_rng(3)
    p = rng.random(4000)
    y = (rng.random(4000) < p).astype(int)
    d = calibration_tests.brier_decomposition(y, p, n_bins=20)
    assert d["brier"] == pytest.approx(d["reliability"] - d["resolution"] + d["uncertainty"], abs=0.01)


def test_calibration_slope_near_one_for_calibrated_predictions():
    rng = np.random.default_rng(4)
    p = rng.uniform(0.02, 0.98, 20000)
    y = (rng.random(20000) < p).astype(int)
    out = calibration_tests.calibration_slope_intercept(y, p)
    assert out["slope"] == pytest.approx(1.0, abs=0.08)
    assert out["intercept"] == pytest.approx(0.0, abs=0.08)


def test_hosmer_lemeshow_reports_caveat():
    out = calibration_tests.hosmer_lemeshow(np.array([0, 1] * 20), np.full(40, 0.5))
    assert 0 <= out["p_value"] <= 1 and "power" in out["caveat"]


def test_psi_zero_for_identical_and_material_for_shifted():
    rng = np.random.default_rng(5)
    a = rng.normal(0, 1, 5000)
    assert stability.psi(a, a) == pytest.approx(0.0, abs=1e-9)
    assert stability.psi(a, a + 1.5) > 0.25
    assert stability.interpret(0.3) == "material shift"


def test_leave_one_issuer_out_never_trains_on_the_test_issuer():
    seen = []

    class Spy:
        def fit(self, X, y, sample_weight=None):
            seen.append(set(X[:, 0].astype(int)))
            return self

        def predict_proba(self, X):
            return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])

    groups = np.repeat(np.arange(4), 3)
    X = groups.reshape(-1, 1).astype(float)
    y = np.array([0, 0, 1] * 4)
    backtest.leave_one_issuer_out(X, y, groups, build_model=Spy)
    for held_out, trained_on in zip(range(4), seen):
        assert held_out not in trained_on


def test_annual_backtest_expected_equals_sum_of_pds():
    preds = pd.DataFrame({
        "as_of": pd.to_datetime(["2019-06-30"] * 3 + ["2020-06-30"] * 2).date,
        "calibrated_pd": [0.1, 0.2, 0.3, 0.05, 0.05],
        "label": [0, 1, 0, 0, 0],
    })
    out = backtest.annual_default_backtest(preds)
    assert out.loc[out["year"] == 2019, "expected_defaults"].iloc[0] == pytest.approx(0.6)
    assert out.loc[out["year"] == 2019, "realised_defaults"].iloc[0] == 1
