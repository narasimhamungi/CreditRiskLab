import numpy as np
import pandas as pd
import pytest

from creditrisklab.config import feature_names
from creditrisklab.models import altman, gbm, logistic, ohlson
from creditrisklab.models.calibration import cross_fitted_calibration, prior_correct


def test_z_double_prime_known_value():
    row = {"wc_ta": 0.1, "re_ta": 0.2, "ebit_ta": 0.05, "equity_tl": 0.5}
    expected = 6.56 * 0.1 + 3.26 * 0.2 + 6.72 * 0.05 + 1.05 * 0.5
    assert altman.z_double_prime(row) == pytest.approx(expected)
    assert altman.z_double_prime(row, include_constant=True) == pytest.approx(expected + 3.25)


def test_z_zones():
    assert altman.zone(3.0) == "safe"
    assert altman.zone(0.5) == "distress"
    assert altman.zone(2.0) == "grey"
    assert altman.zone(None) == "unscored"


def test_z_missing_input_is_unscored_not_zero():
    assert altman.z_double_prime({"wc_ta": 0.1, "re_ta": None, "ebit_ta": 0.1, "equity_tl": 1.0}) is None


def test_ohlson_leverage_raises_score():
    base = {"total_assets": 1000, "total_liabilities": 400, "current_assets": 300, "current_liabilities": 200,
            "net_income": 50, "cfo": 80, "prior_net_income": 40}
    riskier = {**base, "total_liabilities": 950}
    assert ohlson.o_score(riskier) > ohlson.o_score(base)
    assert 0 < ohlson.o_score_pd(base) < 1


def test_ohlson_missing_inputs_return_none():
    assert ohlson.o_score({"total_assets": 1000}) is None


def test_logistic_fits_and_predicts_probabilities(panel, fast_cfg):
    pipe = logistic.fit(panel, fast_cfg)
    X, _, _ = logistic.design_matrix(panel, fast_cfg)
    p = pipe.predict_proba(X)[:, 1]
    assert p.shape == (len(panel),) and ((p >= 0) & (p <= 1)).all()


def test_winsorizer_bounds_come_from_training_data_only():
    w = logistic.Winsorizer(0.1, 0.9).fit(np.arange(100, dtype=float).reshape(-1, 1))
    clipped = w.transform(np.array([[1e9]]))
    assert clipped[0, 0] == pytest.approx(np.quantile(np.arange(100), 0.9))


def test_coefficient_table_carries_sign_check(panel, fast_cfg):
    table = logistic.coefficients(logistic.fit(panel, fast_cfg), fast_cfg)
    assert set(table["feature"]) == set(feature_names(fast_cfg))
    assert "sign_consistent" in table.columns


def test_design_matrix_rejects_missing_features(panel, fast_cfg):
    with pytest.raises(KeyError):
        logistic.design_matrix(panel.drop(columns=["wc_ta"]), fast_cfg)


def test_gbm_balanced_weights():
    w = gbm.class_weights(np.array([1, 0, 0, 0]))
    assert w[0] * 1 == pytest.approx(w[1] * 3)


def test_prior_correction_known_value():
    # sample rate 50%, population 2%: odds scale by 0.02/0.98 exactly
    out = prior_correct(np.array([0.5]), 0.02, 0.5)
    assert out[0] == pytest.approx(0.02)


def test_prior_correction_preserves_ranking_and_is_identity_at_equal_rates():
    p = np.array([0.05, 0.2, 0.6, 0.9])
    corrected = prior_correct(p, 0.02, 0.1)
    assert (np.diff(corrected) > 0).all() and (corrected < p).all()
    assert prior_correct(p, 0.1, 0.1) == pytest.approx(p)


def test_prior_correction_rejects_invalid_rates():
    with pytest.raises(ValueError):
        prior_correct(np.array([0.5]), 0.0, 0.1)


def test_cross_fitted_calibration_uses_platt_on_small_samples(panel):
    rng = np.random.default_rng(0)
    y = panel["label"].to_numpy()
    p = np.clip(0.1 + 0.6 * y + rng.normal(0, 0.05, len(y)), 0.01, 0.99)
    out, method = cross_fitted_calibration(p, y, panel["ticker"].to_numpy())
    assert out.shape == p.shape and method == "platt"
