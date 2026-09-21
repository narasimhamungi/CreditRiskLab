import shutil

import numpy as np

from creditrisklab.reporting import model_validation_report as report


def test_end_to_end_run_produces_every_section(run_result):
    r = run_result
    assert r.predictions is not None and len(r.predictions) > 50
    assert {"raw_pd", "calibrated_pd", "pd", "grade", "z_zone"} <= set(r.predictions.columns)
    assert r.calibration["cross_fitted"] is True
    assert r.benchmark_comparison["verdict"]
    assert r.annual_backtest is not None and not r.annual_backtest.empty
    assert r.control_hardness["verdict"]


def test_prior_correction_lowers_every_pd(run_result):
    p = run_result.predictions
    assert (p["pd"] <= p["calibrated_pd"] + 1e-12).all()


def test_portfolio_is_a_true_cross_section(run_result):
    dates = set(run_result.exposures["as_of"].astype(str))
    assert dates == {"2019-06-30"}
    assert run_result.portfolio["expected_loss"] > 0


def test_ead_is_labelled_as_a_proxy(run_result):
    assert run_result.exposures["ead_basis"].str.startswith("PROXY").all()


def test_report_puts_limitations_before_performance(run_result, recovery_cfg):
    text = report.render(run_result, recovery_cfg)
    assert text.index("Scope and limitations") < text.index("Discrimination")
    assert "cross-fitted" in text.lower()
    assert "PROXY" in text or "proxy" in text


def test_demo_output_is_watermarked(tmp_path, monkeypatch):
    from creditrisklab import cli, config

    root = config.project_root()
    shutil.copytree(root / "config", tmp_path / "config")
    monkeypatch.setenv("CREDITRISKLAB_ROOT", str(tmp_path))
    for loader in (config.load_universe, config.load_model_config, config.load_recovery_config):
        loader.cache_clear()
    try:
        cfg = config.load_model_config()
        cfg["validation"]["cv_repeats"] = 2
        cfg["validation"]["bootstrap_iterations"] = 100
        assert cli.main(["demo"]) == 0
        text = (tmp_path / "data" / "outputs" / "MODEL_VALIDATION_SYNTHETIC.md").read_text(encoding="utf-8")
        assert text.startswith("> **SYNTHETIC DATA")
        assert not (tmp_path / "data" / "outputs" / "MODEL_VALIDATION.md").exists()
    finally:
        for loader in (config.load_universe, config.load_model_config, config.load_recovery_config):
            loader.cache_clear()
