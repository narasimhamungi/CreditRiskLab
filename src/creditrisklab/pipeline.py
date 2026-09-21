"""End-to-end run: panel -> models -> validation -> calibrated PD -> EL/capital -> grades.

The ordering here is deliberate. Validation runs before calibration, and calibration before
expected loss, because each stage can invalidate the next: if out-of-fold discrimination is
no better than Altman's Z'', the calibrated PDs are not worth computing and the report says
so instead of proceeding to a confident-looking loss number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from creditrisklab.config import feature_names, load_model_config, load_recovery_config
from creditrisklab.features.controls import control_group_hardness, hardness_verdict
from creditrisklab.features.panel import coverage_warnings, panel_diagnostics
from creditrisklab.models import altman, gbm, logistic
from creditrisklab.models.calibration import cross_fitted_calibration, prior_correct, sensitivity
from creditrisklab.risk import ead as ead_mod
from creditrisklab.risk import grades as grades_mod
from creditrisklab.risk import lgd as lgd_mod
from creditrisklab.risk.expected_loss import portfolio_expected_loss
from creditrisklab.universe import Issuer
from creditrisklab.validation import backtest, calibration_tests, discrimination, stability


@dataclass
class RunResult:
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    discrimination: dict[str, Any] = field(default_factory=dict)
    benchmark_comparison: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    coefficients: pd.DataFrame | None = None
    sign_violations: list[str] = field(default_factory=list)
    stability: pd.DataFrame | None = None
    grade_table: pd.DataFrame | None = None
    monotonicity: dict[str, Any] = field(default_factory=dict)
    portfolio: dict[str, Any] = field(default_factory=dict)
    exposures: pd.DataFrame | None = None
    predictions: pd.DataFrame | None = None
    pd_sensitivity: dict[float, dict[str, float]] = field(default_factory=dict)
    annual_backtest: pd.DataFrame | None = None
    portfolio_date: Any = None
    defaulters_without_positive: list[str] = field(default_factory=list)
    control_hardness: dict[str, Any] = field(default_factory=dict)


def run(
    panel: pd.DataFrame,
    issuers: list[Issuer],
    model_cfg: dict | None = None,
    recovery_cfg: dict | None = None,
    split_year: int | None = None,
    portfolio_date: Any = None,
) -> RunResult:
    cfg = model_cfg or load_model_config()
    rec = recovery_cfg or load_recovery_config()
    val = cfg["validation"]
    result = RunResult()

    result.diagnostics = panel_diagnostics(panel)
    result.warnings = coverage_warnings(panel)
    positives = set(panel.loc[panel["label"] == 1, "ticker"]) if not panel.empty else set()
    result.defaulters_without_positive = sorted(i.ticker for i in issuers if i.is_default and i.ticker not in positives)
    if result.defaulters_without_positive:
        result.warnings.append(
            "defaulted issuers with no usable pre-default observation (typically stopped filing "
            f"or filed too late to be visible at the observation date): {result.defaulters_without_positive}"
        )
    if not panel.empty:
        result.control_hardness = control_group_hardness(panel)
        result.control_hardness["verdict"] = hardness_verdict(result.control_hardness)
    if panel.empty or panel["label"].sum() == 0:
        result.warnings.append("no positive labels in the panel — nothing can be validated")
        return result

    X, y, groups = logistic.design_matrix(panel, cfg)
    features = feature_names(cfg)

    # ---- fitted models -------------------------------------------------------------
    fitted = logistic.fit(panel, cfg)
    result.coefficients = logistic.coefficients(fitted, cfg)
    result.sign_violations = logistic.sign_violations(fitted, cfg)

    # ---- out-of-fold predictions ---------------------------------------------------
    oof_lr = backtest.grouped_oof(
        X, y, groups,
        build_model=lambda: logistic.build_pipeline(cfg),
        n_splits=int(val.get("cv_folds", 5)),
        repeats=int(val.get("cv_repeats", 20)),
        random_state=int(val.get("random_state", 42)),
    )
    loo_lr = backtest.leave_one_issuer_out(X, y, groups, build_model=lambda: logistic.build_pipeline(cfg))

    scored = ~np.isnan(oof_lr)
    z_risk = altman.as_risk_score(panel)

    result.discrimination = {
        "logistic_oof": discrimination.bootstrap_auc(
            y[scored], oof_lr[scored], groups[scored],
            iterations=int(val.get("bootstrap_iterations", 2000)),
            random_state=int(val.get("random_state", 42)),
        ),
        "logistic_oof_ks": discrimination.ks_statistic(y[scored], oof_lr[scored]),
        "altman_zpp": discrimination.bootstrap_auc(
            y, z_risk, groups,
            iterations=int(val.get("bootstrap_iterations", 2000)),
            random_state=int(val.get("random_state", 42)),
        ),
    }
    loo_scored = ~np.isnan(loo_lr)
    if loo_scored.sum() > 0 and len(np.unique(y[loo_scored])) > 1:
        result.discrimination["logistic_leave_one_issuer_out"] = {
            "auc": discrimination.auc(y[loo_scored], loo_lr[loo_scored]),
            "n": int(loo_scored.sum()),
        }

    if gbm.enabled(cfg):
        oof_gbm = backtest.grouped_oof(
            X, y, groups,
            build_model=lambda: gbm.build_model(cfg),
            n_splits=int(val.get("cv_folds", 5)),
            repeats=int(val.get("cv_repeats", 20)),
            random_state=int(val.get("random_state", 42)),
            sample_weight_fn=gbm.class_weights,
        )
        g_scored = ~np.isnan(oof_gbm)
        if len(np.unique(y[g_scored])) > 1:
            result.discrimination["gbm_oof"] = discrimination.bootstrap_auc(
                y[g_scored], oof_gbm[g_scored], groups[g_scored],
                iterations=int(val.get("bootstrap_iterations", 2000)),
                random_state=int(val.get("random_state", 42)),
            )

    # ---- does the model actually beat the benchmark? -------------------------------
    result.benchmark_comparison = discrimination.paired_auc_difference(
        y[scored], oof_lr[scored], z_risk[scored], groups[scored],
        iterations=int(val.get("bootstrap_iterations", 2000)),
        random_state=int(val.get("random_state", 42)),
    )
    result.benchmark_comparison["verdict"] = _verdict(result.benchmark_comparison)

    # ---- out-of-time ----------------------------------------------------------------
    if split_year is not None:
        oot, test_mask = backtest.out_of_time(panel, X, y, lambda: logistic.build_pipeline(cfg), split_year)
        valid = test_mask & ~np.isnan(oot)
        if valid.sum() > 0 and len(np.unique(y[valid])) > 1:
            result.discrimination["logistic_out_of_time"] = {
                "auc": discrimination.auc(y[valid], oot[valid]),
                "split_year": split_year,
                "n_test": int(valid.sum()),
            }
        result.stability = stability.psi_by_feature(panel, features, split_year)

    # ---- calibration (cross-fitted by issuer) ------------------------------------
    cal_cfg = cfg["calibration"]
    sample_rate = float(np.mean(y[scored]))
    calibrated, cal_method = cross_fitted_calibration(
        oof_lr[scored], y[scored], groups[scored],
        method=str(cal_cfg.get("method", "isotonic")),
        n_splits=int(val.get("cv_folds", 5)),
        random_state=int(val.get("random_state", 42)),
    )
    tau = float(cal_cfg.get("population_default_rate", 0.02))
    corrected = prior_correct(calibrated, tau, sample_rate) if cal_cfg.get("prior_correction", True) else calibrated

    result.calibration = {
        "method": cal_method,
        "cross_fitted": True,
        "sample_default_rate": sample_rate,
        "population_default_rate": tau,
        "prior_correction_applied": bool(cal_cfg.get("prior_correction", True)),
        "mean_pd_before_correction": float(np.mean(calibrated)),
        "mean_pd_after_correction": float(np.mean(corrected)),
        "brier": calibration_tests.brier_decomposition(y[scored], calibrated),
        "slope_intercept": calibration_tests.calibration_slope_intercept(y[scored], calibrated),
        "hosmer_lemeshow": calibration_tests.hosmer_lemeshow(y[scored], calibrated),
        "reliability": calibration_tests.reliability_table(y[scored], calibrated),
    }

    grid = [float(g) for g in cal_cfg.get("sensitivity_grid", [tau])]
    for tau_i, values in sensitivity(calibrated, sample_rate, grid).items():
        result.pd_sensitivity[tau_i] = {"mean_pd": float(np.mean(values)), "max_pd": float(np.max(values))}

    keep = ["ticker", "name", "sector", "as_of", "label", "total_debt"]
    predictions = panel.loc[scored, [c for c in keep if c in panel.columns]].copy()
    predictions["raw_pd"] = oof_lr[scored]
    predictions["calibrated_pd"] = calibrated
    predictions["pd"] = corrected
    predictions["z_double_prime"] = [altman.z_double_prime(r) for _, r in panel.loc[scored].iterrows()]
    predictions["z_zone"] = [altman.zone(v) for v in predictions["z_double_prime"]]
    predictions["grade"] = grades_mod.assign_grades(predictions["pd"].to_numpy(), cfg)
    result.predictions = predictions.reset_index(drop=True)
    result.annual_backtest = backtest.annual_default_backtest(result.predictions, pd_column="calibrated_pd")

    # ---- grades ----------------------------------------------------------------------
    result.grade_table = grades_mod.grade_table(corrected, y[scored], cfg)
    result.monotonicity = grades_mod.monotonicity_check(result.grade_table)

    # ---- exposure, LGD, EL, capital ---------------------------------------------------
    result.portfolio_date = portfolio_date
    result.exposures = _build_exposures(result.predictions, issuers, rec, portfolio_date)
    if result.exposures is not None and not result.exposures.empty:
        result.portfolio = portfolio_expected_loss(result.exposures)

    return result


def _verdict(comparison: dict[str, Any]) -> str:
    if np.isnan(comparison.get("difference", np.nan)):
        return "not computable"
    if comparison.get("significant"):
        return "model beats Altman Z''" if comparison["difference"] > 0 else "Altman Z'' beats the model"
    return (
        "no distinguishable difference from Altman Z'' at this sample size — the fitted model "
        "is not demonstrably better than the published benchmark"
    )


def _build_exposures(
    predictions: pd.DataFrame,
    issuers: list[Issuer],
    recovery_cfg: dict,
    portfolio_date: Any = None,
) -> pd.DataFrame:
    """Issuer-level PD x LGD x EAD.

    With `portfolio_date`, the portfolio is a genuine cross-section: every issuer observed on
    that date, scored with information available then. Without it, each issuer is taken at its
    latest observation — different dates per issuer — which is a mechanics demonstration, not a
    portfolio, and the report labels it that way.

    EAD is a PROXY: reported total debt, fully drawn. Real EAD needs a facility schedule
    (drawn balances, committed undrawn lines), which no public filer discloses. The CCF
    machinery for real facility data lives in `risk.ead` and is tested.
    """
    by_ticker = {i.ticker: i for i in issuers}
    default_mix = {"senior_secured": 0.5, "senior_unsecured": 0.5}

    if portfolio_date is not None:
        mask = predictions["as_of"].astype(str).str[:10] == str(portfolio_date)[:10]
        chosen = predictions.loc[mask.to_numpy()]
    else:
        chosen = predictions.sort_values("as_of").groupby("ticker", as_index=False).last()

    rows = []
    for record in chosen.to_dict("records"):
        ticker = record["ticker"]
        issuer = by_ticker.get(ticker)
        mix = dict(issuer.exposure) if (issuer and issuer.exposure) else default_mix
        debt = record.get("total_debt")
        exposure = float(debt) if debt is not None and np.isfinite(float(debt)) and float(debt) > 0 else np.nan
        rows.append(
            {
                "ticker": ticker,
                "name": record.get("name"),
                "sector": record.get("sector"),
                "as_of": record.get("as_of"),
                "pd": float(record["pd"]),
                "lgd": float(lgd_mod.blended_lgd(mix, recovery_cfg)),
                "lgd_downturn": float(lgd_mod.downturn_lgd(mix, recovery_cfg)),
                "ead": exposure,
                "ead_basis": "PROXY — reported total debt, fully drawn",
                "lgd_mix_basis": "issuer mix from universe.yaml" if (issuer and issuer.exposure) else "default 50/50 secured/unsecured",
                "grade": record.get("grade"),
                "defaulted_within_12m": int(record.get("label", 0)),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    if frame["ead"].isna().all():
        frame["ead"] = 1.0
        frame["ead_basis"] = "PROXY — equal weighted; reported debt unavailable"
    else:
        frame["ead"] = frame["ead"].fillna(float(np.nanmedian(frame["ead"].to_numpy(dtype="float64"))))
    return frame


def facility_ead_example(facilities: list[ead_mod.Facility], recovery_cfg: dict | None = None) -> float:
    """Entry point for real facility data when a loan tape is available."""
    return ead_mod.portfolio_ead(facilities, recovery_cfg)
