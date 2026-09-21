"""MODEL_VALIDATION.md generator.

The report is generated from the run, never hand-written. A hand-written validation report
drifts from the code within one commit, and the drift always runs in the flattering
direction. Limitations are emitted first, before any performance number, so a reader cannot
absorb the AUC before they absorb the sample size.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from creditrisklab.pipeline import RunResult
from creditrisklab.risk.lgd import citations


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{float(value) * 100:.{digits}f}%"


def _table(frame: pd.DataFrame | None, max_rows: int = 30) -> str:
    if frame is None or frame.empty:
        return "_no data_\n"
    return frame.head(max_rows).to_markdown(index=False, floatfmt=".4f") + "\n"


def render(result: RunResult, recovery_cfg: dict | None = None) -> str:
    d = result.diagnostics
    parts: list[str] = []
    parts.append("# CreditRiskLab — Model Validation Report\n\n")
    parts.append(f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from the pipeline run. Not hand-edited._\n")

    # ---- limitations first -------------------------------------------------------
    parts.append("\n## 1. Scope and limitations — read before any performance figure\n\n")
    parts.append(
        "This is a methodology demonstration on a deliberately small sample of real SEC filings, "
        "not a production PD model. It should not be used to underwrite anything.\n"
    )
    parts.append(
        f"- Observations: **{d.get('observations', 0)}** across **{d.get('issuers', 0)}** issuers\n"
        f"- Positive (default-within-12-months) observations: **{d.get('positive_labels', 0)}**, "
        f"drawn from **{d.get('defaulted_issuers_represented', 0)}** distinct defaulted issuers\n"
        f"- Sample default rate: **{_pct(d.get('sample_default_rate'))}** — by construction, far above the population rate\n"
    )
    if result.warnings:
        parts.append("\n**Warnings raised by the run:**\n")
        for warning in result.warnings:
            parts.append(f"- {warning}\n")
    parts.append(
        "\nThe non-default cohort is investment-grade large caps. The model therefore separates "
        "'large investment-grade issuer' from 'issuer approaching bankruptcy'. It has not been shown "
        "to discriminate *within* speculative grade, which is the harder and commercially relevant task. "
        "Expect every model on this sample — including the published benchmarks — to score a high AUC; "
        "a high AUC here says the task is easy, not that the model is good.\n"
        "\nThe prior correction in section 6 fixes the base rate. It does **not** fix a control group "
        "that is not a random sample of surviving firms. Corrected PDs are therefore indicative of "
        "level, not population-calibrated estimates.\n"
    )

    ch = result.control_hardness
    if ch:
        parts.append(
            "\n**Control-group hardness (rule-based, S&P leverage bands):** "
            f"{_pct(ch.get('speculative_like_share_true_survivors'), 1)} of surviving-issuer observations "
            "carry debt/EBITDA above 4x or interest cover below 2x.\n"
            f"Verdict: **{ch.get('verdict')}**\n"
        )

    # ---- data provenance ----------------------------------------------------------
    parts.append("\n## 2. Data provenance and point-in-time discipline\n")
    parts.append(
        "Fundamentals are SEC XBRL 10-K facts. Every fact carries its SEC `filed` date, and the "
        "as-of snapshot filters on that date before selecting the latest reported vintage. A "
        "restatement filed after the observation date is structurally unreachable rather than "
        "avoided by convention.\n\n"
        f"- Median reporting lag (period end to filing): **{_fmt(d.get('median_report_lag_days'), 0)} days**\n"
        f"- Source mix: {d.get('source_mix')}\n"
        f"- Observation years: {d.get('observation_years')}\n"
        f"- Highest feature missingness: {d.get('worst_missingness')}\n"
    )

    # ---- discrimination -----------------------------------------------------------
    parts.append("\n## 3. Discrimination\n")
    disc = result.discrimination
    rows = []
    for key in ("logistic_oof", "gbm_oof", "altman_zpp"):
        block = disc.get(key)
        if isinstance(block, dict) and "auc" in block:
            rows.append(
                {
                    "model": key,
                    "AUC": _fmt(block.get("auc")),
                    "95% CI (cluster bootstrap)": f"[{_fmt(block.get('lower'))}, {_fmt(block.get('upper'))}]",
                    "Gini": _fmt(2 * block["auc"] - 1 if block.get("auc") == block.get("auc") else float("nan")),
                }
            )
    for key in ("logistic_leave_one_issuer_out", "logistic_out_of_time"):
        block = disc.get(key)
        if isinstance(block, dict):
            rows.append({"model": key, "AUC": _fmt(block.get("auc")), "95% CI (cluster bootstrap)": "n/a", "Gini": ""})
    parts.append(_table(pd.DataFrame(rows)))
    parts.append(f"\nKS statistic (logistic, out-of-fold): **{_fmt(disc.get('logistic_oof_ks'))}**\n")
    parts.append(
        "\nConfidence intervals come from a bootstrap that resamples **issuers**, not rows. "
        "Resampling rows would treat one issuer's five annual observations as five independent "
        "facts and halve the apparent interval width.\n"
    )

    # ---- benchmark ----------------------------------------------------------------
    parts.append("\n## 4. Benchmark test — does the fitted model beat Altman Z''?\n")
    bc = result.benchmark_comparison
    parts.append(
        f"- Paired AUC difference (logistic minus Z''): **{_fmt(bc.get('difference'))}**\n"
        f"- 95% interval: [{_fmt(bc.get('lower'))}, {_fmt(bc.get('upper'))}]\n"
        f"- Verdict: **{bc.get('verdict', 'n/a')}**\n\n"
        "Z'' (Altman, Hartzell & Peck 1995) is used rather than the original 1968 Z because the "
        "default cohort is retail, telecom, media and energy — not listed manufacturers — and "
        "market value of equity is not available from XBRL filings.\n"
    )

    # ---- coefficients -------------------------------------------------------------
    parts.append("\n## 5. Coefficient review\n")
    parts.append(_table(result.coefficients))
    if result.sign_violations:
        parts.append("\n**Sign violations against credit priors (reported, not corrected):**\n")
        for violation in result.sign_violations:
            parts.append(f"- {violation}\n")
    else:
        parts.append("\nNo material coefficient contradicts its expected economic sign.\n")

    # ---- calibration --------------------------------------------------------------
    parts.append("\n## 6. Calibration and prior correction\n")
    cal = result.calibration
    parts.append(
        "Calibration is **cross-fitted by issuer**: each issuer's PD is mapped by a calibrator learned "
        "only from other issuers, so the metrics below are out-of-sample rather than a fitted curve "
        "scored on its own data.\n\n"
        f"- Calibration method actually fitted: **{cal.get('method')}**\n"
        f"- Sample default rate: **{_pct(cal.get('sample_default_rate'))}**\n"
        f"- Population default rate used for correction: **{_pct(cal.get('population_default_rate'))}** (ASSUMPTION)\n"
        f"- Mean PD before prior correction: **{_pct(cal.get('mean_pd_before_correction'))}**\n"
        f"- Mean PD after prior correction: **{_pct(cal.get('mean_pd_after_correction'))}**\n"
    )
    parts.append(
        "\nThe sample over-represents defaults by roughly an order of magnitude — unavoidable when "
        "building a default sample from SEC filings, and the reason the correction exists. Without it "
        "every PD, expected loss and capital figure in this report would be inflated by a similar "
        "multiple. The correction is monotone, so discrimination is unaffected.\n"
    )
    brier = cal.get("brier", {})
    slope = cal.get("slope_intercept", {})
    hl = cal.get("hosmer_lemeshow", {})
    parts.append(
        f"\n- Brier score: **{_fmt(brier.get('brier'))}** "
        f"(reliability {_fmt(brier.get('reliability'), 4)}, resolution {_fmt(brier.get('resolution'), 4)}, "
        f"uncertainty {_fmt(brier.get('uncertainty'), 4)})\n"
        f"- Calibration slope: **{_fmt(slope.get('slope'))}** (1.0 = perfect; below 1 means predictions are too extreme)\n"
        f"- Calibration intercept: **{_fmt(slope.get('intercept'))}**\n"
        f"- Hosmer-Lemeshow: chi2 {_fmt(hl.get('statistic'))}, dof {hl.get('dof')}, p {_fmt(hl.get('p_value'))} "
        f"— {hl.get('caveat', '')}\n"
    )
    parts.append("\n**Reliability by predicted-PD bucket**\n\n")
    parts.append(_table(cal.get("reliability")))

    if result.pd_sensitivity:
        parts.append("\n**PD sensitivity to the assumed population default rate**\n\n")
        sens = pd.DataFrame(
            [{"population_rate": k, "mean_pd": v["mean_pd"], "max_pd": v["max_pd"]} for k, v in sorted(result.pd_sensitivity.items())]
        )
        parts.append(_table(sens))

    # ---- stability ----------------------------------------------------------------
    if result.stability is not None and not result.stability.empty:
        parts.append("\n## 7. Population stability (early vs. late window)\n")
        parts.append(_table(result.stability))
        parts.append("\nPSI thresholds (0.10 / 0.25) are industry convention, not a statistical test.\n")

    # ---- grades -------------------------------------------------------------------
    parts.append("\n## 8. Risk-grade masterscale\n")
    parts.append(
        "Grades are assigned on prior-corrected PD. The observed default rate per grade is the "
        "**in-sample, oversampled** rate — read it for ordering across grades, not as a level to "
        "compare against the grade's PD band.\n\n"
    )
    parts.append(_table(result.grade_table))
    mono = result.monotonicity
    parts.append(
        f"\n- Monotonic across tested grades: **{mono.get('monotonic')}**\n"
        f"- Grades tested: {mono.get('grades_tested')}\n"
        f"- Grades excluded for small bucket size (< {mono.get('min_bucket')}): {mono.get('grades_excluded_small_n')}\n"
    )
    if mono.get("violations"):
        parts.append("- Violations:\n")
        for v in mono["violations"]:
            parts.append(
                f"  - grade {v['from_grade']} -> {v['to_grade']}: observed default rate falls "
                f"{_pct(v['from_rate'])} -> {_pct(v['to_rate'])}\n"
            )

    # ---- EL / capital --------------------------------------------------------------
    parts.append("\n## 9. Expected loss, capital and default backtest\n")
    if result.annual_backtest is not None and not result.annual_backtest.empty:
        parts.append(
            "\n**Annual backtest — expected vs. realised defaults (sample-scale PDs).** Sample-scale, "
            "not prior-corrected, because the realised counts are sample counts. z is the normal "
            "approximation to the Poisson-binomial; with one or two defaults a year it is indicative only.\n\n"
        )
        parts.append(_table(result.annual_backtest))
    if result.portfolio:
        p = result.portfolio
        if result.portfolio_date is not None:
            parts.append(
                f"\n**Cross-sectional portfolio as of {result.portfolio_date}** — every issuer observed on "
                "that date, scored only with filings visible then.\n\n"
            )
        else:
            parts.append(
                "\n**Illustrative aggregation — not a portfolio.** Each issuer is taken at its latest "
                "observation, so dates differ across issuers. This demonstrates the PD x LGD x EAD and "
                "IRB mechanics only.\n\n"
            )
        parts.append(
            f"- Issuers: **{len(result.exposures) if result.exposures is not None else 0}**\n"
            f"- Total EAD (proxy): **{p.get('total_ead', 0):,.0f}**\n"
            f"- Expected loss: **{p.get('expected_loss', 0):,.0f}** "
            f"({_pct(p.get('expected_loss_rate'))} of EAD)\n"
            f"- Exposure-weighted PD (prior-corrected): **{_pct(p.get('exposure_weighted_pd'))}**\n"
            f"- Exposure-weighted LGD: **{_pct(p.get('exposure_weighted_lgd'), 1)}**\n"
            f"- Basel IRB capital requirement: **{p.get('capital_requirement', 0):,.0f}**\n"
            f"- Risk-weighted assets: **{p.get('rwa', 0):,.0f}**\n"
        )
        if result.exposures is not None and "defaulted_within_12m" in result.exposures.columns:
            n_def = int(result.exposures["defaulted_within_12m"].sum())
            parts.append(f"- Issuers in this cross-section that defaulted within 12 months: **{n_def}**\n")
        parts.append(
            "\nEAD is a proxy — reported total debt, treated as fully drawn, excluding operating "
            "leases. Real EAD requires a facility schedule with committed undrawn lines, which no "
            "public filer discloses. The CCF machinery for facility data is implemented and tested "
            "in `risk.ead`.\n"
        )
    else:
        parts.append("_Portfolio aggregation not computed for this run._\n")

    parts.append("\n**Per-issuer exposure detail**\n\n")
    if result.exposures is not None and not result.exposures.empty:
        cols = ["ticker", "sector", "as_of", "grade", "pd", "lgd", "lgd_downturn", "ead", "defaulted_within_12m"]
        parts.append(_table(result.exposures[[c for c in cols if c in result.exposures.columns]]))

    # ---- citations -----------------------------------------------------------------
    parts.append("\n## 10. Sourced inputs\n")
    parts.append("Every recovery assumption below is SOURCED to published research, not assumed.\n\n")
    for name, citation in citations(recovery_cfg).items():
        parts.append(f"- **{name}** — {citation}\n")

    parts.append(
        "\n## 11. What would change the verdict\n"
        "- More defaulted issuers. The binding constraint is positives, not features.\n"
        "- A speculative-grade non-default cohort, so the model is tested on the discrimination "
        "task that actually matters commercially.\n"
        "- Macro covariates, which would separate point-in-time from through-the-cycle PD.\n"
        "- Facility-level exposure data, replacing the EAD proxy with a measured number.\n"
    )
    return "".join(parts)


def write(result: RunResult, path: str, recovery_cfg: dict | None = None) -> str:
    content = render(result, recovery_cfg)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path
