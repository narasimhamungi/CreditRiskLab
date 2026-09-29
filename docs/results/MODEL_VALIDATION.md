# CreditRiskLab — Model Validation Report

_Generated 2026-09-29 17:16 UTC from the pipeline run. Not hand-edited._

## 1. Scope and limitations — read before any performance figure

This is a methodology demonstration on a deliberately small sample of real SEC filings, not a production PD model. It should not be used to underwrite anything.
- Observations: **105** across **15** issuers
- Positive (default-within-12-months) observations: **10**, drawn from **10** distinct defaulted issuers
- Sample default rate: **9.52%** — by construction, far above the population rate

The non-default cohort is investment-grade large caps. The model therefore separates 'large investment-grade issuer' from 'issuer approaching bankruptcy'. It has not been shown to discriminate *within* speculative grade, which is the harder and commercially relevant task. Expect every model on this sample — including the published benchmarks — to score a high AUC; a high AUC here says the task is easy, not that the model is good.

The prior correction in section 6 fixes the base rate. It does **not** fix a control group that is not a random sample of surviving firms. Corrected PDs are therefore indicative of level, not population-calibrated estimates.

**Control-group hardness (rule-based, S&P leverage bands):** 2.0% of surviving-issuer observations carry debt/EBITDA above 4x or interest cover below 2x.
Verdict: **EASY TASK — under 10% of surviving-issuer observations look speculative-grade. A high AUC on this design measures the gap between mega-cap IG and bankruptcy, not model skill.**

## 2. Data provenance and point-in-time discipline
Fundamentals are SEC XBRL 10-K facts. Every fact carries its SEC `filed` date, and the as-of snapshot filters on that date before selecting the latest reported vintage. A restatement filed after the observation date is structurally unreachable rather than avoided by convention.

- Median reporting lag (period end to filing): **54 days**
- Source mix: {'edgar': 105}
- Observation years: [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
- Highest feature missingness: {'wc_ta': 0.029, 'current_ratio': 0.029, 'two_year_loss': 0.029, 'interest_cover': 0.01, 'equity_tl': 0.0}

## 3. Discrimination
| model                         |    AUC | 95% CI (cluster bootstrap)   |   Gini |
|:------------------------------|-------:|:-----------------------------|-------:|
| logistic_oof                  | 0.8240 | [0.679, 0.935]               | 0.6480 |
| gbm_oof                       | 0.7460 | [0.577, 0.902]               | 0.4920 |
| altman_zpp                    | 0.7730 | [0.659, 0.912]               | 0.5450 |
| logistic_leave_one_issuer_out | 0.8350 | n/a                          |        |
| logistic_out_of_time          | 0.9480 | n/a                          |        |

KS statistic (logistic, out-of-fold): **0.653**

Confidence intervals come from a bootstrap that resamples **issuers**, not rows. Resampling rows would treat one issuer's five annual observations as five independent facts and halve the apparent interval width.

## 4. Benchmark test — does the fitted model beat Altman Z''?
- Paired AUC difference (logistic minus Z''): **0.052**
- 95% interval: [-0.097, 0.220]
- Verdict: **no distinguishable difference from Altman Z'' at this sample size — the fitted model is not demonstrably better than the published benchmark**

Z'' (Altman, Hartzell & Peck 1995) is used rather than the original 1968 Z because the default cohort is retail, telecom, media and energy — not listed manufacturers — and market value of equity is not available from XBRL filings.

## 5. Coefficient review
| feature         |   coefficient |   expected_sign |   actual_sign | sign_consistent   |
|:----------------|--------------:|----------------:|--------------:|:------------------|
| cfo_total_debt  |       -1.2781 |              -1 |            -1 | True              |
| equity_tl       |       -1.1680 |              -1 |            -1 | True              |
| current_ratio   |       -1.0373 |              -1 |            -1 | True              |
| ebit_ta         |        0.7263 |              -1 |             1 | False             |
| two_year_loss   |        0.6978 |               1 |             1 | True              |
| interest_cover  |       -0.6738 |              -1 |            -1 | True              |
| negative_equity |       -0.5457 |               1 |            -1 | False             |
| net_margin      |        0.4580 |              -1 |             1 | False             |
| debt_ebitda     |        0.4254 |               1 |             1 | True              |
| log_assets      |       -0.3518 |              -1 |            -1 | True              |
| wc_ta           |       -0.3497 |              -1 |            -1 | True              |
| re_ta           |       -0.2070 |              -1 |            -1 | True              |

**Sign violations against credit priors (reported, not corrected):**
- ebit_ta: coefficient +0.726 contradicts the expected sign (higher value => lower PD)
- negative_equity: coefficient -0.546 contradicts the expected sign (higher value => higher PD)
- net_margin: coefficient +0.458 contradicts the expected sign (higher value => lower PD)

## 6. Calibration and prior correction
Calibration is **cross-fitted by issuer**: each issuer's PD is mapped by a calibrator learned only from other issuers, so the metrics below are out-of-sample rather than a fitted curve scored on its own data.

- Calibration method actually fitted: **platt**
- Sample default rate: **9.52%**
- Population default rate used for correction: **2.00%** (ASSUMPTION)
- Mean PD before prior correction: **9.66%**
- Mean PD after prior correction: **2.29%**

The sample over-represents defaults by roughly an order of magnitude — unavoidable when building a default sample from SEC filings, and the reason the correction exists. Without it every PD, expected loss and capital figure in this report would be inflated by a similar multiple. The correction is monotone, so discrimination is unaffected.

- Brier score: **0.081** (reliability 0.0130, resolution 0.0169, uncertainty 0.0862)
- Calibration slope: **0.947** (1.0 = perfect; below 1 means predictions are too extreme)
- Calibration intercept: **-0.109**
- Hosmer-Lemeshow: chi2 7.517, dof 8, p 0.482 — low power at this sample size; a non-rejection is not evidence of calibration

**Reliability by predicted-PD bucket**

|   bucket |       n |   mean_predicted_pd |   observed_default_rate |   observed_defaults |
|---------:|--------:|--------------------:|------------------------:|--------------------:|
|   1.0000 | 21.0000 |              0.0064 |                  0.0000 |              0.0000 |
|   2.0000 | 21.0000 |              0.0244 |                  0.0000 |              0.0000 |
|   3.0000 | 21.0000 |              0.0556 |                  0.0476 |              1.0000 |
|   4.0000 | 21.0000 |              0.1206 |                  0.1905 |              4.0000 |
|   5.0000 | 21.0000 |              0.2758 |                  0.2381 |              5.0000 |

**PD sensitivity to the assumed population default rate**

|   population_rate |   mean_pd |   max_pd |
|------------------:|----------:|---------:|
|            0.0100 |    0.0117 |   0.0732 |
|            0.0200 |    0.0229 |   0.1377 |
|            0.0300 |    0.0337 |   0.1948 |

## 7. Population stability (early vs. late window)
| feature         |    psi | reading        |
|:----------------|-------:|:---------------|
| log_assets      | 4.2218 | material shift |
| current_ratio   | 3.7600 | material shift |
| debt_ebitda     | 2.8306 | material shift |
| wc_ta           | 2.5943 | material shift |
| cfo_total_debt  | 2.5187 | material shift |
| interest_cover  | 2.4337 | material shift |
| equity_tl       | 1.7216 | material shift |
| ebit_ta         | 1.5225 | material shift |
| re_ta           | 1.3723 | material shift |
| net_margin      | 1.2338 | material shift |
| negative_equity | 0.0000 | stable         |
| two_year_loss   | 0.0000 | stable         |

PSI thresholds (0.10 / 0.25) are industry convention, not a statistical test.

## 8. Risk-grade masterscale
Grades are assigned on prior-corrected PD. The observed default rate per grade is the **in-sample, oversampled** rate — read it for ordering across grades, not as a level to compare against the grade's PD band.

|   grade |   n |   observed_defaults |   mean_pd |   observed_default_rate | label       |
|--------:|----:|--------------------:|----------:|------------------------:|:------------|
|       1 |   9 |                   0 |    0.0003 |                  0.0000 | Minimal     |
|       2 |  13 |                   0 |    0.0020 |                  0.0000 | Low         |
|       3 |  27 |                   0 |    0.0058 |                  0.0000 | Moderate    |
|       4 |  27 |                   3 |    0.0170 |                  0.1111 | Acceptable  |
|       5 |  23 |                   7 |    0.0487 |                  0.3043 | Watch       |
|       6 |   6 |                   0 |    0.1062 |                  0.0000 | Substandard |

- Monotonic across tested grades: **False**
- Grades tested: ['1', '2', '3', '4', '5', '6']
- Grades excluded for small bucket size (< 3): []
- Violations:
  - grade 5 -> 6: observed default rate falls 30.43% -> 0.00%

## 9. Expected loss, capital and default backtest

**Annual backtest — expected vs. realised defaults (sample-scale PDs).** Sample-scale, not prior-corrected, because the realised counts are sample counts. z is the normal approximation to the Poisson-binomial; with one or two defaults a year it is indicative only.

|      year |   issuers |   expected_defaults |   realised_defaults |       z |
|----------:|----------:|--------------------:|--------------------:|--------:|
| 2015.0000 |   13.0000 |              0.9800 |              0.0000 | -1.0700 |
| 2016.0000 |   14.0000 |              1.1900 |              0.0000 | -1.1900 |
| 2017.0000 |   15.0000 |              1.6800 |              2.0000 |  0.2900 |
| 2018.0000 |   13.0000 |              1.3900 |              0.0000 | -1.3300 |
| 2019.0000 |   13.0000 |              1.3100 |              4.0000 |  2.6400 |
| 2020.0000 |    9.0000 |              1.0500 |              0.0000 | -1.1900 |
| 2021.0000 |    9.0000 |              1.0700 |              1.0000 | -0.0800 |
| 2022.0000 |    8.0000 |              0.9900 |              2.0000 |  1.1900 |
| 2023.0000 |    6.0000 |              0.3200 |              1.0000 |  1.2500 |
| 2024.0000 |    5.0000 |              0.1500 |              0.0000 | -0.4000 |

**Cross-sectional portfolio as of 2019-06-30** — every issuer observed on that date, scored only with filings visible then.

- Issuers: **13**
- Total EAD (proxy): **222,415,709,000**
- Expected loss: **1,651,082,740** (0.74% of EAD)
- Exposure-weighted PD (prior-corrected): **1.58%**
- Exposure-weighted LGD: **46.9%**
- Basel IRB capital requirement: **18,067,537,071**
- Risk-weighted assets: **225,844,213,382**
- Issuers in this cross-section that defaulted within 12 months: **4**

EAD is a proxy — reported total debt, treated as fully drawn, excluding operating leases. Real EAD requires a facility schedule with committed undrawn lines, which no public filer discloses. The CCF machinery for facility data is implemented and tested in `risk.ead`.

**Per-issuer exposure detail**

| ticker   | sector                             | as_of      |   grade |     pd |    lgd |   lgd_downturn |               ead |   defaulted_within_12m |
|:---------|:-----------------------------------|:-----------|--------:|-------:|-------:|---------------:|------------------:|-----------------------:|
| AAPL     | Technology                         | 2019-06-30 |       4 | 0.0139 | 0.4700 |         0.6200 | 102519000000.0000 |                      0 |
| AMZN     | Retail — e-commerce                | 2019-06-30 |       3 | 0.0075 | 0.4700 |         0.6200 |  24866000000.0000 |                      0 |
| BBBY     | Retail — home goods                | 2019-06-30 |       3 | 0.0037 | 0.4700 |         0.6200 |   1487934000.0000 |                      0 |
| CHK      | Energy — E&P                       | 2019-06-30 |       5 | 0.0591 | 0.4750 |         0.6250 |   7722000000.0000 |                      1 |
| COST     | Retail — warehouse club            | 2019-06-30 |       3 | 0.0073 | 0.4700 |         0.6200 |   6577000000.0000 |                      0 |
| FTR      | Telecom                            | 2019-06-30 |       5 | 0.0336 | 0.4850 |         0.6350 |  17172000000.0000 |                      1 |
| HTZ      | Consumer services — vehicle rental | 2019-06-30 |       4 | 0.0140 | 0.4500 |         0.6000 |  16324000000.0000 |                      1 |
| JCP      | Retail — department stores         | 2019-06-30 |       4 | 0.0192 | 0.4650 |         0.6150 |   3808000000.0000 |                      1 |
| JNJ      | Healthcare — pharma                | 2019-06-30 |       2 | 0.0026 | 0.4700 |         0.6200 |  30284000000.0000 |                      0 |
| NKE      | Consumer discretionary             | 2019-06-30 |       1 | 0.0004 | 0.4700 |         0.6200 |   3474000000.0000 |                      0 |
| PRTY     | Retail — specialty                 | 2019-06-30 |       4 | 0.0208 | 0.4550 |         0.6050 |   1635279000.0000 |                      0 |
| RAD      | Retail — pharmacy                  | 2019-06-30 |       4 | 0.0197 | 0.4500 |         0.6000 |   3470696000.0000 |                      0 |
| REV      | Consumer staples — cosmetics       | 2019-06-30 |       6 | 0.1085 | 0.4600 |         0.6100 |   3075800000.0000 |                      0 |

## 10. Sourced inputs
Every recovery assumption below is SOURCED to published research, not assumed.

- **bank_loan_secured** — Moody's Ultimate Recovery Database, average discounted ultimate recovery on defaulted bank loans of roughly 80%, as reported in Moody's annual corporate default and recovery studies (Emery, Ou, Tennant et al., Moody's Investors Service).
- **senior_secured** — Altman & Kishore (1996), "Almost Everything You Wanted to Know about Recoveries on Defaulted Bonds", Financial Analysts Journal 52(6): senior secured bonds, average recovery of about $58 per $100 face, 1978-1995, measured at post-default prices.
- **senior_unsecured** — Altman & Kishore (1996): senior unsecured bonds, average recovery of about $48 per $100 face, post-default prices.
- **senior_subordinated** — Altman & Kishore (1996): senior subordinated bonds, about $34 per $100 face.
- **subordinated** — Altman & Kishore (1996): subordinated bonds, about $31 per $100 face.

## 11. What would change the verdict
- More defaulted issuers. The binding constraint is positives, not features.
- A speculative-grade non-default cohort, so the model is tested on the discrimination task that actually matters commercially.
- Macro covariates, which would separate point-in-time from through-the-cycle PD.
- Facility-level exposure data, replacing the EAD proxy with a measured number.
