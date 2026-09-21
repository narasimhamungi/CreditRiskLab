# CreditRiskLab — Model Validation Report

_Generated 2026-09-21 11:37 UTC from the pipeline run. Not hand-edited._

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
| logistic_oof                  | 0.8200 | [0.676, 0.934]               | 0.6400 |
| gbm_oof                       | 0.7460 | [0.576, 0.900]               | 0.4930 |
| altman_zpp                    | 0.7710 | [0.658, 0.910]               | 0.5410 |
| logistic_leave_one_issuer_out | 0.8380 | n/a                          |        |
| logistic_out_of_time          | 0.9270 | n/a                          |        |

KS statistic (logistic, out-of-fold): **0.632**

Confidence intervals come from a bootstrap that resamples **issuers**, not rows. Resampling rows would treat one issuer's five annual observations as five independent facts and halve the apparent interval width.

## 4. Benchmark test — does the fitted model beat Altman Z''?
- Paired AUC difference (logistic minus Z''): **0.049**
- 95% interval: [-0.094, 0.216]
- Verdict: **no distinguishable difference from Altman Z'' at this sample size — the fitted model is not demonstrably better than the published benchmark**

Z'' (Altman, Hartzell & Peck 1995) is used rather than the original 1968 Z because the default cohort is retail, telecom, media and energy — not listed manufacturers — and market value of equity is not available from XBRL filings.

## 5. Coefficient review
| feature         |   coefficient |   expected_sign |   actual_sign | sign_consistent   |
|:----------------|--------------:|----------------:|--------------:|:------------------|
| equity_tl       |       -1.2671 |              -1 |            -1 | True              |
| cfo_total_debt  |       -1.2289 |              -1 |            -1 | True              |
| current_ratio   |       -0.9951 |              -1 |            -1 | True              |
| ebit_ta         |        0.7467 |              -1 |             1 | False             |
| two_year_loss   |        0.6777 |               1 |             1 | True              |
| interest_cover  |       -0.6574 |              -1 |            -1 | True              |
| negative_equity |       -0.5995 |               1 |            -1 | False             |
| debt_ebitda     |        0.4622 |               1 |             1 | True              |
| net_margin      |        0.4186 |              -1 |             1 | False             |
| wc_ta           |       -0.4014 |              -1 |            -1 | True              |
| log_assets      |       -0.3718 |              -1 |            -1 | True              |
| re_ta           |       -0.1588 |              -1 |            -1 | True              |

**Sign violations against credit priors (reported, not corrected):**
- ebit_ta: coefficient +0.747 contradicts the expected sign (higher value => lower PD)
- negative_equity: coefficient -0.600 contradicts the expected sign (higher value => higher PD)
- net_margin: coefficient +0.419 contradicts the expected sign (higher value => lower PD)

## 6. Calibration and prior correction
Calibration is **cross-fitted by issuer**: each issuer's PD is mapped by a calibrator learned only from other issuers, so the metrics below are out-of-sample rather than a fitted curve scored on its own data.

- Calibration method actually fitted: **platt**
- Sample default rate: **9.52%**
- Population default rate used for correction: **2.00%** (ASSUMPTION)
- Mean PD before prior correction: **9.64%**
- Mean PD after prior correction: **2.30%**

The sample over-represents defaults by roughly an order of magnitude — unavoidable when building a default sample from SEC filings, and the reason the correction exists. Without it every PD, expected loss and capital figure in this report would be inflated by a similar multiple. The correction is monotone, so discrimination is unaffected.

- Brier score: **0.081** (reliability 0.0159, resolution 0.0190, uncertainty 0.0862)
- Calibration slope: **0.942** (1.0 = perfect; below 1 means predictions are too extreme)
- Calibration intercept: **-0.114**
- Hosmer-Lemeshow: chi2 12.166, dof 8, p 0.144 — low power at this sample size; a non-rejection is not evidence of calibration

**Reliability by predicted-PD bucket**

|   bucket |       n |   mean_predicted_pd |   observed_default_rate |   observed_defaults |
|---------:|--------:|--------------------:|------------------------:|--------------------:|
|   1.0000 | 21.0000 |              0.0051 |                  0.0000 |              0.0000 |
|   2.0000 | 21.0000 |              0.0222 |                  0.0000 |              0.0000 |
|   3.0000 | 21.0000 |              0.0541 |                  0.0476 |              1.0000 |
|   4.0000 | 21.0000 |              0.1191 |                  0.1429 |              3.0000 |
|   5.0000 | 21.0000 |              0.2815 |                  0.2857 |              6.0000 |

**PD sensitivity to the assumed population default rate**

|   population_rate |   mean_pd |   max_pd |
|------------------:|----------:|---------:|
|            0.0100 |    0.0117 |   0.0690 |
|            0.0200 |    0.0230 |   0.1303 |
|            0.0300 |    0.0338 |   0.1850 |

## 7. Population stability (early vs. late window)
| feature         |    psi | reading        |
|:----------------|-------:|:---------------|
| log_assets      | 4.2218 | material shift |
| debt_ebitda     | 3.8311 | material shift |
| current_ratio   | 3.7600 | material shift |
| interest_cover  | 2.6772 | material shift |
| wc_ta           | 2.5943 | material shift |
| cfo_total_debt  | 2.5187 | material shift |
| equity_tl       | 1.7216 | material shift |
| re_ta           | 1.3723 | material shift |
| net_margin      | 1.2338 | material shift |
| ebit_ta         | 0.4244 | material shift |
| negative_equity | 0.0000 | stable         |
| two_year_loss   | 0.0000 | stable         |

PSI thresholds (0.10 / 0.25) are industry convention, not a statistical test.

## 8. Risk-grade masterscale
Grades are assigned on prior-corrected PD. The observed default rate per grade is the **in-sample, oversampled** rate — read it for ordering across grades, not as a level to compare against the grade's PD band.

|   grade |   n |   observed_defaults |   mean_pd |   observed_default_rate | label       |
|--------:|----:|--------------------:|----------:|------------------------:|:------------|
|       1 |   9 |                   0 |    0.0002 |                  0.0000 | Minimal     |
|       2 |  14 |                   0 |    0.0018 |                  0.0000 | Low         |
|       3 |  28 |                   0 |    0.0056 |                  0.0000 | Moderate    |
|       4 |  26 |                   3 |    0.0181 |                  0.1154 | Acceptable  |
|       5 |  20 |                   7 |    0.0470 |                  0.3500 | Watch       |
|       6 |   8 |                   0 |    0.1022 |                  0.0000 | Substandard |

- Monotonic across tested grades: **False**
- Grades tested: ['1', '2', '3', '4', '5', '6']
- Grades excluded for small bucket size (< 3): []
- Violations:
  - grade 5 -> 6: observed default rate falls 35.00% -> 0.00%

## 9. Expected loss, capital and default backtest

**Annual backtest — expected vs. realised defaults (sample-scale PDs).** Sample-scale, not prior-corrected, because the realised counts are sample counts. z is the normal approximation to the Poisson-binomial; with one or two defaults a year it is indicative only.

|      year |   issuers |   expected_defaults |   realised_defaults |       z |
|----------:|----------:|--------------------:|--------------------:|--------:|
| 2015.0000 |   13.0000 |              0.9500 |              0.0000 | -1.0500 |
| 2016.0000 |   14.0000 |              1.1400 |              0.0000 | -1.1600 |
| 2017.0000 |   15.0000 |              1.6400 |              2.0000 |  0.3200 |
| 2018.0000 |   13.0000 |              1.3900 |              0.0000 | -1.3300 |
| 2019.0000 |   13.0000 |              1.3200 |              4.0000 |  2.6300 |
| 2020.0000 |    9.0000 |              1.0800 |              0.0000 | -1.2100 |
| 2021.0000 |    9.0000 |              1.1000 |              1.0000 | -0.1100 |
| 2022.0000 |    8.0000 |              1.0600 |              2.0000 |  1.0900 |
| 2023.0000 |    6.0000 |              0.3200 |              1.0000 |  1.2800 |
| 2024.0000 |    5.0000 |              0.1400 |              0.0000 | -0.3900 |

**Cross-sectional portfolio as of 2019-06-30** — every issuer observed on that date, scored only with filings visible then.

- Issuers: **13**
- Total EAD (proxy): **222,415,709,000**
- Expected loss: **1,643,512,205** (0.74% of EAD)
- Exposure-weighted PD (prior-corrected): **1.57%**
- Exposure-weighted LGD: **46.9%**
- Basel IRB capital requirement: **17,792,496,378**
- Risk-weighted assets: **222,406,204,729**
- Issuers in this cross-section that defaulted within 12 months: **4**

EAD is a proxy — reported total debt, treated as fully drawn, excluding operating leases. Real EAD requires a facility schedule with committed undrawn lines, which no public filer discloses. The CCF machinery for facility data is implemented and tested in `risk.ead`.

**Per-issuer exposure detail**

| ticker   | sector                             | as_of      |   grade |     pd |    lgd |   lgd_downturn |               ead |   defaulted_within_12m |
|:---------|:-----------------------------------|:-----------|--------:|-------:|-------:|---------------:|------------------:|-----------------------:|
| AAPL     | Technology                         | 2019-06-30 |       4 | 0.0127 | 0.4700 |         0.6200 | 102519000000.0000 |                      0 |
| AMZN     | Retail — e-commerce                | 2019-06-30 |       3 | 0.0075 | 0.4700 |         0.6200 |  24866000000.0000 |                      0 |
| BBBY     | Retail — home goods                | 2019-06-30 |       3 | 0.0034 | 0.4700 |         0.6200 |   1487934000.0000 |                      0 |
| CHK      | Energy — E&P                       | 2019-06-30 |       5 | 0.0562 | 0.4750 |         0.6250 |   7722000000.0000 |                      1 |
| COST     | Retail — warehouse club            | 2019-06-30 |       3 | 0.0066 | 0.4700 |         0.6200 |   6577000000.0000 |                      0 |
| FTR      | Telecom                            | 2019-06-30 |       5 | 0.0400 | 0.4850 |         0.6350 |  17172000000.0000 |                      1 |
| HTZ      | Consumer services — vehicle rental | 2019-06-30 |       4 | 0.0166 | 0.4500 |         0.6000 |  16324000000.0000 |                      1 |
| JCP      | Retail — department stores         | 2019-06-30 |       4 | 0.0160 | 0.4650 |         0.6150 |   3808000000.0000 |                      1 |
| JNJ      | Healthcare — pharma                | 2019-06-30 |       2 | 0.0021 | 0.4700 |         0.6200 |  30284000000.0000 |                      0 |
| NKE      | Consumer discretionary             | 2019-06-30 |       1 | 0.0003 | 0.4700 |         0.6200 |   3474000000.0000 |                      0 |
| PRTY     | Retail — specialty                 | 2019-06-30 |       4 | 0.0202 | 0.4550 |         0.6050 |   1635279000.0000 |                      0 |
| RAD      | Retail — pharmacy                  | 2019-06-30 |       4 | 0.0208 | 0.4500 |         0.6000 |   3470696000.0000 |                      0 |
| REV      | Consumer staples — cosmetics       | 2019-06-30 |       6 | 0.1108 | 0.4600 |         0.6100 |   3075800000.0000 |                      0 |

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
