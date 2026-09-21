# CreditRiskLab — Build Roadmap

Project 5 of an applied finance/analytics portfolio. Planned to consume Trellis as an
installed package; in the real run every row came from the direct EDGAR client because
Trellis does not carry SEC filing dates. Trellis was not modified.

**Locked scope decision (made before any code):** default labels come from real
non-financial US issuers that actually filed for bankruptcy, with pre-filing 10-Ks on
EDGAR. Trellis's existing investment-grade universe supplies the non-default class.
Rejected alternatives: a pre-packaged academic bankruptcy dataset (breaks the real-SEC-data
thread, heavily recycled in portfolios), and a scoring-only Z-score tool (no validated
model, weaker for Credit Risk / Model Risk roles).

---

## Stage 0 — Foundations

| Item | Decision |
|---|---|
| Package layout | `src/creditrisklab`, installable, `pip install -e .` |
| Trellis coupling | Consume as installed package via adapter; **zero edits to Trellis** |
| Fallback ingestion | Direct SEC XBRL `companyfacts` client if Trellis is not installed |
| Config | YAML: universe, recovery rates, model params — no hardcoded assumptions in code |
| Tests | pytest, offline-runnable on synthetic fixtures (no network in CI) |

## Stage 1 — Universe & default events

- 10 defaulted non-financial issuers (retail, pharmacy retail, telecom, energy, consumer).
- Investment-grade non-defaulters from Trellis: 5 confirmed in `universe.yaml`, extended to
  Trellis's full validated universe with `scripts/sync_trellis_universe.py`.
- Each default event recorded with: filing date, chapter, and seniority/exposure mix.
- Every issuer carries a `source` field. Nothing enters the universe without an EDGAR CIK.

## Stage 2 — Point-in-time ingestion

- Pull annual fundamentals **with SEC `filed` dates attached**, per fact.
- `as_of_snapshot()` returns only records with `filed <= as_of`. Restatements filed after
  the as-of date are structurally unreachable — hindsight bias is prevented by the data
  layer, not by discipline.
- Observation dates for non-defaulters are drawn from the same calendar window as the
  default cohort, so the model cannot learn "2019–2023 = risky era".

## Stage 3 — Feature engineering

Credit-relevant ratios only, each with an economic rationale documented:
liquidity, leverage, coverage, profitability, cash generation, size, and two distress flags.
Winsorised at 1/99; missing values median-imputed **within the training fold only**.

## Stage 4 — Models

| Model | Role |
|---|---|
| Altman Z''(EM) | Benchmark. Z'' not original Z — the default cohort is retail/service, not manufacturing |
| Ohlson O-score | Second benchmark, published coefficients |
| Logistic regression | Primary model. Interpretable, coefficient signs checkable, standard in credit |
| HistGradientBoosting | Challenger. Reported, not promoted, given sample size |

## Stage 5 — Calibration

- Sample default rate ≈ 40%. Population corporate default rate ≈ 1–3%.
- Uncorrected probabilities would overstate PD by an order of magnitude.
- Prior correction applied (rare-event odds shift) to every reported PD.
- This is stated in the README, not buried — it is the single most common silent error in
  portfolio PD models.

## Stage 6 — Validation

- Discrimination: AUC, Gini, KS, bootstrap confidence intervals.
- Calibration: Brier score, calibration slope/intercept, Hosmer–Lemeshow.
- Resampling: leave-one-default-out + repeated stratified CV (appropriate for small n;
  a single 80/20 split would be noise).
- Benchmark test: does the fitted model beat Altman Z'' on out-of-fold AUC? If not, that
  is reported as the finding.
- Stability: PSI across the feature panel.

## Stage 7 — LGD / EAD / Expected Loss

- LGD from published recovery-rate studies segmented by seniority and collateral,
  cited in `config/recovery_rates.yaml`. Tagged **Sourced**, never Assumed.
- EAD = drawn + CCF × undrawn, CCF from published credit-line-usage research.
- EL = PD × LGD × EAD, computed per issuer and aggregated to portfolio.

## Stage 8 — Risk grading

- 7-grade masterscale mapped from calibrated PD bands.
- Monotonicity test: realised default rate must be non-decreasing across grades.
- Grade assignment reported alongside Z'' zone for cross-reference.

## Stage 9 — Reporting

- `MODEL_VALIDATION.md` generated from the run, not written by hand.
- Charts: ROC, calibration curve, score distribution by outcome, grade distribution, EL waterfall.

## Stage 10 — Documentation & honest limits

- README states sample size, class imbalance, survivorship considerations, and that this is
  a **validated methodology demonstration, not a production PD model**.
- Limitations section written before the results section, so it cannot be softened afterwards.

---

## Build status

| Stage | Status |
|---|---|
| 0 Foundations | Built — package, config validation, CI workflow |
| 1 Universe | Built — gate enforces verified CIKs and verified default events |
| 2 Point-in-time ingestion | Built — tag migration and wrong-fallback bugs fixed and tested |
| 3 Features | Built — 12 ratios with sign priors; lease adjustment deferred (see limits) |
| 4 Models | Built |
| 5 Calibration | Built — cross-fitted by issuer, prior-corrected, sensitivity grid |
| 6 Validation | Built — plus annual default backtest and control-group hardness test |
| 7 LGD / EAD / EL | Built — plus Basel IRB capital, reconciled to BCBS risk weights |
| 8 Grades | Built |
| 9 Reporting | Built — real runs publish to `docs/results/`; synthetic runs cannot |
| 10 Docs | Built — README findings written from the real run |
| **Real-data run** | **Done** — 15 CIKs verified, 10 default events confirmed, 8,671 XBRL facts, 105 observations |

Tests: 103, offline, ~8 seconds.

## Known limits, declared up front

1. ~23 issuers, ~10 default events. Statistical power is low; CIs will be wide and are reported as such.
2. Non-defaulters are large-cap IG; the model discriminates "large IG vs. distressed" rather than fine gradations within speculative grade.
3. Recovery assumptions are from published studies, not issuer-specific workout data (commercial, Moody's/S&P).
4. No macro covariates in v1 — a through-the-cycle vs. point-in-time PD distinction is noted, not implemented.
5. Debt excludes operating leases; ASC 842 (2019) breaks retailer balance-sheet comparability mid-panel.
6. Controls are investment-grade, so discrimination is easy; the run measures and reports this.
