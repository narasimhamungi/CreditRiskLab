# CreditRiskLab

Probability of default, LGD, EAD, expected loss and Basel IRB capital — built on real SEC
filings from US companies that actually went bankrupt, validated against the published
distress models a credit analyst would reach for first.

Project 5 of an applied finance/analytics portfolio. Consumes
[Trellis](https://github.com/narasimhamungi/trellis) as an installed package; does not modify it.

> **Status: machinery complete, first real run pending.** 103 tests pass on synthetic
> fixtures. No performance figure is claimed below until the pipeline has run on verified
> SEC data — see [Key findings](#key-findings).

---

## Problem

A PD model is only as real as its default labels. The rest of this portfolio runs on
investment-grade large caps — none of which has ever defaulted — so it contains no ground
truth for credit risk at all. Most portfolio credit models solve this by downloading a
pre-labelled academic dataset. This one builds the labels from the source: companies that
filed for Chapter 11, scored using only the 10-K data that was public *before* they filed.

## Why it matters

Three errors make most small-sample PD models misleading, and none of them shows up in an AUC:

1. **Look-ahead in the data.** Using financials restated after the default date leaks the outcome.
2. **Oversampled defaults.** A sample built to contain defaults has a default rate several
   times the population's. Uncorrected, every PD, expected loss and capital figure inherits
   that inflation.
3. **An easy comparison group.** Bankrupt retailers against mega-cap investment-grade names
   produce a high AUC for almost any score. A high number on that design says the task is
   easy, not that the model is good.

This project is built to expose each of those rather than benefit from them.

## What was built

| Layer | What it does |
|---|---|
| Universe | 10 non-financial Chapter 11 filers (2017–2023) + investment-grade controls from Trellis. The pipeline refuses to train until every CIK is verified against EDGAR and every default date is matched to an Item 1.03 bankruptcy 8-K. |
| Ingestion | SEC XBRL `companyfacts`, Trellis-first with a direct EDGAR fallback. The SEC `filed` date travels with every fact; a `source` column records which path each row took. |
| Point-in-time | Filters on `filed <= as_of` **before** selecting the latest reported vintage — the reverse order is the classic look-ahead bug. Stale filings are dropped, not carried forward. |
| Panel | Discrete-time hazard design (Shumway, 2001): one row per issuer-year, label = default within 12 months. |
| Models | Penalised logistic regression (primary), gradient boosting (challenger), Altman Z''(EM) and Ohlson O-score (benchmarks). |
| Validation | Issuer-clustered bootstrap CIs, paired AUC test vs. Z'', leave-one-issuer-out, out-of-time split, cross-fitted calibration, Brier decomposition, calibration slope, Hosmer–Lemeshow, PSI, annual expected-vs-realised default backtest, coefficient sign checks, control-group hardness test. |
| Risk | LGD from cited recovery studies by seniority, downturn LGD, EAD with credit conversion factors, EL = PD × LGD × EAD, Basel IRB corporate capital and RWA, 7-grade masterscale with a monotonicity test. |
| Reporting | `MODEL_VALIDATION.md` generated from the run — limitations first, performance second. |

## Data and tools

- **Data:** SEC EDGAR XBRL (`companyfacts`, `submissions`). No commercial data.
- **Recovery inputs:** Altman & Kishore (1996) bond recoveries by seniority (post-default
  trading-price basis) and Moody's ultimate loan recoveries. Every seniority-class figure in
  `config/recovery_rates.yaml` is tagged `SOURCED` with its citation and declares its recovery
  basis; the loader rejects either omission, and the LGD blend refuses to mix trading-price
  with ultimate recoveries. The one analyst adjustment (industry-distress haircut) is tagged
  `ASSUMPTION` and is off by default.
- **Stack:** Python, pandas, NumPy, scikit-learn, SciPy, matplotlib, pytest, GitHub Actions.

## Methodology

**Labels.** Default = Chapter 11 petition date, verified against the issuer's Item 1.03 8-K.
Observations after the default date are dropped, never labelled 0.

**Features.** Twelve ratios, each with a stated credit rationale and an expected coefficient
sign: liquidity, leverage (debt/EBITDA), coverage, profitability, cash generation, size, and
two distress flags. Negative EBITDA maps to worst-case leverage; a *missing* interest tag stays
missing rather than defaulting to strong coverage.

**Leak control.** Winsorisation, imputation and scaling are fitted inside each training fold.
Cross-validation is grouped by issuer, so no company appears in both train and test.

**Benchmark test.** The headline question is not "what is the AUC" but "does the fitted model
beat Altman Z'' on the same observations" — answered by a paired bootstrap over issuers. If the
interval includes zero, the report says the model is not demonstrably better.

**Calibration.** Cross-fitted by issuer (no calibrator is scored on its own fit), then
prior-corrected from the sample default rate to an assumed population rate (King & Zeng, 2001),
with sensitivity over 1–3%.

**Capital.** Basel IRB corporate risk-weight function; the implementation reproduces the BCBS
illustrative risk weight (PD 1%, LGD 45%, M 2.5 → 92.3%) in the test suite.

## Limitations

Written before the results, so they cannot be softened after them.

- **Ten default events.** Every discrimination estimate carries a wide interval, and the report prints it.
- **Easy control group.** Investment-grade controls make discrimination easy. The run measures
  this directly (share of surviving-issuer observations with debt/EBITDA > 4x or cover < 2x) and
  prints a verdict. A rule-selected distressed-survivor cohort is the planned fix.
- **Prior correction is not a representativeness fix.** It corrects the base rate, not a
  control group that is not a random sample of survivors. PD levels are indicative.
- **Leases.** Debt excludes operating lease liabilities; ASC 842 (2019) also creates a
  structural break in retailer balance sheets mid-panel.
- **EAD is a proxy** — reported total debt, fully drawn. No public filer discloses a facility schedule.
- **LGD is study-average**, not issuer workout data (commercial, Moody's/S&P).
- **No macro covariates** — point-in-time vs. through-the-cycle PD is noted, not modelled.

This is a validated-methodology demonstration, not a production PD model.

## Key findings

**Pending the first real run.** Results are written to `docs/results/MODEL_VALIDATION.md` by
`creditrisklab run` and will be summarised here from that file — not before.

What the offline suite establishes about the *machinery* (not about any issuer): the
point-in-time invariant holds on every panel row; the benchmark test returns a null result
when two scores are indistinguishable; the hardness check flags an easy control design; the
IRB function reconciles to published Basel risk weights.

## Insight demonstrated

The difficult part of credit modelling on public data is not the classifier. It is building
labels that are real, features that are knowable at the time, probabilities that mean what
they say, and a validation that is allowed to conclude the model adds nothing.

## Employer takeaway

For **model risk / validation** roles: the full validation toolkit — clustered inference,
benchmark testing, cross-fitted calibration, stability, backtesting — applied to a model
with every weakness declared. For **credit risk** roles: PD, LGD, EAD, EL, IRB capital and
a masterscale, built from primary filings with sourced assumptions.

---

## Quickstart

```bash
pip install -e ".[dev]"
python -m pytest                         # offline test suite

python -m creditrisklab.cli demo         # synthetic smoke run, watermarked, never published
```

Real data (network required; set `CREDITRISKLAB_SEC_UA` in `.env` — see `.env.example`):

```bash
python -m creditrisklab.cli resolve          # verify every CIK against EDGAR; writes verified ones to universe.yaml
python -m creditrisklab.cli verify-defaults  # match default dates to Item 1.03 8-Ks; writes verified: true when found
python -m creditrisklab.cli ingest
python -m creditrisklab.cli run              # writes docs/results/MODEL_VALIDATION.md + charts
```

## Structure

```
config/            universe, recovery rates (cited), model parameters
src/creditrisklab/
  ingest/          EDGAR client, Trellis adapter, XBRL schema
  features/        point-in-time snapshots, ratios, panel, control-group hardness
  labels/          default labelling on the hazard panel
  models/          logistic, GBM, Altman Z'', Ohlson O, calibration
  validation/      discrimination, calibration tests, stability, backtests
  risk/            LGD, EAD, expected loss + IRB capital, grades
  reporting/       generated validation report and charts
tests/             103 offline tests
```

See [`ROADMAP.md`](ROADMAP.md) for the build plan and the scoping decision behind the label source.
