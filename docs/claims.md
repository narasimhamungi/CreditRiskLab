# Claims ledger

Generated 2026-11-04 from the repo's own README.

| Claim | Tag | Evidence |
|---|---|---|
| PD/LGD/EAD/EL/Basel IRB capital built on real SEC filings from companies that actually filed Chapter 11, scored on pre-filing 10-K data only | Demonstrated | README "Problem"/"Why it matters" |
| On 105 issuer-years, fitted PD model reaches out-of-fold AUC 0.824 (95% CI 0.679-0.935) vs 0.773 for Altman Z''; paired bootstrap cannot distinguish them, reported as a null result | Demonstrated — and correctly reported as a null finding rather than a win | README headline |
| Guards against look-ahead bias (restated financials), oversampled-default inflation, and an "easy comparison group" inflating AUC | Demonstrated — explicitly designed against these three failure modes | README "Why it matters" |
| Model configuration ships inside the v0.2.1 package (per Keystone's README, for clean-install validation) | Demonstrated | keystone-claims.md cross-reference |

## Action
No overclaim found. The wide confidence interval (0.679-0.935) on 105 issuer-years is already disclosed alongside the headline AUC, not hidden — correct practice for a small-sample result.
