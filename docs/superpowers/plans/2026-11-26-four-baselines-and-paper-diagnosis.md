# Four baselines and original-paper diagnosis

User explicitly requested implementation and execution in the existing checkout. No Git metadata or CodeGraph index exists; edit directly and preserve prior results.

## Fixed design

- Add GMV and inverse sample volatility over the full 359-day estimation period.
- Add momentum Top20 using 126 daily returns with the most recent 21 excluded; add low-volatility Top20 over the last 60 returns. Lexical asset ID breaks ties; both hold 20 equal positions.
- All four use the existing 100-stock pool, decision 2026-06-30, entry close 2026-07-01, first return 2026-07-02, final valuation 2026-11-10, no rebalancing or costs. Do not tune on holdout results.
- Preserve all existing methods; new methods are main comparisons, for nine strategy main rows and twenty total rows including sensitivity, extensions and index reference.
- Preserve the current l1_cv as the comparison target. Diagnose original-paper ridge, quadrature, threshold and daily-rebalanced accounting separately rather than silently changing this target.
- Original paper uses CSI300 and April 2025, while the current experiment uses a CSI500 subset and July-November 2026. Positive performance in a different sample is not a replication target.

## Implementation and evidence

- [x] Test-first four baseline functions and integrate task03 run/protocol/verify/charts, including hand calculations, labels, ties, windows, invalid inputs and real frozen-data export.
- [x] Add independently checkable stock attribution, concentration, period/regime comparison and return-accounting diagnostics.
- [x] Re-run original appendix numeric choices on the current frozen matrix and separately inspect the paper's stated portfolio and positive return.
- [x] Independently review changed code, run all relevant suites, regenerate formal outputs and inspect updated figures.
- [x] Document comparison, quantified loss causes, original-paper differences and remaining limitations; preserve previous output under task03/archive/2026-11-26-before-four-baselines.

Parallel scopes: baseline implementation writes only task03; PDF verification is read-only on the repository and saves evidence outside it; parent owns analysis code and root documentation. Integration and output regeneration are sequential.
