# Execution ledger: four-baselines-and-paper-diagnosis

Task complete. Added four fixed baselines to the same 100-stock/359-estimation-day/88-return-day experiment, retaining the existing main model and all original rows. Formal output has 20 rows and 9 main strategies. Previous stage03 output is archived under task03/archive/2026-11-26-before-four-baselines.

Test-first evidence: new baseline module and missing independent rule verification were observed failing before implementation. Attribution/compatibility tests were observed failing before implementation. Independent review's label/date/entry-anchor gaps were reproduced, fixed, and covered by regression tests. Strict reconciliation then exposed the existing extension's 8e-10 solver budget tail; it is now recorded explicitly, without altering source weights or public NAV.

Independent review: all new selections, 359/88 matrices, date identity, inverse-volatility weights, GMV KKT and paper-compatible numerical choices verified. Reported validation gaps were addressed; current exact frozen data had not been affected by them.

Fresh final tests: 8 task02 model + 8 CV + 22 evaluator + 7 simple baseline + 10 diagnostics = 55 tests, all passed. Stage03 formal run passed 95 contract and 198 independent checks. All stage02 portfolios reconciled with frozen price endpoints. New baseline figures visually inspected.

Original-paper check: 28 supplier snapshots are preserved with SHA256 and portable offline reconstruction. Printed April 2025 portfolio is positive under raw and hfq definitions; exact 5.51% not fully reproduced. Original 601127 benchmark's 3.49% and .0313 sd are matched at published precision. Current-data ridge/quad/threshold/rebalanced compatibility remains about -60.67%, separating real method differences from changed sample performance.

Artifacts: analysis/README.md, analysis/results, analysis/paper_evidence, task03/results. Frozen data and task02 main source/weights remain unchanged. No Git operations apply to this supplied ZIP checkout.
