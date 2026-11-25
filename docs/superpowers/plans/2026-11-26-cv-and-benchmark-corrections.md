# Cross-validation and benchmark corrections implementation plan

> **For agentic workers:** Use superpowers:executing-plans for the main-model work; independently scoped task03 fixes may use the dispatching-parallel-agents workflow. Tests precede behavior changes.

**Goal:** Restore the original shuffled ten-fold lambda selection as the main nonnegative L1 method, retain fixed-eta sensitivity, and correct benchmark risk reporting and VaR search claims.

**Architecture:** Preserve the frozen data and current L1 objective. Each CV fold estimates its own constant response using training rows only, selects an absolute lambda by validation raw-coefficient variance matching, then the mean of ten selected lambdas is refitted on all estimation rows. Export both coefficient-risk and invested-weight-risk diagnostics. Keep the existing fixed eta rows as sensitivity experiments. Update task03 to evaluate the CV main row and feasible heuristic rows honestly.

**Tech Stack:** Existing Python, NumPy, SciPy, pandas, mpmath, PySCIPOpt; no new mandatory library.

**Spec:** User instructions dated 2026-11-26 in this chat; preserve random ten-fold selection, directly correct source and clarify model positioning.

## Global constraints

- Edit the supplied checkout directly; it is a ZIP without Git metadata or CodeGraph index.
- Archive current task02/results and task03/results before regenerating outputs.
- Do not change frozen data or tune settings against July-November returns.
- CV defaults: 10 shuffled folds, seed 42, 100 absolute lambda candidates from 0 through 10, matching the original appendix's linspace call.
- Aggregate by the mean of per-fold selected lambdas, retaining the original procedure.
- Zero solutions and solver failures must be visible and cannot become invested weights.
- Main-model positioning: empirical nonnegative L1 allocation; raw-risk calibration does not guarantee normalized or future portfolio risk.
- Existing eta grid is supplemental sensitivity, not a replacement for CV.
- Preserve benchmark risk limits while correcting their stated basis; this is a retrospective revision, not a new preregistration.
- No commits or branch operations are required for this non-Git checkout.

## Review focus

- Fold target/covariance accidentally computed using validation or test rows.
- Confusing raw coefficient variance with normalized allocation variance.
- CV errors or zero solutions silently exported as successful portfolios.
- New heuristic success status rejected by downstream evaluation/verification.
- Old main-model IDs or hard-coded hashes hiding the new CV result.

## Task 1: Main model and reproducible CV

Files: add task02/cross_validation.py; modify task02/test_models.py, run.py, verify.py, protocol.json and README.md.

- [x] Add failing tests for deterministic exhaustive shuffled folds, one-asset closed-form CV scoring, invalid and all-zero grids, fold-local estimation, and coefficient/invested risk distinction.
- [x] Implement random_kfold_cv(R, sigma, n_splits=10, seed=42, lambda_grid=None) with complete per-fold/per-candidate diagnostics.
- [x] Export l1_cv as main plus cv_scores.csv, cv_folds.csv and cv_selection.json; retain all l1_eta rows.
- [x] Extend independent verification to CV candidates, aggregate lambda, final weights, constraints and export hashes.
- [x] Document exact selection criterion, random-fold limitation, parameter scaling and model scope.

## Task 2: Risk reporting and honest VaR benchmark

Files: task03 benchmarks/evaluation/loader/runner/verification/tests/protocol/documentation/chart code.

- [x] Add failing tests for risk-statistic labels, known feasible warm starts, honest heuristic/no-candidate statuses and downstream acceptance.
- [x] Export actual estimation equal-weight VaR/CVaR statistics at 95%/99%.
- [x] Retain original risk limits; correct the mistaken documentation and mark revision date.
- [x] Include deterministic known feasible CVaR allocations in VaR candidates and report feasible_heuristic, never global optimal.
- [x] Evaluate l1_cv as main and l1_eta_0.01 as sensitivity; remove stale hard-coded result hashes.

## Task 3: Integration, evidence and baseline recommendations

Files: root README and task cards as needed; generated phase02/03 results.

- [x] Run both complete existing test suites plus new regression tests.
- [x] Regenerate and independently verify phase02 and phase03 outputs; preserve historical outputs in archives.
- [x] Review changes independently and resolve material findings.
- [x] Provide source-backed baseline suggestions, separating allocation baselines from stock-selection strategies and matching data/rebalancing requirements.
