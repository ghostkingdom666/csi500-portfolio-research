# Execution ledger: 2026-11-26-cv-and-benchmark-corrections.md

Ruling: edit the supplied ZIP checkout directly, as the user requested; no Git metadata exists, so no branch/worktree/commit operations apply. Historical phase02/03 outputs, protocols and READMEs were archived before regeneration.

Ruling: preserve the original raw-coefficient variance criterion and mean of fold-optimal lambdas. Clearly separate normalized allocation risk. Fixed eta remains supplemental sensitivity; no new hard risk constraint or profit guarantee is introduced.

Task 1: complete. CV tests were observed red before module/runner implementation. Eight model tests and eight CV tests pass, including closed-form scoring, deterministic original fold membership, fold-local targets, all-fold zero diagnostics, export integration and partial solver/KKT failures. Formal 10 folds / 1000 candidates independently rescored; lambda=0.222222222222 and all 8 portfolios verified.

Task 2: complete. Risk/heuristic/status tests observed red before repair; final 22 task03 tests pass. Original risk limits retained; VaR has known feasible warm starts within the 3000 total budget and honestly reports feasible_heuristic/no_feasible_candidate. Model identities, row statuses and provenance are bound across source files and outputs.

Review: independent read-only review identified P2 partial-candidate verification and stale upstream delivery acceptance. Both reproduced with failing tests, repaired in one fix pass, and covered by the final green suites. Coefficient-risk and fold presentation fields were also independently checked.

Task 3: complete. Formal task03 rebuild passed 95 contract checks and 140 independent checks. Eight portfolios reconciled with direct frozen price ratios. Four chart artifacts regenerated and main NAV visually inspected. Baseline recommendations saved in docs/baseline_options.md; they are proposals, not falsely included in results.

Final tests: task02/test_models.py 8/8; task02/test_cross_validation.py 8/8; task03/test_eval.py 22/22. Evidence in phase02/03 results/tests.txt and verification.txt. Frozen data files and data pipeline source were not changed.
