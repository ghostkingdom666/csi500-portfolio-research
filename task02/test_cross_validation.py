"""CV tests use exact fold membership and a one-asset closed-form optimizer."""
import unittest
import csv
import json
import subprocess
import sys
import tempfile
from unittest.mock import patch
from pathlib import Path
import numpy as np
from models import target

try:
    import cross_validation as cv
except ModuleNotFoundError:
    cv = None


class CrossValidationChecks(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(cv, "random ten-fold cross-validation is missing")
        t = np.arange(60)
        self.R = (.015 + .02*np.sin(t) + .008*np.cos(.23*t))[:, None]
        self.grid = np.array([0., .01, .03, .1, 1.])

    def test_original_shuffle_and_exhaustive_disjoint_folds(self):
        folds = cv.shuffled_kfold_indices(10, 3, 42)
        self.assertEqual([v.tolist() for _, v in folds],
                         [[0, 1, 5, 8], [2, 7, 9], [3, 4, 6]])
        for train, val in folds:
            self.assertFalse(set(train) & set(val))
            self.assertEqual(sorted([*train, *val]), list(range(10)))

    def test_one_asset_scores_match_closed_form_raw_variance(self):
        result = cv.random_kfold_cv(self.R, .04, lambda_grid=self.grid)
        self.assertEqual(len(result["folds"]), 10)
        self.assertEqual(len(result["scores"]), 50)
        selected = []
        for fold in result["folds"]:
            train, val = self.R[fold["train_indices"]], self.R[fold["validation_indices"]]
            y, _ = target(train, .04)
            denom = float(np.sum(train**2))
            eligible = []
            for lam in self.grid:
                coefficient = max(0., (float(train[:, 0] @ y)-lam/2)/denom)
                row = next(s for s in result["scores"]
                           if s["fold"] == fold["fold"] and s["lambda_value"] == lam)
                expected_var = float(np.var(val[:, 0]*coefficient, ddof=1))
                self.assertAlmostEqual(row["raw_variance"], expected_var, places=13)
                self.assertAlmostEqual(row["variance_error"], abs(expected_var-.04**2), places=13)
                if coefficient > 0:
                    self.assertAlmostEqual(row["invested_variance"], float(val.var(ddof=1)), places=13)
                    eligible.append((abs(expected_var-.04**2), -lam))
                else:
                    self.assertEqual(row["status"], "zero_solution")
                    self.assertIsNone(row["invested_variance"])
            best = -min(eligible)[1]
            self.assertEqual(fold["selected_lambda"], best)
            selected.append(best)
        self.assertAlmostEqual(result["lambda_value"], sum(selected)/10)

    def test_validation_rows_cannot_enter_fold_target(self):
        before = cv.random_kfold_cv(self.R, .04, lambda_grid=self.grid)
        changed = self.R.copy()
        changed[before["folds"][0]["validation_indices"]] += .003
        after = cv.random_kfold_cv(changed, .04, lambda_grid=self.grid)
        self.assertEqual(before["folds"][0]["target"], after["folds"][0]["target"])

    def test_determinism_and_raw_invested_risks_are_distinct(self):
        first = cv.random_kfold_cv(self.R, .04, lambda_grid=self.grid)
        again = cv.random_kfold_cv(self.R, .04, lambda_grid=self.grid)
        self.assertEqual(first, again)
        valid = [r for r in first["scores"] if r["status"] == "optimal"]
        self.assertTrue(any(abs(r["raw_variance"]-r["invested_variance"]) > 1e-5 for r in valid))

    def test_all_zero_candidates_fail_instead_of_exporting_allocation(self):
        with self.assertRaisesRegex(ValueError, "no_investable") as caught:
            cv.random_kfold_cv(self.R, .04, lambda_grid=[1e6])
        self.assertEqual(len(caught.exception.diagnostics["folds"]), 10)
        self.assertEqual(len(caught.exception.diagnostics["scores"]), 10)

    def test_invalid_grids_and_insufficient_fold_samples_fail(self):
        for grid in [[], [-1.], [np.nan], [0., 0.]]:
            with self.assertRaises(ValueError):
                cv.random_kfold_cv(self.R, .04, lambda_grid=grid)
        with self.assertRaises(ValueError):
            cv.random_kfold_cv(self.R[:10], .04)
        with self.assertRaises(ValueError):
            cv.shuffled_kfold_indices(10, 1, 42)

    def test_runner_exports_cv_main_and_preserves_eta_sensitivity(self):
        here = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            completed = subprocess.run([sys.executable, "-B", str(here/"run.py"),
                                        "--output", str(out)], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)
            with (out/"runs.csv").open(encoding="utf-8") as stream:
                runs = {r["model_id"]: r for r in csv.DictReader(stream)}
            self.assertIn("l1_cv", runs, "runner still replaces CV with fixed eta")
            self.assertEqual(runs["l1_cv"]["selection_method"], "random_kfold")
            self.assertIn("l1_eta_0.01", runs)
            selected = json.loads((out/"cv_selection.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(float(runs["l1_cv"]["lambda_value"]), selected["lambda_value"])
            with (out/"cv_scores.csv").open(encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1000)
            manifest = json.loads((out/"manifest.json").read_text(encoding="utf-8"))
            self.assertIn("cv_selection.json", manifest["outputs"])

    def test_failed_candidates_do_not_break_successful_export_verification(self):
        import run as model_run
        import verify as model_verify
        real_lasso = cv.lasso
        calls = 0

        def unreliable_candidate(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("simulated candidate solver failure")
            w, info = real_lasso(*args)
            if calls == 2:
                info.update(status="failed_kkt", dual_violation=1e-6)
            return w, info

        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            with patch("cross_validation.lasso", side_effect=unreliable_candidate):
                model_run.run(out)
            with (out/"cv_scores.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["status"], "solver_failure")
            self.assertEqual(rows[1]["status"], "failed_kkt")
            try:
                model_verify.verify(out)
            except Exception as exc:
                self.fail(f"valid remaining candidates must survive independent verification: {exc}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
