import unittest
import numpy as np
import pandas as pd

try:
    import portfolio_diagnostics as diagnostics
except ModuleNotFoundError:
    diagnostics = None


class DiagnosticChecks(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(diagnostics, "portfolio attribution and paper diagnostics are missing")

    def test_stock_contributions_sum_to_buy_and_hold_return(self):
        returns = pd.DataFrame([[.1, 0], [0, .1]], columns=["A", "B"])
        result = diagnostics.stock_attribution(pd.Series({"B": .5, "A": .5}), returns)
        np.testing.assert_allclose(result.loc[["A", "B"], "contribution"], [.05, .05])
        self.assertAlmostEqual(result.contribution.sum(), .10)

    def test_budget_roundoff_is_reconciled_explicitly(self):
        R = pd.DataFrame([[.1,.1]],columns=["A","B"])
        result = diagnostics.stock_attribution(pd.Series({"A":.4999999996,"B":.4999999996}),R)
        self.assertIn("initial_budget_residual",result.attrs)
        self.assertAlmostEqual(result.attrs["initial_budget_residual"],-8e-10,places=15)
        self.assertAlmostEqual(result.contribution.sum()+result.attrs["initial_budget_residual"],.09999999912,places=14)

    def test_negative_contributions_and_invalid_allocation(self):
        returns = pd.DataFrame([[-.5, .2], [0, 0]], columns=["A", "B"])
        result = diagnostics.stock_attribution(pd.Series({"A": .8, "B": .2}), returns)
        self.assertAlmostEqual(result.contribution.sum(), -.36)
        with self.assertRaises(ValueError):
            diagnostics.stock_attribution(pd.Series({"A": 1., "B": 1.}), returns)
        with self.assertRaises(ValueError):
            diagnostics.stock_attribution(pd.Series({"A": .5, "B": .5}), returns*10)

    def test_monthly_blocks_compound_and_include_initial_capital(self):
        nav = pd.Series([1., .9, .81], index=["2026-07-01", "2026-07-31", "2026-08-31"])
        result = diagnostics.monthly_returns(nav)
        self.assertEqual(result.index.tolist(), ["2026-07", "2026-08"])
        np.testing.assert_allclose(result, [-.1, -.1])
        self.assertAlmostEqual(float((1+result).prod()-1), -.19)

    def test_original_covariance_ridge_is_applied_before_theta(self):
        R = np.array([[1,0],[-1,0],[0,1],[0,-1],[0,0],[0,0]])*np.sqrt(2.5)*.02+[.01,.005]
        _, stats = diagnostics.paper_target(R, .04, ridge=1e-6, method="stable")
        self.assertAlmostEqual(stats["theta_s"], .000125/.000401)
        y_quad, quad = diagnostics.paper_target(R, .04, ridge=1e-6, method="quad")
        y_stable, _ = diagnostics.paper_target(R, .04, ridge=1e-6, method="stable")
        np.testing.assert_allclose(y_quad, y_stable, rtol=1e-11)
        self.assertLess(quad["beta_relative_error_vs_stable"], 1e-10)

    def test_original_threshold_is_inclusive_and_renormalized(self):
        w = diagnostics.paper_normalize(np.array([.0001, .0001001, 1.]))
        self.assertEqual(w[0], 0.)
        self.assertGreater(w[1], 0.)
        self.assertAlmostEqual(w.sum(), 1.)
        with self.assertRaises(ValueError):
            diagnostics.paper_normalize(np.array([0., .0001]))

    def test_paper_cv_matches_one_asset_closed_form_and_original_zero_scoring(self):
        t = np.arange(60)
        R = (.015+.02*np.sin(t)+.008*np.cos(.23*t))[:,None]
        grid = [0.,.01,.03,.1,1.]
        raw, invested, info = diagnostics.fit_paper_style(R,grid=grid,method="stable")
        order = np.arange(60)
        np.random.RandomState(42).shuffle(order)
        selected = []
        for ids in np.array_split(order,10):
            tr = R[np.setdiff1d(np.arange(60),ids)]
            val = R[ids,0]
            y, _ = diagnostics.paper_target(tr,.04,method="stable")
            values = []
            for lam in grid:
                w = max(0.,(float(tr[:,0]@y)-lam/2)/float(np.sum(tr**2)))
                values.append((abs(float(np.var(val*w,ddof=1))-.04**2),-lam))
            selected.append(-min(values)[1])
        self.assertAlmostEqual(info["lambda_value"],sum(selected)/10)
        self.assertEqual(invested.tolist(),[1.])
        self.assertGreater(raw[0],0.)

    def test_alignment_rejects_missing_or_unknown_main_labels(self):
        align = getattr(diagnostics, "aligned_weights", None)
        self.assertIsNotNone(align, "diagnostics must validate labels before filling extension zeros")
        for w in [pd.Series({"A": 1.}), pd.Series({"A": 1., "B": 0., "C": 0.})]:
            with self.assertRaises(ValueError):
                align(w,["A","B"])
        result = align(pd.Series({"A": 1.}),["A","B"],allow_subset=True)
        self.assertEqual(result.to_dict(),{"A":1.,"B":0.})
        with self.assertRaises(ValueError):
            align(pd.Series({"A":1.,"C":0.}),["A","B"],allow_subset=True)

    def test_nav_reconciliation_checks_dates_and_absolute_tolerance(self):
        verify = getattr(diagnostics, "verify_nav", None)
        self.assertIsNotNone(verify, "NAV reconciliation must check date identity")
        nav = pd.Series([1., .5],index=["2026-07-01","2026-07-02"])
        with self.assertRaises(AssertionError):
            verify(nav,pd.Series([1.,.5],index=["2026-07-02","2026-07-03"]))
        with self.assertRaises(AssertionError):
            verify(nav,nav+1e-9)
        verify(nav,nav.copy())

    def test_unadjusted_price_diagnostic_requires_the_entry_anchor(self):
        convert = getattr(diagnostics, "raw_close_returns", None)
        self.assertIsNotNone(convert, "raw close diagnostic must pin its entry anchor")
        prices = pd.DataFrame({"A":[200.,100.,110.,99.]},
                              index=["2026-06-30","2026-07-01","2026-07-02","2026-07-03"])
        actual = convert(prices,"2026-07-01",["2026-07-02","2026-07-03"],["A"])
        np.testing.assert_allclose(actual.A,[.1,-.1])
        with self.assertRaises(ValueError):
            convert(prices.drop("2026-07-01"),"2026-07-01",["2026-07-02","2026-07-03"],["A"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
