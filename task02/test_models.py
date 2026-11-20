import itertools
import unittest

import mpmath as mp
import numpy as np
from scipy.optimize import minimize_scalar
from models import target, adjusted_theta, lasso, extension
from run import check_labels, inputs, read_json, HERE


class ModelChecks(unittest.TestCase):
    def test_hand_lasso(self):
        w,d=lasso(np.eye(2),np.array([.2,.1]),.1)
        np.testing.assert_allclose(w,[.15,.05])
        np.testing.assert_allclose(w/w.sum(),[.75,.25])
        self.assertEqual(d["status"],"optimal")
        self.assertAlmostEqual(d["objective"],.025)

    def test_zero(self):
        w,d=lasso(np.eye(2),np.array([.2,.1]),.5)
        self.assertEqual(d["status"],"zero_solution")
        self.assertEqual(w.sum(),0)

    def test_duplicate_columns(self):
        w,d=lasso(np.ones((2,2)),np.ones(2),.4)
        self.assertEqual(d["status"],"optimal")
        self.assertAlmostEqual(w.sum(),.9)
        self.assertAlmostEqual(d["objective"],.38)

    def test_invalid_inputs(self):
        for lam in [-1,np.nan]:
            with self.assertRaises(ValueError):
                lasso(np.eye(2),np.ones(2),lam)
        with self.assertRaises(ValueError):
            lasso(np.eye(2),np.array([1,np.nan]),0)
        with self.assertRaises(ValueError):
            check_labels(["date","b","a"],["a","b"])

    def test_known_target(self):
        # Two orthogonal centered columns with sample variance 1.
        R=np.array([[1,0],[-1,0],[0,1],[0,-1],[0,0],[0,0]])*np.sqrt(2.5)+[.2,.1]
        y,d=target(R,.04)
        with mp.workdps(80):
            s=mp.mpf(".05")
            B=mp.betainc(1,2,0,s/(1+s))
            a=((2*s-2)+2*s*(1+s)**(-2)/B)/6
            self.assertAlmostEqual(d["theta_s"],float(s))
            self.assertAlmostEqual(d["theta_a"],float(a))
            np.testing.assert_allclose(y,float(mp.mpf(".04")*(1+a)/mp.sqrt(a)))

    def test_target_boundaries(self):
        for R in [np.eye(4),np.ones((10,2))]:
            with self.assertRaises(ValueError):
                target(R,.04)
        with self.assertRaisesRegex(ArithmeticError,"underflow"):
            adjusted_theta(1e-15,359,100)
        with self.assertRaises(ValueError):
            adjusted_theta(0,10,2)

    def test_integer_vs_enumeration(self):
        R=np.random.default_rng(23).normal(.001,.02,(90,5))
        gamma,cap=15,.6
        def objective(w):
            r=R@w
            return gamma/2*r.var(ddof=1)-r.mean()
        values=[]
        for i,j in itertools.combinations(range(5),2):
            def value(x):
                w=np.zeros(5)
                w[i],w[j]=x,1-x
                return objective(w)
            fit=minimize_scalar(value,bounds=(1-cap,cap),method="bounded")
            values.append(min(value(1-cap),value(cap),fit.fun))
        u,d=extension(R,gamma,cap,2)
        self.assertEqual(d["status"],"optimal")
        self.assertAlmostEqual(objective(u),min(values),places=8)
        self.assertLessEqual(sum(u>1e-8),2)
        cont,dc=extension(R,gamma,cap)
        self.assertLessEqual(objective(cont),objective(u)+1e-8)
        self.assertEqual(dc["status"],"optimal")
        self.assertEqual(extension(R,gamma,.2,2)[1]["status"],"infeasible")

    def test_real_three_stocks(self):
        R,*_=inputs(read_json(HERE/"protocol.json"))
        R=R[:,:3]
        y,_=target(R,.04)
        _,d=lasso(R,y,.01*2*max(R.T@y))
        self.assertEqual(d["status"],"optimal")


if __name__=="__main__":
    unittest.main(verbosity=2)
