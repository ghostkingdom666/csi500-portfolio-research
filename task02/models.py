"""Task 02 estimators. Inputs and objectives use daily decimal returns."""
import time
import numpy as np
from scipy import special, optimize
from pyscipopt import Model, quicksum


def adjusted_theta(s, t, n):
    if t <= n + 2 or not np.isfinite(s) or s <= 0:
        raise ValueError("invalid_theta_input")
    a, b = n / 2, (t - n) / 2
    ix = special.betainc(a, b, s / (1 + s))
    if ix == 0:
        raise ArithmeticError("beta_underflow")
    log_ratio = a*np.log(s) - (t-2)/2*np.log1p(s) - special.betaln(a,b) - np.log(ix)
    theta = ((t-n-2)*s - n + 2*np.exp(log_ratio)) / t
    if not np.isfinite(theta) or theta <= 0:
        raise ArithmeticError("invalid_adjusted_theta")
    return float(theta)


def target(R, sigma):
    t, n = R.shape
    if t <= n+2 or not np.isfinite(R).all() or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("invalid_estimation_window_or_sigma")
    mu = R.mean(axis=0)
    cov = np.atleast_2d(np.cov(R, rowvar=False, ddof=1))
    eig = np.linalg.eigvalsh(cov)
    if eig[0] <= n * np.finfo(float).eps * eig[-1]:
        raise ValueError("singular_covariance")
    s = float(mu @ np.linalg.solve(cov, mu))
    a = adjusted_theta(s,t,n)
    rc = sigma*(1+a)/np.sqrt(a)
    return np.full(t,rc), dict(theta_s=s,theta_a=a,r_c=rc,eigen_min=eig[0],
                               eigen_max=eig[-1],condition=eig[-1]/eig[0],
                               symmetry_error=np.max(abs(cov-cov.T)))


def lasso(R, y, lam):
    if y.shape != (len(R),) or not all(np.isfinite(x).all() for x in [R,y,lam]) or lam < 0:
        raise ValueError("invalid_lasso_input")
    start = time.perf_counter()
    # For full column rank, rewrite the linear penalty as a shift of the NNLS response.
    shift, _, rank, _ = np.linalg.lstsq(R.T, np.full(R.shape[1],lam/2), rcond=None)
    if rank == R.shape[1]:
        w, _ = optimize.nnls(R,y-shift)
        solver = "scipy.nnls"
    else:
        def fun(w):
            r = R@w-y
            return r@r + lam*w.sum(), 2*R.T@r+lam
        fit = optimize.minimize(fun,np.zeros(R.shape[1]),jac=True,bounds=[(0,None)]*R.shape[1],
                                method="L-BFGS-B",options={"gtol":1e-12,"ftol":1e-15})
        if not fit.success:
            raise RuntimeError(str(fit.message))
        w, solver = fit.x, "scipy.L-BFGS-B"
    r = R@w-y
    g = 2*R.T@r+lam
    primal = max(0.,-w.min())
    dual = max(0.,-g.min())
    comp = max(abs(w*g))
    status = "optimal" if max(primal,dual,comp) <= 1e-7 else "failed_kkt"
    if status == "optimal" and not w.any():
        status = "zero_solution"
    return w, dict(status=status,solver=solver,objective=r@r+lam*w.sum(),
                   primal_violation=primal,dual_violation=dual,complementarity=comp,
                   seconds=time.perf_counter()-start)


def extension(R, gamma, cap, k=None, limit=120):
    n = R.shape[1]
    if gamma <= 0 or not 0 < cap <= 1 or (k is not None and (k < 1 or int(k)!=k)):
        raise ValueError("invalid_extension_parameters")
    if min(n,k or n)*cap < 1-1e-12:
        return None, dict(status="infeasible",reason="insufficient_capacity")
    start = time.perf_counter()
    mu, cov = R.mean(axis=0), np.cov(R,rowvar=False)
    model = Model()
    model.hideOutput()
    model.setRealParam("limits/time",float(limit))
    model.setRealParam("numerics/feastol",1e-10)
    u = [model.addVar(lb=0,ub=cap) for _ in range(n)]
    model.addCons(quicksum(u)==1)
    if k is not None:
        z = [model.addVar(vtype="B") for _ in u]
        for ui,zi in zip(u,z):
            model.addCons(ui<=cap*zi)
        model.addCons(quicksum(z)<=k)
    # Scale both objectives identically for SCIP feasibility tolerances.
    v = model.addVar(lb=None)
    quad = quicksum(float(cov[i,j])*u[i]*u[j] for i in range(n) for j in range(n))
    model.addCons(v >= 10000*(gamma/2*quad-quicksum(float(m)*ui for m,ui in zip(mu,u))))
    model.setObjective(v,"minimize")
    model.optimize()
    reason = str(model.getStatus())
    info = dict(status=reason,reason=reason,solver="SCIP",seconds=time.perf_counter()-start,
                best_bound=model.getDualbound()/10000, nodes=model.getNNodes())
    if model.getNSols()==0:
        return None, info
    raw = np.array([model.getVal(ui) for ui in u])
    # Only remove numerical excursions beyond bounds and binary-forbidden positions.
    weight = np.clip(raw,0,cap)
    if k is not None:
        weight[[model.getVal(zi)<0.5 for zi in z]] = 0
    violation = max(abs(weight.sum()-1),max(abs(weight-raw)))
    obj = gamma/2*weight@cov@weight-mu@weight
    info.update(objective=obj,solver_gap=model.getGap(),primal_violation=violation,
                solver_incumbent=model.getPrimalbound()/10000,raw_weights=raw.tolist())
    if violation>1e-7 or abs(info["solver_incumbent"]-obj)>1e-7:
        info["status"]="failed_residual"
        return None, info
    info["status"]="optimal" if reason=="optimal" else "feasible_"+reason
    return weight, info
