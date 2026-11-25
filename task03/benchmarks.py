"""任务3 四类基准：等权、均值—方差（风险约束）、CVaR 线性规划、VaR 启发式。

约定：
- 输入 R 为参数估计期 T×N 小数日收益（NumPy），mu 为同一估计期样本均值；
- 输出 (weights: pd.Series | None, info: dict)，info 至少含 status 与 solver；
- 失败/不可行返回 None + 明确状态，由调用方保留失败行；
- 参数（sigma_max、alpha、loss_max、budget、seed）必须来自冻结 protocol，不得按检验期结果调整。
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from evalcore import empirical_var_cvar, validate_weights


def _as_arrays(R, mu=None):
    R = np.asarray(R, dtype=float)
    if R.ndim != 2 or R.size == 0:
        raise ValueError("R 必须是 T×N 非空矩阵")
    if not np.isfinite(R).all():
        raise ValueError("R 含非有限值")
    if mu is None:
        mu_vec = R.mean(axis=0)
    else:
        mu_vec = np.asarray(mu, dtype=float)
        if mu_vec.shape != (R.shape[1],) or not np.isfinite(mu_vec).all():
            raise ValueError("mu 维度不匹配或含非有限值")
    return R, mu_vec


def _to_series(u, assets):
    return pd.Series(np.asarray(u, dtype=float), index=list(assets), dtype=float)


def equal_weight(assets) -> tuple[pd.Series, dict]:
    assets = list(assets)
    n = len(assets)
    if n == 0:
        return None, {"status": "fail", "reason": "没有资产", "solver": "closed-form"}
    return _to_series(np.full(n, 1.0 / n), assets), {
        "status": "optimal", "solver": "closed-form", "objective": None,
        "seconds": 0.0,
    }


def _long_only_min_variance(R, x0):
    n = R.shape[1]
    Sigma = np.cov(R, rowvar=False, ddof=1)
    Sigma = np.atleast_2d(Sigma)

    def fun(u):
        return float(u @ Sigma @ u)

    def jac(u):
        return 2.0 * Sigma @ u

    res = minimize(
        fun, x0, jac=jac, method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda u: u.sum() - 1.0,
                      "jac": lambda u: np.ones(n)}],
        options={"maxiter": 1000, "ftol": 1e-14},
    )
    u = np.clip(res.x, 0.0, None)
    u = u / u.sum()
    return u, res


def mean_variance(R, sigma_max: float, assets, mu=None) -> tuple[pd.Series | None, dict]:
    """max mu'u  s.t. 1'u=1, u>=0, sqrt(u'Sigma u) <= sigma_max（日频波动上限）。"""
    start = time.time()
    R, mu_vec = _as_arrays(R, mu)
    n = R.shape[1]
    if not (sigma_max > 0):
        return None, {"status": "fail", "reason": "sigma_max 必须为正", "solver": "SLSQP"}
    Sigma = np.atleast_2d(np.cov(R, rowvar=False, ddof=1))
    eq = np.full(n, 1.0 / n)

    min_var_u, min_var_res = _long_only_min_variance(R, eq)
    min_vol = float(np.sqrt(min_var_u @ Sigma @ min_var_u))
    if min_vol > sigma_max * (1.0 + 1e-10):
        return None, {
            "status": "infeasible",
            "reason": f"全投资非负组合最小日波动 {min_vol:.6f} 已超过上限 {sigma_max:.6f}",
            "solver": "SLSQP", "min_feasible_vol": min_vol,
            "seconds": round(time.time() - start, 6),
        }

    x0 = eq if np.sqrt(eq @ Sigma @ eq) <= sigma_max else min_var_u
    cons = [
        {"type": "eq", "fun": lambda u: u.sum() - 1.0, "jac": lambda u: np.ones(n)},
        {"type": "ineq", "fun": lambda u: sigma_max ** 2 - float(u @ Sigma @ u),
         "jac": lambda u: -2.0 * Sigma @ u},
    ]
    res = minimize(
        lambda u: -float(mu_vec @ u), x0, jac=lambda u: -mu_vec, method="SLSQP",
        bounds=[(0.0, 1.0)] * n, constraints=cons,
        options={"maxiter": 2000, "ftol": 1e-14},
    )
    u = np.clip(res.x, 0.0, None)
    if u.sum() <= 0:
        return None, {"status": "fail", "reason": "解全为零", "solver": "SLSQP",
                      "seconds": round(time.time() - start, 6)}
    u = u / u.sum()
    vol = float(np.sqrt(u @ Sigma @ u))
    info = {
        "status": "optimal" if res.success and vol <= sigma_max * (1 + 1e-6) else "optimal_inaccurate",
        "solver": "SLSQP", "iterations": int(res.nit), "message": str(res.message),
        "objective": float(mu_vec @ u), "risk_daily": vol,
        "constraint_slack": float(sigma_max ** 2 - vol ** 2),
        "seconds": round(time.time() - start, 6),
    }
    if info["status"] != "optimal":
        return None, info
    return _to_series(u, assets), info


def cvar_portfolio(R, alpha: float, loss_max: float, assets, mu=None
                   ) -> tuple[pd.Series | None, dict]:
    """min -mu'u  s.t. zeta + 1/(T(1-alpha)) sum xi_t <= loss_max,
    xi_t >= -r_t'u - zeta, xi_t >= 0, u>=0, 1'u=1, zeta 自由（全实轴）。"""
    start = time.time()
    R, mu_vec = _as_arrays(R, mu)
    T, n = R.shape
    if not (0 < alpha < 1):
        return None, {"status": "fail", "reason": "alpha 必须在 (0,1)", "solver": "highs"}
    tail = T * (1.0 - alpha)
    if tail < 1.0:
        return None, {"status": "fail",
                      "reason": f"尾部样本不足：T(1-alpha)={tail:.3f}<1", "solver": "highs"}

    # 变量顺序：[u(n), zeta(1), xi(T)]
    c = np.concatenate([-mu_vec, [0.0], np.zeros(T)])
    A_eq = np.concatenate([np.ones(n), [0.0], np.zeros(T)])[None, :]
    A_ub = np.empty((T + 1, n + 1 + T))
    A_ub[0, :n] = 0.0
    A_ub[0, n] = 1.0
    A_ub[0, n + 1:] = 1.0 / tail
    for t in range(T):
        row = np.zeros(n + 1 + T)
        row[:n] = -R[t, :]
        row[n] = -1.0
        row[n + 1 + t] = -1.0
        A_ub[t + 1] = row
    b_ub = np.concatenate([[loss_max], np.zeros(T)])
    bounds = [(0.0, None)] * n + [(None, None)] + [(0.0, None)] * T
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.array([1.0]),
                  bounds=bounds, method="highs")
    if not res.success:
        return None, {
            "status": "infeasible" if res.status == 2 else "fail",
            "reason": str(res.message), "solver": "highs",
            "seconds": round(time.time() - start, 6),
        }
    u = np.clip(res.x[:n], 0.0, None)
    u = u / u.sum()
    losses = -(R @ u)
    var, cvar = empirical_var_cvar(losses, alpha)
    return _to_series(u, assets), {
        "status": "optimal", "solver": "highs", "zeta": float(res.x[n]),
        "empirical_var": var, "empirical_cvar": cvar,
        "objective": float(mu_vec @ u),
        "loss_max": loss_max,
        "seconds": round(time.time() - start, 6),
    }


def var_candidates(n: int, budget: int, seed: int, warm_starts=None) -> np.ndarray:
    """Fixed total budget: equal weight, vertices, warm starts, then Dirichlet(1).

    Warm starts consume the same budget as random points; duplicate allocations
    retain distinct candidate IDs so their solver provenance remains traceable.
    """
    warm_starts = list(warm_starts or [])
    if n < 1:
        raise ValueError("n must be positive")
    required = 1 + n + len(warm_starts)
    if budget < required:
        raise ValueError(f"budget={budget} cannot hold {required} deterministic candidates")
    rng = np.random.default_rng(seed)
    rows = [np.full(n, 1.0 / n), *np.eye(n), *warm_starts]
    rows.extend(rng.dirichlet(np.ones(n), size=budget - required))
    return np.asarray(rows, dtype=float)


def var_heuristic(R, alpha: float, loss_max: float, assets, budget: int,
                  seed: int, mu=None, warm_starts=None) -> tuple[pd.Series | None, dict]:
    """Select the highest estimation mean among feasible sampled candidates.

    This provides neither global optimality nor an infeasibility certificate.
    warm_starts maps IDs to labeled invested weights from estimation-period
    solvers; every supplied allocation is included within the total budget.
    """
    start = time.time()
    R, mu_vec = _as_arrays(R, mu)
    T, n = R.shape
    assets = list(assets)
    if len(assets) != n or len(set(assets)) != n:
        raise ValueError("assets must uniquely label every return column")
    if not (0 < alpha < 1):
        return None, {"status": "fail", "reason": "alpha must be in (0,1)",
                      "solver": "random-search"}
    warm_starts = dict(warm_starts or {})
    warm_vectors = [validate_weights(w, assets) for w in warm_starts.values()]
    U = var_candidates(n, budget, seed, warm_starts=warm_vectors)
    candidate_ids = ["equal", *[f"single:{a}" for a in assets], *warm_starts,
                     *[f"random:{i}" for i in range(budget - 1 - n - len(warm_vectors))]]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("warm-start IDs must be distinct from reserved candidate IDs")
    means = U @ mu_vec
    losses = -(R @ U.T)
    ordered = np.sort(losses, axis=0)
    k = int(np.ceil(alpha * T))
    var = ordered[k - 1, :]
    risk_tolerance = 1e-10
    feasible = var <= loss_max + risk_tolerance
    warm_trace = []
    for i, candidate_id in enumerate(warm_starts, start=1 + n):
        warm_trace.append({
            "candidate_id": candidate_id, "candidate_index": i,
            "weights": dict(zip(assets, U[i].tolist())),
            "objective": float(means[i]), "empirical_var": float(var[i]),
            "feasible": bool(feasible[i]),
        })
    info = {
        "solver": "random-search", "budget": budget, "total_candidates": len(U),
        "random_budget": budget - 1 - n - len(warm_vectors), "seed": seed,
        "random_distribution": "Dirichlet(1)", "risk_tolerance": risk_tolerance,
        "feasible_count": int(feasible.sum()), "warm_starts": warm_trace,
    }
    if not feasible.any():
        return None, {**info, "status": "no_feasible_candidate",
                      "reason": "No feasible sampled candidate; infeasibility is not established",
                      "seconds": round(time.time() - start, 6)}
    pick = int(np.argmax(np.where(feasible, means, -np.inf)))
    u = U[pick]
    emp_var, emp_cvar = empirical_var_cvar(-(R @ u), alpha)
    return _to_series(u, assets), {
        **info, "status": "feasible_heuristic",
        "selected_candidate_id": candidate_ids[pick], "selected_candidate_index": pick,
        "empirical_var": emp_var, "empirical_cvar": emp_cvar,
        "objective": float(mu_vec @ u),
        "seconds": round(time.time() - start, 6),
    }
