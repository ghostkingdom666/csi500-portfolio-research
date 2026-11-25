"""Four fixed estimation-only baselines, returning labeled weights and audit info.

The caller supplies only estimation-period returns. No function reads files,
entry/test returns, performance, or a target mean. Top-K is a single ranking at
the estimation end; these functions do not implement periodic rebalancing.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _inputs(R, assets, minimum):
    labels = list(assets)
    A = np.asarray(R, dtype=float)
    if A.ndim != 2 or A.shape[0] < minimum or A.shape[1] == 0:
        raise ValueError(f"R must have at least {minimum} return days and one asset")
    if len(labels) != A.shape[1] or len(set(labels)) != len(labels):
        raise ValueError("assets must uniquely label every return column")
    if not all(isinstance(a, str) and a for a in labels):
        raise ValueError("asset_id must be a nonempty string")
    if not np.isfinite(A).all() or (A <= -1).any():
        raise ValueError("returns must be finite decimals strictly greater than -1")
    if isinstance(R, pd.DataFrame):
        if list(R.columns) != labels:
            raise ValueError("assets must match the labeled return column order")
        if not R.index.is_unique or not R.index.is_monotonic_increasing:
            raise ValueError("estimation return days must be unique and increasing")
        dates = [str(x) for x in R.index]
    else:
        dates = [str(i) for i in range(len(A))]
    estimation = {"start": dates[0], "end": dates[-1], "n_days": len(A),
                  "n_assets": len(labels), "ddof": 1}
    return A, labels, dates, estimation


def gmv(R, assets) -> tuple[pd.Series | None, dict]:
    """min u' sample_cov(R, ddof=1) u, u>=0, sum(u)=1, without a mean target."""
    start = time.perf_counter()
    A, labels, _, estimation = _inputs(R, assets, 2)
    covariance = np.atleast_2d(np.cov(A, rowvar=False, ddof=1))
    n = len(labels)
    # Scaling leaves the minimizer unchanged and avoids absolute SLSQP stopping
    # tolerances accepting the initial weights for small daily-return variances.
    scale = float(np.max(np.diag(covariance))) or 1.
    Q = covariance / scale
    res = minimize(lambda u: float(u @ Q @ u), np.full(n, 1/n),
                   jac=lambda u: 2 * Q @ u, method="SLSQP",
                   bounds=[(0., 1.)] * n,
                   constraints=[{"type": "eq", "fun": lambda u: u.sum() - 1.,
                                 "jac": lambda u: np.ones(n)}],
                   options={"maxiter": 2000, "ftol": 1e-13})
    u = np.asarray(res.x)
    # SLSQP bounds may leave insignificant roundoff below zero.
    u = np.maximum(u, 0.)
    if np.isfinite(u).all() and u.sum() > 0:
        u = u / u.sum()
    gradient = 2 * Q @ u
    multiplier = float(u @ gradient)
    active = u > 1e-8
    stationarity = float(np.max(np.abs(gradient[active] - multiplier))) if active.any() else float("inf")
    dual = float(np.max(np.maximum(multiplier - gradient[~active], 0.))) if (~active).any() else 0.
    kkt = max(stationarity, dual, abs(float(u.sum()) - 1.), max(0., -float(u.min())))
    info = {"status": "optimal" if res.success and kkt <= 1e-6 else "fail",
            "solver": "SLSQP", "objective": float(u @ covariance @ u),
            "objective_name": "sample_portfolio_variance", "iterations": int(res.nit),
            "message": str(res.message), "parameters": {"covariance_ddof": 1, "return_target": None},
            "estimation": estimation,
            "diagnostics": {"covariance_scale": scale, "kkt_residual_scaled": kkt,
                            "stationarity_scaled": stationarity, "dual_violation_scaled": dual,
                            "weight_sum": float(u.sum()), "minimum_weight": float(u.min())},
            "seconds": round(time.perf_counter() - start, 6)}
    if info["status"] != "optimal":
        info["reason"] = "GMV solver or scaled KKT validation failed"
        return None, info
    return pd.Series(u, index=labels, dtype=float), info


def inverse_volatility(R, assets) -> tuple[pd.Series | None, dict]:
    """Full-estimation sample-volatility inverse allocation; no asset exclusions."""
    start = time.perf_counter()
    A, labels, _, estimation = _inputs(R, assets, 2)
    sd = A.std(axis=0, ddof=1)
    # Exact constant histories have zero sample variance even when summation
    # roundoff in np.std leaves a tiny positive value.
    sd[np.ptp(A, axis=0) == 0] = 0.
    info = {"solver": "closed-form", "estimation": estimation,
            "parameters": {"volatility_ddof": 1, "zero_volatility": "fail"},
            "scores": dict(zip(labels, sd.tolist()))}
    invalid = ~np.isfinite(sd) | (sd <= 0)
    if invalid.any():
        return None, {**info, "status": "fail", "objective": None,
                      "reason": "Zero or nonfinite sample volatility",
                      "invalid_volatility_assets": [a for a, bad in zip(labels, invalid) if bad],
                      "seconds": round(time.perf_counter() - start, 6)}
    # Equivalent to 1/sd but resistant to overflow for tiny positive sd.
    inverse = sd.min() / sd
    u = inverse / inverse.sum()
    return pd.Series(u, index=labels, dtype=float), {
        **info, "status": "optimal", "objective": float(np.var(A @ u, ddof=1)),
        "objective_name": "sample_portfolio_variance_diagnostic",
        "diagnostics": {"weight_sum": float(u.sum()), "minimum_weight": float(u.min())},
        "seconds": round(time.perf_counter() - start, 6)}


def _selection(scores, labels, k, descending):
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= len(labels):
        raise ValueError("k must be an integer in [1, n_assets]")
    if not np.isfinite(scores).all():
        raise ValueError("selection scores must be finite")
    ranking = sorted(range(len(labels)), key=lambda i: ((-scores[i] if descending else scores[i]), labels[i]))
    selected = [labels[i] for i in ranking[:k]]
    weights = pd.Series(0., index=labels, dtype=float)
    weights.loc[selected] = 1. / k
    return weights, {"scores": dict(zip(labels, scores.tolist())),
                     "ranking": [labels[i] for i in ranking], "selected_assets": selected,
                     "diagnostics": {"holdings": k, "tie_break": "asset_id_lexicographic_ascending"}}


def momentum_top20(R, assets, k=20, lookback=126, skip=21) -> tuple[pd.Series, dict]:
    """Rank prod(1+R[-126:-21])-1 descending, then asset_id; equal Top20."""
    start = time.perf_counter()
    if (isinstance(lookback, bool) or isinstance(skip, bool)
            or not isinstance(lookback, int) or not isinstance(skip, int)
            or lookback < 1 or not 0 <= skip < lookback):
        raise ValueError("lookback/skip must be integers with 0 <= skip < lookback")
    A, labels, dates, estimation = _inputs(R, assets, lookback)
    stop = len(A) - skip
    begin = len(A) - lookback
    scores = np.expm1(np.log1p(A[begin:stop]).sum(axis=0))
    weights, selection = _selection(scores, labels, k, descending=True)
    return weights, {**selection, "status": "optimal", "solver": "fixed-ranking",
                     "objective": float(weights.to_numpy() @ scores),
                     "objective_name": "mean_selected_cumulative_signal_return",
                     "estimation": estimation, "parameters": {"k": k, "lookback": lookback, "skip": skip},
                     "signal": {"start": dates[begin], "end": dates[stop-1], "n_days": lookback-skip,
                                "excluded_recent_days": skip},
                     "seconds": round(time.perf_counter() - start, 6)}


def low_volatility_top20(R, assets, k=20, lookback=60) -> tuple[pd.Series, dict]:
    """Rank sample sd(R[-60:], ddof=1) ascending, then asset_id; equal Top20."""
    start = time.perf_counter()
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 2:
        raise ValueError("lookback must be an integer >= 2")
    A, labels, dates, estimation = _inputs(R, assets, lookback)
    scores = A[-lookback:].std(axis=0, ddof=1)
    weights, selection = _selection(scores, labels, k, descending=False)
    return weights, {**selection, "status": "optimal", "solver": "fixed-ranking",
                     "objective": float(weights.to_numpy() @ scores),
                     "objective_name": "mean_selected_sample_volatility",
                     "estimation": estimation, "parameters": {"k": k, "lookback": lookback},
                     "signal": {"start": dates[-lookback], "end": dates[-1], "n_days": lookback, "ddof": 1},
                     "seconds": round(time.perf_counter() - start, 6)}
