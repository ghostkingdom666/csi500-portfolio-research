"""Original shuffled-fold lambda selection, restricted to estimation data.

The criterion matches raw-coefficient variance to sigma**2, as in the
paper. Normalized invested variance is exported separately and is NOT
constrained by this criterion. Random folds are not chronological validation.
"""
import numpy as np
from models import target, lasso


class CVSelectionError(ValueError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def shuffled_kfold_indices(n_samples, n_splits=10, seed=42):
    """Match KFold(shuffle=True, random_state=seed), without sklearn."""
    if (not isinstance(n_samples, (int, np.integer)) or
            not isinstance(n_splits, (int, np.integer)) or
            not 2 <= n_splits <= n_samples):
        raise ValueError("invalid_fold_count")
    shuffled = np.arange(n_samples)
    np.random.RandomState(seed).shuffle(shuffled)
    sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    sizes[:n_samples % n_splits] += 1
    folds, start = [], 0
    for size in sizes:
        val = np.sort(shuffled[start:start+size])
        mask = np.ones(n_samples, dtype=bool)
        mask[val] = False
        folds.append((np.flatnonzero(mask), val))
        start += size
    return folds


def random_kfold_cv(R, sigma, n_splits=10, seed=42, lambda_grid=None):
    """Select each fold's lambda, average them, then let the caller refit.

    A tie selects the largest lambda, matching the original <= update.
    Zero or failed fits remain in diagnostics but cannot define an allocation.
    The input must contain estimation rows only; no holdout data is accepted.
    """
    R = np.asarray(R, dtype=float)
    grid = np.asarray(np.linspace(0., 10., 100) if lambda_grid is None else lambda_grid,
                      dtype=float)
    if R.ndim != 2 or not R.size or not np.isfinite(R).all():
        raise ValueError("invalid_cv_matrix")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("invalid_cv_sigma")
    if (grid.ndim != 1 or not grid.size or not np.isfinite(grid).all()
            or (grid < 0).any() or (np.diff(grid) <= 0).any()):
        raise ValueError("invalid_lambda_grid")
    splits = shuffled_kfold_indices(len(R), n_splits, seed)
    if any(len(tr) <= R.shape[1]+2 or len(va) < 2 for tr, va in splits):
        raise ValueError("insufficient_fold_samples")
    result = dict(status="cv_failure", n_splits=int(n_splits), seed=int(seed),
                  criterion="absolute_raw_variance_error", target_variance=float(sigma**2),
                  aggregation="mean_of_fold_selected_lambdas", lambda_grid=grid.tolist(),
                  lambda_value=None, folds=[], scores=[], failed_folds=[])
    for number, (train_ids, val_ids) in enumerate(splits, 1):
        train, validation = R[train_ids], R[val_ids]
        try:
            y, stats = target(train, sigma)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            raise CVSelectionError(f"fold_{number}_target_failure: {exc}", result) from exc
        fold = dict(fold=number, train_indices=train_ids.tolist(),
                    validation_indices=val_ids.tolist(), train_T=len(train), validation_T=len(validation),
                    target={k: float(v) for k, v in stats.items()}, selected_lambda=None)
        result["folds"].append(fold)
        eligible = []
        for lam in grid:
            row = dict(fold=number, lambda_value=float(lam), status="solver_failure",
                       coefficient_sum=None, raw_variance=None, invested_variance=None,
                       variance_error=None, objective=None, primal_violation=None,
                       dual_violation=None, complementarity=None, reason="")
            try:
                w, info = lasso(train, y, float(lam))
                total = float(w.sum())
                variance = float(np.var(validation @ w, ddof=1))
                row.update(status=info["status"], coefficient_sum=total,
                           raw_variance=variance, variance_error=abs(variance-sigma**2),
                           objective=float(info["objective"]))
                row.update({key: float(info[key]) for key in
                            ["primal_violation","dual_violation","complementarity"]})
                if info["status"] == "optimal" and total > 1e-12:
                    row["invested_variance"] = float(np.var(validation @ (w/total), ddof=1))
                    eligible.append(row)
                elif info["status"] == "optimal":
                    row["status"] = "zero_solution"
            except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
                row["reason"] = str(exc)
            result["scores"].append(row)
        if not eligible:
            fold["status"] = "no_investable_candidate"
            result["failed_folds"].append(number)
            continue
        best = min(eligible, key=lambda r: (r["variance_error"], -r["lambda_value"]))
        fold.update(status="selected", selected_lambda=best["lambda_value"], variance_error=best["variance_error"],
                    raw_variance=best["raw_variance"], invested_variance=best["invested_variance"],
                    coefficient_sum=best["coefficient_sum"])
    if result["failed_folds"]:
        raise CVSelectionError(f"no_investable_candidate_in_folds_{result['failed_folds']}", result)
    result["lambda_value"] = float(np.mean([f["selected_lambda"] for f in result["folds"]]))
    result["status"] = "selected"
    return result
