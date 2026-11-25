"""Attribution and original-appendix compatibility checks, separate from main results."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from scipy import integrate, special

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"task02"))
sys.path.insert(0, str(ROOT/"task03"))
from models import adjusted_theta, lasso
from evalcore import validate_weights, buy_and_hold_nav, rebalanced_nav, performance_metrics


def aligned_weights(weights, assets, allow_subset=False):
    s = pd.Series(weights,dtype=float)
    if allow_subset:
        if not s.index.is_unique or set(s.index)-set(assets):
            raise ValueError("invalid_extension_asset_labels")
        validate_weights(s,list(s.index))
        return s.reindex(assets,fill_value=0.)
    validate_weights(s,assets)
    return s.reindex(assets)


def verify_nav(calculated, published):
    assert calculated.index.equals(published.index), "NAV date identity mismatch"
    np.testing.assert_allclose(calculated.to_numpy(),published.to_numpy(),rtol=0,atol=1e-12)


def raw_close_returns(prices, entry, return_dates, assets):
    window = prices.reindex(index=[entry,*return_dates],columns=assets)
    if not np.isfinite(window.to_numpy()).all() or (window.to_numpy() <= 0).any():
        raise ValueError("raw_price_entry_anchor_or_valuation_missing")
    return window.pct_change(fill_method=None).iloc[1:]


def stock_attribution(weights, returns):
    """Initial-weight times full holding-period stock return; sums to portfolio P&L."""
    if not np.isfinite(returns.to_numpy()).all() or (returns.to_numpy() <= -1).any():
        raise ValueError("invalid_simple_returns")
    u = validate_weights(weights, list(returns.columns))
    stock_return = np.prod(1+returns.to_numpy(), axis=0)-1
    result = pd.DataFrame(dict(weight=u, asset_return=stock_return, contribution=u*stock_return),
                          index=returns.columns)
    # The public evaluator fixes initial NAV at 1 even when solver weights have
    # a budget residual within tolerance; keep that tiny reconciliation explicit.
    result.attrs["initial_budget_residual"] = float(u.sum()-1.)
    return result


def monthly_returns(nav):
    daily = nav.pct_change(fill_method=None).iloc[1:]
    months = pd.to_datetime(daily.index).to_period("M").astype(str)
    return (1+daily).groupby(months).prod()-1


def paper_normalize(w, threshold=1e-4):
    w = np.asarray(w, dtype=float).copy()
    if not np.isfinite(w).all() or (w < -1e-12).any():
        raise ValueError("invalid_paper_coefficients")
    w[w <= threshold] = 0
    if w.sum() <= 1e-12:
        raise ValueError("paper_zero_solution_after_threshold")
    return w/w.sum()


def paper_target(R, sigma, ridge=1e-6, method="quad"):
    """Reproduce the paper's covariance ridge and optional literal quadrature."""
    R = np.asarray(R, dtype=float)
    t, n = R.shape
    if t <= n+2 or not np.isfinite(R).all() or sigma <= 0 or ridge < 0:
        raise ValueError("invalid_paper_target_input")
    mu = R.mean(axis=0)
    cov = np.atleast_2d(np.cov(R, rowvar=False, ddof=1)) + ridge*np.eye(n)
    s = float(mu @ np.linalg.inv(cov) @ mu)
    a, b, x = n/2, (t-n)/2, s/(1+s)
    beta_stable = float(np.exp(special.betaln(a,b))*special.betainc(a,b,x))
    if method == "quad":
        beta, error = integrate.quad(lambda z: z**(a-1)*(1-z)**(b-1), 0, x)
        if not beta > 0:
            raise ArithmeticError("paper_quadrature_underflow")
        theta = ((t-n-2)*s-n)/t + 2*s**a*(1+s)**(-(t-2)/2)/(t*beta)
    elif method == "stable":
        beta, error, theta = beta_stable, 0., adjusted_theta(s,t,n)
    else:
        raise ValueError("unknown_paper_beta_method")
    if not np.isfinite(theta) or theta <= 0:
        raise ArithmeticError("paper_invalid_theta")
    rc = float(sigma*(1+theta)/np.sqrt(theta))
    return np.full(t,rc), dict(theta_s=s,theta_a=float(theta),r_c=rc,ridge=float(ridge),
        beta_method=method,beta_value=float(beta),quad_absolute_error_estimate=float(error),
        beta_relative_error_vs_stable=float(abs(beta-beta_stable)/beta_stable))


def fit_paper_style(R, sigma=.04, ridge=1e-6, method="quad", seed=42, grid=None):
    """Post-hoc diagnostic: raw-risk shuffled CV, then original coefficient trim.

    Keeps the original zero-candidate scoring and tie rule; never chooses on OOS.
    This does not reconstruct the missing 278-stock original estimation matrix.
    """
    grid = np.linspace(0,10,100) if grid is None else np.asarray(grid,dtype=float)
    order = np.arange(len(R))
    np.random.RandomState(seed).shuffle(order)
    folds = []
    for fold, validation_ids in enumerate(np.array_split(order,10),1):
        validation_ids = np.sort(validation_ids)
        training_ids = np.setdiff1d(np.arange(len(R)),validation_ids)
        train, validation = R[training_ids], R[validation_ids]
        y, stats = paper_target(train,sigma,ridge,method)
        lambda_max = max(0.,float(2*max(train.T@y)))
        candidates = []
        for lam in grid:
            if lam >= lambda_max:
                w = np.zeros(R.shape[1])
            else:
                w, info = lasso(train,y,float(lam))
                if info["status"] not in {"optimal","zero_solution"}:
                    raise ArithmeticError("paper_candidate_failed_kkt")
            variance = float(np.var(validation@w,ddof=1))
            candidates.append((abs(variance-sigma**2),-float(lam),variance))
        error, negative_lambda, variance = min(candidates)
        folds.append(dict(fold=fold,selected_lambda=-negative_lambda,variance_error=error,
                          raw_variance=variance,target=stats))
    lam = float(np.mean([f["selected_lambda"] for f in folds]))
    y, stats = paper_target(R,sigma,ridge,method)
    raw, info = lasso(R,y,lam)
    if info["status"] != "optimal":
        raise ArithmeticError("paper_refit_not_investable")
    invested = paper_normalize(raw)
    return raw, invested, dict(lambda_value=lam,folds=folds,target=stats,
        coefficient_sum_before_trim=float(raw.sum()),removed_coefficient_mass=float(raw[raw<=1e-4].sum()),
        holdings=int((invested>1e-8).sum()),max_weight=float(max(invested)),status=info["status"])


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diagnose(output):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    data = ROOT/"data_delivery_01/data"
    periods = json.loads((data/"periods.json").read_text(encoding="utf-8"))
    returns = pd.read_csv(data/"model_returns_wide.csv",index_col=0)
    train = returns.loc[periods["estimation_start"]:periods["estimation_end"]]
    test = returns.loc[periods["proposed_first_return_date"]:periods["test_end"]]
    ids = list(train.columns)
    names = pd.read_csv(data/"model_assets.csv").set_index("asset_id").name
    all_weights = pd.concat([pd.read_csv(ROOT/"task02/results/weights.csv"),
                             pd.read_csv(ROOT/"task03/results/weights_benchmarks.csv")])
    metrics = pd.read_csv(ROOT/"task03/results/metrics.csv").set_index("model_id")
    published_nav = pd.read_csv(ROOT/"task03/results/nav.csv",index_col=0)
    index_returns = pd.read_csv(data/"index_daily.csv",index_col=0).daily_simple_return
    attributes, summaries, month_rows = [], [], []
    for model, group in all_weights.groupby("model_id",sort=False):
        if group.status.iloc[0] not in {"optimal","feasible_heuristic"}:
            continue
        u = aligned_weights(group.set_index("asset_id").invested_weight,ids,
                            allow_subset=metrics.loc[model,"role"] == "extension")
        contribution = stock_attribution(u,test)
        contribution.insert(0,"model_id",model)
        contribution.insert(1,"name",names.reindex(contribution.index))
        contribution["train_daily_mean"] = train.mean()
        contribution["test_daily_mean"] = test.mean()
        attributes.append(contribution.rename_axis("asset_id").reset_index())
        nav = buy_and_hold_nav(u.to_numpy(),test,periods["proposed_entry_date"])
        budget_residual = contribution.attrs["initial_budget_residual"]
        np.testing.assert_allclose(contribution.contribution.sum()+budget_residual,
                                   metrics.loc[model,"cumulative_return"],rtol=0,atol=1e-12)
        verify_nav(nav,published_nav[model])
        train_port = train.to_numpy()@u.to_numpy()
        realized = nav.pct_change().iloc[1:].to_numpy()
        train_idx = index_returns.loc[train.index].to_numpy()
        test_idx = index_returns.loc[test.index].to_numpy()
        summary = metrics.loc[model].to_dict()
        summary.update(model_id=model,initial_budget_residual=budget_residual,train_daily_mean=float(train_port.mean()),
            train_daily_sd=float(train_port.std(ddof=1)),train_beta=float(np.cov(train_port,train_idx)[0,1]/np.var(train_idx,ddof=1)),
            test_beta=float(np.cov(realized,test_idx)[0,1]/np.var(test_idx,ddof=1)),
            effective_holdings=float(1/(u@u)),top3_weight=float(u.nlargest(3).sum()),
            losing_positions=int(((contribution.asset_return<0)&(u>1e-8)).sum()))
        summaries.append(summary)
        previous_value = float(nav.iloc[0])
        month_labels = pd.to_datetime(nav.index).to_period("M").astype(str)
        for month,value in monthly_returns(nav).items():
            end_value = float(nav.loc[month_labels == month].iloc[-1])
            month_rows.append(dict(model_id=model,month=month,return_value=float(value),
                                  start_nav=previous_value,end_nav=end_value,
                                  pnl_from_initial_capital=end_value-previous_value))
            previous_value = end_value
    attribution = pd.concat(attributes,ignore_index=True)
    attribution.to_csv(output/"stock_attribution.csv",index=False,encoding="utf-8-sig")
    comparison = pd.DataFrame(summaries)
    comparison.to_csv(output/"model_comparison.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(month_rows).to_csv(output/"monthly_returns.csv",index=False,encoding="utf-8-sig")
    main = attribution.loc[attribution.model_id.eq("l1_cv")&attribution.weight.gt(1e-8)].sort_values("contribution")
    active = attribution[attribution.model_id.eq("l1_cv")].set_index("asset_id").copy()
    equal = attribution[attribution.model_id.eq("equal")].set_index("asset_id")
    active["active_contribution_vs_equal"] = active.contribution-equal.contribution
    active.to_csv(output/"active_attribution_vs_equal.csv",encoding="utf-8-sig")
    regime = pd.DataFrame(dict(name=names,train_daily_mean=train.mean(),test_daily_mean=test.mean(),
                              train_return=(1+train).prod()-1,test_return=(1+test).prod()-1))
    regime["train_mean_quintile"] = pd.qcut(regime.train_daily_mean.rank(method="first"),5,labels=False)+1
    regime["main_weight"] = active.weight
    regime.to_csv(output/"stock_regimes.csv",encoding="utf-8-sig")
    regime.groupby("train_mean_quintile").agg(stock_count=("name","size"),
        train_daily_mean=("train_daily_mean","mean"),test_stock_return=("test_return","mean"),
        main_weight=("main_weight","sum")).to_csv(output/"regime_groups.csv",encoding="utf-8-sig")
    source_group = all_weights[all_weights.model_id.eq("l1_cv")].set_index("asset_id")
    u = source_group.invested_weight.reindex(ids).to_numpy()
    raw = source_group.raw_coefficient.reindex(ids).to_numpy()
    entry = periods["proposed_entry_date"]
    modes = []
    variant_meta = {}

    def evaluate(label, weights, note, return_matrix=test):
        for holding, fn in [("buy_and_hold",buy_and_hold_nav),("daily_rebalanced",rebalanced_nav)]:
            nav = fn(weights,return_matrix,entry)
            modes.append(dict(variant=label,holding=holding,note=note,holdings=int(sum(weights>1e-8)),
                              max_weight=float(max(weights)),**performance_metrics(nav)))

    evaluate("current_cv",u,"Current main coefficients; diagnostic accounting alternatives")
    prices = pd.read_csv(data/"prices_long.csv",usecols=["date","asset_id","close_raw"])
    raw_prices = prices.pivot(index="date",columns="asset_id",values="close_raw").reindex(columns=ids)
    raw_returns = raw_close_returns(raw_prices,entry,list(test.index),ids)
    evaluate("current_cv_raw_close_diagnostic",u,
             "Unadjusted close only, excludes dividend/action cash flows; not a replacement total return",raw_returns)
    trimmed = paper_normalize(raw)
    evaluate("current_cv_original_trim",trimmed,"Apply only the original <=0.0001 coefficient threshold")
    for label,method in [("paper_ridge_stable","stable"),("paper_literal_quad","quad")]:
        coefficients, weights, meta = fit_paper_style(train.to_numpy(),ridge=1e-6,method=method)
        variant_meta[label] = meta
        evaluate(label,weights,"Refit ten folds with original ridge, threshold and specified Beta computation")
        pd.DataFrame(dict(asset_id=ids,name=names.reindex(ids),raw_coefficient=coefficients,
                          invested_weight=weights)).to_csv(output/(label+"_weights.csv"),index=False,encoding="utf-8-sig")
    paper_lambda = .4141
    target_value = json.loads((ROOT/"task02/results/manifest.json").read_text(encoding="utf-8"))["target"]["r_c"]
    coefficients, fit = lasso(train.to_numpy(),np.full(len(train),target_value),paper_lambda)
    if fit["status"] == "optimal":
        evaluate("paper_reported_lambda_on_current_data",paper_normalize(coefficients),
                 "Post-hoc diagnostic only: transplant paper reported lambda 0.4141 onto current data")
    pd.DataFrame(modes).to_csv(output/"paper_compatibility.csv",index=False,encoding="utf-8-sig")
    full = pd.read_csv(data/"returns_wide.csv",index_col=0).loc[test.index]
    if not np.isfinite(full.to_numpy()).all():
        raise ValueError("full_pool_test_missing_returns")
    evidence = dict(data_version=periods["data_version"],periods=periods,
        main_model="l1_cv",main_holdings=main.to_dict("records"),
        main_minus_equal=float(metrics.loc["l1_cv","cumulative_return"]-metrics.loc["equal","cumulative_return"]),
        main_minus_index=float(metrics.loc["l1_cv","cumulative_return"]-metrics.loc["csi500_index_price","cumulative_return"]),
        selected_stocks_losing=int(((1+test).prod()<1).sum()),
        full_500_equal_diagnostic=float(((1+full).prod()-1).mean()),
        cross_section_train_vs_test_mean_correlation=float(train.mean().corr(test.mean())),
        original_trim_weight_max_change=float(abs(trimmed-u).max()),paper_variants=variant_meta,
        inputs={str(p.relative_to(ROOT)):_sha(p) for p in
                [data/name for name in ["model_returns_wide.csv","model_assets.csv","returns_wide.csv",
                                       "index_daily.csv","prices_long.csv","periods.json"]]+
                [ROOT/"task02/results/weights.csv",ROOT/"task02/results/manifest.json",
                 ROOT/"task03/results/metrics.csv",ROOT/"task03/results/nav.csv",
                 ROOT/"task03/results/weights_benchmarks.csv",Path(__file__)]},
        caveats=["Post-hoc explanations are not new parameter selection.",
                 "Paper-compatible choices on current data do not recreate the missing original estimation data.",
                 "Current constituent lookahead and supplier-adjusted proxy limitations remain."])
    (output/"diagnostics.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(comparison.loc[comparison.role.eq("main"),["model_id","cumulative_return","train_daily_mean","train_daily_sd","top3_weight","effective_holdings"]].to_string(index=False))
    print("\nPAPER COMPATIBILITY\n"+pd.DataFrame(modes)[["variant","holding","cumulative_return","holdings"]].to_string(index=False))
    print("\nMAIN STOCK CONTRIBUTIONS\n"+main[["asset_id","name","weight","asset_return","contribution"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"analysis/results")
    diagnose(parser.parse_args().output)
