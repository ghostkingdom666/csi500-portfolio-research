"""Recompute exported objectives and constraints without calling either optimizer."""
import csv
from pathlib import Path
import sys

import mpmath as mp
import numpy as np
from scipy.optimize import nnls
from run import HERE, inputs, read_json, digest


def verify_cv(out, R, p):
    """Re-estimate fold targets and rescore the path without calling CV/lasso helpers."""
    cv = read_json(out/"cv_selection.json")
    cfg = p["cross_validation"]
    grid = np.linspace(cfg["lambda_min"], cfg["lambda_max"], cfg["n_lambdas"])
    np.testing.assert_allclose(cv["lambda_grid"], grid, rtol=0, atol=1e-15)
    assert cv["status"] == "selected" and cv["n_splits"] == cfg["n_splits"] and cv["seed"] == cfg["seed"]
    with (out/"cv_scores.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == cfg["n_splits"]*len(grid)
    keys = [(int(row["fold"]), float(row["lambda_value"])) for row in rows]
    assert len(set(keys)) == len(rows)
    scores = dict(zip(keys, rows))
    permutation = np.arange(len(R))
    np.random.RandomState(cfg["seed"]).shuffle(permutation)
    expected_val = np.array_split(permutation, cfg["n_splits"])
    selected, all_val = [], []
    assert len(cv["folds"]) == cfg["n_splits"]
    for i, fold in enumerate(cv["folds"], 1):
        tr, va = fold["train_indices"], fold["validation_indices"]
        assert fold["fold"] == i and va == sorted(expected_val[i-1].tolist())
        assert tr == sorted(set(range(len(R)))-set(va)) and not set(tr)&set(va)
        all_val.extend(va)
        train, validation = R[tr], R[va]
        t, n = train.shape
        mu = train.mean(axis=0)
        with mp.workdps(70):
            theta_s = mp.mpf(float(mu @ np.linalg.solve(np.cov(train.T),mu)))
            B = mp.betainc(mp.mpf(n)/2,mp.mpf(t-n)/2,0,theta_s/(1+theta_s))
            theta_a = ((t-n-2)*theta_s-n+2*theta_s**(mp.mpf(n)/2)*(1+theta_s)**(-mp.mpf(t-2)/2)/B)/t
            rc = float(p["sigma_daily"]*(1+theta_a)/mp.sqrt(theta_a))
        np.testing.assert_allclose([float(theta_s),float(theta_a),rc],
                                   [fold["target"][k] for k in ["theta_s","theta_a","r_c"]],rtol=1e-10)
        y = np.full(t,rc)
        shift = np.linalg.lstsq(train.T,np.ones(n)/2,rcond=None)[0]
        val_cov = np.atleast_2d(np.cov(validation.T,ddof=1))
        candidates = []
        for lam in grid:
            row = scores[(i,float(lam))]
            if row["status"] == "solver_failure":
                assert row["reason"].strip() and row["invested_variance"] == ""
                for key in ["raw_variance","variance_error","coefficient_sum","objective"]:
                    assert row[key] == "" or np.isfinite(float(row[key]))
                continue
            residual_keys = ["primal_violation","dual_violation","complementarity"]
            residual = max(float(row[key]) for key in residual_keys)
            if row["status"] == "failed_kkt":
                assert residual > 1e-7 and row["invested_variance"] == ""
                assert all(np.isfinite(float(row[key])) for key in
                           ["raw_variance","variance_error","coefficient_sum","objective"])
                continue
            assert row["status"] in {"optimal","zero_solution"} and residual <= 1e-7
            w = np.zeros(n) if lam >= max(0.,2*max(train.T@y)) else nnls(train,y-lam*shift)[0]
            total = float(w.sum())
            raw_var = float(w@val_cov@w)
            error = abs(raw_var-p["sigma_daily"]**2)
            np.testing.assert_allclose([raw_var,error,total],
                [float(row[k]) for k in ["raw_variance","variance_error","coefficient_sum"]],rtol=1e-8,atol=1e-12)
            if total > 1e-12:
                assert row["status"] == "optimal"
                np.testing.assert_allclose(raw_var/total**2,float(row["invested_variance"]),rtol=1e-8,atol=1e-12)
                candidates.append((error,-float(lam)))
            else:
                assert row["status"] == "zero_solution" and row["invested_variance"] == ""
        best = -min(candidates)[1]
        assert fold["selected_lambda"] == best
        best_row = scores[(i,best)]
        for key in ["raw_variance","invested_variance","variance_error","coefficient_sum"]:
            np.testing.assert_allclose(fold[key],float(best_row[key]),rtol=1e-10,atol=1e-13)
        selected.append(best)
    assert sorted(all_val) == list(range(len(R)))
    np.testing.assert_allclose(cv["lambda_value"],np.mean(selected),rtol=0,atol=1e-14)
    with (out/"cv_folds.csv").open(encoding="utf-8-sig") as stream:
        exported_folds = list(csv.DictReader(stream))
    assert len(exported_folds) == len(cv["folds"])
    with (HERE.parent/"data_delivery_01/data/calendar.csv").open(encoding="utf-8-sig") as stream:
        days = [r["date"] for r in csv.DictReader(stream) if r["period"] == "estimation"]
    assert cv["estimation_dates"] == days
    for row,fold in zip(exported_folds,cv["folds"]):
        for key in ["train_indices","validation_indices"]:
            assert read_json_value(row[key]) == fold[key]
        assert read_json_value(row["validation_dates"]) == [days[j] for j in fold["validation_indices"]]
        assert read_json_value(row["train_dates"]) == [days[j] for j in fold["train_indices"]]
        for key in ["selected_lambda","raw_variance","invested_variance","variance_error","coefficient_sum"]:
            np.testing.assert_allclose(float(row[key]),fold[key],rtol=1e-10,atol=1e-13)
        np.testing.assert_allclose(float(row["r_c"]),fold["target"]["r_c"],rtol=1e-10)
    print(f"CV: verified {len(cv['folds'])} shuffled folds, {len(rows)} candidates, lambda={cv['lambda_value']:.12g}")
    return cv["lambda_value"]


def read_json_value(value):
    import json
    return json.loads(value)


def verify(out):
    manifest=read_json(out/"manifest.json")
    p=read_json(HERE/"protocol.json")
    R,assets,dates,hashes=inputs(p)
    assert manifest["inputs"]==hashes and manifest["protocol"]==p
    assert manifest["periods"] == dates
    t, n = R.shape
    mu = R.mean(axis=0)
    with mp.workdps(70):
        s = mp.mpf(float(mu @ np.linalg.solve(np.cov(R.T), mu)))
        beta = mp.betainc(mp.mpf(n)/2, mp.mpf(t-n)/2, 0, s/(1+s))
        theta = ((t-n-2)*s-n+2*s**(mp.mpf(n)/2)*(1+s)**(-mp.mpf(t-2)/2)/beta)/t
        rc = float(p["sigma_daily"]*(1+theta)/mp.sqrt(theta))
    lambda_max = max(0.0, 2*max(R.T @ np.full(t,rc)))
    np.testing.assert_allclose([float(s),float(theta),rc,lambda_max],
        [manifest["target"][k] for k in ["theta_s","theta_a","r_c"]]+[manifest["lambda_max"]],
        rtol=1e-11, atol=1e-13)
    for name,value in manifest["code_sha256"].items():
        assert digest(HERE/name)==value
    for name,value in manifest["outputs"].items():
        assert digest(out/name) == value
    cv_lambda = verify_cv(out,R,p)
    index={r["asset_id"]:i for i,r in enumerate(assets)}
    def read(name):
        with (out/name).open(encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    rows=read("weights.csv")
    runs=read("runs.csv")
    expected={f"l1_eta_{eta}" for eta in p["eta_grid"]}|{"l1_cv","mv_continuous","mv_cardinality"}
    assert len(runs)==len(expected) and {r["model_id"] for r in runs}==expected
    assert {r["model_id"] for r in rows}==expected
    ext_ids={a["asset_id"] for a in sorted(assets,key=lambda a:int(a["liquidity_rank"]))[:p["extension_n"]]}
    for run in runs:
        name=run["model_id"]
        group=[r for r in rows if r["model_id"]==name]
        assert all(r["status"] == run["status"] for r in group)
        assert run["data_version"] == p["data_version"]
        for field in ["decision_date","proposed_entry_date","estimation_end"]:
            assert run[field] == dates[field]
        assert run["first_return_date"] == dates["proposed_first_return_date"]
        ids=[r["asset_id"] for r in group]
        assert set(ids)==(set(index) if name.startswith("l1") else ext_ids)
        assert len(ids)==len(set(ids))
        u=np.array([float(r["invested_weight"]) for r in group])
        A=R[:,[index[s] for s in ids]]
        daily=A@u
        assert (u>=0).all() and abs(u.sum()-1)<1e-7
        assert int(run["holdings"])==sum(u>p["support_reporting_threshold"])
        np.testing.assert_allclose([daily.mean(),daily.std(ddof=1),u.max()],
            [float(run[k]) for k in ["daily_mean_in_sample","daily_sd_in_sample","max_weight"]])
        if name.startswith("l1"):
            w=np.array([float(r["raw_coefficient"]) for r in group])
            np.testing.assert_allclose(u,w/w.sum())
            np.testing.assert_allclose([float(run["coefficient_sum"]),float(run["raw_daily_sd_in_sample"])],
                                       [w.sum(),(A@w).std(ddof=1)],rtol=1e-10,atol=1e-13)
            r=A@w-rc
            lam=float(run["lambda_value"])
            if name == "l1_cv":
                assert run["selection_method"] == "random_kfold" and run["role"] == "main"
                np.testing.assert_allclose(lam,cv_lambda,rtol=0,atol=1e-14)
            else:
                eta = float(run["eta"])
                assert name == f"l1_eta_{run['eta']}" and eta in p["eta_grid"] and run["role"] == "sensitivity"
                np.testing.assert_allclose(lam, eta*lambda_max, rtol=1e-11)
            g=2*A.T@r+lam
            assert min(w)>=0 and min(g)>-1e-7 and max(abs(g*w))<1e-7
            objective=r@r+lam*w.sum()
        else:
            assert run["reason"] == "optimal" and float(run["solver_gap"]) <= 1e-7
            assert float(run["gamma"]) == p["gamma"] and float(run["cap"]) == p["cap"]
            assert run["K"] == (str(p["K"]) if name=="mv_cardinality" else "")
            assert u.max()<=p["cap"]+1e-8
            if name=="mv_cardinality":
                assert sum(u>1e-8)<=p["K"]
            objective=p["gamma"]/2*daily.var(ddof=1)-daily.mean()
            assert abs(objective-float(run["best_bound"]))<1e-7
        assert abs(objective-float(run["objective"]))<1e-8
        assert run["status"]=="optimal" and run["decision_date"]==dates["decision_date"]
        print(f"{name}: verified {len(u)} labelled weights, objective={objective:.12g}")


if __name__=="__main__":
    verify(Path(sys.argv[1]) if len(sys.argv)>1 else HERE/"results")
