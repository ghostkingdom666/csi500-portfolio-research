import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import pyscipopt
from models import target, lasso, extension
from cross_validation import random_kfold_cv, CVSelectionError

HERE = Path(__file__).resolve().parent
DATA = HERE.parent/"data_delivery_01"/"data"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def check_labels(header, ids):
    if header != ["date",*ids] or len(set(ids)) != len(ids):
        raise ValueError("asset_label_mismatch")


def inputs(p):
    manifest = read_json(DATA/"manifest.json")
    dates = read_json(DATA/"periods.json")
    hashes = {}
    for name in ["periods.json","calendar.csv","model_assets.csv","model_returns_wide.csv"]:
        hashes[name] = digest(DATA/name)
        if hashes[name] != manifest["files"][name]["sha256"]:
            raise ValueError("hash mismatch: "+name)
    if any(dates[key]!=p[key] for key in ["data_version","estimation_start","estimation_end"]):
        raise ValueError("frozen_protocol_mismatch")
    with (DATA/"model_assets.csv").open(encoding="utf-8-sig") as f:
        assets = list(csv.DictReader(f))
    ids = [a["asset_id"] for a in assets]
    with (DATA/"calendar.csv").open(encoding="utf-8-sig") as f:
        days = [r["date"] for r in csv.DictReader(f) if r["period"]=="estimation"]
    with (DATA/"model_returns_wide.csv").open(encoding="utf-8-sig") as f:
        rows = csv.reader(f)
        check_labels(next(rows),ids)
        matrix = []
        for expected in days:
            row = next(rows)
            if row[0]!=expected or not p["estimation_start"]<=expected<=p["estimation_end"]:
                raise ValueError("estimation_date_mismatch")
            matrix.append([float(x) for x in row[1:]])
    R = np.array(matrix)
    if R.shape != (manifest["counts"]["estimation_T"],100) or not np.isfinite(R).all():
        raise ValueError("invalid_frozen_matrix")
    return R, assets, dates, hashes


def run(out):
    p = read_json(HERE/"protocol.json")
    R, assets, dates, hashes = inputs(p)
    pool = np.argsort([int(a["liquidity_rank"]) for a in assets])[:p["extension_n"]]
    y, stats = target(R,p["sigma_daily"])
    max_lam = max(0.,2*(R.T@y).max())
    runs, weights, detailed = [], [], {}
    def record(name, indices, u, raw, info, params):
        info = dict(info)
        if "raw_weights" in info:
            detailed[name] = info.pop("raw_weights")
        row = dict(model_id=name,data_version=p["data_version"],estimation_end=dates["estimation_end"],
                   decision_date=dates["decision_date"],proposed_entry_date=dates["proposed_entry_date"],
                   first_return_date=dates["proposed_first_return_date"],T=len(R),N=len(indices),**params,**info)
        if u is not None:
            daily = R[:,indices]@u
            row.update(holdings=int(sum(u>p["support_reporting_threshold"])),max_weight=max(u),
                       daily_mean_in_sample=daily.mean(),daily_sd_in_sample=daily.std(ddof=1),
                       budget_error=abs(u.sum()-1))
        if raw is not None:
            row.update(coefficient_sum=float(raw.sum()),
                       raw_daily_sd_in_sample=float((R[:,indices]@raw).std(ddof=1)))
        runs.append(row)
        for i,j in enumerate(indices):
            weights.append(dict(model_id=name,asset_id=assets[j]["asset_id"],name=assets[j]["name"],
                                raw_coefficient=None if raw is None else raw[i],
                                invested_weight=None if u is None else u[i],status=info["status"]))
    cfg = p["cross_validation"]
    cv_start = time.perf_counter()
    cv_result = None
    w = u = None
    try:
        cv_result = random_kfold_cv(R, p["sigma_daily"], cfg["n_splits"], cfg["seed"],
                                   np.linspace(cfg["lambda_min"], cfg["lambda_max"], cfg["n_lambdas"]))
        w, info = lasso(R, y, cv_result["lambda_value"])
        if info["status"] == "optimal" and w.sum() > 1e-12:
            u = w/w.sum()
        elif info["status"] == "optimal":
            info["status"] = "zero_solution"
    except CVSelectionError as exc:
        cv_result = exc.diagnostics
        info = dict(status="cv_failure", reason=str(exc), solver="random-kfold/scipy.nnls")
    except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
        info = dict(status="solver_failure", reason=str(exc), solver="random-kfold/scipy.nnls")
    info["seconds"] = time.perf_counter()-cv_start
    if cv_result is None:
        cv_result = dict(status="cv_failure", lambda_value=None, folds=[], scores=[])
    record("l1_cv",range(len(assets)),u,w,info,
           dict(role="main",selection_method="random_kfold",lambda_value=cv_result["lambda_value"],
                sigma=p["sigma_daily"],cv_folds=cfg["n_splits"],cv_seed=cfg["seed"]))
    for eta in p["eta_grid"]:
        lam=eta*max_lam
        w=u=None
        try:
            w,info=lasso(R,y,lam)
            if max_lam==0:
                info["status"]="lambda_max_zero"
            if info["status"]=="optimal":
                u=w/w.sum()
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            info=dict(status="solver_failure",reason=str(exc))
        record(f"l1_eta_{eta}",range(len(assets)),u,w,info,
               dict(role="sensitivity",selection_method="fixed_eta",eta=eta,lambda_value=lam,sigma=p["sigma_daily"]))
    for name,k in [("mv_continuous",None),("mv_cardinality",p["K"])]:
        u,info=extension(R[:,pool],p["gamma"],p["cap"],k,p["time_limit_seconds"])
        record(name,pool,u,None,info,dict(role="extension",gamma=p["gamma"],cap=p["cap"],K=k))
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"runs.csv",runs)
    write_csv(out/"weights.csv",weights)
    with (DATA/"calendar.csv").open(encoding="utf-8-sig") as calendar_file:
        estimation_dates = [row["date"] for row in csv.DictReader(calendar_file) if row["period"] == "estimation"]
    cv_summary = {key: value for key,value in cv_result.items() if key != "scores"}
    cv_summary.update(estimation_dates=estimation_dates, model_id="l1_cv",
                      risk_scope="raw coefficient variance; normalized and future risk are not constrained")
    (out/"cv_selection.json").write_text(json.dumps(cv_summary,indent=2,allow_nan=False),encoding="utf-8")
    write_csv(out/"cv_scores.csv", cv_result["scores"])
    fold_rows = []
    for fold in cv_result["folds"]:
        row = {key: value for key,value in fold.items() if key != "target"}
        row["train_dates"] = [estimation_dates[i] for i in fold["train_indices"]]
        row["validation_dates"] = [estimation_dates[i] for i in fold["validation_indices"]]
        for key in ["train_indices","validation_indices","train_dates","validation_dates"]:
            row[key] = json.dumps(row[key])
        row["r_c"] = fold["target"]["r_c"]
        fold_rows.append(row)
    write_csv(out/"cv_folds.csv",fold_rows)
    manifest=dict(protocol=p,periods=dates,inputs=hashes,code_sha256={n:digest(HERE/n) for n in
                  ["models.py","cross_validation.py","run.py","protocol.json"]},python=platform.python_version(),
                  numpy=np.__version__,scipy=scipy.__version__,pyscipopt=pyscipopt.__version__,
                  scip=pyscipopt.Model().version(),target=stats,lambda_max=max_lam,
                  extension_solver_weights_in_export_order=detailed, evaluation="pending_task03",
                  main_model="l1_cv", model_positioning=p["model_positioning"],
                  outputs={name:digest(out/name) for name in
                           ["weights.csv","runs.csv","cv_selection.json","cv_scores.csv","cv_folds.csv"]})
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding="utf-8")
    for r in runs:
        print(r["model_id"],r["status"],r.get("holdings"),r.get("objective"))
    if any(r["status"]!="optimal" for r in runs):
        raise SystemExit("See runs.csv for non-optimal scenarios")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=HERE/"results")
    run(parser.parse_args().output)
