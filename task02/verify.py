"""Recompute exported objectives and constraints without calling either optimizer."""
import csv
from pathlib import Path
import sys

import mpmath as mp
import numpy as np
from run import HERE, inputs, read_json, digest


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
    index={r["asset_id"]:i for i,r in enumerate(assets)}
    def read(name):
        with (out/name).open(encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    rows=read("weights.csv")
    runs=read("runs.csv")
    expected={f"l1_eta_{eta}" for eta in p["eta_grid"]}|{"mv_continuous","mv_cardinality"}
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
            r=A@w-rc
            lam=float(run["lambda_value"])
            eta = float(run["eta"])
            assert name == f"l1_eta_{run['eta']}" and eta in p["eta_grid"]
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
