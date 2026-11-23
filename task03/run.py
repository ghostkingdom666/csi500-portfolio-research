"""任务3 公共评价与四类基准：一次跑通冻结数据、任务2权重与比较表。

用法（仓库根目录）：
    python task03/run.py --output task03/results

输入：
    data_delivery_01/data/{manifest,periods,calendar,model_assets,model_returns_wide,
                           missing_mask,index_daily}
    task02/results/{weights.csv,runs.csv,manifest.json}

输出：
    metrics.csv, comparison.csv, nav.csv, daily_returns.csv,
    weights_benchmarks.csv, run_manifest.json, checks.txt
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from benchmarks import (cvar_portfolio, equal_weight, mean_variance,
                        var_heuristic)
from evalcore import (buy_and_hold_nav, performance_metrics, validate_weights)
from frozen import (check_frozen, check_task02, load_frozen, load_task02,
                    read_json, sha256, write_checks, write_csv)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data_delivery_01" / "data"
TASK02 = ROOT / "task02"

ROLES = {
    "l1_eta_0.01": "main", "equal": "main", "mv": "main",
    "cvar_a95": "main", "var_a95_s0": "main",
    "mv_continuous": "extension", "mv_cardinality": "extension",
    "csi500_index_price": "reference",
}

METRIC_FIELDS = [
    "model_id", "role", "source", "status", "solver", "n_assets", "n_days",
    "cumulative_return", "annualized_return", "annualized_vol", "sharpe",
    "max_drawdown", "final_nav", "max_weight", "daily_mean", "daily_sd",
    "seconds", "data_version", "entry_date", "test_end", "note",
]

COMPARISON_FIELDS = [
    "model_id", "role", "source", "status", "n_assets", "n_days",
    "cumulative_return", "annualized_return", "annualized_vol", "sharpe",
    "max_drawdown", "final_nav", "max_weight",
    "T", "N", "eta", "lambda_value", "sigma", "alpha", "loss_max", "budget",
    "seed", "gamma", "cap", "K", "objective", "solver", "seconds",
    "data_version", "estimation_end", "entry_date", "test_end", "note",
]


def role_of(model_id: str) -> str:
    return ROLES.get(model_id, "sensitivity")


def nav_from_weights(weights, returns: pd.DataFrame, entry_date):
    u = validate_weights(pd.Series(weights), list(returns.columns))
    nav = buy_and_hold_nav(u, returns, entry_date)
    return u, nav


def main(output: Path) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    protocol = read_json(HERE / "protocol.json")
    frozen = load_frozen(DATA)
    checks = check_frozen(frozen, DATA)
    t02 = load_task02(TASK02)
    checks += check_task02(frozen, t02)
    failed = [c for c in checks if not c["passed"]]
    write_checks(output / "checks.txt", checks)
    if failed:
        for c in failed:
            print(f"FAIL {c['check']} :: {c['detail']}", file=sys.stderr)
        raise SystemExit(f"{len(failed)} 项契约检查未通过，停止评价")

    periods = frozen["periods"]
    ids = frozen["ids"]
    returns = frozen["returns"]
    entry = periods["proposed_entry_date"]
    first = periods["proposed_first_return_date"]
    test_end = periods["test_end"]
    est_start = periods["estimation_start"]
    est_end = periods["estimation_end"]
    version = periods["data_version"]

    est = returns.loc[(returns.index >= est_start) & (returns.index <= est_end)]
    window = returns.loc[(returns.index >= first) & (returns.index <= test_end)]
    if window.empty or window.index[0] != first or window.index[-1] != test_end:
        raise SystemExit(f"检验窗口端点应为 {first}..{test_end}，"
                         f"实际 {window.index[0] if not window.empty else '空'}.."
                         f"{window.index[-1] if not window.empty else '空'}")

    metrics_rows, comparison_rows, navs, daily_rows = [], [], {}, []

    def record(model_id, source, status, solver, nav, u, extra=None,
               seconds=None, note=""):
        row = {
            "model_id": model_id, "role": role_of(model_id), "source": source,
            "status": status, "solver": solver,
            "n_assets": int((u > 1e-8).sum()) if u is not None else "",
            "n_days": "", "cumulative_return": "", "annualized_return": "",
            "annualized_vol": "", "sharpe": "", "max_drawdown": "",
            "final_nav": "", "max_weight": "" if u is None else float(np.max(u)),
            "daily_mean": "", "daily_sd": "",
            "seconds": seconds, "data_version": version,
            "entry_date": entry, "test_end": test_end, "note": note,
        }
        if nav is not None:
            m = performance_metrics(nav, rf=protocol["evaluation"]["rf"],
                                    D=protocol["evaluation"]["D"])
            row.update({k: m[k] for k in
                        ("n_days", "cumulative_return", "annualized_return",
                         "annualized_vol", "sharpe", "max_drawdown", "final_nav",
                         "daily_mean", "daily_sd")})
            if row["sharpe"] is None:
                row["sharpe"] = ""
            navs[model_id] = nav
            diff = nav.pct_change().dropna()
            for date, r in diff.items():
                daily_rows.append({"date": date, "model_id": model_id,
                                   "gross_return": float(r)})
        metrics_rows.append(row)
        comp = {k: row.get(k) for k in COMPARISON_FIELDS}
        if extra:
            comp.update({k: v for k, v in extra.items() if k in COMPARISON_FIELDS})
        comparison_rows.append(comp)
        return row

    # ---------- 任务2 交付的 7 个模型 ----------
    runs = t02["runs"].set_index("model_id")
    for model_id in runs.index:
        w = t02["weights"][model_id]
        status = t02["statuses"][model_id]
        run = runs.loc[model_id]
        extra = {k: (None if pd.isna(run.get(k)) else run.get(k)) for k in
                 ("T", "N", "eta", "lambda_value", "sigma", "gamma", "cap", "K",
                  "objective", "solver", "seconds")}
        if status != "optimal":
            record(model_id, "task02", status, run.get("solver"), None, None,
                   extra=extra, note=str(run.get("reason", "")))
            continue
        u, nav = nav_from_weights(w, window[w.index], entry)
        record(model_id, "task02", status, run.get("solver"), nav, u,
               extra=extra, seconds=extra.get("seconds"), note="")

    # ---------- 任务3 四类基准（估计期定权，预登记参数） ----------
    mu = est.mean()
    bench = {}
    cfg = protocol["benchmarks"]
    plans = [("equal", lambda: equal_weight(ids))]
    plans.append(("mv", lambda: mean_variance(
        est.to_numpy(), cfg["mv"]["sigma_max_daily"], ids, mu=mu.to_numpy())))
    for spec in cfg["cvar"]:
        model_id = f"cvar_a{int(round(spec['alpha'] * 100))}"
        plans.append((model_id, lambda s=spec: cvar_portfolio(
            est.to_numpy(), s["alpha"], s["loss_max"], ids, mu=mu.to_numpy())))
    for seed in cfg["var_heuristic"]["seeds"]:
        model_id = f"var_a{int(round(cfg['var_heuristic']['alpha'] * 100))}_s{seed}"
        plans.append((model_id, lambda s=seed: var_heuristic(
            est.to_numpy(), cfg["var_heuristic"]["alpha"],
            cfg["var_heuristic"]["loss_max"], ids,
            cfg["var_heuristic"]["budget"], s, mu=mu.to_numpy())))

    for model_id, solve in plans:
        w, info = solve()
        bench[model_id] = (w, info)
        extra = {
            "T": len(est), "N": len(ids),
            "sigma": (cfg["mv"]["sigma_max_daily"] if model_id == "mv" else None),
            "alpha": (cfg["var_heuristic"]["alpha"]
                      if model_id.startswith("var") else
                      next((s["alpha"] for s in cfg["cvar"]
                            if model_id == f"cvar_a{int(round(s['alpha'] * 100))}"), None)),
            "loss_max": (cfg["var_heuristic"]["loss_max"]
                         if model_id.startswith("var") else
                         next((s["loss_max"] for s in cfg["cvar"]
                               if model_id == f"cvar_a{int(round(s['alpha'] * 100))}"), None)),
            "budget": cfg["var_heuristic"]["budget"] if model_id.startswith("var") else None,
            "seed": int(model_id.rsplit("s", 1)[1]) if model_id.startswith("var") else None,
            "objective": info.get("objective"), "solver": info.get("solver"),
            "seconds": info.get("seconds"),
        }
        if w is None:
            record(model_id, "task03", info.get("status", "fail"),
                   info.get("solver", ""), None, None, extra=extra,
                   note=info.get("reason", ""))
            continue
        try:
            u, nav = nav_from_weights(w, window[ids], entry)
        except ValueError as exc:
            record(model_id, "task03", "fail", info.get("solver", ""), None, None,
                   extra=extra, note=str(exc))
            continue
        note = ""
        if model_id.startswith("cvar"):
            note = (f"zeta={info.get('zeta', float('nan')):.6g}; "
                    f"经验 VaR={info.get('empirical_var', float('nan')):.6g}; "
                    f"经验 CVaR={info.get('empirical_cvar', float('nan')):.6g}")
        if model_id.startswith("var"):
            note = (f"候选 {info.get('budget')}，可行 {info.get('feasible_count')}，"
                    f"经验 VaR={info.get('empirical_var', float('nan')):.6g}")
        record(model_id, "task03", info.get("status"), info.get("solver"),
               nav, u, extra=extra, seconds=info.get("seconds"), note=note)

    # ---------- 中证500 价格指数作市场参照（不是策略） ----------
    idx = frozen["index"].loc[first:test_end, "daily_simple_return"].astype(float)
    if idx.isna().any():
        raise SystemExit("指数参照曲线存在缺失收益")
    nav = pd.Series(np.concatenate([[1.0], np.cumprod(1.0 + idx.to_numpy())]),
                    index=[entry, *idx.index], name="nav")
    record("csi500_index_price", "reference", "reference", "n/a", nav,
           None, note="中证500价格指数，仅作市场参照，不参与策略排名")

    # ---------- 写盘 ----------
    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metrics_rows)[METRIC_FIELDS]
    comparison = pd.DataFrame(comparison_rows)[COMPARISON_FIELDS]
    metrics.to_csv(output / "metrics.csv", index=False, encoding="utf-8")
    comparison.to_csv(output / "comparison.csv", index=False, encoding="utf-8")

    all_models = [r["model_id"] for r in metrics_rows if r["model_id"] in navs]
    nav_frame = pd.DataFrame({m: navs[m] for m in all_models})
    nav_frame.index.name = "date"
    nav_frame.to_csv(output / "nav.csv", encoding="utf-8")
    write_csv(output / "daily_returns.csv", daily_rows,
              ["date", "model_id", "gross_return"])

    name_map = {a["asset_id"]: a["name"] for a in frozen["assets"]}
    weight_rows = []
    for model_id, (w, info) in bench.items():
        if w is None:
            continue
        for asset_id in ids:
            weight_rows.append({
                "model_id": model_id, "asset_id": asset_id,
                "name": name_map.get(asset_id, ""),
                "raw_coefficient": "", "invested_weight": float(w[asset_id]),
                "status": info.get("status"),
            })
    write_csv(output / "weights_benchmarks.csv", weight_rows,
              ["model_id", "asset_id", "name", "raw_coefficient",
               "invested_weight", "status"])

    manifest = {
        "task": "task03",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_version": version,
        "entry_date": entry,
        "first_return_date": first,
        "test_end": test_end,
        "n_test_return_days": int(len(window)),
        "protocol": protocol,
        "inputs": {name: sha256(DATA / name) for name in
                   ("periods.json", "calendar.csv", "model_assets.csv",
                    "model_returns_wide.csv", "missing_mask.csv", "index_daily.csv")},
        "task02_weights_sha256": t02["weights_sha256"],
        "code_sha256": {name: sha256(HERE / name) for name in
                        ("evalcore.py", "benchmarks.py", "frozen.py", "run.py",
                         "protocol.json")},
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
        },
        "outputs": {name: sha256(output / name) for name in
                    ("metrics.csv", "comparison.csv", "nav.csv",
                     "daily_returns.csv", "weights_benchmarks.csv", "checks.txt")},
        "checks_passed": len(checks),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    show = metrics[["model_id", "status", "n_assets", "n_days",
                    "cumulative_return", "annualized_return", "annualized_vol",
                    "sharpe", "max_drawdown"]]
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", lambda x: f"{x:.4f}"):
        print(show.to_string(index=False))
    print(f"\n契约检查 {len(checks)} 项全部通过；输出目录 {output}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "results")
    main(parser.parse_args().output)
