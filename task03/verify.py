"""任务3 独立复核：用不同代码路径重算净值、指标与第2人目标估计。

不调用 run.py 的中间结果，只读冻结数据、task02 交付与 task03 输出文件；
不一致即失败，写 results/verification.txt。
"""

from __future__ import annotations

import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd
from mpmath import mp

from frozen import load_frozen, read_json, read_csv_rows, sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data_delivery_01" / "data"
TASK02 = ROOT / "task02"
OUT = HERE / "results"

TOL_NAV = 1e-10
TOL_METRIC = 1e-10
TOL_TARGET = 1e-11


def verify_simple_baselines(output: Path, est: pd.DataFrame) -> list[tuple]:
    """Reconstruct fixed signals and GMV KKT without production solver imports."""
    output = Path(output)
    manifest = read_json(output / "run_manifest.json")
    metrics = pd.read_csv(output / "metrics.csv").set_index("model_id")
    comparison = pd.read_csv(output / "comparison.csv").set_index("model_id")
    weights = pd.read_csv(output / "weights_benchmarks.csv")
    infos = manifest["benchmark_runs"]
    checks = []

    def add(name, ok, detail=""):
        checks.append((name, bool(ok), str(detail)))

    models = ["gmv", "inverse_volatility", "momentum_top20", "low_volatility_top20"]
    expected_main = {"l1_cv", "equal", "mv", "cvar_a95", "var_a95_s0", *models}
    add("roles:nine_main_models", set(metrics.index[metrics.role == "main"]) == expected_main
        and set(comparison.index[comparison.role == "main"]) == expected_main)
    labels = list(est.columns)
    A = est.to_numpy()
    T, n = A.shape
    estimation = {"start": str(est.index[0]), "end": str(est.index[-1]),
                  "n_days": T, "n_assets": n, "ddof": 1}
    # Independent centered cross-product rather than the production np.cov call.
    centered = A - A.mean(axis=0)
    covariance = centered.T @ centered / (T - 1)
    sd_full = np.array([math.sqrt(math.fsum(float(x)**2 for x in centered[:, j]) / (T-1))
                        for j in range(n)])
    for model in models:
        present = model in infos and model in metrics.index and model in comparison.index
        add(f"simple:{model}:present", present)
        if not present:
            continue
        info = infos[model]
        add(f"simple:{model}:estimation_only", info.get("estimation") == estimation)
        add(f"simple:{model}:status_and_role", info.get("status") == metrics.loc[model, "status"]
            == comparison.loc[model, "status"] and metrics.loc[model, "role"] == "main"
            and comparison.loc[model, "role"] == "main")
        got = weights[weights.model_id == model].set_index("asset_id")
        if info.get("status") != "optimal":
            add(f"simple:{model}:failure_has_no_weights", got.empty)
            continue
        labels_ok = len(got) == n and got.index.is_unique and set(got.index) == set(labels)
        add(f"simple:{model}:all_asset_rows", labels_ok)
        if not labels_ok:
            continue
        u = got.loc[labels, "invested_weight"].to_numpy()
        add(f"simple:{model}:simplex", np.isfinite(u).all() and u.min() >= -1e-12
            and abs(u.sum() - 1) <= 1e-8 and set(got.status) == {"optimal"})
        variance = float(u @ covariance @ u)
        if model == "gmv":
            add(f"simple:{model}:parameters", info.get("parameters") == {
                "covariance_ddof": 1, "return_target": None})
            # For a convex PSD quadratic on the simplex these KKT conditions
            # certify global optimality, including zero-weight inequalities.
            scale = float(np.max(np.diag(covariance))) or 1.
            gradient = 2 * (covariance / scale) @ u
            multiplier = float(u @ gradient)
            active = u > 1e-8
            stationary = float(np.max(np.abs(gradient[active] - multiplier))) if active.any() else np.inf
            dual = float(np.max(np.maximum(multiplier - gradient[~active], 0))) if (~active).any() else 0.
            complementarity = float(np.max(np.abs(u * (gradient - multiplier))))
            residual = max(stationary, dual, complementarity)
            add(f"simple:{model}:independent_kkt", residual <= 1e-6,
                f"scaled_stationarity={stationary:.3e} dual={dual:.3e} complementarity={complementarity:.3e}")
            add(f"simple:{model}:variance_objective", abs(info["objective"] - variance) <= 1e-12)
            continue
        if model == "inverse_volatility":
            add(f"simple:{model}:parameters", info.get("parameters") == {
                "volatility_ddof": 1, "zero_volatility": "fail"})
            valid = np.isfinite(sd_full).all() and (sd_full > 0).all()
            add(f"simple:{model}:positive_finite_volatility", valid)
            if not valid:
                continue
            inv = 1 / sd_full
            expected_weights = inv / inv.sum()
            scores = sd_full
            add(f"simple:{model}:variance_objective", abs(info["objective"] - variance) <= 1e-12)
        else:
            if model == "momentum_top20":
                params = {"k": 20, "lookback": 126, "skip": 21}
                scores = np.array([math.prod(1 + float(x) for x in A[-126:-21, j]) - 1 for j in range(n)])
                ranking = sorted(range(n), key=lambda j: (-scores[j], labels[j]))
                signal = {"start": str(est.index[-126]), "end": str(est.index[-22]),
                          "n_days": 105, "excluded_recent_days": 21}
            else:
                params = {"k": 20, "lookback": 60}
                tail = A[-60:]
                scores = np.array([math.sqrt(math.fsum((float(x) - math.fsum(tail[:, j])/60)**2
                                                      for x in tail[:, j]) / 59) for j in range(n)])
                ranking = sorted(range(n), key=lambda j: (scores[j], labels[j]))
                signal = {"start": str(est.index[-60]), "end": str(est.index[-1]), "n_days": 60, "ddof": 1}
            add(f"simple:{model}:parameters", info.get("parameters") == params)
            add(f"simple:{model}:signal_window", info.get("signal") == signal)
            selected = [labels[j] for j in ranking[:20]]
            expected_weights = np.array([.05 if a in selected else 0. for a in labels])
            add(f"simple:{model}:selection_audit", info.get("selected_assets") == selected
                and info.get("ranking") == [labels[j] for j in ranking]
                and info.get("diagnostics", {}).get("tie_break") == "asset_id_lexicographic_ascending")
            add(f"simple:{model}:signal_objective", abs(info["objective"] - float(expected_weights @ scores)) <= 1e-12)
        got_scores = info.get("scores", {})
        add(f"simple:{model}:all_scores", set(got_scores) == set(labels)
            and all(abs(got_scores[a] - float(s)) <= 1e-12 for a, s in zip(labels, scores)))
        add(f"simple:{model}:independent_weights", np.allclose(u, expected_weights, atol=1e-12, rtol=0))
    return checks


def verify_task02_provenance(output: Path, task02_dir: Path) -> list[tuple]:
    """Bind recorded input hashes and model/status rows to the current delivery."""
    output, source = Path(output), Path(task02_dir) / "results"
    recorded = read_json(output / "run_manifest.json")
    current = read_json(source / "manifest.json")
    expected = current.get("outputs", {})
    checks = []

    def add(name, ok, detail=""):
        checks.append((name, bool(ok), str(detail)))

    add("task02:outputs_manifest_present", bool(expected) and {"weights.csv", "runs.csv"} <= set(expected))
    add("task02:recorded_outputs_match_current_manifest", bool(expected)
        and recorded.get("task02_outputs") == expected)
    actual_weights = sha256(source / "weights.csv")
    add("task02:recorded_weights_sha256_matches_current_source",
        recorded.get("task02_weights_sha256") == actual_weights == expected.get("weights.csv"))
    for name, digest in expected.items():
        path = source / name
        add(f"task02:output_sha256:{name}", path.is_file() and sha256(path) == digest)

    runs = read_csv_rows(source / "runs.csv")
    weights = read_csv_rows(source / "weights.csv")
    run_models = {row["model_id"] for row in runs}
    weight_models = {row["model_id"] for row in weights}
    add("task02:l1_cv_required", "l1_cv" in run_models and "l1_cv" in weight_models)
    add("task02:model_sets_match", run_models == weight_models)
    add("task02:run_models_unique", len(runs) == len(run_models))
    run_statuses = {row["model_id"]: row.get("status", "").strip() for row in runs}
    for model in sorted(run_models | weight_models):
        statuses = {row.get("status", "").strip() for row in weights if row["model_id"] == model}
        want = run_statuses.get(model, "")
        add(f"task02:{model}:statuses_match", bool(want) and statuses == {want})
    for name in ["metrics.csv", "comparison.csv"]:
        rows = read_csv_rows(output / name)
        rows = [row for row in rows if row["source"] == "task02"]
        models = {row["model_id"] for row in rows}
        add(f"task02:{name}:models_match_source", models == run_models and len(rows) == len(models))
        add(f"task02:{name}:l1_cv_main", any(row["model_id"] == "l1_cv" and row["role"] == "main" for row in rows))
        add(f"task02:{name}:statuses_match_source", all(
            row.get("status", "").strip() == run_statuses.get(row["model_id"])
            for row in rows))
    return checks


def verify_benchmark_artifacts(output: Path, est: pd.DataFrame) -> list[tuple]:
    """Independently check exported risks, heuristic search and failure rows."""
    output = Path(output)
    checks = []

    def add(name, ok, detail=""):
        checks.append((name, bool(ok), str(detail)))

    manifest = read_json(output / "run_manifest.json")
    for name, expected in manifest["outputs"].items():
        path = output / name
        add(f"output_sha256:{name}", path.is_file() and sha256(path) == expected)
    add("risk_statistics.csv:manifest_present", "risk_statistics.csv" in manifest["outputs"])
    for name, expected in manifest["code_sha256"].items():
        path = HERE / name
        add(f"code_sha256:{name}", path.is_file() and sha256(path) == expected)

    risks = pd.read_csv(output / "risk_statistics.csv")
    add("risk_statistics.csv:rows", len(risks) == 2 and set(risks["alpha"]) == {.95, .99})
    losses = sorted(-float(row.mean()) for row in est.to_numpy())
    for row in risks.to_dict("records"):
        alpha = float(row["alpha"])
        var = losses[int(np.ceil(alpha * len(losses))) - 1]
        cvar = var + sum(max(x - var, 0) for x in losses) / (len(losses) * (1 - alpha))
        ok = (row["portfolio_id"] == "equal" and row["n_days"] == len(est)
              and row["n_assets"] == est.shape[1]
              and str(row["estimation_start"]) == str(est.index[0])
              and str(row["estimation_end"]) == str(est.index[-1])
              and row["unit"] == "decimal_daily_loss"
              and abs(row["empirical_var"] - var) < 1e-12
              and abs(row["empirical_cvar"] - cvar) < 1e-12
              and abs(row["daily_sd"] - np.std(losses, ddof=1)) < 1e-12)
        add(f"risk_statistics.csv:alpha={alpha}:independent_recompute", ok)

    metrics = pd.read_csv(output / "metrics.csv").set_index("model_id")
    comp = pd.read_csv(output / "comparison.csv").set_index("model_id")
    nav = pd.read_csv(output / "nav.csv", index_col=0)
    weights = pd.read_csv(output / "weights_benchmarks.csv")
    successful = {"optimal", "feasible_heuristic", "reference"}
    perf = ["n_assets", "n_days", "cumulative_return", "annualized_return",
            "annualized_vol", "sharpe", "max_drawdown", "final_nav", "max_weight"]
    for model_id, row in metrics.iterrows():
        if row["status"] in successful:
            add(f"status:{model_id}:has_nav", model_id in nav.columns)
        else:
            add(f"status:{model_id}:empty_failure", row[perf].isna().all()
                and comp.loc[model_id, perf].isna().all() and model_id not in nav.columns
                and model_id not in set(weights["model_id"]))
    add("roles:l1_cv_main", "l1_cv" in metrics.index and "l1_cv" in comp.index
        and metrics.loc["l1_cv", "role"] == "main" and comp.loc["l1_cv", "role"] == "main")
    if "l1_eta_0.01" in metrics.index:
        add("roles:l1_eta_0.01_sensitivity", metrics.loc["l1_eta_0.01", "role"] == "sensitivity")

    checks.extend(verify_simple_baselines(output, est))

    cfg = manifest["protocol"]["benchmarks"]["var_heuristic"]
    infos = manifest["benchmark_runs"]
    assets = list(est.columns)
    A, mu = est.to_numpy(), est.to_numpy().mean(axis=0)
    n, T = len(assets), len(est)
    var_models = {f"var_a{round(100 * cfg['alpha'])}_s{seed}" for seed in cfg["seeds"]}
    add("var:configured_runs_present", var_models <= set(infos))
    for model_id in sorted(var_models & set(infos)):
        info = infos[model_id]
        warm = info["warm_starts"]
        expected_ids = [m for m in cfg["warm_start_ids"]
                        if m in infos and infos[m]["status"] == "optimal"]
        add(f"var:{model_id}:warm_start_ids", [r["candidate_id"] for r in warm] == expected_ids)
        vectors = []
        for i, row in enumerate(warm, start=n + 1):
            u = np.array([row["weights"][a] for a in assets])
            vectors.append(u)
            original = weights[weights["model_id"] == row["candidate_id"]].set_index("asset_id")
            ok = (set(row["weights"]) == set(assets) and len(original) == n
                  and np.allclose(original.loc[assets, "invested_weight"], u, atol=1e-12, rtol=0)
                  and row["candidate_index"] == i and abs(u.sum() - 1) < 1e-8
                  and np.min(u) >= -1e-12)
            add(f"var:{model_id}:warm:{row['candidate_id']}:weights", ok)
        budget = cfg["budget"]
        random_budget = budget - 1 - n - len(vectors)
        seed = int(model_id.rsplit("s", 1)[1])
        add(f"var:{model_id}:budget_seed", info["budget"] == budget
            and info["total_candidates"] == budget and info["random_budget"] == random_budget
            and info["seed"] == seed and info["random_distribution"] == "Dirichlet(1)"
            and info["risk_tolerance"] == 1e-10)
        rng = np.random.default_rng(seed)
        U = np.array([np.full(n, 1 / n), *np.eye(n), *vectors,
                      *rng.dirichlet(np.ones(n), size=random_budget)])
        candidate_ids = ["equal", *[f"single:{a}" for a in assets], *expected_ids,
                         *[f"random:{i}" for i in range(random_budget)]]
        means = U @ mu
        sorted_losses = np.sort(-(A @ U.T), axis=0)
        risk = sorted_losses[int(np.ceil(cfg["alpha"] * T)) - 1]
        feasible = risk <= cfg["loss_max"] + 1e-10
        add(f"var:{model_id}:feasible_count", info["feasible_count"] == int(feasible.sum()))
        for row in warm:
            i = row["candidate_index"]
            add(f"var:{model_id}:warm:{row['candidate_id']}:statistics",
                abs(row["objective"] - means[i]) < 1e-12
                and abs(row["empirical_var"] - risk[i]) < 1e-12
                and row["feasible"] == bool(feasible[i]))
        if not feasible.any():
            add(f"var:{model_id}:honest_no_candidate", info["status"] == "no_feasible_candidate"
                and metrics.loc[model_id, "status"] == "no_feasible_candidate")
            continue
        pick = int(np.argmax(np.where(feasible, means, -np.inf)))
        got = weights[weights["model_id"] == model_id].set_index("asset_id")
        ok = (info["status"] == "feasible_heuristic"
              and metrics.loc[model_id, "status"] == "feasible_heuristic"
              and comp.loc[model_id, "status"] == "feasible_heuristic"
              and info["selected_candidate_index"] == pick
              and info["selected_candidate_id"] == candidate_ids[pick]
              and len(got) == n and set(got["status"]) == {"feasible_heuristic"}
              and np.allclose(got.loc[assets, "invested_weight"], U[pick], rtol=0, atol=1e-12)
              and abs(info["objective"] - means[pick]) < 1e-12
              and abs(info["empirical_var"] - risk[pick]) < 1e-12)
        add(f"var:{model_id}:independent_search_replay", ok)
        feasible_warm = [r["objective"] for r in warm if r["feasible"]]
        add(f"var:{model_id}:not_worse_than_warm_start", not feasible_warm
            or info["objective"] >= max(feasible_warm) - 1e-12)
    return checks


def main(output: Path = OUT) -> int:
    output = Path(output)
    checks = []

    def add(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    frozen = load_frozen(DATA)
    periods = frozen["periods"]
    ids = frozen["ids"]
    returns = frozen["returns"]
    entry = periods["proposed_entry_date"]
    first = periods["proposed_first_return_date"]
    test_end = periods["test_end"]
    window = returns.loc[(returns.index >= first) & (returns.index <= test_end)]
    dates = list(window.index)

    # 1) 第2人目标估计与 λ_max：mpmath 60 位独立重算
    est = returns.loc[(returns.index >= periods["estimation_start"])
                      & (returns.index <= periods["estimation_end"])]
    A = est.to_numpy()
    T, N = A.shape
    mu = A.mean(axis=0)
    Sigma = np.cov(A, rowvar=False, ddof=1)
    theta_s = float(mu @ np.linalg.solve(Sigma, mu))
    mp.dps = 60
    a, b = mp.mpf(N) / 2, mp.mpf(T - N) / 2
    x = mp.mpf(theta_s) / (1 + mp.mpf(theta_s))
    Bx = mp.betainc(a, b, 0, x, regularized=False)
    theta_a = (((T - N - 2) * mp.mpf(theta_s) - N) / T
               + 2 * mp.power(mp.mpf(theta_s), a)
               * mp.power(1 + mp.mpf(theta_s), -(T - 2) / 2) / (T * Bx))
    sigma = 0.04
    r_c = mp.mpf(sigma) * (1 + theta_a) / mp.sqrt(theta_a)
    y = np.full(T, float(r_c))
    lambda_max = max(0.0, 2.0 * float((A.T @ y).max()))
    t2m = read_json(TASK02 / "results" / "manifest.json")
    target = t2m["target"]
    add("task02:theta_s_reproduced",
        abs(theta_s - target["theta_s"]) < TOL_TARGET,
        f"{theta_s:.15f} vs {target['theta_s']:.15f}")
    add("task02:theta_a_reproduced",
        abs(float(theta_a) - target["theta_a"]) < TOL_TARGET,
        f"{mp.nstr(theta_a, 15)} vs {target['theta_a']:.15f}")
    add("task02:r_c_reproduced",
        abs(float(r_c) - target["r_c"]) < TOL_TARGET,
        f"{mp.nstr(r_c, 15)} vs {target['r_c']:.15f}")
    add("task02:lambda_max_reproduced",
        abs(lambda_max - t2m["lambda_max"]) < 1e-11,
        f"{lambda_max:.15f} vs {t2m['lambda_max']:.15f}")
    checks.extend(verify_task02_provenance(output, TASK02))
    checks.extend(verify_benchmark_artifacts(output, est))

    # 2) 独立重算每个模型的买入持有净值（逐资产市值循环，不用 pandas 复利）
    nav = pd.read_csv(output / "nav.csv", index_col=0)
    weights = {}
    for row in read_csv_rows(TASK02 / "results" / "weights.csv"):
        if row["invested_weight"].strip() in ("", "None"):
            continue  # 失败/不可行模型没有可投资权重，绩效行由 metrics.csv 保留
        weights.setdefault(row["model_id"], {})[row["asset_id"]] = float(row["invested_weight"])
    for row in read_csv_rows(output / "weights_benchmarks.csv"):
        weights.setdefault(row["model_id"], {})[row["asset_id"]] = float(row["invested_weight"])

    for model_id, w in weights.items():
        if model_id not in nav.columns:
            add(f"nav:{model_id}:present", False, "缺少净值列")
            continue
        if abs(sum(w.values()) - 1.0) > 1e-8:
            add(f"nav:{model_id}:budget", False, f"权重和={sum(w.values())}")
            continue
        value = dict(w)
        base = {asset: w[asset] for asset in w}
        series = [1.0]
        for date in dates:
            for asset in w:
                base[asset] *= 1.0 + float(returns.at[date, asset])
            series.append(sum(base.values()))
        got = nav[model_id].to_numpy(dtype=float)
        ok = got.shape == (len(series),) and np.allclose(got, series, rtol=0, atol=TOL_NAV)
        add(f"nav:{model_id}:independent_recompute", ok,
            "" if ok else f"max|diff|={np.max(np.abs(got - np.array(series))):.3e}")

    # 3) 独立重算指标
    metrics = pd.read_csv(output / "metrics.csv").set_index("model_id")
    for model_id in nav.columns:
        v = nav[model_id].to_numpy(dtype=float)
        daily = v[1:] / v[:-1] - 1
        n = len(v) - 1
        total = v[-1] / v[0] - 1
        annual = (v[-1] / v[0]) ** (252 / n) - 1
        sd = daily.std(ddof=1)
        vol = np.sqrt(252) * sd
        sharpe = np.sqrt(252) * daily.mean() / sd if sd > 0 else np.nan
        peak = np.maximum.accumulate(v)
        mdd = (v / peak - 1).min()
        row = metrics.loc[model_id]
        checks_local = [
            abs(total - row["cumulative_return"]) < TOL_METRIC,
            abs(annual - row["annualized_return"]) < TOL_METRIC,
            abs(vol - row["annualized_vol"]) < TOL_METRIC,
            (abs(sharpe - row["sharpe"]) < TOL_METRIC) if np.isfinite(sharpe) else pd.isna(row["sharpe"]),
            abs(mdd - row["max_drawdown"]) < TOL_METRIC,
            int(n) == int(row["n_days"]),
        ]
        add(f"metrics:{model_id}:independent_recompute", all(checks_local),
            "" if all(checks_local) else str(checks_local))

    # 4) 逐日收益与净值一致
    daily = pd.read_csv(output / "daily_returns.csv")
    for model_id in nav.columns:
        want = nav[model_id].pct_change().dropna().to_numpy()
        got = daily[daily["model_id"] == model_id]["gross_return"].to_numpy()
        add(f"daily:{model_id}:matches_nav", np.allclose(got, want, atol=1e-12),
            "" if np.allclose(got, want, atol=1e-12) else "不一致")

    # 5) 建仓日可交易性：各模型持仓资产在 2026-07-01 无缺失标记
    mask = frozen["missing"]
    if entry not in mask.index:
        add("entry:tradeability", False, "缺少建仓日缺失标记")
    else:
        bad = []
        for model_id, w in weights.items():
            for asset in w:
                if float(mask.at[entry, asset]) != 0.0:
                    bad.append(f"{model_id}:{asset}")
        add("entry:all_held_assets_tradeable", not bad, ",".join(bad[:5]))

    # 6) 比较表与指标表一致
    comp = pd.read_csv(output / "comparison.csv").set_index("model_id")

    def same_value(a, b):
        if pd.isna(a) and pd.isna(b):
            return True
        return (not pd.isna(a)) and (not pd.isna(b)) and abs(float(a) - float(b)) < 1e-12

    same = all(same_value(comp.loc[m, "cumulative_return"], metrics.loc[m, "cumulative_return"])
               for m in metrics.index)
    add("comparison:consistent_with_metrics", same, "")

    lines = [f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" :: {detail}" if detail else "")
             for name, ok, detail in checks]
    failed = [c for c in checks if not c[1]]
    summary = [f"checks={len(checks)} failed={len(failed)}",
               f"data_version={periods['data_version']}",
               f"entry={entry} first_return={first} test_end={test_end}",
               f"theta_s={theta_s:.15f} theta_a={mp.nstr(theta_a, 15)} r_c={mp.nstr(r_c, 15)}",
               f"lambda_max={lambda_max:.15f}"]
    (output / "verification.txt").write_text("\n".join(summary + lines) + "\n",
                                          encoding="utf-8")
    print("\n".join(summary))
    for line in lines:
        if line.startswith("[FAIL]"):
            print(line)
    return 0 if not failed else 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    sys.exit(main(parser.parse_args().output))
