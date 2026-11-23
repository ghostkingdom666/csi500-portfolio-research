"""任务3 独立复核：用不同代码路径重算净值、指标与第2人目标估计。

不调用 run.py 的中间结果，只读冻结数据、task02 交付与 task03 输出文件；
不一致即失败，写 results/verification.txt。
"""

from __future__ import annotations

import sys
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


def main() -> int:
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
    # 第2人权重文件与 README 声明一致
    add("task02:weights_sha256_matches_readme",
        sha256(TASK02 / "results" / "weights.csv")
        == "fbe0d3e385cc1dc993c28a82610bd7ab8ae7d2df068591f545f2ab39a8a75bb8",
        sha256(TASK02 / "results" / "weights.csv")[:16])

    # 2) 独立重算每个模型的买入持有净值（逐资产市值循环，不用 pandas 复利）
    nav = pd.read_csv(OUT / "nav.csv", index_col=0)
    weights = {}
    for row in read_csv_rows(TASK02 / "results" / "weights.csv"):
        if row["invested_weight"].strip() in ("", "None"):
            continue  # 失败/不可行模型没有可投资权重，绩效行由 metrics.csv 保留
        weights.setdefault(row["model_id"], {})[row["asset_id"]] = float(row["invested_weight"])
    for row in read_csv_rows(OUT / "weights_benchmarks.csv"):
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
    metrics = pd.read_csv(OUT / "metrics.csv").set_index("model_id")
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
    daily = pd.read_csv(OUT / "daily_returns.csv")
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
    comp = pd.read_csv(OUT / "comparison.csv").set_index("model_id")

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
    (OUT / "verification.txt").write_text("\n".join(summary + lines) + "\n",
                                          encoding="utf-8")
    print("\n".join(summary))
    for line in lines:
        if line.startswith("[FAIL]"):
            print(line)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
