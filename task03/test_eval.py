"""任务3 人工可核对测试：评价核心 + 四类基准的边界分支。

运行（仓库根目录）：python task03/test_eval.py
"""

import numpy as np
import pandas as pd

from benchmarks import (cvar_portfolio, equal_weight, mean_variance,
                        var_candidates, var_heuristic)
from evalcore import (WeightError, buy_and_hold_nav, empirical_var_cvar,
                      performance_metrics, rebalanced_nav,
                      reject_raw_coefficients, validate_weights)


def expect_error(fn, kind=ValueError):
    try:
        fn()
    except kind:
        return True
    raise AssertionError("应当报错但未报错")


def test_buy_and_hold():
    R = pd.DataFrame([[0.1, 0.0], [0.0, 0.1]], index=["d1", "d2"], columns=["A", "B"])
    w = pd.Series([0.5, 0.5], index=["A", "B"])
    nav = buy_and_hold_nav(validate_weights(w, list(R.columns)), R, "d0")
    assert np.allclose(nav.to_numpy(), [1.0, 1.05, 1.10]), nav.to_numpy()
    # 与逐日持仓市值累积的独立实现一致（不是固定权重逐日点乘）
    v, out = np.array([0.5, 0.5]), [1.0]
    for t in range(len(R)):
        v = v * (1.0 + R.iloc[t].to_numpy())
        out.append(float(v.sum()))
    assert np.allclose(nav.to_numpy(), out), (nav.to_numpy(), out)


def test_rebalanced():
    R = pd.DataFrame([[0.1, 0.0], [0.0, 0.1]], index=["d1", "d2"], columns=["A", "B"])
    w = pd.Series([0.5, 0.5], index=["A", "B"])
    nav = rebalanced_nav(validate_weights(w, list(R.columns)), R, "d0")
    assert np.allclose(nav.to_numpy(), [1.0, 1.05, 1.1025]), nav.to_numpy()


def test_single_asset():
    r = np.array([0.01, -0.02, 0.03])
    R = pd.DataFrame({"A": r}, index=["d1", "d2", "d3"])
    nav = buy_and_hold_nav(np.array([1.0]), R, "d0")
    assert np.allclose(nav.to_numpy()[1:], np.cumprod(1 + r))


def test_label_alignment():
    rng = np.random.default_rng(3)
    R = pd.DataFrame(rng.normal(0, 0.01, (60, 5)), columns=list("ABCDE"))
    w = pd.Series(1 / 5, index=list("ABCDE"))
    n1 = buy_and_hold_nav(validate_weights(w, list(R.columns)), R, "d0")
    perm = list("EDCBA")
    n2 = buy_and_hold_nav(validate_weights(w.reindex(perm), perm), R[perm], "d0")
    assert np.allclose(n1.to_numpy(), n2.to_numpy())


def test_weight_validation():
    expect_error(lambda: validate_weights(pd.Series({"A": 1.1}), ["A"]), WeightError)
    expect_error(lambda: validate_weights(pd.Series({"A": 1.0}), ["A", "B"]), WeightError)
    expect_error(lambda: validate_weights(pd.Series({"A": 0.5, "B": 0.5}), ["A"]), WeightError)
    expect_error(lambda: validate_weights(pd.Series({"A": 0.5, "B": 0.4}), ["A", "B"]), WeightError)
    expect_error(lambda: validate_weights(pd.Series({"A": 1.1, "B": -0.1}), ["A", "B"]), WeightError)
    expect_error(lambda: validate_weights(pd.Series({"A": np.nan, "B": 1.0}), ["A", "B"]), WeightError)
    expect_error(lambda: validate_weights(pd.Series([0.5, 0.5], index=["A", "A"]), ["A"]), WeightError)
    expect_error(lambda: reject_raw_coefficients(np.array([0.3, 0.2])), WeightError)
    reject_raw_coefficients(None)


def test_metrics():
    nav = pd.Series([1.0, 1.2, 0.9, 1.1], index=["d0", "d1", "d2", "d3"])
    m = performance_metrics(nav)
    assert abs(m["cumulative_return"] - 0.10) < 1e-12
    assert abs(m["max_drawdown"] - (0.9 / 1.2 - 1)) < 1e-12
    n = 3
    assert abs(m["annualized_return"] - ((1.1 / 1.0) ** (252 / n) - 1)) < 1e-12
    daily = np.array([0.2, -0.25, 1.1 / 0.9 - 1])
    assert abs(m["annualized_vol"] - np.sqrt(252) * daily.std(ddof=1)) < 1e-12
    assert abs(m["daily_sd"] - daily.std(ddof=1)) < 1e-12


def test_zero_vol_sharpe():
    nav = pd.Series([1.0, 1.0, 1.0])
    m = performance_metrics(nav)
    assert m["annualized_vol"] == 0.0
    assert m["sharpe"] is None
    expect_error(lambda: performance_metrics(pd.Series([1.0])))
    expect_error(lambda: performance_metrics(pd.Series([1.0, np.nan])))


def test_var_cvar_hand():
    var, cvar = empirical_var_cvar([-0.02, -0.01], 0.5)
    assert abs(var - (-0.02)) < 1e-12
    assert abs(cvar - (-0.01)) < 1e-12
    # 离散并列值：损失 [-0.03,-0.02,-0.02,-0.01]，α=0.5。
    # 最小化定义给出 VaR=-0.02，CVaR=-0.02+(0.01)/(4*0.5)=-0.015；
    # 直接平均最差两个点会得到 -0.025，是任务卡明确不采用的口径。
    var2, cvar2 = empirical_var_cvar([-0.03, -0.02, -0.02, -0.01], 0.5)
    assert abs(var2 - (-0.02)) < 1e-12
    assert abs(cvar2 - (-0.015)) < 1e-12


def test_cvar_lp():
    R = pd.DataFrame({"A": [0.02, 0.01]})
    w, info = cvar_portfolio(R.to_numpy(), alpha=0.5, loss_max=-0.005,
                             assets=["A"])
    assert w is not None and info["status"] == "optimal", info
    # ζ 必须自由：若被限制非负，loss_max=-0.005 会被误判不可行。
    # 注意 ζ 未必等于经验 VaR/CVaR（目标不含 ζ），指标从损失重算。
    assert info["zeta"] < 0, "zeta 被限制非负说明边界设错"
    assert abs(info["empirical_cvar"] - (-0.01)) < 1e-9
    w2, info2 = cvar_portfolio(R.to_numpy(), alpha=0.5, loss_max=-0.02,
                               assets=["A"])
    assert w2 is None and info2["status"] == "infeasible", info2


def test_mv():
    rng = np.random.default_rng(11)
    R = rng.normal(0, 0.01, (200, 3)) + np.array([0.0, 0.001, 0.002])
    assets = ["A", "B", "C"]
    w, info = mean_variance(R, sigma_max=1.0, assets=assets)
    assert w is not None and info["status"] == "optimal", info
    assert abs(w.sum() - 1) < 1e-8
    assert w["C"] > 0.99, w
    # 风险上限低于全投资最小波动 → 明确不可行
    w2, info2 = mean_variance(R, sigma_max=1e-6, assets=assets)
    assert w2 is None and info2["status"] == "infeasible", info2


def test_var_heuristic():
    rng = np.random.default_rng(5)
    R = rng.normal(0, 0.01, (200, 4))
    assets = ["A", "B", "C", "D"]
    w1, i1 = var_heuristic(R, alpha=0.9, loss_max=1.0, assets=assets,
                           budget=500, seed=7)
    w2, i2 = var_heuristic(R, alpha=0.9, loss_max=1.0, assets=assets,
                           budget=500, seed=7)
    assert w1 is not None and i1["status"] == "feasible_heuristic", i1
    assert np.allclose(w1.to_numpy(), w2.to_numpy()), "同种子应可复现"
    assert i1["feasible_count"] <= 500
    U = var_candidates(4, 500, 7)
    assert np.allclose(U[0], 0.25) and np.allclose(U[1:5], np.eye(4))
    w3, i3 = var_heuristic(R, alpha=0.9, loss_max=-10.0, assets=assets,
                           budget=500, seed=7)
    assert w3 is None and i3["status"] == "no_feasible_candidate", i3


def test_var_warm_start_preserves_known_feasible_solution():
    # All legacy deterministic candidates fail: only weights A in [.30, .342857]
    # satisfy the .008 loss limit; the optimal CVaR allocation is (.30, .70).
    R = np.tile([[-0.10, 0.04], [0.02, -0.02]], (100, 1))
    assets = ["A", "B"]
    w95, info95 = cvar_portfolio(R, .95, .008, assets)
    w99, info99 = cvar_portfolio(R, .99, .008, assets)
    assert w95 is not None and w99 is not None, (info95, info99)
    old, old_info = var_heuristic(R, .95, .008, assets, budget=3, seed=0)
    assert old is None and old_info["status"] == "no_feasible_candidate", old_info
    warm = {"cvar_a95": w95, "cvar_a99": w99}
    w, info = var_heuristic(R, .95, .008, assets, budget=5, seed=0,
                            warm_starts=warm)
    assert w is not None and info["status"] == "feasible_heuristic", info
    assert info["objective"] >= info95["objective"] - 1e-12
    assert info["total_candidates"] == 5 and info["random_budget"] == 0, info
    assert info["feasible_count"] == 2, info
    assert info["selected_candidate_id"] in warm, info
    assert [row["candidate_id"] for row in info["warm_starts"]] == list(warm)
    for row in info["warm_starts"]:
        assert row["weights"] == dict(warm[row["candidate_id"]]), row
        assert row["feasible"] and row["candidate_index"] in (3, 4), row
    reordered = {"cvar_a95": w95.reindex(["B", "A"])}
    w2, _ = var_heuristic(R, .95, .008, assets, budget=4, seed=0,
                           warm_starts=reordered)
    assert np.allclose(w2.to_numpy(), [.3, .7]), w2
    expect_error(lambda: var_heuristic(R, .95, .008, assets, budget=4,
                                      seed=0, warm_starts=warm))


def test_estimation_risk_statistics_are_labeled_separately():
    from evalcore import estimation_risk_statistics

    R = pd.DataFrame({"A": [-.10, -.04, .02, .04],
                      "B": [-.06, -.02, .02, .04]})
    rows = estimation_risk_statistics(R, alphas=(.5, .75))
    assert len(rows) == 2
    assert rows[0]["portfolio_id"] == "equal" and rows[0]["alpha"] == .5
    # Equal-weight losses sorted [-.04, -.02, .03, .08].
    assert abs(rows[0]["empirical_var"] - (-.02)) < 1e-12
    assert abs(rows[0]["empirical_cvar"] - .055) < 1e-12
    assert abs(rows[1]["empirical_var"] - .03) < 1e-12
    assert abs(rows[1]["empirical_cvar"] - .08) < 1e-12
    assert all(row["n_days"] == 4 and row["n_assets"] == 2 for row in rows)


def test_feasible_heuristic_status_validates_real_weights():
    from frozen import check_task02

    R = pd.DataFrame({"A": [.01, -.02], "B": [.03, .00]}, index=["d1", "d2"])
    w = pd.Series({"A": .5, "B": .5})
    port = np.array([.02, -.01])
    runs = pd.DataFrame([{"model_id": "l1_cv", "status": "feasible_heuristic", "data_version": "v-test",
                         "estimation_end": "d2", "daily_mean_in_sample": .005,
                         "daily_sd_in_sample": float(port.std(ddof=1)), "holdings": 2}])
    frozen = {"ids": ["A", "B"], "returns": R,
              "periods": {"data_version": "v-test", "estimation_start": "d1",
                          "estimation_end": "d2"}}
    t02 = {"weights": {"l1_cv": w}, "statuses": {"l1_cv": "feasible_heuristic"}, "runs": runs,
           "weights_rows": [{"model_id": "l1_cv", "asset_id": asset, "status": "feasible_heuristic"}
                            for asset in ["A", "B"]]}
    checks = check_task02(frozen, t02)
    assert any(c["check"] == "l1_cv:budget_one" for c in checks), checks
    assert all(c["passed"] for c in checks), checks
    t02["weights"]["l1_cv"] = pd.Series({"A": .8, "B": .8})
    assert not all(c["passed"] for c in check_task02(frozen, t02))


def test_equal_weight():
    w, info = equal_weight(["A", "B", "C"])
    assert np.allclose(w.to_numpy(), 1 / 3)
    assert info["status"] == "optimal"


def test_failed_status_is_preserved():
    """上游 solver_failure 等状态不得阻断整批评价，且不校验空权重。"""
    from frozen import check_task02

    ids = ["A", "B"]
    R = pd.DataFrame({"A": [0.01, -0.02, 0.03], "B": [0.0, 0.01, -0.01]},
                     index=["d1", "d2", "d3"])
    periods = {"data_version": "v-test", "estimation_start": "d1",
               "estimation_end": "d2"}
    w = pd.Series({"A": 0.5, "B": 0.5})
    port = R.loc[["d1", "d2"]].to_numpy() @ w.to_numpy()
    runs = pd.DataFrame([
        {"model_id": "l1_cv", "status": "optimal", "data_version": "v-test", "estimation_end": "d2",
         "daily_mean_in_sample": float(port.mean()),
         "daily_sd_in_sample": float(port.std(ddof=1)), "holdings": 2},
        {"model_id": "bad", "status": "solver_failure", "data_version": "v-test", "estimation_end": "d2",
         "daily_mean_in_sample": np.nan, "daily_sd_in_sample": np.nan,
         "holdings": np.nan},
    ])
    t02 = {"weights": {"l1_cv": w, "bad": pd.Series({"A": np.nan, "B": np.nan})},
           "statuses": {"l1_cv": "optimal", "bad": "solver_failure"}, "runs": runs,
           "weights_rows": [{"model_id": model, "asset_id": asset, "status": status}
                            for model, status in [("l1_cv", "optimal"), ("bad", "solver_failure")]
                            for asset in ["A", "B"]]}
    checks = check_task02({"ids": ids, "returns": R, "periods": periods}, t02)
    failed = [c for c in checks if not c["passed"]]
    assert not failed, failed
    names = [c["check"] for c in checks]
    assert "bad:status_present" in names
    assert not any(n.startswith("bad:budget") for n in names)


def test_run_exports_risks_and_preserves_heuristic_and_failure_states():
    import contextlib
    import io
    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory
    import run
    from frozen import load_frozen, sha256

    frozen = load_frozen(run.DATA)
    periods = frozen["periods"]
    est = frozen["returns"].loc[periods["estimation_start"]:periods["estimation_end"]]
    ids = frozen["ids"]
    port = est.to_numpy().mean(axis=1)
    original_task02 = run.TASK02
    with TemporaryDirectory(dir=Path(__file__).parent) as temporary:
        base = Path(temporary)
        t2 = base / "task02" / "results"
        t2.mkdir(parents=True)
        weights = []
        for model, status, weight in [("l1_cv", "feasible_heuristic", 1 / len(ids)),
                                       ("bad", "solver_failure", None)]:
            for asset in ids:
                weights.append({"model_id": model, "asset_id": asset, "name": asset,
                                "invested_weight": weight, "status": status})
        pd.DataFrame(weights).to_csv(t2 / "weights.csv", index=False)
        rows = []
        for model in ["l1_cv", "bad"]:
            rows.append({"model_id": model,
                         "status": "feasible_heuristic" if model == "l1_cv" else "solver_failure",
                         "data_version": periods["data_version"],
                         "estimation_end": periods["estimation_end"],
                         "daily_mean_in_sample": float(port.mean()),
                         "daily_sd_in_sample": float(port.std(ddof=1)), "holdings": len(ids)})
        pd.DataFrame(rows).to_csv(t2 / "runs.csv", index=False)
        (t2 / "manifest.json").write_text(json.dumps({"outputs": {
            name: sha256(t2 / name) for name in ["weights.csv", "runs.csv"]}}))
        output = base / "results"
        try:
            run.TASK02 = base / "task02"
            with contextlib.redirect_stdout(io.StringIO()):
                run.main(output)
        finally:
            run.TASK02 = original_task02
        metrics = pd.read_csv(output / "metrics.csv").set_index("model_id")
        assert set(metrics.index[metrics.role == "main"]) == {
            "l1_cv", "equal", "mv", "cvar_a95", "var_a95_s0", "gmv",
            "inverse_volatility", "momentum_top20", "low_volatility_top20"}
        assert metrics.loc["l1_cv", "status"] == "feasible_heuristic"
        assert np.isfinite(metrics.loc["l1_cv", "cumulative_return"]), metrics.loc["l1_cv"]
        assert metrics.loc["bad", "status"] == "solver_failure"
        assert pd.isna(metrics.loc["bad", "cumulative_return"])
        nav = pd.read_csv(output / "nav.csv")
        assert "l1_cv" in nav and "bad" not in nav
        risks = pd.read_csv(output / "risk_statistics.csv").set_index("alpha")
        # Frozen estimation data, independently audited decimals (not percent units).
        assert abs(risks.loc[.95, "empirical_var"] - .031404997) < 1e-9
        assert abs(risks.loc[.95, "empirical_cvar"] - .042619536) < 1e-9
        assert abs(risks.loc[.99, "empirical_var"] - .042893239) < 1e-9
        assert abs(risks.loc[.99, "empirical_cvar"] - .068470454) < 1e-9
        manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["outputs"]["risk_statistics.csv"] == sha256(output / "risk_statistics.csv")
        for seed in [0, 1, 2]:
            model = f"var_a95_s{seed}"
            info = manifest["benchmark_runs"][model]
            assert info["status"] == "feasible_heuristic" and info["total_candidates"] == 3000
            assert info["random_budget"] == 2897
            assert {x["candidate_id"] for x in info["warm_starts"]} == {"cvar_a95", "cvar_a99"}
            feasible_warm = [x["objective"] for x in info["warm_starts"] if x["feasible"]]
            assert info["objective"] >= max(feasible_warm) - 1e-12
        from verify import verify_benchmark_artifacts
        checks = verify_benchmark_artifacts(output, est)
        assert all(ok for _, ok, _ in checks), checks
        original_metrics = (output / "metrics.csv").read_bytes()
        original_comparison = (output / "comparison.csv").read_bytes()
        metrics.drop(index="l1_cv").to_csv(output / "metrics.csv")
        comp_without_main = pd.read_csv(output / "comparison.csv")
        comp_without_main[comp_without_main.model_id != "l1_cv"].to_csv(output / "comparison.csv", index=False)
        checks = verify_benchmark_artifacts(output, est)
        assert any(name == "roles:l1_cv_main" and not ok for name, ok, _ in checks), checks
        (output / "metrics.csv").write_bytes(original_metrics)
        (output / "comparison.csv").write_bytes(original_comparison)
        manifest["benchmark_runs"]["var_a95_s0"]["status"] = "optimal"
        (output / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        checks = verify_benchmark_artifacts(output, est)
        assert any(name == "var:var_a95_s0:independent_search_replay" and not ok
                   for name, ok, _ in checks), checks
        manifest["benchmark_runs"]["var_a95_s0"]["status"] = "feasible_heuristic"
        (output / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        risks.loc[.95, "empirical_cvar"] = risks.loc[.95, "empirical_var"]
        risks.to_csv(output / "risk_statistics.csv")
        checks = verify_benchmark_artifacts(output, est)
        assert not all(ok for _, ok, _ in checks), checks
        assert any("risk_statistics.csv" in name and not ok for name, ok, _ in checks)
        # An explicitly failed main model remains present with blank performance.
        failed_weights = pd.read_csv(t2 / "weights.csv")
        failed_weights.loc[failed_weights.model_id == "l1_cv", "status"] = "solver_failure"
        failed_weights.loc[failed_weights.model_id == "l1_cv", "invested_weight"] = np.nan
        failed_weights.to_csv(t2 / "weights.csv", index=False)
        failed_runs = pd.read_csv(t2 / "runs.csv")
        failed_runs.loc[failed_runs.model_id == "l1_cv", "status"] = "solver_failure"
        failed_runs.to_csv(t2 / "runs.csv", index=False)
        (t2 / "manifest.json").write_text(json.dumps({"outputs": {
            name: sha256(t2 / name) for name in ["weights.csv", "runs.csv"]}}))
        try:
            run.TASK02 = base / "task02"
            with contextlib.redirect_stdout(io.StringIO()):
                run.main(output)
        finally:
            run.TASK02 = original_task02
        failed_metrics = pd.read_csv(output / "metrics.csv").set_index("model_id")
        assert failed_metrics.loc["l1_cv", "status"] == "solver_failure"
        assert failed_metrics.loc["l1_cv", "role"] == "main"
        assert pd.isna(failed_metrics.loc["l1_cv", "cumulative_return"])
        assert "l1_cv" not in pd.read_csv(output / "nav.csv")
        assert all(ok for _, ok, _ in verify_benchmark_artifacts(output, est))
        from verify import verify_task02_provenance
        assert all(ok for _, ok, _ in verify_task02_provenance(output, base / "task02"))


def task02_contract_fixture():
    """Two valid models; literal stats correspond to returns [.02, -.01]."""
    R = pd.DataFrame({"A": [.01, -.02], "B": [.03, .00]}, index=["d1", "d2"])
    frozen = {"ids": ["A", "B"], "returns": R,
              "periods": {"data_version": "v-test", "estimation_start": "d1",
                          "estimation_end": "d2"}}
    models = ["l1_cv", "l1_eta_0.01"]
    t02 = {"weights": {m: pd.Series({"A": .5, "B": .5}) for m in models},
           "statuses": {m: "optimal" for m in models},
           "weights_rows": [{"model_id": m, "asset_id": a, "status": "optimal"}
                            for m in models for a in ["A", "B"]],
           "runs": pd.DataFrame([{"model_id": m, "status": "optimal", "data_version": "v-test",
                                  "estimation_end": "d2", "daily_mean_in_sample": .005,
                                  "daily_sd_in_sample": .021213203435596427, "holdings": 2}
                                 for m in models])}
    return frozen, t02


def test_task02_rejects_delivery_without_l1_cv():
    from frozen import check_task02

    frozen, t02 = task02_contract_fixture()
    assert all(c["passed"] for c in check_task02(frozen, t02))
    for key in ["weights", "statuses"]:
        del t02[key]["l1_cv"]
    t02["weights_rows"] = [r for r in t02["weights_rows"] if r["model_id"] != "l1_cv"]
    t02["runs"] = t02["runs"][t02["runs"].model_id != "l1_cv"]
    checks = check_task02(frozen, t02)
    assert any(not c["passed"] for c in checks), checks


def test_task02_rejects_different_weight_and_run_model_sets():
    from frozen import check_task02

    for missing_in in ["weights", "runs"]:
        frozen, t02 = task02_contract_fixture()
        if missing_in == "weights":
            del t02["weights"]["l1_eta_0.01"]
            del t02["statuses"]["l1_eta_0.01"]
            t02["weights_rows"] = [r for r in t02["weights_rows"] if r["model_id"] == "l1_cv"]
        else:
            t02["runs"] = t02["runs"][t02["runs"].model_id == "l1_cv"]
            # A failed orphan previously bypassed runs.loc access entirely.
            t02["statuses"]["l1_eta_0.01"] = "solver_failure"
            for row in t02["weights_rows"]:
                if row["model_id"] == "l1_eta_0.01":
                    row["status"] = "solver_failure"
        checks = check_task02(frozen, t02)
        assert any(not c["passed"] for c in checks), (missing_in, checks)


def test_task02_rejects_weight_and_run_status_mismatch():
    from frozen import check_task02

    frozen, t02 = task02_contract_fixture()
    t02["runs"].loc[t02["runs"].model_id == "l1_cv", "status"] = "solver_failure"
    checks = check_task02(frozen, t02)
    assert any(not c["passed"] for c in checks), checks


def test_task02_rejects_mixed_statuses_inside_weight_rows():
    from frozen import check_task02

    frozen, t02 = task02_contract_fixture()
    t02["weights_rows"][1]["status"] = "solver_failure"
    checks = check_task02(frozen, t02)
    assert any(not c["passed"] for c in checks), checks


def test_verify_detects_task02_provenance_drift():
    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from frozen import sha256
    from verify import verify_task02_provenance

    with TemporaryDirectory(dir=Path(__file__).parent) as temporary:
        base = Path(temporary)
        t2 = base / "task02" / "results"
        t2.mkdir(parents=True)
        for name in ["weights.csv", "runs.csv"]:
            (t2 / name).write_text("model_id,status\nl1_cv,optimal\n", encoding="utf-8")
        outputs = {name: sha256(t2 / name) for name in ["weights.csv", "runs.csv"]}
        (t2 / "manifest.json").write_text(json.dumps({"outputs": outputs}), encoding="utf-8")
        out = base / "results"
        out.mkdir()
        for name in ["metrics.csv", "comparison.csv"]:
            (out / name).write_text("model_id,status,source,role\nl1_cv,optimal,task02,main\n", encoding="utf-8")
        metadata = {"task02_outputs": outputs.copy(), "task02_weights_sha256": outputs["weights.csv"]}
        path = out / "run_manifest.json"
        path.write_text(json.dumps(metadata), encoding="utf-8")
        assert all(ok for _, ok, _ in verify_task02_provenance(out, base / "task02"))
        for field in ["task02_outputs", "task02_weights_sha256"]:
            bad = dict(metadata)
            bad[field] = {"weights.csv": "0" * 64} if field == "task02_outputs" else "0" * 64
            path.write_text(json.dumps(bad), encoding="utf-8")
            checks = verify_task02_provenance(out, base / "task02")
            expected_check = ("task02:recorded_outputs_match_current_manifest"
                              if field == "task02_outputs" else
                              "task02:recorded_weights_sha256_matches_current_source")
            assert any(name == expected_check and not ok for name, ok, _ in checks), (field, checks)
        path.write_text(json.dumps(metadata), encoding="utf-8")
        for name in ["weights.csv", "runs.csv"]:
            (t2 / name).write_text("model_id,status\nl1_cv,solver_failure\n", encoding="utf-8")
        for name in ["metrics.csv", "comparison.csv"]:
            (out / name).write_text("model_id,status,source,role\nl1_cv,solver_failure,task02,main\n", encoding="utf-8")
        new_outputs = {name: sha256(t2 / name) for name in ["weights.csv", "runs.csv"]}
        (t2 / "manifest.json").write_text(json.dumps({"outputs": new_outputs}), encoding="utf-8")
        checks = verify_task02_provenance(out, base / "task02")
        for expected_check in ["task02:recorded_outputs_match_current_manifest",
                               "task02:recorded_weights_sha256_matches_current_source"]:
            assert any(name == expected_check and not ok for name, ok, _ in checks), checks


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"{fn.__name__} OK")
    print(f"\n全部 {len(tests)} 组测试通过")
