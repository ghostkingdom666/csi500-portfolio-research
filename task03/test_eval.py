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
    assert w1 is not None and i1["status"] == "optimal", i1
    assert np.allclose(w1.to_numpy(), w2.to_numpy()), "同种子应可复现"
    assert i1["feasible_count"] <= 500
    U = var_candidates(4, 500, 7)
    assert np.allclose(U[0], 0.25) and np.allclose(U[1:5], np.eye(4))
    w3, i3 = var_heuristic(R, alpha=0.9, loss_max=-10.0, assets=assets,
                           budget=500, seed=7)
    assert w3 is None and i3["status"] == "infeasible", i3


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
        {"model_id": "ok", "data_version": "v-test", "estimation_end": "d2",
         "daily_mean_in_sample": float(port.mean()),
         "daily_sd_in_sample": float(port.std(ddof=1)), "holdings": 2},
        {"model_id": "bad", "data_version": "v-test", "estimation_end": "d2",
         "daily_mean_in_sample": np.nan, "daily_sd_in_sample": np.nan,
         "holdings": np.nan},
    ])
    t02 = {"weights": {"ok": w, "bad": pd.Series({"A": np.nan, "B": np.nan})},
           "statuses": {"ok": "optimal", "bad": "solver_failure"}, "runs": runs}
    checks = check_task02({"ids": ids, "returns": R, "periods": periods}, t02)
    failed = [c for c in checks if not c["passed"]]
    assert not failed, failed
    names = [c["check"] for c in checks]
    assert "bad:status_present" in names
    assert not any(n.startswith("bad:budget") for n in names)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"{fn.__name__} OK")
    print(f"\n全部 {len(tests)} 组测试通过")
