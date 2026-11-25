"""Hand-checkable fixed-rule tests; integration uses temporary output only."""

import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import simple_baselines as simple


def expect_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def diagonal_returns():
    a = np.sqrt(3) * .01 / 2
    return pd.DataFrame([[-a, -2*a], [-a, 2*a], [a, -2*a], [a, 2*a]],
                        columns=["A", "B"])


def test_gmv_and_inverse_volatility_hand_solution():
    R = diagonal_returns()
    assert np.allclose(R.cov(), np.diag([.0001, .0004]), atol=1e-16)
    w, info = simple.gmv(R, list(R.columns))
    assert info["status"] == "optimal", info
    assert np.allclose(w, [.8, .2], atol=1e-8), w
    assert abs(info["objective"] - .00008) < 1e-12
    assert info["diagnostics"]["kkt_residual_scaled"] < 1e-6
    w, info = simple.inverse_volatility(R, list(R.columns))
    assert info["status"] == "optimal", info
    assert np.allclose(w, [2/3, 1/3], atol=1e-12), w
    assert info["estimation"]["ddof"] == 1
    assert info["scores"] == {"A": .01, "B": .02}


def test_momentum_exact_boundaries_and_skipped_recent_days():
    # Row 0 is before the 126-day window, row 1 is its first day,
    # row 105 is its last included day; rows 106..126 are skipped.
    R = pd.DataFrame(0., index=[f"d{i:03}" for i in range(127)], columns=["B", "A", "C"])
    R.loc["d000", "C"] = .9
    R.loc["d001", "A"] = .1
    R.loc["d105", "A"] = .1
    R.loc["d106":, "B"] = .5
    w, info = simple.momentum_top20(R, list(R.columns), k=1)
    assert w.to_dict() == {"B": 0., "A": 1., "C": 0.}, w
    assert abs(info["scores"]["A"] - .21) < 1e-12
    assert info["scores"]["B"] == 0. and info["scores"]["C"] == 0.
    assert info["signal"]["start"] == "d001" and info["signal"]["end"] == "d105"
    assert info["signal"]["n_days"] == 105
    assert info["signal"]["excluded_recent_days"] == 21
    # Exactly 126 observations suffice and must use its first observation.
    w2, _ = simple.momentum_top20(R.iloc[1:], list(R.columns), k=1)
    assert w.equals(w2)


def test_low_volatility_uses_only_last_60_days():
    tail = np.tile([-.01, .01], 30)
    R = pd.DataFrame({"B": np.r_[0., tail * 2], "A": np.r_[.9, tail]},
                     index=[f"d{i:03}" for i in range(61)])
    w, info = simple.low_volatility_top20(R, list(R.columns), k=1)
    assert w.to_dict() == {"B": 0., "A": 1.}
    assert info["signal"] == {"start": "d001", "end": "d060", "n_days": 60, "ddof": 1}
    assert abs(info["scores"]["A"] - np.sqrt(60/59) * .01) < 1e-12
    w2, _ = simple.low_volatility_top20(R.iloc[1:], list(R.columns), k=1)
    assert w.equals(w2)


def test_top20_ties_are_broken_by_asset_id_after_column_reordering():
    labels = [f"A{i:02}" for i in range(25)][::-1]
    R = pd.DataFrame(np.tile(np.linspace(-.01, .01, 126)[:, None], (1, 25)), columns=labels)
    for solve in [simple.momentum_top20, simple.low_volatility_top20]:
        w, info = solve(R, labels)
        w2, info2 = solve(R[labels[::-1]], labels[::-1])
        assert np.allclose(w.sort_index(), w2.sort_index())
        assert info["selected_assets"] == info2["selected_assets"] == sorted(labels)[:20]
        assert len(w) == 25 and (w > 0).sum() == 20
        assert set(w[w > 0]) == {.05} and abs(w.sum() - 1) < 1e-12


def test_invalid_history_returns_and_k_are_rejected():
    R = diagonal_returns()
    for solve in [simple.gmv, simple.inverse_volatility, simple.momentum_top20, simple.low_volatility_top20]:
        for invalid in [np.nan, np.inf, -1., -1.1]:
            bad = pd.DataFrame(.01, index=range(126), columns=["A", "B"])
            bad.iloc[0, 0] = invalid
            expect_error(lambda s=solve, b=bad: s(b, ["A", "B"]))
        expect_error(lambda s=solve: s(R, ["A", "A"]))
        expect_error(lambda s=solve: s(R, ["A"]))
    for solve, rows in [(simple.gmv, 1), (simple.inverse_volatility, 1),
                        (simple.momentum_top20, 125), (simple.low_volatility_top20, 59)]:
        expect_error(lambda s=solve, t=rows: s(np.full((t, 2), .01), ["A", "B"]))
    R = pd.DataFrame(np.full((126, 2), .01), columns=["A", "B"])
    for solve in [simple.momentum_top20, simple.low_volatility_top20]:
        for k in [0, -1, 3, 1.5, True]:
            expect_error(lambda s=solve, value=k: s(R, ["A", "B"], k=value))


def test_inverse_volatility_zero_vol_is_an_explicit_failure():
    for constant in [0., .01]:
        R = pd.DataFrame({"A": np.full(60, constant), "B": np.linspace(-.01, .01, 60)})
        w, info = simple.inverse_volatility(R, list(R.columns))
        assert w is None and info["status"] == "fail", (constant, w, info)
        assert info["invalid_volatility_assets"] == ["A"]


def test_real_end_to_end_rules_dates_provenance_and_independent_verification():
    import run
    import verify
    from frozen import load_frozen, sha256

    frozen = load_frozen(run.DATA)
    p = frozen["periods"]
    est = frozen["returns"].loc[p["estimation_start"]:p["estimation_end"]]
    assert len(est) == 359 and est.index[-1] == "2026-06-30"
    names = ["gmv", "inverse_volatility", "momentum_top20", "low_volatility_top20"]
    # Inspect the actual arguments at the estimation boundary, while executing
    # each real implementation. A test/entry row passed to a solver fails here.
    originals = {name: getattr(run, name) for name in names}
    seen = []
    def observed(name, solve):
        def wrapped(R, *args, **kwargs):
            assert isinstance(R, pd.DataFrame) and R.equals(est), name
            seen.append(name)
            return solve(R, *args, **kwargs)
        return wrapped
    with TemporaryDirectory(dir=Path(__file__).parent) as temporary:
        output = Path(temporary)
        try:
            for name, solve in originals.items():
                setattr(run, name, observed(name, solve))
            with contextlib.redirect_stdout(io.StringIO()):
                metrics = run.main(output).set_index("model_id")
        finally:
            for name, solve in originals.items():
                setattr(run, name, solve)
        assert sorted(seen) == sorted(names)
        assert len(metrics) == 20 and (metrics.role == "main").sum() == 9
        assert set(names) <= set(metrics.index)
        assert (metrics.loc[names, "status"] == "optimal").all()
        assert (metrics.loc[names, "n_days"] == 88).all()
        assert (metrics.loc[names, "role"] == "main").all()
        nav = pd.read_csv(output / "nav.csv", index_col=0)
        assert nav.index[0] == "2026-07-01" and len(nav) == 89
        daily = pd.read_csv(output / "daily_returns.csv")
        target_dates = list(daily[daily.model_id == "l1_cv"].date)
        assert len(target_dates) == 88 and target_dates[0] == "2026-07-02" and target_dates[-1] == "2026-11-10"
        for name in names:
            assert list(daily[daily.model_id == name].date) == target_dates
        w = pd.read_csv(output / "weights_benchmarks.csv")
        for name in names:
            got = w[w.model_id == name]
            assert len(got) == 100 and set(got.asset_id) == set(est.columns)
            assert (got.invested_weight >= 0).all() and abs(got.invested_weight.sum() - 1) < 1e-8
        manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["code_sha256"]["simple_baselines.py"] == sha256(Path(__file__).parent / "simple_baselines.py")
        for name in names:
            info = manifest["benchmark_runs"][name]
            assert info["estimation"] == {"start": "2025-01-02", "end": "2026-06-30", "n_days": 359,
                                           "n_assets": 100, "ddof": 1}
            assert len(info["scores"]) == 100 if name != "gmv" else "diagnostics" in info
        assert manifest["benchmark_runs"]["momentum_top20"]["parameters"] == {"k": 20, "lookback": 126, "skip": 21}
        assert manifest["benchmark_runs"]["low_volatility_top20"]["parameters"] == {"k": 20, "lookback": 60}
        with contextlib.redirect_stdout(io.StringIO()):
            assert verify.main(output) == 0, (output / "verification.txt").read_text(encoding="utf-8")
        import make_charts
        with contextlib.redirect_stdout(io.StringIO()):
            make_charts.main(output)
        for name in ["nav_simple_baselines.png", "drawdown_simple_baselines.png", "metrics_simple_baselines.png"]:
            assert (output / "figures" / name).stat().st_size > 1000
        # Independent verifier must reject altered weights even after refreshing
        # the output hash. This catches rules/KKT rather than only provenance.
        original = w.copy()
        for name in names:
            altered = original.copy()
            mask = altered.model_id == name
            altered.loc[mask, "invested_weight"] = 0.
            altered.loc[altered.index[mask][0], "invested_weight"] = 1.
            altered.to_csv(output / "weights_benchmarks.csv", index=False)
            manifest["outputs"]["weights_benchmarks.csv"] = sha256(output / "weights_benchmarks.csv")
            (output / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            checks = verify.verify_benchmark_artifacts(output, est)
            assert any(check.startswith(f"simple:{name}:") and not ok for check, ok, _ in checks), checks


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"{test.__name__} OK")
    print(f"\nAll {len(tests)} groups passed")
