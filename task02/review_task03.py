"""Audit task 02 outcomes against task 03 exports using frozen price ratios."""
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_delivery_01" / "data"
RESULTS = ROOT / "task03" / "results"


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def audit():
    weight_file = ROOT / "task02" / "results" / "weights.csv"
    manifest = json.loads((ROOT/"task02/results/manifest.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(weight_file.read_bytes()).hexdigest() == manifest["outputs"]["weights.csv"]
    weights = defaultdict(dict)
    for row in rows(weight_file):
        if row["status"] != "optimal":
            continue
        group = weights[row["model_id"]]
        assert row["asset_id"] not in group
        group[row["asset_id"]] = float(row["invested_weight"])
    assets = {a for group in weights.values() for a in group}
    days = [r["date"] for r in rows(DATA/"calendar.csv")
            if "2026-07-01" <= r["date"] <= "2026-11-10" and r["is_open"] == "1"]
    assert len(days) == 89 and days[:2] == ["2026-07-01","2026-07-02"] and days[-1] == "2026-11-10"
    price = defaultdict(dict)
    for row in rows(DATA/"prices_long.csv"):
        d, a = row["date"], row["asset_id"]
        if d < days[0] or d > days[-1] or a not in assets:
            continue
        if d == days[0]:
            assert row["tradable_status"] == "observed_trading" and float(row["volume"]) > 0
            assert float(row["close_raw"]) > 0 and row["quality_invalid"] == row["key_conflict"] == "0"
        assert row["valuation_valid"] == "1" and a not in price[d]
        price[d][a] = float(row["close_adj"])
    nav = {r["date"]:r for r in rows(RESULTS/"nav.csv")}
    daily = {(r["date"],r["model_id"]):float(r["gross_return"]) for r in rows(RESULTS/"daily_returns.csv")}
    expected = {r["model_id"] for r in rows(ROOT/"task02/results/runs.csv") if r["status"] == "optimal"}
    assert list(nav) == days and set(weights) == expected and "l1_cv" in weights and len(assets) == 100
    tables = [{r["model_id"]:r for r in rows(RESULTS/name)} for name in ["metrics.csv","comparison.csv"]]
    for model, group in weights.items():
        # Use the exported weights as-is; the entry NAV is the agreed unit of account.
        budget = math.fsum(group.values())
        assert abs(budget-1) < 1e-8
        units = {a:u/price[days[0]][a] for a,u in group.items()}
        path = [1.0] + [math.fsum(q*price[d][a] for a,q in units.items()) for d in days[1:]]
        ret = [v/u-1 for u,v in zip(path,path[1:])]
        mean, sd = statistics.mean(ret), statistics.stdev(ret)
        values = dict(final_nav=path[-1], cumulative_return=path[-1]-1,
                      annualized_return=path[-1]**(252/len(ret))-1, annualized_vol=sd*math.sqrt(252),
                      sharpe=math.sqrt(252)*mean/sd, max_drawdown=min(v/max(path[:i+1])-1 for i,v in enumerate(path)))
        error = max(abs(v-float(nav[d][model])) for v,d in zip(path,days))
        assert error < 1e-10 and max(abs(r-daily[d,model]) for r,d in zip(ret,days[1:])) < 1e-8
        for table in tables:
            row = table[model]
            assert row["status"] == "optimal" and int(row["n_days"]) == 88
            assert all(abs(value-float(row[key])) < 1e-8 for key,value in values.items())
        print(f"{model}: first={path[1]:.9f} final={path[-1]:.9f} drawdown={values['max_drawdown']:.9f} max_nav_error={error:.2g}")
    print(f"PASS: 100 entry bars, {len(weights)} strategies, 89 NAV dates, 88 returns per strategy")


if __name__ == "__main__":
    audit()
