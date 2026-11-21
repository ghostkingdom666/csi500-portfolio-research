"""冻结输入读取与契约检查：只读 data_delivery_01 与 task02 的交付文件。

任何哈希、标签、日期、单位不一致都显式失败并记录，不静默修正。
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_DATA_FILES = [
    "periods.json",
    "calendar.csv",
    "model_assets.csv",
    "model_returns_wide.csv",
    "missing_mask.csv",
    "index_daily.csv",
]


def read_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv_rows(path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_frozen(data_dir) -> dict:
    """读取冻结数据包，返回结构化输入（不做业务判断）。"""
    data_dir = Path(data_dir)
    manifest = read_json(data_dir / "manifest.json")
    periods = read_json(data_dir / "periods.json")
    calendar = read_csv_rows(data_dir / "calendar.csv")
    assets = read_csv_rows(data_dir / "model_assets.csv")
    ids = [a["asset_id"] for a in assets]

    with (data_dir / "model_returns_wide.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    returns = pd.DataFrame(
        [[float(x) for x in r[1:]] for r in body],
        index=[r[0] for r in body], columns=header[1:],
    )
    index_rows = read_csv_rows(data_dir / "index_daily.csv")
    index = pd.DataFrame(index_rows).set_index("date")
    missing = pd.read_csv(data_dir / "missing_mask.csv", index_col=0)
    return {
        "manifest": manifest, "periods": periods, "calendar": calendar,
        "assets": assets, "ids": ids, "returns": returns,
        "index": index, "missing": missing, "header": header,
    }


def check_frozen(frozen, data_dir) -> list[dict]:
    """对冻结数据包做契约检查，返回 [{check, passed, detail}]。"""
    data_dir = Path(data_dir)
    manifest = frozen["manifest"]
    periods = frozen["periods"]
    ids = frozen["ids"]
    returns = frozen["returns"]
    calendar = frozen["calendar"]
    checks = []

    def add(name, passed, detail=""):
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    add("data_version_matches_periods",
        manifest["data_version"] == periods["data_version"],
        manifest["data_version"])
    for name in REQUIRED_DATA_FILES:
        expect = manifest["files"].get(name, {}).get("sha256")
        actual = sha256(data_dir / name)
        add(f"sha256:{name}", expect == actual, "" if expect == actual else "hash mismatch")

    add("asset_labels_unique", len(set(ids)) == len(ids), f"N={len(ids)}")
    add("asset_label_order", frozen["header"] == ["date", *ids],
        "" if frozen["header"] == ["date", *ids] else "header/asset order mismatch")
    add("returns_finite", bool(np.isfinite(returns.to_numpy()).all()), "")

    open_dates = [r["date"] for r in calendar
                  if str(r["is_open"]) == "1" and str(r.get("is_anchor")) != "1"]
    add("returns_rows_match_open_calendar",
        list(returns.index) == open_dates,
        f"returns={len(returns)} calendar_open={len(open_dates)}")
    add("returns_rows_sorted", list(returns.index) == sorted(returns.index), "")
    counts = manifest["counts"]
    add("estimation_T_matches_manifest",
        int((returns.index <= periods["estimation_end"]).sum()) == counts["estimation_T"],
        f"{int((returns.index <= periods['estimation_end']).sum())} vs {counts['estimation_T']}")
    add("test_T_matches_manifest",
        int((returns.index >= periods["test_start"]).sum()) == counts["test_trading_days_including_entry_date"],
        f"{int((returns.index >= periods['test_start']).sum())} vs {counts['test_trading_days_including_entry_date']}")
    add("last_date_is_last_complete_trading_date",
        returns.index[-1] == counts["last_complete_trading_date"],
        returns.index[-1])
    add("entry_date_in_calendar_and_open",
        periods["proposed_entry_date"] in open_dates,
        periods["proposed_entry_date"])
    add("first_return_date_is_next_open",
        periods["proposed_first_return_date"] in open_dates
        and open_dates.index(periods["proposed_first_return_date"])
        == open_dates.index(periods["proposed_entry_date"]) + 1,
        periods["proposed_first_return_date"])
    add("test_missing_return_cells_zero",
        int(frozen["missing"].loc[returns.index >= periods["test_start"]].to_numpy().sum()) == 0,
        int(frozen["missing"].loc[returns.index >= periods["test_start"]].to_numpy().sum()))
    return checks


def load_task02(task02_dir) -> dict:
    """读取第2人的权重、运行清单与 manifest。"""
    task02_dir = Path(task02_dir)
    weights_rows = read_csv_rows(task02_dir / "results" / "weights.csv")
    runs = pd.read_csv(task02_dir / "results" / "runs.csv", encoding="utf-8-sig")
    manifest = read_json(task02_dir / "results" / "manifest.json")
    weights: dict[str, pd.Series] = {}
    statuses: dict[str, str] = {}
    names: dict[str, str] = {}
    for row in weights_rows:
        model = row["model_id"]
        w = float(row["invested_weight"]) if row["invested_weight"] not in ("", "None") else np.nan
        if model not in weights:
            weights[model] = {}
            statuses[model] = row["status"]
        weights[model][row["asset_id"]] = w
        names[row["asset_id"]] = row["name"]
    return {
        "weights_rows": weights_rows,
        "weights": {m: pd.Series(v, dtype=float) for m, v in weights.items()},
        "statuses": statuses,
        "names": names,
        "runs": runs,
        "manifest": manifest,
        "weights_sha256": sha256(task02_dir / "results" / "weights.csv"),
    }


def check_task02(frozen, t02) -> list[dict]:
    """第2人交付的可评价性检查。"""
    checks = []
    ids = frozen["ids"]
    returns = frozen["returns"]
    periods = frozen["periods"]
    runs = t02["runs"].set_index("model_id")

    def add(name, passed, detail=""):
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    add("task02_data_version_matches",
        str(runs["data_version"].iloc[0]) == periods["data_version"], runs["data_version"].iloc[0])
    add("task02_estimation_end_matches",
        str(runs["estimation_end"].iloc[0]) == periods["estimation_end"], runs["estimation_end"].iloc[0])
    allowed = {"optimal", "infeasible", "fail", "zero_solution", "optimal_inaccurate"}
    for model, w in t02["weights"].items():
        status = t02["statuses"][model]
        add(f"{model}:status_allowed", status in allowed, status)
        if status != "optimal":
            # 失败/不可行模型保留结果行，不做权重校验，也不阻断其他模型评价。
            continue
        unknown = [a for a in w.index if a not in ids]
        add(f"{model}:assets_in_frozen_pool", not unknown, unknown)
        add(f"{model}:no_nan", bool(np.isfinite(w.to_numpy()).all()), "")
        add(f"{model}:nonnegative", bool((w >= -1e-12).all()), float(w.min()))
        add(f"{model}:budget_one", abs(float(w.sum()) - 1.0) <= 1e-8, float(w.sum()))
        # 估计期样本均值/波动应与第2人 runs.csv 一致（描述初始权重对估计样本的暴露）
        est = returns.loc[(returns.index >= periods["estimation_start"])
                          & (returns.index <= periods["estimation_end"]), w.index]
        port = est.to_numpy() @ w.to_numpy()
        add(f"{model}:in_sample_daily_mean_matches",
            abs(float(port.mean()) - float(runs.loc[model, "daily_mean_in_sample"])) < 1e-12,
            f"{port.mean():.12f} vs {runs.loc[model, 'daily_mean_in_sample']:.12f}")
        add(f"{model}:in_sample_daily_sd_matches",
            abs(float(port.std(ddof=1)) - float(runs.loc[model, "daily_sd_in_sample"])) < 1e-12,
            f"{port.std(ddof=1):.12f} vs {runs.loc[model, 'daily_sd_in_sample']:.12f}")
        add(f"{model}:holdings_match",
            int((w.to_numpy() > 1e-8).sum()) == int(runs.loc[model, "holdings"]),
            f"{int((w.to_numpy() > 1e-8).sum())} vs {runs.loc[model, 'holdings']}")
    return checks


def write_csv(path, rows, fieldnames=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fieldnames or list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


def write_checks(path, checks) -> None:
    lines = []
    for c in checks:
        lines.append(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['check']}"
                     + (f" :: {c['detail']}" if c["detail"] else ""))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
