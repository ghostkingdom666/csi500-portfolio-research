"""任务3 图表：净值、回撤、指标柱状图。数据全部来自 results/，可重复生成。

运行（仓库根目录）：python task03/make_charts.py
输出：task03/results/figures/{nav_main,drawdown_main,metrics_main,nav_all}.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
FIG = OUT / "figures"

MAIN = ["l1_eta_0.01", "equal", "mv", "cvar_a95", "var_a95_s0"]
LABELS = {
    "l1_eta_0.01": "Non-negative L1 (eta=0.01)",
    "equal": "Equal weight",
    "mv": "Mean-variance (sigma<=3%)",
    "cvar_a95": "CVaR 95% (L<=3%)",
    "var_a95_s0": "VaR heuristic 95% (seed 0)",
    "csi500_index_price": "CSI 500 price index (reference)",
}


def load():
    nav = pd.read_csv(OUT / "nav.csv", index_col=0, parse_dates=True)
    metrics = pd.read_csv(OUT / "metrics.csv").set_index("model_id")
    return nav, metrics


def plot_nav(nav, models, path, title):
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    for m in models:
        ax.plot(nav.index, nav[m], label=LABELS.get(m, m), linewidth=1.6)
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("NAV (entry close = 1)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_drawdown(nav, models, path):
    dd = nav[models] / nav[models].cummax() - 1.0
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    for m in models:
        ax.plot(dd.index, dd[m], label=LABELS.get(m, m), linewidth=1.6)
    ax.set_title("Drawdown from running peak (peak includes entry NAV=1)", fontsize=12)
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_metrics(metrics, models, path):
    sub = metrics.loc[models]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    axes[0].bar([LABELS.get(m, m) for m in models],
                sub["cumulative_return"] * 100, color="#3b6ea5")
    axes[0].set_title("Cumulative return, 2026-07-02 to 2026-11-10 (%)")
    axes[0].tick_params(axis="x", labelrotation=25, labelsize=7)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[1].bar([LABELS.get(m, m) for m in models], sub["sharpe"], color="#a5643b")
    axes[1].set_title("Annualized Sharpe (rf = 0)")
    axes[1].tick_params(axis="x", labelrotation=25, labelsize=7)
    axes[1].axhline(0, color="black", linewidth=0.8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    nav, metrics = load()
    plot_nav(nav, MAIN + ["csi500_index_price"], FIG / "nav_main.png",
             "Out-of-sample NAV: entry 2026-07-01 close, hold to 2026-11-10")
    plot_nav(nav, list(nav.columns), FIG / "nav_all.png",
             "All evaluated models (task02 weights + task03 benchmarks + index)")
    plot_drawdown(nav, MAIN + ["csi500_index_price"], FIG / "drawdown_main.png")
    plot_metrics(metrics, MAIN, FIG / "metrics_main.png")
    print(f"charts written to {FIG}")


if __name__ == "__main__":
    main()
