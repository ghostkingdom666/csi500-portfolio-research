"""任务3 公共评价核心：权重校验、一次建仓持有的净值、统一指标、经验 VaR/CVaR。

口径（见 task03/contract.md，冻结后不得单方修改）：
- 只接受 invested_weight（非负、和为 1），拒绝未归一化的 raw_coefficient；
- 建仓日收盘 V0=1，不产生收益；首笔收益为建仓后首个交易日收盘到收盘；
- 一次建仓后持有，权重随价格漂移：V_t = Σ_i u_i ∏_{s<=t}(1+r_s,i)；
- 收益为小数日简单收益，0.02 表示 2%；
- 年化 D=252，样本标准差 ddof=1，无风险利率 rf=0（研究假设）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class WeightError(ValueError):
    """权重不满足评价契约。"""


def validate_weights(weights, assets, tol: float = 1e-8) -> np.ndarray:
    """校验并返回按 assets 顺序排列的 numpy 权重向量。

    拒绝：重复标签、缺资产、多资产、NaN、负值、和不为 1。
    """
    s = pd.Series(weights, dtype=float)
    if not s.index.is_unique:
        dup = sorted(set(s.index[s.index.duplicated()]))
        raise WeightError(f"权重存在重复资产标签：{dup}")
    asset_set = set(assets)
    missing = [a for a in assets if a not in s.index]
    extra = [a for a in s.index if a not in asset_set]
    if missing or extra:
        raise WeightError(f"资产不匹配：缺少 {missing}，多出 {extra}")
    s = s.reindex(list(assets))
    values = s.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise WeightError("权重含 NaN 或无穷值")
    if (values < -tol).any():
        raise WeightError(f"权重存在负值：{values.min():.6g}")
    total = values.sum()
    if abs(total - 1.0) > tol:
        raise WeightError(f"权重和不为 1：{total:.12f}")
    return values


def reject_raw_coefficients(raw) -> None:
    """评价入口显式拒收原始系数，避免未归一化向量进入净值。"""
    if raw is not None:
        raise WeightError("评价只接受 invested_weight，不接受 raw_coefficient")


def buy_and_hold_nav(u, returns: pd.DataFrame, entry_date,
                     v0: float = 1.0) -> pd.Series:
    """一次建仓后持有：返回 index=[entry_date, 首个收益日, ...] 的净值序列。"""
    if returns.empty:
        raise ValueError("收益窗口为空")
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("收益矩阵含 NaN 或无穷值")
    w = np.asarray(u, dtype=float)
    if w.shape != (returns.shape[1],):
        raise ValueError(f"权重维度 {w.shape} 与收益列数 {returns.shape[1]} 不一致")
    growth = np.cumprod(1.0 + returns.to_numpy(dtype=float), axis=0)
    values = (growth * w[None, :]).sum(axis=1) * v0
    index = pd.Index([entry_date, *returns.index])
    return pd.Series(np.concatenate([[v0], values]), index=index, name="nav")


def rebalanced_nav(u, returns: pd.DataFrame, entry_date,
                   v0: float = 1.0) -> pd.Series:
    """每日恢复固定权重的对照口径，仅用于测试与诊断，不作为主结果。"""
    w = np.asarray(u, dtype=float)
    port = returns.to_numpy(dtype=float) @ w
    values = np.cumprod(1.0 + port) * v0
    index = pd.Index([entry_date, *returns.index])
    return pd.Series(np.concatenate([[v0], values]), index=index, name="nav_rebalanced")


def performance_metrics(nav: pd.Series, rf: float = 0.0, D: int = 252) -> dict:
    """由净值序列计算统一指标；nav[0]=V0。零波动或样本不足时 Sharpe 返回 None。"""
    v = np.asarray(nav, dtype=float)
    if v.size < 2:
        raise ValueError("净值序列至少需要 V0 和一个收益日")
    if not np.isfinite(v).all() or (v <= 0).any():
        raise ValueError("净值序列非正或含非有限值")
    n = v.size - 1
    daily = v[1:] / v[:-1] - 1.0
    total = v[-1] / v[0] - 1.0
    annual = (v[-1] / v[0]) ** (D / n) - 1.0
    sd = float(daily.std(ddof=1)) if n > 1 else float("nan")
    vol = float(np.sqrt(D) * sd) if np.isfinite(sd) else None
    mean = float(daily.mean())
    sharpe = float(np.sqrt(D) * (mean - rf / D) / sd) if (np.isfinite(sd) and sd > 0) else None
    peak = np.maximum.accumulate(v)
    mdd = float((v / peak - 1.0).min())
    return {
        "cumulative_return": float(total),
        "annualized_return": float(annual),
        "annualized_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "n_days": int(n),
        "final_nav": float(v[-1]),
        "daily_mean": mean,
        "daily_sd": sd if np.isfinite(sd) else None,
    }


def empirical_var_cvar(losses, alpha: float) -> tuple[float, float]:
    """经验 VaR / CVaR。

    VaR 取损失升序第 ceil(alpha*T) 个；CVaR 用最小化定义
    CVaR = min_zeta zeta + 1/(T(1-alpha)) * sum_t (L_t - zeta)^+。
    """
    x = np.sort(np.asarray(losses, dtype=float))
    if x.size == 0:
        raise ValueError("损失序列为空")
    if not (0 < alpha < 1):
        raise ValueError("alpha 必须在 (0,1)")
    if not np.isfinite(x).all():
        raise ValueError("损失序列含非有限值")
    k = int(np.ceil(alpha * x.size))
    var = float(x[k - 1])
    tail = float(np.maximum(0.0, x - var).sum())
    cvar = var + tail / (x.size * (1.0 - alpha))
    return var, cvar
