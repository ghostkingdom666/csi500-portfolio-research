# 任务3 评价契约（v1.0，2026-11-20 冻结）

本文件是第 3 人与第 2、4 人之间的接口约定。**字段、口径与预登记参数冻结后不得单方修改**；新增成本、调仓或滚动窗口必须另开情景并标新版本。实现见 `evalcore.py`、`benchmarks.py`、`frozen.py`、`run.py`，验证见 `test_eval.py` 与 `verify.py`。

## 1 输入契约

### 1.1 冻结数据（第 1 人交付，只读）

| 文件 | 本任务使用的字段 | 校验 |
|---|---|---|
| `periods.json` | `data_version`、`estimation_start/end`、`decision_date`、`proposed_entry_date`、`proposed_first_return_date`、`test_start/end` | 与 manifest 版本一致 |
| `model_returns_wide.csv` | 首列 `date`，其余 100 列按 `model_assets.csv` 顺序 | 列序、唯一性、有限值、行数=官方开市日 |
| `model_assets.csv` | `asset_id`、`name`、`selected`、`liquidity_rank` | 顺序即收益矩阵列序 |
| `calendar.csv` | `date`、`is_open`、`is_anchor` | 建仓日为首个 7 月开市日，首收益日为其下一开市日 |
| `missing_mask.csv` | 建仓日与检验期缺失标记 | 检验期缺失格数必须为 0 |
| `index_daily.csv` | `daily_simple_return` | 仅作市场参照，价格指数 |

六个必需文件的 SHA256 全部对照 `manifest.json` 校验；任一不符即停止评价。

### 1.2 第 2 人交付（只读）

`task02/results/weights.csv`：

| 字段 | 规则 |
|---|---|
| `model_id` | 方法标识；本任务按 `runs.csv` 顺序逐行评价 |
| `asset_id` | 必须与 `model_returns_wide.csv` 列名一致，按标签对齐 |
| `invested_weight` | 投入比例：非负、和为 1（容差 1e-8）；零权重行必须保留 |
| `raw_coefficient` | **不得进入评价**；评价入口显式拒收 |
| `status` | `optimal` / `infeasible` / `fail` / `zero_solution` / `optimal_inaccurate`；非 optimal 保留行、不出净值 |

`task02/results/runs.csv`：提供 `T/N`、`eta`、`lambda_value`、`sigma`、`gamma`、`cap`、`K`、`objective`、`solver`、`seconds`、KKT 残差与状态，用于 `comparison.csv` 关联，不参与净值计算。

`task02/results/manifest.json`：提供 `target`（θ_s、θ_a、r_c）与 `lambda_max`，由本任务独立重算复核。

## 2 评价口径

- 建仓：`periods.proposed_entry_date`（2026-07-01）收盘，`V0 = 1`，建仓日不计收益；
- 首笔收益：`proposed_first_return_date`（2026-07-02）起，收盘到收盘；
- 持有：一次建仓后持有，权重随价格漂移，不每日恢复比例：

$$
V_t=\sum_i u_i\prod_{s\le t}(1+r_{s,i}),\qquad V_{\text{entry}}=1 .
$$

- 累计收益 `V_end/V_0-1`；年化收益 `(V_end/V_0)^(252/n)-1`；年化波动 `sqrt(252)*std(r,ddof=1)`；夏普 `sqrt(252)*mean(r)/std(r)`，`rf=0`（研究假设）；最大回撤的峰值序列包含初始净值 1；
- 零波动或样本不足时 Sharpe 输出空值（不填 0，不参与排名）；
- 经验 VaR 取损失升序第 `ceil(αT)` 个；经验 CVaR 用最小化定义 `min_ζ ζ + 1/(T(1-α)) Σ(L_t-ζ)^+`，不用"只平均严格超过 VaR 的点"；
- 成本：零成本假设，未模拟费用与冲击；不声称实现了全部交易限制。

## 3 风险量对照（不得混用）

| 符号 | 归属 | 含义 | 是否硬约束 |
|---|---|---|---|
| `sigma=0.04` | 任务2 | 原模型目标向量 `r_c` 的日频波动参数 | 否。原 ℓ1 模型没有显式 `u'Σu≤σ²`，实际波动不受其保证 |
| `sigma_max=0.03` | 任务3 均值—方差基准 | `max μ'u s.t. sqrt(u'Σu) ≤ σ_max` | 是（估计期口径） |
| `loss_max=0.03`（95%）/`0.045`（99%） | 任务3 CVaR / VaR | 日损失口径的 CVaR / VaR 限额 | 是（估计期口径） |
| `gamma=15, cap=0.25, K=5` | 任务2 拓展 | 风险厌恶系数、单股上限、最大持仓数 | 是，仅用于 20 只配对实验 |

**三者含义不同，数值相近也不得称为等风险。** 另外，估计期满足 `sigma_max=3%` 的组合，检验期实现年化波动可达 70% 量级，说明约束只对估计期样本成立，不能外推。

`comparison.csv` 的 `sigma` 列随 `source` 含义不同：任务2 行是目标构造参数（0.04），任务3 `mv` 行是风险硬上限（0.03）。跨来源读取时必须同时看 `source` 与 `role`，不得把两列当作同一参数比较。

## 4 预登记参数

| 参数 | 取值 | 依据 |
|---|---|---|
| 评价期 | 2026-07-01 收盘建仓；2026-07-02 至 2026-11-10，88 个收益日 | 第 1 人 `periods.json` |
| D / ddof / rf | 252 / 1 / 0 | 任务卡建议 |
| MV `sigma_max` | 0.03（日频） | 估计期等权日波动 1.93% 的约 1.55 倍，约束有约束力且可行 |
| CVaR 主情景 | α=0.95，`loss_max=0.03` | 估计期等权 CVaR95 为 3.14%，限额略紧 |
| CVaR 敏感性 | α=0.99，`loss_max=0.045` | 估计期等权 CVaR99 为 4.29% |
| VaR 启发式 | α=0.95，`loss_max=0.03`，budget=3000，seeds={0,1,2} | 候选含等权、100 个单资产边界点与 Dirichlet 随机点；三个种子全部保留 |
| 候选预算与种子 | 见 `protocol.json` | 在首次评价检验期之前登记 |

参数只使用估计期统计与任务卡建议量级确定，**未查看任何检验期收益**；冻结后不得按检验期表现调整。

## 5 输出契约

`task03/results/` 下：

| 文件 | 内容 |
|---|---|
| `metrics.csv` | 每模型一行：`model_id, role, source, status, solver, n_assets, n_days, cumulative_return, annualized_return, annualized_vol, sharpe, max_drawdown, final_nav, max_weight, daily_mean, daily_sd, seconds, data_version, entry_date, test_end, note` |
| `comparison.csv` | 指标表与 `runs.csv` 参数合并；失败行保留，`status` 不为空 |
| `nav.csv` | 宽表：`date` × `model_id`，首行建仓日 NAV=1 |
| `daily_returns.csv` | 长表：`date, model_id, gross_return` |
| `weights_benchmarks.csv` | 任务3 基准权重，字段与第 2 人 `weights.csv` 相同（`raw_coefficient` 留空） |
| `run_manifest.json` | 输入/代码 SHA256、协议、环境版本、输出哈希 |
| `checks.txt` | 76 项契约检查 |
| `verification.txt` | 51 项独立复算 |
| `figures/` | `nav_main.png`、`nav_all.png`、`drawdown_main.png`、`metrics_main.png` |

`role` 取值：`main`（五方法主比较）、`sensitivity`（η 网格、α=0.99、VaR 种子）、`extension`（20 只数量约束配对）、`reference`（中证500价格指数，不参与排名）。

**复现性**：`nav.csv`、`daily_returns.csv`、`weights_benchmarks.csv`、`checks.txt` 的内容在相同环境下逐字节稳定；`metrics.csv`、`comparison.csv` 含求解墙钟时间 `seconds`，文件哈希随机器变化。`run_manifest.json` 的 `outputs` 是以当次运行文件计算的哈希，不承诺跨运行一致；对外引用数字时应同时记录 `generated_at`。

## 6 复核清单

第 2 人签认本评价（对应任务卡 §2.6、§3.3）：

- [ ] 权重按 `asset_id` 对齐，零权重行保留，未用 `raw_coefficient` 计算净值；
- [ ] 建仓日 2026-07-01、首收益日 2026-07-02、截止 2026-11-10 与数据包一致；
- [ ] 买入持有口径：两资产手算 1 → 1.05 → 1.10 通过；未误用固定权重逐日复利（那会得到 1.1025）；
- [ ] 指标公式（累计、年化、波动、夏普、最大回撤）与任务卡 §3.2 一致；
- [ ] 失败/不可行模型保留行，未从表中消失；
- [ ] 抽算一个模型（建议 `l1_eta_0.01`）的净值与指标，与 `metrics.csv` 一致；
- [ ] 确认任务3 未改动第 2 人权重：`task02/results/weights.csv` SHA256 =
      `fbe0d3e385cc1dc993c28a82610bd7ab8ae7d2df068591f545f2ab39a8a75bb8`。

第 4 人使用本交付：

- 主表用 `role=main` 的 5 行加指数参照行；`extension` 两行是 20 只小池，**不得与 100 只主表混排比较**；
- 每张图引用 `figures/` 或由 `make_charts.py` 从 `nav.csv` 重生成，不截屏；
- 报告注明 88 个收益日、单窗口、零成本、当前成分回溯样本（幸存者偏差与成分前视）。

## 7 未覆盖

交易成本、调仓频率、滚动窗口、真实可交易性（涨跌停/停牌后复牌规则）、历史成分名单、严格总回报、参数组会确认。以上均不得在报告中写成已完成。

复权口径遗留问题：`close_adj/close_raw` 比值在样本期内频繁小幅变动，并非只在除权日跳变，与严格后复权因子不一致。本评价沿用冻结的 `close_adj`（第 1 人数据口径）；用原始价格重算的敏感性显示累计收益低 0.9–3.6 个百分点、方法排序不变。该问题由第 1 人核验，报告中如需讨论复权影响应引用此敏感性，不得自行更换收益口径重跑主表。
