# 任务3 评价契约（v1.2，2026-11-26 更正与固定基准扩展）

本文件是第 3 人与第 2、4 人之间的接口约定。原 v1.0 于 2026-11-20 冻结；2026-11-26 按用户授权更正风险标签、VaR 候选与状态，恢复随机十折 ℓ1 主模型，并回溯新增四个固定简单基准。限额保持原值，修订不是新事前登记；新增成本、调仓或滚动窗口仍需另开情景并标新版本。实现见 `evalcore.py`、`benchmarks.py`、`simple_baselines.py`、`frozen.py`、`run.py`，验证见 `test_eval.py`、`test_simple_baselines.py` 与 `verify.py`。

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
| `status` | 上游任意非空状态原样保留（如 `optimal`、`zero_solution`、`failed_kkt`、`solver_failure`、`lambda_max_zero`、`feasible_timelimit`、`infeasible`、`failed_residual`）；`optimal` 与 `feasible_heuristic` 进入权重校验与净值，其他状态保留空业绩行、不阻断整批 |

`task02/results/runs.csv`：提供 `T/N`、`eta`、`lambda_value`、`sigma`、`gamma`、`cap`、`K`、`objective`、`solver`、`seconds`、KKT 残差与状态，用于 `comparison.csv` 关联，不参与净值计算。

`task02/results/manifest.json`：提供 `target`（θ_s、θ_a、r_c）与 `lambda_max`，由本任务独立重算复核；`outputs` 提供当前 `weights.csv`、`runs.csv` 和 CV 诊断文件的 SHA256，复核不得硬编码旧权重哈希。

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

**三者含义不同，数值相近也不得称为等风险。** 估计期满足 `sigma_max=3%` 不保证检验期风险，不能把样本约束外推为未来风险保证。

`comparison.csv` 的 `sigma` 列随 `source` 含义不同：任务2 行是目标构造参数（0.04），任务3 `mv` 行是风险硬上限（0.03）。跨来源读取时必须同时看 `source` 与 `role`，不得把两列当作同一参数比较。

## 4 原有参数与更正记录

| 参数 | 取值 | 依据 |
|---|---|---|
| 评价期 | 2026-07-01 收盘建仓；2026-07-02 至 2026-11-10，88 个收益日 | 第 1 人 `periods.json` |
| D / ddof / rf | 252 / 1 / 0 | 任务卡建议 |
| MV `sigma_max` | 0.03（日频） | 估计期等权日波动 1.93% 的约 1.55 倍，约束有约束力且可行 |
| CVaR 主情景 | α=0.95，`loss_max=0.03` | 原限额保留；等权 VaR95=3.1404997%、CVaR95=4.2619536%（原说明误标） |
| CVaR 敏感性 | α=0.99，`loss_max=0.045` | 原限额保留；等权 VaR99=4.2893239%、CVaR99=6.8470454%（原说明误标） |
| VaR 启发式 | α=0.95，`loss_max=0.03`，budget=3000，seeds={0,1,2} | 总预算内含等权、100 个单资产边界点、可用 CVaR95/99 解与 Dirichlet(1) 随机点；两组 warm start 均有时随机点为 2897；三个种子全部保留 |
| 候选预算与种子 | 见 `protocol.json` | 保留原预算与种子；2026-11-26 追加确定性 warm start，明确为回溯修订 |

原协议将等权 VaR95 3.14%、VaR99 4.29% 写成 CVaR，原有 CVaR 限额依据因此不成立。2026-11-26 修订保留已有风险限额并按估计期数据重算 `risk_statistics.csv`，不按检验期表现调参，也不重新宣称事前登记。

VaR 可行候选按估计期均值择优：成功仅标 `feasible_heuristic`；搜不到可行点标 `no_feasible_candidate`，不能据此证明模型不可行。manifest 记录 `budget`、`total_candidates`、`random_budget`、种子、可行数、选中候选 ID/序号，以及每个 warm start 的 ID、序号、按资产权重、风险、目标与可行性。可行 warm start 的目标构成候选择优的下界。

### 4.1 回溯新增的固定简单基准

四者仅使用2025-01-02至2026-06-30的359行估计期数据和同一100股票池；函数不读取文件或检验期。新增参数沿用既定baseline建议，2026-11-26按用户授权固定，不能宣称事前登记，也不得按检验结果调参。

| model_id | 估计及配置规则 |
|---|---|
| `gmv` | 完整估计期样本协方差（ddof=1），`min u'Σu s.t. u>=0, sum(u)=1`，无收益目标；SLSQP目标按最大资产方差缩放，记录实际方差目标与缩放KKT诊断 |
| `inverse_volatility` | 完整估计期样本标准差（ddof=1），`u_i=(1/s_i)/Σ(1/s_j)`；任一零或非有限波动明确失败，不静默剔除 |
| `momentum_top20` | 过去126个收益日剔除最近21日，即 `prod(1+R[-126:-21])-1`；按分数降序及 `asset_id` 字典升序，前20只等权 |
| `low_volatility_top20` | 最近60个收益日的样本标准差（ddof=1）；按分数升序及 `asset_id` 字典升序，前20只等权 |

动量信号的105日区间为2025-12-18至2026-05-29，排除2026-06-01至2026-06-30的21日；低波动区间为2026-04-01至2026-06-30。窗口按收益行数取，不按自然日取。动量是单次排序建仓，不含月度轮动。所有权重在估计结束时冻结，7月1日收盘建仓，统一88日持有与零成本。

输出 `(pd.Series | None, info)`；成功Series有全100个唯一资产标签（包括零权重）、非负且和为1。非法K、历史不足、非有限收益或收益≤−1抛出明确ValueError；零波动的波动率倒数返回 `None` 与 `status=fail`，失败不填净值。info记录 `status/solver/objective/objective_name/estimation/parameters/diagnostics`；选股方法和波动率倒数另记录全部资产 `scores`，选股方法另记录完整 `ranking/selected_assets/signal`。动量目标是选中资产信号收益均值，低波动目标是选中标准差均值，均不应误读成组合估计收益或组合方差；GMV和波动率倒数目标字段为组合样本方差（后者只作诊断）。

## 5 输出契约

`task03/results/` 下：

| 文件 | 内容 |
|---|---|
| `metrics.csv` | 每模型一行：`model_id, role, source, status, solver, n_assets, n_days, cumulative_return, annualized_return, annualized_vol, sharpe, max_drawdown, final_nav, max_weight, daily_mean, daily_sd, seconds, data_version, entry_date, test_end, note` |
| `comparison.csv` | 指标表与 `runs.csv` 参数合并；失败行保留，`status` 不为空 |
| `nav.csv` | 宽表：`date` × `model_id`，首行建仓日 NAV=1 |
| `daily_returns.csv` | 长表：`date, model_id, gross_return` |
| `weights_benchmarks.csv` | 任务3 基准权重，字段与第 2 人 `weights.csv` 相同（`raw_coefficient` 留空） |
| `risk_statistics.csv` | 估计期等权 VaR/CVaR95/99，含日频小数单位、区间与样本数 |
| `run_manifest.json` | 输入/代码 SHA256（含 `simple_baselines.py`）、协议、环境版本、输出哈希（含风险统计）、完整VaR搜索诊断与四新增基准的估计/信号/评分/选股/参数/KKT信息 |
| `checks.txt` | 当次契约检查（数量随模型与状态变化） |
| `verification.txt` | 当次独立复算，含风险统计、候选重放、状态、四基准规则与GMV KKT检查 |
| `figures/` | `nav_main.png`、`nav_all.png`、`drawdown_main.png`、`metrics_main.png`；新增 `nav_simple_baselines.png`、`drawdown_simple_baselines.png`、`metrics_simple_baselines.png` 专图 |

`role` 取值：`main`（`l1_cv`、`equal`、`mv`、`cvar_a95`、`var_a95_s0`、`gmv`、`inverse_volatility`、`momentum_top20`、`low_volatility_top20` 九方法主比较）、`sensitivity`（全部固定 η 网格，含原 η=0.01；α=0.99；其余 VaR 种子）、`extension`（20 只数量约束配对）、`reference`（中证500价格指数，不参与排名）。当前任务2交付下统一20总行，原有各行全部保留。两个Top20的 `comparison.csv:K=20`。

**复现性**：`nav.csv`、`daily_returns.csv`、`weights_benchmarks.csv`、`risk_statistics.csv`、`checks.txt` 的内容在相同环境下逐字节稳定；`metrics.csv`、`comparison.csv` 含求解墙钟时间 `seconds`，文件哈希随机器变化。`run_manifest.json` 的 `outputs` 是以当次运行文件计算的哈希，不承诺跨运行一致；对外引用数字时应同时记录 `generated_at`。

## 6 复核清单

第 2 人签认本评价（对应任务卡 §2.6、§3.3）：

- [ ] 权重按 `asset_id` 对齐，零权重行保留，未用 `raw_coefficient` 计算净值；
- [ ] 建仓日 2026-07-01、首收益日 2026-07-02、截止 2026-11-10 与数据包一致；
- [ ] 买入持有口径：两资产手算 1 → 1.05 → 1.10 通过；未误用固定权重逐日复利（那会得到 1.1025）；
- [ ] 指标公式（累计、年化、波动、夏普、最大回撤）与任务卡 §3.2 一致；
- [ ] 失败状态原样保留：`solver_failure`/`zero_solution` 等不触发整批停止，空权重行不报 float 错误；
- [ ] 抽算一个模型（建议 `l1_cv`）的净值与指标，与 `metrics.csv` 一致；
- [ ] 确认任务3 未改动第 2 人权重：当前 `task02/results/manifest.json` 的 `outputs` 中各 SHA256 与实际交付一致，不使用历史固定哈希。
- [ ] `risk_statistics.csv` 与独立重算一致；VaR 状态为 `feasible_heuristic` / `no_feasible_candidate`，warm start 权重可追踪且估计目标不劣于可行 warm start。
- [ ] 新增四基准全100行权重合法，估计/信号窗口和全部评分/排序/选股可追踪，四者与 `l1_cv` 的88个收益日期完全一致；独立复算波动率倒数和Top20规则，GMV验证非负全投资凸二次规划KKT（缩放残差≤1e-6），不以调用同一个生产函数代替复核。

第 4 人使用本交付：

- 主表用 `role=main` 的 9 行加指数参照行；`extension` 两行是 20 只小池，**不得与 100 只主表混排比较**；
- 每张图引用 `figures/` 或由 `make_charts.py` 从 `nav.csv` 重生成，不截屏；
- 报告注明 88 个收益日、单窗口、零成本、当前成分回溯样本（幸存者偏差与成分前视）。

## 7 未覆盖

交易成本、调仓频率、滚动窗口、真实可交易性（涨跌停/停牌后复牌规则）、历史成分名单、严格总回报、参数组会确认。以上均不得在报告中写成已完成。

复权口径遗留问题：`close_adj/close_raw` 比值在样本期内频繁小幅变动，并非只在除权日跳变，与严格后复权因子不一致。本评价沿用冻结的 `close_adj`（第 1 人数据口径）；历史原始价格敏感性属于修订前模型结果，保存在 archive 中，不能代替本次 CV 主模型与改进 VaR 的新敏感性结论。该问题由第 1 人核验，不得自行更换收益口径重跑主表。
