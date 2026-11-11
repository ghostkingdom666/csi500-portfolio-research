# 01任务卡：中证500数据交付

本目录只完成数据收集、清洗、质量核查与交接。请先阅读 `data/质量报告.md`，再阅读本说明；实际程序验收结果为 `data/acceptance.json`。

**版本：csi500_20261111T020022_current_v1。样本性质：当前成分股回溯样本。** 官方名单参考日2026-11-10，未核实2026-06-30的完整历史名单，按用户批准降级；不能声称无成分前视或存活偏差。

## 本版实际数据

- 收集500只，全部取得原始日线和后复权日线；各223,201条，额外保留2024-12-31锚点。
- 研究期2025-01-02至2026-11-10，共448个交易日；估计期截至2026-06-30，共359个交易日。
- 估计期合格469只，按估计期平均日成交额选择100只；T=359、N=100，满足T>N+2的必要维度条件。未检查02任务的协方差和目标估计式。
- 全500只收益表有1,303个估计期缺失单元，源于上市历史不足、停牌和紧接缺价后的收益；保留空值，不填零。检验期全池与100只子集均无缺失收益。
- 最新日499只具有原始交易条，另1只中国稀土有正式停牌状态证据，延续已核实估值；完整估值截至2026-11-10。全500只均有原始交易条的最近日期另记为2026-10-30。
- 请求截止2026-11-11；实际抓取时为香港时间当日凌晨，11月11日尚未开市。所有返回日期均来自真实响应，不把请求截止当成行情截止。
- 建议决策日6月30日、7月1日收盘建仓、7月2日起计收益。数据交付不代表实际建仓、可交易性签认或回测结果。

## 文件导航

|位置|用途|
|---|---|
|`data/prices_long.csv`|按正式日历对齐的500只行情、状态与估值，含锚点；唯一键date+asset_id|
|`data/returns_wide.csv` / `missing_mask.csv`|完整收集池收益与缺失掩码，448×500；日期为首列|
|`data/assets.csv` / `selection_audit.csv`|500只名单、来源、上市信息、子集标记及全部筛选理由|
|`data/model_assets.csv` / `model_returns_wide.csv`|建议100只子集；资产顺序与全池名单中的选中顺序一致|
|`data/calendar.csv` / `calendar_all_days.csv`|正式开市日及含周末、休市日的完整日历|
|`data/missing_details.csv` / `status_details.csv`|缺失原因及停牌、上市前等状态证据路径|
|`data/three_stock_sample.csv`|白云机场、中国宝安、特锐德的真实完整双价样例|
|`data/corporate_action_checks.csv`|三股共6次研究期除权窗口及手算参照|
|`evidence/*_adjustment_comparison.csv`|两源逐日复权对照，不用于替换主源|
|`data/index_daily.csv`|辅助中证500价格指数日线；不是全收益指数或额外模型|
|`data/cleaning_audit.csv` / `off_calendar_records.csv` / `extreme_return_review.csv`|去重、冲突、日历外记录及异常涨跌审查；不裁剪收益|
|`data/periods.json` / `manifest.json`|日期约定、版本、来源、单位、质量限制、数据与代码哈希|
|`data/raw_inventory.csv`|全部原始快照及元信息文件的SHA256|
|`raw/`|官方名单、休市安排、双价、逐股元信息、停复牌、公司行动和公告原始响应|
|`requests.jsonl`|URL、请求参数、抓取时间、重试、失败与响应哈希|
|`sample_validation.json` / `enrichment_results.json`|真实样例核验及补充证据下载结果|
|`src/` / `pipeline.py` / `requirements.txt`|可运行处理入口与固定依赖|

CSV为UTF-8 BOM、逗号分隔；日期ISO格式；收益为小数，空字段是缺失。用Python读取代码字段时指定字符串，避免丢前导零。浮点输出17位有效数字，文件顺序固定，哈希为SHA256。

## 运行入口

依赖为Python 3.12及requirements.txt固定版本。首次在新机器准备环境：

```powershell
Set-Location D:\Project\data_delivery_01
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe pipeline.py rebuild --output rebuilt_data
.\.venv\Scripts\python.exe pipeline.py validate --output rebuilt_data
```

环境安装完成后，rebuild和validate完全离线，不调用行情或状态接口。validate会在包内.validation_cache临时目录独立重建一次，比较所有交付数据文件及manifest的字节哈希，然后生成acceptance.json，成功后自动清理临时副本。它还执行标签、矩阵、锚点、量纲、缺失传播、停牌、重复冲突及检验期扰动不影响选股等检查。验收脚本失败会返回非零退出码，不生成伪成功记录。

本机已使用Codex捆绑Python 3.12.14及numpy/pandas/openpyxl/pypdf；xlrd和baostock放在vendor目录。它们的来源与许可证文件随包保留。无需使用原论文中的校内数据库连接或凭据。

在线获取入口为 `python pipeline.py acquire`。**本快照已经冻结，入口会拒绝向本版本继续抓取。** 新抓取需新建一个空版本目录，只复制pipeline.py、src、requirements.txt和vendor（或重新安装依赖），再运行acquire；不要复制旧raw、data、snapshot_config.json、日志及样例验证结果。请求与配置在新目录重新生成，不能在不同版本间混用数据。acquire可在尚未冻结的新目录断点续跑，成功原始响应不会覆盖。

## 下游读取示例

```python
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd

root = Path(r"D:\Project\data_delivery_01\data")
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
for name, meta in manifest["files"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == meta["sha256"]
assets = pd.read_csv(root / "assets.csv", dtype={"asset_id": str, "source_code": str})
ret = pd.read_csv(root / "returns_wide.csv")
assert ret.columns[1:].tolist() == assets.asset_id.tolist()
selected = assets.loc[assets.selected == 1, "asset_id"].tolist()
est = ret.loc[ret.date.between("2025-01-01", "2026-06-30"), ["date"] + selected]
R = est[selected].to_numpy(dtype=float)
assert R.shape == (359, 100) and np.isfinite(R).all()
```

全池收益不能直接把空值送优化器；主模型建议读取子集。模型输入只用估计期，7月1日收盘只用于建仓估值，7月2日收益才用于后续持有期记账。数据负责人未实现任何组合模型、净值或绩效指标。

## 来源、核验与限制

1. 名单：[中证指数000905官方成分表](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/cons/000905cons.xls)。原表日期20261110，500只、含交易所。公告日、可得日、生效日无法由表确定，留空；抓取时间独立记录。历史接口连接错误见raw/references/baostock_history_probe.json，不能把该尝试当成成功的历史证据。
2. 价格：东方财富push2his，fqt=0与fqt=2；原始字段及单位依据[AKShare官方发行文档](https://akshare.akfamily.xyz/data/stock/stock_quote.html)与发行包源码。未直接运行AKShare，记录其1.20.37文档/源码只作为端点和单位依据；代码通过urllib请求原始供应商接口。成交量原单位手，乘100转股；成交额元。腾讯fqkline仅作抽样原价、成交量、复权对照。
3. 复权：仅使用东方财富后复权收盘构造收益，标“供应商复权收益代理”。两源原始收盘和成交量三股全449条一致；中国宝安的复权日收益最大差约0.0011675（约11.68基点），超过舍入解释，未证明两源等价。白云机场和特锐德满足两源公布精度的舍入界限。所有差异均留存，未切换或拼接价格源；主源字段可用与跨供应商复权等价是两项独立检查，后一项保留为限制。
4. 日历：[上交所2025年休市安排](https://www.sse.com.cn/disclosure/dealinstruc/closed/c/c_20241223_10767110.shtml)、[2026年休市安排](https://www.sse.com.cn/disclosure/dealinstruc/closed/)。按周末加公告休市区间生成沪深共同开市日，并与中证500真实指数日线日期集合交叉核对。原页面随包冻结；不是将普通工作日当成交易日。
5. 状态：东方财富RPT_CUSTOM_SUSPEND_DATA_INTERFACE及逐股f189上市日期。发现历史筛选响应含未来记录时，仍必须按每条实际起止日期过滤，并获取完整分页。湖南黄金2026年1月停牌被该接口遗漏，已用[原始停牌公告](https://static.cninfo.com.cn/finalpage/2026-01-12/1224929055.PDF)和[深交所托管复牌公告](https://disc.static.szse.cn/disc/disk03/finalpage/2026-01-26/ed783f50-c513-4f81-9713-20bde02af77c.PDF)补证；PDF、提取文本、核实区间和哈希保留于raw/status_announcements。估计期仍不补价，因此核实状态不改变子集排名。
6. 公司行动：东方财富RPT_SHAREBONUS_DET。三股除权窗口和全部存在历史非上市缺口的资产均获取事件列表；未声称核验500只全部事件或严格总回报。检验期19行停牌估值延续均有逐日状态和公司行动列表依据。
7. 授权与共享：以上接口可公开访问，无需账户或付费token；公开访问并不自动授予公开再分发权，本次未核实供应商额外再分发许可。包供本项目内部课程核验；公开发布前核实相关条款。数据说明保留所有来源，可用脚本重新获取合法快照，不共享原稿数据库凭据。

历史池偏差、供应商复权口径差异、退市历史未独立核实及真实组员签认待完成，都保留为限制。程序验收通过仅说明本次交付文件和处理行为符合所记录口径，不代表这些限制被消除。
