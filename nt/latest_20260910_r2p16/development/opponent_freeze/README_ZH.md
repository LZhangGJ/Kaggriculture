# R2 新公开对手池：7 个冻结 Python Agent

冻结日期：2026-09-10（JST）。本目录只完成下载、静态提取和导入验收；**没有在此子任务执行对战，也未修改 R2**。

所有 7 本 Notebook 均重新通过官方 `/kernels/pull` 单线程下载，版本与此前扫描一致，原始 Notebook SHA256 也全部一致。
不以页面的历史最好分数作为这些版本的实力证明。最近运行时间沿用目录 API 快照；pull 元数据的 `lastRunTime` 可能不同，故原样保存但不替代目录排序。

## 可直接对战的入口

| 本地 ID | 来源 | Notebook 版本 / 代码 | 可加载入口 |
|---|---|---|---|
| soil_v219g | [The Soil Remembers Rain](https://www.kaggle.com/code/prvsiyan/kaggriculture-frontier-the-soil-remembers-rain) | 30 / V219G | `soil_v219g/working/main.py` |
| moon_v215 | [The Moon Counts Melons](https://www.kaggle.com/code/prvsiyan/kaggriculture-frontier-the-moon-counts-melons) | 33 / V215 | `moon_v215/working/main.py` |
| flexon_v5 | [Most Powerfull Route](https://www.kaggle.com/code/flexonafft/kaggriculture-most-powerfull-route) | 5 / E182 128-search | `flexon_v5/working/main.py` |
| market_smart_v8 | [Market Smart Farming](https://www.kaggle.com/code/tetsutani/market-smart-farming-kaggriculture) | 8 / R31 | `market_smart_v8/working/main.py` |
| nagatakengo_v70 | [Nagatakengo Kaggriculture](https://www.kaggle.com/code/nagatakengo/kaggriculture) | 70 / V5.9 | `nagatakengo_v70/working/main.py` |
| aurax_reactive_v1 | [Reactive Router](https://www.kaggle.com/code/aurax7/kaggriculture-reactive-router) | 1 | `aurax_reactive_v1/working/main.py` |
| thomas_955_v2 | [95.5% Replay Routing](https://www.kaggle.com/code/thomastschinkel/kaggriculture-95-5-win-rate-via-replay-routing) | 2 / v5/2 | `thomas_955_v2/working/main.py` |

Aurax 此处是 **reactive-router**，不是此前已测的 **shop-router-reactive-v2**。本目录不包含原有 4 个对手，应由上层评测 manifest 合并。

## 加载要求

- Python 标准库即可加载所有入口，不需要运行 Notebook、不需要联网安装依赖。
- 始终调用 `agent(observation, configuration)`。Nagatakengo 的第二参数无默认值，必须传入；不为统一接口而修改其源码。
- Flexon、Market Smart 的 `working` 是完整多文件包，不能只拷走 `main.py`。
- 这两个包使用 `unit_model`、`router_parent`、`_e182_shop_terminal` 等通用模块名；必须隔离全局模块，不能在同一导入缓存中混用不同包。
- 推荐每局独立进程；若复用进程，每局重新载入所有包内模块，清理相应 `sys.modules` 和临时 `sys.path` 条目，防止跨局状态污染。
- 其他 5 个是独立单文件，但同样有持久状态，仍建议每局重新导入。
- 源码里有 step=0/reset 检查，**不等于已经证明完整跨局隔离**。本轮只做导入 smoke。

## 提取方式及完整性

- Soil / Moon：通过 AST 读取唯一 `AGENT_SOURCE` 字符串，不执行外围的安装、图表、样例比赛。
- Flexon：静态提取各个 `%%writefile` 单元；literal SETTINGS 写回同一 JSON 格式；安全解压内嵌路线 JSON。原作者全部 runtime member SHA256 核验通过。
- Market Smart：仅解码内嵌 tar；逐一校验原作者 archive/member SHA256；只允许普通文件、大小上限和受限目录内路径，拒绝链接/路径穿越。
- Nagatakengo / Aurax / Thomas：静态提取 `%%writefile main.py`；按 IPython 行为补缺失的末尾 LF；Aurax 核对作者期待的 main.py SHA256 通过。
- 原作者注释、许可证文本和策略逻辑保留不变。

Nagatakengo 元数据附带 `nagatakengo/kaggriculture-data-class`，但实际 main.py 没有 import、文件读取或数据集引用；本任务不运行整本 Notebook，因此这个未用数据附件不下载。其他 6 个没有 dataset/model/kernel 输入依赖。全部共用已有官方 1.32.7 环境作裁判，不另下竞赛大数据。

## 已通过及未通过的门

已通过：7/7 原始下载、7/7 与扫描版本和源码一致、7/7 Python 语法、7/7 隔离导入、7/7 两参数接口绑定。
导入 smoke 强制禁止网络、子进程、文件写入；仅创建函数/状态与读取内嵌或包内数据。

尚未在本子任务验证：完整 719 步运行、真实对战状态 reset、每步执行耗时、200 局胜率、任何强度声明。
不能把作者的 95.5% 自测当作对 R2 的结果。

## 复核命令

项目根目录 PowerShell：

```powershell
& '.\.venv\python.exe' -X utf8 'research\public_opponents_r2_20260910\verify_packages.py'
```

`freeze_public.py download <id>` 遇到已有 receipt 会跳过，不覆盖冻结版本。
`freeze_public.py extract <id>` 可从保存的原 Notebook 幂等提取；遇到不同内容拒绝覆盖。
所有文件哈希、绝对入口、元数据和 smoke 明细见 `MANIFEST.json` 及各包 `manifest.json`。

这里的工具脚本仅负责下载/静态打包。比赛统一使用主线程评测 harness，不执行公开 Notebook 的整本流程。
