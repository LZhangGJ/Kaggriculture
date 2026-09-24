# A06 R12 动态规则方案：源码与对战交接（2026-09-24）

这里保存 13 份 GPT Pro 交付包中**实际入场的主 Agent**、冻结公开前十的可运行入口、官方 Kaggriculture 1.32.7 Python 规则、逐局结果和复核资料。原交付压缩包共约 170 MB，未重复塞入 Git；本目录保留提取后的运行文件和 SHA-256 来源清单。`agents/<id>/main.py` 必须与同目录 `policy/` 一起使用。随附的 `.so` 是本次评测所用的 Linux x86-64 二进制，因精确复现需要而保留；系统不兼容时可从源码重编，但新二进制需重新验收。

先读 [每个方案改了什么](VARIANTS_ZH.md)，再读 [对战结果与解释](RESULTS_ZH.md)。`evaluation/internal13`、`evaluation/cashflow_public10`、`evaluation/three_public10` 分别保留原报告、冻结协议、汇总和脚本。`games.jsonl.gz` 是完整逐局收据，解压后原文 SHA-256 见 `PACKAGE_MANIFEST.json`。原脚本中的 `/mnt/e/...` 路径是当时的运行现场，仅作方法和来源证据；在别的机器上需改为本目录的相对路径。`python verify_bundle.py` 可校验全部交接文件、13 个 Agent 与 10 个公开对手的 SHA-256，以及三组逐局记录的局数和错误数。在兼容的 Linux x86-64/WSL 环境，可用 `python reproduce_one.py --a r14_liquidity --b n14_prvsiyan --seed 1805214525 --seat-a 0` 跑一局实时对战；这局应得到我方 88,091、对手 89,947，与逐局收据完全一致。换成协议中的种子、座位和对手 ID 可逐局复核。这个单局入口不代表已重跑全部面板。

**这批实验的主要发现：** Cashflow 在 13 方案内部循环赛得分率 96.33%，但对冻结公开前十仅 29.5%；Liquidity 在内部仅 19.75%，对公开十方案却达 84.0%。因此内部名次不能代替外部对手面板。Liquidity 对其中两名强对手仅 32/100 和 29/100，整体也未达到 90%。公开十条入口只有九份不同的 `main.py`，同源性和样本依赖均需保留在判断里。

评测双方都按当前观察实时决策，50 个共同种子、双方换位，官方规则执行到每局 719 次状态转换；结果不是 Kaggle 线上提交或沙箱超时验收。各交付包文件名前的百分数是其作者在**其他面板**中的历史数字，不是本目录实验成绩。

目录用途：

- `agents/`：13 个冻结主 Agent 的源码、配置、测试和匹配原生库；`evaluation/internal13/POOL.json` 记载原交付包名与哈希。
- `opponents/`：这次公开十方案评测所用的冻结 `main.py` 和原有许可/署名文件；来源、版本和哈希见两个 public10 的 `PROTOCOL.json`。
- `referee/`：用于本地比赛的官方 1.32.7 规则与外围 Python 驱动。官方规则字节保持冻结版本；外围 `policy_host.py` 仅将导入路径改为本目录布局。外围驱动不等于 Kaggle 线上沙箱。
- `evaluation/`：完整结果和历史运行脚本。每个 `games.jsonl.gz` 都能还原为原始 JSONL。

没有按文件名、对手身份或评测种子给任何 Agent 赛时切换版本。本目录只归档并解释已完成实验，不宣称这 13 份均可直接作为最终提交。
