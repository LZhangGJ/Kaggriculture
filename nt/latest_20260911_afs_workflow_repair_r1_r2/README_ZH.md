# AFS R1 / R2 完整工作链与工人排班修复

这是 2026-09-11 验收通过的 AFS R1、AFS R2 执行完整性修复交接包。修复思想来自队友提交 `db109ce316151929622f2612223cc942dbb844de`，但以独立开关接入两个 AFS 版本，并经过固定 50 seed、双座位、11 个实时对手各 1,100 局复验。

## 结论

本次修复解决的是：原规划器已经决定并证明可完成的工作链，不再被后续工人重排静默漏掉。它不改变产业目标、购买、雇工、销售、市场估值或对手路由，也没有证据表明总体胜率上升。

| 版本 | 胜 / 平 / 负 | 胜率 | 修复效果 |
|---|---:|---:|---|
| AFS R1 | 856 / 4 / 240 | 77.82% | 1 局动作改变，现金和胜负完全不变；最终漏项为 0 |
| AFS R2 | 824 / 4 / 272 | 74.91% | 未解决断裂 40→0，deadline miss 2→0；平均现金 +1.20 |

R2 的已确认目标场景每座位现金增加 658、分差改善 1,473，但胜负未翻转。完整证据见 [`ACCEPTANCE_REPORT_ZH.md`](ACCEPTANCE_REPORT_ZH.md) 和 [`ACCEPTANCE.json`](ACCEPTANCE.json)。

## 开关

- `P16_AFS_WORKFLOW_REPAIR=0`：关闭修复，保持冻结策略行为。
- `P16_AFS_WORKFLOW_AUDIT=1`：仅记录断裂，不修改动作。
- `P16_AFS_WORKFLOW_REPAIR=1`：发现已承诺工作链断裂时恢复仍可执行的完整见证。

`r1/build/` 与 `r2/build/` 均保存 `workflow_off.so`、`workflow_audit.so`、`workflow_on.so` 和对应构建回执。每个版本 `policy/agent.so` 是已验收的 `workflow_on.so` 副本，供 `main.py` 默认加载。

## 文件索引

| 内容 | 位置 |
|---|---|
| R1 可调用入口 | [`r1/main.py`](r1/main.py) |
| R2 可调用入口 | [`r2/main.py`](r2/main.py) |
| 两版完整 C++ / Python 源码 | `r1/policy/`、`r2/policy/` |
| 工作链修复核心 | `r1/policy/executor/workflow_repair.hpp`、`r2/policy/executor/workflow_repair.hpp` |
| 三种开关二进制及构建回执 | `r1/build/`、`r2/build/` |
| 机制测试 | `r1/tests/test_workflow_repair.cpp`、`r2/tests/test_workflow_repair.cpp` |
| 1,100 局与单进程延迟证据 | `r1/evidence/`、`r2/evidence/` |
| 构建和面板工具 | `tools/` |

调试目录、Python 缓存、临时可执行文件及无关大规模 trace 均未提交。

## 运行与重建

需要 Linux x86-64 / WSL2。直接加载：

```python
import importlib.util
from pathlib import Path

entry = Path("nt/latest_20260911_afs_workflow_repair_r1_r2/r1/main.py").resolve()
spec = importlib.util.spec_from_file_location("afs_workflow_r1", entry)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
action = module.agent(observation, configuration)
```

把路径中的 `r1` 换成 `r2` 即可加载 R2。

重建三种开关版本：

```bash
cd nt/latest_20260911_afs_workflow_repair_r1_r2
python3 tools/build_variants.py r1
python3 tools/build_variants.py r2
```

复跑面板时复用相邻的完整 11 对手包：

```bash
python3 tools/run_panel.py \
  --package ../latest_20260911_p16_jointafs_r1/agent \
  --binary r1/build/workflow_on.so \
  --out /tmp/afs_r1_workflow_panel \
  --seed-start 2610100000 --seed-count 50 --workers 16
```

R2 只需把 `r1` 换成 `r2`。原始构建回执保留了开发机绝对路径，它只是审计记录；实际构建脚本按传入的相对目录运行。

## 验收边界

- R1 机制测试 70/70，R2 机制测试 72/72。
- R2 审计版与冻结版 1,100/1,100 动作哈希一致；开关关闭回归通过。
- 单进程最慢单步：R1 0.202 秒，R2 0.468 秒，低于线上 1 秒限制。
- 源码没有写入验收 seed、对手名称或目标坐标特判。
- 本地胜率只适用于这套冻结对手池，不能外推为当前 Public 天梯实力。
