# P16 JointAFS R1：完整策略、实战证据与 Public 回执

这是 2026-09-10 冻结的 **P16 JointAFS R1** 交接包。AFS 在 P16/T3 执行修复基础上，增加联合动物、饲料与后继项目比较；它是规则动态规划 Agent，不是 Replay 播放器，也不含 ML/RL 模型。

## 1. 结论先行

同一批 11 个原始实时 Python 对手、50 个新 seed、双座位，共 1,100 局：

| 版本 | 胜 | 负 | 平 | 原始胜率 |
|---|---:|---:|---:|---:|
| T3R1 TakeoverMerged 父版 | 797 | 299 | 4 | 72.45% |
| **JointAFS R1** | **856** | **240** | **4** | **77.82%** |

AFS 相对父版救回 125 局，同时丢失 66 场父版胜局，净增 59 胜。它是有价值但有取舍的开发分支，**不是逐对手 90% Agent**。

两版直接对打的两个独立 200 局块方向相反：第一块 AFS 93–101，第二块 AFS 101–93；合计各胜 194、12 平，比分正好 50%。因此不能声称 AFS 单独全面取代父版。

Public Submission [56146577](https://www.kaggle.com/competitions/kaggriculture/submissions?submissionId=56146577) 已 `COMPLETE`，分数 **1430.3**，错误字段为空。这个结果明显弱于本地面板，必须作为后续复盘重点；不能用本地胜率掩盖线上失败。

## 2. 文件索引

| 内容 | 位置 |
|---|---|
| 可调用入口 | [`agent/main.py`](agent/main.py) |
| Python observation 编码与状态隔离 | [`agent/policy/agent.py`](agent/policy/agent.py) |
| 冻结运行配置 | [`agent/policy/config.json`](agent/policy/config.json) |
| 冻结开发二进制 | `agent/policy/joint.so` |
| 完整 C++ 源码 | [`agent/policy/`](agent/policy/) |
| AFS 联合候选 | [`agent/policy/joint_candidates.hpp`](agent/policy/joint_candidates.hpp) |
| 联合承诺状态 | [`agent/policy/joint_bundle_state.hpp`](agent/policy/joint_bundle_state.hpp) |
| 编译及契约测试 | [`agent/build.py`](agent/build.py)、[`agent/tests/`](agent/tests/) |
| 11 对手与本地裁判 | [`agent/opponents/`](agent/opponents/)、[`agent/referee/`](agent/referee/) |
| 1,100 局公平对照 | [`evidence/external11_50seeds/ACCEPTANCE_REPORT_ZH.md`](evidence/external11_50seeds/ACCEPTANCE_REPORT_ZH.md) |
| 第二批 200 局直接对打 | [`evidence/head_to_head_newseeds2/ACCEPTANCE_REPORT_ZH.md`](evidence/head_to_head_newseeds2/ACCEPTANCE_REPORT_ZH.md) |
| Public 发布包与回执 | [`public_submission/README_ZH.md`](public_submission/README_ZH.md) |
| 实际上传 tar.gz | `public_submission/AFS/submission.tar.gz` |

`agent/MANIFEST_SHA256.json` 是 GPT 提供的原始冻结清单；`agent/verify_package.py` 会校验其覆盖的 503 个文件。`public_submission/` 中的发布脚本保留为本次构建和上传审计记录；它引用原研发工作区，不是本交接包的独立重建入口。

## 3. 队友直接运行

需要 Linux x86-64 / WSL2；原生 Windows Python 不能加载 `.so`。从 Git 根目录执行：

```bash
cd nt/latest_20260911_p16_jointafs_r1/agent
python3 verify_package.py
python3 check_binary.py --binary policy/joint.so --out /tmp/jointafs_behavior.json
python3 panel_small.py --help
```

加载 Agent：

```python
import importlib.util
from pathlib import Path

entry = Path("nt/latest_20260911_p16_jointafs_r1/agent/main.py").resolve()
spec = importlib.util.spec_from_file_location("jointafs_entry", entry)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
action = module.agent(observation, configuration)
```

每局/每个座位必须使用独立状态；入口会在新的 step 0 自动 reset。策略只接收官方当前 observation，不应输入 seed、对手程序身份、私有仓库或真实未来。

## 4. 发布验收

Public 使用的并不是开发环境中依赖较新 `libstdc++` 的原始 `joint.so`，而是 GCC 11、通用 x86-64、静态 libstdc++/libgcc 的便携重编译版：

- 冻结源码哈希与 `RELEASE_BUILD.json` 一致；
- 发布版与开发版对 11 对手、双座位 22 局的动作哈希和终局字段全部一致；
- 解压后的提交文件在官方 1.32.7 环境完成 2 局 × 719 步；
- 最慢单步 323.335 ms，0 次超过 1 秒；
- Kaggle 只调用一次上传，Submission ID 为 `56146577`。

详见 [`public_submission/AFS/ACTION_IDENTITY.json`](public_submission/AFS/ACTION_IDENTITY.json)、[`OFFICIAL_FILE_ACCEPTANCE.json`](public_submission/AFS/OFFICIAL_FILE_ACCEPTANCE.json) 和 [`submission_record.json`](public_submission/AFS/submission_record.json)。

## 5. 边界

- 本地 77.82% 来自固定 11 对手池，不能外推为当前天梯胜率。
- Public 1430.3 是当前最重要的反证：AFS 存在线上分布或强对手适应问题。
- 本包保留完整对手资产以复现实验，但不包含 venv、编译工具链、Kaggle 凭据、下载缓存或大型 Replay。
- 不要根据单 seed 或这 11 个对手继续硬编码；后续修改必须用新 seed、双座位和独立对手池复验。
