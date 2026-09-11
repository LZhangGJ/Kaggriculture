# Triad-DP T1 完整交接包（2026-09-09）

## 后续追加：重编译警告（2026-09-09）

**本机 GCC 13.3 `-O3` 已确认会把动物收益预测编译错，不能以“编译成功”作为行为验收。** 原发布 `package/` 保持不变；新原因报告、独立复现、诊断库和 400 局逐动作回归位于 [build_issue_20260909](build_issue_20260909/README_ZH.md)。源码不改、编译追加 `-fno-ipa-modref` 已恢复该批全部 287,600 步与 G001 两座位已知结果。执行下方原构建流程之前，请先阅读问题包并复验工具链。

以下“本次交接”指首次原包交接范围；新原因调查和验证见上述追加包。原 1,400 局整体面板仍是随包结果，没有借本次局部验证宣称全量复现。

## 先看结论和边界

这是 GPT 提供的 **Kaggriculture_Triad_DP_T1_20260908** 原包，放在 `package/`，未修改策略、配置、对手或随包实验记录。

- 默认策略：规则经营规划 + 每日公共情景续跑比较 + 动态执行器；**不是 RL，也没有默认启用训练出的模型**。
- 随包独立面板：100 seed × 7 个实时对手 × 双座位，共 1,400 局，1,132 胜，**80.86%**。
- 同面板 C3＋J7 对照：1,090 胜，77.86%。提高 3 个百分点，但随包配对区间约为 −2 至 +8 个百分点，尚不能称为稳定全面升级。
- **本次交接仅做文件完整性、源码检查、原始对局记录重统计；没有在本机重新编译/跑完 1,400 局，也没有验证 Kaggle 提交。** 原包所述无缓存重编译、官方一致性与耗时均是随包证据，不是这次新增的实测。
- 策略来源为 9 月 5 日 C3 基础，不能将其对照理解为当前工作区最新版 F3。

| 冻结对手 | T1 胜场/200 | T1 胜率 | 随包 C3＋J7 胜率 |
|---|---:|---:|---:|
| G001 | 183 | 91.5% | 78.0% |
| G003 | 110 | 55.0% | 100.0% |
| Boatlee V29 | 163 | 81.5% | 59.5% |
| Kaito V58 | 167 | 83.5% | 100.0% |
| Lynn V5 | 161 | 80.5% | 88.5% |
| Yhay81 Six-day | 170 | 85.0% | 58.5% |
| Yhay81 Three-day | 178 | 89.0% | 60.5% |

并非逐对手 80% 或 90%；G003 是明显短板。T1 平均现金 105,113，平均分差 +11,664。

## 包含什么

原发布清单内 1,885 个文件加原清单本身，共 **1,886 个原文件，127,641,268 字节**。包括完整源码、Linux 策略库、7 对手和模拟器、构建入口、实验数据与结果、历史快照、模型实验、官方复验资产及说明。遵照原发布清单，不复制工作区后来生成的 Python 缓存；不混入其他未完成实验。

| 入口 | 用途 |
|---|---|
| [原包 README](package/README_ZH.md) | 构建、运行、验证及参数说明 |
| [设计与范围](package/DESIGN_AND_SCOPE_ZH.md) | 三层实现及局限 |
| [原报告](package/REPORT_ZH.md) | 原开发、消融与独立面板结果 |
| [策略入口](package/policy/agent.py) / [冻结配置](package/policy/config.json) | 实际导出的 Agent；两者都必须保留 |
| [经营与状态账本](package/policy/triad.hpp) | 生成计划、承诺保留、实际成交反馈 |
| [候选](package/policy/proposals.hpp) / [每日比较](package/policy/search.hpp) | 当日候选与前瞻评分 |
| [公共情景](package/policy/public_flow_scenario.hpp) | 对手公开产能与市场近似，不是真实未来 |
| [执行器](package/policy/executor/) | 取料、移动、维护、收获和交付 |
| [实时对战面板](package/src/panel.cpp) / [arena](package/arena/) | 7 个冻结对手与规则环境 |
| [最终原始记录](package/runs/release_holdout100/rows.json) | 1,400 局，可自行重统计 |
| [同面板基线](package/runs/baseline_j7_holdout100/rows.json) | 配对比较，不能换面板混比 |
| [全部实验索引](package/EXPERIMENT_INDEX.json) | 包括成功、失败和中止的开发实验 |
| [来源](package/SOURCE_PROVENANCE_TRIAD.json) / [环境](package/ENVIRONMENT.json) | 依赖版本和来源边界 |
| [交接校验程序](verify_package.py) / [校验结果](HANDOFF_ACCEPTANCE.json) | 字节哈希完整性和结果重统计，不等于模拟重验 |

## 如何运行

使用 **Linux x86-64 或 WSL2**，需要 C++20 的 g++、Python 开发头文件。附带 `.so` 不是 Windows DLL；换系统或 Python 版本应从源码重建。原构建环境是 Python 3.13.5、GCC 14.2.0，见原环境文件。

从仓库根目录进入交接目录，先做不需要第三方库的文件/结果校验：

```bash
cd nt/latest_20260909_triad_dp_t1
python3 verify_package.py
cd package
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 build_all.py --jobs 2
python3 run.py --tag teammate_smoke --count 4 --threads 4
```

最后一条是 56 局，不是 4 局。`--tag` 必须是未存在的新目录，避免覆盖历史证据。单测一个对手可加 `--opponents g001`。CPU 宽裕可提高运行线程数；这里沿用原包保守的构建/运行设置，没有宣称本机吞吐。

重现随包的 1,400 局及对照：

```bash
python3 run.py --tag teammate_t1_reproduction --count 100 --start 260909000 --threads 4
python3 run.py --tag teammate_j7_reproduction --baseline j7 --count 100 --start 260909000 --threads 4
```

这些 seed 已公开，属于结果复现，**不能再称未见种子验收**。修改后需另选未用过的种子。

单元测试及官方规则/导出动作一致性：

```bash
python3 tests/run_tests.py
python3 run.py --tag teammate_trace --count 1 --trace
python3 tests/official_parity.py teammate_trace --out tests/results/teammate_trace --limit 14
```

仅编译策略：`python3 build_all.py --policy-only`。策略入口为 `policy/agent.py::agent` 或其中的 `Agent`。每个实例只用于一个座位，必须一并带上 `config.json`。显式传 `Agent(config=...)` 时应传完整冻结配置或明确修改后的配置，不要误以为它一定自动合并发布配置。

## 值得借鉴与不能夸大的地方

每天比较保持、交付速度、动物规模、竞争惩罚等少量调整，用执行器在**公共近似情景**内续跑到日末，再加远期估值；真实对战仍逐步按当前状态执行。已融资但未完工的承诺尽量保留，次日重新组织用工。

但它不是全局最优规划器：默认最多 6 个候选方向，去重后往往更少；对手供给时点仍有近似；`layout` 没有实际作用，`rotation/repeat/tour_dp` 默认关闭，不能把代码里有字段当作已启用能力。默认不使用附带的监督模型，也没有复现历史动作来兜底。

下一步应先在队友机器重建并复验，再看 G003 等真实短板。不要把这次上传称作已经解决所有对手，或已经满足线上 1 秒限制。
