# C3 / F3 / J7+C3 与七对手 C++ 交接包

日期：2026-09-06。给队友做 Day0–28 经营决策搜索，不是 Kaggle 提交包。

## 1. 先说明交接内容和边界

- 三个独立入口：C3 全自主、F3 全自主（已修预测账本）、J7 经营参考＋C3 执行器。
- 七个实时 C++ 对手及其必须的路线/参数资产，完整 C++ 比赛模拟器。
- 自包含构建、整局对战、Day0–28 的 29 节点覆盖适配器、最小 Beam 示例。
- 原始源码来源、SHA256、历史官方一致性报告、近期 RL 失败及候选价值审计。
- **没有携带大型 Replay、训练缓存、编译缓存、预编译对象、神经网络权重或 GPU 运行时。**

重要：本包可运行的 `search_demo.py` 使用近期 RL 的**窄接口**，每天最多替换/增加/暂缓一个未投入项目，最多 10 个选择。它是交接接口验证，不是旧的约 321 候选宽池，也不代表全规划器能力上限。

用户希望后续恢复宽候选、允许单节点批量乃至多项目联合调整。完整 C3/F3 源码已交付；具体迁移位置与尚未实现项见 [宽候选迁移说明](WIDE_CANDIDATE_MIGRATION_ZH.md)。**不得把尚未接通的宽池当成已完成，也不得承诺能够找到必胜路径。**

## 2. 三个我方入口

| 参数 `--arm` | 经济规划 | 底层执行 | 是否参考 J7 |
|---|---|---|---|
| `c3auto` | C3 的完整自主投资方法，从 Day0 开始 | C3 | 不调用 J7 配方；每局检查 reference=0、完整规划次数正确 |
| `f3` | F3 从 Day0 开始自主投资；完整项目流撤销/替换修复版 | F3 | 不参考 J7 |
| `c3j7` | 原 C3 条件 J7 经营配方参考、回退和本地调整 | C3 | 是，但不是播放 J7 原始单位动作，也不保证所有参考意图都被兑现 |

`KEEP` 意思是让该版本的原生动态规划继续运行，**不是工人 PASS、不投资或全局静止**。

名称上的“全自主”指无 J7 经营参考，不是模型独立输出全农场决策。本包不使用此前未获提升的 PPO 权重。

| 已冻结历史面板，100 seed × 双座位/对手 | C3 自主 | F3 自主 | J7+C3 |
|---|---:|---:|---:|
| 七对手平均胜率 | 43.50% | 60.14% | 78.14% |
| G001 | 50.50% | 76.00% | 74.00% |
| G003 | 24.00% | 45.00% | 100.00% |
| Boatlee V29 | 73.00% | 98.50% | 63.00% |
| Kaito V58 | 40.00% | 59.50% | 100.00% |
| Lynn V5 | 48.00% | 56.50% | 89.00% |
| Six-Day | 24.00% | 39.00% | 57.00% |
| Three-Day | 45.00% | 46.50% | 64.00% |

来源：[三组原实验报告](evidence/three_arm/G3_FINAL_REPORT_ZH.md)。这些不是此次移机小测试的新实力估计。F3 修账后另一批独立 100 seed 得到 64.57%；**不能拿不同 seed 的 64.57% 和此表直接作提升比较**。修复对本次原/新 PPO 训练数组没有造成变化，见 [复跑结论](evidence/ledger_rerun/ACCEPTANCE_ZH.md) 和 [同数据审计](evidence/ledger_rerun/IDENTITY_AUDIT_ZH.md)。

## 3. 安装与运行

支持已验环境：Ubuntu 24.04 / WSL2、g++ 13、Python 3.12。无需 GPU、JAX、Torch。Boost JSON 小型源码依赖已经带入 `vendor/`。

在 Linux / WSL 终端中，进入本 README 所在目录：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python build.py --jobs 3
.venv/bin/python smoke.py --out runs/smoke_first
.venv/bin/python run.py --arm f3 --seeds 4 --threads 16 --out runs/f3_first
```

如果系统没有编译器或 Python 头文件，由使用者安装 `g++`、`python3-dev`、`python3-venv`。不要把本机旧 `.o` 复制过来绕过构建。

Windows PowerShell 用户应调用 WSL 内的 Python；进入仓库对应 `/mnt/.../nt/latest_20260906_c3_f3_j7c3_search` 后运行上述命令。它不是 Windows 原生 MSVC 工程，也不是可以直接提交 Kaggle 的纯 Python 文件。

`--jobs` 是编译进程数，`--threads` 是 C++ 对局工作线程数，1–16 可配。只有四核心的机器设为 `--jobs 2`、`--threads 4` 即可；吞吐须在当地实测。

七对手全部运行：`--seeds 4` = 7 对手 × 4 seed × 双座位 = **56 局**。单对手示例：

```bash
.venv/bin/python run.py --arm c3j7 --opponents g001 --seeds 8 --out runs/j7c3_g001
.venv/bin/python run.py --arm c3auto --seeds 4 --out runs/c3_auto
```

输出目录必须不存在，以免覆盖历史结果。`games.json` 保存逐局现金/胜负/异常；`decisions.npz` 保存逐日候选特征、mask、实际选择及修改标记。

## 4. Day0–28 搜索接口

直接运行一条 29 节点策略：

```bash
.venv/bin/python run.py --arm f3 --plan examples/keep_day0_28.txt --seeds 4 --out runs/keep_plan
.venv/bin/python run.py --arm f3 --plan examples/mixed_day0_28.txt --seeds 4 --out runs/mixed_plan
```

第二条是**激活接口的测试策略，不是已证实高收益策略**。

文件恰好 29 个整数，对应引擎 Day0…Day28。候选 ID：

| ID | 含义 |
|---:|---|
| 0 | KEEP，保留该日原生计划 |
| 1 / 2 / 3 / 4 / 5 | 对选定未投入项目使用小麦 / 胡萝卜 / 番茄 / 草莓 / 甜瓜 |
| 6 / 7 / 8 | 鹅 / 牛 / 羊 |
| 9 | 撤回或暂缓该选定新项目，具体语义由版本编译器处理 |

这个“选定项目/地块”仍由冻结接口选取；不是用户任意指定地块。当前 mask 包含版本规划约束，**不是官方全部合法动作空间**。

节点执行顺序：

```text
真实公开盘面 + 我方私有状态
→ 原生规划器构建当天计划
→ 生成当前候选，按本日 ID 选择（不可用则 KEEP）
→ 更新该版本计划/账本/订单
→ 由原执行器根据实时盘面执行每一步
→ 双方同步提交动作，由 C++ 规则结算
→ 下一天从真实结果重新规划
```

Day29 保留原生维护和终局管理，仍会运行并记录第 30 次日规划，不在本 29 节点文件中改动。719 次实际动作转换，不能误当每局 720 个可执行动作。

适配器保留 `requested`、实际 `choice`、`mask`、`fallback` 和 `changed`。请求不合法时回 KEEP，不得把这种空改动计作成功实施。计划改变也不等于最终已产出/售出，经济成效必须看实际对局。

最小 Beam 示例：

```bash
# 仅验证程序接通：单对手、前两个节点。
.venv/bin/python search_demo.py --arm f3 --opponents g001 --width 2 --last-day 1 --train-seeds 1 --test-seeds 1 --out runs/search_smoke
# 后续如要运行完整 29 节点窄接口搜索，由队友主动执行。
.venv/bin/python search_demo.py --arm f3 --width 4 --last-day 28 --train-seeds 4 --test-seeds 32 --out runs/f3_beam4
```

Beam4 保留四个前缀，每前缀尝试最多十个本接口选择；剩余未来先 KEEP 再续跑终局。评价按最弱对手胜率→总体胜率→平均资金差。冻结训练胜者后才复验未见 seed，另跑同 seed 的 KEEP 对照。

此示例只有数字计划去重，不做实际行为聚类；无快照缓存，所有分支从 reset 完整执行，安全但并非搜索速度最优。`Pool` 每次完整对局持有独立对手状态；对手一直实时响应，绝非冻结动作回放。

## 5. 从哪里看代码

- [全部 Agent 索引](AGENT_INDEX_ZH.md)：三个我方、七对手的逐项入口。
- [宽候选迁移说明](WIDE_CANDIDATE_MIGRATION_ZH.md)：恢复宽池时必须处理的语义、账本、状态承诺。
- [交付验收](DELIVERY_ACCEPTANCE_ZH.md)：本次源码重编译、213局接口测试和42局原机对照。
- [C++ 公共策略 ABI](src/rl_common.hpp)：观察边界、候选记录、生命周期。
- [C++ runner](src/runner.hpp)：完整双方对战；`src/runner_module.cpp` 保留既有共享桥接工具。
- [节点适配器](src/day_plan_policy.cpp)：仅改变每日选择，不替换原子执行器。
- [来源清单](SOURCE_PROVENANCE.json)、[对手注册表](OPPONENTS.json)。

历史报告中的 `E:/...`、`/mnt/e/...` 是原机证据路径，不是本包运行依赖。当前可运行入口统一是本目录 `build.py / run.py / smoke.py / search_demo.py`。

## 6. 不应误解的结论

- 没有任何版本已证明对全部七对手 90% 胜率，也没有全局最优保证。
- 旧 PPO 的确定性策略仍 KEEP；本次只交接底座及接口，没有声称 PPO 学出了好拳法。
- 搜索阶段可以用仿真已发生的完整结局比较；上线 Agent 只能看当时可见状态，不能读 seed、对手私仓、对手代码或真实未来。
- 官方一致性是有限夹具验证，不是对所有可能局面的形式证明；已有证据随包保留。
- 稳定地多 seed 打赢某对手，应称测试范围内高胜率，不称“任何盘面必胜”。
