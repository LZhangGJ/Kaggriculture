# Kaggriculture Replay Route Switch（干净工程）

## 2026-09-21 当前结论：默认 80.02%，组合优化仍不部署

在未参与训练或筛选的 seeds `2610100000..2610100063` 上，对 `opponents/` 七个强对手、同 seed
双座（每臂 896 局）：纯 R1 `528/896`（58.9%），冷接管 `705/896`（78.7%），**当前默认
G275 + 路线浅树 + delay1 温接管 `717/896`（80.02%）**。这是当前部署的正式成绩。

实验性的 R1 handoff candidate-diff 浅树为 `718/896`（80.13%），相对当前默认仅净增 1 胜
（救败 28、致败 27，McNemar `p=1.0`），并使 Thomas `-5`、Herd `-6`；因此**不部署**，
资产只保留作离线机制研究。线上默认仍不加载 handoff selector。

2026-09-21 对 R1 的作物/动物组合选择增加了严格 opt-in 的局部优化实验：
`portfolio_swaps=1` 做单地块替换，`portfolio_swaps=2` 做受限双地块联动替换。默认均为 0。
单换两批独立 16-seed 合并净胜 0；双换 8-seed pilot 为 Thomas `-1`、其余 0，七强平均
分差全部或近乎全部下降。因此它们只保留作机制实验，**不进入部署，也不扩大正式验证**。
现有价值函数本来就包含已知商店需求、未来商店期望、双方公开资产产量、库存价格曲线、劳动/工资、
预算和竞争收益，并把对手供给各一半放在我方成交前后；失败说明当前主要问题不是少枚举一两个组合，
而是小的模型估值差会被后续滚动重规划放大。

这是 2026-09-20 从多个历史工作树收拢出的独立研究工程。主线只有一条：从近期高手 replay 提取经营路线并聚类，使用高速原生仿真做路线互打和反事实切换搜索，以浅层决策树选择前期 replay 路线；中期把真实公开状态交给**没有开局模板的 JointAFS R1** 动态策略。

当前代码可以完整运行，默认配置为 **固定 G275 opening + 浅树切换 + 三块地状态触发接管 R1**。

当前正式 64-seed 四臂结果以本文开头的 `528/705/717/718` 为准；更早的 `531/683/706`
分别对应旧接管语义，只用于解释演进，不再代表部署成绩。接管仍是强变量：过早切换会丢失 replay
已安排的扩地与资本形成，因此现在由“达到三块地后再等 1 天”触发，step288 是最晚截止。

## 目录

| 路径 | 内容 |
|---|---|
| `agent/` | 前期成熟 replay 路线执行器、147 维浅树和 replay→R1 组合入口 |
| `policy/r1/` | 真正无开局模板的 JointAFS R1 源码、二进制与最薄外部观察桥 |
| `meta_agent/` | 成熟路线执行、特征、切换控制器代码 |
| `scripts/` | replay 分析、聚类、原生互打、切换搜索、鲁棒浅树训练工具 |
| `fast_kaggriculture/` | 高速 C++ 仿真器源码、Python 扩展与已编译二进制 |
| `native_deps/` | 重建高速仿真器所需的最小 C++ 依赖源码 |
| `opponents/` | 7 个近期强公开对手；正式验证不混入弱对手 |
| `data/replays/raw/` | 本轮选定的 609 份完整原始 replay |
| `data/replays/daily/` | 成功通过旧日级 receipt 提取器的 177 份派生记录 |
| `data/artifacts/` | 609 replay 的特征、聚类、245 路线、互打矩阵、切换矩阵和树 |
| `experiments/` | 官方环境并行 A/B、扫描程序及历史结果 |
| `docs/` | 架构、数据、运行方法和结果解释 |

## 快速验证

```bash
cd /root/kaggriculture-replay-switch-clean-v1
PYTHONPATH=. /root/miniforge3/envs/torch-npu/bin/python verify_project.py
PYTHONPATH=. /root/miniforge3/envs/torch-npu/bin/python experiments/smoke_official.py
PYTHONPATH=. /root/miniforge3/envs/torch-npu/bin/python experiments/test_submission_contract.py
```

结构验证检查 609 replay、245 路线、7 个强对手、24,460,800 局切换数据、资产哈希、R1 接管符号和原生仿真器加载。官方烟测完整运行 719 步并跨过默认 day12 接管点。提交契约测试按两种 observation 视图驱动官方引擎，两个 seat 都必须跑满并完成接管（见交接中的视图保真度记录）。

## 当前部署语义

- 开局：固定 `G275`，不使用 Nash 混合。
- 前期：成熟 `TeammateExpandedRouteAgent` 执行 replay 路线；它含杂草修补、市场/现金保护和喂养保护，不是盲目动作磁带。
- replay 内切换：使用 `route_policy.json` 的浅树；现有树只覆盖五个 opening，检查点为 step 144/168/216（day6/day7/day9），当前控制器最多切换一次。
- 动态接管：达到 3 块地后延迟 1 天转交无开局模板 R1，step288 是最晚截止。R1 在此前持续接收公开观察及实际动作，用于公开交易、作物时钟与联合项目账本。
- 可用 `REPLAY_FORCED_OPENING` 和 `REPLAY_HANDOFF_STEP` 做离线实验覆盖；正式结论必须同 seed 双座、多 seed。

详见 [架构](docs/ARCHITECTURE_ZH.md)、[数据](docs/DATA_AND_PROVENANCE_ZH.md)、[复现命令](docs/REPRODUCTION_ZH.md)、[文件索引](docs/FILE_INDEX_ZH.md) 、[交接](HANDOFF_ZH.md) 与 [本轮会话记录](docs/SESSION_20260920_ZH.md)。
