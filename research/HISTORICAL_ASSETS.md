# Kaggriculture 历史资产总表

盘点时间：2026-09-03（Asia/Tokyo）。这是历史资产的权威入口；新增、移动或废弃资产时
更新本文件，不再新建平行的“最终版清单”。

## 盘点范围

本次只读检查了：GitHub 远端全部可见分支、`D:\Kaggriculture` 下与 Agent、Replay、
NT、聚类和实验有关的目录，以及用户提供的两份 `E:\` 代码文件。Git 分支在盘点前执行
了 `git fetch origin --prune`。没有运行任何历史 Agent、训练、仿真或外部文件中的代码。

大小使用文件系统递归统计，单位为 GiB；目录仍可能继续变化。结果可信度分为：

- `已核对元数据`：本次检查了文件、manifest、hash 或 Git tree。
- `历史报告，本次未重跑`：结论来自资产自带报告，不能当作新复现。
- `未验证`：只确认位置或文件存在，尚不能据此引用效果。

## 推荐入口

| 目的 | 当前入口 | 判断 |
|---|---|---|
| 了解全部模块、实验和数据 | [`../agent.md`](../agent.md) | 仓库统一入口 |
| 当前 Route Clustering 源码 | [`../agents/route_clustering_switch_agent/`](../agents/route_clustering_switch_agent/) | 可读源码；新克制树尚未实现 |
| 当前已归档的克制实验 | [`experiments/counter_cluster_round2_semantic_v1_20260903/`](experiments/counter_cluster_round2_semantic_v1_20260903/) | 训练门失败，coverage 0/300 |
| 当前离线 NT 搜索器 | Git 分支 `agent/add-nt-simulator-orbit-migrations` 的 `nt/latest_20260901_candidate8_width12_mcts/` | 离线 Oracle；不是在线分类器 |
| 当前本地 Replay 主库 | `D:\Kaggriculture\top40` | 约 1.03 TiB；先用 manifest，禁止全量复制 |
| 当前路线聚类生成 Agent | `D:\Kaggriculture\route_clustering_top40_20260831\output-v3\agent-dynamic\main.py` | v3 manifest 标为推荐；效果是历史本地验证 |
| Replay 人工诊断 | Git 分支 `tool/replay-diagnostic-simulator-20260827` | 只读可视化，不生成拳法 |
| 自适应规则执行器候选 | `E:\Daily_DP_Rule_Agent_20260903.zip` | 可作 NT 参数化执行器；尚未集成 |

任何“最新”“最强”只在对应冻结协议内成立，不代表当前榜单或新 Replay 上仍最强。

## Git 分支

远端状态以 2026-09-03 fetch 后的 ref 为准。

| 分支 / commit | 主要内容 | 已有结果与边界 | 建议 |
|---|---|---|---|
| [`main`](https://github.com/LZhangGJ/Kaggriculture/tree/eec409775bc538cf65b3676172ee3ce741ca2469) `eec4097` | 通用 CPU/GPU 实验室和 Route Clustering 基线 | 历史基线；没有本次统一资产索引 | 只作共同祖先 |
| [`agent/add-nt-simulator-orbit-migrations`](https://github.com/LZhangGJ/Kaggriculture/tree/agent/add-nt-simulator-orbit-migrations) `c242cba` | NT/JAX 模拟器、FC 系列、68 Agent 池、Replay 收集、Candidate8、Orbit 迁移 | 3,403 files changed；多个交接包，结论口径不同 | NT 代码只从各 `latest_*` 包的 README 进入 |
| [`agent/add-route-level-training`](https://github.com/LZhangGJ/Kaggriculture/tree/agent/add-route-level-training) `693670a` | Agent 原生反事实续跑、路线数据集、route selector、短期任务 DAG | 有测试和工作流文档；未找到可晋级的最终策略结果 | 可复用数据生成接口，不引用为已验证拳法 |
| [`agent/macro-intent-clustering-20260825`](https://github.com/LZhangGJ/Kaggriculture/tree/agent/macro-intent-clustering-20260825) `e7de35b` | 早期 NT、宏观意图聚类、UG0 历史 | G034/G096 提交在分支顶端已回滚；被后续分支覆盖 | 历史参考 |
| [`agent/triton-inventory-market`](https://github.com/LZhangGJ/Kaggriculture/tree/agent/triton-inventory-market) `982b3ef` | GPU 随机事件/市场一致性、层级 RL V3–V8、belief/league、Top25 刷新 | 有实现和测试；没有本次复跑或已晋级最终模型 | 研究基础设施，不直接并入生产 Agent |
| [`agent/unseen-generalization-v1-20260824`](https://github.com/LZhangGJ/Kaggriculture/tree/agent/unseen-generalization-v1-20260824) `00a80de` | Replay 盘点、特征、去重和防泄漏切分 UG0 | 合成 smoke 9/9；真实 D 盘语料和 UG1 未运行 | 可复用 inventory/split 代码 |
| [`codex/full-local-models-20260814`](https://github.com/LZhangGJ/Kaggriculture/tree/codex/full-local-models-20260814) `429261f` | Git LFS 模型、BC/DAgger/PPO、本地 neural/script champion | 报告 neural champion 对 11-Agent 池 106/220（48.18%）；未晋级超过脚本策略 | 模型档案，不作为当前冠军 |
| [`local/route-genome-v1`](https://github.com/LZhangGJ/Kaggriculture/tree/local/route-genome-v1) `d42f7a0` | RouteGenome 进化、7 路线 selector、block/tail 实验、贝叶斯对手路由设计 | 历史报告：652 对手池 95.51%，最低 82.81%，对旧动态 Agent 81.84%；贝叶斯 counterbank 仍是规格，不是完成实现 | 生成路线与 response-cluster 设计的主要历史依据 |
| [`tool/replay-diagnostic-simulator-20260827`](https://github.com/LZhangGJ/Kaggriculture/tree/tool/replay-diagnostic-simulator-20260827) `4b59583` | 官方 1.32.7 Replay 画面、动作结果和风险提示 | 合成测试工具；Replay 只能看到回合净变化 | 可独立使用 |
| [`agent/counter-cluster-round2-semantic-v1-20260903`](https://github.com/LZhangGJ/Kaggriculture/tree/agent/counter-cluster-round2-semantic-v1-20260903) | 语义 PlanDelta Round2、统一文档规范和本资产表 | 4,992 局训练；0/300 coverage；validation/holdout 未运行 | 当前协作与后续实验分支 |

### NT 快照关系

`agent/add-nt-simulator-orbit-migrations` 同时保存多代完整快照，不应把整个 `nt/` 当成一个
版本。主要入口如下：

| 目录 | 内容 | 历史报告，本次未重跑 |
|---|---|---|
| `nt/latest_20260819/` | M3.9 三层规划器、JAX rules core、5 个公开 Agent | 早期 1.32.7 开发快照 |
| `nt/latest_20260821_arena_fusion_fc2a/` | 28-Agent strict-JAX Arena 和规则融合 | 报告包含 37,800 局互打 |
| `nt/latest_20260823_fc24b/` | FC0–FC24B 链、34-Agent 验收、submission 55708153 | 报告包含 8,704 局验收 |
| `nt/latest_20260825_front40_public_v48/` | Rank1–40 重建、34 notebook 扫描、Kaito V48 | 报告 V48 对 FC24B 为 851/1,024 |
| `nt/latest_20260827_complete_68_agent_pool/` | 完整 68-Agent 验证池 | 候选/验收资产集合 |
| `nt/latest_20260827_macro_route_submit_agents/` | G034/G096 单文件路线树 | 后续宏观聚类分支曾回滚相同提交，不视为当前生产版本 |
| `nt/latest_20260829_candidate8_rolling_oracle/` | 八类 PlanDelta 候选与 rolling Oracle | 114 个 manifest 记录、约 7.71 MiB；线上 ranker 未通过，禁止部署 |
| `nt/latest_20260901_candidate8_width12_mcts/` | 8 阶段 Beam width 2/4/8/12、C++ MCTS、R8 执行器 | 322 个 manifest 记录、约 10.91 MiB；交接验收 PASS |
| `nt/gpu_sim/` | parity-first JAX GPU 规则实现和冻结参考 | 修改规则后必须重做 parity receipt |
| `nt/agents/eba26v2/` | EBA26v2 submission、holdout 和上传回执 | 独立历史提交资产 |
| `nt/handoff/gpt_route_bundle_1327_v2/` | 1.32.7 JAX 路线搜索与官方裁判交接包 | 面向四核 CPU 审阅环境 |
| `nt/replay_collection/` | 每日 Top60 计分 submission Replay 收集规范 | 只负责收集；文档内 `E:\ai_coding` 路径是旧机器位置 |
| `nt/orbit_wars/` | 11 篇公开 Orbit Wars 资料摘要及 Kaggriculture 迁移设计 | 设计研究，不是已实现 Agent |

Candidate8 的决策日为 `0,1,3,6,9,12,18,24`。Width12 报告在
`G001/G004/G013 × 2 seeds × 2 seats` 上 width=12 为 12/12 不败、10/12 命中四档联合
最佳；MCTS 只作为小预算多样性补充。所有这些都是看得到真实终局的离线 Oracle 搜索，
不能写成“线上能根据 Replay 自动生成并选择克制拳法”。

2026-08-29 rolling Oracle 报告还显示：静止对手下真实未来 Oracle 平均现金 167,747，
但对 G001 平均分差仍为 -36,051；这证明候选空间存在经济上限，不证明强对手克制成立。

## 本地 Agent 与输出

### Route Clustering Top40 工作区

根目录：`D:\Kaggriculture\route_clustering_top40_20260831`，11,798 个文件，约
2.323 GiB。

| 子目录 | 文件数 | 大小 | 内容与状态 |
|---|---:|---:|---|
| `source/` | 264 | 49.9 MiB | Git 源码；当前有 7 个 modified、20 个 untracked，禁止覆盖 |
| `output/` | 45 | 548.4 MiB | v1 路线库、矩阵和结果 |
| `output-v2/` | 62 | 711.5 MiB | v2 动态/静态 Agent 和完整互打结果 |
| `output-v3/` | 33 | 112.8 MiB | v3 动态 Agent 和刷新后结果 |
| `agent-v1-pre-2107-update/` | 2 | 4.3 MiB | v1 `main.py` 与 manifest |
| `.conda_toolchain/`、`.venv/` | 11,392 | 952.1 MiB | 可重建环境，不是源码 |

四个历史 Agent 已核对如下：

| 版本 | 路径 | 字节 | SHA-256 | manifest 结论 |
|---|---|---:|---|---|
| v3 动态推荐版 | `output-v3\agent-dynamic\main.py` | 5,449,006 | `F81C7143...A8AF3F5A` | G003/G032/G040 Nash；step 144 可切 G001；453,632 配对局；条件提升 3.4722% |
| v2 动态版 | `output-v2\agent-dynamic\main.py` | 4,899,342 | `5F75A460...F19FAC6D` | G003/G033/G041 Nash；step 144 可切 G001；404,480 配对局；条件提升 3.8981% |
| v2 静态版 | `output-v2\agent-static\main.py` | 4,898,954 | `DD6F6FE9...360881CD` | 三路线 Nash；395 条路线、684 族、9,960,320 配对局；无切换 |
| v1 归档 | `agent-v1-pre-2107-update\main.py` | 4,473,269 | `FF32E2E4...0C3FA1EF` | manifest 为 G003 开局并含 step 144/216 两个切换节点；不是纯固定路线 |

v1 存在口径冲突：此前口头记录称“固定 G003”，但同目录 `MANIFEST.json` 明确写有两个
`switch_nodes`，并报告条件提升 4.4006%。在复跑前，以 manifest 描述代码结构，但不把
该提升和 v2/v3 的不同面板直接比较。

### 其他本地代码与实验

| 位置 | 文件数 / 大小 | 内容 | 可用性 |
|---|---:|---|---|
| `D:\Kaggriculture\counter_cluster_round2_semantic_v1_20260903` | 35 / 3.44 MiB | Round2 完整 Windows 归档 | 已核对；结果同步进 Git |
| `D:\Kaggriculture\cpp_route_experiments_20260827` | 628 / 4.606 GiB | causal tree、block、tail、residual、smoke 和导入实验 | 未统一归档；不能从目录名引用结论 |
| `D:\Kaggriculture\K68_nt_20260827` | 2,096 / 0.323 GiB | 较早 NT 68-Agent 本地快照 | 被 Git `latest_*` 包部分取代 |
| `D:\Kaggriculture\agents` | 14 / 1.22 MiB | boatlee/kaito 等 4 个公开 Agent 副本 | 来源候选，不是我方生产源码 |
| `D:\Kaggriculture\public_agents` | 521 / 17.98 MiB | 公开 Agent 源码池 | 输入数据；按 manifest/hash 去重 |
| `D:\Kaggriculture\public_agent_candidates` | 320 / 24.91 MiB | 公开候选下载/提取目录 | 与 `public_agents` 有重叠 |
| `D:\Kaggriculture\public_high_score` | 257 / 11.34 MiB | 高分 notebook/Agent 快照 | 外部历史材料 |
| `D:\Kaggriculture\our_latest_two` | 701 / 20.216 GiB | submission 55663355–55918053 的 Replay 归档为主 | 目录名过时；多数子目录没有 Agent 源码 |
| `D:\Kaggriculture\investigations` | 86 / 2.574 GiB | 单独调查及 Replay | 未标准化 |

`cpp_route_experiments_20260827` 顶层包含 `causal_tree_v1..v5`、`block_mvp_*`、
`block_sequence_oracle_*`、`adaptive_tail_oracle_v0`、`route_residual_adapter_v0`、
`multi_farmer_path_residual_v1`、`nt68_import_v1/v2`、`live_replay_pool/v2` 和多个
`pytest_*` 临时目录。后续只迁移仍需复用的实验；临时 smoke 不逐个写长期报告。

其他支持目录：`ke1327`（769 文件/0.132 GiB）是本地 1.32.7 环境副本；
`frozen_pools`、`league_candidates`、`pool_sources`、`submissions` 和 `reports` 是小型
候选/回执仓；`cache`、`uv-cache`、`pycache_jit_adapter_a2` 及根目录 `.pyc` 是可重建
缓存。`publish` 和 `kaggriculture-ops-lab` 是独立工作区，不能替代当前 Git 源码。

## 本地数据

| 位置 | 文件数 | 大小 | 已核对范围 | 使用规则 |
|---|---:|---:|---|---|
| `D:\Kaggriculture\top40` | 35,824 | 1,053.334 GiB | current manifest：40 队、5,256 行、4,413 唯一 episode、4,500 个现存路径 | 主 Replay 库；只从 manifest 选样本 |
| `D:\Kaggriculture\top25` | 2,773 | 84.041 GiB | 25 队、3,547 行、2,893 唯一 episode、2,534 个现存路径 | Top40 的另一时间快照，不能直接相加 |
| `D:\Kaggriculture\gold_top10` | 2,920 | 83.979 GiB | 未进一步审计 | 历史 Replay 集 |
| `D:\Kaggriculture\ranks_21_40` | 1,874 | 54.182 GiB | 未进一步审计 | 历史 Replay 集 |
| `D:\Kaggriculture\data` | 7,431 | 175.045 GiB | 多代 processed/raw 数据混合 | 必须使用具体 manifest，不能把目录当一个数据集 |
| `D:\Kaggriculture\artifacts` | 116 | 3.924 GiB | 模型/训练生成物 | 结果缓存，不是原始数据 |
| `D:\Kaggriculture\publish` | 24,492 | 2.401 GiB | 发布工作区 | 与源码和数据均可能重复 |
| `D:\Kaggriculture\kaggriculture-ops-lab` | 23,912 | 0.917 GiB | 独立运维工作区 | 未纳入当前 Git 规范 |

Top25/Top40 的 manifest 行数、唯一 episode 和现存路径数不同，说明存在重复引用与缺失
路径。任何实验必须冻结所用 manifest 和文件 hash，不能用“D 盘 top40”作为充分数据说明。

## 外部单文件

以下文件没有提交 Git，本次只读检查，未执行。

| 文件 | 字节 / SHA-256 | 做什么 | 已有证据与边界 |
|---|---|---|---|
| `E:\v20_integrated_g1_guard_water.py` | 48,451 / `188638AD...6B24B3A` | 每日启发式状态反馈：资产目标、作物分配、动物槽位、G1 guard water、市场和工人任务 | 无 Replay 分类、无 NT 搜索、无随附独立验证 |
| `E:\Daily_DP_Rule_Agent_20260903.zip` | 38,255 / `C1412650...759901` | 有界资产 DP、出售 DP、空间角色、原子任务队列、多人路线编译和操作重校验 | 包内 `main.py` 48,098 bytes，SHA-256 `48927784...214C5A` |

ZIP 自带结果声称在 `kaggle-environments==1.32.7`、25 seeds、双座位、对手 `pass` 的
50 局中平均终局现金 151,038.2，范围 102,302–182,535，50/50 DONE。该结果本次没有
独立重跑，而且对手不行动，所以不能证明对主动对手的克制能力。它可以作为 NT 要搜索
参数的动态执行器底座，但当前 ZIP 本身没有对手后验、response clustering 或反制搜索。

## 未归档与风险

1. `D:\Kaggriculture\route_clustering_top40_20260831\source` 有 27 个未提交项：7 个源码
   修改、1 个新脚本和 19 个运行时 DLL。其所有者未知，本次没有复制、提交或清理。
2. D 盘根目录存在多个孤立 `.pyc` 和 `pycache_jit_adapter_a2`。它们不是可维护源码，
   不能作为唯一实现；在确认无进程依赖前也不删除。
3. `our_latest_two` 名称已失真，后五个目录主要是 Replay，不是可直接提交的 Agent。
4. 多个 Git 分支各自带完整 NT/workspace 副本。选择代码时必须写 branch+commit+子目录，
   禁止从不同 snapshot 随意拼文件。
5. 历史报告使用 1.32.6/1.32.7、不同对手池、seed、座位和指标。只有协议一致才可横向比较。
6. 未列入 Git 的绝对路径只在当前机器有效；迁移前先核对 hash，不能仅凭文件名覆盖。

## 维护规则

- 新增外部代码、Replay 集、模型或生成 Agent：在本文件对应表增加位置、用途、版本/hash、
  证据级别和推荐状态。
- 移动或删除资产：保留旧路径及迁移去向；不要静默改表。
- 历史结果复跑后，在独立实验目录写标准 `agent.md`，再把本表的可信度从“历史报告”改为
  “已执行”。
- 新旧版本并存时只指定一个推荐入口；其余标明 superseded、archive 或 data-only。
- 本文件只登记资产，不复制原始日志、Replay、模型和生成二进制。
