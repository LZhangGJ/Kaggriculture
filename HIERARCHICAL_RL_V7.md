# Kaggriculture 层级强化学习 V7

V7 修复了 V6 在长局状态、专家数据、价值监督和策略更新上的主要闭环缺口。网络主体仍是
双棋盘 CNN + Candidate Transformer，但训练目标和运行时状态已经升级。

## 1. 完整决策流程

每个环境步按以下顺序执行：

1. 把自己和对手的 10×10 棋盘编码为两个空间分支，同时编码经济、市场、单位和私有库存；
2. CNN proposal head 对 `任务类型 × y × x` 产生热图，并预测物品、ETA、价值和 slack；
3. 规则候选、持续任务候选、学习候选和可选专家候选合并，进入严格结构/资源可行性过滤；
4. Candidate Transformer 建模候选之间的竞争与互补关系；
5. 全局 matching 在容量约束下给每名工人分配一个任务；
6. 选中任务形成资源 plan context，缺少的种子、饲料、肥料或动物生成强制市场订单；
7. 市场 GRU 在共享预算下先执行必要订单，再生成自由订单；
8. BFS/space-time reservation executor 把任务编译成该步的原子动作；
9. 运行时状态机更新 chain、stage、ETA、完成/取消原因和内部 trace。

策略的随机因子为：proposal 无放回抽样、每日模式决策、全局 worker-task matching、自由市场
序列。强制资源订单不计入策略概率。

## 2. 持久任务链

`UnitTaskState` 现在保存：

- 稳定 `chain_id` 和 `chain_family`；
- `stage_index`、`chain_tasks`、`completed_stages`；
- `stage_started_step`、真实 ETA 和预约路线；
- `termination_reason`。

同一格的 crop lifecycle（种植、浇水、施肥）和 animal lifecycle（建造、放置、喂养、照料、
收获）可以跨日延续。只有已执行阶段的前置条件在下一观察中消失时，系统才将其判定为
`STAGE_ADVANCED` 或 `GOAL_SATISFIED`。未经验证的策略替换会记录为
`ASSIGNMENT_REPLACED / POLICY_IDLED / PRECONDITION_INVALID`，不再伪装成完成。

内部 trace 使用真实 chain ID 和 stage，不再按日期重新生成 ID，也不再固定写 stage 0。

## 3. 专家数据与价值监督

`train_hierarchical_bc.py` 可以读取
`artifacts/agent_behavior_clustering_v1/behavior_clusters.json`。每个 epoch 先均衡抽取行为簇，
再在簇内轮换样本，避免高度同质化的大簇控制梯度。未匹配到审计表的教师会被分配独立簇，
不会静默混入某个公开策略族。

逆规划窗口允许跨日观察后续动作，chain ID 使用工人、任务族和目标格生成。诊断文件包含：

- 每个教师和行为簇的样本数；
- 低置信度率、歧义率、概率归一化错误；
- 缺失 chain ID、非法 stage index、路线不连续；
- phase/task 分布和 candidate closure。

价值标签不再把最终净资产差复制给所有时间步。V7 用与 self-play 相同的定义：

```text
r_t = reward_scale × (liquidatable_asset_potential[t+1]
                      - liquidatable_asset_potential[t])
terminal += win_bonus × outcome
V_target[t] = discounted_return(r_t ... r_T)
```

这能区分“何时创造价值”，并保持双方潜势和终局奖励反对称。

## 4. 精确动作重放 PPO

V6 的 A2C 每批只使用一次旧 log-prob。V7 保存固定候选张量和所有实际抽样记录：

- proposal 的完整 Plackett–Luce draw sequence；
- constrained matching 每一轮抽中的 worker-task edge；
- 初始模式、每日 KEEP/SWITCH、切换后的新模式；
- 市场序列以及哪些订单由资源计划强制产生。

PPO 更新时，在新参数下逐项重算同一动作概率。测试会检查更新前 replay log-prob 与 rollout
log-prob 在 `1e-5` 内一致。训练使用 clipped policy objective、clipped value loss、GAE、KL
早停和梯度裁剪。proposal 监控记录 entropy、独立 approximate KL、旧 log-prob 均值和
proposal task head 的 L1 参数变化。

PPO 推理阶段将模型置于 `eval()`，以免 Transformer dropout 使“同一动作概率”本身随机；
`eval()` 不会关闭梯度，优化仍正常进行。

## 5. 候选热路径

proposal 合法格掩码直接复用已编码棋盘的 locked channel，在 GPU 上广播为
`batch × task × y × x`，移除了训练内环的 Python 三重循环。状态棋盘编码也会在 proposal、
训练候选、deploy closure 和 oracle closure 间复用，避免同一 batch 重复编码。离散 TaskCard
构造、严格操作前置条件和最终 executor 仍在 CPU/Python；这是下一阶段进一步批量化的边界。

## 6. 运行命令

```powershell
$env:PYTHONPATH = "$PWD\src"

.venv\Scripts\python.exe scripts\train_hierarchical_bc.py `
  --teachers path\to\agent_a.py path\to\agent_b.py `
  --teacher-clusters-json artifacts\agent_behavior_clustering_v1\behavior_clusters.json `
  --episode-steps 720 --proposal-top-k 24 --device cuda `
  --output artifacts\hierarchical_bc_v7.pt

.venv\Scripts\python.exe scripts\train_hierarchical_rl.py `
  --init-checkpoint artifacts\hierarchical_bc_v7.pt `
  --episode-steps 720 --rollout-steps 64 `
  --ppo-epochs 4 --ppo-clip 0.2 --minibatch-steps 8 `
  --device cuda --output artifacts\hierarchical_ppo_v7.pt

.venv\Scripts\python.exe scripts\validate_hierarchical_v7.py `
  --checkpoint artifacts\hierarchical_ppo_v7.pt `
  --episodes 1 --episode-steps 720 --official-differential `
  --device cuda --output artifacts\hierarchical_v7_validation.json
```

V3–V6 checkpoint 可以迁移；V7 checkpoint 包含新的训练元数据。旧 checkpoint 没有持久链内存，
因此链状态只在新 episode 中建立，不需要迁移序列化状态。

## 7. 本轮验证结果

- 全部本地测试：`60 passed`；
- CUDA BC 烟雾训练成功，输出 V7 closure、trace audit 和 value-target 诊断；
- CUDA PPO 烟雾训练成功，proposal task head 单次更新 L1 变化 `0.1301`；
- seed 701 的完整 720 配置：实际 719 次转移全部与官方 runner 逐步一致；
- 最终状态：双方 `DONE`，executor failure `0`；
- 覆盖 708 个市场与单位同时动作的 actor-step；
- 平均层级推理延迟约 `147.5 ms/batch(2 actors)`，P95 约 `206.2 ms`。

该长局使用小型烟雾 checkpoint，只用于验证执行闭环，不代表胜率。它仍出现 14 条超过 48 步的
长链和较频繁采购；正式训练应把 stalled-chain rate、重复采购、胜率、净资产、closure、
proposal KL 和 executor failure 同时作为验收门槛。

## 8. 显式每日预算层

预算头输出现金储备比例、`LABOR/CROP/ANIMAL/LAND/SUPPLIES` 支出 simplex，以及最高为
日初现金 25% 的紧急额度。`BudgetPlan` 在日界生成并写入 `HierarchicalMemory`，日内保持不变；
销售收入不会自动扩大已经确定的类别额度。

市场动作现在同时满足模拟器合法性、类别剩余额度和现金底线。普通市场动作不能突破计划；任务卡
所需的强制资源订单只能在紧急额度内突破，超出时任务在执行前被替换为安全 IDLE。PPO replay
记录计划特征、日初现金和动作前累计支出，因此硬掩码仍可精确重放。

官方或在线专家轨迹按天反推软标签：实际采购金额只作为预算下限，日终现金作为储备目标。BC
联合优化 reserve loss、类别分布交叉熵、预算下限 hinge loss 和紧急额度正则。新 checkpoint
架构名为 `hierarchical_candidateformer_v7_budget_v1`；旧 V7 checkpoint 以非严格方式迁移，
新增预算头从随机初始化开始训练。

PPO 中预算计划是日界显式动作：现金储备用 Beta，五类支出 simplex 用 Dirichlet，紧急比例用
缩放到 `[0, 0.25]` 的 Beta。replay 保存实际采样的连续值和 `budget_action_active`；日内 KEEP
步骤的预算 log-prob 为零。PPO 在新参数下重算三部分连续密度并加入总 policy log-prob，预算
monitor 单独记录 approximate KL、日界动作数和 category head 参数变化。PPO checkpoint 架构名
为 `hierarchical_candidateformer_ppo_v7_budget_v2`。

当前 reward 仍是每步可变现净资产潜势差加终局胜负，并没有单独加入 1/3/7 天资源增长项。
多尺度资源增长更适合作为辅助 value heads 或 potential-based shaping；直接把三个累计增长同时相加
会重复计算同一份财富变化，并可能诱导囤积无法变现的土地、工人或动物。
