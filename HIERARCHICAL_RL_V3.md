# Kaggriculture 分层任务匹配 RL V3

V3 保留 V1/V2，重点解决 V2 的四个限制：任务重复绑定到每个工人、市场每回合只有一个订单、跨回合战略没有显式状态、30 个策略 agent 只能提供人工先验而不能直接预训练。

## 一回合的完整决策

```text
双方棋盘 + 公开经济 + 自己私有库存
                 │
                 ▼
       共享三层 CNN + 实体/全局编码器
                 │
       ┌─────────┴──────────┐
       ▼                    ▼
  8 类战略模式          最多 64 张任务卡
  （跨回合保持）        （任务不绑定工人）
       │                    │
       └─────────┬──────────┘
                 ▼
       Task Transformer 关系建模
                 │
                 ▼
      工人 × 任务的全局约束匹配
                 │
       ┌─────────┴──────────┐
       ▼                    ▼
确定性取货/路径/操作       GRU 自回归市场头
                        最多 10 个有序订单
       │                    │
       └─────────┬──────────┘
                 ▼
             官方 action
```

战略模式包括均衡、作物扩张、动物现金流、高价值市场、保守经营、终局清仓、反镜像和对手应对。模式首次选择后写入 episode memory，后续回合的任务和市场决策都以它为条件；新一局会重置。

## 候选、先验与匹配

- 任务池只保存一次语义任务，例如“给 (3, 5) 的奶牛喂食”，不再为 17 个工人复制 17 次。
- 单独构造 `worker × task` 特征，包括距离、是否已在目标、是否延续旧计划、携带/仓库物品、截止时间和工人类型。
- 每个普通任务容量为 1，空闲任务容量为 17；全局匹配保证同一工人只接一个任务、同一普通任务不会被多次分配。
- 强制维护和安全回收优先保留，其余候选按 21 个任务家族轮转填充，避免高分但同质的候选挤掉小类任务。
- 规则先验不是硬编码最终动作，而是可学习缩放的 logit 偏置：`最终分数 = 网络分数 + α × 归一化规则先验`。`α` 从 4.0 开始并由训练更新。

## 多订单市场

市场头用 GRU 自回归生成最多 10 个订单。每生成一个订单，`MarketBudget` 会更新现金、当日雇佣数、土地、种子、仓库容量和库存，再为下一位置重算合法掩码。`NONE` 是 STOP；若前 10 个位置全是订单，则不强制额外 STOP。

这比一次独立选择 10 个槽位更可靠，因为“先卖出获得现金再购买”“连续雇佣成本递增”“购买占用仓库容量”等顺序效应进入了动作合法性。

## 从 30 个策略 agent 行为克隆

`scripts/train_hierarchical_bc.py` 在线跑完整专家对局，并把底层轨迹转换成高层监督：

- 连续移动、仓库取货和最终操作被折叠为同一意图；例如“走向仓库 → 取小麦 → 走向牛舍 → 喂牛”在前面的状态都标成喂养任务。
- 教师市场列表转换成 10 槽有序序列，并用 teacher forcing 训练。
- agent 名称映射为一个初始战略模式标签。
- 最终相对资金经过与 RL 相同的 `log1p` 归一化势能变换，用于预训练 value head，避免 BC→RL 时目标量纲突变。
- 日志中的 `candidate_closure` 表示专家高层意图能否在当前候选池中找到精确对应任务；它是扩充候选空间时最重要的诊断量。未闭合样本只计入诊断，不会用错误的近似任务污染监督损失。

示例（agent 可以是内置名称、Python 文件路径或 `module:callable`）：

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts\train_hierarchical_bc.py --teachers starter D:\agents\agent_a.py D:\agents\agent_b.py --envs 8 --collection-rounds 4 --device cuda
```

先用很小规模验证整条链路：

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts\train_hierarchical_bc.py --teachers starter --envs 1 --collection-rounds 1 --episode-steps 4 --epochs 1 --batch-size 8 --hidden-size 64 --board-width 16 --d-model 64 --transformer-layers 1 --device cpu --output artifacts\hierarchical_bc_smoke.pt
```

## 30-agent 行为聚类与采样权重

`scripts/diagnose_agent_behavior.py` 先在共同对手面板上构造每个 agent 的
`[胜率, 平均现金, 平均分差] × 对手` 响应向量，再做稳健标准化、确定性
k-means 和 silhouette 选 K。输出 JSON、CSV、Markdown 三种格式，并给出：

- agent 的 cluster、最近邻和距离；
- 每组成员及同质化规模；
- 按 `1 / sqrt(cluster_size)` 计算、均值归一到 1 的 BC 采样权重；
- 每个 agent 的证据局数、来源和被插补的自博弈空单元数；
- 完整 silhouette 曲线以及最优 K 是否碰到搜索上界。

这个结果衡量的是“面对同一批对手时的交互响应”，不会冒充动作级路线聚类。
后续拿到所有 agent 的逐步轨迹后，应把雇工时点、土地扩张、市场订单、原语占比和
路线状态加入特征，再与当前响应聚类交叉验证。

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts\diagnose_agent_behavior.py `
  --pair-results D:\arena\all_exact_jax_round_robin_pair_results_v1.csv `
  --panel fc2a_shadow_prt_fix=D:\arena\fc2a_panel.json `
  --panel fc2b_k320_clone_aware_fusion=D:\arena\fc2b_panel.json `
  --output-dir artifacts\agent_behavior_clustering_v1
```

## Closure 与执行器失败日志

BC 不再只输出一个总 closure 数字。每个工人意图都会记录任务类型以及以下唯一原因：

- `EXACT`：任务类型、目标、物品和 worker-task 边都精确存在；
- `TASK_TYPE_MISSING`：候选池没有该类任务；
- `TARGET_MISSING`：有任务类型，但目标格不存在；
- `ITEM_MISMATCH`：目标存在，但物品语义错误；
- `PAIR_MASKED`：任务存在，但对该工人不可选；
- `UNRESOLVED`：以上检查都通过却仍未闭合，用来捕获实现漂移。

训练 checkpoint 和同名 `.diagnostics.json` 都会保存逐任务计数及闭合率。

分层执行器把确定失败写入 `HierarchicalMemory` 的有界日志，包括非法任务/市场索引、
被 mask 的 worker-task 边、缺资源、目标冲突、路径无合法第一步和目标前置条件变化。
日志同时按原因、按 `任务:原因` 聚合；RL checkpoint 和同名
`.diagnostics.json` 会保存累计结果。合法的等待（例如作物当天已经浇水但尚未成熟）
不会被误报为失败。

## 强化学习微调

从行为克隆 checkpoint 开始自博弈 A2C：

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts\train_hierarchical_rl.py --init-checkpoint artifacts\hierarchical_bc_v3.pt --envs 16 --updates 500 --device cuda
```

CPU 冒烟训练：

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts\train_hierarchical_rl.py --envs 1 --updates 1 --rollout-steps 2 --episode-steps 4 --hidden-size 64 --board-width 16 --d-model 64 --transformer-layers 1 --device cpu --output artifacts\hierarchical_a2c_smoke.pt
```

奖励仍采用逐步相对资金势能变化，并在终局加入胜负奖励。A2C 同时训练任务匹配、市场序列、战略模式和 value head；规则先验的缩放参数也参与更新。

## 第一版边界与下一步

- 任务执行器沿用 V2 的曼哈顿逐步路径和保守避碰，不是完整时空 A*；多工人拥堵时需要进一步加入预约表或 MAPF。
- 当前战略模式在整局内保持不变。下一版可在每天开始或局面突变时增加 `KEEP/SWITCH` 选项，同时避免每回合抖动。
- 行为克隆标签来自规则化轨迹反推，候选未覆盖的标签会回落为空闲并计入 closure 失败；应先用 30 个 agent 的 closure 分布指导扩充任务生成器。
- PyTorch 项目锁定 `kaggle-environments==1.32.6`，而部分最新参考策略来自 1.32.7；提交前必须在目标环境做完整回归。
- 当前 RL 奖励偏向现金。后续可加入资产净值、维护失败、任务完成和空转惩罚，并通过消融确认 shaping 不改变最终目标。
