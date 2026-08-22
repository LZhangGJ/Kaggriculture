# Kaggriculture 任务链、资源联合规划与可训练 Proposal RL V6

V6 针对 V5 最关键的三个缺口做闭环修改：顺序动作不再互相竞争、任务与市场资源不再
各自决策、proposal 不再只能依赖 BC。

## 1. 从扁平 Top-M 改为任务链与当前阶段

逆规划先按工人、目标格和任务族把未来语义事件分段。例如：

```text
移动 → PLANT → WATER → HARVEST
```

现在产生一条 crop chain，而不是三个互斥任务。BC 只用该链最早尚未完成的 stage 监督
worker-task matching，同时在 trace 中保留后续 stages、operation、ETA 和稳定 chain id。
PICKUP 会作为 acquisition stage 附着到后续任务；只有找不到后续语义操作时才成为独立链。

trace schema 升级为 `kaggriculture.agent-trace.v2`，新增：

- `chain_id`、完整 remaining chain 和 stage ETA；
- 当前 `phase`：`IDLE / ACQUIRE_RESOURCE / NAVIGATE / EXECUTE / WAITING`；
- 网络的 per-worker phase auxiliary head；
- assignment 与 phase loss 都按逆规划置信度加权。

## 2. 严格可行性与任务—资源—市场联合规划

学习卡和专家卡进入 Candidate Transformer 前必须通过：

- 当前地块/动物/作物的真实操作前置条件；
- owner 限制、deadline、路线可达性；
- 工人携带资源、仓库资源或当前可合法购买的市场资源。

每条 worker-task 边增加 `required_market_indices`。当任务缺少种子、小麦、肥料或动物时，
编码对应的 `BUY_SEED / BUY_PRODUCT / BUY_ANIMAL` 订单。任务匹配完成后：

1. 用共享 `MarketBudget` 复核多个工人的合计采购是否可支付；
2. 共同预算无法支持的任务回退到安全 idle；
3. 选中任务、物品、pair features 和资源订单形成 plan context；
4. plan context 同时调制市场 GRU 初始状态和每个订单 logits；
5. 可支付的必需资源订单优先于自由市场动作执行。

因此 `FEED`、`FERTILIZE`、`PLACE` 或空地 `PLANT` 不会再与它们需要的采购计划脱节。

## 3. Proposal 获得 RL 信号

训练时 proposal 不再固定使用 `detach + deterministic Top-K`。V6 在合法空间格上进行无放回
随机 proposal 采样，并计算归一化 Plackett–Luce log-prob surrogate：

```text
proposal sample log-prob
  + mode log-prob
  + worker-task matching log-prob
  + market log-prob
  → actor loss
```

因此回报可以直接更新 `proposal_task_head`。确定性评估仍使用 Top-K，并把 proposal
log-prob 置零。BC 的 focal/item/ETA/value/slack losses 继续提供低方差预训练信号。

## 运行

```powershell
$env:PYTHONPATH = "$PWD\src"
.venv\Scripts\python.exe scripts\train_hierarchical_bc.py `
  --teachers starter path\to\expert.py `
  --inverse-top-m 3 --proposal-top-k 24 --device cuda `
  --output artifacts\hierarchical_bc_v6.pt

.venv\Scripts\python.exe scripts\train_hierarchical_rl.py `
  --init-checkpoint artifacts\hierarchical_bc_v6.pt `
  --proposal-top-k 24 --device cuda `
  --output artifacts\hierarchical_a2c_v6.pt
```

V3–V5 checkpoint 以 `strict=False` 迁移；phase、resource-plan 和 proposal-RL 新参数随机
初始化。V6 checkpoint 严格加载。

## 仍需注意

当前 proposal 的离散 TaskCard/ETA 编码仍在 CPU/Python 执行，因而还不是大规模 GPU actor
的最终形态。Plackett–Luce 使用归一化 surrogate 控制 K 带来的梯度尺度；正式实验应分别
监控 proposal gradient、Recall@K、任务执行成功率、强制采购次数和实际胜率。
