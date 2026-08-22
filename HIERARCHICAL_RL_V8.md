# Kaggriculture 层级强化学习 V8

V8 在 V7 的任务状态机、路径执行器、显式每日预算和精确 PPO replay 上增加了“对手建模 +
稳健训练”层。主 PPO 奖励和执行器保持不变，所有新损失与 league 都可由系数或命令行参数消融。

## 按执行顺序实现的 8 项

1. `meta_strategy.py` 从官方 replay manifest 建立行为簇两两收益矩阵，输出 coverage、PSRO
   regret-matching mixture 和最差对手权重。入口为 `scripts/build_cluster_payoff_matrix.py`。
2. `opponent_model.py` 仅用公开信息编码对手，并为每个样本保存 120/72/24/0 步历史、mask、
   对手行为簇和终局 outcome；在线 `HierarchicalMemory` 使用同一编码。
3. 网络用 GRU 编码历史，8 类 belief head 以对手簇交叉熵训练，同时输出归一化熵作为不确定性。
4. belief embedding 加到共享 state，因此影响 mode、budget、task matching 和 critic；CNN
   proposal 另有显式 belief projection。
5. 前 7 天的不确定对手提高 `ROBUST_OPENING`/`CONSERVATIVE` 先验。硬预算解码器使用
   `max(采样 reserve, 不确定性底线)`，但 PPO log-prob 仍精确计算原始 Beta 动作；底线单独写入 replay。
6. 三个辅助 critic 预测未来 24/72/168 步的生产性资源增长。跨 episode 的标签被 mask；主奖励仍是
   可变现净资产潜势增量加终局 outcome。
7. BC 每轮混合行为簇均衡抽样与最差簇 CVaR 抽样，避免平均分掩盖脆弱 matchup。
8. PPO 可重复传入 `--league-checkpoint`，冻结对手并轮换 learner seat；若提供
   `--league-payoff-json`，按兼容的 PSRO 与 robust 权重混合抽样，否则均匀抽样。

## 训练入口

```powershell
# 建立真实对局元博弈
python scripts/build_cluster_payoff_matrix.py `
  D:\Kaggriculture\data\processed\official_v7_2026-08-21\manifest.json `
  --output artifacts/cluster_payoff_matrix_v1.json

# V8 BC
python scripts/train_hierarchical_bc.py `
  --official-v7-manifest D:\Kaggriculture\data\processed\official_v7_2026-08-21\manifest.json `
  --official-only --output artifacts/hierarchical_bc_v8.pt

# V8 league PPO（可重复指定多个 checkpoint）
python scripts/train_hierarchical_rl.py `
  --init-checkpoint artifacts/hierarchical_bc_v8.pt `
  --league-checkpoint artifacts/opponent_a.pt `
  --league-checkpoint artifacts/opponent_b.pt `
  --league-payoff-json artifacts/cluster_payoff_matrix_v1.json `
  --output artifacts/hierarchical_ppo_v8.pt
```

旧 V3–V7 checkpoint 会按名称和形状加载兼容张量。V8 新增的 belief、三时域 critic 以及扩展后的
mode head 保持随机初始化，并在启动时报告未兼容的旧张量数量。
