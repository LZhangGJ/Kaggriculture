# 第 11 名路线迁移：一致风格 IL + 跨策略 League

源方案见 [第 11 名摘要](../sources/11_IL_LEAGUE_SOURCE_DIGEST_ZH.md)。

## 迁移结论

适合作为第 5 名 BC 路线的低风险版本：先用少量一致且可解释的强策略得到可玩 policy，再以 PFSP 和跨策略 agents 扩展。重点不是做庞大 league，而是明确每个示范风格、每个对手角色和覆盖盲区。

## 示范数据选择

不要把所有高分 replay 混在一起。先按公开行为聚类：

- 快速 wheat/carrot 周转；
- melon/long-horizon；
- tomato/strawberry ongoing；
- animal；
- 高 hands/land expansion；
- market timing/压价；
- terminal liquidation。

每个 BC model 只学习一个或两个内部一致的 cluster；保留胜局和败局。输出 cluster coverage、episode count、actor decisions、各日/状态分布。若官方 replay 不足，使用 v16 和规则 specialists 自生成；标记 synthetic provenance。

## 两种架构

先用 medium 2–4M：d192、4层，快速筛 features/action。large 不应直接照抄 15M，先试 d256、6层约 6–10M；只有 medium 已因容量而非 opponent/data plateau 才扩。

输入遵循共用 token，加 schedule edge bias。BC 同时学 task、target、mode 和 value/未来辅助；旋转不变通过固定 farm side/relative coordinates 实现，不依靠不必要 augmentation。

## IL anchor

初始 PPO 可用 KL-to-IL，但采用三段：

- 0–20M：KL 0.03–0.05，entropy 低；
- 20–100M：KL 逐步降至 0.005–0.01，对 rare/exploration heads 保留 entropy；
- 100M 后：只在 policy drift/collapse 时对 delayed best teacher 加小 KL，不永久锚定原始 IL。

这是对 Orbit 第 11 名配方的必要修改：Kaggriculture 数据覆盖更可能不完整，永久 KL 会把 animal/market blind spot 固化。

## Cross-strategy league

不是训练 5 个同结构大模型。建议一个 main learner + 4 个便宜固定/慢更新角色：

- exploit-current：专门从 arena replay 找当前弱点；
- market specialist；
- animal/long-horizon specialist；
- historical champion mix。

main 60%、PFSP history 20%、specialists 20%。每个 specialist 最低配额，PFSP 不得把弱但风格独特对手淘汰。每次 promotion 把 main checkpoint 加入 history，并冻结训练 schema/version。

## 训练/评估

IL-only 必须先做座位平衡 arena；它是可用 baseline，不是最终结论。PPO 100M 后比较：

- current-only self-play；
- history PFSP；
- history + specialists。

同模型同数据预算。如果 league 只提高对内部 opponents、不提高固定外部 panel，拒绝。记录策略分层胜率，防止平均胜率掩盖对 animal/market 的崩溃。

## 本机资源

主 learner 2–6M，specialists 多为规则 agent 或 0.5–1M，不需要并行训练 5 个大模型。预计 8k–30k transitions/s；100M 0.9–3.5 小时。相比 Orbit 1150 GPU-hours，本机版通过角色冻结、共享 rollout 和少量 lineage 把工程压到单卡可迭代范围。

## 何时使用

当已有高质量 replay/v16 轨迹且 from-scratch PPO 在基本农业行为上浪费很多 steps 时很有价值。如果 IL-only 不能稳定击败 deterministic baseline，先修标签/动作抽象，不要用 league 掩盖 BC 问题。
