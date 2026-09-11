# 第 11 名：Imitation Learning → PFSP/Cross-run League

- 作者：Piiiiiiiii、morim3、qistripute、msd0110
- 官方文章：[Standing on the Shoulders of Supergiants](https://www.kaggle.com/competitions/orbit-wars/writeups/11th-place-solution-standing-on-the-shoulders-of)
- 开源代码：[msdsm/kaggle-orbit-wars-11th-solution](https://github.com/msdsm/kaggle-orbit-wars-11th-solution)
- 发布时间：2026-07-20

## 一句话

用 Jake Will 与 Isaiah 的约 3500 episodes 做一致风格 BC，IL 单独约第 25；随后 PPO + PFSP 历史池 + 5 条不同 agent lineage 的 cross-run league 到第 11。最终 15M 模型只做 2P RL，却用同一权重玩 4P。

## BC 数据与表示

只选两个强且风格一致的玩家，保留 rating≥1600 的胜局和败局；只保留胜局会丢失“落后/失败状态”监督。混合 Top5 反而把不同风格平均成弱策略。坐标非旋转不变，训练做 4 个 90° rotations，并尝试 seat canonicalization。

每个 planet 48 features，覆盖位置、relative owner、garrison/production、incoming fleet、nearest-neighbour、local force balance、capture/defense economics、board rank、comet、no-launch projection。fleet 折进 target planet。global 11 features 包括进度、角速度、玩家数、自己 planet/ship/fleet shares 和按强度排序的 incoming pressure。

Transformer 使用 2D RoPE 和 arrival-time ALiBi bias。medium：d192/4层/6heads/3.7M；large：d320/6层/8heads/约15M。policy 为 target+noop/fraction，value 用 outcome MSE。

## PPO + league

IL 不只是初始化，还是稳定锚点：PPO 关闭 entropy，固定约 0.05 的 KL 把 policy 拉向 IL。去掉 KL 或加入 entropy 都曾 collapse；whole-batch advantage normalization 使 policy gradient 比只中心化强约 7×。terminal-only reward；shaping 无收益。

只与 current self 训练没有超过 IL。PFSP 把 past checkpoints 和弱 heuristic 加入池，优先困难对手，才越过 IL ceiling。之后 5 个并行 agent 交换 checkpoint，包括 strongest self-play 和 4 个 top-player IL clones。cross-run 多样性减少 monoculture，但没有单独超过最强 lineage；强 base 和训练量仍是主要杠杆。

## 工程、规模与提交

- Rust 环境/feature，多 env parallel；learner+opponent forward 融合，GPU sample；
- 从 188 提到 6068 SPS，单一配置可约 10K；
- 最终 lineage 约 0.4B transitions；全 co-evolution league 约 6B；所有 RL 实验合计约 14B；
- self-play/league 约 1150 GPU-hours，另有 A100 large IL→RL 约 40–60 小时；
- large JAX 首次提交因首步 JIT 超时；删除 value head并启动 warmup 后修复。

最终 winning weights 只做了 2P RL。专门 4P run 不如复用强 2P；作者认为原因包括 binary reward 太粗、动作只能 all-in/noop，无法表达 4P 常见 partial split，以及 4P throughput 较低。

## 证据边界

文章末尾比较表总结了其他前排名次的模型/步数/硬件，其中部分不是相应作者文章的逐字数据。本目录只把这张表当作第 11 名作者的二手比较，不据此建立第 4 名独立来源。

## 迁移价值

它最直接回答“数据不足会怎样”：少而一致的数据能得到强起点，但 broad mix 会冲突，未覆盖策略不会凭空出现。Kaggriculture 应把 BC 作为起点，用规则 specialists、合成 rare-strategy demos 和 self-play exploiters 主动补覆盖，不能让 IL KL 永久锁死策略。
