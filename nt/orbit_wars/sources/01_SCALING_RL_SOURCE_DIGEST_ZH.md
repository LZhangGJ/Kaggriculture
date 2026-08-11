# 第 1 名：Scaling Reinforcement Learning to the Stars

- 作者：IsaiahP（Tufa Labs）
- 官方文章：[Kaggle writeup](https://www.kaggle.com/competitions/orbit-wars/writeups/1st-place-solution-scaling-reinforcement-learnin)
- 开源代码：[IsaiahPressman/kaggle-orbit-wars](https://github.com/IsaiahPressman/kaggle-orbit-wars/tree/main)
- 发布时间：2026-07-10
- 证据口径：以下模型、步数、硬件和结果均为作者报告。

## 一句话

用 Rust 高速环境、约 200M 参数 Transformer、PPO 和约 15B 纯 self-play steps，把“尽量少手工设计、尽量扩大模型和训练”推到提交限制的边界。

## 状态与动作

状态保留较低层的 entity set：planet、fleet、comet，再加 4 个 player summary 和 1 个 global summary。各实体特征先经独立 MLP 投影到 768 维。网络还含 player summary、global、4 个 actor plan、4 个 value 和 4 个 scratch token；scratch token 只是注意力共享工作区。

最初让模型直接预测 launch/angle/fleet size，能学会但不够强。最终把角度抽象成“选目标星球”，连续舰队大小仍由模型决定；Rust 规划器再把 source-target 转成合法角度。这里的关键不是完全无工程，而是把对策略困难、对几何确定的问题交给不同模块。

## 网络

- 38 个 residual self-attention blocks；
- hidden 768，16 heads，MLP hidden 1536；
- 每个玩家的 source/target stream 决定是否发射和目标；
- fleet size 用 8-component truncated discretized logistic mixture；
- critic 对剩余玩家输出胜率分布；
- 单次 forward 同时给所有玩家算动作，作者称节省约 2–4× 推理计算。

## 训练

- PPO + GAE-λ、clipped policy loss、advantage normalization、entropy bonus；
- 2P/4P 同时训练，早期均衡采样；胜者 +1、败者 -1；
- 纯 self-play，没有 imitation 初始化；
- 周期性对 previous-best 做 1v1 和 2v2；新模型胜率 >70% 才替换 best；
- 加入对 previous-best 的 policy KL 和 value cross-entropy 以稳定更新；
- 最终阶段曾把 2P 比例提高到 90%，事后认为忽视 4P 和历史 league 是错误。

作者试过遮掉明显撞太阳等坏动作，训练反而变弱；最后只在 finetune 和推理恢复 mask。作者的解释是无 mask 迫使模型学习物理，但这只是其假设，不是通用结论。

## 引擎、规模与提交

官方 Python 环境太慢，重写为 Rust，以真实 replay 做大量 parity；环境同时负责 raw observation→tensor、source-target→angle。采用多线程并行环境、预分配和复用 pinned memory，降低 CPU→GPU 阻塞。

- 早期模型 1–5M；最终约 200M；
- 大模型实验：8×B200、2048 env、64-step rollout；
- 最终：4 台 8×B200 节点、8192 env；
- 报告吞吐约 6.3M steps/GPU-hour；最终训练约 2400 B200-hours。

提交侧受 1 秒/turn、60 秒 overage 和 100MiB 限制：linear 做 int8，权重用 group-size 128 的 4-bit NormalFloat；可见 fleets 截断为最大者。约 8% 4P 慢机上 overage 快耗尽时切换到 5M 小模型收尾。作者没有把这一过程描述成 teacher→student 部署蒸馏。

## 文章给出的失败教训

- `gamma=1` 让领先者没有尽快结束的动力，训练浪费在已决定的局面；作者建议 early truncation/surrender。
- 只看 previous-best 且后期偏 2P，会产生策略循环、自我风格过拟合和 4P 不稳。
- 这条路线的优势来自规模，但规模不是 3090 可复现部分；可迁移的是动作抽象、规则 parity、历史 checkpoint、固定晋级赛和训练/部署一体化。
