# 第 5 名：Behavior Cloning + IMPALA/V-trace

- 作者：TonyK
- 官方文章：[Orbit Wars 5th place solution](https://www.kaggle.com/competitions/orbit-wars/writeups/orbit-wars-5th-place-solution)
- 开源代码：[tonykozlovsky/orbit-wars-2026-pub](https://github.com/tonykozlovsky/orbit-wars-2026-pub)
- 发布时间：2026-07-20
- 特别说明：文章称正文由 GPT-5.6 Sol 根据公开代码和作者笔记生成，并由作者审核认可。

## 一句话

先用强玩家 replay 做 BC，随后用异步 IMPALA/V-trace、历史 checkpoint pool 和 delayed moving teacher 做终局胜负 RL；单 RTX 5090 训练约一周。

## 两阶段训练

BC 把 replay 还原成动作前 observation 和合法的 per-planet action class，以 masked cross-entropy 学强玩家。RL 延续同一个 model/action representation，无需重新对齐标签空间。

RL 使用很多 CPU actor：模拟器产生短 rollout，推理请求批量送到模型，共享 buffer 交给 GPU learner；actor 使用稍旧策略造成的 off-policy lag 用 V-trace 修正。loss 包括 V-trace policy gradient、value、entropy 和 teacher KL。

主要 reward 是终局 win/loss，关闭中间 fleet、planet、production shaping。对手混合 live policy、训练各阶段 frozen checkpoints、2P/4P specialists。delayed teacher 不是永久冻结，而是周期性加载较旧 learner checkpoint；它既保留 BC 学到的可玩性，又随训练逐步变强。

## 模型

状态被看成 planet set 和有向 planet-to-planet edges：

- planet：ownership、production、position/motion；
- arrival horizon；
- source-destination：distance、tactical margin；
- existing/valid edge/active-player/action masks；
- relative player identity。

连续量归一化，部分离散量 embedding。planet、arrival、edge 分别编码后，以 16 层 planet/edge cross-attention 融合；hidden 128、4 heads，无 dropout。policy 对每个 source planet 输出 destination × send bucket，value 池化全局状态。

## 工程与部署

- C++ 环境与 observation；
- 异步 CPU actors、shared-memory buffers、固定 batch GPU inference；
- BF16 forward、FP32 policy/value heads；
- `torch.compile`；
- 2P/4P 分别 benchmark、训练和打包；
- CPU 提交用 PyTorch AOTInductor + 小型 C++ runner。

文章报告最终模型在单 RTX 5090 上约一周；代表性最后阶段目标是数亿 environment steps，但文章没有给出统一精确总步数。第 11 名作者的二手比较表把该方案估为约 300M steps；应把它视为二手整理，不与第 5 名作者报告混为一谈。

## 风险与启示

- BC 的上限受 replay 策略和覆盖率限制；如果动物、某作物或罕见反制样本少，初始化会系统性忽略它们。
- IMPALA 的价值在 actor/learner 解耦和高吞吐，但单 GPU 上 CPU actor、推理 server、learner 的平衡比端到端 JAX PPO 更复杂。
- 迁移到 Kaggriculture 时，BC 应是 warm start 和结构搜索工具，必须以覆盖平衡、专项 synthetic demonstrations 和 self-play league 补盲区。
