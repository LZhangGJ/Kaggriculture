# 第 19 名：GPU Simulator + BC + PPO

- 作者：Maciej Sypetkowski、Natalia Jaśkowska
- 官方文章：[19th Place Gold Writeup](https://www.kaggle.com/competitions/orbit-wars/writeups/19th-place-gold-writeup)
- 发布时间：2026-07-10

## 一句话

把 simulator 大批量放到 GPU，以高维 future/source-target/fleet-size features 支撑小模型；BC 快速迭代，再分别对 2P/4P 做 PPO self-play 和部署 bias calibration。

## 作者报告的速度

在相近随机发射负载下：

- reference CPU environment：约 150–300 transitions/s；
- batched GPU simulator：约 300K–700K transitions/s/GPU。

这是 Orbit Wars 作者报告，不是本项目实测。它说明 GPU 重写可以跨三个数量级，但必须同时计算规则和候选 future features，单纯把神经网络放 GPU 不够。

## 表示与输出

模型把游戏看成 planet set、所有 directed source-target launches、每条 launch 上的 fleet-size grid。

- global：live opponents、turn、ship rank、fleet/garrison/production、comet、per-player shares；
- planet：owner、signed garrison、production、comet TTL；
- future planet：32 future buckets 的 incoming ships、target balance、reachable garrison；
- source-target：相同 future buckets 上的时间、相对未来位置距离、sun clearance；
- fleet-size：每条 edge 上 64 个候选 sizes，包括 ships、arrival、defenders、commitment、post-combat margin。

模型输出 launch count、source-target logits、fleet-size logits、PPO value。动作编译器选择 launch count、edge、size，瞄准预测未来 target，并丢弃非法/超预算发射。

## BC、PPO 与 calibration

BC 数据来自 Kaggle competitors 和内部 model self-play。replay 要追踪实际 fleet，反推出 intended target，再监督 launch count/target/size。BC 使架构实验更便宜，也用于小→大模型转移。

PPO 通常从 BC 起步，先 warm up value，再更新 policy；用多个历史版本 self-play，clipped PPO、KL check、entropy、LR schedule；作者说 from-scratch 也能变强但更慢。2P/4P 从 BC、RL 到 calibration 全部分开。

部署时 greedy 选择，但搜索 launch-count bias 和 fleet-size bias，用 head-to-head 修正系统性 under/over-sending。这是一种便宜的 policy calibration，不是重新训练。

## 缺失信息

文章没有给出模型层数、参数量、精确 PPO 总步数、GPU 型号或端到端 learner SPS，也没有公开代码链接。因此不能仅凭 300K–700K simulator SPS 推断完整训练速度。

## 迁移价值

本项目已完成同一思想的 Kaggriculture JAX 重写，并实测 batch4096 simulator-only 1.56M transitions/s。第 19 名最值得迁移的是“GPU 上批量构造 future/candidate features + BC 做快消融 + PPO 收尾 + 小规模部署 calibration”，但其 all-pairs×64 size 张量在 Kaggriculture 要改成 unit-task/market quantity candidates，并先做 top-k 稀疏化。
