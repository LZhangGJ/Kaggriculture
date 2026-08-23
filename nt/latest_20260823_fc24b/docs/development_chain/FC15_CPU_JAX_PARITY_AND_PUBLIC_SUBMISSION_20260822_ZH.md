# FC15 CPU / JAX 一致性与 Public 提交记录

## 结论

FC15 的 Python 提交版已经通过严格一致性验收，并提交到 Kaggle Public。

- Kaggle reference：`55684835`
- 提交时间：`2026-08-22 06:23:46.177000`（Kaggle 返回时间）
- 提交状态：`PENDING`
- 描述：`NT FC15 strict JAX-parity fusion plus split weed-HIRE guard`

## 实际提交物

- `main.py` SHA256：`9E486F99B8138ABFA8016A0D176CCB5C5144407650A21494F4C6C67C59D02526`
- `submission.tar.gz` SHA256：`8104A68AC24A63E6911C0E0997DC62675C353A92CE0C9F6974EB896AD91F4549`
- 压缩包只包含根目录下的 `main.py`。

## 严格 JAX / Python 验收

在官方 `kaggle-environments 1.32.7` 下生成 Python 逐步轨迹，再与 GPU JAX FC15 比较：

- 2 个对手：Ray K320、G04 Soil Rain；
- 8 个固定 seed；
- 双座位交换；
- 共 32 个完整对局上下文；
- 动作完全一致：32/32；
- 每一帧官方状态完全一致：32/32；
- 终局资金完全一致：32/32；
- 首个动作或状态差异：无。

因此，本次提交的 Python 版不是近似移植，而是在该验收域中逐步复现已接受的 JAX FC15。

## CPU 推理验收

- 顺序动作：23,008 次，全部与冻结轨迹一致；
- 平均动作耗时：0.4115 ms；
- P99：1.2943 ms；
- 最大：80.3329 ms；
- 验收上限：1,000 ms；
- 结果：PASS。

## 本地强度摘要

Critical15 配对 Arena 共 1,920 局：

- 平均得分率：94.3229%；
- 最低单对手得分率：87.5%；
- 相对 FC14：救回 18 局，伤害 0 局。

Public 分数尚未返回；该分数用于验证当前天梯分布下的真实弱点，不反向改变上述逐步一致性结论。
