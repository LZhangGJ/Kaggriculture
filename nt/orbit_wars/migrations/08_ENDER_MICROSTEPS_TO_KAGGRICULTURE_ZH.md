# 第 8 名路线迁移：同回合联合动作 Micro-steps

源方案见 [第 8 名摘要](../sources/08_ENDER_MICROSTEPS_SOURCE_DIGEST_ZH.md)。

## 迁移结论

Kaggriculture 的 farmer、hands 和 market orders 有强资源依赖，最适合用 micro-step 自回归构造联合动作。但不要像 Orbit 每个 micro-step 重跑完整 trunk；先编码一次状态，再用轻量 GRU/policy heads 顺序决定。

## 自回归顺序

一个环境 turn 内：

1. `HALT/ASSIGN_UNIT`；
2. 选 unit（farmer 或 hand）；
3. 选 `continue/new task/one-shot action`；
4. 选 target 和 mode；
5. 写入临时资源账本并 mask 已占用 unit；
6. 重复至所有 units 或 halt；
7. `HALT_MARKET/ADD_ORDER`；
8. 选 order type/item/qty；写入临时 cash、shed、seed 和 order-slot 状态；
9. 最多 2–4 market intents 后 halt。

临时账本必须重现：同作物 PLANT 原子种子统计、每单位只动作一次、单位阶段早于市场、shed capacity、SELL 只从 shed、最多 10 orders、HIRE 同回合不能行动。

## 模型

- plot/unit/market/global Transformer，4 层 d192，约 2–4M；
- trunk 每环境 turn 只跑一次；
- GRU context 128–192 串起 unit→task→target→mode 和 market slots；
- 每采样一步用小 MLP 更新相关 token/ledger embedding，而不是全量重算；
- value 以环境 turn 为主；PPO log-prob 是所有 micro-actions 的 joint sum。

是否把每个 micro-step当 PPO 时间步要消融。Kaggriculture 同回合动作按解释器固定顺序结算，但 agent 一次提交整个 dict；更稳妥的 V0 是把整组 micro-actions视为一个 environment transition，GAE 只跨真实 turns，内部只累计 log-prob/entropy。

## 先验与 entropy

对高频安全行为使用非均匀先验：

- 大多数 unit 默认 `CONTINUE`；
- hard maintenance 候选 prior 高；
- market 默认 NOOP，不鼓励每回合乱下单；
- terminal window 提高 DROP/SELL prior。

用 KL-to-prior 代替所有类别均匀 entropy，另给战略投资 action 一定探索下限。先验只影响探索正则，不应硬编码最终选择。

## 训练分阶段

1. 只有 farmer + 1 market slot；
2. 加固定 1–2 hands；
3. 加可变 hands mask 和 2 market slots；
4. 最后开放多 unit 同 crop PLANT、动物物流和 order sequencing。

每阶段从上一 checkpoint warm start。对每个动作组统计平均 micro-steps、halt 位置、mask rate、joint entropy 和无效 raw action。

## 搜索增强

像 Ender 一样可在日初/投资/终局节点采样 8–16 个整组 micro-action sequences，用 JAX 推进 6–24 turns，再按 value 选。对手动作由一个小 opponent model 或历史 policy 批量采样。搜索模型可以蒸馏成 0.3–0.8M 小网络，但这属于后期增强，不是训练前提。

## 性能

单次 trunk + head autoregression 目标 8k–30k transitions/s；若每 slot 重跑 trunk，可能降到 2k–12k。100M 约 0.9–3.5 小时（轻版）或 2.3–13.9 小时（重版）。因此优先 policy-head-only，并把 max units/market slots 设为可观测合理上限。

## 风险与判据

- 表达力高但 credit assignment 更难；对比 independent heads 和 micro-step heads。
- 账本 bug 会造成训练合法、官方 no-op；要求 raw action parity 和 100% legality。
- 如果多 unit cooperation 没有提高固定 arena，而平均 micro-steps/turn 上升，应收窄动作而非继续加槽。
