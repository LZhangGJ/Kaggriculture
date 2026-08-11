# 第 19 名路线迁移：GPU Candidate Tensor + BC/PPO/Calibration

源方案见 [第 19 名摘要](../sources/19_GPU_SIM_BC_PPO_SOURCE_DIGEST_ZH.md)。

## 迁移结论

模拟器部分已经落地并超过 Orbit 第 19 名报告区间：本地 Kaggriculture simulator-only batch4096 为 1.56M transitions/s。下一步应迁移其“GPU 上生成 future/candidate features、BC 快速选架构、PPO 提升、最后做小 bias calibration”，而不是再次重写规则核心。

## Candidate tensor

Orbit 的 planet-pair×64 fleet sizes 改为两个张量：

### Unit-task tensor

`[B, U, K, M, F]`：U 为 masked unit slots，K 为 top-k tasks，M 为 2–4 个 modes。F 包含 ETA、deadline、cash/material cost、expected yield、shed delta、maintenance/terminal margin、legal flags。

### Market tensor

`[B, product/purchase, quantity_bucket, order_slot, F]`：包含成交前/后价格、库存冲击、cash、shed、未来城镇需求、自己 planned supply、terminal slack。

不要做所有 `U×36 tiles×所有 task×所有 quantity` 的稠密巨张量。先 hard candidates + heuristic top16，再 model top4+random4，精算最终候选。

## 小模型

第 19 名强调 CPU inference 小模型。本项目建议 1–3M：

- plot/unit/market encoders；
- 4 层 d128–192 attention；
- launch-count 对应 `job-edit count`；
- source-target 对应 `unit-task`；
- fleet-size 对应 task mode/quantity；
- value 预测胜率。

输出 job-edit count 可以避免每 turn 给所有 units 改任务：先决定改 0/1/2/3 个，再选 unit-task-mode；未改 jobs 继续。

## BC 用作架构筛选

对相同 train/validation episode split，快速比较：future bucket 数、top-k、edge features、model width、quantity bins。指标不仅看 action accuracy，还看 task-family accuracy、legal rate 和用 BC policy 打固定 arena 的胜率。BC winner 再进入 PPO；from-scratch 小对照必须保留。

若扩模，可用旧 learned agent 自生成演示做 teacher→larger transfer；这不是最终部署蒸馏，而是训练 warm start。数据必须记录 generator checkpoint，避免评估混入。

## PPO 和历史对手

先 value warmup 10–20M，再 terminal W/D/L PPO 100M；对手为 current frozen、history、v16、specialists。每 10M 做 head-to-head/round-robin。2P-only Kaggriculture 不需要拆 2P/4P，但建议按 strategy family 做 separate calibration table。

## Deployment calibration

借鉴 launch-count/fleet-size bias，对少量可解释 logits 做搜索：

- job-edit-count bias：更保守地延续计划还是频繁重排；
- task-mode bias：MINIMUM/SAFE/MAX_ROI；
- sell-quantity bias；
- terminal liquidation bias。

在冻结 checkpoint 后，用固定 train-calibration seeds 搜索；最终只在完全独立 audit seeds 验证。bias 不得用 Public LB 单点调。

## 本机速度

现有 simulator 不是瓶颈，重点测 feature+model+PPO：目标 15k–60k transitions/s；100M 0.5–1.9 小时热循环。若 candidate tensor 使 SPS 降到 <10k，先稀疏化/减少 future buckets；第 19 名的 300K–700K 是 simulator 指标，不能用来承诺完整 learner。

## 实现顺序

1. 在现有 `gpu_sim` 外新增 feature/action wrapper，不改已验收 rule core。
2. 做 unit-task/market tensor reference 实现与 JAX parity。
3. BC 架构赛。
4. 100M PPO + history pool。
5. bias calibration + policy-only warmup/fallback submission。
