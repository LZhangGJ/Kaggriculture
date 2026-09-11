# 第 7 名路线迁移：Top-k 任务剪枝 + 单因素 100M 实验

源方案见 [第 7 名摘要](../sources/07_STRUCTURED_JAX_PPO_SOURCE_DIGEST_ZH.md)。

## 迁移结论

这不是一套单独网络，而是本项目所有 RL 路线的实验操作系统。建议把它设为默认治理：每个改动固定 50M/100M transitions，单因素训练，再过固定 round-robin；同时把昂贵任务规划从全枚举剪成 model top-k + random exploration。

## Candidate pruning

每个 unit 候选生成顺序：

1. 硬性任务全集：今日未浇且会死亡、动物第二天未喂、格上 yield 将满/衰减、终局前必须入仓/卖出；这些永不剪。
2. 便宜规则 shortlist：按距离、deadline、保守 ROI 取 top16。
3. 模型先用低成本 logits 排序，保留 top4。
4. 从剩余合法候选随机取 4，保证发现新策略。
5. 只对最终 hard + 4+4 做精确 schedule feasibility、future market 和 edge MLP。

market 候选同理：保留 cash/terminal 必需项、top2 model items + 1 random item，再决定数量和 order slot。

需要记录 `candidate recall`：最终强 action 是否曾被预剪掉。可用低频全枚举 oracle rollout 抽检；若 hard task recall<100% 或 champion action recall<95%，先修候选，不怪模型。

## Auxiliary future heads

从 rollout 免费监督：

- t+6/24/72/168 的 plot type/yield/maintenance；
- 下一日 own production 和预计 shed occupancy；
- market price buckets；
- terminal banked cash、是否完成 liquidation；
- 对手公开资产变化。

辅助 loss 只训练 trunk，提交删除。分别消融每一组，不把五种辅助头一次性加入。

## Early truncation

Kaggriculture 没有消灭终局。安全版只截断明显非法/无行为 debug episodes；训练优化可对 critic win probability>0.995 且通过保守 cash upper-bound proof 的状态 value-bootstrap。没有 proof 时不能把“领先很多”当终局，因为市场/动物资产可能翻盘。

## 实验台账

每个实验固定保存：base checkpoint、代码/feature schema hash、seed set、JIT signature、参数、实际 SPS、100M 完成时间、训练曲线、512/2048-game matchup、失败原因。禁止一条 run 同时改 observation、model size、entropy 和 opponent pool。

推荐一轮：

- 50M smoke：排除坏实现；
- 100M正式：与 current baseline + 4 个历史/专家对手各 512 games；
- 若胜率差在噪声区，重复第二 seed 或拉到 2048；
- 合格 changes 汇入新 baseline，再进入下一轮。

## 必测负假设

Orbit 第 7 名发现 live-self、double batch、reward shaping、复杂 league 可能更差。Kaggriculture 不直接采信这些结果，但应优先做对应消融：

- frozen recent vs live self vs history pool；
- terminal-only vs potential shaping；
- batch 翻倍但保持 update 数/保持 step 数两种公平口径；
- top-k 4+4 vs 8+8 vs full candidate；
- entropy 固定 vs per-head schedule。

## 本机迭代速度

1–3M 主模型目标 15k–50k transitions/s。100M 热循环 0.6–1.9 小时，加固定 arena/写 receipt 后，正常可在 3–8 小时形成一个可信单因素结论；一天 2–4 个正式实验，而不是声称 200 个 Orbit 规模实验可原样复现。

## 价值

单看上分潜力不一定最高，但它能显著降低“以为变强、实际只会打自己”的风险。推荐从第一条 RL run 就执行，而不是模型成型后补评估。
