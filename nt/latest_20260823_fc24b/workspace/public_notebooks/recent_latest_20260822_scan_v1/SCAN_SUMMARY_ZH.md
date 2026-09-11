# Kaggriculture 公开方案实时更新扫描（2026-08-22）

## 扫描口径

- Kaggle Code 按 `dateRun` 和 `voteCount` 实时扫描；
- 与本地 2026-08-20 及更早快照比较；
- 重新下载候选的最新 Notebook 源码并计算代码哈希；
- 标题中的分数、排名和作者本地胜率不自动视为当前 Public LB 成绩；
- 优先判断是否存在新策略、新路由或新动作字节，而不只看最后运行时间。

## 结论

除 Boatlee V21 外，本次至少发现五个值得继续验证的实质更新：

1. Prvsiyan Soil V26-H；
2. Prvsiyan Moon V92；
3. Kaito V39 History Gate；
4. Steven E284 Hadouken；
5. Salem HarvestForge-X（3094 页面最新版本）。

其中 Boatlee V21、Soil V26-H、Moon V92 和 Kaito V39 最值得优先迁入 JAX Arena；E284 值得先做两个经济规则的独立消融。

## A 级：应优先进入 JAX 验证

### 1. Boatlee V21-R1 Public-State Route Portfolio

- Kaggle：`boatlee/v21-r1-public-state-route-portfolio`
- 最近运行：2026-08-21 15:17:59 UTC；扫描时 9 票；
- 源码 SHA256：`c6f96a8521dc9aa369b6f27e5b36b9d481e5c1688f50c8ca215c3bb53f1f9eb8`；
- 不是 V20 的小补丁：V20/V21 行级相似度只有约 0.024；
- 同时内置 Moon、Mutoy、Munib 三条完整路线；
- 根据公开状态选择路线：对手开局雇工/现金、前三个商店、对手在 step 217 附近的公开花费；
- 在路线兼容时只迁移市场出售差量，避免破坏完整生产链；
- 作者本地屏：192 局 164/0/28，Top-front 48 局 45/0/3，回归 32 局 26/0/6。

判断：这是本次最重要的新对手。它与我们目前“完整拳法 + 动态选择”的方向高度一致，而且比单一路线更接近当前天梯动态博弈。

### 2. Prvsiyan Soil V26-H

- Kaggle：`prvsiyan/kaggriculture-frontier-the-soil-remembers-rain`
- 最近运行：2026-08-22 01:43:50 UTC；81 票；
- 相比本地 8 月 11 日快照，代码从约 60k 字符增加到约 163k，属于实质更新；
- 保留牛、羊、瓜主路线，新增：
  - 基于公共相对利润的七格番茄替换；
  - 终局羊饲料收益期限判断；
  - 只在不会产生下一次可收羊毛时停止喂养；
- 作者报告 current-top 34-8、hard-64 58-6，并包含 exact-public-2913 的 16-0 门；
- 作者明确说明仍怕蛋路线和重羊毛开局。

判断：比普通“看到番茄贵就改种”严谨，值得作为 FC15 的新强对手和番茄/终局饲料消融来源。

### 3. Prvsiyan Moon V92

- Kaggle：`prvsiyan/kaggriculture-frontier-the-moon-counts-melons`
- 最近运行：2026-08-22 03:19:17 UTC；81 票；
- 相比本地 8 月 11 日快照，代码从约 20k 字符增加到约 140k；
- 新增受限的胡萝卜、番茄、鸡蛋、草莓流量、容量和终局模块；
- 作者本地报告 score-stratified 52-4、current-top 41-1；
- 但 hard-100 只有 59-41，且对 Fabian 为 0-20，作者主动披露了明显整条开局克制。

判断：平均面很强，但不是无短板方案。适合进入 Arena，当作强大但可被路线克制的组合对手。

### 4. Kaito V39 History Gate

- Kaggle：`kaitofukami/103-126-untouched-future-top-30-v39-history-gate`
- 最近运行：2026-08-21 16:38:07 UTC；38 票；
- 不是只看单个时刻，而是在 step 96、120、122、132、144 保存公开历史；
- 只有候选路线在前缀上保持兼容，才允许在 step 144 切换；
- 用距离门拒绝陌生状态，陌生局面回退到强通用路线；
- 作者报告 untouched future 103/126；26 个公开 Notebook 面板为 159/208；
- 标题的 103/126 是本地 Replay 结果，不是 Public LB 分数。

判断：它的价值主要是动态路由架构，尤其证明“同一时刻盘面相同，但历史不同，最佳后续可能不同”。应优先研究其历史特征和安全切换，而不是只照搬路线。

### 5. Steven E284 Hadouken

- Kaggle：`stevenleehans/kaggriculture-e284-hadouken`
- 最近运行：2026-08-22 06:04:26 UTC；6 票；
- 1.32.7 下对 Starter 报告 181,022；这是终局现金，不是天梯 rating；
- 两个机制补丁：
  - 当牛奶/羊毛接近价格下限且小麦昂贵时，跳过无净收益的晚期喂养，同时避免连续两次不喂导致逃跑；
  - 肥料只保留执行路线所需安全库存，额外肥料尽早出售；
- 作者 disjoint confirmation 40 个格子为 26/0/4，相对基线平均约 +1,154；
- 核心仍是 E284 固定动作路线，对 1.32.7 非常敏感。

判断：整套路线未必是最强，但两个经济补丁简单、可解释、值得在 FC15 上分别做消融。

### 6. Salem HarvestForge-X（3094 页面）

- Kaggle：`salemali7/3094-score-kaggriculture`
- 最近运行：2026-08-21 18:50:10 UTC；64 票；
- 当前源码和本地 8 月 11 日源码哈希不同，属于实质更新；
- 从三个公开高分 Replay 的多数动作重建 8 牛 4 羊路线；
- 增加手数对齐和杂草阻塞恢复；
- 作者报告本地双座位 30 seeds 共 60/60，但明确说不是官方竞赛分数。

判断：值得作为新的 8C4S 锚点，但“3094”不能直接当作当前这份字节的已确认 Public rating。

## B 级：有变化，但优先级较低

### Steven X567（仍发布在 X544 slug 下）

- 当前 Notebook 内容已经是 X567，而不是标题中的 X544；
- 只在非镜像、公开牛奶预测较低时，把 `SELL MILK 6` 从 step 264 推迟到 265；
- 独立确认平均仅约 +4.72，属于小幅市场时机补丁；
- 可留作后续微调，不应排在新完整路线前。

### EcoBot V2

- 12 票；具有动态牛羊规模、工作量雇工、自产小麦、作物 conveyor 和任务分配；
- 但 Notebook 的主要验收仍是对 Starter 终局现金超过 60,000；
- 目前没有足够强的跨高分对手证据。

判断：架构思路可参考，现阶段不应当作高分强敌。

### Premium-First Market Agent

- 4 票；报告对 Starter 约 175k；
- 主要是 8C4S 固定链、市场出售排序、杂草修复和前置出售；
- 缺少可信的强对手、固定 seed、双座位大样本比较。

判断：可作为机制参考，不足以按当前证据进入第一批强敌。

## C 级：看起来更新，实际不应视为新强方案

### Flex Multi-Route 当前版本

- 85 票，但当前 Notebook 的 `VARIANT` 是 `v20_adaptive_r1_multi_route_exact`；
- 解码后预期 SHA256 为 `8ac34abce129cf5c9456776c90edf7d2233b3a280bbdcf7622628825ef3669a0`；
- 与本地 Boatlee V20 `main.py` 完全相同。

判断：这是 Boatlee V20 的相同字节，不是一个新的独立 Agent，不需要重复迁入 JAX。

### Steven X578 slug 当前版本

- 当前 Notebook 实际发布的是 X594 Ryo-live 实验；
- 作者明确写明相对 X578 的本地配对门为负，不主张替代 X578。

判断：保留源码作失败研究，不加入强方案池。

## 推荐下一步顺序

1. 先将 Boatlee V21 精确迁入 JAX，并与 FC15、FC2B、K320、G04、X562 做双座位 Arena；
2. 迁入 Soil V26-H、Moon V92；
3. 迁入 Kaito V39，单独验证历史门是否真正增加胜局；
4. 对 FC15 做 E284 的“晚期喂养价格门”和“肥料早售”两项独立消融；
5. 将 Salem 最新版加入 8C4S 对手组；
6. 暂不投入 EcoBot、Premium-first、X594；Flex 当前版直接映射为已有 Boatlee V20。

## 本地快照

本次下载目录即本文所在目录，包含每个候选的最新 `.ipynb` 和 `kernel-metadata.json`。原始文件保持不改动，便于后续提取 `main.py`、计算哈希和执行 JAX 精确迁移。
