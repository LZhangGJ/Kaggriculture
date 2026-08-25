# Top 10 + Ranks 21–40 宏观意图聚类 v1

本目录对 `top10_plus_ranks21_40_macro_intent_v1` 中的 3,397 个座位先做独立聚类。它回答的是“这批新 replay 自身形成什么路线结构”，还没有把路线映射回历史 `G001…G275`。

## 方法

- 成对距离：`D = 0.45C + 0.35G + 0.20T`
- C：五个布局锚点的生产类别计数差异
- G：五个布局锚点的格位类别差异
- T：十个阶段的累计宏观动作差异
- 聚类：hierarchical agglomerative clustering，average linkage
- 主切点：`D ≤ 0.12`
- 临时编号：`N001…`，按族规模降序、奖励中位数降序排列
- 代表样本：精确 medoid（族内平均距离最小的真实座位）

## 主结果

- 677 个临时路线族
- 566 个单例族
- 最大族 N001：1,764 个座位，占 51.9%
- 前 10 族覆盖 73.6%；前 56 族覆盖 80.1%
- 切点 0.04 / 0.12 / 0.20 分别得到 902 / 677 / 377 个族

长尾不是均匀分布的。566 个单例全部来自 6 个 submission，其中 rank 3、rank 1、rank 2 和 rank 39 占绝大多数。最大族则横跨 25 个 submission，说明当前池同时存在一个共享的主流宏观骨架和少数高度逐局变化的 Agent。

## 文件

- `cluster_assignments_v1.csv`：每个座位的临时族、族规模、medoid 标记和到 medoid 的距离
- `cluster_profiles_v1.csv`：每族规模、奖励、胜率、代表样本和终局构成
- `submission_cluster_summary_v1.csv`：每个 submission 的族数、单例比例和主族覆盖
- `cluster_prototypes_v1.npz`：临时标签、medoid、共识阶段表与众数布局
- `threshold_sensitivity_v1.csv`：0.04–0.20 的切点敏感性
- `macro_intent_distance_v1.npy`：5,768,106 个成对距离
- `average_linkage_v1.npy`：完整 average-linkage 树
- `macro_intent_clustering_QA_v1.ipynb`：已执行的质量检查 notebook
- `cluster_overview_v1.png`：阈值与集中度概览图

## 验收结论

聚类产物已通过：输入粒度与数组对齐、距离有限且非负、linkage 单调、族规模回算、每族唯一 medoid、阈值单调性、199 个随机样本对的独立距离复算，以及不依赖 `fcluster` 的 0.12 切分重建。抽样距离最大绝对误差为 `5.13e-08`，手工切分与保存的 677 族完全一致。

下一步应把这些 `N###` 与历史 275 族的 medoid / 原始成员联合比较，再决定延续旧 `G###`、合并，或创建真正的新族。
