# Kaggriculture Top 10 + Ranks 21–40 宏观意图聚类

这是 2026-08-25 从公开 replay 恢复并独立聚类的宏观路线结果。数据粒度是一个目标 submission 在某个 Episode 中的一个座位，共 3,397 个座位、30 个 submission。

## 核心结果

- 距离：`D = 0.45C + 0.35G + 0.20T`
- 聚类：hierarchical agglomerative clustering，average linkage
- 主切点：`D ≤ 0.12`
- 临时路线族：677
- 单例族：566
- 最大族 N001：1,764 个座位，占 51.9%
- 前 10 族覆盖：73.6%
- 0.04 / 0.12 / 0.20 三个切点分别得到 902 / 677 / 377 族

`N001…` 是这批新数据内部的临时编号，尚未与历史 `G001…G275` 对齐。

## 目录

- `input/intent_features_v1.npz`：布局、阶段动作和累计动作特征
- `input/intent_seats_v1.csv`：特征行对应的 Episode / 座位 / submission 元数据
- `clusters/cluster_assignments_v1.csv`：逐座位聚类结果
- `clusters/cluster_profiles_v1.csv`：族画像、medoid 与终局构成
- `clusters/submission_cluster_summary_v1.csv`：逐 submission 的路线多样性摘要
- `clusters/cluster_prototypes_v1.npz`：medoid、共识阶段表和众数布局
- `clusters/macro_intent_distance_v1.npy`：5,768,106 个成对距离
- `clusters/average_linkage_v1.npy`：完整 average-linkage 树
- `clusters/macro_intent_clustering_QA_v1.ipynb`：已执行的 QA notebook
- `clusters/validation_report_v1.json`：独立复算和一致性验收
- `scripts/`：恢复、合并、聚类及 notebook 构建脚本

## 复现聚类

```powershell
python -m pip install -r requirements-analysis.txt
python scripts/cluster_macro_intent.py `
  --input-dir input `
  --output-dir clusters_rebuilt `
  --threshold 0.12
```

如果需要重建 notebook：

```powershell
python scripts/build_cluster_notebook.py `
  --feature-dir input `
  --cluster-dir clusters_rebuilt `
  --output clusters_rebuilt/macro_intent_clustering_QA_v1.ipynb
```

## 验收

- 3,397 个座位键唯一且与特征数组逐行对齐
- 距离全部有限、非负，linkage 距离单调
- 199 个随机样本对独立复算，最大绝对误差 `5.12e-08`
- 不依赖 `fcluster` 手工重建 0.12 切分，与保存的 677 族完全一致
- cophenetic correlation：0.9744
- QA notebook 16 个单元、6 个代码单元，零错误执行

逐动作明细 `macro_events_v1.csv` 约 109.8 MB，没有提交到 Git：它超过 GitHub 普通 Git 的单文件限制，而且聚类只依赖这里提交的两个 `input/` 文件。原始 replay 也不随仓库分发。
