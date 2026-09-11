# EBA26v2

这是 Kaggle submission `55576378` 的确切本地归档。

## 设计

- 完整赛季 Route17 主路线作为稳定计划。
- 第 168 步读取公开商店：存在 `YARN_STORE` 且不存在 `ICE_CREAM_SHOP` 时，切到 Route18（6 牛、12 羊、100 地）。
- 运行时只使用公开状态；禁止读取对手身份、seed、未来事件、终局结果和对手私有状态。
- 局部控制包括杂草修复、需求感知卖出顺序、近镜像高价商品提前出售、移位数量偿还和终局清仓。

## 文件

| 文件 | 含义 |
|---|---|
| `main.py` | 线上提交使用的 exact Agent |
| `submission.tar.gz` | Kaggle 上传包，只含 `main.py` |
| `local_acceptance.json` | CPU 包、双座位和在线延迟验收 |
| `build_receipt.json` | 路由覆盖条件和源码哈希 |
| `submission_record_at_upload.json` | 上传时快照，保留当时的 `PENDING` 状态 |
| `upload_receipt.json` | Kaggle 上传成功收据 |
| `leaderboard_receipt_20260818.json` | 后续查询得到的 `COMPLETE` 与 Public 分数 |
| `official_weak6_holdout_rows.csv` | 官方 1.32.7 的 384 局逐局明细 |

## 验证结果

- 官方环境：`kaggle-environments==1.32.7`。
- CPU 双座位 smoke：2/2 完成，每局 720 帧。
- 在线决策平均延迟：0.1397 ms。
- weak6 未见面板：384 局、317 胜、82.55%，平均分差 +5,517，最弱对手胜率 70.31%。
- public G02：62/64，96.88%。
- public G04：59/64，92.19%。
- Public：2026-08-18 09:08 JST 查询为 **2378.4**。

## 哈希

```text
main.py            DD8CA878FB7C71195EF5CFDDCF61D9B1161F317B052B0E45104B8442B6E59072
submission.tar.gz  9B906F81189179B442649F0B4388EF0D0E1CA914190F7250848993FF320B9E65
```

`submission_record_at_upload.json` 中的 `PENDING` 是历史事实，不应改写；当前榜单结果单独记录在 `leaderboard_receipt_20260818.json`。
