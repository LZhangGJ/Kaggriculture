# 高手 replay 中盘 outcome-SFT pilot

本实验与 R1 `corpus_slot_v2` 完全隔离，只回答一个较窄的问题：能否从日界前可见状态，安全地监督
“下一日内真实成功建立的格子最终选择了哪一种作物/动物”。它不是 R1 精确 prefix ABI 的替代品，也未接入
生产策略。

## 数据与冻结提取链

- 输入索引：`work/expert-replay-slot-sft-v1/pilot-inventory.csv`。
- 从 top100/12 局、旧 top60/8 局和 `data/replays/raw` 扫描 1,976 个严格命名的完整 JSON，按 episode ID
  全局去重；pilot 从当前 top100 manifest 以队伍 round-robin、队内 `SHA256(episode_id)` 排序选 128 局，
  不按胜负或终局钱过滤，共覆盖 100 支队伍、170 个目标玩家侧，LB 分数 2772.3–3094.8。
- 先运行成熟的 `scripts/analyze_macro_route_library.py`，得到 170 个目标侧、100 支队伍的既有宏路线聚类；
  没有另写简化聚类器。
- 日内因果证据完全来自冻结
  `full-daily-corpus-v1/frozen/build_full_daily_route_corpus.py` 的 native ordered receipts，31 进程处理耗时
  24.09 秒。48/128 局严格通过，80 局 fail-closed：73 局为新版/异常空动作数组，7 局为旧 aligner
  未知 action op。失败局没有被猜测解析。

## 安全标签语义

只取目标高手侧、`step >= 288`、`quality_status=computed` 且 native unit receipts 完整的日状态。输入严格是
`steps[start_step][player].observation` 经成熟 `ObservationTokenizer` 编码的日界状态；标签来自随后一天窗口
内 receipt 证明 `effect=true` 且 before/after 证明真正创建资产的 `PLANT/PLACE`。

- 直接新建：日界时为空地或相应空畜舍，随后 native-success 建立。
- 收获后继：日界时为成熟有限期作物，同日先有 native-success HARVEST 将其移除，再有 native-success
  PLANT。
- 固定顺序为距中央仓库距离、再按原棋盘 cell index。
- mask 只声明可由日界棋盘证明的结构合法性，契约明确为
  `mask_precision=structural_only/resource_unknown`。日内现金、购买、PICKUP 和单位背包无法从日界状态给出
  与 R1 相同的精确 prefix resource mask，因此没有伪造。
- 未发生 placement 的格子没有被猜成 `SKIP` 标签；当前损失只覆盖 native-success 正标签。因此这是
  conditional kind SFT，不是已经完整监督的九类 actor。

最终 mmap shard 为 995 个状态、8,228 个 slot（889 train / 106 heldout），来自 48 个 native-complete
replay。10,249 个真实成功创建事件中，8,228 个（80.28%）可从日界安全归因；结构合法率 100%，相同输入
标签冲突为 0。归因包括 945 个直接新建和 7,283 个收获后继。类分布为 WHEAT 5,109、CARROT 2,262、
TOMATO 496、STRAWBERRY 307、MELON 53、SHEEP 1；本 pilot 没有安全的 GOOSE/COW 正标签。

模型 forward 不包含 receipt 的 action-time `before.inventory`、未来 payload、seed、玩家/队伍/对手身份。
玩家与 LB 分数仅留在 manifest 的 audit metadata。当前 observation tokenizer 包含完整当前公开棋盘、市场和
行动方 private，但没有重建 R1 内部 commitment 或公开历史 ledger belief。

## 首次 CPU 训练

独立 93,121 参数 GRU actor 在 CPU 单线程完成 5 epoch：

- one-batch（8 个真实状态）loss `1.8368 -> 0.000260`，masked accuracy 100%；
- train loss `1.8240 -> 0.6567`，masked accuracy `11.93% -> 75.66%`；
- heldout loss `1.8591 -> 0.7648`，masked accuracy `8.13% -> 69.60%`；
- train/heldout illegal argmax 均为 0；训练吞吐约 864 slot-example/s。

这些数字只证明二进制数据、因果输入、mask 和优化闭环可运行。由于没有负 `SKIP`、精确资源日历和历史
belief，该 checkpoint 禁止部署，也不能与 `autoregressive-slot-bc-v2` 的 R1 actor 指标直接比较。

稳定产物：

- `work/expert-replay-slot-sft-v1/expert-outcome-shard-pilot128/manifest.json`
- `work/expert-replay-slot-sft-v1/expert-outcome-student-pilot128.pt`
- `work/expert-replay-slot-sft-v1/expert-outcome-student-pilot128.metrics.json`
- `work/expert-replay-slot-sft-v1/macro-clusters.json`
- `work/expert-replay-slot-sft-v1/native-daily-pilot128/manifest.json`

复现实验入口是 `scripts/build_expert_replay_slot_sft.py` 的 `inventory`、`shard`、`train` 三个子命令；下载
仍由原单线程 collector 负责，本脚本只并行处理已经完整落盘、严格命名的文件。
