# 中盘 Student 输入特征审计

> 2026-09-22。结论先行：`midgame_autofill_student.py` 中的 `global_dim=192`、
> `slot_dim=32` 目前只是网络 shape，**不是已经定义或已经抽取的 192/32 个字段**。
> 49,497 参数只证明小网络可行，不能证明输入信息完整。真实 slot teacher、mask、资源状态和
> 完整 canonical frame/commitment shard 与 loader 完成前，不应开始正式训练。

## 1. 线上信息边界

线上可以精确获得：

- `step/day/hour/player`；双方公开现金；双方 100 格棋盘的完整 tile 字段；双方 farmer、hands
  坐标、土地 mask、当日雇工数。
- 我方 private：12 类 shed、5 类 seeds、每个单位的随身库存及字典插入顺序。
- 9 类市场库存和价格、已解锁商店的有序列表。
- 从上述连续公开历史和我方已知动作推导出的统计量。

线上不能获得、也禁止作为 student 输入：

- 对手 shed、seeds、随身库存、当前内部计划和未来动作；未来商店、杂草 RNG；seed；对手名称、
  脚本或 family identity。
- 轨迹中保存的对手 private 只能作为 belief 辅助 target。`opponent_group/source_family` 只能用于
  split 和审计。

## 2. 逐字段状态

| 特征组 | 必需内容 | 当前真实来源 | 当前 student 状态 |
|---|---|---|---|
| 时钟 | step、day、hour、剩余总 tick、今日剩余 tick、剩余天数 | observation 精确；终局最后动作是 step718/day29/hour22 | **未接入** |
| 终局窗 | 各 kind 能否在 day29 前成熟/首次产出、最终收获和入仓剩余路程 | 可由时钟、tile、规则精确推导 | **未接入**；mask 也未实现 |
| 市场 | 9×库存/价格、当前价相对 base/地板距离 | observation 精确 | 原始轨迹有，模型未接入 |
| 商店/需求 | 8-bit 解锁、解锁顺序、剩余确定需求 cadence | observation + 官方规则精确 | 原始轨迹有，模型未接入 |
| 我方农场 | 100 tile 全字段、土地、现金、单位位置、hires_today | observation 精确 | 原始轨迹有，模型未接入 |
| 我方库存 | shed12、seeds5、每单位 bag12 与顺序 | own private 精确 | 原始轨迹有，模型未接入 |
| 对手公开资产 | 同样的公开 farm/tile/单位/现金 | observation 精确 | 原始轨迹有，模型未接入 |
| 市场历史 | 每步库存/价差、累计和近 1/3 日成交、有效性 mask | 原始轨迹可重建；C++ `PublicTradeLedger` 已跟踪产品1..7 | **未抽取**；WHEAT/FERTILIZER 无 ledger 反推 |
| 对手库存 belief | lower/upper、added/certain_added、最近销售时序 | C++ ledger + `SaleClock`；边界/地板处只有区间 | v2 ABI 已导出 step288 原状态；真库存不可观测 |
| 作物移除历史 | WHEAT/CARROT/MELON 最近最多32个移除年龄及有效计数 | `ObservedCropClock` 已在线维护 | v2 ABI 已导出 step288 ring/count/cursor |
| 自有承诺 | `book[100]` 的 kind/birth/chosen_day/length/successor/funded | C++ `Controller::live` 内精确 | v2 ABI 已导出首次 handoff 状态；后续恢复未完成 |
| joint 状态 | 两个 slot 的 stage/deadline/next_kind 等 | C++ `JointBundleState` 内精确 | v2 ABI 已导出首次 handoff 状态；后续恢复未完成 |
| executor pending | phase、market queue、每单位 plan/index/target、daily_need、expected、prepared seed、T3 missing water | C++ executor 内精确 | **未导出** |
| 销售承诺 | hold deadline、sale plan step/bucket/carry target | C++ `LocalSaleTiming/Planner` 内；正式 sale_dp 当前为0 | **未导出**；配置关闭时可置零并带 mask |
| 当前 partial bundle | 已选 cell→kind、各类计数 | teacher 顺序中精确 | 合成契约能重建，**真实 teacher 尚未输出** |
| partial 资源 | 剩余现金/种子/动物、未来 labor/flow/fixed calendar、feed/fertilizer/shed容量 | C++ portfolio 与 feasibility 可精确更新 | 目前只有11个占位字段，**不完整且无真实值** |
| legal mask | SKIP+5 crop+3 animal 在当前 prefix 下的可行性 | 必须复用 C++ planner/executor | **尚无导出**，不能由静态价格相减伪造 |
| 候选价值 | 356维聚合 state/portfolio 特征 | `portfolio_features()` 已实现 | 可作 critic 基线；缺历史、逐格空间和内部承诺，不能代替完整输入 |

## 3. 为什么 192 维 flat state 不足

仅双方棋盘就是 `2×100` 个结构化 token，每格至少含 kind/item、出生日期、产量、干旱/饥饿、
施肥/护理、寿命和当日服务 flags。再加可变数量的工人、随身库存和历史，不能先验压成一个未定义的
192 维手工向量而不丢掉关键可行性信息。

合理含义应是：`192` 是**结构化 encoder 输出的 latent 宽度**，不是原始字段数；`32` 同理是
tile/slot encoder 的 latent 宽度。完整输入至少需要：

1. 当前全局标量与 market/shop encoder；
2. 双方 100 tile 的共享权重 spatial/token encoder；
3. 我方单位与库存 token encoder；
4. 公开历史 belief encoder；
5. 自有 commitment/executor-state encoder；
6. 当前 slot + 已虚拟填充 portfolio + C++ 更新后的资源/mask。

这些 encoder 尚未实现，所以当前 scaffold 只能保留作接口草图。

## 4. 自回归 slot 的精确定义仍需 teacher

原始 hourly micro-action 无法确定某个 slot 的 expert kind：动作混有移动、维护、repair，一天内释放的
格子在日初可能仍是作物。九类标签只在 C++ teacher 先固定“本轮决策 slot 与 release 语义”后成立：

- `SKIP` 必须定义为“不在本轮给该 slot 增加 successor”，不能解释成挖掉 incumbent。
- 动物标签隐含 coop/pasture 建造，但资金、劳动、feed 和真实可建地块必须在 mask 中兑现；不把
  `max_animals` 这类人工规划上限当成环境资源或网络输入。
- 当前 R1 的 slot 顺序是 `(near_depot_distance, cell)`，不是普通 row-major；若做 BC，应沿用 teacher
  顺序，或用随机/置换不变训练显式消除顺序偏差。
- 每填一个 slot，必须由同一 C++ feasibility 状态推进 budget、seed/animal stock、portfolio
  flow/fixed/labor、feed/fertilizer 与容量，再产生下一个 mask。

C++ teacher v2 已导出候选 target map、356-D feature/key、四情景 terminal Q 与两类 reference；仍没有
slot mask 或逐 prefix 资源状态。它可服务 candidate critic 数据准备，**不能标成 `autofill-v1`**。

## 5. handoff 特殊点

`ReplayThenDynamicAgent` 在 step288 前逐帧调用 `dynamic.observe_external()`，所以 takeover 时
`PublicTradeLedger`、`SaleClock` 和 `ObservedCropClock` 已经 warm；这段历史不能丢。另一方面，
external observe 并未替 replay 构建 R1 的 `book/queue/plans`，所以首次 takeover 的 executor 承诺为空，
由首个 R1 plan 建立。后续各日则必须携带 R1 自己生成的 book/joint 状态。

因此 BC 样本必须明确 `first_handoff`，并区分“可从公开历史重建的 belief”和“student 自己过去决策形成的
内部承诺”。不能把二者都当成当天 observation 的静态特征。

## 6. 训练目标：只训练逐格 actor

项目已明确拒绝把 **candidate critic / candidate-ranking** 当成首训或过渡主线。唯一 actor 任务是按稳定
格序直接输出 `SKIP / 5 crop / 3 animal`；上一格的选择必须更新预算、库存和资源日历后再进入下一格。
完整候选的 Q/paired-advantage 数据只保留作旧 scorer 审计，不能训练一个替 actor 做整盘候选排序的旁路。

`teacher_bridge.cpp` 的训练专用 ABI 现已从真实 R1 greedy loop 导出每个 slot 的：

- `cell / proposed_kind / exact legal_mask / terminal`；其中标签是与后续 prefix 因果一致的
  preview 前逐格 proposal，最终全局可行性仍由同一 executor/preview 过滤；
- 决策前 `remaining_budget / selected_animals / owned_quadrants / slot_index / slot_count`；
- 12 类剩余库存；
- 未来 30 天逐日 `9 product flow + labor + fixed cash`，合计 347 维前缀资源状态。

独立诊断二进制已在真实 step288 双座 handoff 上通过 schema、标签合法性、有限值和重复确定性检查。下一步
数据 writer 必须把这些逐格字段与同一帧完整公开 observation、private state、历史 belief 和内部承诺一起
写入按 `(seed, opponent family)` 分组的 binary shard；任何一项缺失就 fail closed。`short_score`、候选
ID、对手身份和 RNG seed 都禁止作为 actor 输入。critic 若保留，只能共享 encoder 后预测该逐格策略的
value/baseline 或完整序列回报，不能改变 actor 的逐格动作语义。

同一状态的多个 R1 proposal 使用了不同隐藏超参数，可能在相同 slot 输入上给出冲突标签；它们不能同时
当作独立 BC 专家。首版每个状态只取生产 `SearchController::choose()` 按现行 `score()` 选中的 proposal，
并用完整 `proposal_key` 对齐其 slot trace。这里的慢 score 只在离线决定 teacher 标签，不进入 actor forward；
后续真实 suffix / RL 可以改进该逐格策略，但不得新增 candidate-ranking 网络。

## 7. 哪些字段必须原样保存，哪些只应派生

### 7.1 必须保真的线上可见事实

二进制 canonical state 必须保留下列原始语义；不能只留下 356 维聚合或 192 维 latent：

- 唯一时钟 `step`。本规则最后一次动作是 step 718，故 manifest 同时冻结
  `terminal_action_exclusive=719` 与 `turns_per_day=24`。
- 双方 `money / unlocked_mask / hires_today`，farmer 与有序 hands 的位置。
- 双方 100 格 tile 的 kind、crop/animal、birth day、yield、dry/hungry、fertilized/pending bonus、
  lifespan 以及 watered/fed/cared/fertilizer flags。格子位置由数组下标给出。
- 9 类市场 inventory 与观测 price；8 个商店的**出现顺序与重复项**。
- 我方 shed12、seeds5、每个有序单位的 bag12，以及 bag 的字典插入顺序；后者会影响容量不足时的
  `DROP`，不是可以随意排序的实现细节。
- 已执行的我方历史动作。动作属于产生下一帧状态的记录；构造 step `t` 的输入时只能读取 `<t` 的动作，
  不能把当前行为标签泄漏进输入。

原始轨迹中的对手 private、未来 shop/RNG/seed、对手名称和 family 均不得进入 canonical input。
对手动作也不是线上 observation；只保存公开结果，不把 replay 文件里额外可见的动作当输入。

### 7.2 必须保真的因果记忆与我方承诺

这些量不能从当天静态 observation 唯一恢复，必须由与线上相同的 C++ observer/controller 导出，并带
`present` mask：

- `PublicTradeLedger` 的 lower/upper/added/certain_added、成交量与有效性/地板拒绝 mask；
- `SaleClock` 的最近三日 own/rival 分时销售直方图；
- `ObservedCropClock` 的 ring、count、cursor，而不是只有最终 `weights()`；
- `book[100]` 的 kind/birth/chosen_day/length/successor/funded；
- `JointBundleState` 与两个 `JointSlot` 的 stage、deadline、source/next kind 等状态；
- 会跨帧影响执行的 pending/queue/plan index、target、daily need、prepared seed、deferred/not-before、
  service、T3 missing-water、working-capital 与 sale carry/deadline 状态。

逐格 student 的采样点固定为 `hour=0`、`SearchController::choose()` 之前。这个相位中 `plan()` 会重建很多
临时量，但不能凭阅读代码猜某字段“一定无用”后删掉。commitment exporter 的准入检查是：保存状态后
恢复到新 handle，继续喂同一批 observation，逐步 action、candidate key、mask 必须一致。首个 handoff
内部承诺为空是合法状态，必须用 `first_handoff=1` 区别于“该组根本没有导出”。

### 7.3 应在加载/编码时派生的量

以下不应作为另一份可能互相矛盾的 canonical 真值：

- `day/hour/remaining_ticks/remaining_days`：由 step 和上述终局常量精确计算；这些值仍必须显式送进网络。
- tile age、距离 depot、象限、土地/作物/动物/free/weed 数量、shed 已用容量。
- 相对 base price、离价格地板/拐点距离、价格一阶差分，以及由 shops+step 得到的确定需求 cadence。
- 各类能否在终局前首次产出/成熟/入仓的 mask。
- 当前 slot 之前的 cell→kind prefix 和九类计数：由同一 state 的前序 slot label/生成结果重放。
- 现有 356-D `portfolio_features()`：它是可物化缓存，不是原始状态替代品；字段名和顺序变更必须改变
  feature schema hash。

## 8. 最小二进制 shard 契约

不用发明数据库或引入 Arrow。复用工程已有的 raw `.bin` + `numpy.memmap` 模式：一个 shard 是一个目录，
里面每个数组一个连续 little-endian `.bin`，再放一个很小的 `manifest.json`。C++ 只写固定宽度标量数组；
禁止直接 dump C++ struct（padding、`bool`、`size_t` 和 enum 宽度会随编译器/架构变化），也不用压缩 NPZ
作为热路径。manifest 最后原子发布；没有 manifest 或状态不是 `accepted` 的目录不是 shard。

manifest 至少包含：

```json
{
  "format": "kaggriculture-midgame-mmap",
  "container_version": 1,
  "schema_name": "candidate-critic-v1",
  "schema_sha256": "...",
  "byte_order": "little",
  "validation_status": "accepted",
  "constants": {"terminal_action_exclusive": 719, "turns_per_day": 24},
  "arrays": {
    "frame_step": {"path": "frame_step.bin", "dtype": "<u2", "shape": [0],
                   "role": "online_input", "sha256": "..."}
  }
}
```

`shape` 中的 0 只是这里的文档占位，正式 manifest 必须写实际正整数。loader 对每个数组核对
`file_size == product(shape) * dtype.itemsize` 后才能 `np.memmap`；offset 表必须从 0 开始、单调、末值等于
对应 ragged 表行数。离散事实使用 `u8/i8/i16/i32/i64`，money 和 teacher Q 保留 `f64`，网络连续输入可在
明确记录 cast 的派生数组中用 `f32`；canonical 状态不用 `float16`。

### 8.1 公共 frame/state 表

每局的连续 observation 只存一次，day-boundary state 通过 index 引用，不为每个 candidate 复制：

- `episode_frame_offsets[E+1]`、`frame_step[F]`；
- `farm_money[F,2] / farm_unlocked_mask[F,2] / farm_hires_today[F,2]`；
- tile 的分列数组 `[F,2,100]`：kind、item、birth、yield、dry、fertilized_until、pending_bonus、
  max_lifespan_step、flags；
- `market_inventory/market_price[F,9]`、`shop_count[F] / shops[F,8]`；
- own/rival unit offsets 与有序 cell；我方单位另有 bag `[U,12]`、bag order `[U,12]` 与长度；
- 我方 unit action 与 ragged market action，供历史 encoder/行为 target 使用；
- `state_frame_index[N]`、`first_handoff[N]` 和各因果记忆/commitment group 的 `present[N]`。

训练可以选择较短历史窗，但 canonical frame 表不预先写死 GRU 窗长。若存储压力最终成为实测瓶颈，再从
这个保真表生成带独立 schema hash 的 history cache；不能先丢字段后假设 192 维 latent 能补回来。

### 8.2 历史 candidate scorer 诊断表（不进入当前训练）

- `state_candidate_offsets[N+1]`；
- `candidate_features[C,D]`，当前 `D=356`，manifest 必须同时保存完整 feature names 与其 SHA-256，
  loader 不得静默补零或截断；
- `prepared_index / candidate_id / is_outer / proposal_key_hash128`；
- `short_score`（audit/target-only）、`scenario_q[C,4]`、`paired_advantage[C,4]`、scenario valid mask；
- 四个 scenario 的名字、生成规则 hash、teacher binary/settings hash，以及每个 state 的 reference candidate；
- state 级 causal context 指向公共 state、ledger/SaleClock/CropClock、book/joint，而不是复制对手 identity。

四个现有 scenario 是未校准 support points，不是等概率样本；manifest 不得伪造 `0.25` 权重。此表只供
旧 scorer 归因和复现实验，不是逐格 actor 的输入、标签或训练前置条件。

### 8.3 主训练表：slot actor

- `state_slot_offsets[N+1] / slot_cell[K] / proposed_kind[K] / legal_mask[K]`；mask 用 9-bit `u16`；
- 从当前 state 到 day29 的完整 `flow[K,30,9] / labor[K,30] / fixed[K,30] / day_valid[K,30]`；
- 每步虚拟填充后的 feasibility/resource 字段及字段名、单位、schema hash；
- teacher 来源、proposal key 和最终 executor 可行性结果；可选 value target 不能改变逐格标签语义。

这里**不接受固定 11 维 `remaining_resources`**。在 C++ mask 生成器能导出完整 canonical feasibility
state，并通过“从导出状态重算 9-bit mask 完全一致”的自检之前，`R` 保持未定；任何代码都只能从
manifest 读取 `R` 与字段顺序，不能硬编码 11。

## 9. 数据 allowlist 与验收

binary packer 必须吃显式 source allowlist，禁止 `glob(data/bc/**)`。当前已有
`INTERRUPTED-*/DO_NOT_TRAIN.txt`，且其中数据与干净重跑 seed 重叠；目录名、标记文件或缺结果 manifest
任一命中都必须 fail closed。`train-public7-seed2620000128-n128-trajectories` 这类仍在运行或尚无完整结果
manifest 的目录同样不能进入转换。

allowlist 的每个源条目至少固定并核验：结果 manifest 路径、seed 半开区间、seat 集、对手/source-family
集合、policy SHA-256、trajectory 数、逐步 frame 数、terminal/error 数和人工/程序 validation status。
这里的 opponent/source-family 只存在于 audit catalog 与 group split；trainer 只拿 opaque group id，模型
forward 的 tensor 字典中不得出现这些列。packer 还必须拒绝重复 `(source family, seed, seat, policy hash)`，
检查每局 step `0..718` 连续、meta 与 allowlist 相符、恰有一个 terminal 且没有 error。

发布 shard 前至少完成：

1. 所有 array 的 dtype/shape/byte size/hash 与 ragged offset 不变量；
2. source allowlist 的轨迹数、frame 数、seed/seat 范围和无重叠检查；
3. 从每个 source family 抽样做 JSON→binary→C++ View 字段逐项 parity；
4. commitment snapshot restore 后的 candidate key/mask/action parity；
5. Dataset 的 model-input keys allowlist，证明 split/audit/target 字段未被传给 forward。

## 10. archive 与训练热路径

当前 Kaggle replay 单文件约 34.6 MB，407 局未压缩 JSON 约 13 GiB；Python 每个 epoch 重解析它们当然
不可接受。正确流程是只解析一次并写上述 mmap shard，之后训练只做切片。

原始 archive 可以在一轮下载**完全结束且 collector 不再依赖原路径 resume**后，逐文件转成
`json.gz`（stdlib 即可；若环境已有 zstd 也可用）并记录原始字节 SHA-256、压缩文件 SHA-256、解压后
JSON/episode 计数。collector 仍在 resume 时禁止改名、移动或压缩，避免被误判缺文件而重下。未压缩原件
只有用户明确同意后才可删除；压缩 archive 也不是训练热路径。

## 11. 是否需要全 C++ 训练

当前不需要。C++ 应负责 FastEnv、teacher、精确特征/mask、mmap writer、自博弈和最终小网络 inference；
Python/PyTorch 只负责 `np.memmap -> torch.from_numpy`、batch、autograd/optimizer 和实验迭代。PyTorch 的
矩阵乘、GRU、反向传播本来就在 C++/oneDNN/NPU kernel 中执行，“Python 训练”不等于用 Python 循环算网络。
对 8万–18万参数的首版，改成 LibTorch 通常只会增加 checkpoint、loss、调试和 NPU 适配成本，不会加速
rollout 或 JSON 转换。

只有在 mmap 已启用、batch gather 已向量化后，对代表性完整 epoch profile，并连续三轮同时满足以下条件，
才值得做一个 LibTorch A/B：

- Python Dataset/collate/调度占训练墙钟至少 30%，不是等待 rollout 或磁盘缺页；
- 计算设备利用率长期低于约 70%，扩大 batch/预取/worker 后仍无改善；
- 同 batch、同顺序、同 loss 的最小 C++ 原型能把**完整 epoch**墙钟改善至少 1.5 倍，并保持数值/收敛 parity。

即便达到门槛，也应先把 loader/collate 做成一个小 C++ 扩展，而不是重写整个训练栈。当前最有效的分工仍是
“C++ 生产数据和上线推理，PyTorch 训练”。
