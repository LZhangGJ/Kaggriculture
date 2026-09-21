# Kaggriculture Replay Route Switch（干净工程）

## 2026-09-21 研究口径：只优化真实 warm 部署链

后续性能研究只看 **G275 → 147 维浅树切换 replay 路线 → delay=1 温接管 R1**。
纯 R1 和冷接管不再作为优化方向；它们最多用于一次性兼容性检查，不能占用候选筛选和正式验证预算。
最终验收仍是未参与训练或筛选的 seeds、七个 `opponents/` 强对手、同 seed 双座，并要求每个对手
胜率都严格高于 80%。

### 2026-09-21 最新部署改进：保留浅树，G114 叶回退 G275

新 A8 的直接消融推翻了“固定 G275 更稳”：默认浅树为 `98/112`，stay G275 只有 `84/112`。
按最终路线分解，G024 的 29 局和 G316 的 12 局全部获胜，相对 stay G275 合计净救 28 胜；
G114 的 46 局则包含全部 14 败局，相对 G275 净损 14 胜。因此抓手不是关掉路线树，而是修正
G114 这个错误叶。

G114 回退 G275 在独立 B16 七强逐手净胜为 `+2/+4/+8/+6/+4/+2/0`。原生 stop288 proxy
一度把 G396 排第一，小块 D16 相对 G275 为 Thomas `+2`、Melon `0`；但预注册 H32 完整部署
A/B 明确反转，G275 相对 G396 为 Thomas `39→46/64`、Melon `41→46/64`，分差分别
`+1399/+1748`。因此正式路线配置使用显式 `target_fallbacks: {G114: G275}`；其他树叶、
147 维输入、delay=1 温接管和 R1 均不变。这是一项已重复的实质提升，但 H32 两手都只有
71.9%，总目标尚未完成。

G396/G275/G249 的 G16+H32 warm 标签在原始 G114 条件域共 16 个独立 seeds；严格
leave-one-seed-out 下，depth1–3 门控均不如静态 G275，根特征在库存/价格/对手现金间漂移。
看似有效的 BRUNCH stump 也不能外推，因此不部署进一步门控。剩余败局到 day12 已比胜局多落后约
1.1k–1.4k 现金，day15 后才被 R1 放大：主抓手仍是温前缀状态质量，而不是继续扫 DP 参数。
H32 的六路线逐局 oracle 总体上限也仅为 Thomas `53/64`、Melon `54/64`；其中表内最偏
Thomas 的 G249 在全新七强 I16 上逐手 `Δwin=-4/-2/-2/-4/-6/-4/0`、分差全部下降，已拒绝。
这封死了继续用全局路线替代追胜的方向；后续要改变接管前状态形成机制，而不是扩大路线枚举。

### 2026-09-21 首次不可逆错误：step264 的资本 proposal 错排

H32 的三块地购买发生在 step218/220/222，但 delay1 都在 step264 温接管。全局 delay0 让
Thomas/Melon `-21/-8` 胜，全局 delay2 为 `-5/+1` 且分差均降；G195-only delay2 虽偶尔救
Melon，却在 fresh 七强块中收益稀疏、分差普遍下降，因此 delay1 保持不变。

三个双手都败的 hard seeds（`2616600001/3/7`）在接管时全部选 id1；模拟认为 id1 比 id0 高
`1666–1910`，真实官方 suffix 却是 id0 在 6/6 个 Thomas/Melon cell 都更好，平均分差 +1313。
首个不可逆动作是立即追加牛/羊、MELON/TOMATO 种子、肥料和雇工，而非 sale timing。单独删牛、
换动物或限种子的符号会随对手翻转，不能做动作 guard。

随后严格否定了三种工程补丁：统一 id0 在 O8 曾七强救14伤0，fresh P16 却让 Thomas `-6`、
Ahmed `-4`；两层 candidate-diff veto 在 M16 伤 Demand/Ahmed/Pipe；由四块拟合的窄 predicted
band 在 Q16 七强 `Δwin` 全 0 且多数分差下降。结论是可复现根因已到 **共同尾值/风险评分不能
可靠区分首日资本计划**，下一步应修正统一评价目标，不再堆 route、timing 或后验阈值。

进一步的数值分解把根因缩小到作物尾值：hard 6 cells 中 id1 相对 id0 的 h1 预测优势平均
`+1774`，intervention-day 自身现金只贡献 `+5`，竞争项约 `-0.3`；共同 base tail 的未来作物
交易却贡献 `+2170`，扣除工资、动作和对手收益后仍把符号判反。真实官方 suffix 中 id1 的自身
现金平均少 `3845`，最终 margin 少 `1313`。因此 competition、discount 或 sale timing 都不是
这一批错排的主因，而是候选诱发的未来作物流及后续重规划没有被固定 public-flow 模型兑现。

三个机制候选均只在 warm 链上快速证伪并已从正式源码撤除。把共同 tail 从 `live` 改为完整
rollout controller，在 fresh A8 的候选分数、1–5 日 horizon、features 和 suffix 全部 bit-exact；
View 加 `book/joint` 的状态传递没有 bug。把公开情景滚到终局，在 fresh S8 上 Thomas 胜数不变、
Melon `-4/16`。按当前价格扣除正向未来作物流能修正 hard `6/6`，但 fresh P16 让 Thomas
`rescue2/hurt6`，并伤 Ahmed/Demand/Pipe。最后严格落实“一日干预”，次日恢复干预前 `book`，
T8 曾让 Thomas `+2/16`，但 fresh 七强 U8 使 Melon/Demand/Ahmed/Pipe 各 `-1/16`。这些结果
说明当前可解释的统一修正仍在优化平均现金代理，而不能稳定优化胜负；生产 R1 保持不变。

接缝和算力方向也已排除。G275 在接管前最后一帧卖 12 wheat；保留它的 fresh V8 结果为 Thomas
`-1/16`、Melon `+1/16`，不构成公共修复。H32 中两手同 seed/seat 的前缀几乎相同，胜局在接管前
反而少约 650 现金、少 1.4 个资产；买地时机和牛/羊/麦结构主要是路线标签，不能蒸馏成统一动作。
跨 h1–h5 一致才接受候选的 robust 规则也在 L16 和 M16 反号。Thomas-only C++ 化的理论吞吐上限
仅约 `1.12×`、Melon 约 `1.07×`，因为单局主要耗时是我方已为 C++ 的 R1 决策而非对手 Python；
Thomas 现有 native prefix 也只有 `51/64` 轨迹精确，故不投入对手重写。

现已改用更小的算力抓手：`experiments/run_strong_ab.py --engine fast` 让完整真实部署 agent 原样运行
在已有 C++ `FastEnv`，默认仍是官方引擎。全七强、同 seed 双座的 28 行 parity 中，终局 cash、
对手 cash、margin 和 error 逐项完全一致；平均单局由 `14.29s` 降到 `8.00s`（`1.79×`）。它只
用于候选筛查，最终准入仍回官方引擎。

作物尾值又定位到一个精确机制：hard id1 的 `+24 MELON` 是 day13 收获提前到 day12，整局数量
差为零，并非新增产量。策略改变空地数还会改变 weed 抽样消耗的 RNG，进而让后续商店分叉，所以
同 seed proposal 反事实也含不可观测方差。让 proposals 共用 base 的 incumbent harvest date，
X4 曾使五个弱手全部 `+2/8`；但仅首次 handoff 的严格版本在独立 Z8 为 Melon `+4/16`、
Ahmed/Pipe 各 `+2/16`，同时 Thomas `-4/16`。147 维公开状态无法稳定区分 Thomas/Melon，故不做
bot-style gate，实验宏已撤除。direct-payoff default/base 浅树在 L/O/P leave-one-block-out 中也
持续伤 Thomas，W8 盲测集保持未用于规则选择。

后续 AA16 将 harvest 锚定再次判负（Thomas `-2/32`、Melon `-3/32`）。首次接管日启用
crop succession 的 W8/AB8/AC8 固定策略总 `Δwin=+14/+9/-6`；exact stump 与 depth-2 三块
LOBO 合计都为 0，并在 hold-AC 伤 Demand/Ahmed/Pipe 各 2 胜。为判断是否只是 suffix RNG，
FastEnv 新增离线 `reseed_future()`：固定同一 warm handoff 状态后只重采未来 shop/weed；原 seed
parity 完全一致。fresh A4×4 条件续局仍为 Thomas `+0.5` 等价胜、Melon `-1.5`、Demand `-1.5`、
Ahmed `0`、Pipe `-1.0`，五弱手合计 `-3.5/40`。因此该机制的对手依赖是真实的，拒绝且不跑 A8；
工具保留用于以后对 warm 候选做条件期望归因，不能把 RNG replicas 当独立样本。
唯一跨两个 seed 复现的动作签名是 `+3 WHEAT/-3 TOMATO/少花300`：Thomas `+1.0` 等价胜，
Melon/Demand/Pipe 各 `-1.0`。同一公开动作直接反号，因此不再拆 target/flow 阈值门控。

接管前公开成交时序确有一个高精度但低覆盖的 Thomas 子域：`step257` 对手净卖 3 WHEAT。
A8 为 Thomas `6/8` seeds 命中、其余六手 `0/48` 误报；fresh A16 为 `9/16` 与 `0/96`，但只
覆盖 G195/G275。另有 3 个 fresh seed 的 Thomas 与五弱手在整个 step253..263 全商品公开流完全
相同，故不存在该窗口内的全覆盖风格分类器。更关键的是 oracle 上界已经失败：fresh A8 即使只对
Thomas 强制 handoff-day `crop_succession`，也从 `16/16` 降到 `14/16`、平均分差 `-696.75`。
因此不为这个指纹增加线上门控；生产策略与动作保持不变。

### 2026-09-21 最新抓手：接管首日交易，而不是再训路线树

正式 warm 64-seed 的阶段归因把优先级收敛为：接管首日交易/资本选择 > 中后期 DP 的价格冲击
放大 > 前期路线切换。replay 在 day10 用现金换入土地和生产资产；到 day11 我方仍多约 17 株作物、
约 40 单位日产，因此问题不是前缀没有资本形成。Thomas 是唯一在 day11→12 相对现金差继续恶化
（约 −862）的对手；胜负样本的现金差随后才在 day15 拉大。

进一步审计撤回了“seed `2610100049` 由卖货时机一招翻盘”的归因：旧 `competitive_sale`
diagnostic 从固定 base 派生，而真实 default 是另一套折现率/资本参数，反事实并不单变量。接口已改为
从当局最高分 default 派生后只改 `delay_sale`。修正后的全新 warm A8 上 Thomas/Melon 都是
`8/16→8/16`，平均分差仅 `+181/+178`；原 seed49 也只改善 `174/168`，仍然全败。因此此前
MELON/STRAWBERRY 动作差异主要来自资本估值，不是 sale timing，相关后验门控不再推进。

污染同时提示检查 handoff 的资本 proposal 选择：当前一日 MPC 在新的 32 个 handoff 状态中
32/32 次选择激进 id1（较高 discount/capital_power）。强制原始 base proposal 后，Thomas
胜数不变但救 2/伤 2、均分差 `+921`；Melon 却净 `-2/16`、均分差 `-597`。因此也不能统一降低
首日投资强度，停止扫描 discount/capital 参数；剩余问题是评分对状态/对手风格的可识别性。

两个直接的 DP 供给时序假说已经只在 warm 链上证伪。把全部预计对手供给放到我方之前，Thomas
A8 净 `+2/16`，但 Melon 净 `-2/16`；再把城镇需求放到成交之后，两手胜负均不变且平均分差分别
下降 `1753/2761`。把公开资产的待售上界加入当日对手供给也在独立 B16 反向（Thomas `-1/32`、
Melon `-4/32`）。所以“统一修改 50/50 顺序”不是抓手，实验宏未留在正式源码。下一步只研究
handoff 首次重规划的评分误差，并要求门控能在独立 seed 上由公开盘面复现；不再扩大这些成交
顺序或资本参数实验。

评分链本身也已排除：`score()` 的现金、尾值、跨日恢复口径没有双计或双重折现；把 rollout 从
1 日延长到 2 日仍从不选择 base，3–5 日则在 Thomas 上反预测。seed49 的真实首差是 step264
相同卖货后 auto 额外购买 STRAWBERRY，随后把 MELON 配置换成更多 STRAWBERRY/TOMATO；不是
雇工或 sale timing。进一步把 id1 拆成 discount-only 后，fresh D8 的 Melon 32/32 动作完全相同，
Thomas 唯一差异来自 diagnostic 绕过 joint 搜索且分差更差。一个 `id3→id0` 窄 guard 虽在两个
历史 seed 救败，fresh 七强 A16 对五个未达标对手全部零触发，只改变已 100% 的 Salemali，因此
也不保留。未来商店 Jensen 精确枚举的单地块修正只有 Tomato `+$4.15`、Strawberry `-$6.28`、
Melon `$0`；`feed_cover=2` 的七强 A8 也全部 `Δwin=0` 且多数分差下降，均不实现。剩余抓手不是
proposal 参数、horizon 或这些小尾值修正。

### 2026-09-21 最新失败归因：warm 标签有价值，但不可观测后验不能蒸馏

Thomas 的 sale-race 层是实质强点：同 8 seeds 双座把该层关闭，我方由 `12/16` 变成 `16/16`，
平均分差 `+2769`；分解后我方终局现金 `+2100`、Thomas `-668`，说明它主要通过共享价格抢走
我方收入。但 R1 本身每帧已经卖出全部非保留库存；关闭 R1 的本地等消费 hold、按 base-price
精确释放 hold、以及 1/2 日供给抢卖均为 `0~+1/32` 且分差不正，均不部署。

公开市场守恒账本已经实现并用真实对局核对：对不能购买的七种商品，逐 tick 扣除己方可执行成交和
城镇固定消费即可精确恢复对手销量；报价可能触底 1 的 tick 会拒绝而非伪造数值。但在 step216 前，
Thomas 与 Melon 的 128 个双座样本全部得到同一向量：`MILK=12, WOOL=12`，其余为 0，因此该信号
没有分类信息，不接入线上。

直接按 warm 终局收益选叶的树在首个 B16 上净 `+14/224`，公开状态门控后 E16 净 `+18/224` 且
七手均超过 80%；但全新 F32 反向净 `-10/320`（只计五个弱手），Thomas 仅 59.4%，拒绝。
补入 seat1、以及把叶目标改为逐对手 maximin，也分别在新 8-seed pilot 中伤 Thomas/Herd，均拒绝。
step192 反事实中 `G114→G190` 训练内由 14/24 到 18/24，盲测却让 Thomas `-6/32`、Melon
`-4/32`。这组结果说明主要障碍是 checkpoint 后未来商店/RNG 的不可观测性，而不是缺少树深、
样本并行度或另一个 checkpoint；该轮实验当时未改部署，现已由上节的 G114→G275 回退取代。

最后把输入严格限制为 8 个已开放商店和 2 个历史现金特征，并使用逐对手 maximin 叶目标；I8
对五弱手净 `+10/80`，但保守门控后的独立 J16 仅净 `+3/160`，Melon `-1`、Thomas `0`，
Thomas 绝对胜率仍只有 62.5%。因此连低方差可观测分层也不足以把路线 oracle 转化为目标收益。
后续抓手应是 DP 中对手未来供给的**成交时序**校准，而不是更多 replay 路线树。

当前最明确的失败抓手是路线树标签错配：树按“切换后 replay 跑到 day29”训练，线上却只执行到
约 day12 就温接管。正式 64-seed 追踪中，五个未达 80% 的对手各有 62/128 局被切到 `G114`，
这些局的胜率仅 51.6%–61.3%；15 个 hard seeds 的 150 局中有 120 局落到 `G114`。下一步使用
真实 replay-prefix → warm-R1-suffix 终局结果重做标签，不再继续扫描纯 R1 参数。

真实 warm suffix 重标已完成一轮严格验证：128 个独立训练 seed 上的 9 路线 oracle 对五个弱手
均为 95% 以上，但把逐局 oracle 蒸馏成 depth=2 浅树后，独立 B32 曾净增 53/448，随后 C64
仅净增 12/896，并使 Thomas/Melon/Ahmed 分别 -1/-1/-3；因此**不部署**。这证明标签域修正有
总体价值，但“逐局 argmax 路线分类”仍把不可观测未来随机性当成确定标签。下一步应让叶节点直接
最大化配对期望胜负/分差，而不是继续增加 seed、树深或枚举路线。

## 2026-09-21 当前结论：默认 80.02%，组合优化仍不部署

在未参与训练或筛选的 seeds `2610100000..2610100063` 上，对 `opponents/` 七个强对手、同 seed
双座（每臂 896 局）：纯 R1 `528/896`（58.9%），冷接管 `705/896`（78.7%），**当前默认
G275 + 路线浅树 + delay1 温接管 `717/896`（80.02%）**。这是当前部署的正式成绩。

实验性的 R1 handoff candidate-diff 浅树为 `718/896`（80.13%），相对当前默认仅净增 1 胜
（救败 28、致败 27，McNemar `p=1.0`），并使 Thomas `-5`、Herd `-6`；因此**不部署**，
资产只保留作离线机制研究。线上默认仍不加载 handoff selector。

2026-09-21 对 R1 的作物/动物组合选择增加了严格 opt-in 的局部优化实验：
`portfolio_swaps=1` 做单地块替换，`portfolio_swaps=2` 做受限双地块联动替换。默认均为 0。
单换两批独立 16-seed 合并净胜 0；双换 8-seed pilot 为 Thomas `-1`、其余 0，七强平均
分差全部或近乎全部下降。因此它们只保留作机制实验，**不进入部署，也不扩大正式验证**。
现有价值函数本来就包含已知商店需求、未来商店期望、双方公开资产产量、库存价格曲线、劳动/工资、
预算和竞争收益，并把对手供给各一半放在我方成交前后；失败说明当前主要问题不是少枚举一两个组合，
而是小的模型估值差会被后续滚动重规划放大。

这是 2026-09-20 从多个历史工作树收拢出的独立研究工程。主线只有一条：从近期高手 replay 提取经营路线并聚类，使用高速原生仿真做路线互打和反事实切换搜索，以浅层决策树选择前期 replay 路线；中期把真实公开状态交给**没有开局模板的 JointAFS R1** 动态策略。

当前代码可以完整运行，默认配置为 **固定 G275 opening + 浅树切换 + 三块地状态触发接管 R1**。

当前正式 64-seed 四臂结果以本文开头的 `528/705/717/718` 为准；更早的 `531/683/706`
分别对应旧接管语义，只用于解释演进，不再代表部署成绩。接管仍是强变量：过早切换会丢失 replay
已安排的扩地与资本形成，因此现在由“达到三块地后再等 1 天”触发，step288 是最晚截止。

## 目录

| 路径 | 内容 |
|---|---|
| `agent/` | 前期成熟 replay 路线执行器、147 维浅树和 replay→R1 组合入口 |
| `policy/r1/` | 真正无开局模板的 JointAFS R1 源码、二进制与最薄外部观察桥 |
| `meta_agent/` | 成熟路线执行、特征、切换控制器代码 |
| `scripts/` | replay 分析、聚类、原生互打、切换搜索、鲁棒浅树训练工具 |
| `fast_kaggriculture/` | 高速 C++ 仿真器源码、Python 扩展与已编译二进制 |
| `native_deps/` | 重建高速仿真器所需的最小 C++ 依赖源码 |
| `opponents/` | 7 个近期强公开对手；正式验证不混入弱对手 |
| `data/replays/raw/` | 本轮选定的 609 份完整原始 replay |
| `data/replays/daily/` | 成功通过旧日级 receipt 提取器的 177 份派生记录 |
| `data/artifacts/` | 609 replay 的特征、聚类、245 路线、互打矩阵、切换矩阵和树 |
| `experiments/` | 官方环境并行 A/B、扫描程序及历史结果 |
| `docs/` | 架构、数据、运行方法和结果解释 |

## 快速验证

```bash
cd /root/kaggriculture-replay-switch-clean-v1
PYTHONPATH=. /root/miniforge3/envs/torch-npu/bin/python verify_project.py
PYTHONPATH=. /root/miniforge3/envs/torch-npu/bin/python experiments/smoke_official.py
PYTHONPATH=. /root/miniforge3/envs/torch-npu/bin/python experiments/test_submission_contract.py
```

结构验证检查 609 replay、245 路线、7 个强对手、24,460,800 局切换数据、资产哈希、R1 接管符号和原生仿真器加载。官方烟测完整运行 719 步并跨过默认 day12 接管点。提交契约测试按两种 observation 视图驱动官方引擎，两个 seat 都必须跑满并完成接管（见交接中的视图保真度记录）。

## 当前部署语义

- 开局：固定 `G275`，不使用 Nash 混合。
- 前期：成熟 `TeammateExpandedRouteAgent` 执行 replay 路线；它含杂草修补、市场/现金保护和喂养保护，不是盲目动作磁带。
- replay 内切换：使用 `route_policy.json` 的浅树；现有树只覆盖五个 opening，检查点为 step 144/168/216（day6/day7/day9），当前控制器最多切换一次。
  - **G275 的 cp144 节点已换成用真实对手重训的 depth-3 树**（896 个「状态 / 5 候选目标 / 结局」单元；原树是在自对弈矩阵上训练的）。配对 A/B：**+14 / +4 胜（两段），14/14 对手-段均分差为正**。
  - `target_fallbacks` 除原有的 `G114 → G275` 外，新增 **`G019 → G195`**（G019 是按叶归因找出的坏叶，条件胜率 69.7%）。配对 A/B：**+10 / +14 胜，14/14 对手-段均分差为正**。
- 动态接管：达到 3 块地后延迟 **2** 天转交无开局模板 R1（即固定到 step 288；实测 288 为单峰最优点，264 差 52 胜、312 差 29 胜）。配对 A/B：**+52 胜，z=2.20，3 段 6 手全正**。
- 可用 `REPLAY_FORCED_OPENING` 和 `REPLAY_HANDOFF_STEP` 做离线实验覆盖；正式结论必须同 seed 双座、多 seed。

### 验收口径（重要）

`run_strong_ab.py` 的 `both_seats` 几乎不提供独立信息：本对局对称且双方确定，**约 87% 的 (对手, seed) 配对中 seat0 与 seat1 的 margin 逐元相同**。因此所有历史「N/896」的有效样本约为名义值的 57%。
验收请用 `experiments/eval_seed_paired.py`（按 (对手, seed) 配对 + Wilson 区间），并按 seed 段互斥取数。

### 当前水平（512 个全新 seed，seed 配对，Wilson 95%）

| 对手 | 胜率 | 95% CI |
|---|---|---|
| thomas_2945 | 76.2% | [72.3, 79.7] |
| melon_2749 | 79.9% | [76.2, 83.1] |
| demand_preserving | 80.3% | [76.6, 83.5] |

其余四手（salemali / herd / pipe8 / ahmed）已在 89.6%–100% 区间。**「每个对手 ≥80%」尚未达成。**


详见 [架构](docs/ARCHITECTURE_ZH.md)、[数据](docs/DATA_AND_PROVENANCE_ZH.md)、[复现命令](docs/REPRODUCTION_ZH.md)、[文件索引](docs/FILE_INDEX_ZH.md) 、[交接](HANDOFF_ZH.md) 与 [本轮会话记录](docs/SESSION_20260920_ZH.md)。
