# 工作入口

先读 `README.md`、`HANDOFF_ZH.md`，以及本轮的分析与诊断记录 `docs/ECONOMIC_MODEL_ZH.md`。
本工程唯一主线是：高手 replay 离线提取/聚类 → 路线互打 → 147 维状态上的浅树切换 replay 路线 →
中期接管无开局模板的 JointAFS R1。

## 队友主机交接（`handoff/student-v45-20260923`）

- 仓库：`https://github.com/LZhangGJ/Kaggriculture`；请从本交接分支检出，不要把它当成 `main` 的状态。
- 已训练到 v45 的逐格 actor 快照放在 `models/student-v45/`：`actor.pt`（模型和 AdamW 状态，7.3 MB）、`actor.bin`（同一轮的 C++ 前向权重，2.5 MB）及 `manifest.json`（checkpoint 的固定数据契约，约 10 MB）。v45 是训练链快照，不代表盲测最优，也尚未替换线上 R1。
- 不上传原始/日级 replay、历史逐局 rollout、BC mmap shards、实验缓存或构建目录。续训从 `actor.pt` 的模型与优化器状态出发，每轮只采新的 on-policy 对局；无需重做此前轮次或读取旧 BC shards。仍需同步尚未提交到本分支的实验性 RL 源码并在目标主机重建原生环境；仅克隆此分支暂不能直接续训。
- 本机现成的 Python/C++ `.so` 是 aarch64 构建物。队友若用 x86_64，必须在自己的 Python 环境重建原生扩展和 R1，不能直接复用这些 `.so`；`actor.pt` 与 `actor.bin` 本身是跨主机的数据文件。
- 当前正在后台训练，v45 是刻意冻结的交接点。后续轮次不会自动覆盖仓库里的快照；需要更新时再单独选定完整轮次。

## 硬约束

- 不得把 `experiments/results` 中带原生 opening-template 的旧 Triad 扫描当作 R1 结论。
- 不使用 Nash 选开局。
- 正式公开对手验证只用 `opponents/` 的 7 个强对手，同 seed 双座、多 seed。
- replay 只离线使用。线上 `agent/main.py` 只读取导出的路线、浅树和当前公开观察。
- 先复用 `scripts/` 的成熟聚类、原生对打和鲁棒树训练，不另写简化替代品。

## 模型研究方法论（强约束）

本节约束后续所有数学模型、规划器和执行器研究，优先级高于历史参数 A/B 经验。

1. **理论先于现象。** 先从官方规则写出目标、状态转移、现金/库存/市场守恒和信息边界，再检查实现；
   轨迹只负责证伪、确认触发域和估计量级。不能从败局相关性反推原因，也不能要求一个理论错误必须先与
   败局相关才允许修。
2. **分清四个层次。** 每一项都明确标记为：精确规则/不变量、主动模型假设、计算近似、经验补偿参数。
   `competition=2`、`supply=.85`、`replant=.5` 等有效参数可能在补偿别处的错误，不能反过来当作理论真值。
3. **先做影响量级筛选。** 深挖前先估算
   `误差上界 × 实际触发覆盖率 × 决策敏感度`。数值误差只有在候选间的差分足以跨过 score gap、改变
   `argmax` 或约束可行性时才可能改变计划；同时报告绝对误差和候选差分，不能只报前者。
4. **内部不变量优先验收。** 核心模型修复先用最小子系统诊断：逐项跟踪预测流、现金、库存、约束、
   tail value、候选 score 和排序。能用保存状态或只跑到 checkpoint 验证时，不跑完整对局；能在固定
   portfolio 上隔离估值差时，不把重新规划、执行和未来随机性混进同一个结果。
5. **正确性与当前胜率分开。** 修正一个模型错误后端到端性能下降，只能说明旧参数/旧误差之间存在补偿，
   不能证明修正错误，也不能证明旧近似更正确。应沿依赖顺序重新校准下游假设和参数；不得为了保留当前
   胜率而恢复已确认违反规则或守恒的不变量。
6. **端到端验证是最后一道部署门槛，不是理论裁判。** 顺序固定为：理论推导 → 内部不变量 → 量级与
   排序影响 → 下游重校准 → 固定状态 suffix → 互斥 seed 的七强完整 warm 链。最终是否部署仍以正式
   胜率口径决定，但端到端负结果不能抹掉已经证明的模型缺陷。
7. **保持可归因。** 理论修复先以隔离构建或诊断开关验证，不同时改无关模块；记录修复前后的内部量、
   哪些候选换序以及换序贡献。只有内部机制成立后才扩大实验。

下文“已实测封死”限制的是在旧模型上继续盲扫同类参数或直接部署旧候选，不禁止修复其上游理论错误后，
按依赖顺序重新校准受影响的参数。旧模型下的负 A/B 不能永久否决一个后来已由规则证明正确的不变量。

当前 R1 首要结构问题已进一步定位为 **`score()` 的静态 portfolio tail 不是线上每日重规划策略的
continuation value**。在 8 个固定 handoff 的 6 个非零 outer 反事实上，现有一日静态 score 对真实
终局方向只对 `3/6`；用同一模型真正逐日闭环规划到终局后方向对 `5/6`，聚合预测由错误的 `+919`
改为 `−2,053`，接近真实 `−2,800`。1/3/5/10 日截断均不稳定，只有终局 horizon 在这批状态上恢复
主要方向。因此优先修的是 Bellman/控制时域一致性，不再扩首日 beam 或把长期 `model.value` 搜得更精确。

对手跨日库存/成交响应是其后的重要状态缺失，而不是已经单独解释排序的结论。公开账本在 Thomas seed100
的 day24 给出 WOOL 隐库存区间 `[0,33]`，随后候选使市场库存/价格分叉到 `−51/+195`，对手多赚
`$5,659`；但以同一 forecast flow 做“立即卖 vs 跨日最优清仓”的库存 best-response 诊断，对相关候选
几乎共同扣约 `$6,994`，winner 不变。这证明库存时序对绝对价值很大，却也证明仅加一个库存点估计不够；
剩余误差需要公开历史上的多情景 `rival stock belief + conditional liquidation`，并允许未来重规划与
价格反馈联动。

一次 top-4 终局闭环重排虽令内部 winner 在 `7/8` 状态变化，固定 suffix 仅从原始 outer 的 `−2,800`
改善到 `−1,939`，但该 A/B **不能准入**：baseline 与 outer 各自按短分截 top-4，候选集不嵌套，outer
会漏掉 baseline 经闭环重排后的 normal winner；两例甚至选择了闭环分更低的 outer。下一次必须让 outer
候选集显式包含 baseline winner，再做固定状态比较。所有新逻辑仍是默认关闭的诊断，正式策略未部署。

该协议现已修正为：baseline 评短分前 4 个 normal，outer 评同一组 normal 加短分前 4 个 outer，8/8
满足扩展集合的预测最优值不下降。修正后的固定 suffix 中，闭环 normal 重排相对旧 normal 合计改善
`+9,774` margin；加入 outer 后相对旧 normal 仍改善 `+9,344`，说明闭环 continuation 是有效抓手。
但 extra-outer 相对同一闭环 normal 仍为 `−430`，其预测却为 `+7,970`，4 个非零选择只对 `2/4` 方向；
Thomas 两例分别误判 `+364→−5,697`、`+4,016→−1,586`。这暴露了第二层瓶颈：终局闭环消除静态
tail 错误后，单条 open-loop 对手供给仍会被候选搜索利用。不能直接部署 top-K；下一步在闭环 scorer 内
加入公开信息可生成的对手响应多情景，并用嵌套候选集继续验收。

## 当前部署语义（2026-09-21）

- 开局固定 `G275`；浅树在 step 144 / 168 切换，最多一次。
- **接管延迟 2 天**，即固定到 step 288（实测 288 为单峰最优：264 差 52 胜、312 差 29 胜）。
- `target_fallbacks` = `{G114: G275, G019: G195}`。G019 是按叶归因找出的坏叶（条件胜率 69.7%）。
- G275 的 **cp144 节点已换成用真实对手重训的 depth-3 树**（896 个「状态 / 5 候选目标 / 结局」单元）；
  原树是在自对弈矩阵（对手 = 245 条库内路线）上训练的，标签域与线上不符。

改动明细与配对 A/B 证据见 `README.md`「当前部署语义」。

## 验收口径（容易踩的坑）

`experiments/run_strong_ab.py` 的 `both_seats` 几乎不提供独立信息：本对局对称且双方确定，
**约 87% 的 (对手, seed) 配对中 seat0 与 seat1 的 margin 逐元相同**。所有「N/896」的有效样本
约为名义值的 57%。历史上一批 8–16 seed 块的 accept/reject 决策都建立在这之上。

- 用 `experiments/eval_seed_paired.py`：按 (对手, seed) 配对 + Wilson 区间。
- **seed 段互斥**，不要重复使用同一段。
- 判定「每个对手 ≥80%」需要约 683–1537 seed（现在只有 512）。`engine fast` 跑 64 seed × 7 手 × 双座约 5 分钟。

所有动态策略对 `opponents/` 公开脚本的新评测都必须保存可用于后续 BC 的逐步轨迹。正式入口
`experiments/run_strong_ab.py` 默认写 `<结果文件 stem>-trajectories/` 下的 `jsonl.gz`：每帧一份公共
observation、双方 private、双方 action，并记录策略标签、对手、seed、座位、引擎和终局 reward。
不要只保留比分，也不要把 forced-candidate 诊断轨迹无标记地并入专家数据；训练/验证必须按 seed 与对手
分组切分，避免同 seed 双座和 baseline/candidate 互相泄漏。

中盘学习路线见 `docs/LEARNED_MIDGAME_PLAN_ZH.md`。硬边界：RL/NN 只作用于 step288 后；慢 R1/终局
闭环只作离线 teacher，线上 student 不再每局跑完整 DP。大规模自博弈、反事实 suffix、特征提取和小模型
推理尽量放 C++，Python/NPU 只负责训练与调度。**学习主线只训练自回归逐格 actor**：按稳定格序直接输出
`SKIP / 5 crop / 3 animal`，每步应用精确规则 mask，并把已选前缀的预算、库存和 30 天资源日历传给下一格。
不得退回“R1 先生成整盘 candidate、NN 只做 candidate-ranking”的替代任务。critic 只能作为逐格 actor 的
value/baseline 或完整序列辅助头，不能代替逐格决策；最终动作仍须经过 executor 可行性检查。

现有非 NN 动态 R1 是冻结 baseline：学习代码必须走独立、默认关闭的实验入口，禁止覆盖
`policy/r1/agent.so`、`policy/r1/config.json` 和 `agent/main.py`。learned off 时须做逐步 action parity；
student 发生 OOD、NaN 或 feasibility 失败时必须回退原 R1。模型首版预算 `8万–18万` 参数，CPU 优先；
NPU 只在千万级样本/大规模 ensemble 时使用，不能用来替代 C++ teacher rollout 加速。

学习输入遵守 `docs/MIDGAME_STUDENT_FEATURE_AUDIT_ZH.md`：`192/32` 是 encoder latent 宽度，绝非
已完成的 raw 字段；必须明确输入剩余时域、完整公开棋盘/市场/我方 private、公开历史 belief、内部承诺
和虚拟前缀后的全剩余期资源日历。11 项资源占位和无 mask 的数据不得开训。原始 JSON 仅作档案，训练
只读一次转换并校验过的 mmap binary shards；网络下载仍单线程，下载后的独立文件解析才允许并行。

历史 candidate-critic 的四个公开供给情景是**无概率的 support 向量**，只保留作旧 scorer 诊断，不进入
当前 actor 训练或线上 forward；不得把它们等权平均成“期望”，也不得借这些资产恢复 candidate-ranking
旁路。逐格策略改进若使用真实 suffix，所有 future replicas 仍须按原 checkpoint 成组聚合。

轨迹恢复的有效域也要 fail closed：`observe_external` 只足以 warm 首次 step288 的公开 ledger/SaleClock/
CropClock，此时 R1 commitment 为空是合法状态。任何 step>288 teacher 样本必须从 step288 起实际驱动
同一 R1，逐步核对 action，并在采样点核对 candidate key、book/joint/restore_day；不能从公开盘面猜回
内部承诺。BC corpus 的策略指纹必须覆盖 Python 入口、R1 `.so`/配置、deployment、route policy/tapes
及实际加载的执行模块；单独 `sha256(agent/main.py)` 不是可复现的 policy identity。

step288 后 RL 的 terminal reward 以胜负为主，钱差只能作小且有界的 shaping；现金/资产等中间量不得
直接累加，除非写成终局为零、可望远镜的 potential difference。BC/critic 轨迹可多 epoch 复用；PPO
轨迹必须保存原 policy version、逐 slot mask/log-prob，且只作少量 on-policy epoch，旧轨迹没有明确
importance correction 时不得用于 policy-gradient。所有候选、future replicas、双座按原 checkpoint
聚合并以 `(seed, opponent family)` 整组切分；seed/RNG/对手身份永不进入 forward，保留永久盲测 seed 段。
当前 reward `sign(margin)+0.1*tanh(margin/10000)` 在 v16–v21 的分解中，对所有非零胜负
advantage 都没有翻转符号；tanh 占平均绝对 advantage 约 `4.9%–5.6%`，占平方信号仅
`0.03%–0.05%`。它在 `55%–64%` 胜负完全相同的 seed block 中提供微小 tie-break 信号，
不是当前方差或泛化瓶颈，不据盲段一次“分差升、胜数平”盲改 reward。

逐格 actor 在一天内自回归生成的**整条计划是 PPO action unit**：当天 joint log-ratio 必须等于各个
actionable slot log-ratio 之和，并只对这个 joint ratio 做一次 clip；forced slot 只推进 hidden，不进入
likelihood。终局 advantage 对每个 actionable day bundle 应用一次，**不得再除以当天 slot 数**，否则会
把期望终局收益改成依赖动作长度的另一目标。正式 batch 对同一环境 seed 跑全部「对手 × 双座」，当前局的
baseline 使用同 seed 的其他独立 policy-RNG 轨迹做 leave-one-out；它不读取当前局 reward/action，故只作
无偏 control variate，seed/对手身份仍不得进入网络输入。v11/v12 的两段 1,536 局审计中，该 baseline 将
advantage 方差分别降至旧 `(opponent family, seat)` LOO 的 `65.1% / 54.3%`；更复杂的 two-way 修正没有
额外收益。训练游戏数必须整除 `2 × 对手数`，且每个 seed 恰好覆盖完整的「对手 × 双座」块。每轮必须
连续恢复 AdamW state，并在完整 rollout 上重放行为策略、报告 day-chain KL 后才保存更新。

原生 rollout 只有在三层证据同时成立后才可替换训练入口：任意新 seed 不依赖预生成 cache；固定 cache
逐步比对前288 replay/浅树动作、step288 packed/context、全部 actor event、后缀动作与终局 exact；输出
训练所需的 context/observation/tokens/resources/legal/action/old-logprob 连续数组并通过 CPU/NPU old-policy
replay gate。重复少数 cache 得到的吞吐只算性能 microbenchmark，不算 RL 可用性。

原生 rollout 持久化直接保存 `ppo_arrays()` 的未压缩 NPZ 与最小 JSON metadata，不得先膨胀成几十万
Python event dict 再 `torch.save`。resume 时重建视图并重新执行 policy/binary/manifest、对手 artifact、
native actor weights、JobBatch module、逐局 identity 与 `route=-1` 门；NPZ 不是绕过 old-policy replay gate
的理由。B8 fresh/resume 已做到逐 tensor update exact，保存 12.99MB 仅 `0.0365s`。

RL 对手池按源码家族去重：经 Python↔C++ 逐步 action parity 的最新强公开代表作为主力，
student 历史冻结快照用于 self-play。慢动态 R1 **只做离线 BC teacher**，不得进入 RL rollout、对手池、
reward 计算或 policy-gradient；若事后给 student 状态补 R1 标签，也只能作为下一轮 BC 数据。不得因多个脚本共享
Ahmed/MetaV4 chassis 就将它们当成独立分布重复加权，也不得等所有对手 C++ 化后才开始首训。

截至 2026-09-23，原生 JobBatch train-head 为 v22，dev-best 仍为 v20。v22 首次按源码 family 用
Thomas/Fieldcraft 各 1/2，并启用 whole-seed 6-fold day-state cross-fit control variate；1,536 局
rollout `34.08s`（`45.07 games/s`），非法与 fallback 均为 0。critic 将 advantage 方差降至
`79.66%`、slot-count 加权后 `77.24%`，六折与 17 个决策日全部改善；端到端 rollout、NPU 更新和审计
共 `143.61s`。old-logprob max/mean/KL 为 `1.416e-4 / 2.206e-7 / 8.11e-13`。
学习率在 v19 起由 `2e-5` 降为 `1.8e-5`；v22 day-chain KL 为 `0.01091`，48 个 minibatch
裁剪前 gradient norm 中位数 `68.50`、p95 `100.55`，100% 超过 `max_grad_norm=1`。`target_kl` 是 post-epoch soft
early-stop，在 `epochs=1` 时不回滚 checkpoint。

v15 的 Thomas/Meta/G397 各1/3 完整拆解显示：G397 占 `39.8%` bundle KL 但只贡献 `18.8%`
clipped-surrogate gain，与两强手的 same-seed reward 相关系数仅 `0.016/-0.083`；删掉它后
advantage 方差从 `0.918` 降至 `0.680`。因此 G397 留作 retention eval，不再等权1/3。
Salemali JobBatch code4 已在新 seed 双座与权威 Python 源做到 `11,504/11,504` 动作和 `16/16`
终局 exact；但历史 student 对它 `509/512` 胜，也只作 canary。独立 family Fieldcraft 的 JobBatch
code5 已对 standalone C++ 做到 `5,752/5,752` 动作、`8/8` 终局与 identity exact，正式进入 v22 对手池。
反复查看过的固定 1,024 局集合只能称 dev：v9/v10/v11/v12/v13/v14/v15/v16/v17/v18/
v19/v20 胜局依次为 `542/554/548/559/566/554/577/560/576/598/593/613`。v20 相对
v15 为 `+36` 胜、margin `+490.9`、McNemar `p=.00275`、seed-cluster 胜率 CI
`[+1.37pp,+5.66pp]`。但预留 milestone blind `2631600000..2631600511` 的一次性配对复核仅为
v20/v15 `553/552`（净 `+1`）、margin `+197.7`，Thomas `-3`胜、Meta `+4`胜；固定 dev 的
胜率提升没有跨 seed 复现。因此 v20 保留为 dev-best，v21 按“非灾难性震荡不回滚训练链”作
train-head，不把 dev 改善当作泛化证据。
旧 blind 段已退役；`2633000000..2633000511` 为新的未查看 milestone blind 段，禁止训练。

## 性能研究的边界（已实测封死，别再走）

只评估真实 warm 链（replay 浅树切换 → delay=2 温接管 R1）。以下方向已逐一实测，收益为 0 或负：

| 方向 | 结果 |
|---|---|
| R1 配置杆（21+ 项，含叠加） | 均分差最高 +830，胜率 0（翻转一局需 +5,017） |
| 换 opening（7 条） | −66 ~ −367 |
| 编辑 tape（作物 / 预算 / 无耦合） | −454 ~ −566 |
| 叶回退（G024 / G275 / G316 方向） | −10 ~ −289 |
| 二次切换、更早切换点 cp72 | 胜率 0 / 明显更差 |
| 候选路线前瞻特征（48 步、120 步逐日） | +1.2pp / −3.9pp（后者更差） |
| 树重训扩样本（896 → 2,240 单元）、换模型类别 | 饱和 / 全部差于 depth-2 |
| 移植对手的 sale-lead 抢卖层 | +2 胜（≈0） |

**oracle（逐状态最优目标）比可达高 6–8pp，但用现有 147 维特征 + 任何模型类别都取不到 —— 瓶颈是特征信息量，不是数据量或模型容量。**

## Kaggle 提交

```bash
PYTHONPATH=. python scripts/pack_kaggle_submission.py \
  --so <x86_64 且 Jammy 兼容的 policy/r1/agent.so> \
  --output build/kaggriculture_submission.py
```

两个必须注意的点（否则线上直接得零分）：

1. **架构**：本地 `policy/r1/agent.so` 是 aarch64，Kaggle 是 x86_64。
2. **glibc**：用 Noble 的 `x86_64-linux-gnu-g++` 编出的产物要求 GLIBC 2.36/2.38，Jammy 只有 2.35。
   必须用 Jammy sysroot 工具链（`kaggriculture-t2-ideas-v1/submissions/toolchain-jammy`，GCC 11.2），
   加 `-static-libstdc++ -static-libgcc -march=x86-64 -ffp-contract=off -Wl,-Bsymbolic`，
   产物最高需求 GLIBC 2.34、无 libstdc++ 依赖。

改完编译器/宏后**必须做跨架构 parity**：同一批真实对局观测分别喂 aarch64（本地）与 x86_64（qemu + Jammy sysroot），
逐步比对 `td_observe` 输出。本会话已验证 300 步 0 mismatch。
