# 中盘学习策略：理论目标、训练路线与 C++ 加速边界

> 2026-09-22。只作用于 step288 后的 R1；前期 replay、浅树切换和接管时机完全不动。

## 结论

最终线上策略不应继续每局运行完整终局动态规划。慢规划器应变成离线 teacher；线上 student 只做小网络
前向、合法候选生成/修复和确定性执行。目标不是端到端模仿 719 步动作，而是：

`公开历史 belief encoder → 宏观 bundle policy → 规则 feasibility/repair → 候选 critic → executor`。

当前 Thomas smoke 的 FastEnv 单局决策 CPU 约 `5.85s`、总墙钟约 `9.6s`。student 的工程目标是把
step288 后的决策降到 `<1s/game`，并以 p95 `<2s/game` 作首个硬门槛。训练 teacher 可以慢，但必须离线、
批量和多线程。

## 一、严格问题

按日边界建模为 belief-MDP（更准确是受限部分可观测随机博弈）。状态为

`S_d=(X_d,Y_d,Q_d,B_d)`：

- `X_d`：时间、双方公开农场/现金、市场、商店；
- `Y_d`：我方 shed、种子、携带库存；
- `Q_d`：executor 的 queue/book/joint commitments；
- `B_d`：由公开历史得到的对手库存、产出和条件销售 belief。

动作 `g_d` 是一整天的宏观 bundle，不是一个格子；真实 24 步动作由 executor 展开。终局目标为

`V_T=P(C_own,T>C_rival,T)`，辅助头可以预测 margin 分位数。中间若使用 `Δmargin`，必须可望远镜求和；
不再让 `competition=2`、现金相关 discount 或静态 portfolio tail 定义新版最终效用。

日 Bellman 递推为：

`Q_d(g)=E[V_{d+1}(T_24(S_d,g,opponent,ξ), update(B_d,H_{d+1}))]`，

`V_d=max_{g∈C(S_d)}Q_d(g)`。

其中 continuation 必须是次日重新观察、更新 belief、重新决策的闭环策略。现有终局闭环诊断相对静态 tail
方向由 `3/6` 提高到 `5/6`，闭环 normal 重排在 8 个固定 suffix 上相对旧 normal 合计 `+9,774`
margin，证明这一层值得蒸馏；extra-outer 预测 `+7,970`、实际 `−430` 又证明单条对手点预测不足。

## 二、网络是否逐格决策

网络必须逐格决定种/养什么。对按稳定规则排列的空地 `j`，策略输出：

`πθ(k_j | S_d, B_d, 已选组合 P_{j-1}, 剩余资源 R_{j-1}, 当前格 x_j)`。

每步先由 C++ 规则层屏蔽非法项目，再保留 top-2/3 分支形成 width 2–4 的 bundle beam。完整 bundle 由
critic 统一排序。这样 NN 替换的是旧 `gain/cost^α` 的分支顺序，不替换现金、劳动、市场和执行规则。

首版优先采用 32–64 hidden 的 GRU/TCN 编码最近数日公开增量，并用小 MLP 评价 bundle。浅树只能作为
显式 belief 特征上的基线或最终蒸馏门控；它无法自然携带已选组合和历史。若手工历史统计已经足够，先用
GBDT/MLP，避免为时序网络而时序网络。

这里的 `192` 只能是结构化 encoder 输出的 latent 宽度，不能解释为 192 个已经定义好的 raw 字段。
输入至少覆盖：明确的剩余 tick/天数和终局兑现窗、双方完整公开棋盘 token、我方 private/单位库存、
市场与商店、公开成交/作物移除历史 belief、我方 executor commitments，以及每个虚拟前缀更新后的全
剩余期 flow/labor/fixed 日历。当前 scaffold 尚未实现这些 encoder，不能正式训练；逐字段审计见
`docs/MIDGAME_STUDENT_FEATURE_AUDIT_ZH.md`。

## 三、训练数据与降噪

绝对终局 margin 不是合格的单次标签。对同一 checkpoint、基准动作 `g0`、候选 `g` 和未来情景 `ω`，
训练 paired advantage：

`D(S,g,ω)=R(S,g,ω)-R(S,g0,ω)`。

模型预测 `P(D>0)`、`E[D]` 和若干分位数；主损失使用同一状态内 pairwise/listwise 排序，而不是绝对 cash
MSE。优先学习

`真实闭环 suffix − belief-aware 规则 teacher`

的残差，以消掉经济规则已经解释的公共变化。future replicas 必须按原 checkpoint 聚合，不能冒充独立样本。

用 sequential racing 控制标签成本：所有候选先跑 2 个 future replicas，淘汰明显落后者，只给置信区间
重叠的 top-2/3 追加到 4、8、16 个。common random numbers 用于降方差，但若动作改变 RNG 消耗，仍要用
多 future seed 估计真实期望，不能把同 seed 当成完全相同外生路径。

## 四、BC、自博弈和动态 teacher 的分工

1. **公开强脚本 BC**：学习 step288 后的宏观意图、对手持有/出售 hazard 和价格响应。对手 private 只可
   作为训练期辅助标签，线上输入只能是公开历史；不输入对手身份。
2. **自博弈**：用 student 历史 checkpoint population 扩大状态覆盖；随机化销售模式、库存粒子和市场
   情景，防止两个同源策略共同收敛到盲点。
3. **动态 R1 teacher**：只离线生成逐格 BC 标签。它不进入 RL rollout、对手池、reward 或策略梯度；若给
   student rollout 中的不确定/OOD 状态补标签，也只进入下一轮 BC 数据。
4. **固定 suffix teacher**：给少量候选生成真正的反事实 paired advantage；普通 BC 轨迹没有这一信息。

我方当前动态轨迹只能作为状态覆盖/行为 prior，不能自动当专家。forced-candidate 数据必须保留诊断标签。
训练/验证按 `(seed, opponent family)` 分组；同 seed 双座、baseline/candidate、所有 replicas 必须在同一 split。

首批规模：`128 seeds × 7 × 单座 = 896` 局用于可行性；有 held-out 信号再扩到 `256×7=1,792`，最终
稳健版 `512×7=3,584` 个独立对局。现有压缩格式约 `312KB/game`，存储不是瓶颈。

### 4.1 RL 奖励不能重新混淆胜负与钱差

终局主目标保持胜负，钱差只提供小而有界的连续 shaping。首版可预注册为

`R_T = (2 * 1[margin>0] - 1) + alpha * tanh(margin / scale)`，

其中 `alpha` 先取 `0.05–0.15`；`scale` 只由训练 split 的败局绝对 margin 稳健尺度确定，不能看验证集后
调参。模型仍分别输出校准胜率与 margin/分位数头；上式只是 policy-gradient 的有限方差效用，不把两种
语义永久压成一个 value target。平局单列为 0。

现金、库存、动物数、当日收入等都不能直接累加成 dense reward：投资会暂时减少现金，市场冲击又会把
今天的收入转移到未来，朴素 shaping 会改变最优策略。确需密集信号时只接受可望远镜的
`F_t = Phi(H_{t+1}) - Phi(H_t)`，并保证终局 potential 为 0；同时保留无 shaping 的 terminal-return
评估，防止近似 belief 下的 potential 偏差被隐藏。

### 4.2 batch 与轨迹复用

- 一个训练样本的统计单位是完整 checkpoint；其全部候选、slot prefix、future replicas 和双座必须在
  同一 batch group 与同一 split。candidate loss 先在状态内平均，再在状态间平均，不能让候选多的状态
  自动获得更大权重。
- BC、belief、mask 与 critic 辅助任务可以多 epoch 复用旧轨迹。旧动态 R1 轨迹只是 behavior prior，
  高手 replay 才是外部行为信号；二者必须有 source/policy 标记并分别采样。
- on-policy PPO 数据保存逐 slot legal mask、采样前 logits/log-prob、bundle 总 log-prob 和 policy version；
  同一批只做少量 `2–4` epoch。重复优化不增加独立样本数。
- 更旧 rollout 只在有 importance ratio/V-trace 等明确 off-policy 修正时参与 policy-gradient；否则仅训练
  BC/critic。没有 behavior log-prob 的历史 JSON 绝不反推 PPO ratio。
- 自回归 bundle 的 log-prob 是合法 slot 的条件 log-prob 之和；entropy/advantage 按有效 slot 数归一，
  防止空地多的状态仅因动作长度更长而支配梯度。按 slot 数 bucket/pad，首版每批约 `128–512` 个完整
  state group；模型很小时再 profile 是否扩大，而不是先追求 NPU 大 batch。

### 4.3 降噪与 seed 泛化

- forced-candidate 比较使用同 checkpoint、同 future seed 的 common random numbers，并报告实际方差比；
  多 replicas 先在 checkpoint 内聚合，不能把它们当独立状态。
- value baseline/GAE 只看合法线上信息；优势按 state/future group 居中或使用固定 anchor paired
  advantage，避免绝对终局钱数的跨状态方差淹没动作信号。
- 网络 forward 永不接收 seed、future RNG、对手名/family 或对手 private。family 只用于采样平衡、
  group split 和审计；十个公开脚本中的同源 Ahmed 派生不能按十个独立分布加权。
- `(seed, opponent family)` 是不可拆分 group；同 seed 双座、所有候选、replicas 与 policy arms 一起
  分配。训练 seed 持续生成，验证使用互斥 seed 段，并永久冻结一段 final blind seeds，不能因中间结果
  回收进训练。
- early stop 看 held-out calibration、paired win/margin regret 和严重 harm 率，不看训练 return；重复读取
  同一轨迹只增加优化步数，不增加有效样本量。

2026-09-22 的首批真实证据为 `48` 个 hard-3 step288 状态、`188` 个 normal 候选、
`4` 个成组 future replicas，共 `752` 局（`work/r1-real-suffix-hard3-seed2609820000-n16-r4-v2.json`）。
`140` 个非 anchor 候选中 `76.4%` 的 paired margin 在四个 future 上正负都出现；差值标准差中位数
约 `$1,205`，而状态内 margin-oracle 收益中位数仅 `$285`。CRN 后差值方差相对两条绝对
margin 方差和的中位比为 `0.226`，证明配对有效，但 `4` replicas 仍不足以把单状态
argmax 当作硬分类真值。训练必须使用 group mean/分布头与不确定性权重；单次胜负不得入选
为“最优候选”标签。

## 五、C++ 加速边界

应放进 C++：

- FastEnv 批量环境、checkpoint clone、future reseed；
- teacher 的候选生成、规则 mask、repair、闭环多情景 suffix；
- 每状态共享中间量与 common-random paired rollout；
- 自博弈 student inference、特征抽取和固定尺寸二进制训练 shard；
- 线上 GRU/MLP 前向（导出权重，使用小型手写 dense/GRU kernel 即可）。

保留在 Python/PyTorch：

- gzip 原始轨迹索引和实验调度；
- PyTorch 数据加载、mini-batch 训练、损失与校准；
- 模型选择、离线分析和权重导出。

“Python/PyTorch 训练”不意味着逐样本跑 Python：矩阵运算、GRU、反向传播和优化器实际由 C++/oneDNN
或 NPU kernel 执行。原始 replay/对局 `jsonl.gz` 只读一遍并保留作档案；训练热路径使用带 schema hash
和 group offsets 的定 dtype mmap shards，DataLoader 只切片二进制数组。全 C++ 训练会引入 LibTorch 或
自写 autograd/optimizer，却不解决 rollout 与 JSON 解析瓶颈；只有 profiler 证明 batch 训练本身成为主
瓶颈时才考虑。

首版总参数预算约 `8万–18万`，即使使用 3–5 个 critic ensemble 也控制在 `50万` 以下。这个规模 CPU
训练通常已足够；统一使用 `npu-torch` conda 环境不等于必须把 tensor 放到 NPU。只有训练样本达到千万级、
大规模 ensemble 或超参搜索时才启用 NPU，否则设备初始化和搬运可能比计算更慢。teacher rollout 的
瓶颈始终是 C++ 仿真/规划，NPU 不能加速这部分。

公开对手的 Python 版只承担校准、parity 和最终验收；大吞吐 RL 使用经逐步 action
parity 验证的 C++ 代表实现。对手池以源码家族而非脚本数量计权：主体是最新强公开家族和
student 历史冻结快照。慢动态 R1 只在 RL 循环之外生成离线 BC 标签，不作为对手也不随游戏调用。
C++ 移植未完成不阻塞
首批 BC/self-play；每完成一个家族 parity 才将其加入训练池。

2026-09-23 的速度数字必须按执行层解释：当前单次 native-plan callback smoke 虽使用 C++ `FastEnv` 和
C++ planner/executor，但 141k actor 仍由 Python/CPU PyTorch 在 109 个 slot 上逐次回调。该局 719 步
wall `6.081s`，我方全部 719 次 agent 调用合计 `3.283s`，其中 18 次中盘 callback plan 合计
`0.750s`；它不是纯 C++ student 基准。仓库原生 benchmark 的两个完整 R1 对弈在单线程为
`9.813s/game`，192 局×192线程为 `8.409 games/s` 聚合吞吐；这测的是两个昂贵 R1，不是最终小网络。
相反，纯 C++ replay shell + FastEnv 在 96 线程已有 `9.3k–13.1k games/s`。因此环境不是主瓶颈，正式
RL 必须把 actor 前向和强对手一起留在 C++ 热路径，并在动作契约稳定后重新实测，不能从上述两端数字
臆测最终速度。

## 六、阶段和准入

现有非学习 R1 是冻结 baseline。learned student 必须使用独立实验入口/二进制和默认关闭的选择开关；
关闭时同一 observation 的逐步 action 必须与 baseline bit-exact。任何训练代码不得覆盖
`policy/r1/agent.so`、`policy/r1/config.json` 或 `agent/main.py`，也不得改变早期 replay。student OOD、
非有限输出或可行性失败时无条件回退 baseline R1。

### A. 数据与教师正确性

实现 3–5 个公开可生成的对手库存/销售情景，满足现金、库存、市场逐单位守恒；嵌套候选集做终局闭环
paired suffix。先报告情景校准、候选方向和 regret，不改生产。

### B. Student 蒸馏

直接训练 masked autoregressive slot actor，逐格输出 `SKIP / 5 crop / 3 animal`。首版 BC 每个状态只
模仿生产 R1 选中 proposal 的因果一致 slot trace；不训练 candidate-ranking 网络。held-out 上报告 masked
accuracy、逐类召回、完整序列一致率和非法 argmax（必须为 0），并保留 OOD 回退。

首个真实闭环已于 2026-09-22 跑通。slot ABI v1 会在 preview 把 label 改成 `SKIP` 后继续导出改写前
proposal 的 prefix，违反因果性，已整体拒绝；ABI v2 固定
`label_semantics=greedy_proposal_before_preview`、`prefix_semantics=before_current_slot`。48 个 step288
状态均只取生产 normal-score winner，proposal key 对齐 `48/48`，共 236 个槽，相同输入标签冲突为 0，
teacher label 合法率 100%。141,573 参数 GRU 的 one-batch overfit 为 `2.0086→0.00066 / 100%`；
39/9 group split 的 train/held-out loss 为 `0.2562/1.7322`，masked accuracy 为 `93.30%/45.24%`，非法
argmax 均为 0。50 epoch 的纯训练用时 8.56 秒（CPU，约 1,133 slot-epochs/s）。因此当前结论仅是数据与
优化闭环正确；held-out 泛化尚不可接受，模型不部署。正式记录为
`work/student-v1/autofill-slot-v2-student-v1.metrics.json`，其 schema 和 shard digest 必须由消费者核验。

#### 2026-09-23 更正：旧 live-winner slot 数据不能作为线上 actor 的正式 BC

上面的 ABI v2 只证明「导出的 winner trace 内部因果一致」，没有证明它与线上执行 scaffold 一致。
生产 `SearchController` 先用多套 `Settings` 生成 proposal，再把 score winner 的 controller 安装一天；
旧 slot 数据因此属于 winner 自己的 `discount/capital_power/competition/...`。诊断 student 在线执行时却固定
从 `h.policy.base` 构造逐格 callback。两者不是同一个 MDP 动作接口。

硬门产物 `work/student-v1/base-vs-winner-heldout-hard-gate-v1.json` 在 4 个 held-out 整组、20 个日边界上
逐前缀重放 winner，并逐元素比较 347 维资源日历：

- 只有 `15/20` 状态能在 base scaffold 中完整重放 winner 的格序/动作；5 个状态直接在前缀结束前失配；
- 只有 `14/20` 状态的资源完全相等，可重放状态的最大绝对资源差为 `18`；
- step288 的 `4/4` 样本全部失败。首个样本中 winner 依次处理
  `(84,COW),(22,STRAWBERRY),(94,COW),(1,STRAWBERRY)`，base 在首项后却直接转到 cell94；
- 根因不只是在同一空格上选了不同 kind。`discount` 会改变现有有限作物的 harvest/release 日，设置还会
  改变终止和扩地条件，因而连「今天有哪些可决策格、格序、资源前缀」都会改变。

所以旧 `live-slot-v2-full19328-merged` 及其 checkpoint 只能保留作旧 teacher 档案和接口 smoke；其
`81.24%` held-out masked accuracy 是 winner-scaffold 上的 teacher-forced 指标，不代表 base-scaffold
闭环精度，也不能据此启动正式 RL。在线 smoke 原名 `teacher_agreement` 的量其实只是与当前 base greedy
建议一致，已改名 `base_greedy_agreement`。

固定 base 后重导 base 自己的贪心标签虽然执行一致，但只会蒸馏较弱 base，不会蒸馏 score winner。
正式动作契约必须先补齐 winner 方案中 actor 目前无法表达的控制量：至少包括现有有限作物的当日
`KEEP/RELEASE` 与土地扩张；随后才按确定格序逐格输出 `NONE / 5 crop / 3 animal`。`NONE` 不应继续隐式
等价于“终止所有后续格”，否则网络仍受旧贪心停止规则限制。规则层继续负责 legality、现金/劳动日历与
确定性执行。只有新动作前缀在训练导出与在线 callback 间逐字段一致，BC 才可放行。

### C. BC + population self-play

BC warm start 后只在 C++ 强公开对手与 student 历史快照池中做中盘 RL。R1 不进入 rollout；如离线给
采样状态补逐格标签，也只进入下一轮 BC。不能从一开始做 719 步端到端 PPO，也不优先使用
CQL/Decision Transformer。

### D. 部署验收

依次通过规则不变量、固定 suffix、互斥 seed 七强 warm 链、seed-paired Wilson、x86_64/aarch64 parity。
“每手 >80%”仍需约 683–1537 seed 才能严格声明。

## 七、能达到什么程度

当前综合完成度约 `35–45%`：理论诊断约 65%，正确 scorer 原型约 30%，可部署 learned policy 不足 10%。
已有闭环 normal 的量级证据支持“显著优于当前”是现实目标；但部分可观测、未知对手响应与候选覆盖决定了
不能诚实承诺博弈全局最优或固定 90%+。可证明的目标应是：在公开信息定义的情景分布上，student 的
held-out candidate regret 接近 teacher，并在未知 seed/对手族上持续提高真实胜率。
