# step288 后 PPO day-state control variate 审计

日期：2026-09-23  
状态：理论/离线 probe 已完成；独立诊断与 trainer 的显式 opt-in seam 已实现并通过小测试。
**默认关闭，尚未用它更新 actor；C++ actor、rollout 和生产 agent 均未改变。**

## 1. 结论先行

当前 same-environment-seed leave-one-out（下文记为 `B_peer`）在 v16 的
Thomas/Meta 完整 seed block 上很有效，但固定使用 `R - B_peer` 等价于把 peer
系数硬编码为 1，并非一般的最小方差 control variate。

建议下一轮只做一个训练期、默认关闭的实验入口：

1. 用当前 rollout 已有的 day-start observation 构造下文固定的 163 维状态；
2. 按**环境 seed 整组**做 6-fold cross-fit；
3. 每折拟合一个 global `HistGradientBoostingRegressor`，共同覆盖 17 个 day，预测
   `E[R | H_day, B_peer]`；
4. 将 out-of-fold 预测裁到 reward support `[-1.1, 1.1]`，冻结后用于这一轮 PPO；
5. critic 不共享 actor encoder、不反传进 actor、不导出 C++，也不进入线上 forward。

v16 的可行动 day 上，global HGB 把标量 advantage 方差降到当前 baseline 的
`0.8608`；逐 day 的 102 个模型可到 `0.8276`，但并行拟合仍需 `23.56s`，而 global
6 模型只需 `3.62s`。当前整轮约 `154.48s`，所以首版应选 global：少约 3.3 个百分点
的标量降噪，换来少 20 秒和明显更小的实现面。

这不是让 critic 决策。逐格自回归 actor、day joint ratio、legal mask 和 C++ rollout
语义均保持不变。

## 2. 数学条件

### 2.1 day bundle 的正确 score

对游戏 `i` 的第 `d` 个 day，令 day-start 历史为 `H_id`，整日自回归动作序列为
`A_id`。当天 score 是所有 actionable slot score 之和：

```text
S_id = grad log pi(A_id | H_id)
     = sum_k grad log pi(a_idk | H_id, a_id,<k)
```

forced slot 只推进 hidden，不进入 likelihood。终局 return `R_i` 对每个 actionable
day bundle 使用一次，不能按 slot 数再除。

### 2.2 baseline 何时不增加 policy-gradient 偏差

只要 baseline `b_id` 在采样当前 `A_id` 前已经确定，且在给定 `H_id` 后不依赖当前
day action，就有：

```text
E[S_id * b_id | H_id] = b_id * E[S_id | H_id] = 0.
```

因此下列 baseline 合法：

- 当前局以外、policy RNG 独立、同 environment seed 的 peer returns；
- 只读当前局 **pre-day** 状态的冻结 critic；
- 按 environment seed 整组 cross-fit 后对 held-out seed 给出的预测；
- 只用以前 rollout 训练、在收集本轮之前冻结的 lagged critic。

`seed`/`opponent` 可用于组成 block、切 fold 和审计，但不能作为 critic 或 actor 的输入。
`seat` 只可用于把棋盘规范成 own/rival；规范后也不作为输入。

这里的“无额外偏差”严格对应 on-policy score estimator，或 PPO 在 `theta_old` 的一阶点。
PPO 的 ratio clipping 本身在离开 `theta_old` 后已有偏差；任何 baseline 都不能把 clipped
surrogate 变成全局无偏估计。cross-fit 的保证是它不再额外引入“critic 看过当前 action/
reward”的有限样本泄漏。

### 2.3 为什么同轨迹 leave-in critic 不成立

不能用当前轨迹的 `R_i` 拟合 critic，再立即给同一轨迹算 `R_i - V(H_id)`，即使
对 `V` 调用 `detach()`：

- critic 参数已经经由 `R_i` 依赖当前 action；
- `detach()` 只切断 autograd，不切断统计依赖；
- 极端插值器可令 `V(H_id)=R_i`，把 actor gradient 人为归零。

也不能把同一游戏的 17 个 day 随机切到不同 fold。最小安全分组是完整 environment
seed：同 seed 的全部 opponent、双座、17 个 day 一起进同一 fold。

若根据某一 rollout 的 held-out reward 反复选择特征、超参或决定是否启用 baseline，再
回头用同一 rollout 更新 actor，模型选择本身也形成 leave-in 依赖。本审计的配置应在
**下一份新 rollout 收集前锁定**；v15/v16 只用于确定方案，不应用它们重做一次 actor update。

### 2.4 `beta=1` 不是理论要求

若只使用 peer baseline，最小化普通 residual 方差的系数为：

```text
R ~= alpha + beta * B_peer
beta* = Cov(R, B_peer) / Var(B_peer).
```

v16 的 6-fold cross-fit `beta` 约 `0.813`，仅改成 cross-fit 线性 baseline 就把当前
advantage 方差从 `0.580459` 降到约 `0.5539`，即当前的约 `95.4%`。

但真正使 policy-gradient 方差最小的 state baseline 是 score-norm 加权条件期望：

```text
b_g*(H) = E[R * ||S||^2 | H] / E[||S||^2 | H].
```

本审计报告的是**标量 advantage 方差**，不是完整 gradient covariance 的证明。它是值得
进入下一轮的强证据，但上线前仍应补一个按 day score-norm 或至少 actionable-slot 数加权
的方差审计。

## 3. 163 维 pre-day feature contract

### 3.1 信息边界

只允许 day action 生成前已经存在的字段：

- day-start public board/market/town；
- 我方当时可见 private shed、seed 和 hand inventory；
- `B_peer`。

明确禁止：

- 当前 day 的 `event_resources`、legal mask、action、old log-prob、entropy；
- 当前 day 已选择的任何 autoregressive prefix；
- terminal cash/margin/reward；
- seed、opponent、route、policy seed 等 identity。

对手可通过其**公开状态**表现出的行为留下可泛化信号；这不等于输入对手身份。

### 3.2 NPZ 与反归一化

原生 rollout 已有全部输入，不需要改 C++ capture，也不需要扩 NPZ：

```text
day_session_index   (N_day,)
day_step            (N_day,)
observation         (N_day, 3074)
observation_length  (N_day,)
seed                (N_game,)       # only grouping/audit
seat                (N_game,)       # only identity/parity audit
```

`observation` 是 C++ actor 按行为 checkpoint 标准化后的 float32，不是 `_pack()` 原值。
必须使用生成该 rollout 的 checkpoint normalization 做：

```text
raw = observation * observation_std + observation_mean
```

禁止拿更新后的 checkpoint normalization 反解。反解后须验证所有离散列接近整数、
`raw.step == day_step`、`hour == 0`、`player == 0`，否则 fail closed。

在 hour-zero canonical pack 中，farm 已经是 own-first，且 hands 为空、private inventories
固定为 `[{}]`。固定 offset 为：

| 范围 | 语义 |
|---|---|
| `0:4` | step, day, hour, canonical player |
| `4:10` | own money, farmer x/y, hand count, unlock mask, hires |
| `10:1510` | own 100 tiles × 15 fields |
| `1510:1516` | rival farm header |
| `1516:3016` | rival 100 tiles × 15 fields |
| `3016:3028` | own shed 12 |
| `3028:3033` | own seeds 5 |
| `3033` | inventory bag count（应为 1） |
| `3034:3046` | 当前 hand inventory 12 |
| `3046` | inventory order length（hour zero 应为 0） |
| `3047:3056` | market inventory 9 |
| `3056:3065` | market prices 9 |
| `3065` | unlocked shop count |
| `3066:3066+nshop` | shop ids；其后为 capacity padding |

tile 的 15 列与 `policy/r1/agent.py::_pack` 一致：kind、crop id、animal id、planted day、
placed day、yield、consecutive unwatered/unfed、fertilized-until、pending care、lifespan、
watered/fed/cared/fertilizer-available。

### 3.3 精确 163 维聚合

按固定顺序拼接：

1. 4 维：day、`29-day`、own-rival money、own+rival money；
2. own/rival 各 52 维：
   - 5：money、unlock mask、hires、farmer x、farmer y；
   - 7：kind `0..6` 的 tile 数；
   - 20：5 crops 各自的 count、yield sum、`sum(day-planted_day)`、
     consecutive-unwatered sum；age 只在该 crop 的现存格上求和，不裁剪、不平均；
   - 15：GOOSE/COW/SHEEP 各自的 count、yield sum、`sum(day-placed_day)`、
     consecutive-unfed sum、pending-care sum；age 只在该 animal 的现存格上求和，
     不裁剪、不平均；
   - 5：`fertilized_until_day >= day`、watered、fed、cared、
     fertilizer-available 的 tile 数（后四项分别直接汇总 tile bool 列）；
3. 29 维：own shed 12、seeds 5、hand inventory 12；
4. 18 维：market inventory 9、price 9；
5. 8 维：shop-id multi-hot。

总计 `4 + 2*52 + 29 + 18 + 8 = 163`。首版必须冻结列名、顺序和一个 schema hash，
不得把 3074 维 raw observation 直接喂树，让 padding/顺序偶然性变成模型特征。

## 4. 已有 v15/v16 证据

### 4.1 现 baseline

| rollout | raw reward var | 旧 opponent-seat LOO | 当前 same-seed LOO |
|---|---:|---:|---:|
| v15 Thomas/Meta/G397 | 0.967939 | 0.826009 | 0.918059 |
| v16 Thomas/Meta | 1.067658 | 1.071878 | 0.580459 |

v16 中同 seed 四局相关，当前 baseline 很有效；v15 加入与两强手收益几乎不相关的 G397
后，硬编码 `beta=1` 反而变差。这正是需要学习 `beta` 而非把配对平均当理论真值的原因。

### 4.2 固定 probe 配置

所有 state probe 都按 environment seed 做 6-fold cross-fit；同 seed 的全部局/日不拆分。
HGB 固定为：

```text
max_iter=100
learning_rate=0.05
max_leaf_nodes=7
min_samples_leaf=40
l2_regularization=5
early_stopping=False
random_state=<由 fold 确定的固定值>
```

`early_stopping=False` 很重要，避免 sklearn 的 `auto` 再随机抽内部 validation。建议把
out-of-fold 预测裁到终局 reward 的合法 support `[-1.1, 1.1]`；投影到 target support
不会增加逐样本平方误差，而且仍是 action-independent baseline。下表 global 数值来自未裁剪
预测（其预测未缓存，未重跑）；per-day 同时测了裁剪前后。

v16 的 25,072 个 actionable day：

| baseline | `Var(R-b) / Var(current advantage)` |
|---|---:|
| 当前 `R-B_peer` | 1.0000 |
| cross-fit beta-only | 约 0.954 |
| beta + cash HGB | 0.9113 |
| beta + 163D linear ridge | 0.9107 |
| beta + 163D global HGB | 0.8608 |
| beta + 163D per-day HGB | 0.8280 |
| per-day HGB + support clip | **0.8276** |
| frozen actor hidden + ExtraTrees | 1.0131 |

per-day HGB 六折为 `0.9003/0.8136/0.8289/0.8219/0.7753/0.8469`；按 seed
cluster bootstrap 的 95% CI 为 `[0.7894, 0.8732]`。逐日全部小于 1，step288/312
约 `0.915/0.916`，到 step648/672 为 `0.697/0.650`。

v15 去掉 G397、只在 Thomas/Meta 四格内重算 peer baseline 的后验机制复核为
`0.8085`（clip 后），95% CI `[0.7572,0.8668]`。这不是 v15 原正式采样设计，不能当
独立确认集，但与 v16 接近，说明收益并非单个 batch 的孤立异常。完整 v15 的 `0.526`
主要来自公开状态很容易区分 G397，不应用来外推下一轮两强对手收益。

现有 actor 的 frozen 192D initial hidden 没保留足够 value 信息；线性 probe 基本只复现
beta-only，高维 ridge 还会严重外推，ExtraTrees 略差于当前 baseline。因此不要在现 hidden
后随手挂 value head并宣称已经有 critic。

### 4.3 收益尺度与开销

global HGB 的 `0.8608` 相当于：

- 标量 advantage 方差约减少 `13.9%`；
- 若 gradient score 权重没有系统性反转，标准误约乘 `sqrt(0.8608)=0.928`；
- 样本效率代理约乘 `1/0.8608=1.162`，即 1,536 局约等价于多 249 局。

per-day clip 的 `0.8276` 对应约 `20.8%` 的样本效率代理增益，但不是 gradient variance
的精确保证。

同一机器、底层 OMP=1：

| 方案 | fits | wall |
|---|---:|---:|
| per-day, ThreadPool 32 | 102 | 23.56s |
| per-day, ThreadPool 64 | 102 | 24.92s |
| global, 6-way parallel | 6 | 3.62s |

64 线程没有收益。global 首版最多开 6 个独立 fit；不要为了显示 CPU 满载而制造过度并行。
离线 probe 的 25 秒“解析”来自一次性 Python 逐行聚合，不是必要训练成本；正式实现应直接
在已加载的 NPZ observation 上按上述固定 offset 分块向量化，禁止重新 JSON 化或构建 event dict。

## 5. 最小实验实现路径（尚未执行）

保持正式 trainer 不动，先新增一个独立诊断模块/入口，职责只包括：

1. 从 rollout NPZ、行为 checkpoint 提取并校验 163D day features；
2. 由 `day_session_index` 将 terminal reward 和 `B_peer` 映射到 day；
3. `fold = deterministic(environment_seed) % 6`，验证每个 seed 只在一个 fold；
4. 六折并行 fit global HGB，生成每个 day 唯一的 out-of-fold prediction；
5. clip 到 `[-1.1,1.1]`，输出 day-aligned advantages 和审计 JSON；
6. 不保存可在线加载的 critic，不修改 actor checkpoint 权重。

独立诊断验收后，再给 trainer 增加一个显式 opt-in baseline mode。最小 trainer seam 是：

- 现 `_paired_seed_loo_advantages()` 保留作 frozen baseline；
- 新模式产生**day-aligned** advantage，而不是当前 game advantage；
- `_day_bundle_objective()` 仍先对 slot log-ratio 求和、对 joint ratio clip 一次，然后每个
  valid day 取对应 day advantage；
- 绝不按 slot 数缩放 advantage；
- cross-fit prediction 在完整 PPO epoch 内冻结，不能每个 minibatch 重拟合。

当前 `_batch_terms()` 只返回 minibatch-local `day_games`。接入时可为恢复后的 day 记录增加
稳定的 `day_index`，或在 rollout 恢复时一次性附上 scalar advantage；不要用浮点状态内容做
隐式 join。

## 6. 必须通过的测试/验收门

### 6.1 数据信息边界

- feature schema 精确等于 163 维且 hash 固定；
- identity 字段不在 feature names；
- 反归一化 round-trip 与 `_pack()` checkpoint fixture 逐列一致；
- 每个 day feature 只读该 day 的 state slice；
- 改写当前局 action/reward 后，当前局 critic prediction 不变；其 residual 可变；
- 同 seed 任一局属于 fold `k` 时，该 seed 的任何局/日都不在 fold `k` 的 fit rows。

注意：改写当前局 reward 会合法地改变**其他 peer 局**的 `B_peer`，所以泄漏测试要检查
“当前局自己的预测不读自己的 reward”，不能错误要求整个 batch 所有预测都不变。

### 6.2 PPO 语义

- constant day baseline 模式与旧 game-advantage 路径 loss/gradient exact parity；
- 每个 actionable day 恰好一个 advantage；forced-only day 不进 loss；
- day joint log-ratio、clip fraction、KL 计算与旧路径 exact parity；
- prediction/advantage 含 NaN、fold 不完整、schema 不匹配时 fail closed；
- learned-baseline off 时 checkpoint、optimizer state 和 action parity 完全不变。

### 6.3 统计报告

下一份新 rollout 在更新 actor **之前**固定输出：

- raw/current/global-HGB residual variance；
- 每 fold、每 day 的 variance ratio；
- prediction min/max/p99、clip fraction；
- seed-cluster bootstrap CI（可放离线审计，不必阻塞每轮）；
- 至少按 actionable-slot count 加权的 residual ratio；最好再补实际 day score-norm proxy。

不要用当轮 realized VR 再选择 beta/HGB/per-day 三者后回头更新同一 batch。方案应预注册为
global HGB；统计只作监控和下一轮决策。

## 7. 是否值得进入下一训练轮

**值得，但只能以 opt-in cross-fit global HGB 进入一份新的 rollout。** 理由是：

- 数学上不读取 held seed 的当前 action/reward，不改变期望 on-policy score gradient；
- v16 正式 block 的 CI 全在 1 以下，v15 Thomas/Meta 后验复核方向一致；
- 3.62 秒 fit 相对 154.48 秒整轮开销很小；
- 不碰 actor forward、C++ rollout、生产 agent，回滚面只是一条训练期 baseline seam。

不建议现在做的事：共享 actor encoder 的 value head、GAE/TD bootstrapping、逐 day 102 模型、
把 critic 导出 C++、或再扩大网络。这里的目标只是一个低风险 control variate，不是新建第二套
决策系统。

## 8. 实现与历史复现（2026-09-23）

实现保持实验隔离：

- `experiments/day_state_control_variate.py`：冻结 schema/hash、vectorized 163D extractor、
  whole-seed folds、peer baseline、global HGB 和 day-aligned advantage；
- `experiments/audit_day_state_control_variate.py`：只读历史 NPZ 诊断；
- `experiments/train_student_action_event_rl_v3.py`：只有显式
  `--day-state-crossfit-baseline` 才启用，且强制 native JobBatch、`margin_weight=0.1`；
- off 时仍走原 game advantage 分支；critic 不保存进线上模型，不改 C++。

19 项 targeted tests 通过，覆盖 schema/identity、behavior normalization、pack offsets、whole-seed
leave-out、当前 reward 不进入自己预测、constant day advantage 与旧 game advantage 的 loss/gradient
exact parity，以及排序后的 result 与原生 array-order day 的稳定 identity join。

唯一历史复现产物为：

```text
work/student-v1/v16-day-state-crossfit-hgb-audit.json
rollout SHA256: 9c7c14a45ccba8bd7b2bfcb3b607bb21ed2caa43da2e35f349aa9193a3da0a72
behavior checkpoint SHA256: f53fd66351b23dcc35a1f5bb81da20ed5b9b3eeb7647a5739170bde3fc51a962
```

该实现对 v16 的复现为：

- vectorized extraction `3.119s`，6-fold fit `3.701s`；
- actionable-day variance ratio `0.800867`；
- actionable-slot-count 加权 ratio `0.781831`；
- 六折均小于 1：`.754/.817/.842/.815/.822/.769`；
- 17 个 step 均小于 1，从 step288 的 `.907` 到 step672 的 `.610`；
- prediction clip fraction `0.368%`。

这个 `0.8009` 比一次性 probe 的 global `0.8608` 更好，但旧 probe 没保留可执行预测 artifact，
两者不能事后宣称 exact 同口径。现在以冻结 schema/hash、可重跑 JSON 的实现结果作为工程复现值，
仍只把它当历史诊断；正式 actor 更新必须等待全新的 v22 on-policy rollout，不能回头更新 v16。

真实 v16 pack 还证伪了“unlocked shop id 必须唯一”的假设：列表中确有重复 id。最终 extractor
保留长度与 id 域校验，把 8 维 shop 特征定义为 presence multi-hot，不再错误要求 one-hot 数量等于
原始列表长度。
