# 数学模型与实现静态审计补充

> 日期：2026-09-22。先做源码与产物静态核对，随后只对理论审查得到的候选做配对验证；
> 本文中的新公开对手和 `candidate_extra` 数字来自 FastEnv，对正式部署结论仍以官方口径为准。
> 本文补充 `ECONOMIC_MODEL_ZH.md`，不覆盖其中已有实测结论。

## 已确认的边界

- `max_animals=40`、`replant=0.5` 是已验证的正式优化，不作为风险项。
- 路线树使用单次随机结局标签是当前算力约束下的主动折中：warm R1 太慢，R1 未来价值本身也有较大噪声。后续不能把“大规模 warm-R1 多未来重采样”写成近期必做项。
- 不做对手身份识别。可使用的只有当前公开状态、公开历史和候选动作；对手名字、代码指纹、手工风格分类均不进入线上特征。

## 因果纪律

修改顺序固定为：先从规则、目标函数或执行依赖推出应满足的不变量；再检查共享实现是否违反；
最后才用轨迹确认触发域、量级和副作用。败局相关性本身既不是根因证明，也不是改代码的准入条件：
相关可能来自共同上游状态，缺少相关也可能只是样本不足或效应被后续策略抵消。轨迹只负责证伪理论、
确认代码路径和估计实际收益，不能从现象反向编造机制。

## 仍成立的模型风险

### 1. 三层效用没有统一

- 路线树标签按单次结局先胜负、再 margin 排序。
- R1 主价值为折现后的 `own_cash_flow - competition * rival_cash_flow`，正式配置 `competition=2`。
- sale DP 内部使用 `competition=1`。
- 正式验收看 `P(final_own_cash > final_rival_cash)`。

`competition=2` 可能是有效的经验校准，但它不等价于 margin，更不等价于胜率。不能再把 R1 predicted value、平均 margin 和胜率当成同一个量。近期不重写 R1；至少要求所有实验同时报告配对胜负翻转、seed-level 胜率和 margin，且只以胜负准入。

更精确地说，`Planner::value()` 中的 `enemy` 是固定公开供给假设下的**预计销售收入**，不是对手
终局现金流：它没有对手未来投资、工资、采购和条件策略。因此 `own - 2*enemy` 只是用市场外部性
构造的排序代理，不能解释成 `E[final_own-final_rival]`。把系数机械改回 1 也不会补全缺失状态；
而且该参数方向已经过正式 warm 链筛查，不因这个命名错误重开参数扫描。

还要避免一个看似简单、实际无效的“修复”：对 R1 的单个确定性 margin 预测套 sigmoid、截断或
零点附近加权。任何单调变换都保持候选次序不变。要真正优化胜率/CVaR，必须先让同一候选产生一组
未来情景（至少覆盖未知商店与成交时序），再对分布评分；当前点预测里没有可供尾部目标使用的方差。

### 2. 147 维状态存在候选兼容性别名

现有特征聚合了数量、yield、stress、库存、价格和现金，但没有充分表示地块位置/年龄/过期、工人位置与手持物、以及候选 tape 的动作前置条件。两个 147 维相同的状态可能对同一候选路线具有完全不同的可执行性。

若以后加特征，优先增加 `state × candidate` 的兼容性量：未来现金缺口、资源覆盖率、不可执行动作数、地块/作物时序冲突、候选供给变化乘当前市场边际价值。只统计候选 tape 固有动作数的常数特征没有分裂信息。

### 3. cp168 有条件分布偏移

线上只有 cp144 没切换的状态才能到 cp168；独立从固定 opening 生成的 cp168 训练集并不是这个 survivor distribution。下次重训 cp168 时，应先按正式 cp144 策略 rollout，再只对未切换状态生成 cp168 反事实。该修正不要求调用 warm R1。

### 4. replay 聚类距离忽略了已提取的经济变量

`analyze_macro_route_library.py` 已提取 spend、revenue、money、capital requirement 和 slack，但 `_distance()` 只使用资产数量、布局、粗动作计划和土地/工人。聚类可能合并形态相似而现金节奏不同的路线。下次重建路线库时应复用现有提取结果，把现金谷底、资本 slack、销售/收获时序加入距离；不另写聚类管线。

### 5. R1 的市场和供给预测仍是近似

- 正式构建使用 `R2_MARKET_INTEGRAL=0`，批量平均报价跨越价格地板时不等于官方逐单位成交；代码已有 `ConditionalMarket::execute()` 的精确 primitive。
- 未知商店使用均值需求穿过非线性价格曲线，存在 Jensen 偏差。
- `best_rotation()` 当前按毛商品流估值，未计 seed、劳动、工资、土地租金和自身价格冲击。`replant=0.5` 的规模可以正确，但补种商品构成仍可能偏。
- portfolio 先按完整价值算 gain，再除以 `cost^capital_power` 贪心排序；这是 ROI/NPV 混合启发式，不是声明价值函数下的预算最优解。

这些近似可能互相补偿，因此不能把单点修正未经下游重校直接上线。按当前方法论，先由官方规则与
内部不变量确认问题，再量化候选差分、重校受影响的代理，最后才用真实 warm 链的跨 seed 配对
验证部署；warm 胜率不能投票否定已证明的规则/守恒错误。

## 本轮静态修复

1. 撤除未同步的 40 维 candidate-lookahead：恢复 Python、native、训练矩阵和部署树的 147 维契约。该 lookahead 在固定 checkpoint 内只由 tape 和 step 决定，基本是样本常数；seat1 还会因缺少原始 `step` 从 tape 开头读取。
2. `train_robust_search_route_trees.py` 读取多个 compact 文件时，同一 `(opening, checkpoint)` 现在拼接数据，不再由后一个文件静默覆盖前一个；同时拒绝 target/feature schema 不一致。
3. `verify_project.py` 改为验证当前两个 fallback、147 维 schema 和 `handoff_land_delay_days=2`，不再要求旧浅树产物逐字相等。
4. replay budget guard 已撤除。正常前缀由训练出的浅树和原路线完整执行；只有确认无法由 replay 自愈的硬失败才允许研究兜底，而且必须单独做七强泛化验证。
5. `eval_seed_paired.py` 不再把 `{0, 0.5, 1}` 的双座 seed 单元塞进 Bernoulli Wilson 公式，改报 seed-clustered mean CI。
6. `run_strong_ab.py` 在任一对局报错时不再给出可误用的完整胜率，记录入口文件 hash，并在写出诊断结果后以失败状态结束。

## G195 第三块地失败的精确边界

- G195 在 step 149 成功从一块地扩到两块；step 219 以 1886 现金提交第三块地，尚差 114。
- 同帧单位动作会先把 4 WOOL 放入 shed，随后才结算市场；真正的问题是 overlay 把提前到本帧的 `SELL WOOL 4` 追加在 `BUY_LAND` 后，买地检查资金时尚未获得销售收入。
- replay 此后不再提交 `BUY_LAND`。step 276 仍为两块地；到固定接管帧 step 288，R1 才以 11671 现金成功购买第三块地。因此它不是 replay 自愈，而是 R1 在接管后补买，原计划被延后 69 step。
- 曾实验把买地延至 step 220、等羊毛入仓后再买。它会明显提高接管前产能，但也改变 weed/商店 RNG 消耗及共享市场价格；fresh 16-seed 七强 A/B 为 Thomas +2 胜、Melon -2 胜，其余 0，不能部署。
- 最终修复不延迟买地、不新增销售：只在 step≥168、首槽 `BUY_LAND` 现金不足且同帧既有可执行 SELL 足以补齐时，把这些 SELL 移到买地前。三个互斥 FastEnv 段共 208 seed，17 个独立 seed 触发，按六个会触发的公开对手聚合为 15 正 2 负；64-seed 段名义净胜 `+18`、128-seed 段 `+21`，全部段均为 0 个胜转败。两个已知救败 seed 的 56 局官方解释器复核方向一致。默认开启，可用 `REPLAY_CAPITAL_SELL_FIRST=0` 回退。
- 结论：异常来自市场队列的依赖顺序，不是买地目标或路线时机本身。修复仅让原本同帧已计划的现金流先于资本订单执行，不改浅树、路线内容或正常 replay 经济节奏。

## 公开脚本的“现金兜底”与我方的真实差异

- Thomas / Melon / Demand / Ahmed / Pipe 的公共 chassis 虽保留了一个“预算未来 72 帧购买、不足则额外卖货”的 `budget_guard` 实现，但它们的最终 `_SETTINGS` 均为 `budget_guard=False`。因此不能把这段保留代码当成线上行为。
- 它们实际启用的是：把已定 tape 中未来 1–14 帧的 SELL 提前，从 step 144 起尽量把可执行 SELL 排在购买前，并保护待拾取/待消耗的物资。这是“冻结路线的提前变现”，不是持续维持某个现金余额的通用策略。
- 这五个脚本的 41 条原生路线都在 step 150 买第二块地、step 265 买第三块地；Salemali 的单路线为 step 160 / 240。这是 tape 离线规划出来的延后资本时点，不是“买地失败后等钱够再试”的在线控制器；未发现通用的 BUY_LAND 失败重试。
- 我方 replay 也已有固定路线的提前销售和市场排序 overlay，但不维持通用现金底线。新增的 `sell_before_unfunded_land()` 只修复一个可证明的资本顺序错误：不新卖货，只把同帧已有且保守估值后足够买地的 SELL 移到 BUY_LAND 前。
- 接管后 R1 的规划预算会扣除 `reserve=120`、饲料覆盖和 joint seed reserve；但当前 `feed_finance=0`，`settle_market(preparing)` 在换天采购步会直接返回，因此代码中“出售库存为整组已承诺采购融资”的分支实际上没有运行。当前额外执行兜底只有 `working_capital_gate`，而它只处理饲料缺口，不保障种子、动物、雇工或土地。已有 fresh32 × 7 对手 × 双座轨迹在 step 288 后没有记录到采购未成交，所以这是一个结构性薄弱点，尚不是已观测到的线上故障；不能据此直接打开 `feed_finance`。

## 前缀市场失败的剩余风险

已有 fresh 32-seed 七强追踪显示，当前常见失败不应继续打补丁：

- step 20/24 以及 step 63–65 的种子/雇工失败几乎全局普遍，不区分胜负；`BUY_PRODUCT WHEAT` 的零现金缺口失败是仓容量截断。
- G195 step 180 的买牛失败会在 step 194 卖奶、step 199 以 `[SELL MILK, BUY_ANIMAL COW]` 重新购入，step 200 拾取成功；再加 retry 会过量买牛。
- G024/G316 step 219 的第三块地失败会在 step 220 继续卖羊毛、step 221 再买地并成功；step 221 在已有三块地时的失败是第四块地尝试，超出当前 R1 `max_land=3` 与接管要求。

因此前期的当前判定是：**已有一个由市场队列因果顺序直接推出、且不会自愈的 bug 被修复；其余已检查失败均符合容量约束或路线自愈语义**。后续是否修改不以“与败局显著关联”为门槛，而以是否能先证明它违反规则、资金依赖或部署不变量为门槛；轨迹只用于验证触发范围和收益/副作用。

## 新公开对手域外检查（不计入正式七强验收）

从 Kaggle 新公开 notebook 下载的 MetaV4 后继在 96 个互斥 seed、同 seed 双座 FastEnv 探索中：
Ahmed V56 `145/192`（75.5%）、Master Hybrid 2965 `144/192`（75.0%）、Thomas v13
`150/192`（78.1%），共 576 局无异常。三者逐局 margin 相关系数为 `0.91–0.95`，68/96 个
seed 对三者全胜、17/96 个全败；这更像同一公开 chassis 下的共同环境脆弱性，而不是需要身份门控的
三个独立漏洞。

“终局第三块 SW 恰好一只动物”的指纹没有在已下载脚本中复现：上述三者以及 Fieldcraft、
Market-Smart 的 SW 均为 0 只，Night Harvest 稳定为 2 只羊；MetaV4 后继偶尔购买第四块 SE，布局
是 6 只羊。当前不能把该指纹归因给这些公开版本。

### `candidate_extra` 不能作为纯性能冗余删除

正式七强、全新 32 seed、双座 FastEnv 配对中，关闭 `candidate_extra` 后总胜 `400→398`、平均
margin 变化 `-163`；448 个 seed-对手配对中有 196 个 margin 改变，只节省约 `0.85%` 决策时间。
debug 轨迹也显示 liquid 候选被选 60/864 次。由此否定“额外三个候选从不生效，所以可免费删掉”的
假设，保持 `candidate_extra=1`；该试验不作为正式胜率更新。

## 已确认的产物一致性问题

`agent/route_policy.json` 的 G275/cp144 **树体**已是 15 节点 depth-3 真实对手树，运行时行为正确；
但同一节点外层仍保留旧自对弈树的 `samples=62720`、`depth=4`、CV/稳定性指标以及根级 `sources`。
这些字段不被线上控制器读取，却会误导复现和审计。当前只能把 README 中的 896 单元与独立配对 A/B
作为新树证据，不能把 JSON 内旧 CV 指标归给新树；后续导出树时应原子替换树体与 provenance。

## 近期优先级

1. 保持现有单次结局路线标签，不扩大 warm-R1 训练。
2. 下次路线树训练先修 cp168 survivor distribution，并添加候选兼容性特征；仍保持浅树。
3. 下次路线库重建时修聚类距离，而不是增加路线数量。
4. R1 研究只做统一价值误差归因；不按对手身份分支，不再扫已封死的参数和模型类别。
5. 所有准入以互斥 seed 段、完整配对、seed-clustered 区间为准；错误局不能从分母静默删除。

## 2026-09-22：从参数 A/B 转为模型一致性审计

后续研究不再把“单独改正某一项后端到端胜率下降”解释为该项不重要。现有参数可能是在补偿其他模型
错误；局部修正跨过离散 `argmax` 后还会改变整条轨迹。理论正确性、当前参数下的组合性能、最终部署
性能必须分开报告。具体纪律已写入根目录 `AGENTS.md` 的“模型研究方法论”。

### `PublicFlowScenario` 的状态闭合错误

设 `F(o_t)=r_{t:T}` 是从当前公开盘面生成的对手供给预测。一个自洽场景只能采用以下二者之一：

1. **open-loop**：在场景起点生成一次 `r`，场景内始终沿用尚未兑现的 `r_{t+1:T}`；
2. **closed-loop**：下一日重新计算 `F(o_{t+1})`，但 `o_{t+1}` 必须包含与当日已兑现供给一致的
   对手收获、持有物和资产转移。

当前实现把二者混合：`PublicFlowScenario` 把 `r_t` 直接加入对手 shed 并出售，却不给对手执行产生该
供给的收获/收集动作；随后 tail 从这张没有同步变化的对手棋盘重新调用 `public_rival()`。因此场景的
市场现金状态和资产状态不可能来自同一条状态轨迹。有限作物重复预测只是这一错误的一种表现；重新
计算补种和作物年龄也会产生正负皆有的整条供给漂移。

对候选选择有意义的不是单个候选的绝对误差，而是候选差分：

`Δ_ab = [V_a(F(stale))-V_a(r_future)] - [V_b(F(stale))-V_b(r_future)]`。

其一阶量级由错误供给 `δr`、价格曲线局部斜率和候选供给差 `q_a-q_b` 共同决定；候选公共项会抵消。
所以必须同时报告绝对 tail 偏差、同状态候选间偏差范围、score gap 与排序翻转。

内部诊断只把既有 warm 前缀跑到 step 288 后停止，不运行 suffix。32 个 Thomas/Melon handoff 状态
（8 seed、双座，实际约 16 个独立状态）、126 个正常 proposal 得到：

- 固定同一个 tail portfolio，仅把 stale 重算供给替换为起点剩余供给，tail 修正量范围
  `−3,091..+11,476`，中位数 `−1,710`；绝对状态误差很大。
- 同一状态内候选修正量的范围中位数 `63`、P90 `901`、最大 `987`；旧 top-2 score gap 中位数
  `538`、P10 `90`。这证明误差具备跨越部分决策边界的量级，但该批 handoff 状态尚未实际换 winner。
- 用起点剩余供给重新规划 tail 后，126 个 proposal 中 108 个 `proposal_key` 改变；候选修正范围中位数
  `303`、P90 `1,100`、最大 `1,300`，但该批状态的顶层 winner 仍未改变。
- 未来供给不一致集中在 WHEAT、STRAWBERRY，并在部分状态出现 CARROT `+131.1` 和
  FERTILIZER `+17.5`。这否定了“只有当日成熟作物重复一次”的过窄解释。

这些结果验证了状态闭合 bug 和它对内部计划的实质影响，但不使用终局胜率裁定理论真假。下一步应让
场景评分明确采用 open-loop 剩余供给；随后重新校准依赖该错误分数的 proposal selector，而不是保留
错误以维持旧参数的偶然补偿。正式二进制在完成内部 parity 与下游校准前不替换。

隔离构建 `work/agent-scenario-carry-rival.so` 已实现上述 open-loop 语义。它不从 stale board 重建
场景内的对手供给，而把起点 `r_future` 传给 rollout 内的跨日重规划和 terminal tail。在两个真实
handoff checkpoint、9 个 proposal 上，四个内部验收量全部精确归零：固定组合 tail 差、重新规划
tail 差、未来供给 mismatch、replan `proposal_key` 差均为 0。对应产物为
`work/scenario-carry-rival-invariant-2states.json`。这证明实现已经与选定的 open-loop 理论语义一致；
它尚未替换正式 `agent.so`，因为依赖旧错误分数校准的下游 selector/参数仍需按内部目标重新审计。

### 候选 score 的分量与时间拼接审计

`Planner::value()` 已增加只读 `ValueBreakdown`，逐项记录折现后的固定支出、自身交易、工资、动作成本、
对手收入惩罚和流动性惩罚；默认规划行为不变。最小测试验证六项之和与原返回值误差小于 `1e-9`。
在上述 32 个 handoff 状态、126 个正常 proposal 上，`score = rollout_objective + tail` 的最大恒等式误差
为 `3.64e-11`。

当前 winner 相对 runner-up 的 score gap 中位数为 `538`，但组成并不稳定：一日我方现金差中位数
`-80`，tail 自身交易差中位数 `+240`，tail 对手收入惩罚差中位数 `+348`；按绝对值最大的分量计数，
32 个状态中一日我方现金主导 9 个、tail 自身交易 11 个、tail 对手项 10 个、工资 2 个。流动性项全部
为 0。故不能再把 handoff 排序概括为单一“作物尾值”或单一“竞争项”；必须看候选差分。

把**同一批固定候选**仅从 `competition=2` 重评分为线性 margin 的 `competition=1`，32 个名义状态有
8 个 winner 改变，其中 seat 重复和数值近 tie 占一部分；这说明 `competition` 是能跨过决策边界的
补偿参数，但不构成直接把它改成 1 的部署建议。重新生成候选、修复场景闭合和重校准仍须分开。

更上游的确定性错误是时间拼接。当前实现计算
`C_{t+1}^{own}-cC_{t+1}^{rival}+V_{t+1}`，而 `V_{t+1}` 在新时点把折现指数重置为 0，并按候选后的
现金重新设折现率；它不是从 `t` 出发的同一个 `V_t` 递推。诊断新增“起点日期与现金冻结”的 tail
重估，不重规划、不跑 suffix。首批 4 个独立前缀状态中，当前各候选的绝对修正约 `+191..+457`；
同状态候选修正范围为 `5、156、7、237`，其中一个状态跨过原 top-2 gap `90` 并更换 winner。
这已经证明时间不一致不仅是绝对常数，能够改变排序。产物：
`work/scenario-score-parts-fast-32states.json`、`work/scenario-bellman-audit-4states.json`。

### 理论审计新增的结构项

- `gain/cost^alpha` 加静态 budget 不求解明确的现金约束组合最优问题。应在固定 `V` 下比较 greedy 与
  小规模 exact/beam 上界，报告 surrogate regret；不能把价值误差和求解器误差混在一起。
- 静态 budget 不是逐 tick 现金路径。preview/repair 删除项目后没有回填重优化；应比较计划流与首日
  实际可执行 witness、现金前缀和被删价值。
- 普通 proposal 与 joint bundle 采用嵌套 argmax，且 joint 内先用另一 proxy 筛掉候选；正确内部量是
  全部 `proposal × bundle` 在统一 horizon 下的 regret。当前触发覆盖可能低，先测覆盖再排优先级。
- `proposal_key` 只覆盖当前 target/queue 的一部分，不是完整行为等价键；可能把 Settings、commitment、
  sale memory 不同的方案错误去重。应对 key collision 跑固定 24-step 公共场景并逐动作比较。
- `PublicFlowScenario` 还需核对 requested flow、整数化 flow 与实际成交。负 WHEAT/FERTILIZER 会变成
  受现金和 shed 容量约束的购买，且不执行后续消耗；正流则先注入 shed 再出售。该守恒账应先于 suffix。

### score 全链时间一致性与场景兑现账（4 个独立 handoff 前缀）

只给既有 tail portfolio 换折现基准仍不完整，因为 tail portfolio 本身也是在错误的“边界重新起钟”
目标下生成的。隔离实现现在同时做到：rollout 内逐日现金增量按起点折现、tail 规划全过程沿用起点日期和
起点现金的折现核、continuation 从正确指数接续。它仍是确定性 surrogate，不声称已经等于胜率目标。

Thomas/Melon × 2 seeds × seat0 的 4 个独立 step288 状态中，旧 score 到完整时间一致 score 的候选间
修正范围为 `107 / 157 / 7 / 237`；第二个状态从 proposal 1 换为 proposal 0，旧 top-2 gap 为 `90`。
因此错误不只是绝对常数，也不只是“同一组合少乘一天折现”：错误的时间核还会改变 tail 重规划。
隔离宏 `R2_SCENARIO_FIXED_VALUE_CLOCK=1` 已把同一语义接入 `score()` 和 joint `event_value()`：rollout
逐日现金增量与 tail 共用起点折现核。在上述换序状态，默认构建导出的 diagnostic consistent score 与
隔离修复构建实际返回的两个候选 score 逐项完全相同（最大误差 0，winner 均为 proposal 0），产物为
`work/bellman-fixed-implementation-parity.json`。默认宏仍为 0，尚未替换正式策略。

同一批状态的 `PublicFlowScenario` 兑现账得到：正常候选的 requested flow 相同，非零项为
`WHEAT=-0.7621, MILK=7.65, WOOL=6.8, FERTILIZER=14.45`；整数化/实际成交均为
`[-1, 8, 7, 14]`。整数化绝对 L1 偏差为 `1.2379` 单位，cash/capacity short-fill 为 0，现金守恒
残差为 0，价格地板未触发。虚拟对手 shed 最终残留 1 WHEAT，因为负流被实现为购买却没有执行 FEED；
open-loop tail 不读取该 private shed，所以它证明场景不是合法资产轨迹，但在这四个 h1 状态没有继续
传播成 fill 差分。

同一个 rounded rival flow 因我方候选改变共享报价，对手现金的候选 range 分别为 `$3/$15/$3/$60`；
乘 `competition=2` 后最大交互尺度 `$120`，只有旧 gap=90 的敏感状态达到同量级。该项是共享价格交互，
不能误写成 requested→actual short-fill。合成边界测试另已验证：大额负 WHEAT 会受现金/棚容截断；
在 `$1` 地板上 SELL 全部成交拿钱，但只有到达地板前的单位推进市场库存。

产物：`work/model-internal-audit-4states.json`、`work/public-flow-audit-4states.json`；可运行检查为
`policy/r1/test_public_flow_audit.cpp`。均未运行 suffix 或完整对局。

### greedy 局部内部 regret（同一 4 状态）

对每个 free slot 同时记录当前 `gain/cost^alpha` 选择与 raw `gain` 最大项。在两个 seed0 状态里，
proposal 0/5 各有 2 个 slot 分歧；单格 raw-gain regret 最大约 `74/56`，各状态跨全部正常 proposal
求和约 `146/106`。自动 winner proposal 1 没有发生该分歧，另外两个状态全部为 0；preview 删除项目
也全部为 0。

这些状态的静态 budget 也完全不紧：初始约 `$12,973–13,447`，所有正常计划结束仍至少剩约
`$12,174`。所以已观测分歧不是预算约束迫使的取舍，而是 `cost^alpha` 在松弛预算下主动牺牲 raw
gain；但静态 budget 是否违反逐 tick 现金可行域，在这四个状态没有触发证据。

这只是不改变后续选择的局部反例，不是全局组合 regret 上界。扩展到新的 16 seed、Thomas/Melon、
只取 seat0 的 32 个独立 handoff 状态后，默认 124 个 proposal 共审计 456 个 slot，其中 75 个
`gain/cost^alpha` 选择与最大边际 `model.value` 不同。只看每个状态最终胜出的 proposal，仍有
`9/32` 状态发生分歧；局部 value regret 合计约 `$590`、最大 `$249`。原 top-2 gap 中位数约
`$494`、P10 约 `$65`、最小 `$44`，有 3 个状态的局部 regret 和已经超过当时 gap。它仍不是可同时
兑现的组合改进，但量级足以否定“optimizer 误差必然小于候选间隔”。

现有单换邻域只让 2/32 个状态的最终最高 score 改善，合计 `$367`、最大 `$336`，没有改变 proposal
winner；受限双换在 32 个状态均没有改善最终最高 score。大量 swap gain 发生在本来不会胜出的 proposal，
所以不能把跨全部 proposal 的 gain 相加后当作决策收益。

隔离宏 `R2_GREEDY_VALUE_ORDER=1` 进一步保持候选生成、预算、执行链和外层评分不变，只把 slot 排序改为
边际 `model.value`。同一 32 状态中，内部 winner 的逐 slot value regret 从约 `$590` 降为 0；外层最高
score 有 14 个状态上升、5 个下降、13 个不变，总修正 `+3,047`，单状态范围 `−236..+762`，并有
4/32 个 proposal winner 换序。它证明 ratio/bias/rent 排序对决策是实质变量；同时，“planner value
逐步更优却使外层 score 下降”也再次证明内层 `model.value` 与外层 rollout+tail 不是同一个目标。
该宏默认 0，只作诊断，不进入正式策略。

扩样本也推翻了“当前只有 2–3 个新 slot、最多 `9^3=729` 叶”的早期估计：敏感 winner 实际审计到
5–7 个 slot，朴素枚举最高为 `9^7`。因此不为旧 surrogate 硬写 exact DFS；先完成场景闭合和 Bellman
目标统一，再在同一目标下用有界 beam/branch-and-bound 报告 solver regret。产物：
`work/greedy-audit-independent-32states.json`、`work/greedy-audit-independent-32states-summary.json`、
`work/greedy-value-order-32states.json`、`work/greedy-value-order-comparison-32states.json`。

### 上游语义修正后仍存在的控制时域错配

为避免拿一个已知矛盾的 score 裁定 optimizer，另做了组合隔离构建，同时启用：

- `R2_SCENARIO_CARRY_RIVAL=1`：场景内沿用起点 open-loop 对手供给；
- `R2_SCENARIO_FIXED_VALUE_CLOCK=1`：rollout 与 tail 共用起点折现核；
- `R2_MARKET_INTEGRAL=2`：规划估值使用逐单位精确市场 primitive。

在同一 32 个独立 handoff 状态上，旧 ratio 排序的 125 个 proposal 共 472 个 slot，104 个选择与最大
边际 `model.value` 不同；最终 winner 在 17/32 个状态有分歧，局部 value regret 合计约 `$1,136`、
最大 `$194`。此时 top-2 gap 中位数约 `$555`、P10 `$210`，只有一个状态的局部 regret 和超过 gap。

在同一组合语义上再启用 value-order 后，外层最高 score 仍是 13 个状态上升、10 个下降、9 个不变，
总修正 `+1,220`、范围 `−166..+536`，proposal winner 仅 1/32 换序。10 个负值中 5 个绝对值小于
`$10`，但另外 5 个不是数值噪声。因此状态闭合、市场和 Bellman 时钟修正后，内外目标仍未统一。

剩余根因是**控制时域错配**：`Controller::plan()` 用 `model.value` 把新增项目当作持续到终局的 portfolio
承诺；`SearchController::score()` 却只让 candidate settings 干预一天，第二天恢复 common base，并从
执行后的状态重新规划 tail。于是 proposal generator 优化“长期持有组合”，外层 evaluator 优化“一日
可执行干预 + 重新规划”。更精确地求解前一个目标不能保证提高后一个目标。

下一步不再扩大旧 portfolio swap，而是在固定 handoff 上对首日**实际可执行 bundle**做小宽度 beam：
叶节点统一调用修正后的外层 score，`model.value` 只作 beam heuristic；报告相对现有 proposal 的内部
regret、换序和运行成本。产物：`work/coherent-ratio-32states.json`、
`work/coherent-value-32states.json`、`work/coherent-value-order-comparison-32states.json`。

### 首日外层单-slot 邻域：控制时域错配是高覆盖结构误差

在上述组合修正语义上，诊断宏 `R2_OUTER_NEIGHBOR_AUDIT=1` 从每个状态当前 outer winner 出发：对它
实际经过的每个 greedy slot，逐一强制 8 种项目，完整重跑原生 `plan → preview/repair → 一日 rollout →
common-base tail`，并按 `proposal_key` 去重。它不把局部 gain 相加，也不使用简化执行模型。

32 个独立状态共得到 658 个可行单-slot 邻居：

- 25/32 状态至少有一个邻居优于默认 outer winner，正 score 修正合计约 `+2,461`，最大 `+506`；
- 与 default、`portfolio_swaps=1`、`portfolio_swaps=2` 三者中的最好结果相比，仍有 23/32 状态改善，
  正修正合计约 `+1,844`；改善中位数约 `+31.7`，16 个超过 `$10`、9 个超过 `$50`、5 个超过 `$100`；
- 既有 single-swap 在同一组合语义下只改善 2/32（合计 `+620`），double-swap 为 0。故问题不是再扩大
  旧 `model.value` 邻域，而是邻域选择目标本身错了；
- 25 个默认改善状态中，15 个最佳邻居的长期 `predicted` 反而下降，20 个强制项目不是该 slot 的
  `value_kind`。也就是说，长期 portfolio value 在大多数可改善状态里会主动排除 one-day continuation
  更优的动作，而不只是给这些动作一个略有噪声的排序。

最佳强制项集中在若干位置/品类，但 Thomas 与 Melon 同 seed 上经常给出近乎相同的修正；这支持公共状态
下的统一目标错配，不构成按对手身份或固定位置写 guard 的依据。下一步若扩展 depth-2，必须以 outer
score 选 beam，并报告相对一阶邻域的**增量** regret；不能用这些频繁模式直接做线上启发式。

同一单状态、同一 Python 前缀进程的墙钟为：组合修正基线 `12.38s`，加入 23 个 outer 邻居后
`14.10s`，增量约 `1.72s`。它作为首次 handoff 离线/低频搜索尚可，但若每天原样搜索会增加约数十秒；
正式算法需要复用 rollout/tail、并行安全评分或有可验证 regret 的截断，不能直接把诊断全枚举搬上线。

按真实 outer score 取前两个一阶分支，再分别枚举第二个 slot 的 width-2 beam，共评估 922 个去重二阶
叶。17/32 状态相对 `max(base, single)` 继续改善，新增 score 合计约 `+898`、最大 `+243`。将一阶和二阶
一起与既有 default/single-swap/double-swap 最好结果比较，25/32 状态改善，正修正合计约 `+2,740`、
中位数约 `+89`、最大 `+506`；其中 16 个状态最终由二阶方案胜出。它证明首日 bundle 存在真实组合
交互，不能只做独立单格修正。

搜索宽度已有最小饱和检查：前 8 个独立状态上，width 2 共评估 238 个二阶叶；width 4 增至 450 个，
但最佳 score 的新增改善为 0。因此当前停止加宽，不能把“更多叶”本身当进展。同 seed 单状态 width-2
墙钟约 `18.06s`，相对组合修正基线增加 `5.68s`；它仍只适合作为首次 handoff oracle/低频候选，不能
未经复用和截断直接进入每天重规划。

产物：`work/coherent-outer-neighbor-32states.json`、
`work/coherent-outer-neighbor-summary-32states.json`、
`work/coherent-outer-neighbor-vs-existing-32states.json`、
`work/coherent-outer-neighbor-mechanism-32states.json`、
`work/coherent-outer-beam2-summary-32states.json`、
`work/coherent-outer-beam2-vs-existing-32states.json`、
`work/coherent-outer-beam4-vs2-8states.json`。

### 固定 suffix 分解：首日转移基本兑现，主要漏损在 continuation/tail

为区分“首日动作模型错”与“后续价值模型错”，在 Thomas/Melon × 4 个互斥 seed × seat0 的 8 个真实
step288 handoff 上，baseline 与 outer-beam 从完全相同的 observation 出发，各自只强制安装一次候选，
随后沿原生执行链跑到终局。这里比较的是固定状态 suffix，不把完整胜率当理论裁判。

按与 score 相同的 `c2=own−2·rival` 口径分解候选差分：

- outer score 合计认为候选改善约 `+919`；
- 预测首日 `c2` 差分合计 `+5,615`，真实下一日状态中的差分为 `+4,881`，8/8 的符号一致；
- 因而模型隐含的 continuation 差分为 `919−5,615=−4,696`；真实终局 `c2` 差分为 `−6,932`，扣除
  真实首日后，continuation 为 `−11,813`；模型少计了约 `−7,117` 的后续损失；
- 排除两个零差分状态后，continuation 方向只匹配 `3/6`。真实终局 margin 差分合计 `−2,800`；真实
  首日 margin 合计 `+4,691`，其后的 margin continuation 合计 `−7,491`。

所以 width-2 找到的首日收益并非主要由第一天 transition 幻觉造成：预测收益大体进入了真实下一日状态，
但后续重规划和 tail 没有正确评价它留下的状态。当前优先级应从“继续扩大首日 beam”转为逐日核对：每次
重规划的 portfolio、市场/库存状态、预测 continuation 与真实剩余 suffix 在哪一天开始分叉。不能用
固定位置或对手身份 guard 掩盖这一结构误差。

边界：样本只有 8 个状态；强制 `install_candidate()` 在两臂都绕过了 `SearchController::choose()` 的
joint-bundle 后层，因此这是 portfolio 候选的隔离诊断，不是正式线上策略 A/B，也没有部署依据。原始产物：
`work/coherent-outer-suffix-transition-8states-seed100-103.json`。

### continuation 的重要缺失状态：对手跨日库存与候选相关成交时序

逐日和终局状态把上述 continuation 偏差继续分解：8 个固定状态中，outer 相对 baseline 的终局我方现金
合计其实为 `+1,332`，对手现金合计为 `+4,132`，所以 margin 才是 `−2,800`。8/8 状态里对手公开的
动物、作物、劳工和杂草终局计数均完全相同；主误差不是对手资产数量预测，而是相同资产在不同共享市场
路径上的成交数量、时序与价格反馈。

Thomas seed100 是可复现的高量级反例：

- 默认 coherent 模型下，outer 自己终局多赚 `$2,660`，但对手多赚 `$5,561`，margin `−2,901`；
- 对手非 flush 的 WOOL 卖单数量反而略少（约 200 vs 207），按下单前公开报价作粗略 mark-to-market，
  收入却约为 `$40,404` vs `$32,183`，差约 `$8,221`；这不是“多产了羊”的效果；
- 两臂 WOOL 市场直到 day24 完全相同，day25 后库存差依次扩大，day29 为 `−51`，公开价格差为 `+195`；
- 用历史成交小时密度的 `R2_SALE_CLOCK_MODE=2` 后，普通 proposal 在 8/8 状态仍不换 winner；它拒绝了
  原 6 个非零 outer 改动中的 5 个，却保留这个最差反例，并把内部 outer gain 从 `+89` 提高到 `+176`。
  同模型固定 suffix 中，我方只多 `$146`，对手多 `$5,659`，margin 进一步为 `−5,513`。

代码原因是 `PublicFlowScenario` 只把 `rival[d][i]` 当天在固定窗口成交，`Planner::value()` 也只用单一
early/late share；二者都没有隐藏库存状态 `Z_{t,i}`，所以不能表示“今天收获但以后卖”、终局清仓，或
候选改变价格后对手调整成交。`SaleClock` 的近三日平稳小时密度只能移动同一天订单，不能修复跨日状态。

公开信息并非完全不足。新增的只读区间账本按以下守恒更新：确定收获同时提高 lower/upper，不确定消失只
提高 upper，确认销售同时扣减两界，价格地板或换日不确定时 lower 归零。Thomas seed100 在 day24 的
WOOL 区间已经是 `[0,33]`，两臂相同；随后真实价格分叉证明该区间的候选敏感度上界足以跨越几百元的
score gap。实现位于 `public_trade_ledger.hpp`，默认不参与 score；最小检查为
`test_public_trade_ledger.cpp`，产物为 `work/rival-stock-interval-handoff-8states-seed100-103.json` 和
`work/coherent-saleclock2-outer-daily-stock-thomas-seed100.json`。

下一版不能重复旧 `R2_PENDING_STOCK_WEIGHT` 的语义——把整个 upper 一次性塞进“今天供给”。正确状态为
`Z_{t+1}=Z_t+H_t-S_t`，并对公开历史允许的 `Z_t` 与条件成交策略保留多情景；至少包含即时卖、跨日持有、
终局清仓，并让成交随候选产生的价格路径变化。候选比较必须报告这些情景下的 margin 分布/下界，而不是
再用一个固定权重把不确定库存压成点值。

### 终局闭环诊断：静态 tail 是当前更上游的误差

先用现有 `sale_plan_dp` 做了一个隔离的对手库存 best-response：保持同一条 forecast flow、同一市场
primitive 和同一日内顺序，只比较“预计产出当天卖完”与“连同公开库存 upper 在剩余日期内择时清仓”。
Thomas seed100 的 WOOL 修正对相关候选约为 `−6,994`（另一个空间分支约 `−7,079`），量级远大于
候选 gap，但候选 winner 不变。它说明跨日库存严重影响绝对价值；同时，在这组候选上它近似公共项，
单独加入库存点估计不能修复排序。该诊断不是对手策略模型，也不应直接进入生产 score。

随后不换模型，只把 `SearchController::score()` 的 rollout horizon 延长，并让同一控制器在模拟状态上
每天重新规划。对原始 width-2 outer 相对 normal 的 6 个非零反事实：

- 一日静态 score 的方向只对 `3/6`，聚合预测 `+919`，真实终局 margin 为 `−2,800`；
- 3/5/10 日闭环截断的方向分别为 `4/6`、`3/6`、`3/6`，仍不稳定；
- 到终局的闭环 score 方向对 `5/6`，聚合预测 `−2,053`，已经恢复真实总方向和主要量级。

这比“某个市场近似少算几十元”更上游：正式 score 当前求的是“一日动作 + 静态长期组合”，而线上真正
执行的是“一日动作后每天观察、重规划、再执行”的闭环策略。静态 portfolio tail 因而不是部署策略的
价值函数，且短 horizon 不能被假定为终局值的单调近似。当前主线应先建立可计算的闭环 continuation；
库存 belief 和条件清仓用于解释其剩余误差，而不是反过来用一个库存权重修补静态 tail。

top-4 诊断提供了一个重要的实验协议反例。按短 score 各自截 top-4 后再用终局闭环分重排，内部 winner
在 `7/8` 状态变化；固定 suffix 的 outer-baseline 合计为 `−1,939`，表面上比旧 `−2,800` 好。但两臂
候选集并不嵌套：baseline 会从四个 normal 中选出闭环 winner，outer 的“全体短分 top-4 + 短分最佳
normal”可能漏掉它。Thomas seed103 与 Melon seed103 因此出现 outer 选择的闭环分反而低于 baseline。
所以这次 suffix 只能说明“终局闭环重排会实质改变动作”，不能评价 outer search 是否改善；结果不得作为
部署或拒绝闭环目标的证据。下次必须把 baseline 闭环 winner 强制加入 outer 候选集，再比较嵌套集合。

协议修正后，baseline 使用短分前 4 个 normal，outer 使用完全相同的 4 个 normal 再加前 4 个 outer；
预测集合单调性在 `8/8` 状态成立。新的固定 suffix 给出两个不同层次的结果：

- 闭环 normal 重排相对旧 normal 的真实 margin 合计 `+9,774`；新的 outer winner 相对旧 normal 仍为
  `+9,344`。因此终局闭环 continuation 不只是解释工具，确实改善了这批状态的 proposal 选择；
- 但 extra-outer 相对新的闭环 normal 是 `−430`，而模型预测 `+7,970`。4 个非零 extra-outer 中只
  `2/4` 与真实同向；Thomas seed101/102 分别预测 `+364/+4,016`，实际为 `−5,697/−1,586`。

所以不能把“固定的原始候选方向 `5/6` 正确”外推成“对更大候选集取 argmax 就可靠”。搜索会主动选择
模型误差最大的候选，形成 optimizer's curse。当前证据链是：静态 tail 是第一层主误差；改成闭环后，
单条 open-loop rival flow 成为暴露出来的第二层主误差。下一步应在同一个闭环 scorer 中保留多个公开
可生成的对手库存/清仓响应，再重做嵌套候选集 regret，而不是扩大 top-K 或直接上线闭环重排。

产物：`work/coherent-rival-stock-audit-wool-thomas-seed100.json`、
`work/coherent-outer-longhorizon-8states-seed100-103.json`、
`work/coherent-outer-closed-loop-top4-8states-seed100-103.json`、
`work/coherent-outer-closed-loop-top4-suffix-8states-seed100-103.json`、
`work/coherent-outer-closed-loop-nested-top4-suffix-8states-seed100-103.json`。以上均为诊断构建，正式
决策语义未改。

所有观测代码均有编译期开关：`R2_FLOW_AUDIT`、`R2_OPTIMIZER_AUDIT` 与
`R2_GREEDY_VALUE_ORDER`、`R2_OUTER_NEIGHBOR_AUDIT` 默认 0，正式构建不承担逐订单记账、逐 slot 保存、
强制邻域搜索或新排序语义。这些宏默认关闭后的重建审计二进制，在同一真实 handoff 的三种 swap 模式下
与改动前候选 JSON 逐项相同；
此前默认关闭的生产语义重建二进制与当前策略比较到 step300 也为 0 mismatch。

### 2026-09-23：正式 h1 的控制接缝与 WOOL 外部性隔离

需收紧上文“终局闭环”的定义：`SearchController::score(requested_horizon=30-day)` 在模拟的后续日期调用的是
单个 `Controller::act()`，会逐日重新规划，但**不会逐日调用线上 `SearchController::choose()`**，因此
它是 base-Controller 闭环，不是完整部署策略的 continuation。先前 `5/6` 方向和嵌套 top-4 的 fixed-suffix
结果仍是各自实验口径的事实，不能解释为已经修好这一策略接缝。

在 Thomas seed `2609800100`、seat0、step288 的同一 handoff，正式一日 score 对 normal id1−id0
为 `+$593.87`，选 id1。强制两候选各执行当天 24 步、仅到 step312 后，以实际公开状态重新计算线上
normal 候选：id0 前驱的最佳 day13 分为 `−48,720.22`（选 id1），id1 前驱为 `−60,476.20`
（选 id4）；两步的**模型内部**排序因而改为偏向最初 id0 约 `$11,755.99`。其中 id0 前驱在 day13
由 base 到最佳候选有 `+$11,651.68` 的选择机会，id1 前驱仅 `+$0.25`。这证明正式 h1 的静态 tail
不能充当线上 next-day 选择的 Bellman 值；它**不证明**真实终局 margin 同方向或同量级，也不保证
其他状态换序。day12 首日模型预测的候选间 `own_cash−2*rival_cash` 差为 `+$1,110`，真实为
`+$1,039`；九种产品的候选间库存差逐项吻合，故本例的巨大内部换序不是首日市场转移误差造成。
作为覆盖率反例，同法检查独立 Thomas seed `2609800102`，day12 id1−id0 为 `+$1,152.91`，
day13 normal max Q 仍为 `+$1,222.94`，未换序。当前只有触发机制与两个状态的量级，不能估计
总体换序频率。
在换序例里，day13 两臂资产总数同为 60 作物/15 动物，但 id0 臂有 3 株 MELON、0 株 TOMATO、
shed WOOL12；id1 臂为 1 株 MELON、2 株 TOMATO、shed WOOL8，WOOL 市场库存也高 4。
只有 id0 臂的某个 day13 normal 候选打开后续增羊方案。现阶段不能把选项价值归因于某一株作物、
现金、市场或 commitment 单独变量；结论是完整状态上的可选后继集合不同。

进一步隔离 day13 同一状态 normal id1−id0 的 `+$11,651.68` 分差：首日 `own−2*rival`
为 `+$757`（实际执行 `+$752`），静态 tail 为 `+$10,894.68`。tail 中对手惩罚项改善
`+$14,754.27`，自有交易减少 `$2,453.48`，其余固定/工资/动作合计减少约 `$1,406`。
按产品看，WOOL 未来自有供给从 96 增到 132，模型给 WOOL 对手收入惩罚的差为
`+$14,112.40`；即每新增 WOOL 的固定供给路径外部性约 `$392`，不是本商品一次售价。
这种长时域外部性来自 `−competition × Σ_t rival_sales_t × price(market_inventory_t)`，
候选自产对后续库存与价格的冲击反复作用于对手销量。它是正式 `competition=2` 与固定对手流下的
主动模型假设，不能直接当成真实收益。固定候选把系数代数改为 1 后本例仍约 `+$4,274`，所以
也不能把问题简化为“只因权重为 2”。

同状态强制 day13 id0/id1 到 step336，首日候选间 `own−2*rival` 预测 `+$757`、真实
`+$752`，市场库存候选差仍逐项吻合；id1 臂 day14 在线首个动作确实买 2 只 SHEEP，全部 normal
候选的后续 WOOL 流保持 132，id0 臂保持 96。这证明增羊方向至少经过一次次日重规划并产生在线订单，
并非 tail 中当场消失的虚构计划；下一步市场成交记录显示两只羊全部买入、进入 shed。
继续只走到 day15，羊从 5 只增至 7 只、shed 羊归零，落地亦已确认；
护理和最终 WOOL 产量仍未确认。另一方面，day14 的预测仍是同一固定对手供给下的模型值，未验证
对手跨日库存、条件清仓与后续投资/生产响应。单独允许 WOOL 用公开库存上界跨日最优清仓，id0/id1
分别扣 `−$7,563.78/−$6,960.55`，候选差反而增加约 `$603`，不换序。该日公开 WOOL 隐库存
区间实际为 `[0,0]`，所以这里的差来自**未来预测产出**的择时清仓，而非已有隐藏库存；它只否定
“固定一条未来流、仅改卖出日期就能修复排序”，不能代表真实对手条件生产/销售响应。
固定起点折现时钟使该 day13 tail 差只缩小约 `$121`，
沿用起点对手供给只缩小约 `$20`，因此此例的主导信号是固定流下的 WOOL 价格冲击，不是这两项 seam。
同一 tail portfolio、同一对手流与需求下，仅把估值中的批量平均价换成 `ConditionalMarket::execute`
逐单位市场 primitive（整数销量遵守官方地板规则；小数预测流仍为分段线性松弛），
id0/id1 tail 分别从 `−$36,848.89/−$25,954.22` 变成 `−$39,317.13/−$29,149.29`；
候选差由 `+$10,894.68` 缩至 `+$10,167.85`，少约 `$727`，仍不换序。其中 WOOL 贡献约
`−$639` 的差分修正。这个 fixed-portfolio 内部对拍既否定“价格地板误差能解释整笔 +$10.9k”，
也否定“地板误差必然只有几十元”；它不包含因精确估值而重新生成候选/portfolio 的反馈。

正确目标应区分：精确终局效用（胜负、margin）、公开 belief 条件下的对手状态转移、线上策略的
`V^{SearchController}`，以及当前经验代理 `own−2*rival`。形式上候选比较需要
对 margin 用 `Q_t(c)=m_t+E[Δm_t(c)+V^{SearchController}_{t+1}(s_{t+1})\mid public_history,c]`，
其中 `m_t` 是当前钱差、`Δm_t` 是当日钱差增量、`V` 只包含以后增量，避免重复计现金；若目标是胜率，
则必须直接比较 `P(m_T>0\mid public_history,c,SearchController)`，不能把钱差的可加形式当作胜率。
目前的静态 tail
把 `V` 换成一次静态 portfolio 估值，base-Controller 终局 rollout 也仍缺线上 `choose()`。
这里的 `s_{t+1}` 必须是公开观测 **加** 已由真实动作序列推进的控制器内部状态
（`book/joint/restore_day`、SaleClock/公开账本及候选安装后的执行状态），不是仅凭下一日棋盘
重建的公开快照；否则同一盘面下不同承诺被错误合并，仍不满足 Bellman 状态充分性。
完整线上动作还包含 normal winner 之后的 `joint_candidates` 二层覆写；其 `event_value()`
同样用有限期 base-Controller rollout 加静态 tail。若只替换 normal `score()` 而不统一二层
比较，最终部署策略仍有两个互相不同的 continuation 目标。
此外，二层只从**已经选中的一个** normal winner 生成 joint bundle，`event_value()` 再用
`age+2` 日 horizon 比 keep/exchange；即使两层各自评价准确，其他 normal 的 joint 分支也未
进入同一个 `argmax`。正确候选比较要以共同起点、共同对手/商店情景、同一线上后继策略的 Q
在嵌套集合上验收，不能拿 normal h1 分数直接与 joint event 分数相减。
joint 候选生成内部还先用旧 `keep.model.value` 在 feeder 年龄 `2..4` × successor 作物 5 种中
只留一个 best；`event_value` 无法重新找回被此预筛丢弃的组合。因此即使只统一最终 scorer，
还需核对生成覆盖，不能把 final rescore 误当作全部组合优化。
当前覆盖率只作小范围筛选：Thomas seed `2609800100..103`、seat0，从部署入口真实 warm 链
进入 day12，含 handoff selector 和每日正式 `SearchController::choose()`，到 day20 即截断，
共 32 次日决策。joint 搜索 5 次（day12 三次、day16/day19 各一次），共 7 个子候选；
仅 2 个通过 feasibility gate，exchange−keep 内部分数为 `−$3,730/−$948`；其余 5 个
虽然也有诊断分数，但被拒绝，不能当作可执行反事实。0 次覆写。因此这些前缀里二层不改变动作；
不能推论更晚日期或其他公开状态的覆写率，
更不是终局性能证据。产物：`experiments/audit_r1_joint_prefix.py`、
`work/r1-joint-online-prefix-thomas-seed100-103.json`。此前强制 day12 id0 的 3/28 搜索也保留
在 `work/r1-joint-day13-20-thomas-seed*.json`，只作反事实，不作为部署覆盖率。
下一步只做隔离的同状态、多候选 next-day `normal max Q` 和对手条件响应支持诊断；候选集、预测流与
真实 suffix 要分别归因，不能据一个内部换序改生产 R1。

隔离产物：`experiments/audit_r1_first_day_transition.py`、
`work/r1-day13-asset-kinds-thomas-seed100.json`、`work/r1-day13-tail-items-thomas-seed100.json`、
`work/r1-day13-wool-stock-thomas-seed100.json`、
`work/r1-day13-exact-tail-thomas-seed100.json`、
`work/r1-day13-two-arm-thomas-seed102.json`、
`work/r1-day14-tail-items-after-day13-id0.json`、`work/r1-day14-tail-items-after-day13-id1.json`。
`R2_SHOP_BRANCH_AUDIT` 与 `R2_RIVAL_STOCK_AUDIT_ITEM` 默认关闭，未改正式策略、开局或训练入口。

### 2026-09-23：作物收获日子目标与整盘目标不一致

`triad::Controller::scalar()` 给现有有限作物选择收获日、给轮作等子问题计价时，使用固定的商品路径价
`px[d][i]`、线性 `work_price × labor[d]` 与单独的 `discount`。正式配置 `work_price=4`、
`labor_hours=10`；而最终 `Planner::value()` 对同一组合用候选供给引起的共享市场价格路径，
把对手成交收入乘 `competition=2`，再扣 `wages(total_labor[d])` 的非线性每日雇工成本近似、
`action_cost=4 × total_labor[d]` 及现金流动性罚。这不是同一目标的分解。
在约 130h/天的预测负荷处，雇工成本函数 `wages(L)` 的边际约 `$18/h`，加动作成本约 `$22/h`，
而 `scalar` 的线性工时价只有 `$4/h`；更关键的是跨日移动工时能降低拥挤日工资，即使总工时上升。

理论上，固定其余组合 `A`，子方案 `a` 与 `b` 应比较
`F(A+a)−F(A+b)`，其中 `F=Planner::value`，同一现金时钟、市场成交 primitive 和对手情景。
其边际由自有成交、对手成交价格外部性、逐日工资差、动作成本、固定现金与流动性罚共同组成；
固定 `px` 加线性劳动只是一阶近似，不是这个差分。若要廉价近似，应从同一 `F` 导出逐日商品与
工时边际影子价；WOOL 等接近价格地板处一阶近似可能失效，应对少量可行收获日直接重算 `F`。

只读内部诊断在 day13 Thomas seed100 的五个 normal portfolio 中，各有 22–25 株可审有限作物；
逐株用相同观测、相同其他资产把原作物路径替换为生物学合法的收获日，原选中路径重建的
`model.value` 偏差全部为 0。每个组合有 9–11 株存在更高的同口径 `F`，单株最大改进 `$29–$92`。
例如 id0 的 WHEAT pos90 从 day15 改 day13：`scalar` 差 `−$68.23`，完整 `F` 差
`+$91.70`，分项为自有成交 `−$100.68`、对手惩罚 `+$110.26`、工资 `+$65.53`、动作
`+$16.59`。id1 的 MELON pos1 从 day13 改 day14：`scalar` 差 `−$22.59`，完整 `F`
差 `+$74.87`，其中工资差 `+$77.57`，虽然全期总劳动反而多 1h，表明逐日拥挤才是关键。
各单株改进的算术和 `$220–$776` 不是可同时兑现的组合收益：替换会改变互相的市场/工资边际，
诊断也未重排工人路线与资源日历。这是严格的内外目标不一致证据与局部量级，不是端到端性能结论。

修正顺序应为：先定义外层 `score()` 的统一 Bellman continuation 与对手情景，再让已证实错位的
harvest-age 子问题对同一目标至少达到受约束的局部最优，并用 executor 可行日历过滤替换；
maintenance 子 DP 虽也使用固定影子价，但已有 `service_reconcile=2` 以整盘值复核生成的路径，
仍需单独审计其备选覆盖，不能把收获日证据直接套用；
最后重校 `work_price`、`labor_hours`、`competition` 等可能在旧误差上互相补偿的经验参数。
不能为了保留旧胜率把已确认的子目标差继续当作精确最优。产物：
`work/r1-day13-harvest-age-parts-thomas-seed100.json`；审计函数仅在 `R2_SHOP_BRANCH_AUDIT` 构建出现。

### 2026-09-23：原生逐格贪心的 STOP 条件缺少支配证明

正式非 NN `Controller::plan()` 按距仓库的顺序扫描 `free`，在某一格找不到正 `rank` 时执行
`if(bestkind<0)break`，终止整个后续格序列。这个剪枝只有在后续格的可行动作集与边际收益都被当前格
单调支配时才成立；源码没有建立该性质。`free` 混有当天可收获的已有作物格、空格及其他可清理格，
在作物格上动物候选被禁止，后续空格却允许动物；前缀资产改变市场与工资边际，位置距离也不保证
综合收益单调。因此“当前格无正项目”一般不推出“后续无正项目”。`portfolio_swaps` 只访问已经新建的
`new_positions`，不能补回这种截断遗漏。该结论是**非 NN 原生贪心的搜索完备性问题**，不是 student v3
执行器问题：v3 的 `NONE_OR_KEEP` 会 `continue` 越过此格，只有显式 `STOP` 才 `break`。

诊断分三层：① 后续格按相同现金、库存、动物上限与原 `F` 有正边际；② 原 `rank` 也为正，若访问确会
进入计划；③ executor 资源日历可兑现。①②仍不是实际利润，③需要单独验证。仅在
`R2_OPTIMIZER_AUDIT=1` 下记录停止位置、被跳过的原生空格数，并对前三个空格计算同一前缀的最好
`rank` 分子；正式构建宏为 0，生产策略没有改变。是否在当前 handoff 触发及候选差分量级，待内部诊断。

小样本内部结果：Thomas seed `2609800100..107` 的 8 个 step288 handoff、33 个 normal 候选均未
早停；seed `2609800100` 从 normal id0 实际只走一天到 step312 后，5 个 normal 候选都在某格
早停，但后面没有**原生 EMPTY** 格（5/5 的 `stop_skipped_empty=0`），因此这两个检查点并未证明
被截断的正收益项目，更不能把该机制归因于前述 day13 大额候选换序。后续仍可能有 WEED/建筑格或
可收获作物格，当前诊断只量化原生 EMPTY，不宜说“所有后续可行格都为零”。完整触发率未测。
产物：`work/r1-greedy-stop-thomas-seed100-107.json`、
`work/r1-greedy-stop-day13-thomas-seed100.json`；诊断二进制 `work/agent-r1-stop-tail-audit.so`。

更一般地，纵使所有资产流精确，整盘 `F` 同时含共享市场价格、凸雇工/容量、饲料—动物互补与
整数资本约束，不具备逐格 `gain/cost^0.4` 贪心所需的统一交换/支配性质；单次扫描不构成
近最优性保证。应先统一真实 continuation 与可执行性，再用嵌套候选集检验小范围组合搜索；
在错误代理上加深搜索只会更善于利用代理漏洞。
现有 32 handoff 的 normal 去重候选仅 `2..7` 个、平均 `3.875` 个；统一 Q 只能保证在实际
生成的候选集内排序一致，不能由此宣称全局近最优。扩邻域时应沿用“outer 包含 baseline
winner”的嵌套候选协议，先证明新增可执行候选的统一 Q 值与排序贡献。

### 2026-09-23：连续工时代理不能冒充实际雇工现金流

正式 `Planner::value()` 对每天 `a.labor[d]` 扣 `wages(L)`，即以 `L/labor_hours−1` 代入 Fibonacci
总雇工价的连续近似；`labor_hours=10` 是经验换算，不是官方规则。官方只对实际执行且成交的每个
`HIRE` 按当天已有雇工数收 Fibonacci 单价；场上工人数、市场订单带宽、路线可执行性和现金均是
离散约束。故统一目标中应把 `wages(L)` 明确标成**工作量/容量代理**，不能同时声称它是终局现金
的精确预测；若作为尾值成本，还须与 scenario 首日实际雇工现金和子 DP 的影子工价校准到同一尺度。

内部对拍：Thomas seed `2609800100`、day13 同一实际状态，normal id0/id1 的第一日预测劳动分别
`158.40/154.18`，`wages(L)` 分别约 `$1,477.2/$1,205.5`，模型仅工资项给 id1 约
`+$271.7` 候选优势。两候选 executor 均排 11 次 `HIRE`；各强制执行到 day14 的当日峰值
`hires_today=11`，官方同为 `1+1+2+3+5+8+13+21+34+55+89=$232`，**实际雇工差为 0**。
绝对高估分别约 `$1,245/$974`，候选间错误差约 `$272`；前者不能直接代表排序影响。
而正式 `max_hands=14` 时一天最多付 14 次雇工、总价 `$986`；这里的 `$1,205–1,477`
甚至超过可执行策略的整日雇工支出上界，故不能再将它解释成未精确拟合的真实工资。
这隔离出一个确定的现金流量纲/边际错位，未说明尾值总差 `$10.9k` 由它造成：正式 h1 的第一日
scenario 已用真实订单现金，错误主要可能出现在静态 tail、候选生成及未来工作量影子成本。候选
下一日总现金差还有其他交易与动作效应，不能将其与 `$271.7` 相加当作真实收益。
现有同状态 day13 的正式静态 tail 分项中，id1−id0 的工资项反而是 `−$336`（从下一日开始），
与第一日代理的 `+$272` 不同号；不能用第一日代理差去解释这笔 `$10.9k` tail 排序。

建议的内部不变量是：同一候选、同一天的 `HIRE` 现金预测必须等于实际可执行订单的逐单价格和；
工作量机会成本另列为约束的对偶项，并检查其边际与项目取舍是否一致。先在固定 portfolio 与同一
候选集内重估首日/未来日工资差，追踪候选差分和 `argmax`，再处理在线 continuation；不能直接
把 `labor_hours` 调到碰巧使旧胜率好看。产物：`work/r1-day13-hire-arm0-thomas-seed100-v2.json`、
`work/r1-day13-hire-arm1-thomas-seed100-v2.json`、`work/r1-greedy-stop-day13-thomas-seed100-v2.json`。
量级筛选只读已有 32 个 handoff：每状态 formal normal 前两名 score gap 中位数 `$493.7`，
两名**静态 tail 工资分项差的绝对值**中位数 `$164.3`；12/32 状态里后者大于 gap。
这说明“把工资代理重新解释/校准”足以进入排序敏感域，**不代表**代理误差本身就有这么大，
也不代表修正后必换序；须在同一候选和公开情景下先计算实际执行雇工路径的差分。
同一 32 状态的前两名有 24/32 计划 `HIRE` 数相同，其中 22/32 的预测劳动仍相差超过 1；
重复 seed/不同对手状态相关，不能当成 32 个独立触发样本。实际成交相同目前只验证上述 day13
两臂，其他状态仍须按订单顺序核对现金与工人上限。

更深的待验证约束是：`Asset.f[d]` 先按全部护理/收获成功生成产量，再用 `wages(L)+4L` 软罚
劳动；真实规则里产量取决于离散路线、24 步容量、种子/饲料与服务动作的完成。软罚不能同时充当
雇工现金和执行可行性证明。统一目标至少要分开：实际可执行 `HIRE` 现金、资源日历约束下的条件
产量，以及可选的工作量对偶价。是否发生未来少产/漏服务须对拍执行任务与产量，不能由雇工
现金错位直接推断。

另一个严格量纲点：`Planner::value()` 令 `cash=a.fixed+own_trade−wages(L)−action_cost×L`，
再把这个 `cash` 累加到 `balance` 计算未来负现金惩罚。`action_cost×L` 只是影子机会成本，
不是官方会扣除的现金，故这个 `balance` 不是资金余额，不能用它证明采购/雇工可融资。
当前风险项的覆盖率很低：已有 32 handoff 的 124 个 normal 静态 tail 中
`liquidity_penalty` 全为 0；此缺陷在这批状态不影响排序，但应在现金紧张样本中按真正可支付
订单和销售时序重建融资约束，而不是调 `risk` 权重补偿。

原则上的分解是：对离散可执行任务/雇工 `x,h` 最大化同一个终局 continuation，约束
路线可行性（`24×(1+h)` 只是工步的粗上界，实际还受移动/雇入时点约束）、真实现金前缀和
库存/饲料前缀；雇工成本用
`C_hire(h)=Σ_{j<h}Fib(j)` 精确计入现金。只有为降低搜索代价而松弛容量约束时，才可引入
对偶价 `μ_t` 惩罚增量工作量；此 `μ_t` 应由拥挤和被挤掉的任务价值决定，不能预设全局常数
`work_price=4`，也不能再次进入可支付现金。目标从 margin 切到胜率时，对偶价还需在相同
终局效用尺度上定义。
已有 executor 的当日 `prepare_orders()` 会用 `jobs()`、`estimate()`、`funding/admit()` 生成
`HIRE` 队列，`hirecost(h)` 正是 Fibonacci 逐单和，可复用来对拍首日现金；但 `estimate()` 在
所有 `h≤max_hands` 都排不下任务时仍返回 `max_hands`，所以**队列雇工数不是任务完成证书**。
须再核对选定 h 的 `pack(...).second==0`、执行后的 receipt/`actual_drop` 与条件产量。

### 2026-09-23：同视图次日 seam 的零差分边界

只读诊断用正式 warm 链在 Thomas seeds `2609800100..103` 的 step288 捕获 4 个 handoff，
共 16 个 normal 候选。对每个候选分别比较：正式 h1 首日 scenario 结束后的
`tail=live; book/joint=roll; tail.plan(next_view)`，以及同一首日 scenario 继续用
`roll.act(next_view)` 时的次日重规划。两者 `predicted` 在 16/16 候选上逐项相等，
次日首日 portfolio 供给也逐项相等；因此这个 seam 的绝对偏移与候选差分均为 0，
按该 seam 值替换 h1 tail 不会发生任何换序。不能再把已观察到的长期闭环差异直接
归因为「首个次日 plan 的模型价值不一致」。

两者 `proposal_key` 字面不等，但审计钩子在 `roll.act()` 返回后记录，此时首个动作已从
执行队列消费；16/16 候选的前 226 项（target 与 planned land）和服务日历尾部 400 项
相等，只有中间队列长度差 30 项，即 10 个订单三元组。key 不等是记录时点不同，
不是同一时点计划分叉的证据。该检查**没有**证明日内执行轨迹或后续多日计划相等，
也不消除静态 tail 不等于终局闭环 continuation 的已确认结构问题。
尤其该零 seam 仅针对 base `Controller::plan/act`，**不涵盖线上下一日
`SearchController::choose` 重新枚举 normal 与 joint 候选**。已有
`work/r1-day13-search-thomas-seed100.json` 在 day12 id0 强制后的实际 step312 状态中，
在线 day13 normal id1 的内部 score 比 normal id0 高约 `$11,652` 并被选择；day12 id1
分支的 day13 则选 id0。它是同一线上搜索层的条件策略分叉，不与 base plan 零 seam 矛盾，
也不能直接当作真实 margin 增益。
另两个只执行到 step312 的同 handoff 双臂内部对照给出尺度：seed101 的 day12 id0/id5
在 next-day 在线搜索中，相对 normal0 的 score 增量分别为 `+$528/+$4`，候选间
option-lift 差 `+$524`；首日预测与实际现金误差为我方两臂均 `+$13`、对手均 `+$412`，
主要是共同项。seed102 的 day12 id1/id5 对应增量 `+$167/+$137`，差仅 `+$30`，
首日现金预测误差也仅我方 `+$9/+$21`、对手 `+$377/+$371`。所以搜索层漏计的
条件机会在 seed101 足以接近 h1 的数百元 gap，但 seed102 不能以此解释终局模型换序；
更后日期/尾值/执行仍要分层核查。产物：`work/r1-day13-search-thomas-seed101.json`、
`work/r1-day13-search-thomas-seed102.json`。这些 online option lift 是**下一日模型内部
score 差**，不是可直接加到上一日 Q 或真实终局 margin 的收益。
seed103 的 day12 id1/id0 也仅执行到 step312，次日 option lift `+$283/+$104`，
差 `+$178`，反而偏向 h1 原 winner id1；首日预测→实际现金误差为我方
`−$51/+$36`、对手 `+$483/+$404`，按 `own−2·rival` 口径候选残差约 `−$245`。
连同 seed102，单个 next-day 搜索机会并不能解释这两例 h1→模型终局的 `$2k–3k`
候选残差范围。产物：`work/r1-day13-search-thomas-seed103.json`。
产物：`work/r1-same-view-tail-seam-thomas-seed100.json`、
`work/r1-same-view-tail-seam-thomas-seed101-103.json`。

同一批状态、同候选及同一条 open-loop 对手供给下，正式 h1 score 与
`score(horizon=30−day, shop=−1)` 的**模型内部**终局 rollout 比较，单状态所有候选的
`Q_full−Q_h1` 存在大的共同偏移，但其候选内极差依次为 `$1,763/$1,962/$2,368/$3,132`；
4 个状态里 3 个 winner 换序（seed101 `0→5`、102 `1→5`、103 `1→0`）。例如 seed102
h1 的 id1 相对 id5 为 `+$1,001`，终局 rollout 却为 `−$1,039`。这证明长期截断误差
进入当前候选差分的尺度，不仅是可忽略的共同截距。这里仍**不是实际终局 margin**，
也不能拆成「未来再规划」单因果：延长 rollout 同时替换了后续静态估值、成交与执行；
`shop=−1` 是固定的平均商店模型，不代表真实未知商店分布。首个次日 seam 精确零差分，
故下一步应沿后续日边界逐层定位首次显著非共同差分。
已有 `scores_horizon[1..5]` 表明 4 个状态中 3 个在 h2 已换 winner：seed100
`1→0`、seed101 `0→5`、seed103 `1→0`；seed102 直到 h5 仍选 id1、终局才选 id5。
h2 同时纳入 day13 执行与 day14 的新静态截尾，所以这里只定位首次可见的排序分叉时域，
尚不能把差分单独归因给当日执行或下一边界估值。h2..h5 的 winner 还会来回变，
有限截断值没有呈现可据以外推终局的单调稳定性。

可执行的下一步不是盲延长 horizon，而是把同一外生情景、同一线上策略与同一起点价值时钟固定，
先明确选择同一个终局效用 `U`（例如 cash margin；胜负效用须用情景分布），
在每个日边界定义 `B_k(c)=已兑现至 k 的 U 增量 + V̂_k(S_k^c)`，并逐候选记录
`D_k(c)=B_{k+1}(c)−B_k(c)`。在终局余值为零时，`Σ_k D_k` 正是 h1 到该情景终局
的误差分解；排序只看相对基准候选的 `D_k(c)−D_k(b)`，公共大偏移不算决策收益。
第一个使累计候选残差跨过原 h1 gap 的日边界，是优先检查执行/尾值/市场状态的定位点。
现有 `scores_horizon[1..5]` **不是**这张 Bellman 残差表：它沿 base Controller
而非线上 SearchController，并重置静态尾值的折现起钟。因此只能作时域敏感度筛查，
不能把相邻 horizon 的变化直接归因给某天的单独机制。

上述边界定义现有默认关闭的 `td_candidate_online_boundaries_json` 诊断入口：固定同一
`PublicFlowScenario` 对手供给，首日安装同 handoff 候选，此后逐日调用真正的
`SearchController::act/choose`；每个边界在共同起点 `ValueBasis` 下记录已兑现的
`own−competition·rival`、静态 `V̂` 及两者之和。该实验仍只评价模型自身的 `competition=2`
代理，不是终局胜负效用；诊断最多向前六天，不跑完整对局。入口仅在
`R2_SHOP_BRANCH_AUDIT=1` 编译，正式宏 0 不改变生产行为。

Thomas seed102 的 day12 normal id1/id5 共用对手 flow，fixed-clock h1 与 day13 首个在线
边界值逐项精确相等，确认这一对照的首日 transition/时钟契约。`B_k(id1)−B_k(id5)`
在 day13..18 依次为 `+$949,+$1,208,+$1,245,+$292,−$35,−$659`，首次换序在
day17。day16→17 的累计已兑现值差反而从 `+$515` 升到 `+$1,528`，静态尾值差从
`−$223` 降到 `−$1,563`；这次换序是边界尾值重估引起，不能解释为此前现金收益已
兑现反转。尾值为何如此变动仍待同边界分项/商品/资产拆解。产物：
`work/r1-online-bellman-thomas-seed102-day12-18.json`。
加入同一 `V̂` 的分项重算后，day16→17 候选静态 tail 差的 `−$1,340` 变化中，
`own_trade` 差从 `−$683→−$1,947`（变化 `−$1,264`）占主导，
`rival_penalty` 差 `+$242→+$391` 反向抵消约 `$149`，工资/动作较小。
商品上主要是 STRAWBERRY 的 id1−id5 估值差 `+$590→−$361`（`−$951`）和
MELON `+$326→−$40`（`−$367`），TOMATO `−$51→+$406`（`+$457`）部分抵消。
两个边界的剩余 STRAWBERRY 总供给差都为 id1 多 4 单位；所以这不是简单的
总供给量下降，需继续检查日别生产/成交与价格时序，不能仅凭分项把根因写成草莓。
重算 `ValueBreakdown.total` 与 `tail.predicted` 每边界均相等，产物：
`work/r1-online-bellman-thomas-seed102-day12-18-v2.json`。
进一步核对日别流与当前库存：id1/id5 两臂在 day16 的 STRAWBERRY 私有总存货均为 8，
静态尾值的 day16/17 流分别为 `40/16`；到 day17 私有存货均升至 32，静态尾值当日流
变为 `48=32 当前存货+16 当日预测产出`。这证实实际在线路径出现**共同的 +24 净结存**，
而静态 tail 将 day16 的 40 单位逐日 flow 当作该日可入市量，没有显式携带到下一边界的
我方库存状态；共同库存冲击经候选不同的后续日别供给与非线性报价可变成排序差分。
逐单市场成交诊断进一步闭合：两臂 day16 草莓起始私有存货均为 8、当日 SELL 实际
成交均为 8、BUY 均为 0，day17 私有存货均为 32。草莓没有其他私有耗用路径时，
当日新产出 32、只卖 8、留存 32；day16 静态尾值却把 `8+32=40` 全量在当天
`exchange()`，没有将 32 单位 carry 纳入下一日市场状态。这里的库存时序违背
确定，但它是两臂共同发生；尚不能把整个约 `$1,264` 的 own-trade 相对 swing
归为单一 carry 因果，因为候选日别供给、后续价格与重规划一起变化。
两候选在同一边界的对手 forecast 总 flow 逐商品完全相同，故这里不是候选特异的对手 belief
漂移。跨边界共同 forecast 仍改变，例如剩余 STRAWBERRY `288→232`、MELON
`0→60`；它与己方时序可交互，但也不能仅凭总量证明是 stale-board 重复收获。
产物：`work/r1-online-bellman-thomas-seed102-day12-17-v6.json`。

对应的数学约束应显式写为每种商品的私有库存
`Z_{d+1,i}=Z_{d,i}+Y_{d,i}+B_{d,i}−S_{d,i}−C_{d,i}`，其中 `S` 是实际可在
日内订单槽/现金/持仓约束下成交的量，且市场库存和我方/对手现金必须按官方逐事件次序
同时推进。当前 `Planner::value()` 对 `a.f[d][i]` 直接 `exchange()`；对正产出的
非必需消费商品，这等价于将该日总可用 flow 强行取 `S=a.f`、跨日 `Z=0`。这不是
执行器 `LocalSaleTiming` 或
收获/入仓时点的语义。下一步先在**固定 portfolio**上将销售/持有日历作为显式状态，
核对库存、成交、价格、现金的内部守恒及候选差分；之后才联动重规划、工资代理和
对手库存 belief 重校。不能单独打开 `sale_dp` 参数当理论修复：其规划销售日历还须与
现有 executor 的实际入仓、销售限制及首日资金时序逐项对齐。

对此错误再做严格的固定 portfolio 反事实：在 day16 同一两臂各自的原 portfolio、
同一市场起点/对手 flow/折现时钟下，只把实际跨日的 32 STRAWBERRY 从 day16
`a.f` 移到 day17，并重算 `Planner::value()`，不让 planner 或后续执行重新决策。
两臂尾值均**精确降低 `$126.262185`**，候选差分为 0，day16 winner 不变。
该相等有结构原因：day16 边界两臂的草莓起始市场库存、day16/day17 原供给
`40/16` 和同日对手预测流均相同；挪量守恒且到 day17 结束时累计草莓供给恢复，
更远价格路径也不因这一挪量而产生候选差。
这区分了确定的守恒错误与当前排序主因：一日 carry 修正本身在此状态只是共同项，
**不能**解释 day17 的尾值差约 `−$1,340` 变化。后者仍须拆更后续的条件重规划、
未来供给时序与价格反馈，不能用此单点误差推动生产改动。产物：
`work/r1-online-bellman-thomas-seed102-day16-shift32.json`。

对手预测流也做同边界固定 portfolio 反事实：day17 保持当时己方 portfolio、
市场起点和价值时钟不变，仅把重新生成的 `tail.model.rival` 换成 day16 已预测的
剩余对手流。id1/id5 的尾值分别降低 `$3,629.88/$3,792.75`，绝对量级很大，
候选间修正差仅 `+$162.87`；对手流重建在本例对 `−$1,340` 的跨日尾值差
有贡献，却不是其主量。商品日别核对表明：day16 对 day17 的 STRAWBERRY 预测为
16，day17 重建为 8；day27 `46→26`、day29 `20→0`，两候选变化相同。
这项比较是固定 portfolio 下的重估差，未把新流重新送入 day17 planner；不能当成
完整策略修正。产物：`work/r1-online-bellman-thomas-seed102-day12-17-v9.json`。

市场起点也做同边界固定 portfolio 反事实：用 day16 静态 tail 的逐日 `trade()`/需求
方程推算它预计的 day17 市场库存，再在 day17 保持己方 portfolio、对手预测流、需求和
时钟不变，仅把实际市场起点换成该旧预测。STRAWBERRY 库存旧预测为 9872，实际场景
为 9846，两臂相同；全商品旧预测市场起点使 id1/id5 tail 分别上调
`+$3,617/+$3,474`，但候选间修正只 `+$143`。因此市场状态错位的绝对量级也很大，
在此例仍主要是共同项，不足单独解释 `−$1,340` 排序差。与旧 rival-flow 的
`+$163` 差分不可机械相加：二者通过同一非线性价格函数交互。产物：
`work/r1-online-bellman-thomas-seed102-day12-17-v10.json`。
两项不能相加的警告已经由联合反事实证实：day17 同一当前 portfolio 下，同时使用
day16 预测的市场起点与对手流，id1/id5 tail 分别改变 `−$101.65/+$113.45`，
候选间差 `−$215.10`，与两个单项候选差 `+$143/+$163` 的和反向，非线性交互约
`−$521`。联合旧状态并未恢复 id1 的排序，反使它相对更差。故应转向同边界旧/新
portfolio 的商品结构与重规划差分，不再把市场/对手的数千元绝对误差当作排名主因。
产物：`work/r1-online-bellman-thomas-seed102-day12-17-v11.json`。

同一 day17 当前市场、对手预测、需求和时钟下，固定 day16 的旧 future portfolio
与 day17 新 portfolio 比较时，必须先把当前实际私有现货补进旧组合的 day17 flow；
否则等于凭空丢掉边界库存。未补现货的旧组合 id1−id5 尾值差为 `+$1,189`；
逐商品补入 day17 实际现货后变为 `−$1,355`，相对变化 `−$2,543`，足以跨越排序；
day17 真正重新规划组合尾值差为 `−$1,563`，相对库存对齐旧组合只再变 `−$209`。
这说明本状态的**边界私有库存/资产状态对齐**对内部 Q 排序更重要，不能笼统把
day17 换序归为“重新选择未来组合”。day17 两臂现货草莓均 32，差别主要包括
MILK `15/18`、FERTILIZER `11/15`；`−$2,543` 是全库存与非线性市场的联合
估值差，不是这 7 件商品的直接售价，也不证明旧组合现实可执行。
产物：`work/r1-online-bellman-thomas-seed102-day12-17-v12.json`。
逐品项把当日现货加入同一旧组合后，候选估值差贡献约为 MILK `−$2,278`、
FERTILIZER `−$321`、STRAWBERRY `+$59`、EGG `−$4`；单项和与全库存联合值
只差约 `$1`。因此此状态下的库存对齐换序主要集中在 **MILK 现货×各自候选市场/旧组合**
的模型边际/竞争估值，而不是双方相同的 32 单位草莓 carry。`−$2,278` 远大于
两臂 MILK 存货数量差的直接售价，但并非“只增加 3 单位 MILK”的纯因果效应：
两臂 day17 MILK 市场库存为 `10013/10010`，旧未来组合流也不同。下一步必须拆
`own_trade` 与 `rival_penalty`、检查价格曲线/地板及
市场冲击，不能将这笔模型分数解释成真实现金。产物：
`work/r1-online-bellman-thomas-seed102-day12-17-v13.json`。
MILK 的 `−$2,278` 候选差现已按同一 `ValueBreakdown` 拆开：补入 id1 的 15
MILK 使自身交易项 **降低 `$1,366`**、对手收入惩罚项改善 `$4,287`，净值
`+$2,921`；补入 id5 的 18 MILK 使自身交易项降低 `$2,146`、对手项改善
`$7,345`，净值 `+$5,199`。因此在各自市场与旧组合条件下补入现货，对手惩罚项
相对偏向 id5 约 `$3,058`，自身销售被压价的相对变化抵消约 `$780`，净差约 `$2,278`。
此处的排序杠杆是 `competition=2` 下的**预期市场外部性**，不是 3 单位 MILK
的直接售价；对手未来是否按该 flow 成交、价格压制是否真实可兑现仍未验证。
仅对此固定 MILK 现货反事实作代数重权重，若把对手项系数从 2 改成 1，
候选净差仍约 `−$749`（旧 `−$2,278`）；系数 2 显著放大，但不是效应存在
的唯一原因。完整 c=1 策略会重选 portfolio、改变市场与对手响应，不能由此
局部重权重推出它的 winner。
不能把模型内部 `own−2·rival` 分数称为真实 cash margin，也不能仅凭本例把
`competition` 调回 1——该系数可能补偿其他模型错位。产物：
`work/r1-online-bellman-thomas-seed102-day12-17-v14.json`。

相同隔离入口扩到另外两个独立 handoff，仍只到 day17：seed101 的 id0−id5
边界分差在 day13..17 为 `+$431,−$66,+$120,−$121,−$28`，day14 首次换序后
继续反复；seed103 的 id1−id0 为 `+$935,−$342,−$225,+$613,+$347`，
day14 翻转又恢复。两例 day13 边界也与各自 fixed-clock formal h1 精确相等。
因此 seed102 的 day17 MILK/库存高杠杆不是所有状态的统一根因；有限 horizon
排序不稳定来自状态、市场、尾值和线上策略的连续反馈，不能用某一个截断日或商品
修补替代 Bellman 一致性。产物：`work/r1-online-bellman-thomas-seed101-day12-17-v14.json`、
`work/r1-online-bellman-thomas-seed103-day12-17-v14.json`。

三个首次换序点的共同分解更明确：旧 h1 winner 的累计已兑现 `own−2·rival`
差当日**都改善**，而静态 tail 差突然恶化并越过它。seed101 id0−id5 从
day13→14：realized `−$80→+$276`（改善 `$356`），tail `+$511→−$342`
（恶化 `$853`），边界 gap `+$431→−$66`；seed103 id1−id0：realized
`+$1,200→+$1,412`（改善 `$212`），tail `−$265→−$1,754`（恶化 `$1,489`），
gap `+$935→−$342`；seed102 id1−id5 day16→17：realized `+$515→+$1,528`
（改善 `$1,013`），tail `−$223→−$1,563`（恶化 `$1,340`），gap
`+$292→−$35`。这是跨状态的**边界尾值不稳定**证据，不等于三例拥有同一商品
根因，也不是实际终局 margin 结论。
两处 day14 首翻的尾值分项进一步证实不能共用一个微观补丁。seed101 id0−id5
的 tail 恶化约 `$853`，自身交易差反而改善 `$183`、对手项只变 `−$9`；主变化是
连续工资代理差 `+$484→−$76`（`−$560`）、`action_cost×labor`
`+$179→−$68`（`−$248`）、fixed `+$130→−$90`（`−$219`）。seed103
id1−id0 的 tail 恶化约 `$1,489`，主要是自身交易差 `−$4→−$1,864`
（`−$1,860`），对手项反向改善约 `$510`，工资不是主项。seed102 又由
MILK 现货×市场外部性放大。统一问题是静态尾值在状态边界重估时不满足稳健的
Bellman/执行语义，不是单个商品、价格 primitive 或某一工资参数。
