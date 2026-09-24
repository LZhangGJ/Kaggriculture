# R1 系统梳理：模块、参数、假设与实测结论

**目的**：把"哪个模块做什么、每个参数喂到哪里、哪些假设已被实测证否、哪些还开着"集中到一处。
之前这些散在几十次实验、记忆文件和代码注释里，导致同一个问题被反复重测、同一个错误被反复犯。

**基线口径**：段间胜率差异极大（thomas 在 64-seed 段上从 70.8% 到 87.5%），**只能比配对差，不能比绝对值**。
效果的判据是 +2,200 margin ≈ +6pp（由 `max_animals` 20→40 实测得出）。

---

## 一、线上管线

```
Kaggle / FastEnv 观测
  └─ agent/main.py :: agent()
       ├─ replay_deployment() 就绪 → ReplayThenDynamicAgent
       │     ├─ step < 288 : 浅树选一条 replay 路线（route_policy.json + route_actions.json）
       │     └─ step ≥ 288 : 交接给下面的动态策略（handoff_step 默认 288）
       └─ policy.Agent  (policy/r1/agent.py)
            ├─ _pack(observation) → double[]     ← 观测打包成扁平数组
            └─ ctypes → policy/r1/agent.so (bridge.cpp)
                 └─ SearchController (search.hpp)
                      └─ Controller::live  ← ★ 线上唯一真正决策的对象
                           └─ Planner::model  (planner.hpp)  ← 估值 + 组合选择
                           └─ dp7::Controller core (executor/policy.hpp) ← 执行/调度
```

**关键事实**：`proposals.hpp` / `generate_proposals` / `candidate_*` 参数 / `portfolio_passes`
在 `scenario=1` 时进入线上每日 `choose()`：候选先各自 `plan()`，经一日 public-flow rollout
评分后，胜出副本整体安装为 `Controller::live`。147/356 维候选特征只服务于离线诊断或显式启用的
selector，不参与默认 `scenario=1` 的规则评分。不能把“候选特征离线”误写成“proposal 不在线运行”。

### ⚠️ `Controller::plan` 在 live 控制器上**从不执行**（重要且反直觉）

`SearchController::act` 的顺序是：

```cpp
if(base.scenario!=0&&o.day!=live.core.day)choose(o);   // config 里 scenario=1 ⇒ 每天触发
auto out=live.act(o);                                  // Controller::act
```

而 `choose()` 结尾是 `install(proposals[winner], o.day)` →
`live = proposals[winner].policy;` —— **整体替换 `live`**。
那个候选副本的 `core.day` 已经在它自己被评估时（`proposals.hpp` 的 `p.plan(o)`）设成了当天，
于是随后 `live.act(o)` 里的 `if(o.day!=core.day)plan(o)` **条件为假，`Controller::plan` 被跳过**。

**后果（两条，都会咬人）：**
1. **任何挂在 `Controller::plan` 上的每日初始化，在 live 控制器上都得不到执行。**
2. 挂在 `Controller::plan` 上的计算只会发生在**每个被评估的候选**上（实测 **305 次/局**），
   而执行器实际继承的是**胜出候选**的那份状态 —— 不是最终承诺组合的。

卖出计划因此被移到 `SearchController::act` 里、`choose()` 之后、`live.act()` 之前，
用 `live` 自己的组合构建一次（实测 **18 次/局**，耗时从 +370% 降到 +21%）。
判据必须比 **天**（`exec_plan_day != o.day`），比 step 会在当天 24 步里每步重建（实测 431 次/局）。

---

## 二、线上 `Controller::act()` 的执行顺序

```
1. joint.observe(o)                      观测 → 内部状态
2. if (o.day != core.day) plan(o)        ★ 换天时重建"承诺组合"（每天一次）
      ├─ demand(o)                       需求模型（见 4.1）
      ├─ public_rival(o)                 对手供给模型（见 4.2）
      ├─ 逐地块选项目 → portfolio        组合选择（见 3）
      ├─ predicted = value(o, portfolio) 估值
      └─ build_exec_plan(portfolio, o)   卖出计划（仅 sale_dp>0）
3. local_sale.plan / carry / seed_bucket 把计划交给执行器
4. working_capital_gate(o)               流动性兜底（仅 P16_WORKING_CAPITAL_GATE=1）
5. core.act(o)                           dp7 执行器出动作
6. settle_market(o, out, preparing)      ★ 卖出过滤（见 5）
7. joint.record(o, out); return out
```

---

## 三、组合选择（"种什么"）

**决策点**：`triad.hpp` 的 `Controller::plan` 内层循环
```cpp
double gain = val - current - s.land_rent * (k>=9 ? 29-o.day : len);
double rank = gain / pow(max(10., cost) + (k>=9?100:0),
                         s.capital_power * max(0., 1-o.own.money/16000.));
rank *= (k>=9 ? s.animal_bias : s.crop_bias);     // ← 对所有作物乘同一个数，无法单独偏向某个作物
```
候选集 `k ∈ {0,1,2,3,4, 9,10,11}` = WHEAT/CARROT/TOMATO/STRAWBERRY/MELON + GOOSE/COW/SHEEP。

**被实测证否的"种什么"干预**（不要再走）：
| 干预 | 结果 |
|---|---|
| 强制多种胡萝卜（×3/×10/×30/×100，128 seed） | +65 [−420,+571]，无效。胡萝卜与小麦在引擎模型里是 1:1 替代 |
| `max_land` 3→4 | **−2,889 / −1,318 / −2,283**，三手全显著差 |
| `rotation` 0→1 | **−3,008 [−4197,−1822]** |
| `animal_bias` 0.5/0.7/1.3 | 四臂全在噪声内 |
| 象限假设（thomas 4 块地 vs 我们 3 块） | **反向**：输了 7 局里它一块地都不多；赢的 17 局里有 4 局它是 4 块 |

**已上线且验证**：`max_animals` 20→40。2670 段 thomas 75.0%→81.2%（+2,211 margin）；
2970 段 71.9%→75.0%（+634 [+4,+1312]）。

**已上线且验证**：`replant=0.5`（见 4.2）。

---

## 四、估值模型 `planner.hpp::value()`

```cpp
for (int d = day; d < 30; d++) {
  double cash = a.fixed[d], enemy = 0;
  for (int i = 0; i < 9; i++) {
    inv[i] -= dem[d][i] * .5;                    // ① 当天需求的前一半
    double r = cfg.supply * rival[d][i] * .5;    // ② 对手供给的前一半
    enemy += trade(i, inv[i], r);
    cash  += trade(i, inv[i], a.f[d][i]);        // ③ 我们：整天的产量一次卖出
    enemy += trade(i, inv[i], r);                // ④ 对手后半
    inv[i] -= dem[d][i] * .5;                    // ⑤ 需求后半
    if (prices) (*prices)[d][i] = price(i, inv[i]);
  }
  cash -= wages(a.labor[d]); cash -= cfg.action_cost * a.labor[d];
  balance += cash; liquidity += max(0, -balance);
  val += (cash - cfg.competition * enemy) / pow(1 + cfg.discount*max(0,1-money/20000.), d-day);
}
return val - cfg.risk * liquidity;
```

**`trade(i, inv, q)`（`R2_MARKET_INTEGRAL=0`）**：
```cpp
double p = quote(i, inv, q);          // 价格曲线在 [inv, inv+q] 上的三点 Gauss 均值
if (q<0 || p>1.) inv += q;            // 库存被推进（价格=1 时不推进）
return q * p;
```
**对 q 是凹的** ⇒ `enemy` 对 `supply` 不是线性的 ⇒ `competition × supply` **不是**一个有效乘积。

### 4.1 需求模型 `demand()`
```cpp
dem[d][i] = (i<8 ? 1 : 0)                       // 城镇中心：每商品每天 +1
          + known[i]                             // 已解锁商店：每家 +6/天
          + future_shop * max(0,min(8,d/3)-已解锁) * avg[i];   // 未解锁商店的期望
```
与引擎 `simulator.cpp::town_consume` 逐字对应（每 4 步每店扣 1 ⇒ 6/天；每 24 步每商品扣 1）。

### 4.2 对手供给模型 `public_rival()` —— **已测出结构性缺陷**
```cpp
for (pos) { if (plant(t)) {
   ... a = crop_cycle(...);
   if (next<29 && cfg.replant>0)                       // 当前 config 为 0.5
      add(a, crop_dp(best_rotation(...), ...), cfg.replant);
}}
rival[d][i] = Σ 对手地块的 a.f[d][i];
```
以下“没有轮作、没有补种”的后果描述的是历史 `replant=0` 基线；当前 0.5 已修正供给规模，
但 `best_rotation()` 对补种商品构成的估值仍是近似。

**实测（一局 vs thomas，模型 vs 实际）**：
| 商品 | 模型 | 实际 |
|---|---|---|
| CARROT | **0** | 2 / 26 / 26 |
| TOMATO | **0** | 14 / 14 / −50 |
| STRAWBERRY | **0** | 10 / 33 / 34 |
| MELON | **0** | 14 / 14 / 14 |
| MILK | 21 / 15 / 14 | −25 / 10 / 49 |
| WOOL | 12 / 4 / 4 | −2 / 1 / 1 |

（负值来自我的恒等式缺了"价格=1 时卖出不推进库存"这一项，逐日数值不可信；
但"模型恒为 0、实际为正"的定性结论不受影响。）

**这条曲线是给我们自己的卖单定价用的** ⇒ 我们在拿一条"对手会消失"的价格曲线定价，系统性偏乐观。

**`replant` 实测**：
| 对手 | replant=0 | **0.5** | 1.0 | 1.5 |
|---|---|---|---|---|
| thomas | 75.0% | **79.7%** | **79.7%** | 73.4% |
| melon | 82.8% | **85.9%** | **85.9%** | 84.4% |
| demand | 85.9% | **89.1%** | 87.5% | 81.2% |
| 合并胜率 | 81.25% | **84.90%** | 84.38% | 79.69% |

**胜率三手全涨（+4.7/+3.1/+3.2pp），margin 持平。** 过度修正（1.5）反噬。

**合并两段（448 seed 单位，4 个对手）—— 已上线 `replant: 0.5`**：

| 对手 | base | rp05 | Δ胜率 | Δmargin | W/L |
|---|---|---|---|---|---|
| thomas | 80.5% | **83.6%** | +3.1pp | +99 | +9/−5 |
| melon | 85.9% | **87.5%** | +1.6pp | +434 | +6/−4 |
| demand | 86.7% | **88.3%** | +1.6pp | +7 | +5/−3 |
| ahmed | 92.2% | **93.8%** | +1.6pp | +530 | +2/−1 |
| **合计** | 85.5% | **87.5%** | **+2.0pp** | +230 [−146,+610] | **+22W/−13L** |

四个对手的胜率全部上升，符号在两个独立段上一致。

---

## 五、卖出层

### 5.1 本地规则 `local_sale_timing.hpp`（`R2_LOCAL_SALE_TIMING=1`）
- `due(step) = ((step+3)/4)*4+1`：下一个 4 步桶边界（**引擎每 4 步消费一次需求，这是它的自然周期**）
- `amount_now()`：在每个桶内比较"现在卖 q、需求扣掉后再卖剩下的"的总收入，取最优 q
- `rival` 传 0（mode 1）
- **它只在非换天步执行**：`settle_market` 第一行 `if(delay_sale<0||(preparing&&feed_finance<=0))return;`
  在换天步（`preparing = core.phase==1`，`feed_finance=0`）恒真 ⇒ 换天步完全不过 filter

### 5.2 卖出 DP `sale_plan_dp.hpp`（`sale_dp`，默认 0 = 未上线）
**已实现且通过离线自测**（`cases=400 self-inconsistent=0 beaten by reference=0`）：
- 周期 = **4 步桶**（108 期），需求用引擎自己的 `consumption` 节奏
- 状态 = **我们持有的库存**，硬约束 `s ≤ 棚容`
- 目标 = `own − rival`（`competition=1`）
- **计划语义是"这个桶结束时留多少"（carry target），执行器卖到目标** —— 不是"这个桶卖多少"
- 棚容预算 = `100 − 每只动物 × feed_cover`（饲料必须先占位，否则牲口饿死）
- `budget` 之外无上限（`1e9`）

**实测（三次独立测量，结论一致）**：

| 版本 | 条件 | margin | 胜率 |
|---|---|---|---|
| 第一版 | 日粒度、carry-target、对手模型带 `replant=0` | +356 [+182,+528] | +5W/−5L |
| 第二版 | 同上，但对手模型已修（`replant=0.5`） | +218/+307/+151/+200 | +7W/−3L |
| **第三版** | **修正计划所有权（承诺组合）+ 代价降 17 倍** | **+215/+266/+20/+41** | **+7W/−8L** |

**结论**：DP 有两个真 bug（计划挂在从不执行的 `Controller::plan` 上、执行器跟的是胜出候选而非承诺组合），
但修掉后**结果不变** ⇒ 这两个 bug 不是它无效的原因。**DP 在本系统中的真实上限是"小幅正 margin、胜率持平"。**

**未上线** —— 目标是胜率。

**已知的坑（都修过）**：
| 现象 | 原因 |
|---|---|
| 计划恒为"全卖" | 状态轴锚在市场库存上，k=0 时唯一可行状态不可表示 |
| −24k/局 | 禁止卖入饱和区的 clamp ⇒ 大部分商品大部分时间卖 0 |
| −7.8k/局 | 计划囤货超过棚容 ⇒ 引擎销毁溢出 |
| −4.3k/局 | `plan_done` 记"打算卖"而非"卖掉" ⇒ 当天预算一次烧光 |
| −3,365/局 | 视界之外的桶被填成小数 ⇒ 执行器当成上限 ⇒ 终局抱着 28.9 单位 |
| −28,178/局（单 seed） | 在换天步绑定计划 ⇒ 走进一条**原始代码从不执行**的融资分支 ⇒ 采购断粮 ⇒ 牲口饿死 |
| 计划不生效 | 两个卖出通道各自对"当天目标"记账 |

---

## 六、执行器层（`executor/`，19 个模块）

线上真正决定"每个工人这一步做什么"的地方，也是 bug 最集中的地方。

| 模块 | 作用 |
|---|---|
| `policy.hpp` (dp7::Controller, 1388 行) | 主调度：把 `core.target` 展开成工人的动作序列 |
| `settle_market` (在 triad.hpp) | 卖出过滤：本地规则 / 卖出计划 |
| `working_capital_gate` | 流动性兜底：现金不够付饲料时，卖抵押品 + 买饲料 |
| `intraday_admission` | 盘中接纳新项目 |
| `joint_portfolio` / `joint_candidates` | P16 联合捆绑（动物+饲料+种子的整包决策） |
| `t3_execution_repair` / `t3_repair_state` | 执行修复（义务/产能/收据） |
| `service_recovery` | 服务（喂/照顾）的补救 |
| `crop_delivery` / `day_consequence` | 作物交付 / 单日后果投影 |
| `observed_day_scenario` | 用观测重放一天的模拟器（**注意：`project_sales` 只在这里被调用**） |
| `rotation_calendar` / `rotation_cash` | 轮作日历 / 现金日历 |

---

## 七、参数表

**判据**：`s.X` / `cfg.X` 的实际消费点（扫描 `policy/r1/**/*.hpp`）。

### 7.1 估值模型（直接改变 `value()`）
| 参数 | 默认 | config | 喂给 | 实测 |
|---|---|---|---|---|
| `competition` | 0.8 | **2** | `val = cash − c×enemy` | **1 更差**（demand −12.5pp, p=0.008）；2 兼职"抑制过度生产" |
| `supply` | 0.85 | 0.85 | `r = supply×rival×0.5` | **已在最优点**：0.5 → −1,180，1.2 → −1,205，2.0 → −8,831 |
| `future_shop` | 0.7 | 0.7 | 未解锁商店的期望需求 | **未测** |
| `discount` | 0.015 | **0.005** | 现金贴现 | 子 agent：0.06 → +725 ns |
| `capital_power` | 0.4 | 0.4 | `rank` 的分母指数 | 子 agent：0.9 → −270 ns |
| `land_rent` | 2 | 2 | `rank` 的地租 | 子 agent：8/20 → −680 ns |
| `risk` | — | — | `val − risk×liquidity` | 被 configure 固定为 1 |

### 7.2 产能/预算
| 参数 | 默认 | config | 实测 |
|---|---|---|---|
| `max_animals` | 20 | **40** | **已上线**：+6.2pp (2670)、+634 (2970) |
| `max_land` | 4 | 3 | 4 更差（−2,889） |
| `max_hands` | 14 | 14 | 20 被 C++ 校验拒绝（合法上限 14） |
| `reserve` | 120 | 120 | 未单独测 |
| `feed_cover` | 2 | 1 | 未单独测 |

### 7.3 执行/搜索（只影响执行器，不改估值）
`labor_hours` 15/10、`work_price` 1.2/4、`animal_work`、`preview`、`delivery`、`intraday`、
`service`、`rotation`（**实测 1 更差**）、`repeat`（**单独完全无作用 = 惰性**）、
`crop_fert`、`harvest_threshold`、`delay_sale`、`opening_budget`、`keep_commitments`、
`tour_dp`、`animal_bias`、`crop_bias`、`replant`（**0.5 已上线，见 4.2**）、`scenario`。

### 7.4 死参数
- **`layout`** —— 在 `Settings` 里声明，**任何地方都没有读取**
- `portfolio_passes` —— 被线上 `scenario=1` 的 `proposals.hpp` 用作候选宽度，不是死参数
- `finite_fertilize`（`Planner::Config`）—— **从未被 Settings 赋值**，所以 `planner.hpp:233` 那条施肥分支是死代码；
  真正施肥的是执行器（`core.p.finite_fertilizer ← crop_fert`，走 `finite_fertilize_due`）

---

## 八、已证否的方向（不要再走）

| 方向 | 证据 |
|---|---|
| 模型化对手未来种植 / 避免互相踩踏 | 预测最不准的 seed 反而**赢更多**（反例判定） |
| 胡萝卜作为第二收入线 | 128 seed 强制 ×10 → +65 ns；与小麦 1:1 替代 |
| 四个象限 | 反向证据 |
| 卖出择时的日内分布 | 按桶摊开净 −5 胜（在 `plan_done` 修好前测的，**结论待重测**） |
| 二分开销（λ 影子价格） | 比硬约束更慢且更不准 |
| `competition=1`（"物理正确"） | 实测更差 |
| 8-seed 诊断当效应量 | `rotation` 在 8 seed 上 +873，64 seed 上 −3,008 |
| 用声明订单量反解成交量 | 末期抛单大量失败；必须用 `last_market_fills` |

---

## 九、开着的线

1. **统一 proposal 评分与最终胜负口径** —— `competition=2` 是经验代理，不等于胜率
2. **候选兼容性特征与 cp168 survivor distribution** —— 见 `STATIC_MODEL_AUDIT_ZH.md`
3. **sale DP 的 +356 margin 为什么不转化胜率**
4. **thomas 的 15pp 缺口**：全谱系基线（2970 段）是 thomas 75.0 / demand 82.8 / melon 82.8 /
   ahmed 89.1 / pipe8 92.2 / herd 98.4 / salemali 100.0 —— **3 个已过 90%**
5. **最终提交物必须重打包并做跨架构 parity**：以当前 `max_animals=40`、`replant=0.5` 的
   Jammy-compatible x86_64 二进制为准，不能复用旧内嵌产物
