# Orbit Wars 全部公开金牌方案迁移比较与 RTX 3090 推荐顺序

本页比较官方金牌区第 1–19 名中实际公开的 11 篇 writeup。没有公开文章的第 4、12–18 名不虚构方案；详见 [覆盖台账](WRITEUP_COVERAGE_LEDGER.md)。

## 1. 总结先行

对本机 `i9-12900K + 128GB + RTX 3090 24GB`，最合适的不是独立照搬某一名，而是组合：

> **第 10 名的 Strategic task wrapper + 第 19 名的 GPU candidate tensor + 第 3 名的 semantic feasibility/PFSP + 第 7 名的实验治理**，在现有第 9 名风格的 end-to-end JAX simulator 上训练；稳定后加入第 6 名 edge/search，最后才考虑第 8 名完整 micro-step。

原因：本项目已经解决最昂贵的前置问题——精确 JAX simulator。当前 simulator-only 达 1.56M transitions/s，608K policy+sim 达 188K/s，完整基准 PPO 达 57.3K/s。现在上分瓶颈会从“能否跑 RL”转成“状态、任务动作、对手池和评估是否正确”。

## 2. 方案对比总表

评分是本项目迁移判断，不是原比赛名次的再评分。实现难度 1=低、5=高；上分潜力是对 Kaggriculture 的方向性判断；速度为迁移成本文建议的 1–6M 本地版本后的**未实测估算**，不是原作者速度。

| Orbit 名次 | 核心 | 迁移实现难度 | 3090 适配 | Kaggriculture 上分潜力 | 估算端到端 transitions/s | 100M 热循环 | 最大风险 |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | 200M pure PPO scaling | 5 | 1 | 5（理论）/2（本机复刻） | 3k–15k（缩小版） | 1.9–9.3h | 原路线 15B/B200 不可复现，低层动作样本效率差 |
| 2 | Future sequence + 4.3M ModernBERT | 3.5 | 2 | 3.5 | 8k–30k | 0.9–3.5h | 极简模板形成策略盲区；原训练 8×H100/10B |
| 3 | Schedule/reachability tensor + PFSP | 4.5 | 4 | 5 | 8k–30k | 0.9–3.5h | 可行性 kernel/候选错误会系统性误导 policy |
| 5 | BC + IMPALA/V-trace + teacher | 5 | 3 | 4 | 5k–25k | 1.1–5.6h | 数据覆盖、actor lag、单卡系统复杂度 |
| 6 | 小 edge Transformer + league + search | 4 | 5 | 4.5 | 10k–40k | 0.7–2.8h | 重 feature 工程；搜索可能只过拟合内部对手 |
| 7 | JAX PPO + top-k + 严格消融 | 4 | 4 | 4.5 | 8k–30k | 0.9–3.5h | 若只评自家模型仍会 self-play 过拟合 |
| 8 | 自回归 micro-steps + search | 4.5 | 5 | 4.5 | 5k–20k | 1.4–5.6h | 联合动作 credit assignment 和多次推理成本 |
| 9 | 全 JAX rich future PPO | 4.5 | 4 | 5 | 10k–40k | 0.7–2.8h | rich features 吞吐/显存；4P 类似的评估不稳对应市场循环 |
| 10 | ScheduleCache + StrategicEnv | 3 | 5 | 4 | 15k–60k | 0.5–1.9h | strategic wrapper 太粗会限制最终上限 |
| 11 | 一致 IL + PFSP/cross-run league | 4.5 | 3 | 4 | 8k–30k | 0.9–3.5h | replay 风格/覆盖、league 计算分散、IL anchor 锁死 |
| 19 | GPU all-candidate + BC/PPO/calibration | 3.5 | 5 | 4.5 | 15k–60k | 0.5–1.9h | 全候选张量易爆；simulator SPS 不等于 learner SPS |

热循环时间只等于 `100M / SPS`，不含 JIT、arena、checkpoint 和开发。实际一个正式 100M 实验建议预留 3–8 小时；300M 约半天到一天；1B 约 1–3 天，取决于最终模型和评估密度。

## 3. 逐方案迁移评价

### 第 1 名：最高天花板，最低本机优先级

可迁移：通用 entity Transformer、target-level action、规则 parity、规模曲线、previous-best gate。

不可照搬：200M、15B steps、2400 B200-hours、低层连续动作探索。本机应只做 1→3→8M scale test；如果 3M 已被结构化 2M 模型击败，就没有理由追求“大而通用”。

### 第 2 名：最快得到能玩的 RL baseline

多时间尺度 future sequence 很适合植物/动物生命周期；极简 template action 也能快速学。但它很可能只学会一种高频生产循环。适合作为表示与吞吐 baseline，不适合作为最终 action ceiling。

### 第 3 名：样本效率与上限最平衡

Schedule-feasibility tensor 把 Kaggriculture 最难的“走过去是否来得及、材料在哪、维护是否冲突、能否入仓卖出”直接提供给 policy，同时仍让模型选战略。工程较重，但每一项都与规则可验证，适合单卡用工程换算力。

### 第 5 名：有数据时强，但不先做

IMPALA 适合大量 CPU actors + GPU learner，而本项目 JAX simulator 已在 GPU 极快。新建异步栈只有在端到端 JAX 的 policy/update 成为瓶颈时才值得。BC 可独立复用，IMPALA 不必绑定。

### 第 6 名：最符合“本机金牌范式”

原作者前半就在 3090 开发，2.5M 模型配 edge features，后期才加 A100。迁移后还可让本地 JAX 并行搜索。它的推理搜索是便宜的最后 5–10% 增强，但必须在多样对手面板验证。

### 第 7 名：不是可选模型，是默认研发流程

Top-k+random 候选、future auxiliary、历史池和单因素 round-robin 应从第一天执行。近 200 个 100M 实验不能原样复刻，但“一天 2–4 个可信实验”在本机可能实现。

### 第 8 名：最强联合动作表达，后期开放

Kaggriculture 同回合多单位+有序市场最需要自回归账本。它也最容易降低 SPS 和增加 bug。先用一次 trunk + GRU heads；只有 independent task heads 已成为明确瓶颈时再升级。

### 第 9 名：长期训练框架首选

本项目现有 `gpu_sim` 已经是其核心系统路线。下一步不是“再做全 JAX”，而是把 future/calendar/task masks 和 PPO 接到已验收规则 core。用第 3/7 名稀疏化，避免复刻其数十亿步才收敛的形态。

### 第 10 名：最快做出第一版 learned strategic agent

ScheduleCache 和 job wrapper 开发清晰、吞吐高、易排错。它应是第一个 implementation milestone；之后可以逐步把过粗 template 拆成第 3/8 名的语义/自回归动作。

### 第 11 名：解决“只会一种 replay 风格”

它证明少量一致数据比冲突 broad mix 好，也证明 IL-only 可以很强。但其原 cross-run league 计算巨大。本机只需 main learner + 固定 specialists，不需要同时训练 5 个大模型。

### 第 19 名：已经完成一半

其最大贡献是 GPU simulator，而本项目这一步已完成且本地测得更高 simulator-only SPS。剩余价值是 GPU candidate tensor、BC 架构赛、PPO 和 frozen-checkpoint bias calibration。它与第 10/3 名天然组合。

## 4. 推荐实施顺序

### 顺序 0：立即启用第 7 名的实验治理

冻结 seed panel、座位交换、opponent families、promotion 公式、run receipts。任何模型之前先有可信 arena。

### 顺序 1：第 10 + 19 名，建立最快闭环

实现 ScheduleCache、job state、candidate tensor、合法动作编译器；0.6–1M 模型，先 BC/规则轨迹做 sanity，再 30M PPO。目标不是金牌强度，而是能稳定种植、维护、入仓、卖出且 100% legal。

### 顺序 2：第 3 名，升级为主建模路线

实现 schedule-feasibility tensor、semantic modes、edge bias 和 PFSP history pool。2–4M，100M 单因素实验。它是最可能在单卡上获得高样本效率的主线。

### 顺序 3：第 9 名，补 rich future/calendars

逐个加入 plot maintenance calendar、plot future、logistics calendar、market future；每项单独 100M 消融。只保留 tournament 证明有用的通道。

### 顺序 4：第 6 名，edge 精炼与窄搜索

在主 policy 稳定后做 distributional value、triggered 6–24-turn search。搜索开/关做 2048+ games。它适合最后上分，而不适合掩盖基础 policy 弱。

### 顺序 5：第 8 名，联合动作表达

若日志显示 independent jobs 无法协调 hands、同回合种植或市场订单，再加轻量 micro-step GRU；先 policy-head-only，不重跑 trunk。

### 顺序 6：第 11 名，coverage-aware IL/league

当官方高分 replay 积累且已完成策略覆盖账本时启用。一致风格 BC models 也可作为 opponent specialists，而不必都作为 main 初始化。

### 顺序 7：第 2 名 future-sequence 作为结构消融

其 Conv1D future encoder 可与第 9 名 calendar 对打；极简动作只保留为快速 baseline。

### 顺序 8：第 5 名 IMPALA

只有 profile 证明 synchronous JAX PPO 因 learner/actor 耦合不能充分利用 3090，且 replay BC 明显有效时才投入。否则 V-trace 系统复杂度不划算。

### 顺序 9：第 1 名 scaling

在所有结构路线稳定后做 1M/3M/8M controlled scaling；不规划 200M/15B 复刻。

## 5. 推荐的第一个可训练版本

### Observation

- 72 plot + 8 masked unit + 9 market + 1 global tokens；
- `d_model=128`，4 layers，4 heads，约 1–2M；
- time features：+6/+24/+72 和终局；
- unit-task top-k：hard tasks + model top4 + random4。

### Action

- 每 turn 0–2 个 job edits；
- 任务：continue/water/plant/harvest/depot logistics/feed/care/build/place/sell/buy/hire/land；
- modes：MINIMUM/SAFE；
- 最多 2 个 market macro slots；
- deterministic compiler 输出官方 raw action。

### Training

- 1024 env 起步，rollout 64；
- 0.6–1M smoke 30M transitions；
- 1–2M 正式 100M；
- terminal W/D/L，small auxiliary future losses；
- recent frozen 50%、history 25%、v16 15%、specialists 10%；
- 每 10M checkpoint，固定 512 games/matchup，晋级候选 2048。

### Go/no-go gates

- feature/action parity 通过；
- raw action legal rate 100%；
- hard maintenance candidate recall 100%；
- end-to-end PPO ≥15k transitions/s；
- 对 deterministic baseline 座位平衡胜率显著 >50%；
- 对至少四个策略族无大幅回退。

## 6. 最终推荐

如果只能选一个“文章方案”开始：**第 3 名**，因为它最能用工程提高 3090 的样本效率。

如果选一个最符合本机经验的完整范式：**第 6 名**，小模型、edge、league、搜索，原作者也从 3090 开发。

如果选一个最匹配现有代码的训练系统：**第 9 名**，因为 JAX simulator 已经完成。

实际执行则按：

> **7（治理）→ 10+19（闭环）→ 3（主模型）→ 9（future features）→ 6（搜索）→ 8（联合动作）→ 11/2（数据与结构补强）→ 5（必要时异步化）→ 1（受控扩模）**。
