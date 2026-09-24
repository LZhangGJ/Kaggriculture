# C++ candidate critic teacher ABI v2

> 2026-09-22。仅用于离线诊断/数据生产；不进入提交，也不修改正式
> `policy/r1/agent.so`。当前 ABI 只接受首次接管 `step=288/hour=0`。

## 冻结契约

`policy/r1/teacher_bridge.cpp` 复用正式 R1 的 `Handle`、候选生成、
`portfolio_features()` 和 scorer，没有替代实现。

- candidate schema：`r1_portfolio_features_v1`，356 维；字段名 SHA-256
  `9af5b62e20c54c34b785134f08399d69e32603dfad166918b0813f1b7b970e38`。
- context schema：`r1_first_handoff_causal_v1`，2233 维；字段名 SHA-256
  `6386bae567618f5c8409075b420dc5dbff3bb616b236bb714aacde151beff3a8`。
- key：`proposal_key()` 的 signed int32 序列，以 little-endian 编码后取 SHA-256
  前 128 bit。`prepared_index` 只在一次 prepare 生命周期内有效。
- family：空 diagnostic name 才是 `normal=0`，`outer*` 是 `outer=1`，其余为
  `diagnostic=2`。诊断候选不进入 normal top-K。
- 每行 19 个标量：index、id、family、is_outer、short score、canonical/selection
  reference、4×raw Q、4×canonical advantage、4×selection advantage。
- canonical anchor 固定为 normal `candidate_id=0`；会随 top-K 变化的 selection reference
  及其 advantage 只作诊断。raw scenario Q 是训练真值。

字段名来自同一真实候选上 `portfolio_features(..., naming=true)`，不是手工列表；每个候选都会
检查 names/width 一致。outer 现在也使用相同 observation、base normal 和 candidate id 生成完整
356 维特征，旧的 `features={}` 已修正。C++ 与 writer 都会对 schema fail closed。

## 两条路径

`td_teacher_prepare_features_observation()` 只运行 observation preamble、
`SearchController::prepare(v,true)` 和特征生成；不调用 `score()`，不生成 outer/diagnostic。
它返回全部线上 normal，可查询 id/key/features 后直接 `td_install(index)`。

`td_teacher_batch_observation()` 是离线标签路径：选择 short-score top-K normal、强制包含
canonical id0，再加 top-K outer，并计算四个终局闭环 Q。四个情景是未校准 support，不是带概率
的分布：point forecast、flush lower、feasible high-stock flush、capacity-limited carry。
`short_score` 只能用于 audit/teacher selection，禁止作为 student 输入。

context 包含 step/剩余时域、原始 ledger lower/upper/valid/成交量、own/rival `SaleClock`
三日逐小时状态、`CropClock` ring/count/cursor、own `book[100]` 和 `JointBundleState`；不含 seed、
对手 identity 或对手 private。

当前 writer 必须保持 `context_complete=false, training_ready=false`：完整 canonical public frame
尚只通过源轨迹及 packed observation hash 引用，executor pending/queue 尚未导出；step288 后由 R1
自身形成的 commitment 也不能靠 `observe_external()` 恢复。`step!=288` 直接拒绝。

## 自检证据

- binary：`work/agent-teacher-feature-v2.so`
- SHA-256：`7f9c8f039df3a608656b575aecf75ea4c2907af699bb1c0737b26c331ed6245c`
- result：`work/teacher-feature-v2-final-thomas-seed200-seat0.json`
- packed frame：3070 float64，SHA-256
  `abc63bbcf334d29342c9ebcdbd1f90cf3e46271bc016e6c9d05d97fd30ef0f0c`
- shape/schema hash、feature/context 重复确定性、完整 teacher 重复确定性、reference pairing：PASS；
  `step=289`：fail closed。

同一真实 Thomas warm 状态的单次时延：feature-only 0.187s，旧 short-score prepare 8.723s，
4-scenario scoring 8.786s。候选+特征不是 8–9 秒瓶颈；未来 critic 可绕过 short scorer，而
scenario scorer 只留在线下。`td_teacher_batch_many` 当前只接受 `batch=1`，跨状态须一进程一状态。

```bash
g++ -std=c++20 -O3 -DNDEBUG -shared -fPIC -pthread \
  -DR2_STARTUP_SUPPLY_MODE=2 -DR2_LOCAL_SALE_TIMING=1 \
  -DR2_FINITE_FERTILIZER=1 -DR2_CROP_CLOCK_MODE=1 \
  -DR2_OBSERVE_PUBLIC_TRADES=1 -DR2_MARKET_INTEGRAL=2 \
  -DR2_SALE_CLOCK_MODE=0 -DP16_WORKING_CAPITAL_GATE=1 \
  -DP16_LIVE_REMAINING_VALUE=1 -DT3_OBLIGATION_REPAIR=1 \
  -DT3_CAPACITY_REPAIR=1 -DT3_RECEIPT_REPAIR=1 \
  -DP16_JOINT_BUNDLES=1 -DR2_SCENARIO_CARRY_RIVAL=1 \
  -DR2_SCENARIO_FIXED_VALUE_CLOCK=1 -DR2_OPTIMIZER_AUDIT=1 \
  -DR2_OUTER_NEIGHBOR_AUDIT=1 -DR2_OUTER_BEAM_WIDTH=2 \
  -Ipolicy/r1 policy/r1/teacher_bridge.cpp \
  policy/r1/executor/vendor/simulator.cpp -o work/agent-teacher-feature-v2.so

/root/miniforge3/envs/torch-npu/bin/python experiments/audit_teacher_batch.py \
  --binary work/agent-teacher-feature-v2.so \
  --trajectory <warm.jsonl.gz> --step 288 --top-k 1 \
  --check-repeat-batch --output <new-diagnostic.json>
```
