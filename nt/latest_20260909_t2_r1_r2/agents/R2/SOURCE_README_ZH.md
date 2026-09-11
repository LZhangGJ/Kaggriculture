# Kaggriculture T2-R2 BatchedDelivery

2026-09-09。基于 T2-R1 的合并卸货与共享仓容修复。

**定点六场败局全部救回；新增 224 局为 214 胜（95.54%），与 R1 总胜率持平。新面板救回三局，也丢掉三场原胜局。不是已证明全面胜过 R1 的版本。**

默认：`batch_delivery=1`、`service_reconcile=2`、`live_ledger=0`。包含源码、动态库、七个冻结对手、R1 对照、逐局结果及构建验收。

```bash
python3 verify_package.py
python3 -m pip install -r requirements.txt
python3 build_all.py --jobs 2 --cxx g++
python3 run.py --tag r2_local224 --start 2609098000 --count 16 --threads 4
```

只构建策略：`python3 build_policy.py --cxx g++`。它执行 8 条完整轨迹回归，不自动跑对战。

R1 同种子对照：

```bash
python3 run.py --tag r1_local224 --baseline --start 2609098000 --count 16 --threads 4
```

每对手 16 seed × 双座位＝32 局，全七对手每版本共 224 局。主 runner 不接受超过 16 个 seed，避免误跑 1400 局。

入口：`policy/agent.py::agent`。**R2 参数数量为 38；请成套使用 policy/，不要只换 .so。**

详见 [完整报告](REPORT_ZH.md)、[逐局配对结果](T2_R2_paired224.csv)、[源码补丁](T2_R1_to_R2.patch)。`evidence/` 是随 R1 继承的历史证据；本轮证据在 `evidence_r2/` 和 `runs/*validation224/`，不可把历史中止的大面板当成此次新测试。
