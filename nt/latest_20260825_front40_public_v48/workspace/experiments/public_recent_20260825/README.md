# Public Recent 2026-08-25

本目录记录 2026-08-25 按 Kaggle `Recently Run` 顺序完成的公开 Notebook 扫描、去重、官方 1.32.7 初筛，以及 Kaito V48 的精确 JAX 迁移。

核心结论：本轮 34 份下载中，Kaito V48 / Adaptive Shop Guard / Shops Remember 属于同一个新强力动态路线家族。Kaito V48 六条路线已完成官方 Python 1.32.7 与 JAX 双座位逐动作、逐状态、终局奖励完全一致；在独立随机库对冻结 FC24B 的 1,024 局测试中取得 83.11% 胜率。

入口：

- `reports/RECENTLY_RUN_PUBLIC_SCAN_20260825_ZH.md`：扫描和去重总结；
- `receipts/kaito_v48_jax_parity_seed1302001_n2x2_v3.json`：严格一致性回执；
- `receipts/kaito_v48_vs_fc24b_seed1304001_n512x2_v1.json`：正式强度回执；
- `artifacts/recent_v48_route_bank_v1.npz`：六条新路线加入现有 Bank 后的冻结路线库；
- `tools/`：路线提取、官方 Trace、逐步一致性和 GPU Arena 工具。

原始下载位于 `public_notebooks/recent_latest_20260825_scan_v1`。研究型 Notebook（KagSim、Top12 X-ray、No Yarn、Market 59）是机制证据，不应被误记为可直接参赛的 Agent。

