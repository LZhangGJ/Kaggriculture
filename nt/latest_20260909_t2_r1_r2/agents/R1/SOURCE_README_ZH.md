# Kaggriculture T2-R1 ServiceAligned

**先交付修复版。未完成 1,400 局，不宣称独立测试达到 90%。**

固定版本：`service_v1` 源码 + `service_reconcile=2`，对应同种子 112 局 **106 胜（94.64%）**；原 T2 89 胜。救回 19、退步 2。五个重点案例救回四个。详细边界和逐局证据见 [报告](REPORT_ZH.md)。

## 使用

```bash
python3 verify_package.py
python3 -m pip install -r requirements.txt
python3 build_all.py --jobs 2 --cxx g++
python3 run.py --tag r1_local112 --start 2609096000 --count 8 --threads 4
```

策略入口 `policy/agent.py::agent`，配置 `policy/config.json`。必须一起使用本包包装器与动态库：新参数 ABI 为 37 个 double，不能沿用 T2 的 33 个参数。

只编译策略：`python3 build_policy.py --cxx g++`。构建时验证 5 条固定轨迹（3595 动作），通过后才发布，不自动执行 112/1400 局。上述 `run.py` 比赛命令由使用者明确执行。

`--baseline` 指原 T2；七个实时对手源码/必需资产在 `arena/`。`T2_to_R1.patch` 是相对原 T2 的策略修改；`T2_R1_paired112.csv` 是完整同种子对照。

本包默认未开启模式 3 和账本实验。其他分支结果只在 `evidence/` 留档。大面板已中止，部分完成结果置于 `evidence/stopped_tests/`，不能当成完整或独立验收。

受测策略二进制 SHA256：`ead6b00d982e798fa570756859304ad40d77aa9a6a278e0ca2d3c1a9140f9b44`。
