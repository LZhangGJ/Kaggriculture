# V1 / V2 对战队友最新提交

用户要求：下载队友最新两个版本，让 V1、V2 各对每个版本打 200 盘。测试对象为 V306 与 V463；现已全部完成，共 800 盘正式测试，零错误，全部到 719 步。另有 8 盘烟测和 6 盘串行复跑，双方动作哈希及现金 14/14 完全复现；这些重复局不计入正式胜率。

| 对手 | V1 | V2 |
|---|---|---|
| V306 | 119/200，59.5% | 123/200，61.5% |
| V463 | 110/200，55.0% | 123/200，61.5% |

无平局。详细现金、配对区间与耗时见 [REPORT_ZH.html](REPORT_ZH.html) 和 RESULTS.json。V463 串行抽查 V1/V2 最大非首步耗时分别为 1.108/1.214 秒，均有超过一秒预算的状态；本次未按超时判负，不应当成严格超时沙箱胜率。

2026-09-24 已发现的来源：

- Git 分支 `handoff/rl-student-v306-kaggle-20260924`，冻结提交 `6ef0f6d1a9ee8036df70cc34863a4a3328d2650e`。
- 已下载 `submissions/student-v306-kaggle.tar.gz`，SHA-256 与队友文档一致；Kaggle 提交 ID 56511458。
- 同一 Git 快照有 V285 模型，但它是训练快照，没有独立 Kaggle 提交原包。
- V463 原包由用户提供本地路径 `gpt_review/gpt_code/gpt-6-dp/student-v463-funded-kaggle.tar.gz`，对应 Kaggle ID 56517329，10,640,087 字节，SHA-256 `7a8fcabeb04096e330bacc3e6d6acb42a9cb3b93567c0bfb1b4cb1de05092445`，与 Git HANDOFF 的原包哈希一致。
- V463 的 25 个文件原样导入，未重新打包或替换权重。与 V306 对比，16 个同路径文件完全一致（含原生执行器 `.so`、路线资源和配置）；模型、入口/模型适配源码更新，新增 `FundedReplayRoute` 开局雇工资金保护。因此两个原包的差异不只在权重，不能把胜率变化全部归因于 RL 训练。

V1 是上一轮保留的 `cf_liq_h12_nointraday`，V2 是未晋级的 `v2_animal06`，均原样冻结，不进行训练或参数调整。

每个对手使用 100 个新环境种子，交换两个座位；V1 和 V2 使用完全相同的环境种子、座位、对手采样 RNG。环境种子不传给策略。采样 RNG 单独生成并记录，保持队友原包的 `sample=True`。

裁判为本地冻结的官方 1.32.7 Python interpreter，与上一轮融合验收相同。双方均实时行动；没有使用对手 replay 动作磁带。当前测试不实施 Kaggle 超时沙箱。

每局审核完整 719 步、对手 17 个模型决策日、0 fallback、0 illegal。异常不计作普通败局。源码和裁判哈希位于 PROTOCOL，逐局结果位于 runs。烟测、串行复跑独立保存，不计入 200 盘。

复跑环境：WSL x86_64，`/home/mitubant/vllm-clean/bin/python`，PyTorch 2.9.1+cu128（实际使用 CPU、单线程），NumPy 2.2.6。

```powershell
wsl.exe -- /home/mitubant/vllm-clean/bin/python /mnt/e/ai_coding/kaggle/kaggriculture/experiments/fusion_vs_teammate_latest_20260924/run.py run --opponent student_v306 --workers 16
wsl.exe -- /home/mitubant/vllm-clean/bin/python /mnt/e/ai_coding/kaggle/kaggriculture/experiments/fusion_vs_teammate_latest_20260924/serial_checks.py --opponent student_v306
wsl.exe -- /home/mitubant/vllm-clean/bin/python /mnt/e/ai_coding/kaggle/kaggriculture/experiments/fusion_vs_teammate_latest_20260924/run.py run --opponent student_v463 --workers 16
wsl.exe -- /home/mitubant/vllm-clean/bin/python /mnt/e/ai_coding/kaggle/kaggriculture/experiments/fusion_vs_teammate_latest_20260924/serial_checks.py --opponent student_v463
python experiments/fusion_vs_teammate_latest_20260924/report.py
```

报告脚本只汇总已完整完成的 400 盘面板。检查 RESULTS.json 中的 opponent 列表，不要把只有一个对手的结果理解为两个对手都已完成。
