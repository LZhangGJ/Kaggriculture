# F3 预测后果实验

入口：PLAN_ZH.md。结果依次见 G0_ACCEPTANCE_ZH.md、G1_ACCEPTANCE_ZH.md、G2_ACCEPTANCE_ZH.md、G3_ACCEPTANCE_ZH.md。

## 实际执行环境

- Windows PowerShell 7.6.5，项目根 E:\ai_coding\kaggle\kaggriculture。
- WSL Ubuntu-24.04；C++ 环境、F3 和 7 对手在 CPU 16线程执行。
- `/home/mitubant/vllm-env/bin/python`：只借用 Python3.12 / PyTorch2.10 cu128，RTX3090 更新参数，不运行vLLM。
- 复用已验收的 F3 修账本动态库与原对手池；不重写环境、不重新编译策略。

## 命令

项目路径以下均为 WSL 路径。调用前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，用于可重复GPU运算。例如在PowerShell：

```powershell
wsl.exe -d Ubuntu-24.04 -- env CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/mitubant/vllm-env/bin/python /mnt/e/ai_coding/kaggle/kaggriculture/experiments/economic_rl_future_aux_20260906/selftest.py
```

按相同调用前缀分别运行：

1. `g1.py generate`：生成成对标签。输出目录已存在时拒绝覆盖。
2. `g1.py learn`：只进行监督学习，训练集/开发集/最终测试seed隔离。
3. `run_g2.py`：冻结协议，依次训练control/aux各两次；所有模型选择冻结后才测最终100seed，自动生成G2/G3报告。

历史结果应保留。重做独立实验需新目录和新协议，不能覆盖本轮结果后再把新结果称为同一次冻结实验。`run_g2.py`仅能跳过完整完成的子运行，不支持从半批次恢复。

## 工程修正记录

G1候选分支已全部生成并验证后，监督学习数据读取发现NPZ字段在逐节点循环中反复解压，且切片保留整份数组引用。第一次学习进程在尚未训练前主动中断。修正为每个归档只读取一次字段数组，再索引节点。未改变任何标签、抽样、划分或训练超参数；未重跑分支，也未读取测试结果来调参。

`g1/PROTOCOL.json`记录生成阶段的脚本hash；G2协议和最终manifest记录这一读取优化之后的版本。因此生成阶段脚本hash与最终g1.py不同是上述已披露的读取修正，并非更换标签。第一步正式学习在修正后运行。

## 结论边界

G1是单节点条件价值，不是在线模型完整对战。G2未来标签来自真实rollout，但后续仍由当时的策略执行，不是理论最佳后续。G3是唯一直接实战证据，训练预算较小、只做两次配对重复，阴性结果不能证明该方向永远无效。
