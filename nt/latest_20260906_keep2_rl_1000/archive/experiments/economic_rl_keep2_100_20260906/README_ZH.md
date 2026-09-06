# KEEP=2、100轮PPO验证入口

- 结论：`SUMMARY_ZH.md`
- 冻结协议：`PLAN_ZH.md`、`PROTOCOL.json`
- 全部数字与配对区间：`ACCEPTANCE_ZH.md`、`FINAL_RESULTS.json`
- 工程验收：`G0_ACCEPTANCE.json`、`FINAL_AUDIT.json`
- 行为统计：`BEHAVIOR_DIAGNOSTIC.json`
- 开发选择冻结：`FINAL_SELECTION_FROZEN.json`

## 模型与数据

`training/control_r0`、`control_r1`、`aux_r0`、`aux_r1`保存各次训练：

- `step000.bin`至`step100.bin`：C++使用的actor/critic 70,402个float32参数。
- `stepNNN.pt`：完整模型、辅助头和优化器，用字典中的`model`、`optimizer`恢复，不是裸state_dict。
- `rollout_NNN`：每轮224局及30条/局日级决策；`input.json`记录产生这些数据的前一轮权重sha。
- `selection.json`、`best.json`：仅按开发集选模，不是终测后选最高分。

`final`保存21个评测版本，每个1,400局。普通与辅助组都保留第20轮、第100轮的greedy/sample及开发选定greedy；不因同分省略失败配置。

## 重要兼容约束

**`.bin`只包含网络参数，不包含KEEP加分。** 必须配合本目录`build/keep2.so`/`policy.cpp`，或者Torch的`bias2_model.py`使用。直接放到旧KEEP=3.58运行器会改变策略，不能叫相同模型成绩。

这仍是10候选的单项目F3 Overlay，不是旧Candidate8宽候选，也不是完整自主经营策略。C++负责F3、7个实时对手、规则模拟和采样；Torch MLP更新使用RTX3090。

## 检查命令（PowerShell工作目录为项目根）

```powershell
wsl.exe -d Ubuntu-24.04 -- /home/mitubant/vllm-env/bin/python /mnt/e/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_100_20260906/audit100.py
```

该命令只核查已保存证据并重生成审计json，不启动新训练和新对局。需要上述WSL环境及既有项目依赖；此目录不是脱离项目可运行的独立提交包。

本次由`run_all.py`编排，`train100.py`执行100轮，`report100.py`汇总。不要为了看结果再次执行`run_all.py`：已完成训练会跳过，但它会重写总耗时收据。若要复现实验，应先建立独立实验目录并冻结新协议，不能覆盖本次证据。

本次没有推Git或提交Kaggle，原F3基座未替换。
