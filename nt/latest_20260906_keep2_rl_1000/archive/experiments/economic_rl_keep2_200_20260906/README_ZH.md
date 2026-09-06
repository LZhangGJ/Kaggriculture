# KEEP=2，从100轮续训至200轮

本目录是续训实验，不覆盖 `../economic_rl_keep2_100_20260906`。

## 结果入口

- `PLAN_ZH.md`、`PROTOCOL.json`：训练前冻结的预算、种子和选模规则。
- `G0_ACCEPTANCE.json`：四组模型权重、Adam状态恢复一致性和真实更新预检。
- `TRAINING_ACCEPTANCE.json`：四组续训完成后生成。
- `FINAL_SELECTION_FROZEN.json`：最终测试前冻结的开发集选模结果。
- `ACCEPTANCE_ZH.md`、`FINAL_RESULTS.json`：全部新种子对战结果，完成后生成。
- `FINAL_AUDIT.json`：训练/终测全量记录、优化器计数、C++/Torch一致性及旧文件保护审计。
- `SUMMARY_ZH.md`：验收完成后人工归纳的简明结论；未完成时不存在。

## 续训语义

四个运行：`control_r0`、`control_r1`、`aux_r0`、`aux_r1`。

每组都从旧目录的 `step100.pt` 恢复完整模型及Adam，而不是从开发最佳轮数、初始模型或仅权重开始。普通组不使用辅助损失，辅助组权重仍为0.1。

每组新增100轮×224局=22,400局，累计200轮、44,800局。每轮4个epoch、14个mini-batch，所以Adam计数从5600连续增加到11200。

新训练使用旧训练seed区间之后的1600个seed/重复，7对手×双座位。开发集仍是69800000起32seed；最终测试换成70100000起100seed，旧69900000终测不进入训练或本次选模。

没有添加早中晚阶段标签，没有改候选或执行器，没有改变KEEP加分、学习率或奖励。这样100→200轮比较反映继续训练，而不是夹带新的策略修改。

## 模型与兼容

每组 `training/<name>` 中：

- `step100.bin/.pt` 是原文件的逐字节复制。
- `step101`至`step200`为新训练保存的模型和优化器。
- `rollout_NNN/input.json`记录生成该批数据的 `step(NNN-1).bin` sha。
- `selection.json`包括旧0–100轮和新增120–200轮的开发评测。
- `best.json`可能指向100轮以前；权重的真实位置以 `FINAL_SELECTION_FROZEN.json` 为准。

`.bin`仅含70,402个float32 actor/critic参数，**不含KEEP加分或辅助头**。必须搭配旧100轮目录的 `build/keep2.so` 或 `bias2_model.py`，不能用KEEP=3.58的旧运行器冒充相同结果。

`.pt`是包含 `model`、`optimizer`、`step`、`bin_sha` 的字典，辅助头保存在其中。C++内核仍负责7个实时对手、F3执行和规则模拟；网络更新使用RTX3090。目录依赖原项目，不是独立Kaggle提交包。

## 命令

从项目根目录的PowerShell执行，使用已有WSL Ubuntu-24.04和Torch环境：

```powershell
wsl.exe -d Ubuntu-24.04 -- /home/mitubant/vllm-env/bin/python /mnt/e/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_200_20260906/run_all200.py
```

只在确认没有同名运行器正在执行时启动。脚本会跳过已完成组，并从未完成组的完整检查点恢复；不要同时启动第二个运行器写同一目录。

完成后的只读数据核查（会写新的审计收据，但不训练、不跑新对局）：

```powershell
wsl.exe -d Ubuntu-24.04 -- /home/mitubant/vllm-env/bin/python /mnt/e/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_200_20260906/audit200.py
```

不要重复执行预检脚本覆盖已有检查。若要改参数、特征或数据，建立新的实验目录。

本实验不推Git、不提交Kaggle，也不自动替换F3基座。
