# KEEP=2 PPO：续训至总计300轮

## 目的与边界

从四个200轮检查点及其 Adam 状态继续100轮，判断增加训练量能否改善实际胜率。普通 PPO 与带后果预测辅助任务的 PPO 各有两次训练重复。不能根据单次训练胜率判断提升；最终比较在同一批新种子上的200轮与300轮模型，并包含全KEEP基准。

本实验不扩大候选，不改奖励、不改KEEP强度、不改F3执行器，不进行事后Oracle搜索，也没有Kaggle提交或Git推送。此前最佳检查点全部保留；300轮不自动成为推荐版本。

## 运行

在项目根目录的 PowerShell 中：

```powershell
wsl.exe -d Ubuntu-24.04 -- /home/mitubant/vllm-env/bin/python /mnt/e/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_300_20260906/run_all300.py
```

首次运行前须完成 `preflight300.py`。正常中断后，主程序会读取已保存进度续跑；不要删除进度文件或同时启动第二份。部分评测文件存在但未完整写完时，先检查残留文件，而不是手动宣称完成。

训练及终测全部完成后执行同目录 `audit300.py`；最终文档完成后执行 `freeze_manifest300.py`。冻结后不要覆盖已审计结果，进一步实验另建目录。

## 依赖和模型格式

- `economic_rl_keep2_100_20260906`：冻结的 C++ 引擎、7对手、F3候选/执行器、网络实现和KEEP=2运行时。
- `economic_rl_keep2_200_20260906`：此次续训的200轮模型、优化器、历史开发集选模记录。
- CPU以16线程模拟完整719步，C++同时进行策略推理；RTX3090执行PPO梯度更新。
- `.pt` 是包含 `model`、`optimizer`、`step`、`bin_sha` 的字典，不是裸权重。
- `.bin` 只包含70,402个actor/critic浮点参数。KEEP=2在运行时实现，不包含在二进制文件内；必须配合100轮目录的 `build/keep2.so`。辅助预测头只在训练使用，不计入线上actor/critic参数。
- `selected` 可以指向历史100轮或200轮目录。这是按固定开发集选择的结果，不是漏训。

## 核心文件

|文件/目录|用途|
|---|---|
|PLAN_ZH.md / PROTOCOL.json|预先固定的训练量、种子、对照、代码与输入哈希|
|G0_ACCEPTANCE.json|恢复权重及优化器、C++/Torch概率一致性检查|
|training/*/step300.pt、step300.bin|四个300轮模型|
|training/*/PROGRESS.json|逐轮训练进度和诊断|
|training/*/selection.json|0–300轮历史开发集选择记录|
|FINAL_SELECTION_FROZEN.json|在最终评测之前冻结的模型选择|
|final/*|相同新种子上200轮、300轮、开发选定模型和KEEP的对局及决策记录|
|FINAL_RESULTS.json / ACCEPTANCE_ZH.md|胜率、配对差异、逐对手和耗时|
|FINAL_AUDIT.json / BEHAVIOR_DIAGNOSTIC.json|独立证据审计及决策分布|
|SUMMARY_ZH.md|面向使用者的结论，实验完成后生成|
|MANIFEST.json|最终文件大小和SHA256清单|

新训练种子与以前训练种子、开发集、所有最终评测集均隔离。每版本最终评测是100个新种子×7对手×双座位=1,400局。21种配置共29,400局，但模型选择只使用开发集。置信区间按种子分组，不能把同局多天决策视为独立对局样本。
