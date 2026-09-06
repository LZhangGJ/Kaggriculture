# KEEP推理偏置消融：使用与边界

报告：ACCEPTANCE_ZH.md。固定实验定义：PLAN_ZH.md / PROTOCOL.json。

在PowerShell启动WSL内命令：

```powershell
wsl.exe -d Ubuntu-24.04 -- /mnt/e/ai_coding/kaggle/kaggriculture/.venv_wsl_cpp/bin/python /mnt/e/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep_bias_20260906/build.py
wsl.exe -d Ubuntu-24.04 -- /home/mitubant/vllm-env/bin/python /mnt/e/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep_bias_20260906/run.py
```

原目录已执行时会拒绝覆盖，重做请建立新的实验目录。环境、对手和F3执行均在C++、CPU16线程；仅工程验收中用GPU核对模型概率，无训练。

本轮新增动态库 `build/keep.so`。mode=100/101/102/103分别为原加分/2/1/0，mode=0固定使用原F3。mode只用于离线控制推理偏置，不进入模型特征。API与已有C++ arena一致，不是独立Kaggle提交包。

模型沿用economic_rl_future_aux_20260906/training下4个step020；原文件不修改。新头文件只把KEEP常数变为实例调用时设定的thread_local值。Policy与Session字段布局不变，原模拟器和对手接口不重编译。

验收范围：宏观选择满足原候选mask，C++无异常且719步完整结束，1/16线程及原加分完整动作一致。不能将这些检查写成“本轮对所有新轨迹做过官方逐原子动作一致性复验”。

原网络是在3.58先验下训练的。去掉先验后的裸网络分数可能包括对这个先验的补偿，不是经过独立校准的经济价值；也可能受到熵奖励影响。本轮不把裸分数高直接解释成更赚钱。

这次仅测试已学参数的推理阈值。若要验证低KEEP先验是否改善探索与学习，需要从训练阶段改变先验并进行独立对照，不能用此次推理改动代替该结论。
