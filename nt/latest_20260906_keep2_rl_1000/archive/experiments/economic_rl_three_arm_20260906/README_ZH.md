# 三组小MLP经济决策RL：代码和结果入口

项目目录：`E:\ai_coding\kaggle\kaggriculture\experiments\economic_rl_three_arm_20260906`。

## 先看哪些文件

- `G3_FINAL_REPORT_ZH.md`：首轮训练及独立评测总报告（正式实验完成后生成）。
- `FINAL_RESULTS.json`：每组两次训练、配对胜率/现金/分差区间、性能。
- `FINAL_AUDIT.json`：来源哈希、所有训练局/评测局、奖励/概率/种子隔离审计。
- `PLAN_ZH.md`：冻结的预算和实验边界；首行是设计时状态，不是最新运行状态。
- `G0_ACCEPTANCE_ZH.md`、`G1_ACCEPTANCE_ZH.md`、`G2_ACCEPTANCE_ZH.md`：阶段验收。早期报告的“尚未完成”指该阶段当时状态。
- `training/<arm>_r0`、`training/<arm>_r1`：全部21个checkpoint、训练数据、开发选择与最终对战结果。
- `baselines`：原版KEEP、均匀随机候选、未训练随机策略三种对照。
- `logs`：顺序执行日志及退出码；失败探针`checks_v1/v2`保留，不可当作最新验收。

## 三组究竟学习什么

|代码名|基础经营计划|底层动作执行|RL决定|
|---|---|---|---|
|c3auto|C3完整自主投资，无J7|C3|保持／改一个未投入项目／当天暂缓一个项目|
|f3|F3从第0天自主，无J7|F3|同上|
|c3j7|C3基础计划＋原有条件J7配方参考及回退|C3，不播放J7底层动作|同上|

重要限制：“全自主”指没有J7经营参考，不代表MLP从零独立决定全农场经济。三组都是原生规划器提供基础计划，RL学习有限的每日调整。尚未训练联合扩地、雇工、销售和多人路径；不能把本轮结果解释成这些能力的RL上限。

128维全局＋32维候选，共用两层128隐藏单元打分器，独立critic，总计70,402参数。最多10个候选；KEEP带固定初始化先验。每天一次，约30次宏观决策对应719次环境推进。

## 实际运行方式

C++一次跑完整双方实时对局，并在CPU执行MLP推理；16工作线程。整个批次完成后，将数组交给PyTorch，在RTX3090更新PPO，再导出下一批固定权重。没有每一步Python往返，没有在训练批次中途换模型，也不使用事后Oracle。

固定对手为G001、G003、Boatlee V29、Kaito V58、Lynn V5、Six-Day、Three-Day。每批224局＝7对手×16新seed×双座位。每组两次独立训练，每次20批＝4480局；开发/最终seed分离。

WSL依赖使用已有环境，没有安装/升级包：

- 原生构建和评测：`.venv_wsl_cpp/bin/python`，Python3.12、numpy、pybind11。
- GPU学习：`/home/mitubant/vllm-env/bin/python`，PyTorch2.10.0+cu128。仅借用Python依赖，不启动或修改vLLM服务。
- g++、OpenMP，以及BUILD_RECEIPT.json列出的原本地原生对象文件。

这不是可上传Kaggle的提交包。构建依赖本地冻结源码和对手资产；源码位置/哈希见SOURCE_FREEZE.json。已有模型权重可由本实验C++ runner直接运行。

## 单独评测一个现有模型

在项目根目录PowerShell执行（输出目录必须尚不存在）：

```powershell
wsl.exe -d Ubuntu-24.04 -- /mnt/e/ai_coding/kaggle/kaggriculture/.venv_wsl_cpp/bin/python /mnt/e/ai_coding/kaggle/kaggriculture/experiments/economic_rl_three_arm_20260906/evaluate.py --arm f3 --checkpoint /mnt/e/ai_coding/kaggle/kaggriculture/experiments/economic_rl_three_arm_20260906/training/f3_r0/step020.bin --mode greedy --seed-start 65000000 --seeds 16 --threads 16 --out /mnt/e/ai_coding/kaggle/kaggriculture/experiments/economic_rl_three_arm_20260906/extra_eval_f3_65000000
```

`--mode sample`按模型概率探索；`--mode keep`为原版，不要求checkpoint；`--mode random`为均匀随机可行候选。不要把random/keep评测数据送进PPO：它们记录的模型分数不是该行为的采样概率。训练接口已拒绝非sample数据。

模型选用只看`best.json`记录的开发集选择，不根据最终测试集改选checkpoint。step000代表未训练模型。如果开发集选中step000，就不能称训练超过了原版。

## 从头复现实验

不要在本目录直接重复run_all.py：输出使用禁止覆盖模式，会报错以保护证据。应在同一项目的experiments下建立新的独立实验目录，复制本实验根目录源码，不复制training/baselines/logs/checks等输出。然后按顺序：

1. 原生Python运行build.py（调用prepare.py，复制原版源码，生成配置并编译）。
2. 原生Python运行checks.py，再运行recheck_features.py。
3. PyTorch Python运行learn_checks.py。
4. PyTorch Python运行run_all.py：六次训练→三种对照→总报告。
5. 原生Python运行audit_results.py，进行最终数据/哈希审计。

训练期间不可修改冻结的模型/runner/策略源码或SO。学习率、候选语义、训练预算或KEEP先验的修改均属于新实验，必须新编号，并保持独立最终测试集，不能悄悄改写本次结论。
