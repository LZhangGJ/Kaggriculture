# F3 KEEP=2 经济决策RL：实验过程、1000轮结果与轻量交接

日期：2026-09-06。接在同仓库 [C3/F3/J7+C3 C++底座包](../latest_20260906_c3_f3_j7c3_search/README_ZH.md) 之后。不修改旧包，不是Kaggle可提交包。

## 先看结论

四组F3日级经济PPO均完成1000轮，每组224,000训练局，合计896,000局。300→1000新增627,200局；每100轮报告；末尾再做40,600局独立测试。

|独立未见seed，7对手平均|1000轮确定性|1000轮采样|开发选定轮数|开发选定模型确定性|
|---|---:|---:|---:|---:|
|KEEP基座|62.93%|不适用|不适用|不适用|
|普通PPO重复1|61.79%|62.71%|200|65.36%|
|普通PPO重复2|64.21%|63.57%|180|62.29%|
|未来预测辅助PPO重复1|71.57%|68.64%|640|71.71%|
|未来预测辅助PPO重复2|64.64%|64.07%|780|65.64%|

辅助第一组的改善在独立seed上仍存在，但第二组没有同等改善，不能称稳定复现。第一组300轮在同批独立seed上已达70.14%，1000轮71.57%，增益区间包含零；并非继续加轮数就持续变强。普通PPO两次1000轮确定性平均63.00%，接近KEEP。

最强一组1000轮确定性仍仅54.00%胜G003、50.50%胜Six-Day，未达全对手高胜率。没有按终测挑选新的组合Agent，没有线上提交。本包给出真实结论，包括失败结果。

## 阅读入口

1. [实验全过程与关键判断](PROCESS_ZH.md)：首轮C3/F3/J7+C3、修账复跑、候选价值、辅助预测、KEEP调整与长训。
2. [1000轮完整报告](archive/experiments/economic_rl_keep2_1000_20260906/FINAL_SUMMARY_ZH.md)：每100轮趋势、独立胜率、现金、逐对手和配对区间。
3. [全部报告索引](REPORT_INDEX_ZH.md)：包含所有归档的Markdown报告，按实验目录排列。
4. [独立测试原始汇总及配对区间](results/independent1000/RESULTS.json)，以及每个配置下的 `games.json.gz`。
5. [关键模型索引](MODEL_INDEX.json)、[导出来源及原始SHA-256](SOURCE_PROVENANCE.json)、[本包验证](PACKAGE_ACCEPTANCE.json)。

## 实验本身是什么

- F3从标准第0天自主开局，无J7开局表。KEEP表示保留F3当天生成的经营方案，不是PASS。
- 每天最多10个候选：KEEP、5种作物、3种动物、暂缓一个未投入项目。仅改接口选出的一个项目；候选mask包含规划器约束，不是全官方合法空间，也不是旧Candidate8宽池。
- 70,402参数MLP actor/critic；全局128维、候选32维。普通/辅助各两次训练重复。
- 辅助组额外1,935参数头，学1/3/7天后的经营后果，权重0.1；仅训练用，不导出到C++。它不是读取未来的线上输入。
- KEEP的logit加分固定2；PPO学习率3e-4、4 epochs、minibatch512、gamma1、lambda0.95、clip0.2、entropy0.01。
- 每轮16个新seed × 7个实时对手 × 双座位 = 224完整局；每局719步，30次日规划，Day29仅KEEP。模型更新56次/轮。
- C++16线程运行环境、F3、对手与MLP推理；GPU完成PPO更新。这里不是当前模型自我对弈，而是固定七对手池训练。
- 终局奖励 `sign(我方现金-对手现金) + 0.1*tanh(资金差/50000)`，输局同样进入PPO更新。
- 开发seed 69800000起32个；每20轮选模。固定监测70300000起100个，只汇报，不据此改策略；最终71100000起100个，训练/选模结束后才测试。
- 300→1000严格恢复末轮权重和Adam，保持一阶/二阶状态，计数16,800→56,000；不从开发最佳回滚再训。

## 包含与排除

包含：

- 10个历史实验目录及400–1000七阶段的代码、报告、配置和验收收据。
- KEEP=2每100轮的四组 `.bin`（100–1000轮）、开发选定旧模型；300轮、1000轮和最终开发选定模型的 `.pt`（含Adam）。准确名单见 `MODEL_INDEX.json`。
- 每轮训练统计、开发选模历史；最终与阶段评测的逐局结果使用无损gzip压缩，而不是只留下胜率。
- 原始冻结C++策略源码及仅修改include路径的可移机推理源码；已有模拟器/七对手直接复用同仓库旧包。

不包含：

- `rollout_*` 的大量逐节点数组、`decisions.npz`、完整Replay、每轮所有中间模型/Adam、编译产物、虚拟环境、JAX/Torch缓存、原机巨型搜索缓存。
- 本机全量数据没有删除；仍在原 `E:/ai_coding/kaggle/kaggriculture/experiments/` 中。

**归档边界**：`archive/` 是原实验的原样证据和代码，不是可从本包一键重放的全量训练工作区。其中原 `MANIFEST.json`、路径、旧报告里的“本轮没有晋升”等均保留历史口径，可能指向有意未上传的数据；它们不是当前轻量包的文件清单。当前包使用根目录 `PACKAGE_MANIFEST.json`。完整旧审计需要原机被省略的rollout。轻量包能独立核对所携带结果、查看所有训练统计，并通过下面入口复现模型对战。

## 队友如何运行

Linux/WSL2、g++13、Python3.12；推理无需GPU或Torch。先进入同仓库的底座包：

```bash
cd nt/latest_20260906_c3_f3_j7c3_search
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python build.py --jobs 3
cd ../latest_20260906_keep2_rl_1000
../latest_20260906_c3_f3_j7c3_search/.venv/bin/python build_policy.py
../latest_20260906_c3_f3_j7c3_search/.venv/bin/python validate_portable.py
```

底座已编译就不必重复编译；换Python版本须重新构建扩展。四核心CPU可把底座编译并行设2；评测线程可设4。不要复制本机旧 `.so` 代替重编译。

```bash
# 默认1000轮辅助第一组，2 seed × 七对手 × 双座位 = 28局
../latest_20260906_c3_f3_j7c3_search/.venv/bin/python evaluate.py --out runs/aux1000_greedy
# 采样模式
../latest_20260906_c3_f3_j7c3_search/.venv/bin/python evaluate.py --mode sample --out runs/aux1000_sample
# 同一批seed的KEEP基座
../latest_20260906_c3_f3_j7c3_search/.venv/bin/python evaluate.py --mode keep --out runs/keep
# 完整重测一个模型1400局；这批seed已用于本报告，不是又一次独立测试
../latest_20260906_c3_f3_j7c3_search/.venv/bin/python evaluate.py --model aux_r0_step640 --seeds 100 --out runs/aux640_full
```

输出目录必须不存在，防止覆盖。其他权重ID见 `MODEL_INDEX.json`。移机验收使用终测原有2个seed，对29配置共812局核对双方现金、胜負、步数及执行计数；结果见 [PORTABLE_ACCEPTANCE.json](PORTABLE_ACCEPTANCE.json)。这是结果复现检查，不是新独立强度验证，也不是全部动作/状态的形式证明。

验证本轻量包的哈希与保存结果无需GPU、无需编译器、仅需标准Python：

```bash
python3 verify_package.py
```

需要继续训练时，先参考归档的 `train1000.py / common1000.py / bias2_model.py / aux_model.py` 以及 `.pt` 的状态；原训练脚本带原机目录和全量审计要求，**不能直接把目录替换后就声称精确续训**。本次没有新增或测试移机续训驱动。

## 本包不做的事

不改变任何经营策略、不重新训练、不按新测试挑更好模型、不新增候选Oracle，不自动提交Kaggle。此前C3/F3包中的“尚未得到改进PPO”是当时实验结论；本包补充后来KEEP=2长训的结果，不改写旧证据。
