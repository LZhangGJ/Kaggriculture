# F3 RL预测账本修复验收 — 2026-09-06

结论：本次针对的预测账本错误已修复，回归通过。未续训、未调超参、未提交；没有新的胜率提升结论。

## 错误与修复

旧RL接口在替换/取消未投入项目时，用当前价格重新计算动物旧收益，或只计算作物第一轮，然后从总预测扣除。它不一定等于原规划器实际加入的预测。重复种植、前后价格变化、动物维护选择变化都会使这种“事后重算再扣除”不可靠。

修复后：

1. 原规划器保留每个项目实际加入的完整30天产出/投入流，包含重复种植和饲料、肥料消耗。
2. 项目重新分配地块时，这份预测跟随项目移动，不能只按原坐标查找。
3. 已有在产资产及其延续预测单独保存，不因取消一个新项目而被删除。
4. 替换时移除旧项目的整份实际记录，只加入一份新记录，并据此更新总预测和预测价格。
5. 候选在各自副本上修改；KEEP不重新算原计划。跨日重新建账，旧日的编辑被拒绝。

这是日级投资计划的**预测账本**，不是重写实际现金/库存/官方结算规则，也不是宣称整个经济规划器已完美。高层候选范围、PPO超参、MLP特征与权重、C3两组均未更改。

## 验收证据

|检查|结果|
|单元及生命周期断言|11,077项通过|
|产业/日期/重复种植测试|98份完整流，18个会触发旧重复周期错误的案例|
|替换与撤销|覆盖作物/动物互换、取消再恢复、重复取消、A→B→A；不残留、不重复入账|
|原版KEEP一致性|7对手×4seed×双座位＝56局，40,264步完整动作和终局现金全部一致|
|随机投资接口|56局全部完成，1,205次项目修改，预测重建断言均通过|
|线程/跨局隔离|同14局的1线程与16线程完整动作流一致|
|官方1.32.7复验|14局、10,066步；双方状态逐步一致|
|旧实验保留|旧源码、旧SO、旧结果和checkpoint未覆盖|

原问题的构造回归：小麦/胡萝卜完整计划29份，旧撤销只扣2份；番茄14份只扣7份；草莓12份只扣8份；甜瓜12份只扣6份。现在取消后，该项目在预测中的剩余贡献均为0。这些是预测流测试，不是实际库存或实战现金。

## 性能

相同KEEP工作负载、16线程、每批112局，四轮交替新旧顺序计时：

- 旧版：中位85.13局/秒。
- 修复版：中位82.81局/秒。
- 本次小样本约2.7%的吞吐开销；不是长期性能定论，也不是GPU/PPO训练吞吐。

## 入口及复现

修正版源码：`f3_policy.cpp`、`stage/f3/agent.cpp`、`stage/f3/project_forecast_ledger.hpp`。

修正版库：`build/f3.so`。**旧实验runtime仍保留旧版路径，以保证历史结果可重现；不能用旧入口误称已使用修复版。** 本目录`evaluate_fixed.py`明确加载新库。

项目根目录PowerShell示例（输出目录不可已存在）：

```powershell
wsl.exe -d Ubuntu-24.04 -- /mnt/e/ai_coding/kaggle/kaggriculture/.venv_wsl_cpp/bin/python /mnt/e/ai_coding/kaggle/kaggriculture/experiments/economic_rl_f3_ledger_fix_20260906/evaluate_fixed.py --mode keep --seed-start 65010000 --seeds 4 --out /mnt/e/ai_coding/kaggle/kaggriculture/experiments/economic_rl_f3_ledger_fix_20260906/example_keep
```

原生构建及验证依次运行`build_fix.py`、`regression.py`。回归输出不允许覆盖，重新运行应使用独立实验副本；本目录依赖旁边冻结的`economic_rl_three_arm_20260906`及其七对手资产。

原训练checkpoint的权重格式仍兼容，`--mode greedy/sample --checkpoint <路径>`可以加载做诊断；但修复改变了候选后果，旧胜率不能套用，旧rollout不可当作修复版的同策略训练数据。后续训练需要新的实验编号和新生成的rollout。

机器可读证据：`BUILD_RECEIPT.json`、`UNIT_TEST_RECEIPT.json`、`NATIVE_REGRESSION.json`、`OFFICIAL_REGRESSION.json`、`PERFORMANCE.json`、`ACCEPTANCE.json`。完整动作和逐局结果在`regression/`。
