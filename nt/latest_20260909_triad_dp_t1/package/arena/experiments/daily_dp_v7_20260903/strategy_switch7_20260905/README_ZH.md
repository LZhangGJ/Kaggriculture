# S9 恢复入口

> 最终状态：Goal已完成。所有学习切换均未通过未见seed晋升门，最终保留本轮范围内最强的`J7_03`。入口和官方逐步验收见`final_local_best_v1/`，总审计见`FINAL_GOAL_ACCEPTANCE_ZH.md`与`FINAL_GOAL_AUDIT.json`。队友节奏和动态三日计划特征的失败消融均保留，不接入最终Agent。下文保留的是执行过程记录。

当前Goal：对1,494套经营策略作行为聚类和代表质检，按队友方法训练最多一次盘面切换，完成七对手独立验收。先读 `PLAN_ZH.md`。

## 历史状态快照（已由页首最终状态取代）

**当时的阶段判断：S9新切换器没有产生更强Agent。** 最终S9选择不切换，38.86%总胜率/21.5%最弱，对照J7_03为58.86%/27.5%，不得晋级或声称更强。后续已完成策略树、检查日和计划特征补充审计，并正式保留J7_03结案；以页首及`FINAL_GOAL_ACCEPTANCE_ZH.md`为准。

P3A粗筛和P3B训练已完成。32后缀粗筛12,544次续跑后保留38/205/846/528/1495五个互补选项；32个新训练seed的15,680次续跑已完成，浅树已生成`switch_learning_v1/nodes.json`。事后可赢398/448不是实际树胜率。当前session47756执行`collect_select`，另32个新seed选择首次切换的检查点组合。后续为冻结选型→100新seed最终对战→观察入口/官方一致性。不能重启已结束的83395、17287、41134，也不能编辑冻结的switch_learning.py。

当前运行：unified exec session **83395**，`switch_learning.py coarse`，不要重复启动。已完成：44271采样83,664局；58693聚类29族；83776代表复验7,168局；20532全动作审计编译；48227全动作审计896局，均exit0。P2正式通过，见`P2_ACCEPTANCE_ZH.md`与`clustering_v1/P2_ACCEPTANCE.json`。32个代表全部保留，按新seed最弱对手优先选开局38，最弱6/32、总83/224，不能说它已很强。粗测开局38、7检查日、32后缀、4发现seed双座位七对手，总12,544后缀续跑。后续阶段尚未执行。

有效采样版本：`receipts_v2`。`receipts_v1`只保留初次编译Result名称冲突的失败日志和源码，不含仿真结果。

运行环境：WSL Ubuntu-24.04，冻结Python：

```text
/mnt/e/ai_coding/kaggle/kaggriculture/research/team_mate/Kaggriculture_main_512631c/agents/route_clustering_switch_agent/fast_kaggriculture/.venv-bench/bin/python
```

在本目录依次运行（已经完成的阶段不要重新执行）：

```text
python run.py build
python run.py check
python run.py collect
python cluster.py cluster
python cluster.py validate
python quality_review.py
```

`build/check`已完成；P0报告见 `P0_ACCEPTANCE_ZH.md`。全量采样按16套一块，每块必须同时有rows、daily与hash收据后才跳过；已有完整块可续跑，半块必须先检查，不能静默覆盖。

`clustering_v1/library.json`已冻结29族和32代表。阈值0.10的seed半分ARI=0.97957，轮廓系数0.71386，最大族19.41%，不预定类别数。新seed最近家族无变化。全83,664局指定生产意图无效果0，代表896局所有实际单位动词无效果0，hash与现金全等。平均每局4.0025次不可用recipe回退，保持软兼容性诊断。阈值选择是可复核启发式，不冒充统计真理。

`switch_learning.py`已适配S8的粗筛、精标签、分组CV浅树、单次检查序列选型和七对手最终测试。9项纯数据单元测试通过（session65296 exit0），含Oracle不重复计算多个检查点的同一个世界；尚未用新数据实测。必须有真实质检完成的 `clustering_v1/P2_ACCEPTANCE.json`（`status=PASS_REPRESENTATIVE_QUALITY_REVIEW`、`accepted_indices`、`opening`）才能启动；不能为了跑通伪造该验收。P3标签、P4模型和部署复验尚未执行，当前不是已完成Agent，不存在最终七对手胜率。

待质检通过后的命令：

```text
python switch_learning.py coarse
python switch_learning.py collect_train
python switch_learning.py train
python switch_learning.py collect_select
python switch_learning.py select
python switch_learning.py final_test
```

若coarse无额外胜局，不要强行越过门：分析利润/风险价值是否有证据，否则以最佳固定Agent继续最终评测。部署适配已写入本轮 `local_agent.py`、`online_bridge.cpp`、`verify.py`，但还未编译/验收。它保留S8 ABI与唯一动态执行器，只精简保存实际使用的S9计划；不能冒充已验证提交。

最终选择冻结后执行：

```text
python verify.py build
python verify.py official
python verify.py report
```

`quality_review.py`只生成复核证据，不自动盖章；`coarse_rescue_harm.json`与train/select对应表分别报告每个后缀、检查日、对手的救回和伤害。原始策略数、树分支数和每局允许切换次数是不同概念：本轮可以保留多个后缀目标，但每局最多切换一次。

`deep_quality.py build/audit`已完成每个保留代表的2个验证seed双座位七对手trace审计；重复执行需先看done，不能重建覆盖。训练集Oracle增益下界不正时不训练树；`keep_fixed`只允许在真实无收益证据下冻结固定基线，不能绕过正确性失败。`switch_learning.py`现在已按SHA冻结（e7b76e0aae33a4dac2181ac79697c6d677260a4c1e63066fd1a23226df81786e），不要边采样边编辑。P0冻结的run.py/profile.cpp/PLAN_ZH.md、聚类脚本也保持原样。

原S8不修改；原节点重组Oracle仅编译、未启动，保持暂停。七实时对手、规则和底层执行器不变。最多16线程，无Kaggle/Git操作。
