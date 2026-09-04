# S3D：Fieldbook完整C++接入及多对手基准

日期：2026-09-03。状态：本里程碑通过；自主v7的整体goal未达标。

## 一、结论

用户追加的 [Six-Day Public-State Fieldbook](https://www.kaggle.com/code/yhay81/six-day-public-state-fieldbook) 已进入完整原生C++对手面板，未降级为单条Replay。

- 原公开C++策略按原样编译，不改选择树、路线、阶段资金保护；适配层仅处理观察/动作映射和每局context。
- 10局×719步，与原公开`main.py + agent.so`动作及官方1.32.7状态一致；原生输入投影逐项比较也一致。
- 100局单线程基准与1,000局16线程重复结果相同；两个局面交错推进、完成后换seed/换座位复用context，完整动作/状态轨迹哈希相同。
- 我方策略及官方规则哈希没改，原PASS/G001/G003开发A的1,200条执行记录保持一致。
- 两批各100局开发测试：S3C03对Fieldbook总计49/200胜，暂定物流修复50/200胜。不能宣布稳定提升，更不能宣布整体90%目标完成。

当前真实对手可用3/7：G001、指定G003、Fieldbook。Boatlee V29、Kaito V58、Lynn V5、公开EcoBot V7已下载冻结，完整C++接入仍待完成。PASS另列。

## 二、Fieldbook实际逻辑

冻结版本为`combined_publicstate_r4`。它把719步分成5段：144、144、144、144、143步。各段可选路线数量为1、2、1、3、5。

在144/288/432/576步，用当前市场、我方现金与库存、双方公开盘面等特征选择下一段。288步的选择器当前只有一个结果。选择后，在阶段起点估计该段已知自身动作带需要的现金和物品，适当出售自身库存保护后续支出；其余步执行该段动作。

这是“分段路线选择＋资金保护”，不是每天从零生成经济计划，也没有调用真实未来随机事件。树中有杂草位置统计等公开特征，本次作为冻结对手保留；不把其阈值或路线身份搬进我方自主规划器。

## 三、接入边界与来源

公开产物保留于`opponents/yhay81_six_day/output/`。主要哈希：

| 文件 | SHA256 |
|---|---|
| `main.py` | `5809846fd528f00d1c01d6f58940633604adb7283d1e2f152603697967e3bb89` |
| 原公开`agent.so` | `eb4117082d3927e55276358897b38d06c25d27f4262aeb2a14acb390e77f7437` |
| `sixday_r4_source/policy.cpp` | `a3f04707615b7c04f3ab4a3c593081c98df2dd091be386aa80a3ffad7c00c1b4` |
| 本项目构建的`_dp7_native.so` | `f4f7ad86964a467cd9f20fab96b811e6316092e3533d2ac61f28e68fab21b312` |

实现文件：`native/fieldbook_adapter.hpp/.cpp`、`native/module.cpp`。构建收据记录所有直接使用的原始C++源文件、头文件和本项目输入哈希。

原公开`submission_bridge.cpp`内含进程级全局Session，不能让16线程直接共享。本项目没有把它链接进并行热循环，而是每局调用原始policy的create/act/destroy接口；符号改名仅用于避免和公开参考二进制重名。

适配输入来自`dp7::View`：双方公开农场、市场、商店与本方私有物品。对方库存/种子/随身物品不传入，真实seed不传入。附带的`sim.hpp`只用于原policy类型/估算函数；实际比赛状态转移仍由冻结的`fastkag::Simulator`执行，未替换为该公开方案的模拟器。

特别处理了两套C++类型中的地块枚举及PICKUP/DROP枚举差异；数量缺省遵守官方语义。对照时官方一侧用原始公开入口产生的动作，C++一侧用新适配器动作，避免仅用同一动作推进两边造成自证。

## 四、验收与范围

| 检查 | 规模 | 结果 |
|---|---:|---|
| 原公开入口动作＋官方状态逐步对照 | 10完整局，7,190步 | 0差异 |
| 原Python打包观察与C++投影逐字段对照 | 21,512,480标量比较 | 0差异 |
| 单线程基准 vs 16线程重复 | 100基准局＋1,000复跑局 | 0现金/路径选择/溢出差异 |
| 同线程交错推进与context重置 | 6条完整轨迹哈希 | 0差异 |
| 原候选机制测试 | 546项 | PASS |
| 旧PASS/G001/G003面板回归 | 1,200条执行记录 | 完全一致 |

官方对照seed：20261401、20261402、20261404、20261410、20261422，各双座位。观察到末段路线0/1/2/4，未穷尽所有状态或所有选择树叶子；不能声称全状态形式证明。

1,000局复跑只验证并行隔离，并不是1,000个独立seed。独立速度测试16线程约921完整局/s，包含双方策略和719步比赛，未包含Python启动/资产加载/官方逐步验证；小批计时不代表所有对手都有该速度。

## 五、两批开发结果

开发A：20261401–20261450；开发B：20261501–20261550。每批50个seed双座位，每配置/对手100局。每局重复一次用于确定性检查，不计为新增样本。两批均为开发集，不使用原最终验收预留seed。

| 配置 | PASS现金A / B | G001胜场A / B | G003胜场A / B | Fieldbook胜场A / B |
|---|---:|---:|---:|---:|
| L3_base（S3C03） | 177,100.73 / 180,035.51 | 36 / 44 | 7 / 11 | 29 / 20 |
| L3_return（暂定） | 179,319.03 / 181,590.46 | 40 / 48 | 9 / 11 | 28 / 22 |

每个胜场数字分母均为100。零平局。不要把A/B平均现金超过18万等同于预先冻结50局正式PASS验收。

| 配置/批次 | 对Fieldbook我方平均现金 | Fieldbook平均现金 | 平均分差 |
|---|---:|---:|---:|
| L3_base A | 96,086.97 | 106,289.97 | -10,203.00 |
| L3_base B | 98,104.42 | 109,887.90 | -11,783.48 |
| L3_return A | 96,325.50 | 106,008.47 | -9,682.97 |
| L3_return B | 98,538.95 | 109,483.58 | -10,944.63 |

这说明新增对手确实暴露差距，但不能只凭终局均值判定具体因果。物流修复在G001上增加胜场，在Fieldbook上却A批少1胜、B批多2胜；总计仅多1/200胜，仍无足够证据宣布通用增强。下一轮改善必须在完整对手池报告各项增退。

报告使用按seed聚类的近似区间，因为双座位相关。零经验方差不输出虚假的[100%,100%]正态区间。保守界也依赖随机seed独立假设，不是线上胜率保证。

## 六、收据索引

- [初始4局官方对照](../receipts/fieldbook_official_parity_v1/acceptance.json)
- [新增6局分支对照](../receipts/fieldbook_official_branches_v1/acceptance.json)；两个目录均保存gzip完整轨迹。
- [context/线程隔离](../receipts/fieldbook_context_isolation_v1/acceptance.json)
- [546项原策略测试](../receipts/fieldbook_unchanged_policy_mechanisms_v1/acceptance.json)
- [开发A面板](../receipts/pool_fieldbook_A50_v1/results.json)
- [开发B面板](../receipts/pool_fieldbook_B50_v1/results.json)
- [接入与旧结果核对总验收](../receipts/fieldbook_integration_acceptance_v1/acceptance.json)
- [机器对手索引](../opponents/registry.json)

## 七、运行方式

本机Windows PowerShell调用WSL，构建：

```powershell
wsl.exe -d Ubuntu-24.04 --exec /mnt/e/ai_coding/kaggle/kaggriculture/research/team_mate/Kaggriculture_main_512631c/agents/route_clustering_switch_agent/fast_kaggriculture/.venv-bench/bin/python /mnt/e/ai_coding/kaggle/kaggriculture/experiments/daily_dp_v7_20260903/tools/build_native.py
```

只测试Fieldbook的C++100局（输出目录必须未存在）：

```powershell
wsl.exe -d Ubuntu-24.04 --exec /mnt/e/ai_coding/kaggle/kaggriculture/research/team_mate/Kaggriculture_main_512631c/agents/route_clustering_switch_agent/fast_kaggriculture/.venv-bench/bin/python /mnt/e/ai_coding/kaggle/kaggriculture/experiments/daily_dp_v7_20260903/tools/run_opponent_panel.py --opponents yhay81_six_day --configs /mnt/e/ai_coding/kaggle/kaggriculture/experiments/daily_dp_v7_20260903/profiles/s3c/hauling_configs.json --labels L3_base,L3_return --seed 20261401 --count 50 --repeat 1 --threads 16 --out /mnt/e/ai_coding/kaggle/kaggriculture/experiments/daily_dp_v7_20260903/receipts/NEW_FIELDBOOK_PANEL
```

加入已完成的其他对手，把`--opponents`改成`pass,g001,g003,yhay81_six_day`。请求未迁完版本会报错，不会替换为PASS或旧版。

逐步官方校验脚本：`tools/check_fieldbook_native.py`；线程/重置测试：`tools/test_fieldbook_isolation.py`；总核对：`tools/verify_fieldbook_panel.py`。构建二进制变化后，面板要求重新运行并登记匹配的适配验收收据；不能沿用旧二进制的验收标识。

## 八、后续

继续接入剩余4个明确版本，保留原决策和状态恢复；全部可用后再做生产兑现、资金恢复和市场择时等通用消融。Fieldbook的阶段支出预留可作为机制审查参照，不直接复制其分段路线或决策阈值。保持原两个正式目标，无训练、无Kaggle提交、无Git推送。
