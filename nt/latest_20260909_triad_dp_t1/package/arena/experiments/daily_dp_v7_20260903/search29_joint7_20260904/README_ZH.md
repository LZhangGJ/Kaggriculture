# 七对手共同经营策略搜索

本轮使用本地完整 C++ Agent 实时行动；不是从它们的冻结对局动作带中寻找事后反制。目标是同一条宏观计划在不同对手和 seed 下仍好用。

## 当前入口

- 开发/冻结计划：`PLAN_ZH.md`。
- 当前有效输出：`receipts_v2`。`receipts_v1`仅保留第一次编译失败的诊断，不包含搜索结果。
- 运行程序：`run.py`，新增 C++ 联合搜索封装：`joint.cpp`；旧生产源码原样引用。
- 搜索结束后：`selected_plan.json` 为选型冠军；最终独立测试在 `receipts_v2/final_test`，报告 `RESULT_ZH.md`。

## 运行方式（WSL / 本机已有环境）

Python：

```text
/mnt/e/ai_coding/kaggle/kaggriculture/research/team_mate/Kaggriculture_main_512631c/agents/route_clustering_switch_agent/fast_kaggriculture/.venv-bench/bin/python
```

脚本：

```text
/mnt/e/ai_coding/kaggle/kaggriculture/experiments/daily_dp_v7_20260903/search29_joint7_20260904/run.py
```

参数依次为：`build` → `check` → `search` → `select` → `final_test` → `official` → `close`。

已经成功结束的阶段不要重复启动；`build`禁止覆盖，`search`检测旧前沿/完成收据后会退出。其他阶段从已存在的冻结结果读取，不重新挑 seed。

完成后可用 `replay --seed 20268001` 对七个对手各双座位运行选中策略。新增结果保存到独立的 `additional_replays`，不写入原验收。

## 分清几个结果

- 训练搜索：4 seed × 双座位 × 7 对手 = 56 世界，每天评估共同候选，保留4条分支。
- 新种子选型：50 seed，比较搜索最终4条，共2800局；DP27额外700局。
- 最终独立测试：另一批50 seed，仅已选冠军和DP27，各700局，不允许再选其他候选。
- 最终14条完整轨迹与官方1.32.7逐步核对。

候选 `KEEP` 是保留当天基础动态规划，不是原子PASS。计划固定的是每天一个语义配方，执行动作由真实状态决定；不可行配方显式回退KEEP。

## 数据与身份边界

对手索引仅用于离线选择目标和逐对手计分；不会进入我方 Controller。未来终局用于离线评分，不作为部署输入。完整前沿的 `outcomes` 只计算终局指标，`hash_recorded=false`；从头 replay 和验证结果才记录真实719步状态/动作hash，不把未记录hash当作动作去重证据。

首次编译失败仅来自测试代码试图对不可赋值的对手控制器做 move-assignment；已改为调用已有的深拷贝构造器。没有修改原对手策略或搜索评分，旧失败收据保留。

Goal工具因旧目标paused/unfinished无法新建；没有谎称旧目标已完成。本轮按用户明确授权执行并以文件收据追踪。无Kaggle/Git操作。
