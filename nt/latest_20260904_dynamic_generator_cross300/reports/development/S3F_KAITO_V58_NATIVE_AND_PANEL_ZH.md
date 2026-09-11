# S3F：Kaito V58 完整 C++ 迁移与多对手回归

日期：2026-09-03。结论：**对手迁移通过，整体 v7 目标未通过。完整 C++ 对手由 4/7 增至 5/7。**

本轮没有改我方经营逻辑、参数或官方规则，没有训练模型、提交 Kaggle 或推送 Git。上一轮仅核对状态的回答不计作开发进展；本轮实际完成源码迁移、差分测试和新对战证据。

## 1. 迁移了什么

来源为冻结的 Kaito V58 Minimax Closed Loop：

- 原码：`../opponents/kaito_v58/output/main.py`。
- 原码 SHA256：`b041058ec187a8d0a01edc0eab8de068b53deca3e6c1973faf74ace6916ddcb9`。
- C++：`../native/kaito_v58.hpp`、`../native/kaito_v58.cpp`，接入 `module.cpp`。
- 数据：`../native/kaito_v58_frozen.json.zlib`，93,674 字节。
- 数据 SHA256：`470c1d931059961e73ab50a8dc8d68d0f559ab850f917c84d4425e3db75048ce`。
- 本轮最终二进制 SHA256：`d509f06be8bf057d485a7e1c4f27a811edfc2d534d92feb4ad97225b4c6b6fec`。

实际热路径是 `V58 router → 10 个独立 residual controller → sparse planner → weed repair → route proposal`，不是把压缩包中所有旧代模块都运行一遍。十套每套均为 719 步：backbone、yarn、pet、recovery、smoothie、clone、known_yarn、ice_minimax、bakery_yarn、pizza_recovery。

保留的行为包括：

- 根据商店、公开资金/产能及持续盘面相等状态切换完整路线。
- 每条路线都有独立状态，即使未选中也按真实观测更新；不按它自己的虚构农场更新。
- BUILD_PASTURE/PLANT 遇杂草先 DIG，下一步重试，再有限延迟该单位的动作；其他单位和市场照常。
- 公开市场供给推断、近镜像置信度、提前出售、从未来原定销量扣回。
- 原 SELL 槽重排、必要时销售前置，以及最终步按公开风险排序清仓。
- 保留原版的保守库存投影；没有擅自换成信息更完整的模拟器投影。

原版 `defer_enabled` 与 `market_maker_enabled` 均关闭，继续关闭，加载时明确检查。Kaito 的固定公开状态指纹是冻结对手原逻辑，仅存在于对手模块，不加入我们的自主规划器。

Python 分十次创建控制器并补放缺失的观测；C++ 一开始创建十个独立状态，每步同步更新。每个 Python 子控制器开始可见时，它与 C++ 已读到同一观测前缀。差分测试逐步比较全部已创建子控制器，而不是仅比较最终选中的动作。

## 2. 发现并修复的一处迁移错误

初次对照在 20261401/座位0/第529步发现：未选中的 ice_minimax 子控制器原订单数量为字符串 `"4"`，通用原生导入器只处理整数，错误读成缺省1。

修复限于 Kaito 数据导入：按官方的 `int(quantity)` 语义读取；未改其他对手导入器，也未改原始源码/数据。比较器区分“原始 JSON 类型”与“官方实际执行数量”，字符串4和整数4等价，但4和1仍不等价。原版按字符串订单推进官方状态，C++按整数订单推进原生状态，逐步结果仍须相同。

失败收据保留在 `kaito_original_official_probe_v1/failure.json`。修复后的历史四局在 `kaito_original_official_parity_v2`；最终构建又重新验证，不拿旧二进制报告冒充新构建通过。

## 3. 验收证据与边界

最终阶段汇总：[kaito_integration_acceptance_v1/acceptance.json](../receipts/kaito_integration_acceptance_v1/acceptance.json)。

| 检查 | 结果 | 能证明什么 |
|---|---|---|
| 当前构建 4 局原版/官方逐步对照 | 719 步/局，动作、子状态和双方状态零差异 | 覆盖正常路线、YARN切换、杂草、抢卖和扣回 |
| 当前构建 2 局同版本镜像对照 | 719 步/局，零差异 | 覆盖近镜像置信度与残差交易；不声称这两局触发最终clone路线 |
| 976 个公开状态路由夹具 | 零差异，覆盖10个输出路线 | 对原版实际路由函数做条件/边界对照；不是976局强度样本 |
| 100 局串行＋1,000 局16线程复跑 | 零差异 | 每局状态隔离；后者只是重复50个seed双座位，不增加独立样本量 |
| 6 次完整轨迹哈希 | 交错、重置、换座位均一致 | 防止跨局/跨座位污染 |
| 原生机制回归 | 546 项通过 | 既有机制未因接入破坏 |
| 原 PASS/G001/G003/Fieldbook/Boatlee A/B面板 | 4,000条执行记录逐项不变 | 我方逻辑、原对手与原规则未退化 |

当前构建总计6场官方差分、4,314个环境步骤；每步还比较所有已创建子控制器的发出动作、供给记忆、置信度、欠售账本和杂草状态。官方实际对照为冻结1.32.7解释器，**不是最终 Kaggle 完整沙箱/超时框架验收**。

这些是有限样本的差分验收，不是穷尽所有合法盘面的数学证明。自然对战样本主要触发 backbone/YARN；其余路由分支由独立合成夹具覆盖，不能混称成自然对战覆盖。

Fieldbook、Boatlee也在新二进制上各补4场官方回归及1,000场隔离复跑，旧收据和旧二进制仍保留。当前入口见 `registry.json`。

## 4. 强度：我方对 Kaito V58

沿用已有开发A：20261401–20261450；开发B：20261501–20261550。每批50个seed双座位=100局。每组复跑第二遍只检查确定性。

| 我方配置 | A胜场 | B胜场 | 合计胜率 | 我方均值现金 | Kaito均值现金 | 平均分差 |
|---|---:|---:|---:|---:|---:|---:|
| L3_base / S3C03 | 20/100 | 16/100 | 36/200，18.0% | 97,288.83 | 110,302.25 | -13,013.42 |
| L3_return / 暂定物流修复 | 20/100 | 17/100 | 37/200，18.5% | 97,527.24 | 109,993.47 | -12,466.23 |

按seed聚类的近似95%胜率区间分别为10.4%–25.6%、10.9%–26.1%。这是开发集估计，不是线上总体保证。增加1胜不支持宣布物流修复明显变强，也没有新候选晋级。

同一批次的当前综合结果如下，每个真实对手每配置200局：

| 对手 | L3_base | L3_return |
|---|---:|---:|
| G001 | 40.0% | 44.0% |
| G003 | 9.0% | 10.0% |
| Fieldbook | 24.5% | 25.0% |
| Boatlee V29 | 20.5% | 21.5% |
| Kaito V58 | 18.0% | 18.5% |
| Lynn V5 | 未迁完/未测 | 未迁完/未测 |
| 公开 EcoBot V7 | 未迁完/未测 | 未迁完/未测 |

PASS两批均值和原结果不变：基础177,100.73/180,035.51，暂定物流179,319.03/181,590.46。两项正式目标依然不能宣告通过，预留最终验收seed未使用。

原始完整面板：[A](../receipts/pool_kaito_A50_v1/results.json)、[B](../receipts/pool_kaito_B50_v1/results.json)。

## 5. 执行速度和使用

16线程纯C++热循环，包含我方决策、Kaito十套控制器和719步规则结算：1,000局复跑1.859秒，约538整局/秒；实际对照面板约526–546局/秒。对应约38.7万环境步/秒，不能与只跑规则转移或GPU训练吞吐混用。数字不含Python启动、源码解压和编译；编译本轮约19秒。

PowerShell示例；输出目录必须尚不存在：

```powershell
$taskPy = '/mnt/e/ai_coding/kaggle/kaggriculture/research/team_mate/Kaggriculture_main_512631c/agents/route_clustering_switch_agent/fast_kaggriculture/.venv-bench/bin/python'
$taskRoot = '/mnt/e/ai_coding/kaggle/kaggriculture/experiments/daily_dp_v7_20260903'
wsl.exe -d Ubuntu-24.04 --exec $taskPy "$taskRoot/tools/run_opponent_panel.py" --opponents pass,g001,g003,yhay81_six_day,boatlee_v29,kaito_v58 --configs "$taskRoot/profiles/s3c/hauling_configs.json" --labels L3_base,L3_return --seed 20261401 --count 50 --repeat 2 --threads 16 --out "$taskRoot/receipts/NEW_UNIQUE_PANEL"
```

复验脚本：`check_kaito_native.py`、`test_kaito_router.py`、`test_kaito_isolation.py`、`verify_kaito_panel.py`。每项报告包含本次构建和源文件哈希。

## 6. 下一步

1. 完成Lynn V5完整包和公开EcoBot V7原生接入，仍禁止用旧版本或单条Replay代替。
2. 七个对手齐备后，继续已冻结的饲料承诺/生产兑现等通用机制消融；不回到G001单对手选型。
3. 当前新证据显示：新增对手普遍压制我方，不能靠小幅物流修复就宣称自主规划器足够强；要用实际现金和生产链诊断决定下一项改动。
4. 保留所有最优候选和失败分支，正式双目标未达标则goal继续。
