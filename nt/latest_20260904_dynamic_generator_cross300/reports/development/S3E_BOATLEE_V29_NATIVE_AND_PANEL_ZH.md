# S3E：Boatlee V29完整C++迁入与开发面板

2026-09-03。本阶段通过；自主v7整体goal仍未完成。本轮只新增对手，不修改我方候选与规则。

## 1. 实际完成

- 来源：[Boatlee V29-R1 Adaptive Market Hysteresis](https://www.kaggle.com/code/boatlee/v29-r1-adaptive-market-hysteresis)。
- 保留全部生产动作、杂草修复、后期自适应出售、市场压力记忆、累计追加额度及近镜像抑制。
- 14局×719步，与原公开Python动作、持久状态和官方1.32.7解释器逐步一致。
- 100单线程基准局＋1,000局16线程复跑一致；另6条完整轨迹验证交错执行、换seed/换座位重置。
- 原候选546项机制检查通过；旧PASS/G001/G003/Fieldbook A/B面板3,200条复跑记录不变。
- Fieldbook在新二进制下追加4局官方回归及1,000局线程隔离，继续通过。

当前完整C++真实对手4/7：G001、G003、Fieldbook、Boatlee V29。剩余Kaito V58、Lynn V5、公开EcoBot V7已冻结源码，原生接入未完成。PASS另列。

## 2. 源码与实现语义

原源码SHA256：`c4a6964cec3c1c99207c32bb1fd91e53c3ec01e6890da5734331cbeab1cc1267`。

静态数据资产：`native/boatlee_v29_frozen.json.zlib`，SHA256：`ec4d6d6a83a6f1cef9f0257c659029afa8a7c5ad540738e2896ba2dffd00ce3a`。

本阶段二进制SHA256：`20bf072dd7e35167adc17e3e0a799fd58c1876cd73a945dcf89c05495d18635c`。

`tools/export_native_boatlee_v29.py`只静态解析AST、解压动作和导出参数，不执行下载代码。原表720条，包括未在正常对局执行的末条；实际调用0–718，共719步。最初导出器错误地要求长度719，检查拒绝后改为核对720并保留全部原索引，未改原表或平移开局。

热路径在`native/boatlee_v29.hpp/.cpp`：只接收公开农场、市场、商店和自身私有物品的View，无真实seed、对手隐藏库存。参数与动作数据从冻结资产读取，每局拥有独立State，常量Data只读共享。

特别保留原版行为：

- `_FR_ITEMS=()`，旧front-run逻辑不能触发，其还款账本也不会变为非空；导出器和原生加载器均拒绝擅自启用此分支的不同版本。
- 遇杂草时先DIG、下一步补原意图，再延迟回放8步。不是本项目另设计的恢复策略。
- 456步起，对草莓/牛奶/羊毛检查库存、需求、市场压力和额度，再补充出售。
- 289步锁定一次近镜像判断；原版镜像额外出售预算为0，不替对手开启。
- 市场压力来自库存变化扣除自己估计销量和城镇消耗；它是原版估计，不改成用真实成交或偷看对手订单。
- 市场满10条时原版可能没有插入新订单却仍消耗追加额度，这一行为也保留。
- 仅Boatlee编译单元使用`-ffp-contract=off`，保持Python浮点乘法、加法分步舍入；没有改变我方或官方引擎的编译选项。

首次C++构建缺少`<stdexcept>`导致编译失败，补齐头文件后成功。没有引擎或策略补丁被用于掩盖对照差异。

## 3. 验收覆盖

| 验证 | 样本与结果 |
|---|---|
| 普通对局 | seed20261401/02双座位，共4局；动作、状态记忆、官方状态0差异 |
| 杂草场景 | seed20261404/11/39双座位，共6局，其中3局触发恢复；0差异 |
| 镜像场景 | seed20261401/02双座位，共4局；近镜像锁定为真、额外销量为0；0差异 |
| 线程与重置 | 100基准＋1,000并行重复；6条交错/重置轨迹哈希；0差异 |
| 旧对手回归 | A/B共3,200条旧面板记录完全不变 |

官方侧用原公开Python产生的对手动作，原生侧用C++翻译动作，两边各自推进；并逐步比较价格、双方农场、双方私有状态、商店、时间。额外比较对手内部pressure/added/last_inventory/last_sold/weed恢复状态，浮点值亦一致。

14局是10,066步有限样本，不是所有合法状态的形式证明；未进行完整Kaggle沙箱/超时规范的最终发布验收。当前通过的只是原对手迁移和回归门。

16线程1,000局热循环约1,122完整局/s，包含我方决策、Boatlee决策和719步规则执行，不包含Python启动和资产加载。重复局用于确定性，不当作1,000个独立强度样本。

## 4. 两批完整开发测试

A：20261401–20261450；B：20261501–20261550。每批每配置/对手50seed×双座位=100局。复跑第二遍仅检查确定性，样本量不翻倍。原最终预留验收集未使用。

| 我方配置 | Boatlee胜场A | Boatlee胜场B | 两批合计 |
|---|---:|---:|---:|
| S3C03 / L3_base | 13/100 | 28/100 | 41/200，20.5% |
| 暂定L3_return | 15/100 | 28/100 | 43/200，21.5% |

| 配置/批次 | 我方现金均值 | Boatlee现金均值 | 平均分差 |
|---|---:|---:|---:|
| L3_base A | 92,833.12 | 103,226.57 | -10,393.45 |
| L3_base B | 105,407.69 | 111,317.41 | -5,909.72 |
| L3_return A | 93,227.43 | 102,950.72 | -9,723.29 |
| L3_return B | 105,873.10 | 110,895.38 | -5,022.28 |

其他结果保持不变，两批合计每对手200局：

| 配置 | G001 | G003 | Fieldbook | Boatlee V29 |
|---|---:|---:|---:|---:|
| L3_base | 40% | 9% | 24.5% | 20.5% |
| L3_return | 44% | 10% | 25% | 21.5% |

不能将两场胜场差解释为稳定提升；A/B差异本身说明必须保留多seed。也不能根据新对手胜率低就直接断言“只需抢卖即可修好”。我方现金差还可能涉及投资规模、生产兑现、库存和时机，需要后续实际成交/生产链归因。

## 5. 文件、命令与后续

- [静态导出收据](../receipts/native_boatlee_v29_static_export.json)
- [普通官方对照](../receipts/boatlee_original_official_parity_v1/acceptance.json)
- [杂草对照](../receipts/boatlee_weed_official_parity_v1/acceptance.json)
- [镜像对照](../receipts/boatlee_mirror_official_parity_v1/acceptance.json)
- [隔离检查](../receipts/boatlee_context_isolation_v1/acceptance.json)
- [开发A原始结果](../receipts/pool_boatlee_A50_v1/results.json)
- [开发B原始结果](../receipts/pool_boatlee_B50_v1/results.json)
- [总核对验收](../receipts/boatlee_integration_acceptance_v1/acceptance.json)

对照目录保存gzip完整轨迹。重新运行必须用新输出目录，不能覆盖历史。

本机PowerShell调用C++面板：

```powershell
wsl.exe -d Ubuntu-24.04 --exec /mnt/e/ai_coding/kaggle/kaggriculture/research/team_mate/Kaggriculture_main_512631c/agents/route_clustering_switch_agent/fast_kaggriculture/.venv-bench/bin/python /mnt/e/ai_coding/kaggle/kaggriculture/experiments/daily_dp_v7_20260903/tools/run_opponent_panel.py --opponents pass,g001,g003,yhay81_six_day,boatlee_v29 --configs /mnt/e/ai_coding/kaggle/kaggriculture/experiments/daily_dp_v7_20260903/profiles/s3c/hauling_configs.json --labels L3_base,L3_return --seed 20261401 --count 50 --repeat 1 --threads 16 --out /mnt/e/ai_coding/kaggle/kaggriculture/experiments/daily_dp_v7_20260903/receipts/NEW_BOATLEE_PANEL
```

逐步校验脚本`tools/check_boatlee_native.py`；加`--mirror`验证镜像。线程检查`tools/test_boatlee_isolation.py`；综合核对`tools/verify_boatlee_panel.py`。

继续先补齐Kaito V58、Lynn V5、公开EcoBot V7。没有新我方参数晋级，不降低原目标，不训练、不提交Kaggle、不Git推送。
