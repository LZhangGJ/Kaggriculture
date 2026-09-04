# S3H：Three-Day 完整对手接入

2026-09-03。结论：对手迁移通过；我方自主v7目标未达标。

## 完整保留什么

源为yhay81/three-day-shop-router，kernel 132926004。原始policy.cpp及两条719步tape未经修改，以独立C++命名空间编译，避免与Six-Day同名但不同布局的kag::State冲突。每局独立原版Context，不共享提交桥的全局Session。

第360步根据首个公开商店、市场肥料库存、对方公开种植格数选择两条路线之一；每72步检查未来一段资金/消耗需求，必要时增加出售订单。它不是每一步重新规划整个农场。所有日期和阈值只存在于冻结对手源码，不加入我方策略。

## 证据

- 10局×719步：输入投影、原发布main.py+agent.so动作、官方1.32.7双方逐步状态均无差异。四局保持路线0，六局切换路线1，覆盖BAKERY/PET_CAFE两类触发。
- 1,000局16线程复跑无跨局污染；50个独立seed的两座位，重复样本不算新增强度证据。另有6条全轨迹隔离/复位核验。
- 旧Boatlee/Kaito/Lynn/Fieldbook在新构建各4局官方对照、1,000局隔离通过；546项原机制回归通过。
- 与此前Lynn面板比较，PASS和旧6个对手合计5,600条记录完全不变。我方policy.hpp和冻结规则源码未变。
- 10局自然样本中资金保护没有新增订单，故不能宣称自然样本覆盖了所有资金保护分支；该源码包含且未改，有限对照不等于全状态数学证明。
- 分析辅助脚本曾误读tape计数字段、town字段，失败v1/v2均保留；修正辅助读取后v3通过。未改运行策略或规则来通过检查。

## 新对手成绩

开发A：20261401–20261450，开发B：20261501–20261550，均双座位。每配置200局，重复运行仅用于确定性核验。

| 我方 | 胜场 | 胜率 | 平均现金 | 对手平均现金 | 平均分差 |
|---|---:|---:|---:|---:|---:|
| L3_base/S3C03 | 34/200 | 17% | 92,230.94 | 103,698.77 | -11,467.83 |
| L3_return暂定物流修正 | 38/200 | 19% | 92,748.86 | 103,160.58 | -10,411.73 |

按seed聚类的近似95%区间约9.73%–24.27%和11.40%–26.60%，不是天梯保证。没有新参数晋级，最终独立验收集未使用。

## 位置与复验

总收据：`receipts/three_day_integration_acceptance_v3/acceptance.json`。
原版一致性和录像：`receipts/three_day_original_official_parity_v1/`、`receipts/three_day_switched_official_parity_v1/`。
对战原始数据：`receipts/pool_three_day_A50_v1/results.json`和B50同名目录。
构建SHA256：`5071e869b6ea0df537a1a954eb46452e30886d1781df82ccc06474e5da742202`。

使用已有WSL Python：

```text
python tools/run_opponent_panel.py --opponents pass,g001,g003,yhay81_six_day,boatlee_v29,kaito_v58,lynn_v5,yhay81_three_day --configs profiles/s3c/hauling_configs.json --labels L3_base,L3_return --seed 20261401 --count 50 --repeat 2 --threads 16 --out receipts/NEW_PANEL
```

目前7/8完整真实对手。下一步补公开EcoBot V7后继续跨对手通用机制消融，不替换原公开EcoBot的估值/调度，不以对手迁移完成冒充自身能力突破。
