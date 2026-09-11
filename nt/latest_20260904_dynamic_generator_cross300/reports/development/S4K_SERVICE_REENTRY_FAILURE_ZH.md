# S4K：完成实验，但新增事务保护有回归，拒绝晋级

2026-09-04。参照 `S4K_SERVICE_REENTRY_PRE_REGISTER_ZH.md`，8配置7200场，N50双座位，8原始实时对手与PASS；5400重复账本、32官方对照、23新机制及4490旧机制，32原生CPU时延检查均运行结束。1800旧对照与S4J2逐场完全相同。

## 结果不能解释为恢复任务没有价值

保留对照对G00145%，恢复版31%；对G00314%→10%；PASS现金180428→170715。新自主对照43%/10%，恢复版31%/8%。全部结果见 `receipts/s4k_interaction_N50_v1/TABLES_ZH.md`。

真实账本指出：对PASS牛奶产出293.58、羊毛122.98均未改变，但新版本终局残留牛奶14.19、草莓18.26，销售金额下降9709。与最初“可能恢复维护不划算”的猜测不同，这次主要出现确定的事务错误。

新增 `service::protect` 把所有SELL限制为动作前库存，却漏掉同回合先DROP入库的货物。特别在终局，这会永久丢失可成交的SELL意图。资源保护本应只约束有真实预留的商品。

## 测试的教训与修复边界

官方/JAX/C++一致和已发动作no-effect=0，都不能证明Agent发出了应有动作：两种引擎一致执行了错误的策略订单。必须补“应存在的SELL意图不能被删”回归。

新增回归在旧build已确证失败：
`receipts/s4k2_regression_before_fix_v1/acceptance.json`
错误：`unreserved terminal DROP SELL was deleted`。

S4K2仅缩窄保护范围，不改任何经济参数。原配置与seed完整重跑后再判断能力有效性，不利用旧面板去挑seed或切版本。

## 冻结证据

- 源码/配置：`profiles/s4k/`。
- 旧build：`74cd0ec4d01806077892d8db1a7d88710ca087fd02e6dc770cbbd2276ea9354f`。
- 阶段收据：`receipts/s4k_stage_acceptance_v1/acceptance.json`，只是运行完成，不是正确性或目标通过。
- 现金链：`receipts/s4k_cash_chain_N50_v1/summary.json`。
- 全部5400账本闭合、已发生产动作no-effect=0、动物逃跑0；不能据此掩盖漏发SELL。
- 原生动作最高46.03ms，不是最终Python线上时延证明。
- 融资开关在N50面板中未触发；专项能力可执行不等于已有实战收益。

Goal仍active，同一冻结Agent对八对手分别超过90%的目标未达到。旧对照和新失败分支全部保留。
