# S4T：先整链全开，再谈单项归因

2026-09-04，响应用户明确要求：过去只打开少数开关，可能没有激活长生产链的组合能力。此次主实验不是继续单项试错，不能用此前某项无增益预先排除它。

## 先验证完整组合

原运行源码不变，固定S4S build。统一N50开发seed、双座位、八个原始实时对手及PASS。每个配置900局，共5配置4500局；两旧控制1800局逐场复现。先执行时延探针和最完整配置官方逐步对照，再全量对战与三个全开配置2700场完整账本。

|配置|作用|
|---|---|
|all_intraday|原保留基线|
|all_intraday_insert|原共享调度对照|
|full_workers25|此前统计的人员专项25开关全部同时开启；包括旧兼容项|
|full_chain|25开关基础上，将日内项目提出/采购、已承诺产能估值、入库销售、流动性恢复、公开对手产能估值一起开启|
|full_chain_autonomous|full_chain再同时开启自主开局、联合项目投资、完整作物日历，检验上层与下层共同激活|

完整顺序：项目生成与估值 → 人员规模 → 多人共享/接力/滚动调度 → 采购融资与生产并行 → 领料/搬运/入库/出售 → 当前日与下一日经营后果比较。**先比较这三个完整组合与两个对照，不提前逐项关闭。** 如果发现组合增益，再从整套开启状态逐组关闭进行归因和后续独立复验。

## 25专项开关及关联依赖

- 分工排程11：preparation_work、split_service_jobs、regret_schedule、regret_compile、shared_task_atoms_v2、stepwise_recoordination、preparation_pipeline_v2、schedule_value_compare、resource_aware_exchange、shared_service_insertions、idle_task_handoff。
- 用工估算3：insertion_hire_estimate、regret_hire_estimate、intraday_future_workforce。
- 物流与恢复9：capacity_hauling、terminal_deposit_schedule、overflow_idle_dispatch、preparation_spawn_guard、midroute_delivery、recover_service_inputs、procure_service_inputs、finance_service_inputs、incremental_pickup_repair。
- 排程估值2：day_consequence_compare、day_value_replan_next_day。
- full_chain关联6：intraday_admission、intraday_procurement、intraday_declared_value、sell_deposits、fix_liquidity、day_value_public_supply（前两项原基线已开，显式保持）。
- full_chain_autonomous关联3：autonomous_start、joint_investment_portfolio、portfolio_crop_calendar。

regret_schedule是regret_compile/hire的旧总开关；preparation_work在preparation_pipeline_v2启用时使用新分支；split_service_jobs和shared_task_atoms_v2的入口为OR。它们是兼容/替代，不会自动多出三种能力。仍全部设置true以如实测试“全开”，同时记录分支实际使用情况。其他作物数量、固定偏好、开局数量、现金权重等参数不扫值；两个自主上层开关会从实际观察生成项目，并非写入单seed路线。

## 验收

组合模式的时延、官方语义及整局异常先检查。若出现交互硬伤，保存失败配置和原因，修复后另版本重测，不能静默关掉能力冒充全开结果。完整对战报告每个对手胜率、资金差、按seed聚类区间、人员/生产/融资/销售变化；胜负与生产效率一并看。机会数不算利润，静止高收入不算对战强。

不使用真实未来、隐藏库存、身份/seed路由或Candidate事后Oracle。冻结之后如组合值得晋级，须使用此前未见P或新seed。全部八对手逐一超过90%才可能满足原Goal；本轮结束无论成功失败都落地复盘。
