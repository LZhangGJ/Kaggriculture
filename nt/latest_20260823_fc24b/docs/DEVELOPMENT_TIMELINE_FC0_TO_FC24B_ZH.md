# FC0 → FC24B 开发时间线

这份索引解释每一阶段解决了什么、哪些尝试失败，以及为什么最终是FC24B而不是某个更早的版本。对应原始报告全部保存在 `development_chain/`。

## 阶段A：建立严格Arena与第一版基座

### FC0：Ray K320基线

先冻结已通过官方/JAX一致性的K320动态Agent，建立逐对手而非只看平均胜率的验收纪律。

报告：`development_chain/FC0_BASELINE_AND_ACCEPTANCE_20260821_ZH.md`

### FC1、FC2：反近镜像市场抢卖

从公开盘面识别接近镜像的高价商品供给竞争，尝试提前出售草莓、牛奶和羊毛。FC2证明这种能力可迁移且没有伤害异构对局，但当时K320聚合得分率仍约89.96%，只能保存为阶段最优，不能宣布达成90%。

报告：

- `development_chain/FC1_K320_RANK14_DIAGNOSIS_AND_FC1B_SCREEN_20260821_ZH.md`
- `development_chain/FC2_K320_ANTI_MIRROR_HETEROGENEOUS_REGRESSION_20260821_ZH.md`

## 阶段B：融合不同强Agent能力并处理状态同步

### FC2A：第一次大融合与Public卡死

FC2A尝试同时保留K320、X562与PRT能力。本地JAX表现较强，但早期Python提交版在Public中途进入长时间全PASS。根因不是官方模拟器，而是冷启动Shadow-PRT时没有从第0步维护其内部状态。修复原则是：影子策略从开局一直更新，只在触发时接管动作。

报告：`development_chain/FC2A_PUBLIC_STALL_ROOT_CAUSE_20260821_ZH.md`

### FC4～FC6：路线救援与早期现金

这一阶段验证了固定后缀、Replay相似度和早期现金补丁。关键结论是：不能因为某个败局像某条公开路线，就在中盘生硬切换完整动作带；必须保持任务账本、销售债务和单位位置同步。

报告：

- `development_chain/FC4_CURRENT_STACK_ROUTE_RESCUE_AUDIT_20260821_ZH.md`
- `development_chain/FC5_PRT_PUBLIC_BOARD_AND_ROUTE_SELECTOR_AUDIT_20260821_ZH.md`
- `development_chain/FC6_EARLY_CASH_REPLAY_ROUTE_COMPLEMENT_AUDIT_20260821_ZH.md`

## 阶段C：高分玩家的项目生命周期和商店路由

### FC7、FC8：Kobe项目退出与作物转向

通过高分Replay研究“动物投资何时停止、作物何时接棒”，并把可迁移能力表达为公共商店与盘面条件，而不是作者识别。多个看似合理的项目切换在独立事件库中没有稳定收益，因此被保留为诊断，不直接进入正式策略。

报告：

- `development_chain/FC7_KOBE_CONDITIONAL_PROJECT_EXIT_AUDIT_20260821_ZH.md`
- `development_chain/FC8_HIGH_RANK_PROJECT_LIFECYCLE_AND_SCALE_AUDIT_20260822_ZH.md`
- `development_chain/FC8_KOBE_20_REPLAY_CROP_SHIFT_AUDIT_20260822_ZH.md`

### FC9～FC11：动物服务、随机商店与PRT窗口

验证动物是否继续喂养、不同公开商店组合下的经营路由，以及PRT何时真正形成威胁。结论是：固定某天切换不足以泛化；应保留实时价值、现金和公开供给条件。

报告：

- `development_chain/FC9_FC2B_ANIMAL_SERVICE_EFFECT_AUDIT_20260822_ZH.md`
- `development_chain/FC10_KOBE_77_PUBLIC_SHOP_ROUTER_AUDIT_20260822_ZH.md`
- `development_chain/FC10_KOBE_PUBLIC_SHOP_ROUTER_AND_HOLDOUT_20260822_ZH.md`
- `development_chain/FC10_KOBE_RANK14_CAPABILITY_AUDIT_20260822_ZH.md`
- `development_chain/FC11_KOBE_83_PUBLIC_SHOP_ROUTER_AUDIT_20260822_ZH.md`
- `development_chain/FC11_PRT_ROUTE_WINDOW_AND_KOBE_REFRESH_AUDIT_20260822_ZH.md`

## 阶段D：可部署的雇工/杂草同步与FC15

### FC12：大规模雇工杂草保护

解决公开方案中大量雇工遇到杂草后动作带错位的问题。该阶段还发现Python部署版与JAX候选曾有语义差异，因此建立“官方Python轨迹 → JAX逐步比较”的硬门。

报告：

- `development_chain/FC12_MASS_HIRE_WEED_GUARD_AUDIT_20260822_ZH.md`
- `development_chain/FC12_DEPLOY_PARITY_CORRECTION_AUDIT_20260822_ZH.md`
- `development_chain/FC12_FULL_ROSTER_PAIRED_AUDIT_20260822_ZH.md`

### FC14、FC15：PRT/X562分离保护与首次严格提交

FC14/FC15把不同单位链的杂草与雇工恢复拆开，避免一条保护规则覆盖另一条。FC15通过32个官方上下文的逐动作、逐状态和终局一致性后提交，但Public仅1988.7，证明旧本地池没有充分覆盖新环境的早期路线。

报告：

- `development_chain/FC14_P100_VS_FC12G_PAIRED_CRITICAL12_20260822_ZH.md`
- `development_chain/FC14_P100_VS_FC12G_PAIRED_FULL_ROSTER_20260822_ZH.md`
- `development_chain/FC15_VS_FC14_PAIRED_CRITICAL15_20260822_ZH.md`
- `development_chain/FC15_CPU_JAX_PARITY_AND_PUBLIC_SUBMISSION_20260822_ZH.md`

## 阶段E：利用Public败局扩充对手覆盖

### FC15早期败局与FC15R

下载FC15的Public Replay，重建双方公开状态，再用冻结公开Agent做反事实比较。这里明确区分：某个Replay可解释，不等于补丁可泛化；所有改动必须进入独立事件库和双座位门。

报告：

- `development_chain/FC15_MOON_MARKET_PAIRED_AUDIT_20260822_ZH.md`
- `development_chain/FC15_EARLY3_EXACT_COUNTERFACTUAL_FIX_REVIEW_20260823_ZH.md`
- `development_chain/FC15R_EARLY_OPENING_LOSS_FIX_20260823_ZH.md`

## 阶段F：最新6个公开强Agent与FC22

2026-08-22冻结并迁移：Boatlee V21、Soil V26-H、Moon V92、Kaito V39、Steven E284和Salem最新版。每个JAX实现均在2个seed、双座位的完整官方轨迹上动作、状态、终局4/4严格一致。

最终形成FC22能力栈：

1. Moon公共市场异常供给观察和提前销售债务环；
2. 开局小麦种子8；
3. 早期批量胡萝卜少买1，保留喂养现金；
4. 第10天后比较动物产品价值与小麦替代成本，决定是否容忍首次漏喂；
5. 以后的小麦购买按信用账本扣回。

FC22对最新6和旧28完成全量回归，但PRT仍为230/256，即89.84%，只差一局，不能过门。

报告：`development_chain/FC22_FULL_LOCAL_REGRESSION_GATE_20260823_ZH.md`

## 阶段G：FC24失败与FC24B最终过门

### 被拒绝的FC24

FC24加入终局最后可行补收，确实救回PRT临界局，但没有计算单位随身货物的机会成本。结果是部分对局为了补收低价作物，把高价牛奶或羊毛推迟到拥挤的最终市场步骤，造成真实现金下降。该版本没有被提升。

### FC24B

FC24B增加通用价值门：待收作物当前清算价值必须至少是单位随身货物价值的2倍。最终结果：

- 最新6：最低91.80%；
- 旧28：最低PRT 90.23%；
- 34个对手全部达到90%；
- 相对FC22：4局败转胜、0局胜转败、1,313局现金增加、0局现金下降；
- 官方/JAX动作、状态、终局36/36严格一致；
- CPU与跨局重入验收通过。

报告：

- `development_chain/FC24B_FULL_LOCAL_NON_REGRESSION_GATE_20260823_ZH.md`
- `development_chain/FC24B_FINAL_OFFICIAL_JAX_ACCEPTANCE_20260823_ZH.md`
- `development_chain/FC24B_PUBLIC_SUBMISSION_55708153_20260823_ZH.md`

## 最终纪律

FC24B是当前冻结池上的本地冠军，不是“已经解决比赛”的证明。新增高分Agent、Public败局或官方规则变化后，必须重新执行：来源冻结、精确JAX迁移、双座位独立事件回归、官方Python/JAX逐步一致性和CPU部署验收。
