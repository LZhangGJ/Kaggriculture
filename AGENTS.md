# 工作入口

先读 `README.md` 和 `HANDOFF_ZH.md`。本工程唯一主线是：高手 replay 离线提取/聚类 → 路线互打 → 147 维状态上的浅树切换 replay 路线 → 中期接管无开局模板的 JointAFS R1。

- 不得把 `experiments/results` 中带原生 opening-template 的旧 Triad 扫描当作 R1 结论。
- 不使用 Nash 选开局；当前部署以 G275 为主线并对 G114 叶回退 G275，选择依据与局限见交接。
- 正式公开对手验证只用 `opponents/` 的 7 个强对手，同 seed 双座、多 seed。
- replay 只离线使用。线上 `agent/main.py` 只读取导出的路线、浅树和当前公开观察。
- 性能研究和候选准入只评估真实 warm 链：replay 浅树切换后 delay=1 温接管无模板 JointAFS R1。纯 R1、冷接管不再做性能探索；接管实现变更只需用它们做必要的默认行为兼容检查。
- 先复用 `scripts/` 的成熟聚类、原生对打和鲁棒树训练，不另写简化替代品。
