# 工作入口

先读 `README.md` 和 `HANDOFF_ZH.md`。本工程唯一主线是：高手 replay 离线提取/聚类 → 路线互打 → 147 维状态上的浅树切换 replay 路线 → 中期接管无开局模板的 JointAFS R1。

- 不得把 `experiments/results` 中带原生 opening-template 的旧 Triad 扫描当作 R1 结论。
- 不使用 Nash 选开局；当前部署固定主流 G001，选择依据与局限见交接。
- 正式公开对手验证只用 `opponents/` 的 7 个强对手，同 seed 双座、多 seed。
- replay 只离线使用。线上 `agent/main.py` 只读取导出的路线、浅树和当前公开观察。
- 任何接管改动必须同时报告纯 R1、冷接管和温接管，并检查旧/新 R1 在无外部观察时动作一致。
- 先复用 `scripts/` 的成熟聚类、原生对打和鲁棒树训练，不另写简化替代品。
