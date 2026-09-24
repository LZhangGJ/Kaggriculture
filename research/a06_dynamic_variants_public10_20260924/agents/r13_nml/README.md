# A06 R13 — 无 ML／RL，固定面板 111/160

**最终成绩 69.38%，原版 50.63%；提高 18.75 个百分点，未达到平均 80% 目标。**

保持 `main.py` 与 `policy/` 的相对路径；Python 入口 `main.py:agent`。预编译库适用于 Linux x86-64 / WSL，不依赖 GPU 或 ML/RL 库。重编译：`python3 build.py --unit`。

相对 A06 R12 revision 4：`discount=0.12`，`portfolio_passes=4`，`candidate_extra=0`，启用 `R2_SALE_CLOCK_MODE=2`；其余最终配置见 `policy/config.json`。旧学习模型已移除，负 `scenario` 被拒绝；没有对手 ID/环境种子选择、固定动作回放或私有信息读取。

完整方法、每对手 16 盘结果、局限和复現命令见 `REPORT_ZH.md`。当前数字来自本地冻结且参与过调优的面板，不是天梯或封存留出集成绩。4 局接口逐步动作比较与独立规则差分通过；五组单元测试通过。Kaggle 沙箱时间/内存限制未获得平台认证。

原版保存在完整证据包的 `inputs/A06_R12/`，不是本提交的运行依赖。

共享库 SHA-256：`a79f90939360c809bdaf52a93b2262633a3e613483c79111fbd5e1794d7d7519`。
