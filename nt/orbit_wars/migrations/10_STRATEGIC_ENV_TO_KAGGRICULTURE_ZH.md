# 第 10 名路线迁移：ScheduleCache + Strategic Farm Env

源方案见 [第 10 名摘要](../sources/10_STRATEGIC_ENV_SOURCE_DIGEST_ZH.md)。

## 迁移结论

Kaggriculture 规则核心已很快，无需再造一个近似游戏；但可以在同一精确 JAX core 上建立“战略动作 wrapper”，预计算静态路径和生产模板，让 policy 只处理 task/job，而不是原始移动。这比独立 StrategicEnv 更安全。

## ScheduleCache

可预计算且与 episode state 无关：

- 6×6 网格任意位置到任意位置、各 depot entrance 的 shortest path/distance；
- 土地按 NE→SW→SE 解锁后的可达 mask；
- crop/animal 从 plant/place day 起的 production、maintenance、decay template；
- HIRE Fibonacci cost、BUY_LAND cost；
- 9 商品完整合法 market price lookup；
- town/store demand schedule；
- task template 的原始 action automaton。

动态部分只 patch 当前 unit position/backpack、tile state、cash、shed、market inventory 和 opponent public state。ScheduleCache 是精确加速，不应把对手订单/随机 weeds 预设成固定未来。

## Strategic action

一个 macro action：

`(unit, task_type, target, mode, budget/quantity)`

wrapper 把它注册成 job，之后每 turn 自动 move/operate，直到完成、失败或 policy cancel。训练早期限制每 turn 1 个 job edit；闭环稳定后加到 2–4。

target-relative quantity 对应第 10 名 ship bins：

- `MIN_NEEDED`：维护/购买/卖出最低量；
- `TARGET_CAP`：补到植物/动物/仓储目标；
- `25/50/75/100%`：库存或预算比例；
- `TERMINAL_ALL`：终局清仓。

policy head 接 direct arithmetic：完成时间、剩余 cash、shed margin、预计 yield、价格冲击、ROI、deadline slack、维护/捕获类 boolean（这里是 plant survives/animal stays/harvest before decay/sell before terminal）。

## 轻量自回归

按第 10 名负结果，优先只重跑 policy head：trunk 编码一次；选择一个 job 后更新 unit/资源摘要和 GRU context，再选下一 job。只有当消融证明 full trunk rerun 明显更强，才承受额外开销。

## 训练路线

1. 一动作/turn、无 hands/animals 的 crop+sell strategic env，10–30M。
2. 加持续作物和日终维护。
3. 加 hands、animal logistics、两个 job slots。
4. 加 market ordering、history opponent pool。
5. 3–5M 模型 100–300M PPO。

每个 curriculum 都使用同一精确 JAX transition，只改变候选/action wrapper；不能训练在删减规则的近似 environment 后直接部署。

## held-out 评估

Orbit MapCache 有 map overfit 风险；Kaggriculture 对应是 seed/weed/market/opponent overfit。训练、arena 和 final audit seed 集严格分开；固定规则 specialist 不参与训练的版本必须保留在 final panel。

## 本机性能

Cache 能让 candidate feature 很便宜，目标 15k–60k transitions/s；1–4M 模型 100M 约 0.5–1.9 小时热循环。主要上分潜力来自动作抽象和算术 features，开发难度低于全 micro-step 和 IMPALA，适合作为第一版 learned strategic agent。
