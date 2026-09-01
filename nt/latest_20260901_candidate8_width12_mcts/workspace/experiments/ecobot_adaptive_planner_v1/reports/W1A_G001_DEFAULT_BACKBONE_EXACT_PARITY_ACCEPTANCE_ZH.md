# W1A：G001 默认骨架逐动作一致性验收

## 结论

**PASS。** 新 EcoBot 的默认回退链已经成功绑定到完整 G001 Agent；在自适应覆盖层关闭时，官方 Python `kaggle-environments==1.32.7` 中的动作序列和终局结果与原 G001 完全一致。

这一步只冻结“安全出发点”，不代表后续方案要针对某个已知 Agent 调参。后续自适应规划器只能读取公开盘面、市场、现金、人员、项目义务、deadline 和计划偏差，禁止使用对手名称、submission id 或 Arena id 做路由。

## G001 的准确含义

本实验中的 G001 不是单独一条静态动作带，而是当前本地的完整 `route_clustering_switch_agent/main.py`：

- 开局固定从 G001 出发；
- 原 Agent 自带第 144、168、216 步的公开状态路线切换能力；
- 这个完整行为被视为默认计划和失败回退，而不是被拆散后重新拼接。

## 实现

- 本地开发入口：`experiments/ecobot_adaptive_planner_v1/agents/g001_overlay_v0/main.py`
- 验收工具：`experiments/ecobot_adaptive_planner_v1/tools/verify_g001_overlay_exact_parity.py`
- 自适应覆盖层当前 `enabled=False`；返回 `None` 时原样执行 G001 动作。
- 覆盖层接口不包含任何对手身份字段。
- 最终提交前会将 G001 载荷与自适应模块打包为单文件；当前绝对/工作区路径加载方式只用于本地开发验收。

## 官方验收设置

| 项目 | 值 |
|---|---:|
| 官方环境 | kaggle-environments 1.32.7 |
| episodeSteps | 720 |
| seed | 1,941,001–1,941,004 |
| 座位 | 双方换座位 |
| 对手类型 | passive、G001 镜像 |
| 对比局数 | 16 组逐动作对比 |
| 实际官方完整对局 | 32（每组分别运行原版与包装版） |

每组同时检查：

1. 720 帧完整结束；
2. 双方状态均为 `DONE`；
3. 包装版候选动作与原 G001 每帧完全相同；
4. 对手动作每帧完全相同，排除环境轨迹偏移；
5. 双方终局资金完全相同。

## 结果

| 指标 | 结果 |
|---|---:|
| 完整组数 | 16 / 16 |
| 候选动作逐帧完全一致 | 16 / 16 |
| 对手动作逐帧完全一致 | 16 / 16 |
| 终局资金完全一致 | 16 / 16 |
| 跨局状态污染 | 0 |
| 崩溃、超时、非法终止 | 0 |

正式收据：`experiments/ecobot_adaptive_planner_v1/receipts/w1a_g001_overlay_exact_parity_seed1941001_n4x2_v1.json`

## SHA256

| 文件 | SHA256 |
|---|---|
| G001 overlay v0 | `BD386BB0EA62EC1AC278D048242E0D69D324F4319D3EE654E9AA69CCB8F0359A` |
| 验收工具 | `0227CCBCF337968BAF13A372A6D9905C5DC70CA359A24D41AA8313ABF89DC189` |
| 正式收据 | `7807C1CD1D423E95B00FBC3308CD990C6CD4CC02140D2CA3AFE3FE404D27218C` |

## 放行决定

W1A 放行。下一阶段可以在 G001 上加入通用的计划偏差状态和事件触发器，但必须满足：

- 关闭新模块时继续保持本报告的逐动作一致性；
- 每个覆盖动作都记录触发原因、估计收益、风险和回退原因；
- 不按对手身份特调；
- 不允许一次局部偏差破坏整套 G001 生产链。

