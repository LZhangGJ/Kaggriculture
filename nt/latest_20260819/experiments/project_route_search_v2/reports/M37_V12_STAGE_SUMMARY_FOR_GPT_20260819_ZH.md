# Kaggriculture M3.7 V12 阶段总结（供 GPT 独立复审）

日期：2026-08-19  
规则版本：官方 `kaggriculture 1.32.7`  
当前候选：M3.7 controller V12（不是 Kaggle 提交版本 V12）  
固定实验：`episode_id=94051618`、`seed=1905263333`、我方座位 1、`NullOpponent`、719 transitions  

## 1. 本阶段要解决什么

M3.7 不搜索新拳法，先验证三层控制器能否把完整经营日历兑现为真实经营动作：

1. 高层日历给出每天的人员、作物、动物购买/维护和土地目标；
2. 项目层把目标变成跨日承诺、资源预留和有截止时间的任务；
3. 调度与执行层把任务分配给单位，并产生官方合法的原子动作。

冻结边界：不修改官方 1.32.7 状态转移，不扩大 `RouteCalendarV3` schema，不播放金牌 Replay 的 raw action，不放松硬错误门，也不在工程门通过前启动正式路线搜索。

## 2. 当前真实结论

V12 已把“硬错误”降到 0，并且修复了已购动物从建设、拾取到放置的完整执行链；但它仍无法兑现 KAWASHIGI 日历中的中期作物扩张，终局现金只有 `50,134`。因此：

- 工程诊断基础：通过；
- 动物落地流水线：通过；
- 作物中期扩张：未通过；
- 金牌里程碑：未通过；
- 经济门 `72,261`：未通过；
- 官方 Python 逐状态复验：尚未对 V12 执行；
- 1,000 / 10,000 / 160,000 候选搜索：均未获准启动。

这不是“只差调参”。目前控制器能够表达金牌日历的高层数量，但调度、空间布局以及收获/回仓/维护之间的资源竞争，使金牌执行路径仍不在当前稳定可达集合内。

## 3. 分章节状态

| 章节 | 状态 | 已证明 | 未证明 |
|---|---|---|---|
| M3.7A 承诺生命周期诊断 | PASS | 2048 batch 的 GPU carry 只保存轻量聚合；逐承诺详细 trace 独立保存；日历跨日债务能更新 | 不代表经济能力 |
| M3.7B 动物 BUILD/PICKUP/PLACE | PASS | 18 个已成交动物对应 18 个 PLACE 任务；18 个全部发出终点 PLACE；0 个跨日被清空；0 硬错误 | 不代表动物路线经济最优 |
| M3.7C 作物 target debt 与中期兑现 | FAIL | V12 的目标冲突抢占消除了 V10/V11 的 deadline/逃逸硬错误 | 大量作物目标仍未兑现，生产任务仍有 34 个没有产生有效动作 |
| M3.7D 固定金牌里程碑 | FAIL | 尚无 | 第 6、11、12 天目标均未达到 |
| M3.7E 官方复验与经济门 | NOT RUN / FAIL | JAX V12 完整跑完 719 transitions，硬错误为 0 | `final_bank=50,134 < 72,261`；未做 V12 官方逐状态复验 |
| M3.7F 正式吞吐 | NOT RUN | 尚无 | 未测 V12 批量正式吞吐，不得声称达到 100k transitions/s |

## 4. V7 到 V12 的修复链

| 版本 | 主要改动 | 硬错误 | 终局现金 | 结论 |
|---|---|---:|---:|---|
| V7 | 紧急饲料购买与基础动物解锁 | 0 | 59,294 | 47 个 PLACE 任务实例中有 29 个未完成，27 个跨日被清空 |
| V8 | 强制划出作物执行 lane | 0 | 31,953 | 粗暴隔离单位，经济严重退化；方案被拒绝并改为默认关闭 |
| V9 | 仅在当日剩余步数足够时启动 BUILD/PLACE 路线 | 0 | 54,680 | 动物流水线通过：18 个承诺、18 个 PLACE、0 跨日丢失 |
| V10 | 只抢占 1 个低优先级任务，为紧急目标腾出 foothold | 2 | 51,705 | 出现一次 deadline miss 和一次羊逃逸；抢占不了解冲突目标 |
| V11 | 可选 FEED 不再自动当作硬任务 | 2 | 51,705 | trace 与 V10 完全相同，说明修复没有命中真正分支 |
| V12 | 优先抢占占用“生存/浇水目标格”的可选任务 | 0 | 50,134 | 工程硬错误消失，但作物扩张仍失败，经济未恢复 |

注意：不同版本现金仅用于该固定实验的诊断，不是正式对战强度排名。

## 5. V12 的具体失败证据

V12 完整运行：

- backend：GPU `cuda:0`；
- 719 transitions；
- `illegal_action=0`；
- `unexplained_effect_error=0`；
- `unplanned_deadline_miss=0`；
- `duplicate_resource_reservation=0`；
- `unplanned_animal_escape=0`；
- `plant_without_same_day_water=0`；
- final bank：`50,134`。

关键 backlog：

| Step | 未兑现作物目标 `[麦, 胡萝卜, 番茄, 草莓, 瓜]` | 未兑现动物维护 `[鸡, 牛, 羊]` |
|---:|---|---|
| 144 | `[3, 0, 0, 6, 1]` | `[0, 2, 0]` |
| 264 | `[13, 0, 0, 33, 0]` | `[0, 2, 5]` |
| 288 | `[26, 0, 0, 29, 0]` | `[0, 2, 7]` |
| 719 | `[17, 0, 0, 0, 0]` | `[0, 0, 0]` |

作物诊断：

- 作物任务实例：512；
- 其中生产任务：132；
- 生产任务未产生有效动作：34；
- deadline preemption 诊断次数：1,004，涉及 293 个 step；
- PLANT：46 次；
- WATER：408 次。

这表明 V12 不是不会生成任何作物动作，而是维护、收获、回仓、动物服务和新种植长期争抢单位与地块，导致中期新增种植规模远低于日历目标。

## 6. 三个金牌 Replay 的用途边界

本包包含 M3.6B 实际用于“完整 30 天日历编译覆盖”的三局官方 Replay：

| 路线 | Episode | 玩家 | Replay reward | 结构 | 本阶段用途 |
|---|---:|---|---:|---|---|
| KAWASHIGI_6C12S_4LAND | 94051618 | カワシギ | 90,326 | 6 牛、12 羊、累计 4 块土地 | V12 固定日历来源和主要诊断目标 |
| TETSUYA_11C4S_3LAND | 94052517 | tetsuya | 78,287 | 11 牛、4 羊、累计 3 块土地 | 验证另一种动物比例可无截断编译 |
| RECURSION_6C7S_3LAND | 94054256 | ReCurSiON | 65,950 | 6 牛、7 羊、累计 3 块土地 | 验证不同作物/动物结构可无截断编译 |

三局均成功编译成完整 30 天高层日历，`event_overflow_count=0`。这只证明 schema 能表达高层日历，不证明控制器能重现金牌终局状态。

## 7. 代码入口

- `code/project_route_search_v2/src/project_route_search_v2/m36_controller.py`：每日经营目标、市场与调度入口；
- `code/project_route_search_v2/src/project_route_search_v2/m36_scheduler.py`：统一任务优先级、任务粘性、V12 目标冲突抢占；
- `code/project_route_search_v2/src/project_route_search_v2/m3_controller.py`：动物建设、拾取、放置、喂养和作物底层执行；
- `code/project_route_search_v2/src/project_route_search_v2/m37_lifecycle.py`：跨日承诺/债务聚合诊断；
- `code/project_route_search_v2/tools/generate_m36d_controller_trace_v1.py`：固定种子生成 719-step GPU trace；
- `code/project_route_search_v2/tools/analyze_m37b_animal_pipeline_v1.py`：动物流水线审计；
- `code/project_route_search_v2/tools/analyze_m37c_crop_pipeline_v1.py`：作物兑现审计；
- `code/project_route_search_v2/tests/`：相关单测；
- `code/kaggriculture_jax/`：本地 JAX 1.32.7 状态转移实现；
- `code/strategic_v5/`：当前任务 schema、原子动作编译器和经济辅助依赖。

## 8. 当前回归结果

2026-08-19 在 Windows 原生 Python 环境重新运行：

```powershell
$env:PYTHONPATH='src;..\..\gpu_sim\src;..\strategic_v5\src'
& '..\..\.venv\python.exe' -m pytest `
  tests\test_m37_lifecycle.py `
  tests\test_m36c_scheduler.py `
  tests\test_m36c_controller.py `
  tests\test_m3_animal_controller.py -q
```

结果：`34 passed in 59.16s`。

## 9. 建议 GPT 重点复审的问题

1. 当前 `RouteCalendarV3` 只有每日数量目标，是否缺少决定金牌物流效率的空间布局、批次和单位分工语义？
2. 中期目标落后时，应该优先改成“项目级批次调度”，还是继续在原子任务优先级中追加抢占规则？
3. 是否应显式构造 `START_CROP_LOT / SERVICE_CROP_LOT / HARVEST_AND_BANK_LOT` 等闭环项目，让单位一次承诺一组地块，而不是让 PLANT/WATER/HARVEST 单任务长期互抢？
4. 怎样在不放松硬错误门、不播放 raw Replay 的前提下，让金牌高层日历进入稳定可达集合？
5. 在恢复大规模搜索前，最小的下一个验收应是什么，才能区分“布局表达不足”“人员数量不足”“任务优先级错误”和“回仓/出售批处理不足”？

## 10. 结论

V12 的价值是把错误从“动物被买入但落不了地、跨日任务消失、硬错误”收敛为一个更清晰的问题：控制器虽然合法、安全，但不会高效地同时兑现多条中期经营承诺。

因此 V12 适合作为 GPT 复审的工程基线，不适合作为已完成路线，也不应启动正式搜索。
