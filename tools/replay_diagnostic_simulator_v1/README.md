# Kaggriculture Replay 诊断模拟器 v1

这是官方 Kaggriculture 1.32.7 Replay 画面的本地只读增强层。它不改变
比赛规则，也不声称从 Replay 反推出 Agent 的内部 `if` 条件。

## 已实现

- 嵌入官方 1.32.7 visualizer，完整播放 0–719 frame。
- 回合、拖动条、播放速度、座位切换双向同步。
- 在每名农场主/雇工头顶显示动作气泡：动作名称以及绿色成功、红色失败、黄色待确认状态；移动后气泡跟随到新位置，同格多人自动堆叠。
- 逐人显示原始位置、动作目标以及动作前后状态证据。
- 将动作分为“确认生效 / 未生效 / 跨日或同回合混合无法精确归因”。
- 对“今日再不浇就会枯死”的植物显示 `快枯死` 气泡，对“今日再不喂就会逃跑”的动物显示 `快逃跑` 气泡；临近日终后升级为红色闪烁。
- 其他报警：植物待浇、动物待喂、新杂草/遗留杂草、动物产物满、仓库接近容量、日终溢出、终局未售库存。
- 市场指令和整回合净变化提示。
- 支持粘贴本地路径或在浏览器选择 Replay JSON。

## 启动

```powershell
py -3 experiments/kaggriculture_replay_diagnostic_simulator_v1/tools/run_simulator.py `
  --replay "E:\\path\\to\\episode.json" `
  --open
```

默认地址：`http://127.0.0.1:8765/`。

仓库版本已经内置官方 Kaggle Environments 1.32.7 visualizer，不依赖开发者
本机的 `research/` 目录。也可以通过 `--official-viewer` 显式指定另一份官方
visualizer `index.html`。

## 目录

- `tools/run_simulator.py`：无第三方运行时依赖的本地 HTTP 服务。
- `src/`：Replay 解析、动作结果确认和报警逻辑。
- `static/`：诊断界面、响应式布局、动作与实体气泡。
- `vendor/official_1_32_7_visualizer/`：官方 1.32.7 画面及许可证。
- `tests/`：合成 Replay 单元测试，不包含真实比赛 Replay。

## 解释边界

- Replay 只记录整回合前后状态。两个单位在同一格连续操作、市场订单与单位
  搬运同回合发生时，只能显示净变化，不能永远精确归因到单个指令。
- `WATER / FEED / CARE` 若发生在第 24 回合，下一状态已经跨日重置相应
  `*_today` 标志，因此标成“需谨慎解释”，不会误报为失败。
- 报警代表盘面暴露，不代表该 Agent 一定在内部读取了该字段；要判断关注程度，
  仍需跨局统计报警后的响应率。

## 官方素材与许可

画面直接复用仓库内冻结的 Kaggle Environments 1.32.7 Kaggriculture
visualizer。上层诊断 UI、动作解释和报警逻辑位于本目录。Kaggle
Environments 项目采用 Apache License 2.0，许可证见
`vendor/official_1_32_7_visualizer/LICENSE`；本工具未修改官方比赛状态转移。
