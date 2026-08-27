# Replay 诊断模拟器交付说明（2026-08-27）

## 范围

- 官方 Kaggriculture 1.32.7 visualizer，支持 0–719 Frame 回放。
- 农场主与雇工动作气泡、动作是否生效及证据说明。
- 植物缺水、动物断粮/逃跑、杂草、仓库、日终与终局库存报警。
- 独立诊断侧栏和响应式画面缩放，不遮挡农场主体。
- 粘贴本地 Replay 路径或浏览器上传 JSON；不包含真实比赛 Replay。

## 可移植性

`vendor/official_1_32_7_visualizer/index.html` 已随工具冻结，启动不依赖
开发者本机的 `research/` 目录。官方许可证位于同目录 `LICENSE`。

## 验证命令

```powershell
py -3 -m pytest -q tools/replay_diagnostic_simulator_v1/tests
py -3 tools/replay_diagnostic_simulator_v1/tools/run_simulator.py `
  --replay "E:\path\to\episode.json"
```

## 本次验收结果

- 诊断与 Replay 集成测试：`5 passed`。
- 从本目录独立启动后，`/`、`/api/meta`、`/official-viewer` 均返回 HTTP 200。
- 测试 Replay：720 frames，官方模块版本 `1.32.7`。
- 内置官方画面 SHA256：
  `4E178925A8AD1E15F4E70F013BD39789C32798DA50A90152FB18048072FC2F53`。

最终 Git 提交与远端分支以本次推送回执为准。
