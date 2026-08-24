# UG0 合成语料 Smoke Test

日期：2026-08-24  
基线 commit：`a00c723681ad5b906d67a36f8fb8b73ba7eb91f5`

## 结果

状态：`PASS_CODE_AND_SYNTHETIC_SMOKE`

- `pytest`：9/9 通过；
- `compileall`：通过；
- Kaggle 同步命令 dry-run：通过；
- Kaggle CLI `submission-download` 能力检测：通过；
- submission archive 根目录 `main.py` 检查：通过；
- Replay 第二座位增量 observation 重建：通过；
- Replay 行为签名与粗路线族提取：通过；
- Submission ID 和行为签名的跨 split 泄漏检查：通过；
- 重复运行时排除自身输出目录：通过。

合成语料包含：1 个 Replay、1 个公开 Notebook 和 1 个 submission archive；识别出 `MILK_PRIMARY` 与 `WOOL_PRIMARY` 两个玩家行为族。

## 测试中发现并修复的问题

初版重建逻辑只在字段完全缺失时从前座 observation 补齐后座公共状态。由于上一帧已经存在 `step`，后座会错误地保留旧值，导致关键时点、每日快照和路线特征整体错位。

修复后，每一帧都执行：

```text
当前前座完整公共状态
→ 作为后座公共字段基底
→ 叠加当前后座增量
→ 保留后座私有状态
```

## 未执行项

当前运行容器无法访问用户本机 `D:\Kaggriculture`，也无法通过 Kaggle 网络下载，因此本报告不包含：

- 最新 submission 的真实 ID 和文件哈希；
- 真实排行榜 Replay 数量；
- 真实路线族分布；
- 官方 Python/JAX replay parity；
- GPU 路线搜索结果。

这些项目必须在本机运行 `tools/run_ug0_windows.ps1` 后，由真实生成的 `live_assets\ug0_corpus_v1\UG0_CORPUS_BASELINE_V1_ZH.md` 取代。
