# Kaggriculture 每日 Top 60 最强 Submission Replay 收集规范

> 当前可执行入口和最新参数请先阅读 [`README_ZH.md`](README_ZH.md)。本文件保留完整的每日收集规范与历史背景。

版本：v1.0

日期：2026-08-26

项目目录：`E:\ai_coding\kaggle\kaggriculture`

竞赛 ID：`147734`

## 1. 任务目标

每天固定读取一次 Kaggriculture 实时 Public Leaderboard，选择当时 Rank 1–60 每支队伍正在计分的最强 submission，并下载该 submission 当前通过官方接口能够取得的全部公开 Replay。

本任务只负责：

1. 冻结每日榜单；
2. 锁定前 60 名各自的计分 submission；
3. 下载这些 submission 的全部可获取公开 Replay；
4. 去重、续传、校验并保存回执。

本任务不负责分析 Replay、复现 Agent、训练模型、提交 Kaggle 或删除历史数据。

## 2. 最重要的身份规则

不得只按玩家名、队伍名或 Team ID 下载最近对局。稳定工作对象必须是：

```text
competition_id + team_id + frozen_submission_id
```

其中 `frozen_submission_id` 必须直接来自当日榜单响应中的：

```text
publicLeaderboard[].submissionId
```

它代表该队伍当时正在榜单上计分的 submission，也就是本次收集要锁定的 Agent。

必须遵守：

- 同一队伍更换 submission：从下一次日快照起按新 Agent 收集；
- 同一 submission 排名发生变化：仍是同一个 Agent，不得混入该队伍其他 submission；
- 下载过程中即使榜单变化，也继续使用本次已经冻结的 submission ID；
- 禁止把同一玩家旧 submission、新 submission 或其他实验 submission 的 Replay 混在一起；
- 一局若同时属于两个前 60 submission，Replay 文件只保存一份，但两个 submission 的 manifest 都必须记录这局。

## 3. “全部 Replay”的严格口径

“某个最强 submission 的全部 Replay”只指在本次快照时：

- Kaggle 官方 `EpisodeService/ListEpisodes` 能返回；
- 至少一方的 `submissionId` 精确等于冻结的 submission ID；
- Episode 状态为 `COMPLETED`；
- Episode 类型为 `PUBLIC`；
- 双方 Agent 信息完整；
- 官方 `replay.json` 端点仍可下载。

不包括：

- Private 对局；
- 尚未完成的对局；
- 同一队伍其他 submission 的对局；
- 官方接口已经不再提供的历史对局。

报告中应写“官方接口当前可获取的全部公开 Replay”，不能写成“该 Agent 实际打过的全部对局”。

禁止传入 `--latest-per-submission`。一旦使用该参数，就只会保留最近 N 局，不再满足“全部公开 Replay”的要求。

## 4. 每日目录与续传规则

每天使用一个固定目录：

```text
E:\ai_coding\kaggle\kaggriculture\replay\top60_daily_YYYYMMDD
```

例如：

```text
E:\ai_coding\kaggle\kaggriculture\replay\top60_daily_20260826
```

规则：

1. 同一天第一次运行时下载并冻结 `leaderboard_service_raw.json`；
2. 同一天重跑时不得覆盖该榜单文件，必须用原快照继续续传；
3. 下载失败后重新执行同一天、同一个目录的命令；
4. 不得换目录重新下载，也不得删除已经成功保存的 Replay；
5. 新的一天创建新目录，并重新冻结当日 Top 60；
6. 使用 `--reuse-root E:\ai_coding\kaggle\kaggriculture\replay` 复用历史 Replay；优先创建硬链接，避免重复占用磁盘。

## 5. 标准每日执行命令

在 Windows Native PowerShell 7 中执行。网络请求保持单线程；16 线程只用于本地缓存扫描、校验和硬链接。

```powershell
$KaggriRoot = 'E:\ai_coding\kaggle\kaggriculture'
$SnapshotTag = Get-Date -Format 'yyyyMMdd'
$SnapshotRoot = Join-Path $KaggriRoot "replay\top60_daily_$SnapshotTag"
$LeaderboardPath = Join-Path $SnapshotRoot 'leaderboard_service_raw.json'
$PythonPath = Join-Path $KaggriRoot '.venv\python.exe'
$DownloaderPath = Join-Path $KaggriRoot 'replay\tools\download_score_band_all_replays.py'

New-Item -ItemType Directory -Path $SnapshotRoot -Force | Out-Null

if (-not (Test-Path -LiteralPath $LeaderboardPath)) {
    Invoke-WebRequest `
        -Uri 'https://www.kaggle.com/api/i/competitions.LeaderboardService/GetLeaderboard?competitionId=147734' `
        -OutFile $LeaderboardPath
}

& $PythonPath `
    $DownloaderPath `
    --leaderboard-json $LeaderboardPath `
    --score-min 0 `
    --rank-max 60 `
    --output-root $SnapshotRoot `
    --workers 16 `
    --api-workers 1 `
    --retries 5 `
    --reuse-root (Join-Path $KaggriRoot 'replay')

if ($LASTEXITCODE -ne 0) {
    throw "Top 60 Replay downloader failed with exit code $LASTEXITCODE"
}
```

参数说明：

| 参数 | 固定值 | 原因 |
|---|---:|---|
| `--score-min` | `0` | 按排名选前 60，不再用分数线裁剪 |
| `--rank-max` | `60` | 精确选择实时榜单前 60 |
| `--workers` | `16` | 只加速本地缓存扫描、校验和硬链接 |
| `--api-workers` | `1` | Episode 列表请求保持单线程，避免限流 |
| Replay 网络 worker | `1` | 下载器内部固定为单线程 |
| `--retries` | `5` | 临时网络失败自动重试 |
| `--reuse-root` | 项目 `replay` 目录 | 复用历史文件，避免重复下载 |

不要加入 `--trust-local-cache`，除非已经单独确认缓存完全可信；默认保留 JSON、EpisodeId、步数和 SHA256 校验。

## 6. 每日硬验收

下载命令返回成功后，读取：

```powershell
$SummaryPath = Join-Path $SnapshotRoot 'DOWNLOAD_SUMMARY.json'
$Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
$Selected = @(Import-Csv -LiteralPath (Join-Path $SnapshotRoot 'selected_teams.csv'))
$Manifests = @(Get-ChildItem -LiteralPath (Join-Path $SnapshotRoot 'submission_manifests') -File -Filter '*.json')

$Summary | Format-List
$Selected |
    Sort-Object { [int]$_.rank } |
    Format-Table rank,team_name,team_id,submission_id,score
```

必须同时满足：

1. `selected_teams == 60`；
2. `selected_teams.csv` 恰好 60 行；
3. Rank 恰好覆盖 1–60，不缺失、不重复；
4. 60 个 Team ID 唯一；
5. 60 个 submission ID 均非空；
6. `failed_unique_replays == 0`；
7. `valid_unique_replays == unique_episode_count`；
8. 每个冻结 submission 都有独立 manifest；
9. 每个 manifest 顶层 submission ID 与 `selected_teams.csv` 的冻结值一致；
10. manifest 内每局至少一方 submission ID 精确等于该 manifest 对应的冻结 submission；
11. Replay 文件中的 `info.EpisodeId` 与文件名一致；
12. Replay JSON 可解析，且包含可用的 `steps`；
13. 所有失败都记录 EpisodeId、错误信息和重试结果，不得静默跳过。

任意一项不满足，该日任务状态必须标记为 `FAILED`，不能标记为成功。失败时保留现场，稍后对同一目录续传。

特别注意：如果当天榜单接口暂时返回不足 60 支队伍，不能偷偷降为 Top 50 或补入历史队伍；应保留原始响应并报告异常。

## 7. 每日产物

每个日目录必须至少包含：

```text
leaderboard_service_raw.json
selected_teams.csv
api_raw/list_episodes_submission_<submission_id>.json
episodes/<episode_id>.json
submission_manifests/rankXX_<submission_id>.json
submission_summary.csv
episode_rows.csv
DOWNLOAD_SUMMARY.json
```

各文件用途：

- `leaderboard_service_raw.json`：证明当天 Top 60 和计分 submission 的官方来源；
- `selected_teams.csv`：冻结 Rank、Team ID、队名、submission ID 和分数；
- `api_raw/`：保存每个 submission 的原始 Episode 列表；
- `episodes/`：全局按 EpisodeId 去重后的 Replay；
- `submission_manifests/`：按同一个 Agent 划分的完整对局清单；
- `submission_summary.csv`：各 submission 的对局数和胜负统计；
- `episode_rows.csv`：包含座位、对手、分差和 rating 变化的逐局索引；
- `DOWNLOAD_SUMMARY.json`：下载数量、缓存复用、失败和榜单 SHA256 总回执。

## 8. 限流、断点续传与故障处理

- EpisodeService 固定 `--api-workers 1`；
- Replay 网络下载由现有脚本固定为单线程；
- 不得为了赶进度增加网络并发；
- 遇到 HTTP 429、403、5xx 或超时，使用脚本重试和退避；
- 连续失败时停止新增请求，保留快照、API 回执、已下载文件和失败信息；
- 稍后重新执行完全相同的命令和目录；
- 不得自动删除历史日目录；
- 不得因单局下载失败而把它从 manifest 中静默移除；
- 如果 Kaggle 接口结构变化，先保存原始响应并停止，不要猜字段或按队名替代 submission ID。

## 9. 定时任务要求

建议每天日本时间 06:15 执行一次；具体时刻可由队友调整，但必须满足：

- 使用 PowerShell 7 的 `pwsh.exe`；
- 直接执行一个 `.ps1` 文件，不嵌套 `powershell -Command`；
- 工作目录固定为 `E:\ai_coding\kaggle\kaggriculture`；
- 使用当前项目的 `.venv\python.exe`；
- 启用“错过计划时间后尽快启动”；
- 设置“已有实例运行时不启动新实例”；
- 标准输出和错误输出保存到当日日志；
- 只有下载器退出码为 0 且全部硬验收通过，定时任务才算成功。

建议的任务名称：

```text
Kaggriculture-Daily-Top60-Scoring-Submission-Replays
```

推荐把第 5、6 节整理成：

```text
E:\ai_coding\kaggle\kaggriculture\replay\tools\run_daily_top60_replay_snapshot.ps1
```

定时任务只调用这个脚本，不要在任务计划程序中拼接长命令。

## 10. 每日交接摘要

每次完成后，至少向团队报告：

```text
日期：YYYY-MM-DD
榜单快照：<absolute path>
Top 60：60/60
计分 submission：60 个
API 当前可获取的唯一公开 Replay：<N>
有效 Replay：<N>
新增网络下载：<N>
历史缓存复用：<N>
失败：0
榜单 SHA256：<hash>
状态：PASS / FAILED
```

若某队当天更换了计分 submission，要在摘要中单独写：

```text
SAME_TEAM_NEW_SUBMISSION: <team_id> <old_submission_id> -> <new_submission_id>
```

这表示出现了一个新的 Agent 数据源，不应与旧 submission 合并。

## 11. Git 与数据保留

- 原始 Replay 体积很大，默认不提交 Git；
- 应保留下载脚本、榜单快照、选择清单、manifest、统计、哈希和验收回执；
- 原始 Replay 使用共享磁盘、压缩包或对象存储交接；
- 若仓库已有更严格的大文件规则，以仓库规则为准；
- 禁止定时任务自动清理旧 Replay。清理必须另行审计和确认。

## 12. 一句话执行原则

每天先冻结 Top 60 榜单中的 60 个计分 submission ID，再只按这些精确 ID 下载官方当前可获取的全部公开 Replay；网络单线程、历史文件硬链接复用、失败原地续传，任何 submission 混用或静默缺失都视为任务失败。
