# Kaggriculture Replay 下载与完整性验收工具

版本：2026-09-11

这个目录保存我们当前使用的 Replay 下载器。它按冻结的 `submissionId` 收集 Kaggle 官方接口当前仍可取得的全部公开已完成对局，并生成可审计的逐局索引、网络回执和 SHA256 验收结果。

## 文件

| 文件 | 用途 |
|---|---|
| `download_score_band_all_replays.py` | 读取冻结榜单、查询每个 submission 的 Episode 列表、续传和校验 Replay |
| `verify_top_snapshot.py` | 不联网，重新核对榜单身份、API 覆盖、manifest、EpisodeId、帧数、大小和 SHA256 |
| `run_snapshot.ps1` | Windows PowerShell 一键执行“下载 + 独立验收” |
| `agent.md` | 原始每日 Top 60 收集规范，保留作详细背景说明 |

Replay 数据不应提交到 Git。建议统一写入本目录下已忽略的 `snapshots/`，或写入仓库外的大容量磁盘目录。

## 严格口径

“全部 Replay”指冻结时官方 `EpisodeService/ListEpisodes` 对目标 submission 返回、且同时满足以下条件的行：

- 状态为 `COMPLETED`；
- 类型为 `PUBLIC` 或 `EPISODE_TYPE_PUBLIC`；
- 恰好两名 Agent，且目标 `submissionId` 恰好出现一次；
- 官方 `replay.json` 端点仍可下载。

它不等于该 Agent 历史上实际打过的全部比赛。私有局、验证局、未完成局、已不再由官方接口提供的旧局都不在范围内。

## 环境

- Windows Native PowerShell 7；
- Python 3.10 或更高版本；
- Python 脚本只使用标准库，不要求安装额外包；
- 能访问 `www.kaggle.com`。

先进入仓库根目录：

```powershell
$RepoRoot = (Get-Location).Path
$ToolRoot = Join-Path $RepoRoot 'nt\replay_collection'
$Python = 'python'
```

如果项目使用自己的虚拟环境，可改成：

```powershell
$Python = Join-Path $RepoRoot '.venv\python.exe'
```

## 最简用法：下载当前 Top 40

先冻结一次官方榜单。一个下载任务从开始到结束都必须使用同一个榜单文件：

```powershell
$Tag = Get-Date -Format 'yyyyMMdd_HHmmss'
$OutputRoot = Join-Path $ToolRoot "snapshots\top40_$Tag"
$Leaderboard = Join-Path $OutputRoot 'leaderboard_service_raw.json'

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
Invoke-WebRequest `
    -Uri 'https://www.kaggle.com/api/i/competitions.LeaderboardService/GetLeaderboard?competitionId=147734' `
    -OutFile $Leaderboard
```

然后执行下载和独立验收：

```powershell
& (Join-Path $ToolRoot 'run_snapshot.ps1') `
    -LeaderboardJson $Leaderboard `
    -OutputRoot $OutputRoot `
    -Top 40 `
    -Workers 16 `
    -Retries 6 `
    -Python $Python `
    -ReuseRoot @((Join-Path $ToolRoot 'snapshots'))
```

联网始终是单线程。`Workers=16` 仅用于本地缓存读取、JSON 校验、硬链接和最终哈希，不会创建 16 个 Kaggle 下载线程。

成功时终端最后会显示 `PASS`，并且：

```powershell
Get-Content -LiteralPath (Join-Path $OutputRoot 'ACCEPTANCE.json') -Raw
```

其中 `status` 必须为 `PASS`、`errors` 必须为空、`valid_replays` 必须等于 `unique_replays`。

## 只下载一个指定 submission

下载器不会按队名猜 submission。先从官方榜单或认证后的我方 submission 列表确认 `teamId` 与 `submissionId`，再准备一个单项冻结选择文件：

```json
{
  "selectionSource": "manually frozen from an authenticated official Kaggle response",
  "publicLeaderboard": [
    {"rank": 1, "teamId": 16704466, "submissionId": 56114767, "displayScore": 2576.7}
  ],
  "teams": [
    {
      "teamId": 16704466,
      "teamName": "QQ Farming",
      "teamMembers": [],
      "lastSubmissionDate": "2026-09-09T05:42:36Z"
    }
  ]
}
```

把它保存为目标目录中的 `leaderboard_service_raw.json`，然后使用 `-Top 1`：

```powershell
& (Join-Path $ToolRoot 'run_snapshot.ps1') `
    -LeaderboardJson 'D:\kaggriculture_replays\my_submission\leaderboard_service_raw.json' `
    -OutputRoot 'D:\kaggriculture_replays\my_submission' `
    -Top 1 `
    -Python $Python `
    -ReuseRoot @('D:\kaggriculture_replays')
```

示例数字只是文件格式示范。实际使用时必须替换为你刚从官方响应确认的值。

## 直接调用 Python

需要自己编排时，可跳过 PowerShell 入口：

```powershell
& $Python -u (Join-Path $ToolRoot 'download_score_band_all_replays.py') `
    --leaderboard-json $Leaderboard `
    --score-min 0 `
    --rank-max 40 `
    --output-root $OutputRoot `
    --workers 16 `
    --api-workers 1 `
    --retries 6 `
    --reuse-root (Join-Path $ToolRoot 'snapshots')

if ($LASTEXITCODE -ne 0) {
    throw "Replay download failed: $LASTEXITCODE"
}

& $Python -u (Join-Path $ToolRoot 'verify_top_snapshot.py') `
    $OutputRoot --top 40 --workers 16

if ($LASTEXITCODE -ne 0) {
    throw "Replay audit failed: $LASTEXITCODE"
}
```

不得把 `--api-workers` 改成大于 1；代码会直接拒绝。若目标是“全部公开 Replay”，也不得传 `--latest-per-submission`。

## 断点续传和 429

遇到 429、403、超时或某个 Replay 重试耗尽时：

1. 不要删除输出目录或已下载文件；
2. 不要重新抓一份榜单替换冻结文件；
3. 确认没有旧下载进程仍在运行；
4. 稍后使用完全相同的命令和 `OutputRoot` 重跑。

下载器会验证已存在文件并从缺失 Episode 继续。每次网络尝试写入 `network_receipts.jsonl`；单局耗尽重试后会停止后续网络请求，并在 `DOWNLOAD_SUMMARY.json` 中把余下项目标为 `not_attempted`，不会伪装成成功。

## 主要产物

```text
leaderboard_service_raw.json
selected_teams.csv
api_raw/list_episodes_submission_<submission_id>.json
episodes/<episode_id>.json
submission_manifests/rankXX_<submission_id>.json
submission_summary.csv
episode_rows.csv
network_receipts.jsonl
DOWNLOAD_SUMMARY.json
ACCEPTANCE.json
VERIFIED_REPLAY_HASHES.json
download.log
verify.log
```

`ACCEPTANCE.json` 是最终验收入口；`VERIFIED_REPLAY_HASHES.json` 可用于以后确认数据没有被改动。

## 安全边界

- 不要把 Kaggle token、Cookie 或浏览器认证数据写入冻结文件、日志或 Git；
- 不要并行运行两个指向同一 `OutputRoot` 的任务，入口脚本使用 `RUN.lock` 阻止这种情况；
- 不要使用 `--trust-local-cache`，除非缓存已由其他流程单独验收；
- 不要提交 `episodes/`、快照目录或下载日志到仓库。
