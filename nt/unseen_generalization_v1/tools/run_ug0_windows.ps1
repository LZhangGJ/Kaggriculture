param(
    [string]$Workspace = "D:\Kaggriculture",
    [string]$Competition = "kaggriculture",
    [int]$MaxReplays = 128,
    [string]$Python = "python",
    [switch]$DownloadLogs,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Experiment = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

$tokenFile = Join-Path $HOME ".kaggle\access_token"
$legacyFile = Join-Path $HOME ".kaggle\kaggle.json"
if (-not $env:KAGGLE_API_TOKEN -and -not (Test-Path $tokenFile) -and -not (Test-Path $legacyFile)) {
    throw "Kaggle authentication is missing. Set KAGGLE_API_TOKEN or create ~/.kaggle/access_token."
}

$syncArgs = @(
    (Join-Path $Experiment "tools\sync_live_assets.py"),
    "--workspace", $Workspace,
    "--competition", $Competition,
    "--max-replays", $MaxReplays
)
if ($DownloadLogs) { $syncArgs += "--download-logs" }
if ($Force) { $syncArgs += "--force" }

& $Python @syncArgs
if ($LASTEXITCODE -ne 0) { throw "Kaggle live-asset synchronization failed." }

$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = (Join-Path $Experiment "src")
    & $Python (Join-Path $Experiment "tools\build_ug0_manifest.py") --workspace $Workspace
    if ($LASTEXITCODE -ne 0) { throw "UG0 manifest construction failed." }

    & $Python -m pytest (Join-Path $Experiment "tests") -q
    if ($LASTEXITCODE -ne 0) { throw "UG0 tests failed." }
}
finally {
    $env:PYTHONPATH = $previousPythonPath
}

Write-Host "UG0 completed. Review: $Workspace\live_assets\ug0_corpus_v1\UG0_CORPUS_BASELINE_V1_ZH.md"
