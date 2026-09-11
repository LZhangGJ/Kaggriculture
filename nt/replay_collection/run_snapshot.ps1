[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$LeaderboardJson,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [ValidateRange(1, 1000)][int]$Top = 40,
    [ValidateRange(1, 16)][int]$Workers = 16,
    [ValidateRange(1, 20)][int]$Retries = 6,
    [string]$Python = 'python',
    [string[]]$ReuseRoot = @()
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

$LeaderboardJson = [IO.Path]::GetFullPath($LeaderboardJson)
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
$FrozenLeaderboard = Join-Path $OutputRoot 'leaderboard_service_raw.json'
$Downloader = Join-Path $PSScriptRoot 'download_score_band_all_replays.py'
$Verifier = Join-Path $PSScriptRoot 'verify_top_snapshot.py'
$LockPath = Join-Path $OutputRoot 'RUN.lock'
$Lock = $null

if (-not (Test-Path -LiteralPath $LeaderboardJson -PathType Leaf)) {
    throw "Leaderboard JSON not found: $LeaderboardJson"
}
if (-not (Test-Path -LiteralPath $Downloader -PathType Leaf)) {
    throw "Downloader not found: $Downloader"
}
if (-not (Test-Path -LiteralPath $Verifier -PathType Leaf)) {
    throw "Verifier not found: $Verifier"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

try {
    $Lock = [IO.File]::Open(
        $LockPath,
        [IO.FileMode]::CreateNew,
        [IO.FileAccess]::Write,
        [IO.FileShare]::None
    )
    $LockBytes = [Text.Encoding]::UTF8.GetBytes("pid=$PID`nstarted_at_utc=$([DateTime]::UtcNow.ToString('o'))`n")
    $Lock.Write($LockBytes, 0, $LockBytes.Length)
    $Lock.Flush()

    if (Test-Path -LiteralPath $FrozenLeaderboard -PathType Leaf) {
        $InputHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $LeaderboardJson).Hash
        $FrozenHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $FrozenLeaderboard).Hash
        if ($InputHash -ne $FrozenHash) {
            throw "OutputRoot already contains a different frozen leaderboard: $FrozenLeaderboard"
        }
    }
    elseif ($LeaderboardJson -ne $FrozenLeaderboard) {
        Copy-Item -LiteralPath $LeaderboardJson -Destination $FrozenLeaderboard
    }

    $DownloadArguments = @(
        '-u', $Downloader,
        '--leaderboard-json', $FrozenLeaderboard,
        '--score-min', '0',
        '--rank-max', $Top,
        '--output-root', $OutputRoot,
        '--workers', $Workers,
        '--api-workers', '1',
        '--retries', $Retries
    )
    foreach ($Root in $ReuseRoot) {
        $DownloadArguments += @('--reuse-root', [IO.Path]::GetFullPath($Root))
    }

    & $Python @DownloadArguments 2>&1 |
        Tee-Object -FilePath (Join-Path $OutputRoot 'download.log')
    $DownloadExit = $LASTEXITCODE
    if ($DownloadExit -ne 0) {
        throw "Downloader failed with exit code $DownloadExit. Resume with the same command and OutputRoot."
    }

    & $Python -u $Verifier $OutputRoot --top $Top --workers $Workers 2>&1 |
        Tee-Object -FilePath (Join-Path $OutputRoot 'verify.log')
    $VerifyExit = $LASTEXITCODE
    if ($VerifyExit -ne 0) {
        throw "Independent verifier failed with exit code $VerifyExit."
    }

    $Acceptance = Get-Content -LiteralPath (Join-Path $OutputRoot 'ACCEPTANCE.json') -Raw | ConvertFrom-Json
    if ($Acceptance.status -ne 'PASS') {
        throw "Acceptance status is not PASS."
    }
    Write-Host "PASS: $($Acceptance.valid_replays)/$($Acceptance.unique_replays) unique Replay files verified."
}
finally {
    if ($null -ne $Lock) {
        $Lock.Dispose()
        Remove-Item -LiteralPath $LockPath -Force -ErrorAction SilentlyContinue
    }
}
