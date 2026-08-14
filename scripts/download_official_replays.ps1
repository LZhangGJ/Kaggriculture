[CmdletBinding()]
param(
    [ValidatePattern('^\d{4}-\d{2}-\d{2}$')]
    [string]$Date,
    [string]$Destination
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
if (-not $Destination) {
    if (-not (Test-Path -LiteralPath "D:\")) {
        throw "D: drive is unavailable; pass an explicit non-C -Destination"
    }
    $Destination = "D:\Kaggriculture\data\raw\replays"
}
$Destination = [System.IO.Path]::GetFullPath($Destination)
$indexPath = Join-Path $Destination "_index"
New-Item -ItemType Directory -Force -Path $indexPath | Out-Null

if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
    throw "The Kaggle CLI is required and must already be authenticated."
}

& kaggle datasets download kaggle/kaggriculture-episodes-index -p $indexPath --unzip
if ($LASTEXITCODE -ne 0) { throw "Official replay index download failed" }

$manifestPath = Join-Path $indexPath "manifest.csv"
$rows = @(Import-Csv -LiteralPath $manifestPath)
if ($rows.Count -eq 0) { throw "Official replay index is empty: $manifestPath" }

if ($Date) {
    $selected = $rows | Where-Object date -EQ $Date | Select-Object -First 1
    if (-not $selected) { throw "Date $Date is not present in the official replay index" }
}
else {
    $selected = $rows | Sort-Object date | Select-Object -Last 1
    $Date = $selected.date
}

$targetPath = Join-Path $Destination $Date
New-Item -ItemType Directory -Force -Path $targetPath | Out-Null
$expectedCount = [int64]$selected.episode_count
$expectedBytes = [int64]$selected.total_bytes
$existing = @(Get-ChildItem -LiteralPath $targetPath -File -Filter "*.json")
$existingBytes = [int64](($existing | Measure-Object -Property Length -Sum).Sum)

if ($existing.Count -ne $expectedCount -or $existingBytes -ne $expectedBytes) {
    $driveRoot = [System.IO.Path]::GetPathRoot($targetPath)
    $freeBytes = [System.IO.DriveInfo]::new($driveRoot).AvailableFreeSpace
    $requiredBytes = $expectedBytes + 2GB
    if ($freeBytes -lt $requiredBytes) {
        throw "Insufficient free space: need at least $([math]::Round($requiredBytes / 1GB, 2)) GiB"
    }

    $dataset = "kaggle/$($selected.daily_dataset_slug)"
    & kaggle datasets download $dataset -p $targetPath --unzip
    if ($LASTEXITCODE -ne 0) { throw "Official replay download failed: $dataset" }
}

$files = @(Get-ChildItem -LiteralPath $targetPath -File -Filter "*.json")
$actualBytes = [int64](($files | Measure-Object -Property Length -Sum).Sum)
if ($files.Count -ne $expectedCount -or $actualBytes -ne $expectedBytes) {
    throw "Replay validation failed: expected $expectedCount files/$expectedBytes bytes, got $($files.Count) files/$actualBytes bytes"
}

[PSCustomObject]@{
    Date = $Date
    Dataset = $selected.daily_dataset_slug
    Path = $targetPath
    EpisodeCount = $files.Count
    TotalBytes = $actualBytes
    TotalGiB = [math]::Round($actualBytes / 1GB, 3)
}
