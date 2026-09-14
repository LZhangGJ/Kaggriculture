param([Parameter(Mandatory=$true)][string]$Root, [Parameter(Mandatory=$true)][string]$Python)
$ErrorActionPreference = 'Stop'
$repoPath = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$rootPath = (Resolve-Path -LiteralPath $Root).Path
$pythonPath = (Resolve-Path -LiteralPath $Python).Path
$action = New-ScheduledTaskAction -Execute $pythonPath -Argument "-m tools.arena.manage --root `"$rootPath`" tick" -WorkingDirectory $repoPath
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 5)
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 2)
Register-ScheduledTask -TaskName 'KaggricultureArena' -Action $action -Trigger $trigger -Settings $settings -Description 'Deterministic CPU arena tick; no LLM polling'
