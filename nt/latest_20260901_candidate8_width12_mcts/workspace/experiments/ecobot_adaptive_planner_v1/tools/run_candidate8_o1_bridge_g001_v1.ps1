[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectRoot = 'E:\ai_coding\kaggle\kaggriculture'
$experimentRoot = Join-Path $projectRoot 'experiments\ecobot_adaptive_planner_v1'
$windowsPython = Join-Path $projectRoot '.venv\python.exe'
$wslPython = '/mnt/e/ai_coding/kaggle/kaggriculture/.venv_wsl_cpp/bin/python'
$wslProject = '/mnt/e/ai_coding/kaggle/kaggriculture'
$wslExperiment = "$wslProject/experiments/ecobot_adaptive_planner_v1"
$wslPythonPath = "$wslProject/research/team_mate/Kaggriculture_main_512631c/agents/route_clustering_switch_agent/src"

function Invoke-O1Pool {
    param(
        [int]$PrefixStart,
        [int]$FutureStart,
        [string]$Base
    )
    & wsl.exe -d Ubuntu-24.04 -- env "PYTHONPATH=$wslPythonPath" OMP_NUM_THREADS=16 `
        $wslPython "$wslExperiment/tools/generate_candidate8_competitive_pool.py" `
        --source "$wslProject/research/team_mate/Kaggriculture_main_512631c/agents/route_clustering_switch_agent/runtime/teammate_base.py" `
        --actions "$wslExperiment/artifacts/oracle/rank1_latest_55714246_ep99954642_route_actions.json.zlib" `
        --metadata "$wslExperiment/artifacts/oracle/rank1_latest_55714246_ep99954642_route_library.json" `
        --genomes "$wslExperiment/configs/w5_autonomous_capacity_r6_challenger_v1.json" `
        --merged-receipt "$wslExperiment/receipts/cxx_clean133_actual_vs_oracle_confirmed_v1.json" `
        --group all --opponents G001 --days 0,1,6 `
        --prefix-seed-start $PrefixStart --prefix-seed-count 8 `
        --future-seed-start $FutureStart --future-count 64 `
        --maximum-arms 64 --candidate-pool shortlist `
        --dataset-output "$wslExperiment/artifacts/o1_bridge/${Base}_v1.npz" `
        --output "$wslExperiment/receipts/${Base}_generation_v1.json"
    if ($LASTEXITCODE -ne 0) { throw "C++ pool generation failed: $Base" }

    & $windowsPython "$experimentRoot\tools\audit_candidate8_o1_bridge.py" `
        --dataset "$experimentRoot\artifacts\o1_bridge\${Base}_v1.npz" `
        --output "$experimentRoot\receipts\${Base}_eval_v1.json"
    if ($LASTEXITCODE -ne 0) { throw "O1 evaluation failed: $Base" }
}

$run1 = 'candidate8_o1_bridge_g001_n8x2_days016_future64'
$run2 = 'candidate8_o1_bridge_g001_n8x2_days016_future64_repro'
Invoke-O1Pool -PrefixStart 2961001 -FutureStart 3961001 -Base $run1
Invoke-O1Pool -PrefixStart 2962001 -FutureStart 3962001 -Base $run2

$trace = 'candidate8_o1_g001_scenario_mean_official_spot8_traces_v1.json.gz'
& wsl.exe -d Ubuntu-24.04 -- env "PYTHONPATH=$wslPythonPath" OMP_NUM_THREADS=16 `
    $wslPython "$wslExperiment/tools/export_candidate8_o1_traces.py" `
    --selection-receipt "$wslExperiment/receipts/${run1}_eval.json" `
    --source "$wslProject/research/team_mate/Kaggriculture_main_512631c/agents/route_clustering_switch_agent/runtime/teammate_base.py" `
    --actions "$wslExperiment/artifacts/oracle/rank1_latest_55714246_ep99954642_route_actions.json.zlib" `
    --metadata "$wslExperiment/artifacts/oracle/rank1_latest_55714246_ep99954642_route_library.json" `
    --genomes "$wslExperiment/configs/w5_autonomous_capacity_r6_challenger_v1.json" `
    --method scenario_mean --limit 8 `
    --output "$wslExperiment/artifacts/o1_bridge/$trace"
if ($LASTEXITCODE -ne 0) { throw 'C++ trace export failed' }

& $windowsPython "$experimentRoot\tools\verify_candidate8_o1_trace_bundle_official.py" `
    --trace-bundle "$experimentRoot\artifacts\o1_bridge\$trace" `
    --output "$experimentRoot\receipts\candidate8_o1_g001_scenario_mean_official_spot8_v1.json"
if ($LASTEXITCODE -ne 0) { throw 'Official Python 1.32.7 spot-check failed' }
