[CmdletBinding()]
param(
    [string]$PythonVersion = "3.11",
    [string]$TorchVersion = "2.11.0",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128",
    [string]$TritonRequirement = "triton-windows>=3.6,<3.7",
    [string]$CacheRoot = "D:\Kaggriculture\cache"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$venvPath = Join-Path $repoRoot ".venv"
$pythonPath = Join-Path $venvPath "Scripts\python.exe"
$env:UV_CACHE_DIR = Join-Path $repoRoot ".uv-cache"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Install it first from https://docs.astral.sh/uv/."
}

Push-Location $repoRoot
try {
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        & uv venv --python $PythonVersion $venvPath
        if ($LASTEXITCODE -ne 0) { throw "uv venv failed" }
    }

    # Run project code through PYTHONPATH below. Reinstalling a local wheel on
    # every invocation can fail when OneDrive temporarily locks its dist-info
    # directory, while an editable .pth can misdecode this non-ASCII path.
    & uv pip install --python $pythonPath "kaggle-environments==1.32.6" "pytest>=9"
    if ($LASTEXITCODE -ne 0) { throw "project dependency installation failed" }

    & uv pip install --python $pythonPath "torch==$TorchVersion" --index-url $TorchIndexUrl
    if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed" }

    & uv pip install --python $pythonPath $TritonRequirement
    if ($LASTEXITCODE -ne 0) { throw "Triton-Windows installation failed" }

    $env:PYTHONPATH = Join-Path $repoRoot "src"
    if ([System.IO.Path]::GetPathRoot([System.IO.Path]::GetFullPath($CacheRoot)) -eq "C:\") {
        throw "GPU caches must not use C:; pass a D: -CacheRoot"
    }
    $env:TRITON_CACHE_DIR = Join-Path $CacheRoot "triton"
    $env:TORCHINDUCTOR_CACHE_DIR = Join-Path $CacheRoot "torchinductor"
    New-Item -ItemType Directory -Force -Path $env:TRITON_CACHE_DIR | Out-Null
    New-Item -ItemType Directory -Force -Path $env:TORCHINDUCTOR_CACHE_DIR | Out-Null
    & $pythonPath -c "import torch, triton; assert torch.cuda.is_available(); print(f'torch={torch.__version__} cuda={torch.version.cuda} triton={triton.__version__} device={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}')"
    if ($LASTEXITCODE -ne 0) { throw "CUDA/Triton validation failed" }
}
finally {
    Pop-Location
}
