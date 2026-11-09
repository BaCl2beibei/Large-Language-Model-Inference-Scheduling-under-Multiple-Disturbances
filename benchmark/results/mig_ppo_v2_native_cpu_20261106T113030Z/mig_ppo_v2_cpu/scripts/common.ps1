$ErrorActionPreference = 'Stop'
$TaskBundleRoot = Split-Path -Parent $PSScriptRoot
$TaskPython = Join-Path $TaskBundleRoot '.venv\Scripts\python.exe'
function Invoke-PpoCpu {
    param([string[]]$Arguments)
    if (!(Test-Path $TaskPython)) { throw 'Run scripts\install_cpu.ps1 first.' }
    Push-Location $TaskBundleRoot
    try {
        & $TaskPython 'benchmark\run_ppo_sim.py' @Arguments
        if ($LASTEXITCODE -ne 0) { throw "CPU command failed: exit $LASTEXITCODE" }
    } finally { Pop-Location }
}
