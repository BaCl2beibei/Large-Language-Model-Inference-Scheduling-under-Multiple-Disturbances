param([string]$PythonExe = 'python')
$ErrorActionPreference = 'Stop'
$TaskBundleRoot = Split-Path -Parent $PSScriptRoot
Push-Location $TaskBundleRoot
try {
    & $PythonExe -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,13) and sys.maxsize > 2**32, "Use 64-bit Python 3.10, 3.11 or 3.12"'
    if ($LASTEXITCODE -ne 0) { throw 'Select a supported Python with -PythonExe.' }
    & $PythonExe -m venv '.venv'
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed.' }
    $TaskPython = Join-Path $TaskBundleRoot '.venv\Scripts\python.exe'
    & $TaskPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'pip setup failed.' }
    & $TaskPython -m pip install -r 'requirements-cpu.txt'
    if ($LASTEXITCODE -ne 0) { throw 'CPU dependency installation failed.' }
    & $TaskPython 'benchmark\run_ppo_sim.py' local-doctor --bundle-root . --output-file 'runs\doctor_report.json'
    if ($LASTEXITCODE -ne 0) { throw 'Doctor failed.' }
} finally { Pop-Location }
