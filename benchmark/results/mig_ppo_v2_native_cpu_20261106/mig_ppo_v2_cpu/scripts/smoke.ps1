param([string]$RunName = 'smoke_001')
. "$PSScriptRoot\common.ps1"
Invoke-PpoCpu -Arguments @('local-smoke','--bundle-root','.','--run-dir',"runs/$RunName")
