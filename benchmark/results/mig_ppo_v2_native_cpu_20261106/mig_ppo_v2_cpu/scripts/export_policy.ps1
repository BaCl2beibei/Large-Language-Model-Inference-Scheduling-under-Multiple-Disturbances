param([string]$RunName = 'local_001', [string]$ExportName = 'local_001')
. "$PSScriptRoot\common.ps1"
Invoke-PpoCpu -Arguments @('local-export-policy','--bundle-root','.','--run-dir',"runs/$RunName",'--output-dir',"exports/$ExportName")
