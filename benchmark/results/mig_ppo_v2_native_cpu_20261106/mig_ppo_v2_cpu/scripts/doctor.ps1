. "$PSScriptRoot\common.ps1"
Invoke-PpoCpu -Arguments @('local-doctor','--bundle-root','.','--output-file','runs/doctor_report.json')
