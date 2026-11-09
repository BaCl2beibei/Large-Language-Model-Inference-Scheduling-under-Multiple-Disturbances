param([string]$RunName = 'local_001', [ValidateSet('development','heldout')][string]$Split = 'heldout')
. "$PSScriptRoot\common.ps1"
Invoke-PpoCpu -Arguments @('local-evaluate','--bundle-root','.','--run-dir',"runs/$RunName",'--split',$Split)
