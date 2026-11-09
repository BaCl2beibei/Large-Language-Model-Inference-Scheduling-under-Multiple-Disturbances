param([string]$RunName = 'local_001', [int]$MaxUpdates = 0, [double]$WallBudgetSec = 0,
      [switch]$Resume, [switch]$SkipEvaluation, [switch]$RecoverLock)
. "$PSScriptRoot\common.ps1"
$TaskArgs = @('local-train','--bundle-root','.','--run-dir',"runs/$RunName")
if ($MaxUpdates -gt 0) { $TaskArgs += @('--max-updates',"$MaxUpdates") }
if ($WallBudgetSec -gt 0) { $TaskArgs += @('--wall-budget-sec',"$WallBudgetSec") }
if ($Resume) { $TaskArgs += '--resume' }
if ($SkipEvaluation) { $TaskArgs += '--skip-evaluation' }
if ($RecoverLock) { $TaskArgs += '--recover-lock' }
Invoke-PpoCpu -Arguments $TaskArgs
