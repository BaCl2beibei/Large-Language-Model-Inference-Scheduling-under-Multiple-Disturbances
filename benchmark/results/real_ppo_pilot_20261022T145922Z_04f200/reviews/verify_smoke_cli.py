"""Run the four frozen CLI dry runs without writing the shared budget clock."""
from contextlib import redirect_stdout, redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('reviewed_cli', ROOT/'benchmark/run_real_ppo.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
clock = json.loads((RUN/'experiment_clock.json').read_text())
remaining = {'total_experiment_wall_time_sec': 28800. - clock['elapsed_sec'],
             **{p+'_sec': cap-clock['phase_usage'][p] for p, cap in
                [('P0_P3', 7200.), ('P4', 7200.), ('P5', 14400.)]}}
def require(seconds):
    if min(remaining['total_experiment_wall_time_sec'], remaining['P0_P3_sec']) < seconds:
        raise RuntimeError('insufficient recorded budget')
    return {'remaining': remaining}
results = []
for row in json.loads((RUN/'commands/05_smoke.json').read_text())['commands']:
    argv = row['validation_argv']
    assert argv[-1] == '--dry-run'
    output, error = io.StringIO(), io.StringIO()
    with patch.object(module, 'BudgetLedger', return_value=SimpleNamespace(require=require)), \
         redirect_stdout(output), redirect_stderr(error):
        code = module.main(argv[2:])
    results.append({'label': row['label'], 'argv': argv, 'exit_code': code,
                    'plan': json.loads(output.getvalue()) if output.getvalue() else None,
                    'stderr': error.getvalue()})
    assert code == 0 and results[-1]['plan']['GPU_will_start'] is False
(RUN/'reviews/smoke_cli.json').write_text(json.dumps({
    'scope': 'exact CLI main/parse/gate/config/trace dry run; only budget I/O replaced with read-only recorded-state check',
    'results': results, 'GPU_started': False, 'not_live_acceptance': True}, indent=2)+'\n')
print(json.dumps({'dry_runs': len(results), 'exit_codes': [r['exit_code'] for r in results]}))
