"""Reproduce step-03 CPU verification. Never starts a GPU service or smoke."""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4];RUN=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rl_scheduler.real_budget import BudgetLedger,atomic_json
from rl_scheduler.real_bridge import digest


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest=json.loads((RUN/'commands/05_smoke.json').read_text())
    sources=list(manifest['source_hashes'])+['llumnix/metrics/causal_events.py',
        'rl_scheduler/tests/test_real_metrics.py','rl_scheduler/tests/test_real_bridge.py',
        'rl_scheduler/tests/test_real_env.py','rl_scheduler/tests/test_real_runner.py']
    commands=[]
    cases=[('tests',[sys.executable,'-m','unittest','rl_scheduler.tests.test_real_metrics',
                    'rl_scheduler.tests.test_real_bridge','rl_scheduler.tests.test_real_env','rl_scheduler.tests.test_real_runner','-v']),
           ('compile',[sys.executable,'-m','py_compile',*sources]),
           ('diff_check',['git','diff','--check']),
           ('repro_trace_contract',[sys.executable,str(RUN/'p2_evidence/repro_trace_contract.py')]),
           ('gpu_process_inventory',['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']),
           ('supervisor_inventory',['supervisorctl','status'])]
    for label,argv in cases:
        started=time.monotonic();result=subprocess.run(argv,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        path=RUN/'p2_evidence'/(label+'.log');path.write_bytes(result.stdout)
        commands.append({'command':argv,'exit_code':result.returncode,'elapsed_sec':time.monotonic()-started,
                         'log':str(path.relative_to(RUN)),'log_sha256':sha(path),
                         'interpretation':('read-only inventory; pre-existing EXITED Jupyter/pyworker can produce code 3' if label=='supervisor_inventory' else 'CPU verification')})
    current_source={name:sha(ROOT/name) for name in sources}
    spec=importlib.util.spec_from_file_location('_verify_real_cli',ROOT/'benchmark/run_real_ppo.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    cli.validate_gate(RUN/'gates/p1_offline_revalidated_03.json',RUN)
    status=BudgetLedger(RUN).status()
    checks={'all_required_CPU_commands_passed':all(c['exit_code']==0 for c in commands if 'supervisorctl' not in c['command']),
            '63_tests_passed':b'Ran 63 tests' in (RUN/'p2_evidence/tests.log').read_bytes() and (RUN/'p2_evidence/tests.log').read_bytes().rstrip().endswith(b'OK'),
            'no_GPU_compute_processes':(RUN/'p2_evidence/gpu_process_inventory.log').read_text().strip()=='',
            'budget_not_started':status['started'] is False and status['elapsed_sec']==0,
            'four_prepared_commands_hashes_match':all(sha(ROOT/name)==expected for name,expected in manifest['source_hashes'].items()),
            'seven_episodes_6060_sec_fit_phase_cap':manifest['planned_total_max_sec']==6060 and 6060<=status['remaining']['P0_P3_sec'],
            'no_live_gates_created':not any((RUN/'gates'/name).exists() for name in ('p1_live.json','p2_live.json')),
            'supervisor_files_not_installed':not any((Path('/etc/supervisor/conf.d')/Path(path).name).exists()
                for path in json.loads((RUN/'p2_evidence/service_preparation.json').read_text())['files'] if path.endswith('.conf'))}
    atomic_json(RUN/'p2_evidence/commands.json',commands)
    value={'status':'passed' if all(checks.values()) else 'failed','checks':checks,
           'tests':63,'GPU_experiments':0,'real_successful_migrations':0,'PPO_updates':0,
           'source_hashes':current_source,'source_hash':digest(current_source),'remaining_budget':status['remaining']}
    atomic_json(RUN/'p2_evidence/verification.json',value)
    print(json.dumps({k:v for k,v in value.items() if k!='source_hashes'},indent=2))
    return 0 if all(checks.values()) else 1


if __name__=='__main__':raise SystemExit(main())
