"""CPU-only command validation, manifests and private supervisor file preparation."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import yaml

ROOT=Path(__file__).resolve().parents[4]
RUN=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rl_scheduler.real_budget import BudgetLedger,atomic_json
from rl_scheduler.real_bridge import digest
from rl_scheduler.real_runner import verify_trace_manifest
from rl_scheduler.real_service import ManagedService


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    config=ROOT/'rl_scheduler/configs/real_pilot.yaml';cfg=yaml.safe_load(config.read_text())
    manifest=verify_trace_manifest(cfg)
    sources=['benchmark/run_real_ppo.py','rl_scheduler/real_bridge.py','rl_scheduler/real_env.py',
             'rl_scheduler/real_runner.py','rl_scheduler/real_service.py','rl_scheduler/real_budget.py',
             'rl_scheduler/real_observation.py','rl_scheduler/real_client.py','rl_scheduler/real_metrics.py',
             'llumnix/real_control.py','llumnix/manager.py','llumnix/llumlet/llumlet.py',
             'llumnix/llumlet/migration_coordinator.py','llumnix/global_scheduler/migration_scheduler.py',
             'llumnix/global_scheduler/global_scheduler.py','llumnix/backends/output_forwarder.py',
             'llumnix/backends/vllm/worker.py','llumnix/backends/vllm/llm_engine.py',
             'llumnix/entrypoints/vllm/api_server.py','llumnix/entrypoints/vllm/real_metrics_stream.py',
             'llumnix/entrypoints/vllm/real_control_api.py']
    hashes={name:sha(ROOT/name) for name in sources}
    (RUN/'commands').mkdir(exist_ok=True)
    commands=[];validation=[]
    for stage,mode,seed in [('telemetry','shadow',6101),('bridge','shadow',6102),('bridge','active',6103),('bridge','overhead',6104)]:
        label='telemetry' if stage=='telemetry' else mode
        argv=[str(ROOT/'.venv/bin/python'),str(ROOT/'benchmark/run_real_ppo.py'),'--config',str(config),
              '--stage',stage,'--mode',mode,'--output-dir',str(RUN),'--manage-services',
              '--invocation-id',f'05_{label}_{seed}']
        result=subprocess.run(argv+['--dry-run'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        path=RUN/'p2_evidence'/('dry_run_'+label+'.json')
        if result.returncode:
            path.with_suffix('.error.log').write_text(result.stdout+result.stderr)
            raise RuntimeError('CPU dry-run failed: '+label)
        plan=json.loads(result.stdout);atomic_json(path,plan)
        row={'label':label,'argv':argv,'env':{'PYTHONUNBUFFERED':'1'},'config_path':str(config),
             'config_sha256':sha(config),'llumnix_config_path':str(ROOT/'rl_scheduler/configs/real_llumnix.yaml'),
             'llumnix_config_sha256':sha(ROOT/'rl_scheduler/configs/real_llumnix.yaml'),
             'expected_episodes':plan['episodes'],'estimated_max_sec':plan['estimated_max_sec'],
             'seed':seed,'trace_sha256':manifest['traces'][str(seed)]['sha256'],
             'validation_argv':argv+['--dry-run'],'validation_exit_code':result.returncode,
             'GPU_execution_status':'not_run','requires_next_stage_review_and_explicit_smoke_execution':True}
        commands.append(row)
        validation.append({'command':argv+['--dry-run'],'exit_code':result.returncode,
                           'log':str(path.relative_to(RUN)),'log_sha256':sha(path)})
    remaining=BudgetLedger(RUN).status()
    total=sum(row['estimated_max_sec'] for row in commands)
    assert total<=remaining['remaining']['P0_P3_sec']
    assert not remaining['started']
    document={'schema':'real-ppo-smoke-commands-v1','status':'prepared_not_executed',
              'run_dir':str(RUN),'run_id':json.loads((RUN/'budget.json').read_text())['run_id'],
              'source_hashes':hashes,'source_hash':digest(hashes),
              'planned_total_max_sec':total,'remaining_budget_at_preparation':remaining['remaining'],
              'commands':commands,'stop_on_failed_or_incomplete_stage':True,'automatic_retries':0,
              'note':'These argv/env are validated by CPU dry-run only. No live P1/P2 gate or research effect is claimed.'}
    atomic_json(RUN/'commands/05_smoke.json',document)
    atomic_json(RUN/'p2_evidence/cli_validation_commands.json',validation)
    files=ManagedService(ROOT,RUN,config).prepare()
    atomic_json(RUN/'p2_evidence/service_preparation.json',{'status':'prepared_not_installed_or_started',
                'files':{str(Path(path).relative_to(RUN)):sha(path) for path in files},
                'no_portal_or_Caddy_changes':True,'no_supervisorctl_start_executed':True})
    spec={'schema_version':'smoke-spec-v1','status':'prepared_not_run','scope':'P1 telemetry and P2 engineering smoke only',
          'protocol_does_not_replace_P3_or_final_evaluation':True,
          'source_hash':digest(hashes),'config_path':str(config),'config_sha256':sha(config),
          'trace_manifest':str(Path(cfg['smoke_trace_manifest']).relative_to(RUN)),
          'trace_manifest_sha256':sha(cfg['smoke_trace_manifest']),
          'seeds':{name:case['seed'] for name,case in cfg['smoke'].items()},
          'seed_search':'p2_evidence/smoke_seed_search.json','cases':cfg['smoke'],
          'sampling':cfg['sampling'],'SLO':{'TTFT_ms':2000,'request_TPOT_ms':150,'single_token':'TTFT_only'},
          'control':{'policy':'SR','directed_pairs':12,'global_inflight_limit':1,
                     'cooldown_sec_from_confirmed_terminal':5,'max_state_age_sec':2,
                     'destination_reserve':'max(16 blocks, ceil(0.05 * total_blocks))',
                     'shadow_zero_actual_migrations_required':True,
                     'active_min_confirmed_terminals':10,'active_min_successes':5,
                     'CPU_decision_path_P95_strictly_less_ms':100},
          'route_disturbance':{'active_only':[.7,.1,.1,.1],'pre_generated_route_seed_per_request':True,
                               'verify_actual_initial_route':True,'PPO_does_not_choose_route':True},
          'overhead':{'sequence':['OFF','ON','ON','OFF'],'paired_repetitions':2,
                      'same_frozen_trace':True,'same_sampling_and_SLO':True,
                      'OFF':'lease+client telemetry; no per-second bridge snapshot/observation',
                      'ON':'shadow per-second bridge snapshots/observations; zero actual migration',
                      'instrumentation_elsewhere_constant':True,'max_token_throughput_degradation':.05,
                      'max_TTFT_P95_degradation':.10,'zero_baseline':'incomplete; no relative pass'},
          'budget':{'persistent_ledger':'experiment_clock.json','immutable_P0_input':'budget.json',
                     'total_remaining_sec':remaining['remaining']['total_experiment_wall_time_sec'],
                     'P0_P3_remaining_sec':remaining['remaining']['P0_P3_sec'],
                     'planned_total_max_sec':total,'independent_cleanup_sec':60,
                     'watchdog_starts_cleanup_when_remaining_sec':60,
                     'startup_included':True,'handoff_wait_included_after_first_GPU_start':True},
          'services':{'supervisor_files':'services/','bind':'127.0.0.1:18080',
                      'reuse_models_within_overhead_pairs':True,'stop_task_services_after_each_command':True,
                      'never_stop_foreign_Ray_or_management_services':True},
          'prerequisites_before_execution':['03 offline hashes intact','Astra review of 03 handoff and commands',
                                            'future prompt explicitly authorizes GPU smoke','sufficient current persistent budget',
                                            'no foreign Ray/GPU service and available private ports'],
          'failure_rule':'Stop on unknown action, event gap, incomplete cleanup or failed gate; preserve evidence; do not alter seed/config/code during running stage.',
          'not_verified':['actual Llumnix/Ray object lifetime and worker flush','actual GPU transfer/cancellation',
                          'actual CPU latency/monitor overhead','budget watchdog under a real GPU failure']}
    (RUN/'smoke_spec.yaml').write_text(yaml.safe_dump(spec,allow_unicode=True,sort_keys=False))
    print(json.dumps({'status':'passed_CPU_preparation','commands':4,'episodes':7,'estimated_max_sec':total,
                      'service_files_installed':False,'GPU_started':False,'budget_started':remaining['started']},indent=2))


if __name__=='__main__':main()
