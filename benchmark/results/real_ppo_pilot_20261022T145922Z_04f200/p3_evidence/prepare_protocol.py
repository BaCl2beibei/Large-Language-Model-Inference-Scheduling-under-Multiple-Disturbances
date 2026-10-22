"""Freeze P3 CPU artifacts without touching bridge gates, budgets or services."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import yaml

ROOT=Path(__file__).resolve().parents[4];RUN=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from disturbance_exp.workload import load_local_tokenizer,_make_prompt,_encode
from rl_scheduler.real_bridge import digest
from rl_scheduler.real_budget import atomic_json
from rl_scheduler.real_opportunity import protocol,make_trace,maximum_duration,readonly_budget
from rl_scheduler.real_observation import RealNormalizer


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    spec=protocol(RUN,ROOT/'rl_scheduler/configs/real_pilot.yaml')
    runtime=yaml.safe_load((ROOT/'rl_scheduler/configs/real_pilot.yaml').read_text())
    spec['sampling']=runtime['sampling'];spec['runtime_config_hash']=sha(ROOT/'rl_scheduler/configs/real_pilot.yaml')
    spec['llumnix_config_hash']=sha(ROOT/'rl_scheduler/configs/real_llumnix.yaml')
    # The frozen snapshot currently uses remaining_steps/no defrag, TP1/non-PD.
    llumnix=yaml.safe_load((ROOT/'rl_scheduler/configs/real_llumnix.yaml').read_text())
    assert llumnix['INSTANCE']['MIGRATION_LOAD_METRIC']=='remaining_steps' and not llumnix['INSTANCE']['ENABLE_DEFRAG']
    assert llumnix['MANAGER']['MIGRATE_OUT_THRESHOLD']==spec['balanced']['threshold']
    atomic_json(RUN/'p3_evidence/user_override_06.json',{'run_id':json.loads((RUN/'budget.json').read_text())['run_id'],
        'user_instruction':'跳过smoke，直接执行06_sol_p3_prepare.md','smoke_skipped':True,
        'scope':'06 preparation and recorded opportunity prerequisite exception; not a live-gate pass',
        'GPU_experiments_authorized_in_06':False,'budget_caps_changed':False,'verification_thresholds_changed':False})
    (RUN/'opportunity_spec.yaml').write_text(yaml.safe_dump(spec,sort_keys=False,allow_unicode=True))
    (RUN/'reward_initial.yaml').write_text(yaml.safe_dump({'schema':'real-reward-v1','weights':spec['reward']['weights'],
        'C_ref':None,'N_ref':None,'status':'unmeasured_initial_weights_not_optimized',
        'scale_rule':spec['reward']['scale_rule'],'revisions_used':0,'revisions_max':1,
        'gap_limit':spec['reward']['gap_limit'],'no_component_clipping':True},sort_keys=False))
    (RUN/'final_burst_spec.yaml').write_text(yaml.safe_dump(spec['final_burst'],sort_keys=False))
    atomic_json(RUN/'p3_evidence/normalizer_frozen.json',RealNormalizer().state_dict())
    tokenizer=load_local_tokenizer(runtime['service']['model_path'])
    if tokenizer is None:raise RuntimeError('existing local tokenizer unavailable; no fallback pretending exact input tokens')
    pool={}
    for size in spec['workload']['input_budgets']:
        text,count=_make_prompt(tokenizer,size,spec['workload']['prompt_pool_seed']+size)
        actual=len(_encode(tokenizer,text));assert actual==count and 0<actual<=size
        pool[str(size)]={'prompt':text,'input_tokens':actual,'input_budget':size}
    directory=RUN/'traces/opportunity07';directory.mkdir(parents=True,exist_ok=True)
    atomic_json(directory/'prompt_pool.json',pool)
    manifest={'protocol_hash':digest(spec),'prompt_pool_hash':digest(pool),'traces':{},
        'tokenizer_path':runtime['service']['model_path']+'/tokenizer.json',
        'tokenizer_sha256':sha(Path(runtime['service']['model_path'])/'tokenizer.json'),
        'final_burst_traces':'deferred to selected A rate; generator and final seeds frozen now'}
    for phase,seeds,duration in [('probe',spec['probe']['seeds'],spec['probe']['arrival_sec']),
                                  ('opportunity',spec['opportunity']['seeds'],spec['horizons']['arrival_sec'])]:
        configurations=zip(spec['probe']['qps'],seeds) if phase=='probe' else ((q,s) for q in spec['probe']['qps'] for s in seeds)
        for qps,seed in configurations:
            key=f'{phase}_q{qps}_seed{seed}';path=directory/(key+'.json')
            rows=make_trace(spec,seed=seed,qps=qps,duration=duration)
            atomic_json(path,rows)
            manifest['traces'][key]={'path':path.name,'sha256':sha(path),'offered':len(rows),'seed':seed,'qps':qps,'phase':phase}
    atomic_json(directory/'manifest.json',manifest)
    sources=['benchmark/run_real_opportunity.py','rl_scheduler/real_reward.py','rl_scheduler/real_opportunity.py',
             'rl_scheduler/real_opportunity_runner.py']
    source_map={name:sha(ROOT/name) for name in sources}
    argv=[str(ROOT/'.venv/bin/python'),str(ROOT/'benchmark/run_real_opportunity.py'),'--config',str(ROOT/'rl_scheduler/configs/real_pilot.yaml'),
        '--opportunity-spec',str(RUN/'opportunity_spec.yaml'),'--stage','opportunity','--output-dir',str(RUN),
        '--manage-services','--invocation-id','07_opportunity_2101_2102_2103']
    command={'label':'frozen_probe_selection_and_12_pairs','argv':argv,'env':{'PYTHONUNBUFFERED':'1'},
        'validation_argv':argv+['--dry-run'],'GPU_execution_status':'not_run',
        'config_sha256':sha(ROOT/'rl_scheduler/configs/real_pilot.yaml'),'protocol_sha256':sha(RUN/'opportunity_spec.yaml'),
        'expected_max_episodes':18,'estimated_max_sec':maximum_duration(spec,runtime),'validation_exit_code':None}
    atomic_json(RUN/'commands/07_opportunity.json',{'schema':'real-opportunity-commands-v1','status':'CPU_prepared_not_executed',
        'run_dir':str(RUN),'source_hashes':source_map,'source_hash':digest(source_map),
        'protocol_hash':digest(spec),'commands':[command],'adaptive_selection_rule':spec['selection'],
        'planned_total_max_sec':maximum_duration(spec,runtime),'automatic_retries':False,'stop_on_failed_or_incomplete_audit':True,
        'user_override':'05 skipped explicitly; no live verification claim',
        'budget_snapshot_readonly':readonly_budget(RUN)})
    print(json.dumps({'status':'prepared_CPU_only','frozen_trace_files':len(manifest['traces']),
        'max_runtime_sec':maximum_duration(spec,runtime),'C_ref':None,'N_ref':None,'GPU_started':False},indent=2))


if __name__=='__main__':main()
