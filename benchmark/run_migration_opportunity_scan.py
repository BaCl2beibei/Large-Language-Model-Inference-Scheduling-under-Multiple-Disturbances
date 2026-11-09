#!/usr/bin/env python3
"""Plan, execute bounded real scan/pairs, or analyze independently. No PPO."""
from __future__ import annotations
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_scheduler.real_budget import atomic_json
from rl_scheduler.real_bridge import digest
from rl_scheduler.migration_scan import coarse_matrix, refinement, paired_schedule, select_thresholds, trace_for


def read(path):
    return json.loads(Path(path).read_text())


def audit():
    dependencies = {}
    for package in ('vllm','ray','aiohttp','transformers','psutil','PyYAML','matplotlib'):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    def command(argv):
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=10)
            return dict(exit_code=p.returncode, stdout=p.stdout.strip(), stderr=p.stderr.strip())
        except (OSError, subprocess.SubprocessError) as exc:
            return dict(error=str(exc))
    return dict(python=sys.executable, dependencies=dependencies,
        gpu=command(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv']),
        services=command(['supervisorctl','status']),
        compute_processes=command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']),
        gpu_time_reservation='unknown; invocation wall budget enforced by runner',
        cpu_affinity=len(os.sched_getaffinity(0)))


def plan(directory):
    directory.mkdir(parents=True, exist_ok=False)
    for name in ('configs','traces','raw_runs','plots'):
        (directory/name).mkdir()
    paths = ['rl_scheduler/migration_scan.py','rl_scheduler/migration_scan_runner.py','rl_scheduler/real_client.py',
        'benchmark/run_migration_opportunity_scan.py','rl_scheduler/migration_scan_analysis.py',
        'rl_scheduler/real_bridge.py','rl_scheduler/real_env.py','rl_scheduler/real_opportunity.py',
        'rl_scheduler/real_opportunity_runner.py','llumnix/backends/vllm/scheduler.py',
        'llumnix/backends/output_forwarder.py',
        'llumnix/backends/vllm/worker.py','llumnix/llumlet/llumlet.py','llumnix/real_control.py',
        'llumnix/llumlet/migration_coordinator.py','llumnix/llumlet/local_migration_scheduler.py',
        'llumnix/entrypoints/vllm/api_server.py','llumnix/entrypoints/vllm/real_control_api.py',
        'llumnix/entrypoints/vllm/real_metrics_stream.py',
        'rl_scheduler/configs/real_pilot.yaml','rl_scheduler/configs/real_llumnix.yaml',
        'rl_scheduler/full_completion_service.py','rl_scheduler/real_service.py','rl_scheduler/full_completion.py',
        'benchmark/run_full_completion_balance.py']
    manifest = dict(schema='migration-opportunity-scan-v1', status='planned_no_gpu_experiments',
        run_id=directory.name, created_utc=datetime.now(timezone.utc).isoformat(),
        code_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        cases=coarse_matrix(), seeds=[91001,91002,91003], scan_seed=91000,
        arrival_sec=120, drain_sec=120, cleanup_sec=240, route_shares=[.7,.1,.1,.1],
        slos=dict(ttft_ms=2000, tpot_ms=150), policy='constrained Balanced; native policy disabled',
        coarse_runs=9, refinement_runs_per_candidate=3, paired_runs_per_candidate=12,
        threshold_rule='finite nonidle arrival-phase F empirical quartiles; reject degeneracy',
        environment=audit())
    atomic_json(directory/'experiment_manifest.json',manifest)
    print(json.dumps({'run_dir':str(directory),'coarse_runs':9,'pairs_for_2_to_3_candidates':'24–36',
        'gpu_started':False},ensure_ascii=False))


def verify_sources(manifest):
    for path, expected in manifest['code_hashes'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('source changed since plan; create a new isolated run: '+path)


def runtime(service_dir, case):
    import yaml
    cfg = yaml.safe_load((ROOT/'rl_scheduler/configs/real_pilot.yaml').read_text())
    server = yaml.safe_load((ROOT/'rl_scheduler/configs/real_llumnix.yaml').read_text())
    server['SERVER']['REQUEST_OUTPUT_QUEUE_TYPE']='rayqueue'
    server['MANAGER']['ENABLE_ROUTINE_MIGRATION']=False
    server['MANAGER']['ENABLE_PRE_STOP_MIGRATION']=False
    server_path=service_dir/'llumnix.yaml'
    server_path.write_text(yaml.safe_dump(server))
    svc=cfg['service']
    svc['ray_num_cpus']=16
    svc['ray_temp_dir']='/tmp/ms_'+digest(str(service_dir))[:12]
    svc['env']['LLUMNIX_SCAN_TELEMETRY']='1'
    argv=svc['server_argv']
    for name,value in [('--config-file',server_path),('--max-num-seqs',case['max_num_seqs']),
        ('--gpu-memory-utilization',case['gpu_memory_utilization']),('--max-model-len',case['max_model_len'])]:
        argv[argv.index(name)+1]=str(value)
    argv.extend(['--max-num-batched-tokens',str(case['max_num_batched_tokens'])])
    cfg['scope']='bounded_migration_opportunity_scan_no_training'
    (service_dir/'runtime_config.yaml').write_text(yaml.safe_dump(cfg))
    return cfg


def freeze(directory, candidates):
    manifest=read(directory/'experiment_manifest.json')
    verify_sources(manifest)
    selected=read(candidates)
    if not 2<=len(selected)<=3:
        raise ValueError('select two or three real audited candidates')
    records=[json.loads(s) for s in (directory/'completed_runs.jsonl').read_text().splitlines()]
    result={}
    for case in selected:
        matching=[r for r in records if r['phase']=='scan' and r['case']==case and r['policy']=='Stay']
        if not matching:
            raise ValueError('candidate lacks current-run Stay evidence: '+case['scenario'])
        summaries=[read(directory/r['summary']) for r in matching]
        if any(s['status']!='passed' for s in summaries):
            raise ValueError('invalid Stay evidence')
        values=[v for s in summaries for v in s['f_values']]
        result[case['scenario']]=dict(thresholds=select_thresholds(values), count=len(values),
            minimum=min(values),maximum=max(values), evidence=[r['summary'] for r in matching],
            evidence_hashes={r['summary']:digest(s) for r,s in zip(matching,summaries)})
    target=directory/'threshold_selection.json'
    if target.exists():
        raise RuntimeError('thresholds already frozen; use a new run to revise')
    atomic_json(target,dict(candidates=selected, selection=result,
        seed_protocol=manifest['seeds'], freeze_utc=datetime.now(timezone.utc).isoformat()))


async def execute(args):
    import aiohttp
    from disturbance_exp.workload import load_local_tokenizer
    from benchmark.run_full_completion_balance import ManagedFullCompletionService
    from rl_scheduler.full_completion import wait_service_ready, exact_prompt
    from rl_scheduler.real_env import RealEnvConfig
    from rl_scheduler.migration_scan_runner import episode, WallBudget, append
    directory=args.run_dir.resolve()
    manifest=read(directory/'experiment_manifest.json')
    verify_sources(manifest)
    environment=audit()
    if environment['dependencies']['vllm']!='0.6.3.post1':
        raise RuntimeError('requires vLLM 0.6.3.post1')
    if environment['compute_processes'].get('stdout') or environment['compute_processes'].get('exit_code')!=0:
        raise RuntimeError('foreign or unknown GPU workload; refusing startup')
    if args.stage=='paired':
        frozen=read(directory/'threshold_selection.json')
        for selection in frozen['selection'].values():
            for path, expected in selection['evidence_hashes'].items():
                if digest(read(directory/path))!=expected:
                    raise RuntimeError('frozen threshold evidence changed')
        schedule=paired_schedule(frozen['candidates'],manifest['seeds'],
            {k:v['thresholds'] for k,v in frozen['selection'].items()})
    else:
        cases=manifest['cases']
        if args.stage=='refine':
            cases=[c for candidate in read(args.candidates) for c in refinement(candidate)]
        schedule=[dict(case=c,seed=manifest['scan_seed'],policy='Stay',threshold=None) for c in cases]
    completed=[]
    ledger=directory/'completed_runs.jsonl'
    if ledger.exists():
        completed=[json.loads(s) for s in ledger.read_text().splitlines()]
    def identity(row):
        return digest({k:row[k] for k in ('case','seed','policy','threshold')})
    done={identity(r) for r in completed}
    if args.stage=='paired':
        # Resume only complete 12-run scenes; rerun an interrupted scene together
        # under fresh stable epochs. The append-only ledger retains old evidence.
        for scenario in {r['case']['scenario'] for r in schedule}:
            expected={identity(r) for r in schedule if r['case']['scenario']==scenario}
            if not expected <= done:
                done-=expected
    budget=WallBudget(args.max_seconds)
    # Group by backend config to keep models resident for all paired runs of a scene.
    groups={}
    for row in schedule:
        if identity(row) not in done:
            key=tuple(row['case'][k] for k in ('max_num_seqs','max_num_batched_tokens','gpu_memory_utilization','max_model_len'))
            groups.setdefault(key,[]).append(row)
    started=time.time_ns()
    atomic_json(directory/f'configs/{args.stage}_{started}_environment.json',environment)
    try:
        for index,rows in enumerate(groups.values()):
            budget.require(300)
            service_dir=directory/'raw_runs'/f'{args.stage}_{started}_{index}'
            service_dir.mkdir()
            (service_dir/'runtime').mkdir()
            cfg=runtime(service_dir,rows[0]['case'])
            service=ManagedFullCompletionService(service_dir)
            try:
                budget.require(cfg['service']['startup_timeout_sec']+300)
                service.start()
                timeout=aiohttp.ClientTimeout(total=15,connect=5)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    await asyncio.wait_for(wait_service_ready(cfg,service.is_running),
                        timeout=cfg['service']['startup_timeout_sec'])
                    tokenizer=load_local_tokenizer(cfg['service']['model_path'])
                    if tokenizer is None:
                        raise RuntimeError('local tokenizer unavailable')
                    # Short separately audited warmup, one forced initial route per GPU.
                    import random
                    route_seeds=[]
                    for low,high in ((0,.7),(.7,.8),(.8,.9),(.9,1.)):
                        route_seeds.append(next(i for i in range(10000) if low<=random.Random(i).random()<high))
                    prompt,count=exact_prompt(tokenizer,32,0)
                    warm_trace=[dict(request_id=f'w{i}',scheduled_arrival_sec=.1*(i+1),prompt=prompt,
                        input_tokens=count,output_tokens=8,route_seed=seed,request_type='warmup')
                        for i,seed in enumerate(route_seeds)]
                    warm=await episode(cfg=cfg,service_dir=service_dir,run_id=manifest['run_id'],
                        episode_id='warmup',policy='Stay',threshold=None,trace=warm_trace,
                        session=session,budget=budget,horizons=RealEnvConfig(arrival_sec=2,drain_sec=8))
                    mapping=warm['mapping']
                    for row in rows:
                        budget.require(300)
                        key=identity(row)
                        trace_path=directory/'traces'/f'{digest(row["case"])[:16]}_{row["seed"]}.json'
                        trace=trace_for(row['case'],row['seed'],tokenizer)
                        if trace_path.exists() and read(trace_path)!=trace:
                            raise RuntimeError('frozen trace mismatch')
                        if not trace_path.exists():
                            atomic_json(trace_path,trace)
                        atomic_json(directory/'configs'/f'{key}.json',row)
                        summary=await episode(cfg=cfg,service_dir=service_dir,run_id=manifest['run_id'],
                            episode_id=key[:20],policy=row['policy'],threshold=row['threshold'],
                            trace=trace,session=session,budget=budget,mapping=mapping)
                        mapping=summary['mapping']
                        path=service_dir/'runs'/key[:20]/'scan_summary.json'
                        append(ledger,dict(**row,phase='paired' if args.stage=='paired' else 'scan',
                            summary=str(path.relative_to(directory)),trace_hash=digest(trace)))
                        print(json.dumps(dict(scenario=row['case']['scenario'],seed=row['seed'],
                            policy=row['policy'],status='completed')),flush=True)
            finally:
                # A supervisor start RPC can time out after its owned program starts.
                if service.installed.exists():
                    service.running=True
                service.stop()
                stopped=service_dir/'service_stopped.json'
                if not stopped.exists() or read(stopped).get('cleanup_error'):
                    raise RuntimeError('owned service cleanup unconfirmed; stop invocation')
        atomic_json(directory/f'{args.stage}_status.json',dict(status='completed',gpu_experiments=True))
    except BaseException as exc:
        atomic_json(directory/f'{args.stage}_status.json',dict(status='incomplete',reason=type(exc).__name__+':'+str(exc)))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['plan','audit','scan','refine','freeze','paired','analyze'])
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--candidates',type=Path)
    parser.add_argument('--max-seconds',type=float,default=14400)
    args=parser.parse_args()
    import signal
    def terminated(signum,frame):
        raise KeyboardInterrupt('termination signal; preserve completed records and clean owned service')
    signal.signal(signal.SIGTERM,terminated)
    if args.stage=='audit':
        print(json.dumps(audit(),indent=2));return
    if args.run_dir is None:
        if args.stage!='plan':
            parser.error('--run-dir required')
        args.run_dir=ROOT/'benchmark/results'/('migration_opportunity_scan_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if args.stage in ('refine','freeze') and args.candidates is None:
        parser.error('--candidates required')
    if args.stage=='plan':
        plan(args.run_dir)
    elif args.stage=='freeze':
        freeze(args.run_dir,args.candidates)
    elif args.stage=='analyze':
        from rl_scheduler.migration_scan_analysis import analyze
        analyze(args.run_dir)
    else:
        try:
            asyncio.run(execute(args))
        except BaseException as exc:
            if args.run_dir.exists():
                atomic_json(args.run_dir/f'{args.stage}_status.json',dict(status='incomplete',
                    reason=type(exc).__name__+':'+str(exc)))
            raise

if __name__=='__main__':
    main()
