"""Bounded real episodes reusing RealEnv, causal IO and the migration arbiter."""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
import subprocess
import time

from .real_bridge import RealBridge, digest
from .real_budget import atomic_json
from .real_env import RealEnv, RealEnvConfig
from .real_metrics import _percentile
from .real_opportunity import metric_summary
from .real_opportunity_runner import (OpportunityLiveEpisodeIO, reconcile_migrations,
    route_audit, request_submission_audit, _nonblocking_controller_heartbeat, _stop_heartbeat)
from .migration_scan import opportunity, resource_summary, migration_statistics, remaining_steps, weighted_summary


def append(path, row):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False)+'\n')
        stream.flush()


class WallBudget:
    def __init__(self, seconds):
        self.deadline = time.monotonic()+seconds
    def require(self, seconds):
        if time.monotonic()+seconds > self.deadline:
            raise RuntimeError('invocation time budget exhausted; completed artifacts retained')


def gpu_sample():
    result = subprocess.run(['nvidia-smi', '--query-gpu=uuid,utilization.gpu,memory.used,memory.total',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=5)
    rows = {}
    for line in result.stdout.splitlines():
        uuid, util, used, total = [v.strip() for v in line.split(',')]
        def number(v):
            try:
                return float(v)
            except ValueError:
                return None
        util, used, total = map(number, (util, used, total))
        rows[uuid] = dict(gpu_pct=util, memory_mib=used,
                         memory_pct=used/total*100 if used is not None and total else None)
    return rows


def gpu_mapping(acks):
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True, timeout=5)
    pids = {}
    for line in result.stdout.splitlines():
        pid, uuid = [v.strip() for v in line.split(',')]
        pids[int(pid)] = uuid
    mapping = {}
    cluster = acks.get('cluster', acks)
    instances = cluster.get('instances', []) if isinstance(cluster, dict) else []
    for instance in instances:
        state = instance.get('state') or {}
        instance_id = state.get('instance_id') or instance.get('instance_id')
        workers = instance.get('workers') or []
        if len(workers) != 1:
            raise RuntimeError('exactly one vLLM worker acknowledgement required per instance')
        worker = workers[0]
        pid = worker.get('worker_pid')
        instance_id = instance_id or worker.get('instance_id')
        if not instance_id or pid is None:
            raise RuntimeError('instance identity and worker PID are required for GPU mapping')
        uuid = pids.get(int(pid))
        if uuid is None:
            raise RuntimeError('worker GPU UUID not found in live NVIDIA process inventory')
        mapping[instance_id] = uuid
    if len(mapping)!=4 or len(set(mapping.values()))!=4:
        raise RuntimeError('four distinct worker PID to GPU UUID mappings required')
    return mapping


async def episode(*, cfg, service_dir, run_id, episode_id, policy, threshold, trace,
                  session, budget, mapping=None, horizons=None):
    horizons = horizons or RealEnvConfig()
    duration = horizons.arrival_sec + horizons.drain_sec
    bridge = RealBridge(cfg['service']['base_url'], session)
    io = OpportunityLiveEpisodeIO(bridge, session, run_id=run_id, episode_id=episode_id,
        run_dir=service_dir, trace=trace, sampling=cfg['sampling'], arrival_sec=horizons.arrival_sec,
        route_shares=[.7, .1, .1, .1])
    samples, decisions, fs = [], [], []
    context = {}
    def sink(record):
        record['opportunity'] = dict(context['opportunity'])
        decisions.append(record)
        append(io.directory/'decisions.jsonl', record)
    env = RealEnv(bridge, io, budget, run_id=run_id, episode_id=episode_id,
        ledger_factory=io.ledger, config=horizons, decision_sink=sink)
    stop = asyncio.Event()
    errors = []
    heartbeat = None
    def acquired():
        nonlocal heartbeat
        heartbeat = asyncio.create_task(_nonblocking_controller_heartbeat(bridge, stop, errors))
    async def release():
        await _stop_heartbeat(heartbeat, stop)
    bridge.on_controller_acquired = acquired
    bridge.before_controller_release = release
    previous = {}
    initial_counters = {}
    try:
        obs, _ = await env.reset(mode='stay' if policy=='Stay' else 'active',
                                 controller_version='migration-scan-constrained-balanced-v1')
        if mapping is not None and mapping != io.mapping:
            raise RuntimeError('paired instance epochs changed')
        gpu_ids = await asyncio.to_thread(gpu_mapping, json.loads((io.directory/'initial_flush.json').read_text()))
        atomic_json(io.directory/'gpu_mapping.json', gpu_ids)
        while env.state in ('arrival','drain'):
            if errors:
                raise RuntimeError('heartbeat failed: '+str(errors))
            snapshot = env.snapshot
            at = snapshot['received_ns']/1e9
            try:
                gpu = await asyncio.to_thread(gpu_sample)
                gpu_at = time.monotonic_ns()/1e9
            except (OSError, subprocess.SubprocessError) as exc:
                gpu = {}
                gpu_at = None
                append(io.directory/'sampling_errors.jsonl', {'at':at, 'error':str(exc)})
            rows = []
            for state in snapshot['instances']:
                key = state['instance_id']
                stamp = state['measured_ns']/1e9
                fresh = 0 <= at-stamp <= 2.
                counters = state['scan_counters']
                initial_counters.setdefault(key, dict(counters))
                old = previous.get(key)
                rates = {}
                for target, source in [('prefill_tps','prefill_tokens'), ('decode_tps','decode_tokens'),
                                       ('output_tps','output_tokens_total')]:
                    value = state.get(source) if source=='output_tokens_total' else counters.get(source,0)
                    prior = old[1].get(source) if old else None
                    dt = stamp-old[0] if old else 0
                    rates[target] = (value-prior)/dt if fresh and dt>0 and value is not None and prior is not None and value>=prior else None
                previous[key] = (stamp, {**counters, 'output_tokens_total':state.get('output_tokens_total')})
                total = state.get('total_blocks')
                rows.append(dict(instance_id=key, gpu_uuid=gpu_ids[key], measured_ns=state['measured_ns'],
                    kv_pct=100*state['used_blocks']/total if fresh and total else None,
                    **{k:state.get(k) if fresh else None for k in ('running','waiting','head_wait_sec','free_blocks','used_blocks')},
                    **gpu.get(gpu_ids[key], {}), **rates, scan_counters=counters,
                    remaining_steps=remaining_steps(state) if fresh else None))
                value = remaining_steps(state)
                if fresh and value is not None and env.start_ns <= state['measured_ns'] < env.arrival_end_ns:
                    fs.append(value)
            sample = dict(at=at, gpu_at=gpu_at, instances=rows)
            samples.append(sample)
            append(io.directory/'resources.jsonl', sample)
            context['opportunity'] = opportunity(snapshot, obs['raw'], threshold, time.monotonic_ns())
            # Keep the existing pending receipt and slot fences; no second submission.
            if env.unresolved_decision:
                receipt = await bridge.query_receipt(env.unresolved_decision)
                env.receipts[env.unresolved_decision] = receipt
                if receipt.get('terminal_status') in ('succeeded','failed','rejected'):
                    env.unresolved_decision = None
            action = context['opportunity']['action'] if policy!='Stay' and env.unresolved_decision is None else 0
            obs, _, done, _, _ = await env.step(action, policy_version=policy+'-scan-v1')
            if done:
                break
        result = await env.finish()
        atomic_json(io.directory/'episode.json', result)
        if result['status'] != 'passed':
            raise RuntimeError('episode or cleanup audit failed: '+str(result['invalid_reasons']))
        through = min(cp.through_ns for cp in env.ledger.checkpoints.values())
        evidence = list(env.ledger.events.values())
        routes = route_audit(io, env.ledger, evidence, io.mapping)
        rows, _ = env.ledger._rows(occurred_by_ns=through)
        submission = request_submission_audit(rows, [r['request_id'] for r in trace], arrival_end_ns=env.arrival_end_ns)
        audits = env.ledger.export(io.directory, cleanup_through_ns=through)
        if routes['status']!='passed' or submission['status']!='passed' or set(audits.values())!={'passed'}:
            raise RuntimeError('route, submission or export audit failed')
        migrations = reconcile_migrations(decisions, env.receipts, evidence)
        by_decision = {d['decision_id']:d for d in decisions}
        for m in migrations:
            d = by_decision[m['decision_id']]
            m.update(src_id=d['src_id'], dst_id=d['dst_id'])
        if policy=='Stay' and migrations:
            raise RuntimeError('unexpected migration under Stay')
        accepted_server_ids={r['server_request_id'] for r in rows if r['accepted_ns'] is not None
            and r['accepted_ns']<=env.arrival_end_ns}
        if any(m['terminal_status']=='succeeded' and m['actual_request_id'] not in accepted_server_ids for m in migrations):
            raise RuntimeError('successful migration outside accepted cohort')
        metrics = metric_summary(env.ledger)
        completed = [r for r in rows if r['outcome']=='success' and r['client_terminal_ns'] is not None
            and r['client_terminal_ns']<=env.deadline_ns]
        e2e = [(r['client_terminal_ns']-r['sent_ns'])/1e6 for r in completed]
        metrics.update(ttft_p95_ms=_percentile([r['ttft_ms'] for r in completed if r['ttft_ms'] is not None],.95),
                       tpot_p95_ms=_percentile([r['tpot_ms'] for r in completed if r['tpot_ms'] is not None],.95),
                       e2e_mean_ms=sum(e2e)/len(e2e) if e2e else None,
                       e2e_p95_ms=_percentile(e2e,.95), completed_throughput_rps=len(completed)/duration)
        stats = migration_statistics(decisions, migrations, metrics['accepted'], duration)
        summary = dict(status='passed', policy=policy, threshold=threshold, trace_hash=digest(trace),
            mapping=io.mapping, metrics=metrics, migrations=migrations, migration_statistics=stats,
            resources=resource_summary(samples, env.start_ns/1e9, env.deadline_ns/1e9),
            f_values=fs, window=dict(start=env.start_ns/1e9, arrival_end=env.arrival_end_ns/1e9, end=env.deadline_ns/1e9),
            routes=routes, submission=submission,
            waiting_fraction=weighted_summary([(sample['at'], next((float(r['waiting']>0) if r.get('waiting') is not None else None
                    for r in sample['instances'] if r['instance_id']==io.mapping[0][0]),None))
                for sample in samples], env.start_ns/1e9, env.arrival_end_ns/1e9)['mean'],
            counter_deltas={key:{name:value-initial_counters[key].get(name,0)
                for name,value in counters.items() if name!='output_tokens_total' and isinstance(value,(int,float))}
                for key,(_,counters) in previous.items()},
            opportunity_rejections=[d['opportunity'] for d in decisions])
        atomic_json(io.directory/'scan_summary.json', summary)
        await bridge.release_episode(run_id,episode_id)
        return summary
    except BaseException as exc:
        if env.state not in ('new','finished'):
            try:
                atomic_json(io.directory/'cleanup_on_error.json', await env.close())
            except BaseException as cleanup_exc:
                append(io.directory/'errors.jsonl', {'cleanup':str(cleanup_exc)})
        atomic_json(io.directory/'interrupted.json', {'status':'incomplete','reason':type(exc).__name__+':'+str(exc)})
        raise
    finally:
        await release()
        await io.close()
