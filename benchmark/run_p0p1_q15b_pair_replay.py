#!/usr/bin/env python3
"""Replay one frozen Q15-B trace as a fresh, same-trace Stay/Balanced pair."""
from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from benchmark.run_full_completion_balance import ManagedFullCompletionService, _warmup_trace
from benchmark.run_p0p1_fast import (can_start, environment, metrics, opportunity_stats,
    pair_gate, read, rows, validate_config)
from rl_scheduler.cohort_ppo import audit_execution
from rl_scheduler.full_completion import run_episode, wait_service_ready
from rl_scheduler.real_bridge import RealBridge, digest
from rl_scheduler.real_budget import atomic_json


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source_dir: Path, run_dir: Path):
    import yaml
    if run_dir.exists():
        raise FileExistsError(run_dir)
    old_plan = read(source_dir / 'plan.json')
    old_comparison = read(source_dir / 'comparison.json')
    if old_plan.get('seeds', []) != [141103, 141104]:
        raise ValueError('unexpected source seed plan')
    if old_comparison.get('p0', {}).get('passed') is not True:
        raise ValueError('source P0 did not pass')
    episode_id = '07_Q15-B_141104_Balanced'
    source_episode = source_dir / 'runs' / episode_id
    old_e = read(source_episode / 'episode.json')
    old_audit = read(source_episode / 'execution_audit.json')
    if old_e.get('status') != 'passed' or old_audit.get('status') != 'passed':
        raise ValueError('source Balanced episode or audit did not pass')
    trace = read(source_dir / 'traces/Q15-B_141104.json')
    trace_hash = digest(trace)
    if trace_hash != old_plan['trace_hashes']['Q15-B_141104'] or trace_hash != old_e['trace_sha256']:
        raise ValueError('source trace hash mismatch')

    for sub in ('traces', 'runs', 'runtime', 'services'):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    (run_dir / 'traces/Q15-B_141104.json').write_text(json.dumps(trace, separators=(',', ':')))
    cfg = yaml.safe_load((source_dir / 'runtime_config.yaml').read_text())
    srv = yaml.safe_load((source_dir / 'real_llumnix.yaml').read_text())
    argv = cfg['service']['server_argv']
    argv[argv.index('--config-file') + 1] = str(run_dir / 'real_llumnix.yaml')
    cfg['service']['ray_temp_dir'] = '/tmp/p0p1_replay_' + hashlib.sha256(str(run_dir).encode()).hexdigest()[:10]
    (run_dir / 'runtime_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (run_dir / 'real_llumnix.yaml').write_text(yaml.safe_dump(srv, sort_keys=False))
    validate_config(cfg, srv)

    plan = {
        'schema': 'p0p1-q15b-pair-replay-v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_run': str(source_dir.relative_to(ROOT)),
        'source_episode': episode_id,
        'source_trace': 'traces/Q15-B_141104.json',
        'trace_sha256': trace_hash,
        'seed': 141104,
        'candidate': 'Q15-B',
        'rate': 0.6,
        'modes': ['Balanced', 'Stay'],
        'same_trace_for_both_modes': True,
        'fresh_attempt_not_continuation_of_source_budget': True,
        'p0_reference': str(source_dir.relative_to(ROOT) / 'comparison.json'),
        'p0_reference_sha256': sha_file(source_dir / 'comparison.json'),
        'balanced_threshold_rule': 'min(actual total_blocks_i)/8 after idle warmup',
        'no_training': True,
        'model': cfg['service']['model_path'],
        'input_tokens': 128,
        'output_tokens': 1536,
        'request_count': 128,
        'hot_fraction': 0.9,
        'budget_sec': 2700,
        'hard_abort_sec': 3600,
        'runtime_config_sha256': sha_file(run_dir / 'runtime_config.yaml'),
        'code_hashes': {
            'benchmark/run_p0p1_q15b_pair_replay.py': sha_file(Path(__file__)),
            'benchmark/run_p0p1_fast.py': sha_file(ROOT / 'benchmark/run_p0p1_fast.py'),
            'rl_scheduler/full_completion.py': sha_file(ROOT / 'rl_scheduler/full_completion.py'),
        },
    }
    atomic_json(run_dir / 'plan.json', plan)
    atomic_json(run_dir / 'source_pair_reference.json', {
        'episode_id': episode_id,
        'metrics': old_e.get('primary_metrics'),
        'execution_audit': old_audit,
        'opportunity': read(source_episode / 'opportunity.json'),
    })
    return plan


async def execute(run_dir: Path):
    import aiohttp
    import yaml
    from tokenizers import Tokenizer

    plan = read(run_dir / 'plan.json')
    for rel, expected in plan['code_hashes'].items():
        if sha_file(ROOT / rel) != expected:
            raise RuntimeError('code changed after plan: ' + rel)
    if sha_file(run_dir / 'runtime_config.yaml') != plan['runtime_config_sha256']:
        raise RuntimeError('runtime config changed after plan')
    source_dir = ROOT / plan['source_run']
    if sha_file(source_dir / 'comparison.json') != plan['p0_reference_sha256']:
        raise RuntimeError('P0 reference changed after plan')
    cfg = yaml.safe_load((run_dir / 'runtime_config.yaml').read_text())
    env = {'before': environment()}
    atomic_json(run_dir / 'environment_audit.json', env)
    service = ManagedFullCompletionService(run_dir)
    started_wall = time.time()
    budget_start = time.monotonic()
    state = {'status': 'running', 'started_wall': started_wall, 'budget_origin_wall': started_wall}
    atomic_json(run_dir / 'run_state.json', state)
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    async def stop_service_on_signal():
        await stop.wait()
        await asyncio.to_thread(service.stop)
    signal_task = asyncio.create_task(stop_service_on_signal())
    async def hard_budget_watchdog():
        await asyncio.sleep(max(0., 3600. - (time.monotonic() - budget_start)))
        state.update(status='incomplete_budget', reason='60 minute hard abort')
        await asyncio.to_thread(service.stop)
    budget_task = asyncio.create_task(hard_budget_watchdog())
    durations = []
    formal = 0
    mapping = None
    threshold = None
    try:
        await asyncio.to_thread(service.start)
        await asyncio.wait_for(wait_service_ready(cfg, service.is_running), timeout=480.)
        timeout = aiohttp.ClientTimeout(total=None, connect=5, sock_connect=5)
        async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(keepalive_timeout=3.)) as session:
            async def episode(name, policy, trace):
                nonlocal mapping, formal
                if name != '00_warmup_141100_Stay':
                    if not can_start(time.monotonic() - budget_start, durations):
                        raise TimeoutError('insufficient remaining budget for a complete episode')
                    formal += 1
                if stop.is_set():
                    raise RuntimeError('user_stop')
                before = time.monotonic()
                print(json.dumps({'event': 'episode_start', 'id': name,
                    'elapsed_sec': before - budget_start}), flush=True)
                e = await run_episode(cfg=cfg, run_id=run_dir.name, episode_id=name,
                    policy=policy, trace=trace, run_dir=run_dir, control_session=session,
                    frozen_mapping=mapping, service_probe=service.is_running, stop_owned_service=service.stop,
                    balanced_threshold=threshold or 9000., no_progress_timeout_sec=120.,
                    hard_stop_monotonic=budget_start + 3600.)
                elapsed = time.monotonic() - before
                if policy != 'Stay' or name != '00_warmup':
                    durations.append(elapsed)
                mapping = e['frozen_mapping']
                local = run_dir / 'runs' / name
                full = read(local / 'episode.json')
                audit = audit_execution(full, rows(local / 'decisions.jsonl'), rows(local / 'events.jsonl'),
                    rows(local / 'ppo_action_intents.jsonl'))
                if policy == 'Stay' and (audit['coordinator_attempts'] or e['actual_migration_attempts']):
                    audit = {'status': 'failed', 'error': 'Stay had actual migrations'}
                atomic_json(local / 'execution_audit.json', audit)
                (local / 'migrations.jsonl').write_text(''.join(
                    json.dumps(r) + '\n' for r in audit.get('reconciled_migrations', [])))
                with (run_dir / 'episode_ledger.jsonl').open('a') as f:
                    f.write(json.dumps({'episode_id': name, 'candidate': 'Q15-B', 'seed': 141104,
                        'mode': policy, 'wall_sec': elapsed}) + '\n')
                    f.flush(); os.fsync(f.fileno())
                print(json.dumps({'event': 'episode_end', 'id': name, 'status': e['status'],
                    'audit': audit['status'], 'wall_sec': elapsed}), flush=True)
                if e['status'] != 'passed' or audit['status'] != 'passed':
                    raise RuntimeError('episode or audit failed; stop this pair')
                return e

            tokenizer = Tokenizer.from_file(str(Path(cfg['service']['model_path']) / 'tokenizer.json'))
            await episode('00_warmup_141100_Stay', 'Stay', _warmup_trace(tokenizer))
            snap = await RealBridge(cfg['service']['base_url'], session).snapshot()
            capacities = [s.get('total_blocks', 0) for s in snap['instances']]
            if len(capacities) != 4 or min(capacities) <= 0:
                raise RuntimeError('unknown or zero block capacity')
            threshold = min(capacities) / 8
            plan['balanced_threshold'] = threshold
            plan['capacity_snapshot'] = snap
            plan['threshold_frozen_wall'] = time.time()
            atomic_json(run_dir / 'plan.json', plan)
            env['after_warmup'] = environment()
            env['capacity_snapshot'] = snap
            atomic_json(run_dir / 'environment_audit.json', env)
            trace = read(run_dir / plan['source_trace'])
            if digest(trace) != plan['trace_sha256']:
                raise RuntimeError('replay trace hash changed')
            await episode('01_Q15-B_141104_Balanced', 'Balanced', trace)
            await episode('02_Q15-B_141104_Stay', 'Stay', trace)

        result = analyze_pair(run_dir)
        state.update(status='single_seed_pair_complete', comparison=result)
    except TimeoutError as exc:
        state.update(status='incomplete_budget', reason=str(exc))
    except BaseException as exc:
        state.update(status='execution_failed', reason=type(exc).__name__ + ': ' + str(exc),
            traceback=traceback.format_exc())
    finally:
        signal_task.cancel(); budget_task.cancel()
        await asyncio.gather(signal_task, budget_task, return_exceptions=True)
        await asyncio.to_thread(service.stop)
        state.update(elapsed_sec=time.monotonic() - budget_start,
            attempt_elapsed_sec=time.monotonic() - budget_start, formal_rounds=formal,
            ended_wall=time.time(), episode_durations_sec=durations)
        env['after_cleanup'] = environment()
        env['service_stopped'] = read(run_dir / 'service_stopped.json') if (run_dir / 'service_stopped.json').exists() else None
        atomic_json(run_dir / 'environment_audit.json', env)
        atomic_json(run_dir / 'run_state.json', state)
        if (run_dir / 'episode_ledger.jsonl').exists():
            analyze_pair(run_dir)
        print(json.dumps(state), flush=True)


def analyze_pair(run_dir: Path):
    plan = read(run_dir / 'plan.json')
    episodes = {}
    output = []
    for item in rows(run_dir / 'episode_ledger.jsonl'):
        if item['episode_id'] == '00_warmup_141100_Stay':
            continue
        e_dir = run_dir / 'runs' / item['episode_id']
        e = read(e_dir / 'episode.json')
        e['execution_audit'] = read(e_dir / 'execution_audit.json')
        e['opportunity'] = opportunity_stats(rows(e_dir / 'decisions.jsonl'), e['migration_receipts'],
            rows(e_dir / 'requests.jsonl'), n=e['planned_request_count'], threshold=plan.get('balanced_threshold'))
        e['opportunity']['descriptive_partial_only'] = e['status'] != 'passed'
        atomic_json(e_dir / 'opportunity.json', e['opportunity'])
        if item['mode'] in ('Stay', 'Balanced'):
            episodes[item['mode']] = e
            m = metrics(e) if e.get('status') == 'passed' and e['execution_audit'].get('status') == 'passed' else {}
            p = e.get('primary_metrics') or {}
            o = e['opportunity']
            output.append({'episode_id': item['episode_id'], 'mode': item['mode'], 'status': e['status'],
                'completed': e.get('measured_request_count', 0), **m,
                'send_lag_p95_ms': p.get('send_lag_ms', {}).get('p95'),
                'control_work_p95_ms': o.get('control_work_ms', {}).get('p95'),
                'N_all': o['N_all'], 'N_opp': o['N_opp'], 'N_forced': o['N_forced'],
                'succeeded': o['succeeded'], 'distinct_requests': o['distinct_requests'],
                'coverage': o['coverage']})
    if output:
        with (run_dir / 'results.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)
    comparison = {'status': 'single_seed_pair_complete', 'seed': 141104, 'trace_sha256': plan['trace_sha256'],
        'same_trace': True, 'p0': {'passed': True, 'source': plan['p0_reference'],
            'scope': 'limited reuse from Q15-A at 1.0 req/s; Q15-B at 0.6 req/s is not an independent P0'},
        'pair': pair_gate([episodes['Stay']] if 'Stay' in episodes else [], episodes.get('Balanced')),
        'recommended': False, 'reason': 'one replayed seed pair is not independent two-seed confirmation'}
    atomic_json(run_dir / 'comparison.json', comparison)
    atomic_json(run_dir / 'training_environment_handoff.json', {
        'recommended': False, 'status': comparison['status'], 'model': plan['model'],
        'evidence': str(run_dir.relative_to(ROOT)), 'candidate': 'Q15-B', 'rate': 0.6,
        'seed': 141104, 'same_trace': True, 'no_PPO_training': True,
        'comparison': 'comparison.json', 'results': output})
    lines = ['# Q15-B 同 trace 配对重放', '',
        '本目录是独立 attempt；旧批次保持不变。Stay 与 Balanced 使用同一冻结 trace，来源为旧批次 Q15-B / seed 141104。',
        '这是一组重放配对，不是新 seed，也不单独构成 PPO 推荐。P0 仅有限复用 Q15-A / 1.0 req/s 的检查。', '',
        '| 模式 | 状态 | mean ms | P95 ms | cost ms | 成功迁移 | 不同请求 | 机会/决策 | 只能 Stay | 控制耗时 P95 ms | 发包延迟 P95 ms |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in output:
        o = episodes[row['mode']]['opportunity']
        lines.append(f"| {row['mode']} | {row['status']} | {row.get('mean')} | {row.get('p95')} | {row.get('cost')} | {row['succeeded']} | {row['distinct_requests']} | {row['N_opp']}/{row['N_all']} | {row['N_forced']} | {row['control_work_p95_ms']} | {row['send_lag_p95_ms']} |")
    lines += ['', f"配对门槛：`{json.dumps(comparison['pair'], ensure_ascii=False)}`",
        '', '没有启动 PPO 训练。完整原始账本在 runs/；阈值、配置、GPU 映射与清理审计见 plan.json 和 environment_audit.json。']
    (run_dir / 'report.md').write_text('\n'.join(lines) + '\n')
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('command', choices=('plan', 'run', 'analyze'))
    args = parser.parse_args()
    source = args.source_dir.resolve(); run_dir = args.run_dir.resolve()
    if args.command == 'plan':
        print(json.dumps(prepare(source, run_dir), indent=2))
    elif args.command == 'run':
        asyncio.run(execute(run_dir))
    else:
        print(json.dumps(analyze_pair(run_dir), indent=2))


if __name__ == '__main__':
    main()
