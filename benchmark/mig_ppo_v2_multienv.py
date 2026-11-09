"""Bounded multi-environment Mig-PPO V2 real screening and calibration.

This module is imported by the existing real and simulator entrypoints. Planning,
reporting, and fitting are CPU-only; only ``collect`` starts its owned service.
"""
from __future__ import annotations

import asyncio
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
ENV_IDS = [f'HOT{gpu}-{level}' for gpu in range(1, 5) for level in ('L', 'M', 'H')] + ['NATIVE']
LEVELS = {'L': .50, 'M': .70, 'H': .90}
EVAL_SEED = 26110521
BACKUP_N_SEED = 26110522
BACKUP_QPS_SEED = 26110523
VALIDATION_SEED = 26110524
EPISODE_HORIZON_SEC = 200.
DECISION_INTERVAL_SEC = 1.5
MIGRATION_COOLDOWN_SEC = 0.
GPU_BUDGET_SEC = 5 * 3600.


def _json(path):
    return json.loads(Path(path).read_text())


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _canonical_hash(value):
    return _sha_bytes(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                 ensure_ascii=False, allow_nan=False).encode())


def _atomic_json(path, value):
    from rl_scheduler.real_budget import atomic_json
    atomic_json(Path(path), value)


def _append_jsonl(path, row):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream, fcntl.LOCK_UN)


def _read_jsonl(path):
    path = Path(path)
    if not path.exists() and path.suffix != '.gz':
        compressed = path.with_name(path.name + '.gz')
        if compressed.exists():
            path = compressed
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _command(*args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=30)
    return {'returncode': result.returncode, 'stdout': result.stdout,
            'stderr': result.stderr}


def _seed_values(value, out):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == 'seed' and type(child) is int:
                out.add(child)
            elif key.endswith('seeds') and isinstance(child, list):
                out.update(x for x in child if type(x) is int)
            _seed_values(child, out)
    elif isinstance(value, list):
        for child in value:
            _seed_values(child, out)


def _registered_seeds():
    used = {}
    results = ROOT / 'benchmark/results'
    relevant_names = {'plan.json', 'manifest.json', 'run_manifest.json', 'data_split.json',
                      'seed_inventory_final.json', 'candidate_ledger.jsonl', 'episode_ledger.jsonl',
                      'attempts.jsonl', 'real_attempts.jsonl'}
    for path in results.rglob('*'):
        if path.name not in relevant_names or not path.is_file():
            continue
        try:
            if path.suffix == '.jsonl':
                values = _read_jsonl(path)
            else:
                values = [_json(path)]
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        found = set()
        for value in values:
            _seed_values(value, found)
        for seed in found:
            used.setdefault(seed, []).append(str(path.relative_to(ROOT)))
    return used


def _trace_base(trace):
    return [{key: value for key, value in row.items()
             if key not in ('initial_instance_index', 'hot_instance_index', 'routing_mode', 'candidate_id')}
            for row in trace]


def _trace_with_routing(trace, environment_id, mode, hot_fraction=None, hot_index=None,
                        permutation_seed=EVAL_SEED):
    result = [dict(row) for row in trace]
    if environment_id == 'NATIVE':
        for row in result:
            row['initial_instance_index'] = None
            row['hot_instance_index'] = None
            row['routing_mode'] = 'native'
        return result
    if hot_fraction is None or hot_index is None:
        raise ValueError('artificial hotspot requires an explicit location and fraction')
    n = len(result)
    rng = random.Random(int(permutation_seed) ^ 0x51A1)
    order = list(range(n))
    rng.shuffle(order)
    hot_count = math.floor(n * float(hot_fraction) + .5)
    routes = [None] * n
    for idx in order[:hot_count]:
        routes[idx] = int(hot_index)
    nonhot = [idx for idx in range(4) if idx != int(hot_index)]
    for offset, idx in enumerate(order[hot_count:]):
        routes[idx] = nonhot[offset % 3]
    for idx, route in enumerate(routes):
        result[idx]['initial_instance_index'] = route
        result[idx]['hot_instance_index'] = int(hot_index)
        result[idx]['routing_mode'] = 'fixed'
    return result


def _make_candidate_trace(*, tokenizer, candidate_id, n, qps, arrival_seed,
                          prompt_seed, environment_id, output_dir, hot_fraction=None,
                          hot_index=None):
    from rl_scheduler.full_completion import make_trace
    family = 'S1' if environment_id != 'NATIVE' else 'C0'
    candidate = {'family': family, 'profile': 'decode1536', 'rho': 1.,
                 'hot_fraction': float(hot_fraction or 0.)}
    prefix = f'trace_{n}_{qps:g}_{prompt_seed}_'
    raw = make_trace(candidate=candidate, n=n, seed=arrival_seed, lambda_ref=qps,
                     tokenizer=tokenizer, episode_id_prefix=prefix,
                     routing_mode='native' if environment_id == 'NATIVE' else 'fixed',
                     hotspot_index=hot_index, prompt_seed=prompt_seed)
    if environment_id == 'NATIVE':
        routed = raw
    else:
        # make_trace's route is replaced by one registered permutation shared
        # across intensity levels and all four physical positions.
        routed = _trace_with_routing(raw, environment_id, 'fixed', hot_fraction,
                                     hot_index, permutation_seed=arrival_seed)
    for row in routed:
        row['candidate_id'] = candidate_id
    full_hash = _canonical_hash(routed)
    base_hash = _canonical_hash(_trace_base(routed))
    trace_path = Path(output_dir)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(trace_path, routed)
    # <run>/traces/{selection,validation}/<environment>/<trace>.json
    run_root = trace_path.parents[3]
    base_path = run_root / 'base_traces' / f'{base_hash}.json'
    if not base_path.exists():
        _atomic_json(base_path, _trace_base(routed))
    hot_count = sum(row.get('initial_instance_index') == hot_index for row in routed) if hot_index is not None else 0
    return {'candidate_id': candidate_id, 'n': n, 'target_qps': qps,
            'arrival_time_domain_sec': n / qps,
            'arrival_generator': 'uniform_order_statistics_fixed_n',
            'arrival_seed': arrival_seed, 'prompt_seed': prompt_seed,
            'trace_path': str(trace_path.relative_to(run_root)),
            'base_trace_path': str(base_path.relative_to(run_root)),
            'base_trace_hash': base_hash, 'full_trace_hash': full_hash,
            'routing_mode': 'native' if environment_id == 'NATIVE' else 'fixed',
            'hotspot_instance_index': hot_index,
            'target_hotspot_fraction': hot_fraction,
            'actual_hotspot_fraction': hot_count / n if n and hot_index is not None else None,
            'hotspot_request_count': hot_count if hot_index is not None else None,
            'route_counts': {str(i): sum(row.get('initial_instance_index') == i for row in routed)
                             for i in range(4)} if hot_index is not None else None}


def _legacy_audit():
    """Index and stream-verify the historical P0/P1/P2/P3 evidence roots."""
    from rl_scheduler.full_completion import cvar95
    roots = [ROOT / 'benchmark/results/p0p1_fast_20261104T104200Z',
             ROOT / 'benchmark/results/p0p1_fast_replay_20261104T124801Z',
             ROOT / 'benchmark/results/p2p3_q15b_20261104T135200Z']
    files, archives = [], []
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.rglob('*')):
            if not path.is_file():
                continue
            rel = str(path.relative_to(ROOT))
            if path.name.endswith('.tar.gz'):
                members = []
                with tarfile.open(path, mode='r|gz') as archive:
                    for member in archive:
                        if not member.isfile():
                            continue
                        stream = archive.extractfile(member)
                        digest = hashlib.sha256(); size = 0
                        if stream:
                            for block in iter(lambda: stream.read(1024 * 1024), b''):
                                digest.update(block); size += len(block)
                        members.append({'name': member.name, 'size': size,
                                        'sha256': digest.hexdigest()})
                archives.append({'path': rel, 'sha256': _sha_file(path),
                                 'members_stream_verified': len(members), 'members': members})
            elif path.suffix in ('.json', '.jsonl', '.gz'):
                if path.suffix == '.gz':
                    h = hashlib.sha256(); size = 0; lines = 0
                    with gzip.open(path, 'rb') as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b''):
                            h.update(block); size += len(block); lines += block.count(b'\n')
                    files.append({'path': rel, 'sha256_compressed': _sha_file(path),
                                  'sha256_uncompressed': h.hexdigest(),
                                  'uncompressed_bytes': size, 'lines': lines,
                                  'gzip_stream_verified': True})
                else:
                    files.append({'path': rel, 'sha256': _sha_file(path), 'bytes': path.stat().st_size})

    p0 = _json(roots[0] / 'plan.json')
    raw_pair = _json(roots[1] / 'comparison.json')
    trace = _json(roots[0] / 'traces/Q15-B_141104.json')
    hot_counts = {str(i): sum(row['initial_instance_index'] == i for row in trace) for i in range(4)}
    recomputed = {}
    for policy in ('Stay', 'Balanced'):
        run_name = '01_Q15-B_141104_Balanced' if policy == 'Balanced' else '02_Q15-B_141104_Stay'
        reqs_path = roots[1] / 'runs' / run_name / 'requests.jsonl'
        if not reqs_path.exists():
            reqs_path = roots[1] / 'runs' / run_name / 'requests.jsonl.gz'
        reqs = _read_jsonl(reqs_path)
        latencies = [float(row['client_e2e_ms']) for row in reqs]
        mean = math.fsum(latencies) / len(latencies)
        tail = cvar95(latencies)
        recomputed[policy] = {'n': len(latencies), 'mean_e2e_ms': mean,
                              'cvar95_e2e_ms': tail, 'C_ms': mean + tail,
                              'successful_output_count': sum(row.get('outcome') == 'success' for row in reqs)}
    gain = (recomputed['Stay']['C_ms'] - recomputed['Balanced']['C_ms']) / recomputed['Stay']['C_ms']
    p2manifest = _json(roots[2] / 'manifest.json')
    split = _json(roots[2] / 'data_split.json')
    model_path = roots[2] / 'calibration/model.json'
    old_model = _json(model_path) if model_path.exists() else None
    archive_bundle = ROOT / 'benchmark/results/p0p1_handoff_20261104T123539Z/p0p1_materials.tar.gz'
    if archive_bundle.exists():
        member_rows = []
        with tarfile.open(archive_bundle, mode='r|gz') as archive:
            for member in archive:
                if member.isfile():
                    stream = archive.extractfile(member); digest = hashlib.sha256(); size = 0
                    if stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b''):
                            digest.update(block); size += len(block)
                    member_rows.append({'name': member.name, 'bytes': size, 'sha256': digest.hexdigest()})
        archives.append({'path': str(archive_bundle.relative_to(ROOT)),
                         'sha256': _sha_file(archive_bundle),
                         'members_stream_verified': len(member_rows), 'members': member_rows})
    return {
        'roots': [str(path.relative_to(ROOT)) for path in roots],
        'indexed_files': files, 'stream_verified_archives': archives,
        'old_q15b': {
            'plan': str((roots[0] / 'plan.json').relative_to(ROOT)),
            'trace_sha256': _canonical_hash(trace), 'trace_route_counts_by_index': hot_counts,
            'hotspot_index_from_seed_rule': 141104 % 4,
            'p0p1_pair_comparison': raw_pair,
            'request_log_recomputation': recomputed,
            'recomputed_gain_C': gain,
            'comparison_gain_C': raw_pair.get('pair', {}).get('relative_improvement', {}).get('cost'),
            'p2p3_split': split, 'p2p3_manifest': p2manifest,
            'old_model_path': str(model_path.relative_to(ROOT)) if model_path.exists() else None,
            'old_model_sha256': _sha_file(model_path) if model_path.exists() else None,
            'old_model': old_model,
        },
        'reuse_decisions': {
            'service_cadence': 'conditional_prior_only; same Qwen2-1.5B and four RTX 3060 topology, must pass new independent per-environment validation',
            'kv_capacity': 'must be re-read at warmup for this run; old 8565 blocks is not frozen input',
            'migration_cost_and_pause': 'historical transfer observations may seed a shared fit; 5s cooldown coverage does not establish 0s repeated-migration behavior or pause time',
            'state_observation_reward_gae': 'not reusable as current contract; old 1s/5s and historical reward/checkpoints remain immutable evidence',
            '141104': 'development-selection evidence only, not independent validation',
            '151101': 'used by historical PPO update; excluded from blind validation',
        },
    }


def _code_paths():
    return [
        'benchmark/run_ppo_real.py', 'benchmark/run_ppo_sim.py',
        'benchmark/mig_ppo_v2_multienv.py', 'benchmark/run_full_completion_balance.py',
        'rl_scheduler/full_completion.py', 'rl_scheduler/full_completion_service.py',
        'rl_scheduler/real_metrics.py', 'rl_scheduler/real_client.py',
        'rl_scheduler/real_opportunity_runner.py', 'rl_scheduler/real_bridge.py',
        'rl_scheduler/real_observation.py', 'rl_scheduler/real_reward.py',
        'rl_scheduler/cohort_ppo.py', 'rl_scheduler/cohort_reward.py',
        'rl_scheduler/q15b_sim.py', 'rl_scheduler/env.py',
        'llumnix/manager.py', 'llumnix/global_scheduler/global_scheduler.py',
        'llumnix/global_scheduler/dispatch_policy.py',
        'llumnix/backends/vllm/llm_engine.py',
        'rl_scheduler/configs/real_pilot.yaml', 'rl_scheduler/configs/real_llumnix.yaml',
        'rl_scheduler/configs/mig_ppo_v2_reward.json',
        'rl_scheduler/docs/mig_ppo_v2_multienv_test_fit_prompt.md',
    ]


def _frozen_matrix_projection(matrix):
    """Return only plan-time fields; execution status is sealed separately."""
    projected = []
    for row in matrix:
        frozen = {key: value for key, value in row.items()
                  if key not in ('status', 'selected_candidate_id', 'selection_passed',
                                 'fit_status', 'independent_validation_status',
                                 'candidate_results', 'selection', 'validation',
                                 'validation_error', 'budget_reason', 'global_failure')}
        candidates = []
        for candidate in frozen.get('candidates', []):
            candidates.append({key: value for key, value in candidate.items()
                               if key not in ('status', 'validation_trace_path',
                                              'validation_base_trace_hash',
                                              'validation_full_trace_hash')})
        frozen['candidates'] = candidates
        projected.append(frozen)
    return projected


def build_plan(run_dir: Path, *, evaluation_seed=EVAL_SEED, validation_seed=VALIDATION_SEED,
               gpu_budget_sec=GPU_BUDGET_SEC, prior_gpu_reservation_sec=0.,
               prior_run_dir: Path | None = None):
    import yaml
    from tokenizers import Tokenizer
    from benchmark.run_p0p1_fast import validate_config
    run_dir = Path(run_dir).resolve()
    if run_dir.exists():
        raise FileExistsError(run_dir)
    planned_seeds = {int(evaluation_seed), int(validation_seed), BACKUP_N_SEED, BACKUP_QPS_SEED}
    if len(planned_seeds) != 4:
        raise ValueError('evaluation, backup, and independent validation seeds must be distinct')
    seed_inventory = _registered_seeds()
    collisions = {seed: seed_inventory[seed] for seed in planned_seeds if seed in seed_inventory}
    if collisions:
        raise RuntimeError('pre-registered seeds collide with prior experiment evidence: ' + json.dumps(collisions))
    prior_attempt = None
    if prior_run_dir is not None:
        prior_run_dir = Path(prior_run_dir).resolve()
        warmup_path = prior_run_dir / 'runs/selection_warmup/episode.json'
        session_path = prior_run_dir / 'service_session.json'
        prior_attempt = {'run_dir': str(prior_run_dir.relative_to(ROOT)),
            'attempts_path': str((prior_run_dir / 'attempts.jsonl').relative_to(ROOT)),
            'attempts_sha256': _sha_file(prior_run_dir / 'attempts.jsonl'),
            'warmup_episode_path': str(warmup_path.relative_to(ROOT)),
            'warmup_episode_sha256': _sha_file(warmup_path),
            'warmup_status': _json(warmup_path).get('status'),
            'warmup_route_audit': _json(warmup_path).get('route_audit'),
            'service_session': _json(session_path),
            'formal_candidate_pairs_started': False,
            'failure': 'GPU PID→UUID join assumed nested event.data; captured journal is flat; service was shut down before any candidate collection',
            'cleanup_budget_write_failure': 'reserved origin was not flushed before entering finally; corrected before this plan'}
    model = ROOT / 'model/Qwen/Qwen2-1.5B'
    required_model_files = ['config.json', 'tokenizer.json', 'model.safetensors']
    missing = [name for name in required_model_files if not (model / name).is_file()]
    if missing:
        raise FileNotFoundError('required Qwen2-1.5B local files missing: ' + ', '.join(missing))
    gpus = _command('nvidia-smi', '--query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.used,utilization.gpu',
                    '--format=csv,noheader')
    if gpus['returncode'] != 0 or len([line for line in gpus['stdout'].splitlines() if line.strip()]) != 4:
        raise RuntimeError('four visible GPUs are required; no reduced GPU/model fallback is permitted')
    active_gpu = _command('nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader')
    if active_gpu['returncode'] != 0 or active_gpu['stdout'].strip():
        raise RuntimeError('GPU is not idle; preserve the active service and wait')
    import subprocess as _sp
    supervisor = _sp.run(['supervisorctl', 'status'], capture_output=True, text=True, timeout=30)
    active_managed = [line for line in supervisor.stdout.splitlines()
                      if any(name in line for name in ('full_completion_', 'p23_'))
                      and ' RUNNING ' in line]
    if active_managed:
        raise RuntimeError('a prior Llumnix experiment is still running: ' + '\n'.join(active_managed))

    run_dir.mkdir(parents=True)
    for sub in ('base_traces', 'traces/selection', 'traces/validation', 'runs', 'comparisons',
                'validation_runs', 'calibration', 'sim_validation', 'services', 'runtime'):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / 'rl_scheduler/configs/real_pilot.yaml').read_text())
    srv = yaml.safe_load((ROOT / 'rl_scheduler/configs/real_llumnix.yaml').read_text())
    cfg['service'].update(ray_num_cpus=16, startup_timeout_sec=480.,
        ray_temp_dir='/tmp/migppo_v2_' + hashlib.sha256(str(run_dir).encode()).hexdigest()[:10],
        request_output_queue_type='rayqueue')
    cfg['service'].setdefault('env', {})['LLUMNIX_DISPATCH_RANDOM_SEED'] = '271828'
    cfg['service']['env']['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    argv = cfg['service']['server_argv']
    argv[argv.index('--config-file') + 1] = str(run_dir / 'real_llumnix.yaml')
    if '--max-num-batched-tokens' not in argv:
        argv.extend(['--max-num-batched-tokens', '2112'])
    srv['SERVER']['REQUEST_OUTPUT_QUEUE_TYPE'] = 'rayqueue'
    srv['MANAGER']['INITIAL_INSTANCES'] = 4
    srv['MANAGER']['DISPATCH_POLICY'] = 'load'
    srv['MANAGER']['DISPATCH_LOAD_METRIC'] = 'remaining_steps'
    srv['MANAGER']['POLLING_INTERVAL'] = .05
    srv['MANAGER']['TOPK_RANDOM_DISPATCH'] = 1
    srv['INSTANCE']['MIGRATION_BACKEND'] = 'rayrpc'
    srv['INSTANCE']['REQUEST_MIGRATION_POLICY'] = 'SR'
    validate_config(cfg, srv)
    (run_dir / 'runtime_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (run_dir / 'real_llumnix.yaml').write_text(yaml.safe_dump(srv, sort_keys=False))
    tokenizer = Tokenizer.from_file(str(model / 'tokenizer.json'))

    matrix = []
    for index, environment_id in enumerate(ENV_IDS):
        if environment_id == 'NATIVE':
            row = {'environment_id': environment_id, 'role': 'test_only', 'routing_mode': 'native',
                   'hotspot_instance_index': None, 'target_hotspot_fraction': None}
        else:
            match = re.fullmatch(r'HOT([1-4])-([LMH])', environment_id)
            hot_index = int(match.group(1)) - 1
            level = match.group(2)
            row = {'environment_id': environment_id, 'role': 'training_candidate',
                   'routing_mode': 'fixed', 'hotspot_instance_index': hot_index,
                   'target_hotspot_fraction': LEVELS[level], 'intensity_id': level}
        candidates = []
        specs = [('primary', 48, .6, int(evaluation_seed)),
                 ('backup_n32_q0p6', 32, .6, BACKUP_N_SEED),
                 ('backup_n48_q1p0', 48, 1., BACKUP_QPS_SEED)]
        for branch, n, qps, arrival_seed in specs:
            candidate_id = f'{environment_id}_{branch}_s{arrival_seed}'
            trace_path = run_dir / 'traces/selection' / environment_id / f'{candidate_id}.json'
            metadata = _make_candidate_trace(tokenizer=tokenizer, candidate_id=candidate_id,
                n=n, qps=qps, arrival_seed=arrival_seed, prompt_seed=int(evaluation_seed),
                environment_id=environment_id, output_dir=trace_path,
                hot_fraction=row.get('target_hotspot_fraction'),
                hot_index=row.get('hotspot_instance_index'))
            metadata.update({'branch': branch, 'status': 'preregistered'})
            candidates.append(metadata)
        row.update({'status': 'not_run', 'candidates': candidates,
                    'selected_candidate_id': None, 'selection_passed': False,
                    'fit_status': 'pending', 'independent_validation_status': 'pending'})
        matrix.append(row)

    code_hashes = {path: _sha_file(ROOT / path) for path in _code_paths()}
    model_identity = {name: _sha_file(model / name) for name in required_model_files}
    run_id = run_dir.name
    run_dir.mkdir(exist_ok=True)
    protocol = {
        'schema': 'mig-ppo-v2-multienv-protocol-v1', 'run_id': run_id,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'stage_order': ['cpu_preflight', 'selection_all_13', 'unified_cpu_fit', 'independent_validation', 'cpu_sim_validation'],
        'single_round_deadline_sec': EPISODE_HORIZON_SEC,
        'max_no_output_progress_sec': 120,
        'decision_interval_sec': DECISION_INTERVAL_SEC,
        'migration_cooldown_sec': MIGRATION_COOLDOWN_SEC,
        'primary': {'n': 48, 'qps': .6, 'prompt_tokens': 128, 'output_tokens': 1536,
                    'arrival_generator': 'uniform_order_statistics_fixed_n'},
        'backup_branch_rule': {'if_any_round_incomplete': 'n32_qps0p6',
                                'else_if_no_positive_migration_gain': 'n48_qps1p0',
                                'at_most_one_branch_per_environment': True},
        'selection_rule': 'complete audited same-trace pair; C=mean(E2E)+exact fractional-tail CVaR95(E2E); gain_C>0 and >=1 audited successful Balanced KV migration',
        'policy_order': 'alternate Stay/Balanced first by environment row; backup uses same environment parity',
        'reward_threshold_rule': 'on selected artificial-hotspot fit snapshots only, classify queue pressure as max(waiting)>0; choose max-Youden threshold on max per-instance KV fraction; ties choose the largest threshold; one-class data means unidentifiable',
        'budget': {'gpu_reserved_wall_limit_sec': float(gpu_budget_sec),
                   'prior_failed_preflight_reservation_sec': float(prior_gpu_reservation_sec),
                   'cpu_soft_limit_sec': 7200, 'cpu_hard_limit_sec': 14400,
                   'pair_start_reserve_sec': 480},
        'no_formal_ppo_training_or_real_update': True,
    }
    (run_dir / 'protocol.md').write_text(
        '# Mig-PPO V2 多环境筛选与拟合协议\n\n'
        f'Run: `{run_id}`。正式顺序为 13 场景完整筛选、统一 CPU 拟合、冻结后新 seed 独立验证。\n\n'
        '主成本 `C=mean(E2E)+CVaR95(E2E)`，尾部分数按精确分数权重计算；采用条件为完整同轨迹配对、`gain_C>0` 且 Balanced 至少一笔有请求/源目标/blocks/终态的成功迁移。\n\n'
        '首选 N=48、0.6 req/s；仅超时走 N=32/0.6，完整但无合格收益走 N=48/1.0。单轮硬窗 200 秒，所有策略统一 1.5 秒决策、0 秒冷却。备选失败则跳过。\n\n'
        '人工热点强度只改变初始路由占比；NATIVE 不发送 `initial_instance_id`，并核对 Manager 原生 dispatch 与首次 engine admission。NATIVE 固定 test_only。\n\n'
        '奖励阈值只在最终选定的人工热点拟合快照中按预注册 Youden 规则校准；数据只有一个标签时保持 pending。验证 trace 不能用于改模型或阈值。正式 PPO 训练、真实 PPO 更新和论文最终评测均不在本轮。\n')

    _atomic_json(run_dir / 'environment_matrix.json', matrix)
    _atomic_json(run_dir / 'data_split.json', {
        'schema': 'mig-ppo-v2-multienv-split-v1',
        'selection_seed': int(evaluation_seed), 'backup_n32_arrival_seed': BACKUP_N_SEED,
        'backup_qps1_arrival_seed': BACKUP_QPS_SEED,
        'independent_validation_seed': int(validation_seed),
        'selection_base_trace_groups': sorted({c['base_trace_hash'] for row in matrix for c in row['candidates']}),
        'validation_base_trace_groups': 'created only after model/reward freeze from independent_validation_seed',
        'shared_evaluation_base_trace_across_hotspot_positions': True,
        'shared_validation_base_trace_across_hotspot_positions': True,
        'no_request_or_time_random_split': True,
        'prior_seed_collisions_checked': True,
    })
    _atomic_json(run_dir / 'environment_audit.json', {
        'phase': 'preflight', 'captured_utc': datetime.now(timezone.utc).isoformat(),
        'gpu_inventory': gpus, 'gpu_processes_before': active_gpu,
        'supervisor_before': {'returncode': supervisor.returncode,
                              'stdout': supervisor.stdout, 'stderr': supervisor.stderr},
        'model_identity': {'path': str(model), 'files_sha256': model_identity},
        'expected_engine': {'model': 'Qwen2-1.5B', 'replicas': 4, 'tensor_parallel_size': 1,
            'dtype': 'half', 'enforce_eager': True, 'max_model_len': 2112,
            'max_num_seqs': 128, 'max_num_batched_tokens': 2112,
            'gpu_memory_utilization': .65, 'ray_num_cpus': 16,
            'queue': 'rayqueue', 'migration_backend': 'rayrpc', 'migration_policy': 'SR'},
        'expected_dispatch': {'policy': 'load', 'metric': 'remaining_steps',
            'topk_random_dispatch': 1, 'state_update_interval_sec': .05,
            'random_seed': 271828, 'routine_migration': 'control lease only'},
    })
    _atomic_json(run_dir / 'reuse_audit.json', _legacy_audit())
    if prior_attempt is not None:
        _atomic_json(run_dir / 'prior_preflight_attempt.json', prior_attempt)
    _atomic_json(run_dir / 'runtime_config.yaml.sha256.json', {
        'runtime_config_sha256': _sha_file(run_dir / 'runtime_config.yaml'),
        'real_llumnix_sha256': _sha_file(run_dir / 'real_llumnix.yaml')})
    _atomic_json(run_dir / 'budget.json', {
        'schema': 'mig-ppo-v2-multienv-budget-v1', 'gpu_budget_limit_sec': float(gpu_budget_sec),
        'prior_gpu_reservation_sec': float(prior_gpu_reservation_sec),
        'cpu_soft_limit_sec': 7200., 'cpu_hard_limit_sec': 14400.,
        'pair_start_reserve_sec': 480., 'gpu_reserved_origin_wall': None,
        'gpu_reserved_elapsed_sec': 0., 'gpu_active_sec': 0., 'gpu_hours': 0.,
        'cpu_elapsed_sec': 0., 'phases': []})
    _atomic_json(run_dir / 'selection_summary.json.pending', {
        'status': 'not_started', 'environment_count': len(matrix), 'environment_ids': ENV_IDS})
    plan = {'schema': 'mig-ppo-v2-multienv-plan-v1', 'run_id': run_id,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'evaluation_seed': int(evaluation_seed), 'backup_n32_seed': BACKUP_N_SEED,
        'backup_qps1_seed': BACKUP_QPS_SEED, 'validation_seed': int(validation_seed),
        'protocol': protocol, 'environment_ids': ENV_IDS,
        'runtime_config_sha256': _sha_file(run_dir / 'runtime_config.yaml'),
        'real_llumnix_sha256': _sha_file(run_dir / 'real_llumnix.yaml'),
        'model_identity': model_identity, 'code_hashes': code_hashes,
        'taskbook_sha256': _sha_file(ROOT / 'rl_scheduler/docs/mig_ppo_v2_multienv_test_fit_prompt.md'),
        'environment_matrix_preregistered_sha256': _canonical_hash(_frozen_matrix_projection(matrix)),
        'preflight_gpu_count': 4, 'evaluation_pair_limit': 26,
        'validation_pair_limit': 13, 'total_cohort_limit': 78,
        'prior_preflight_attempt_sha256': (_sha_file(run_dir / 'prior_preflight_attempt.json')
            if prior_attempt is not None else None),
        'frozen_before_gpu_collection': True}
    plan['freeze_sha256'] = _canonical_hash(plan)
    _atomic_json(run_dir / 'plan.json', plan)
    _atomic_json(run_dir / 'source_index.json', {
        'historical_artifact_index': 'reuse_audit.json:indexed_files/stream_verified_archives',
        'selection_candidates': [{k: c[k] for k in ('candidate_id', 'trace_path', 'base_trace_path',
            'base_trace_hash', 'full_trace_hash', 'arrival_seed', 'prompt_seed')}
            for row in matrix for c in row['candidates']],
        'validation_seed': int(validation_seed), 'validation_traces_materialized_after_freeze': True})
    (run_dir / 'attempts.jsonl').touch()
    (run_dir / 'results.csv').write_text('environment_id,status,candidate_id,n,target_qps,stay_C_ms,balanced_C_ms,gain_C,successful_migrations,distinct_migrated_requests\n')
    return plan


def validate_plan(run_dir: Path):
    run_dir = Path(run_dir).resolve()
    plan = _json(run_dir / 'plan.json')
    verify = dict(plan); expected = verify.pop('freeze_sha256')
    if _canonical_hash(verify) != expected:
        raise RuntimeError('plan freeze hash mismatch')
    if _sha_file(run_dir / 'runtime_config.yaml') != plan['runtime_config_sha256']:
        raise RuntimeError('frozen runtime config changed')
    if _sha_file(run_dir / 'real_llumnix.yaml') != plan['real_llumnix_sha256']:
        raise RuntimeError('frozen Llumnix config changed')
    if _canonical_hash(_frozen_matrix_projection(_json(run_dir / 'environment_matrix.json'))) != plan['environment_matrix_preregistered_sha256']:
        raise RuntimeError('pre-registered environment matrix or candidate traces changed')
    amendment = plan.get('execution_amendment_sha256')
    amendment_path = run_dir / 'execution_amendment.json'
    if amendment and (not amendment_path.is_file() or _sha_file(amendment_path) != amendment):
        raise RuntimeError('execution amendment record changed or is missing')
    for path, digest in plan['code_hashes'].items():
        if _sha_file(ROOT / path) != digest:
            raise RuntimeError('source changed after plan freeze: ' + path)
    if _sha_file(ROOT / 'rl_scheduler/docs/mig_ppo_v2_multienv_test_fit_prompt.md') != plan['taskbook_sha256']:
        raise RuntimeError('taskbook changed after plan freeze')
    for row in _json(run_dir / 'environment_matrix.json'):
        for candidate in row['candidates']:
            if _canonical_hash(_json(run_dir / candidate['trace_path'])) != candidate['full_trace_hash']:
                raise RuntimeError('frozen candidate trace changed: ' + candidate['candidate_id'])
    return plan


def _metrics(episode):
    primary = episode.get('primary_metrics') or {}
    e2e = primary.get('client_e2e_ms') or {}
    if not e2e or not all(e2e.get(key) is not None for key in ('mean', 'p95', 'cvar95')):
        return None
    return {'mean_e2e_ms': float(e2e['mean']), 'p95_e2e_ms': float(e2e['p95']),
            'cvar95_e2e_ms': float(e2e['cvar95']),
            'C_ms': float(e2e['mean']) + float(e2e['cvar95']),
            'ttft_p95_ms': (episode.get('primary_metrics', {}).get('ttft_ms') or {}).get('p95'),
            'tpot_p95_ms': (episode.get('primary_metrics', {}).get('tpot_ms') or {}).get('p95'),
            'n': episode.get('planned_request_count'),
            'cvar_tail_effective_sample_count': .05 * int(episode.get('planned_request_count') or 0)}


def _worker_gpu_mapping(run_dir, instance_mapping):
    """Join engine event PID/instance IDs to the active nvidia-smi UUID list."""
    service = _json(Path(run_dir) / 'service_session.json')
    causal_dir = Path(service['causal_dir'])
    gpu = _command('nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader')
    if gpu['returncode'] != 0:
        raise RuntimeError('cannot read active GPU compute PID mapping')
    pid_gpu = {}
    for line in gpu['stdout'].splitlines():
        fields = [value.strip() for value in line.split(',')]
        if len(fields) >= 2:
            try:
                pid_gpu[int(fields[0])] = {'gpu_uuid': fields[1],
                    'used_memory': fields[2] if len(fields) > 2 else None}
            except ValueError:
                continue
    result = _join_instance_gpu_mapping(instance_mapping, causal_dir, pid_gpu)
    return {'gpu_processes': gpu, 'mapping': result}


def _join_instance_gpu_mapping(instance_mapping, causal_dir, pid_gpu):
    """Join flat or wrapped causal engine events with a captured PID→GPU map."""
    instance_pids = {}
    for path in causal_dir.glob('events*.jsonl'):
        with path.open(encoding='utf-8', errors='replace') as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Causal journals currently serialize event fields at the top
                # level; accept an optional wrapped event for older writers.
                data = row.get('data', row)
                if data.get('event') not in ('request_engine_admitted', 'engine_iteration_completed'):
                    continue
                instance_id, pid = data.get('instance_id'), data.get('pid')
                if instance_id and type(pid) is int and pid in pid_gpu:
                    instance_pids.setdefault(instance_id, {})[pid] = instance_pids.get(instance_id, {}).get(pid, 0) + 1
    result = []
    for index, (instance_id, epoch) in enumerate(instance_mapping):
        choices = instance_pids.get(instance_id, {})
        if not choices:
            raise RuntimeError(f'cannot join instance {instance_id} to an active GPU worker PID')
        pid = max(choices, key=choices.get)
        result.append({'logical_instance_index': index, 'instance_id': instance_id,
            'epoch': epoch, 'worker_pid': pid, **pid_gpu[pid]})
    if len({row['gpu_uuid'] for row in result}) != 4:
        raise RuntimeError('four distinct instance-to-GPU UUID mappings were not observed')
    return result


def _compress_episode(run_dir):
    records = []
    for path in sorted(Path(run_dir).glob('*.jsonl')):
        target = path.with_name(path.name + '.gz')
        digest = hashlib.sha256(); size = 0; lines = 0
        with path.open('rb') as source, target.open('wb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1, mtime=0) as sink:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(block); size += len(block); lines += block.count(b'\n'); sink.write(block)
            raw.flush(); os.fsync(raw.fileno())
        checked = hashlib.sha256(); checked_size = 0
        with gzip.open(target, 'rb') as check:
            for block in iter(lambda: check.read(1024 * 1024), b''):
                checked.update(block); checked_size += len(block)
        if checked.hexdigest() != digest.hexdigest() or checked_size != size:
            target.unlink(missing_ok=True)
            raise RuntimeError('compressed episode journal failed content hash verification: ' + str(path))
        path.unlink()
        records.append({'source': path.name, 'compressed': target.name, 'lines': lines,
                        'uncompressed_bytes': size, 'compressed_bytes': target.stat().st_size,
                        'sha256_uncompressed': digest.hexdigest()})
    if records:
        _atomic_json(Path(run_dir) / 'compression_manifest.json', {'files': records,
            'verification': 'gzip stream and uncompressed SHA-256/byte count'})
    return records


async def _episode(*, cfg, run_dir, run_id, name, policy, trace, frozen_mapping,
                   threshold, service, hard_deadline, traffic_session):
    from rl_scheduler.full_completion import run_episode
    from rl_scheduler.cohort_ppo import audit_execution
    from benchmark.run_p0p1_fast import opportunity_stats
    before = time.monotonic()
    episode_id = name
    _append_jsonl(Path(run_dir) / 'attempts.jsonl', {
        'kind': 'episode_started', 'episode_id': episode_id, 'policy': policy,
        'candidate_id': trace[0].get('candidate_id'), 'started_utc': datetime.now(timezone.utc).isoformat(),
        'n': len(trace), 'planned_horizon_sec': EPISODE_HORIZON_SEC})
    episode = await run_episode(cfg=cfg, run_id=run_id, episode_id=episode_id,
        policy=policy, trace=trace, run_dir=run_dir,
        control_session=traffic_session, frozen_mapping=frozen_mapping,
        service_probe=service.is_running, stop_owned_service=service.stop,
        balanced_threshold=threshold, no_progress_timeout_sec=120.,
        hard_stop_monotonic=hard_deadline,
        episode_horizon_sec=EPISODE_HORIZON_SEC,
        decision_interval_sec=DECISION_INTERVAL_SEC,
        migration_cooldown_sec=MIGRATION_COOLDOWN_SEC)
    elapsed = time.monotonic() - before
    local = Path(run_dir) / 'runs' / episode_id
    full = _json(local / 'episode.json')
    events = _read_jsonl(local / 'events.jsonl')
    decisions = _read_jsonl(local / 'decisions.jsonl')
    if not full.get('route_audit_passed'):
        raise RuntimeError('global initial-route/admission audit failed in ' + episode_id)
    if full.get('cohort_complete') and full.get('status') == 'passed':
        audit = audit_execution(full, decisions, events, [])
    else:
        audit = {'status': 'incomplete', 'reason': 'incomplete cohort is retained as failure evidence',
                 'route_audit_status': full.get('route_audit', {}).get('status')}
    _atomic_json(local / 'execution_audit.json', audit)
    receipts = full.get('migration_receipts', {})
    migrations = audit.get('reconciled_migrations', [])
    if not migrations:
        migrations = [{'decision_id': key, **value} for key, value in receipts.items()]
    (local / 'migrations.jsonl').write_text(''.join(json.dumps(row, allow_nan=False, sort_keys=True) + '\n'
                                                      for row in migrations))
    full['execution_audit'] = audit
    full['opportunity'] = opportunity_stats(decisions, receipts, full.get('requests', []),
        n=full['planned_request_count'], threshold=threshold)
    _atomic_json(local / 'opportunity.json', full['opportunity'])
    _append_jsonl(Path(run_dir) / 'attempts.jsonl', {
        'kind': 'episode_finished', 'episode_id': episode_id, 'policy': policy,
        'status': full.get('status'), 'cohort_complete': full.get('cohort_complete'),
        'execution_audit_status': audit.get('status'), 'route_audit_status': full.get('route_audit', {}).get('status'),
        'wall_sec': elapsed, 'episode_timeout_reason': full.get('episode_timeout_reason'),
        'migration_attempts': full.get('actual_migration_attempts'),
        'completed_requests': full.get('measured_request_count'),
        'ended_utc': datetime.now(timezone.utc).isoformat()})
    _compress_episode(local)
    if not service.is_running() and full.get('episode_timeout_reason') is None:
        raise RuntimeError('owned Llumnix service stopped unexpectedly during ' + episode_id)
    return full, elapsed


async def _collect_stage(run_dir, stage):
    import yaml
    import aiohttp
    from benchmark.run_full_completion_balance import ManagedFullCompletionService, _warmup_trace
    from rl_scheduler.full_completion import wait_service_ready
    from rl_scheduler.real_bridge import RealBridge
    from tokenizers import Tokenizer
    run_dir = Path(run_dir).resolve()
    plan = validate_plan(run_dir)
    if stage not in ('selection', 'validation'):
        raise ValueError('stage must be selection or validation')
    if stage == 'selection':
        if (run_dir / 'selection_summary.json').exists():
            prior = _json(run_dir / 'selection_summary.json')
            if prior.get('status') != 'partial' or not plan.get('execution_amendment_sha256'):
                raise FileExistsError('selection summary already frozen; no retries are allowed')
    else:
        if not (run_dir / 'selection_summary.json').exists() or not (run_dir / 'model_registry.json').exists():
            raise RuntimeError('independent validation is locked until selection and CPU fitting are frozen')
        if not _json(run_dir / 'model_registry.json').get('models_frozen'):
            raise RuntimeError('model registry is not frozen')
        if not (run_dir / 'reward_frozen.json').exists():
            raise RuntimeError('reward/threshold freeze record is missing')

    budget = _json(run_dir / 'budget.json')
    now_wall = time.time()
    if budget.get('gpu_reserved_origin_wall') is None:
        budget['gpu_reserved_origin_wall'] = now_wall
    budget_start_wall = float(budget['gpu_reserved_origin_wall'])
    _atomic_json(run_dir / 'budget.json', budget)
    budget_start_mono = time.monotonic() - max(0., now_wall - budget_start_wall)
    hard_deadline = budget_start_mono + float(budget['gpu_budget_limit_sec'])
    if time.monotonic() >= hard_deadline:
        raise TimeoutError('five-hour GPU reservation wall limit reached')
    cfg = yaml.safe_load((run_dir / 'runtime_config.yaml').read_text())
    tokenizer = Tokenizer.from_file(str(Path(cfg['service']['model_path']) / 'tokenizer.json'))
    matrix_path = run_dir / 'environment_matrix.json'
    matrix = _json(matrix_path)
    service = ManagedFullCompletionService(run_dir)
    service_started = None
    mapping = None
    elapsed_episodes = []
    global_failure = None
    prior_active = _command('nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader')
    if prior_active['stdout'].strip():
        raise RuntimeError('unowned GPU process appeared; do not stop or interfere with it')
    try:
        service_started = time.monotonic()
        service.start()
        remaining = min(480., max(0., hard_deadline-time.monotonic()))
        await asyncio.wait_for(wait_service_ready(cfg, service.is_running), timeout=remaining)
        control_timeout = aiohttp.ClientTimeout(total=10., connect=3., sock_connect=3.)
        traffic_timeout = aiohttp.ClientTimeout(total=None, connect=5., sock_connect=5.)
        async with aiohttp.ClientSession(timeout=control_timeout,
                connector=aiohttp.TCPConnector(limit=4096, keepalive_timeout=3.)) as control_session, \
                aiohttp.ClientSession(timeout=traffic_timeout,
                connector=aiohttp.TCPConnector(limit=4096, keepalive_timeout=3.)) as traffic_session:
            warm = _warmup_trace(tokenizer)
            # The warmup is a separate, non-scored cohort under the same control contract.
            warmup_name = f'{stage}_warmup'
            warmup_index = 1
            while (run_dir / 'runs' / warmup_name).exists():
                warmup_name = f'{stage}_warmup_resume_{warmup_index:02d}'
                warmup_index += 1
            warm_e, warm_sec = await _episode(cfg=cfg, run_dir=run_dir, run_id=plan['run_id'],
                name=warmup_name, policy='Stay', trace=warm, frozen_mapping=None,
                threshold=0., service=service, hard_deadline=hard_deadline,
                traffic_session=traffic_session)
            if warm_e.get('status') != 'passed' or not warm_e.get('cohort_complete'):
                raise RuntimeError('warmup cohort failed lifecycle or completion audit')
            mapping = warm_e['frozen_mapping']
            threshold = 0.
            capacity_snapshot = await RealBridge(cfg['service']['base_url'], control_session).snapshot()
            capacities = [int(state.get('total_blocks') or 0) for state in capacity_snapshot['instances']]
            if len(capacities) != 4 or min(capacities) <= 0:
                raise RuntimeError('four measured nonzero KV capacities are required')
            threshold = min(capacities) / 8.
            worker_map = _worker_gpu_mapping(run_dir, mapping)
            baseline = _json(run_dir / 'environment_audit.json')
            prior_service_mappings = baseline.get('service_mappings_by_stage', {})
            if 'selection' in prior_service_mappings:
                old_uuids = [row['gpu_uuid'] for row in prior_service_mappings['selection']['mapping']]
                new_uuids = [row['gpu_uuid'] for row in worker_map['mapping']]
                if set(old_uuids) != set(new_uuids):
                    raise RuntimeError('service restart did not recover the same four GPU UUIDs')
                if old_uuids != new_uuids:
                    uuid_by_instance = {row['instance_id']: row['gpu_uuid']
                                        for row in worker_map['mapping']}
                    epoch_by_instance = {row['instance_id']: row['epoch']
                                         for row in worker_map['mapping']}
                    instance_by_uuid = {uuid: instance_id
                                        for instance_id, uuid in uuid_by_instance.items()}
                    mapping = [[instance_by_uuid[gpu_uuid],
                                epoch_by_instance[instance_by_uuid[gpu_uuid]]]
                               for gpu_uuid in old_uuids]
                    worker_map = _worker_gpu_mapping(run_dir, mapping)
                    aligned_uuids = [row['gpu_uuid'] for row in worker_map['mapping']]
                    if aligned_uuids != old_uuids:
                        raise RuntimeError('could not restore the frozen logical instance-to-GPU UUID mapping')
            log_path = run_dir / 'api_server.log'
            engine_args_lines = []
            if log_path.exists():
                with log_path.open(encoding='utf-8', errors='replace') as log_stream:
                    for line in log_stream:
                        if 'engine_args:' in line:
                            engine_args_lines.append(line.rstrip())
            if not engine_args_lines:
                raise RuntimeError('API server did not log effective engine_args for the environment audit')
            engine_args = engine_args_lines[-1]
            required_engine_fields = [
                str(Path(cfg['service']['model_path'])), 'tensor_parallel_size=1',
                "dtype='half'", 'max_model_len=2112', 'max_num_seqs=128',
                'max_num_batched_tokens=2112', 'gpu_memory_utilization=0.65',
                'enforce_eager=True']
            engine_args_audit = {'line': engine_args,
                'required_fields': required_engine_fields,
                'passed': all(field in engine_args for field in required_engine_fields)}
            if not engine_args_audit['passed']:
                raise RuntimeError('effective engine args do not match the frozen Qwen2-1.5B configuration')
            if stage == 'selection' and 'selection' in prior_service_mappings:
                prior_service_mappings.setdefault('selection_continuations', []).append(worker_map)
            else:
                prior_service_mappings[stage] = worker_map
            baseline.update({'phase': stage, 'service_started_utc': datetime.now(timezone.utc).isoformat(),
                'service_session': _json(run_dir / 'service_session.json'),
                'frozen_mapping': mapping, 'capacity_snapshot': capacity_snapshot,
                'actual_total_blocks_by_instance': capacities,
                'balanced_threshold_F_min_over_8': threshold,
                'worker_pid_gpu_uuid_mapping': worker_map,
                'service_mappings_by_stage': prior_service_mappings,
                'effective_engine_args': engine_args_audit,
                'actual_runtime_versions': {k: _command(str(ROOT / '.venv/bin/python'), '-c',
                    f"import importlib.metadata;print(importlib.metadata.version('{k}'))")
                    for k in ('torch', 'ray', 'vllm', 'tokenizers')}})
            _atomic_json(run_dir / 'environment_audit.json', baseline)

            if stage == 'selection':
                for index, env_row in enumerate(matrix):
                    # A continuation may only visit rows that have never started.
                    # Completed rows remain immutable, and a partial candidate is
                    # never silently replayed.
                    if env_row.get('status') not in ('not_run', 'not_run_budget',
                                                      'not_run_global_audit_failure'):
                        continue
                    if time.monotonic() + 480. >= hard_deadline:
                        env_row['status'] = 'not_run_budget'
                        for later in matrix[index+1:]:
                            later['status'] = 'not_run_budget'
                        break
                    candidates = env_row['candidates']
                    primary = candidates[0]
                    candidate_results = list(env_row.get('candidate_results') or [])
                    order = ('Stay', 'Balanced') if index % 2 == 0 else ('Balanced', 'Stay')

                    async def run_candidate(candidate, branch):
                        nonlocal mapping
                        # Reserve eight minutes for complete collection plus stop/flush cleanup.
                        recent = [x for x in elapsed_episodes[-2:] if x > 0]
                        estimate_pair = max(400., 2 * max(recent, default=200.) + 30.)
                        if hard_deadline-time.monotonic() < estimate_pair + 480.:
                            raise TimeoutError('insufficient GPU budget for another full pair and cleanup reserve')
                        trace = _json(run_dir / candidate['trace_path'])
                        comparison = {'environment_id': env_row['environment_id'],
                            'candidate_id': candidate['candidate_id'], 'branch': branch,
                            'candidate_metadata': candidate, 'same_trace_for_both_methods': True,
                            'base_trace_hash': candidate['base_trace_hash'],
                            'full_trace_hash': candidate['full_trace_hash'],
                            'order': order, 'episodes': {}, 'status': 'running'}
                        for policy in order:
                            episode_id = f"{env_row['environment_id']}_{candidate['candidate_id']}_{policy.lower()}"
                            result, duration = await _episode(cfg=cfg, run_dir=run_dir,
                                run_id=plan['run_id'], name=episode_id, policy=policy, trace=trace,
                                frozen_mapping=mapping, threshold=threshold, service=service,
                            hard_deadline=hard_deadline, traffic_session=traffic_session)
                            elapsed_episodes.append(duration)
                            mapping = result.get('frozen_mapping', mapping)
                            if result.get('route_audit_passed') is not True:
                                raise RuntimeError('global route/admission audit failed: ' + episode_id)
                            comparison['episodes'][policy] = {
                                'episode_id': episode_id, 'status': result.get('status'),
                                'cohort_complete': result.get('cohort_complete'),
                                'execution_audit_status': result.get('execution_audit', {}).get('status'),
                                'metrics': _metrics(result),
                                'planned_request_count': result.get('planned_request_count'),
                                'completed_request_count': result.get('measured_request_count'),
                                'output_token_total': result.get('output_token_total'),
                                'route_audit': result.get('route_audit'),
                                'decision_interval_sec': result.get('decision_interval_sec'),
                                'migration_cooldown_sec': result.get('migration_cooldown_sec'),
                                'episode_horizon_sec': result.get('episode_horizon_sec'),
                                'episode_timeout_reason': result.get('episode_timeout_reason'),
                                'migration_receipts': result.get('migration_receipts', {}),
                                'opportunity': result.get('opportunity')}
                            if result.get('status') != 'passed' or not result.get('cohort_complete'):
                                comparison['status'] = 'incomplete'
                                comparison['selection_passed'] = False
                                comparison['first_incomplete_policy'] = policy
                                comparison['skipped_policy'] = order[1] if policy == order[0] else None
                                comparison['skipped_reason'] = 'candidate cannot form a complete same-trace pair'
                                break
                            if result.get('execution_audit', {}).get('status') != 'passed':
                                raise RuntimeError('global lifecycle/migration execution audit failed: ' + episode_id)
                            if any(issue in ('migration_terminal_unknown', 'migration_terminal_unconfirmed')
                                   for issue in result.get('control_issues', [])):
                                raise RuntimeError('global migration lifecycle audit failed: ' + episode_id)
                        if comparison['status'] == 'running':
                            stay = comparison['episodes'].get('Stay', {})
                            balanced = comparison['episodes'].get('Balanced', {})
                            sm, bm = stay.get('metrics'), balanced.get('metrics')
                            if sm and bm and sm['C_ms'] > 0:
                                gain = (sm['C_ms'] - bm['C_ms']) / sm['C_ms']
                            else:
                                gain = None
                            receipts = [r for r in balanced.get('migration_receipts', {}).values()
                                        if r.get('terminal_status') == 'succeeded'
                                        and r.get('actual_request_id') and r.get('src_id') and r.get('dst_id')
                                        and int(r.get('copied_blocks') or 0) > 0
                                        and type(r.get('terminal_ns')) is int]
                            distinct = len({r['actual_request_id'] for r in receipts})
                            comparison.update({'status': 'complete', 'stay_C_ms': sm['C_ms'] if sm else None,
                                'balanced_C_ms': bm['C_ms'] if bm else None, 'gain_C': gain,
                                'balanced_successful_migrations': len(receipts),
                                'balanced_distinct_migrated_requests': distinct,
                                'balanced_migration_request_ids': sorted({r['actual_request_id'] for r in receipts}),
                                'selection_passed': bool(gain is not None and gain > 0 and len(receipts) >= 1),
                                'positive_cost_gain_but_no_migration_evidence': bool(gain is not None and gain > 0 and not receipts),
                                'N': len(trace), 'cvar95_tail_effective_sample_count': .05 * len(trace),
                                'uncertainty_note': 'single paired trace; no significance or tail stability claim'})
                        _atomic_json(run_dir / 'comparisons' / f"{candidate['candidate_id']}.json", comparison)
                        return comparison

                    try:
                        if candidate_results:
                            # Recover only from a prior, explicitly recorded
                            # incomplete comparison. Never replay a candidate
                            # whose episode_started event already exists.
                            previous = candidate_results[-1]
                            if previous.get('selection_passed'):
                                chosen = previous
                            else:
                                previous_id = previous.get('candidate_id')
                                if env_row.get('candidate_execution_order') == 'backup_qps1_first':
                                    first = next(c for c in candidates if c['branch'] == 'backup_n48_q1p0')
                                    fallback = primary if previous_id == first['candidate_id'] else None
                                elif previous_id == primary['candidate_id']:
                                    fallback = candidates[1] if previous.get('status') == 'incomplete' else candidates[2]
                                else:
                                    fallback = None
                                if fallback is None or any(x.get('candidate_id') == fallback['candidate_id']
                                                           for x in candidate_results):
                                    chosen = None
                                else:
                                    fallback_result = await run_candidate(fallback, fallback['branch'])
                                    fallback_result['backup_reason'] = 'resume_after_recorded_candidate'
                                    candidate_results.append(fallback_result)
                                    chosen = fallback_result if fallback_result.get('selection_passed') else None
                        elif env_row.get('candidate_execution_order') == 'qps_sweep':
                            # Follow-up NATIVE sweep: each preregistered QPS tier
                            # gets one paired trace, stopping at the first
                            # positive, audited Balanced result.
                            for candidate in candidates:
                                result = await run_candidate(candidate, candidate['branch'])
                                candidate_results.append(result)
                                if result.get('selection_passed'):
                                    chosen = result
                                    break
                        elif env_row.get('candidate_execution_order') == 'backup_qps1_first':
                            first = next(c for c in candidates if c['branch'] == 'backup_n48_q1p0')
                            result = await run_candidate(first, first['branch'])
                            candidate_results.append(result)
                            if result['selection_passed']:
                                chosen = result
                            elif len(candidates) < 3:
                                # A deliberately single-candidate follow-up has
                                # no registered backup branch to run.
                                chosen = None
                            else:
                                # The user's requested backup-first order is an
                                # explicit amendment. If that candidate is not
                                # selected, preserve one paired primary result.
                                fallback = primary
                                fallback_result = await run_candidate(fallback, 'primary_after_backup_first')
                                fallback_result['backup_reason'] = 'qps1_first_not_selected'
                                candidate_results.append(fallback_result)
                                chosen = fallback_result if fallback_result.get('selection_passed') else None
                        else:
                            result = await run_candidate(primary, 'primary')
                            candidate_results.append(result)
                            if result['selection_passed']:
                                chosen = result
                            else:
                                timeout = result.get('status') == 'incomplete'
                                backup_index = 1 if timeout else 2
                                if len(candidates) <= backup_index:
                                    # Some frozen follow-up matrices intentionally
                                    # contain only one candidate. Preserve the
                                    # completed pair and finish this row instead
                                    # of indexing an unregistered backup.
                                    chosen = None
                                else:
                                    backup = candidates[backup_index]
                                    backup_reason = ('primary_incomplete' if timeout else
                                        'primary_complete_without_positive_migration_gain')
                                    backup_result = await run_candidate(backup, backup['branch'])
                                    backup_result['backup_reason'] = backup_reason
                                    candidate_results.append(backup_result)
                                    chosen = backup_result if backup_result.get('selection_passed') else None
                        env_row['candidate_results'] = candidate_results
                        if chosen is not None:
                            env_row['selected_candidate_id'] = chosen['candidate_id']
                            env_row['selection_passed'] = True
                            env_row['status'] = 'selected'
                            env_row['fit_status'] = 'eligible_for_unified_fit'
                            env_row['selection'] = chosen
                        elif env_row['environment_id'] == 'NATIVE':
                            complete = next((item for item in candidate_results if item['status'] == 'complete'), None)
                            if complete:
                                env_row['selected_candidate_id'] = complete['candidate_id']
                                env_row['selection'] = complete
                                env_row['status'] = 'test_only_no_positive_gain'
                                env_row['fit_status'] = 'eligible_for_test_only_fit'
                                env_row['selection_passed'] = False
                            else:
                                env_row['status'] = 'test_only_incomplete'
                                env_row['fit_status'] = 'unavailable'
                        else:
                            env_row['status'] = 'skipped_no_positive_gain'
                            env_row['fit_status'] = 'not_in_fit_set'
                    except TimeoutError as exc:
                        env_row['status'] = 'not_run_budget'
                        env_row['budget_reason'] = str(exc)
                        for later in matrix[index+1:]:
                            later['status'] = 'not_run_budget'
                        break
                    except BaseException as exc:
                        env_row['status'] = 'incomplete_global_failure'
                        env_row['global_failure'] = type(exc).__name__ + ': ' + str(exc)
                        global_failure = {'environment_id': env_row['environment_id'],
                            'error': type(exc).__name__ + ': ' + str(exc),
                            'traceback': traceback.format_exc()}
                        for later in matrix[index+1:]:
                            later['status'] = 'not_run_global_audit_failure'
                        break
                    _atomic_json(matrix_path, matrix)
                    print(json.dumps({'environment': env_row['environment_id'],
                        'status': env_row['status'], 'selected_candidate_id': env_row.get('selected_candidate_id')},
                        ensure_ascii=False), flush=True)

            else:
                summary = _json(run_dir / 'selection_summary.json')
                eligible = [row for row in summary['environments']
                    if row.get('selected_candidate_id') and
                    (row['environment_id'] == 'NATIVE' or row.get('selection_passed'))]
                for index, env_row in enumerate(eligible):
                    if hard_deadline-time.monotonic() < 880.:
                        env_row['independent_validation_status'] = 'not_run_budget'
                        continue
                    candidate = next(c for c in _json(matrix_path)[ENV_IDS.index(env_row['environment_id'])]['candidates']
                                     if c['candidate_id'] == env_row['selected_candidate_id'])
                    hot_fraction = env_row.get('target_hotspot_fraction')
                    hot_index = env_row.get('hotspot_instance_index')
                    trace_path = run_dir / 'traces/validation' / env_row['environment_id'] / f"{candidate['candidate_id']}_validation_s{plan['validation_seed']}.json"
                    validation = _make_candidate_trace(tokenizer=tokenizer,
                        candidate_id=candidate['candidate_id'] + f'_validation_s{plan["validation_seed"]}',
                        n=candidate['n'], qps=candidate['target_qps'], arrival_seed=plan['validation_seed'],
                        prompt_seed=plan['validation_seed'], environment_id=env_row['environment_id'],
                        output_dir=trace_path, hot_fraction=hot_fraction, hot_index=hot_index)
                    candidate['validation_trace_path'] = validation['trace_path']
                    candidate['validation_base_trace_hash'] = validation['base_trace_hash']
                    candidate['validation_full_trace_hash'] = validation['full_trace_hash']
                    if candidate['validation_base_trace_hash'] == candidate['base_trace_hash']:
                        raise RuntimeError('validation trace group collides with selection fit trace')
                    trace = _json(run_dir / validation['trace_path'])
                    order = ('Stay', 'Balanced') if index % 2 == 0 else ('Balanced', 'Stay')
                    record = {'environment_id': env_row['environment_id'],
                        'candidate_id': candidate['candidate_id'], 'validation_seed': plan['validation_seed'],
                        'trace_metadata': validation, 'base_trace_hash': validation['base_trace_hash'],
                        'fit_base_trace_hash': candidate['base_trace_hash'],
                        'order': order, 'episodes': {}, 'status': 'running'}
                    try:
                        for policy in order:
                            episode_id = f"VALIDATION_{env_row['environment_id']}_{candidate['candidate_id']}_{policy.lower()}"
                            result, duration = await _episode(cfg=cfg, run_dir=run_dir,
                                run_id=plan['run_id'], name=episode_id, policy=policy, trace=trace,
                                frozen_mapping=mapping, threshold=threshold, service=service,
                                hard_deadline=hard_deadline, traffic_session=traffic_session)
                            elapsed_episodes.append(duration); mapping = result.get('frozen_mapping', mapping)
                            if result.get('status') != 'passed' or not result.get('cohort_complete'):
                                record['status'] = 'incomplete'
                                record['failed_policy'] = policy
                                break
                            if result.get('route_audit_passed') is not True or result.get('execution_audit', {}).get('status') != 'passed':
                                raise RuntimeError('independent validation route/lifecycle audit failed')
                            record['episodes'][policy] = {'episode_id': episode_id,
                                'status': result['status'], 'metrics': _metrics(result),
                                'route_audit': result.get('route_audit'),
                                'migration_receipts': result.get('migration_receipts', {}),
                                'opportunity': result.get('opportunity')}
                        if record['status'] == 'running':
                            stay = record['episodes']['Stay']['metrics']; balanced = record['episodes']['Balanced']['metrics']
                            record.update({'status': 'complete', 'stay_C_ms': stay['C_ms'],
                                'balanced_C_ms': balanced['C_ms'],
                                'gain_C': (stay['C_ms']-balanced['C_ms'])/stay['C_ms']})
                        _atomic_json(run_dir / 'validation_runs' / f"{env_row['environment_id']}.json", record)
                        env_row['independent_validation_status'] = record['status']
                        env_row['validation'] = record
                    except BaseException as exc:
                        env_row['independent_validation_status'] = 'incomplete_global_failure'
                        env_row['validation_error'] = type(exc).__name__ + ': ' + str(exc)
                        global_failure = {'environment_id': env_row['environment_id'],
                            'error': type(exc).__name__ + ': ' + str(exc),
                            'traceback': traceback.format_exc()}
                        break
                    _atomic_json(run_dir / 'environment_matrix.json', matrix)
                    print(json.dumps({'validation': env_row['environment_id'],
                        'status': env_row['independent_validation_status']}, ensure_ascii=False), flush=True)
        if stage == 'selection':
            prior_summary_path = run_dir / 'selection_summary.json'
            prior_summary_hash = None
            prior_summary_history = []
            if prior_summary_path.exists():
                prior_summary_hash = _sha_file(prior_summary_path)
                history_path = run_dir / 'selection_summary_pre_amendment.json'
                if not history_path.exists():
                    prior_summary_path.replace(history_path)
                else:
                    history_path = run_dir / 'selection_summary_pre_amendment.json'
                prior_summary_history.append({'path': history_path.name,
                    'sha256': _sha_file(history_path), 'status': _json(history_path).get('status')})
            terminal_statuses = {'selected', 'skipped_no_positive_gain', 'skipped_failed_candidate',
                                 'test_only_no_positive_gain', 'test_only_incomplete'}
            selection_complete = all(row.get('status') in terminal_statuses for row in matrix)
            frozen = {'schema': 'mig-ppo-v2-selection-summary-v1',
                'status': 'complete' if not global_failure and selection_complete else 'partial',
                'global_failure': global_failure, 'environments': matrix,
                'evaluation_complete_before_unified_fit': True,
                'fit_started_during_selection': False,
                'selection_frozen_utc': datetime.now(timezone.utc).isoformat(),
                'policy': 'Constrained-Balanced', 'primary_cost': 'mean(E2E)+exact-fractional-tail-CVaR95(E2E)'}
            if plan.get('execution_amendment_sha256'):
                frozen['execution_amendment_sha256'] = plan['execution_amendment_sha256']
            if prior_summary_hash:
                frozen['prior_partial_summary_sha256'] = prior_summary_hash
                frozen['summary_history'] = prior_summary_history
            frozen['selection_sha256'] = _canonical_hash(frozen)
            _atomic_json(run_dir / 'selection_summary.json', frozen)
            (run_dir / 'selection_summary.json.pending').unlink(missing_ok=True)
        else:
            _atomic_json(run_dir / 'environment_matrix.json', matrix)
    finally:
        try:
            await asyncio.to_thread(service.stop)
        finally:
            service_sec = max(0., time.monotonic() - service_started) if service_started else 0.
            budget = _json(run_dir / 'budget.json')
            origin = budget.get('gpu_reserved_origin_wall')
            if origin is None:
                origin = time.time()
                budget['gpu_reserved_origin_wall'] = origin
            budget['gpu_reserved_elapsed_sec'] = max(0., time.time() - float(origin))
            budget['gpu_active_sec'] = float(budget.get('gpu_active_sec', 0.)) + service_sec
            budget['gpu_reserved_total_sec'] = (float(budget.get('prior_gpu_reservation_sec', 0.))
                + budget['gpu_reserved_elapsed_sec'])
            budget['gpu_hours'] = (float(budget.get('prior_gpu_reservation_sec', 0.))
                + budget['gpu_active_sec']) * 4 / 3600.
            budget['phases'].append({'stage': stage, 'service_elapsed_sec': service_sec,
                'ended_utc': datetime.now(timezone.utc).isoformat(),
                'episodes': len(elapsed_episodes), 'episode_wall_seconds': elapsed_episodes})
            _atomic_json(run_dir / 'budget.json', budget)
            after_gpu = _command('nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader')
            supervisor_after = subprocess.run(['supervisorctl', 'status'], capture_output=True,
                                              text=True, timeout=30)
            owned_name = service.name
            owned = [line for line in supervisor_after.stdout.splitlines() if owned_name in line and ' RUNNING ' in line]
            _atomic_json(run_dir / 'cleanup_verification.json', {
                'stage': stage, 'service_stopped': (run_dir / 'service_stopped.json').exists(),
                'owned_supervisor_program_running': owned, 'gpu_processes_after': after_gpu,
                'wall_elapsed_since_gpu_reservation_sec': budget['gpu_reserved_elapsed_sec'],
                'cumulative_reserved_wall_sec': budget['gpu_reserved_total_sec'],
                'gpu_active_sec': budget['gpu_active_sec'], 'gpu_hours_four_cards': budget['gpu_hours'],
                'verified_utc': datetime.now(timezone.utc).isoformat()})
    return {'stage': stage, 'status': 'global_failure' if global_failure else 'completed',
            'global_failure': global_failure, 'episode_count': len(elapsed_episodes),
            'gpu_reserved_elapsed_sec': budget['gpu_reserved_elapsed_sec'],
            'gpu_active_sec': budget['gpu_active_sec'], 'gpu_hours': budget['gpu_hours']}


def collect(run_dir: Path, stage='selection'):
    return asyncio.run(_collect_stage(Path(run_dir), stage))


def _episode_artifacts(run_dir, episode_id):
    directory = Path(run_dir) / 'runs' / episode_id
    episode = _json(directory / 'episode.json')
    decisions = _read_jsonl(directory / 'decisions.jsonl')
    requests = _read_jsonl(directory / 'requests.jsonl')
    events = _read_jsonl(directory / 'events.jsonl')
    return directory, episode, decisions, requests, events


def _fit_one_environment(run_dir, env_row, matrix_row, previous_model):
    selected = env_row.get('selection')
    if not selected or selected.get('status') != 'complete':
        return {'status': 'unavailable', 'reason': 'no complete final selected pair'}
    fit_rows, frame_rows, migrations, request_count = [], [], [], 0
    source_episodes = {}
    for policy in ('Stay', 'Balanced'):
        item = selected['episodes'].get(policy)
        if not item or item.get('status') != 'passed':
            return {'status': 'unavailable', 'reason': f'{policy} fit cohort incomplete'}
        directory, episode, decisions, requests, events = _episode_artifacts(run_dir, item['episode_id'])
        source_episodes[policy] = {'episode_id': item['episode_id'],
            'trace_sha256': episode['trace_sha256'],
            'base_trace_hash': selected['base_trace_hash'],
            'episode_path': str(directory.relative_to(run_dir)),
            'episode_sha256': _sha_file(directory / 'episode.json'),
            'decisions_sha256': _sha_file(directory / 'decisions.jsonl.gz'),
            'requests_sha256': _sha_file(directory / 'requests.jsonl.gz'),
            'events_sha256': _sha_file(directory / 'events.jsonl.gz'),
            'route_audit': episode.get('route_audit'),
            'execution_audit': _json(directory / 'execution_audit.json')}
        request_count += len(requests)
        # Client frames are actual receive timestamps and cumulative token deltas.
        for req in requests:
            increments = req.get('token_increments') or []
            prior_at = None
            for increment in increments:
                at, delta = int(increment[0]), int(increment[1])
                if delta <= 0:
                    continue
                if prior_at is not None and at > prior_at:
                    frame_rows.append({'episode_id': item['episode_id'],
                        'policy': policy, 'request_id': req['request_id'],
                        'delta_tokens': delta, 'frame_elapsed_ms': (at-prior_at)/1e6,
                        'ms_per_token': (at-prior_at)/1e6/delta,
                        'cohort_group': selected['base_trace_hash']})
                prior_at = at
        # Decision snapshots provide measured 1.5s instance-level effective cadence.
        by_instance = {}
        for decision in decisions:
            at = int(decision.get('observation_decision_ns', decision['monotonic_ns']))
            snap = decision['snapshot']
            for state in snap['instances']:
                instance_id = state['instance_id']
                previous = by_instance.get(instance_id)
                by_instance[instance_id] = (at, state)
                if previous is None:
                    continue
                prior_at, prior_state = previous
                dt = (at-prior_at)/1e9
                delta = (state.get('output_tokens_total') or 0) - (prior_state.get('output_tokens_total') or 0)
                running = (float(state.get('running') or 0) + float(prior_state.get('running') or 0)) / 2
                if .1 <= dt <= 5.0 and delta > 0 and running > 0:
                    fit_rows.append({'episode_id': item['episode_id'], 'policy': policy,
                        'instance_id': instance_id, 'dt_sec': dt, 'delta_tokens': delta,
                        'mean_running': running,
                        'effective_ms_per_token': dt * running * 1000. / delta,
                        'mean_waiting': (float(state.get('waiting') or 0) + float(prior_state.get('waiting') or 0)) / 2,
                        'mean_kv_fraction': (float(state.get('used_blocks') or 0) / max(1, float(state.get('total_blocks') or 1))
                            + float(prior_state.get('used_blocks') or 0) / max(1, float(prior_state.get('total_blocks') or 1))) / 2,
                        'cohort_group': selected['base_trace_hash']})
        if policy == 'Balanced':
            for receipt in episode.get('migration_receipts', {}).values():
                start, terminal = receipt.get('migration_start_ns'), receipt.get('terminal_ns')
                blocks = receipt.get('copied_blocks')
                if type(start) is int and type(terminal) is int and terminal >= start and blocks:
                    migrations.append({'migration_id': receipt.get('migration_id'),
                        'request_id': receipt.get('actual_request_id'),
                        'source_id': receipt.get('src_id'), 'target_id': receipt.get('dst_id'),
                        'copied_blocks': int(blocks), 'duration_ms': (terminal-start)/1e6,
                        'terminal_status': receipt.get('terminal_status'),
                        'pause_duration_ms': None, 'pause_measured': False,
                        'cohort_group': selected['base_trace_hash']})
    saved = Path(run_dir) / 'calibration' / env_row['environment_id']
    saved.mkdir(parents=True, exist_ok=True)
    (saved / 'service_samples.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in fit_rows))
    (saved / 'client_frame_samples.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in frame_rows))
    (saved / 'migration_samples.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in migrations))
    capacities = _json(Path(run_dir) / 'environment_audit.json')['actual_total_blocks_by_instance']
    capacity = min(capacities)
    if fit_rows:
        effective = [row['effective_ms_per_token'] for row in fit_rows]
        median = float(sorted(effective)[len(effective)//2])
        frame_values = [row['ms_per_token'] for row in frame_rows]
        frame_median = (float(sorted(frame_values)[len(frame_values)//2]) if frame_values else None)
        service_samples = len(fit_rows)
        service_fit_status = 'refitted'
        edges = [0, 2, 4, 8, 16, 32, 64, 128]
        bucket_medians = [median] * (len(edges)-1)
        ridge = [median, 0., 0.]
    elif previous_model:
        median = float(previous_model.get('global_median_ms', 1.))
        frame_median = None
        service_samples = 0
        service_fit_status = 'reused_prior_unvalidated'
        edges = list(previous_model.get('bucket_edges', [0, 2, 4, 8, 16, 32, 64, 128]))
        bucket_medians = list(previous_model.get('bucket_medians_ms', [median] * (len(edges)-1)))
        ridge = list(previous_model.get('ridge_coefficients', [median, 0., 0.]))
    else:
        return {'status': 'fit_insufficient', 'reason': 'no measured service cadence and no compatible prior'}
    usable_migrations = [row for row in migrations if row['terminal_status'] in ('succeeded', 'failed')]
    migration_status = 'unavailable'
    migration_intercept, migration_per_block = 1., 0.
    if usable_migrations:
        try:
            import numpy as np
            x = np.asarray([[1., row['copied_blocks']] for row in usable_migrations], dtype=float)
            y = np.asarray([row['duration_ms'] for row in usable_migrations], dtype=float)
            coef = np.linalg.lstsq(x, y, rcond=None)[0]
            migration_intercept = max(1., float(coef[0]))
            migration_per_block = max(0., float(coef[1]))
            migration_status = 'refitted' if len({row['copied_blocks'] for row in usable_migrations}) > 1 else 'refitted_single_block_support'
        except Exception:
            migration_status = 'fit_failed'
    elif previous_model:
        migration_intercept = float(previous_model.get('migration_intercept_ms', 1.))
        migration_per_block = float(previous_model.get('migration_ms_per_block', 0.))
        migration_status = 'reused_prior_unvalidated'
    model = {'schema': 'mig-ppo-v2-environment-physical-v1',
        'selected': 'bucket', 'bucket_edges': edges, 'bucket_medians_ms': bucket_medians,
        'global_median_ms': median, 'ridge_coefficients': ridge,
        'migration_intercept_ms': migration_intercept,
        'migration_ms_per_block': migration_per_block,
        'capacity': capacity, 'capacity_by_logical_instance': capacities,
        'block_size_tokens': 16, 'decision_interval_sec': DECISION_INTERVAL_SEC,
        'migration_cooldown_sec': MIGRATION_COOLDOWN_SEC,
        'episode_horizon_sec': EPISODE_HORIZON_SEC,
        'native_dispatch_update_interval_sec': .05,
        'failure_probability': None,
        'routing_mode': env_row['routing_mode'],
        'service_fit_status': service_fit_status, 'migration_fit_status': migration_status,
        'effective_ms_per_token_median': median,
        'client_frame_ms_per_token_median': frame_median,
        'service_sample_count': service_samples, 'client_frame_sample_count': len(frame_rows),
        'migration_sample_count': len(usable_migrations), 'migration_direction_count': len({
            (row['source_id'], row['target_id']) for row in usable_migrations}),
        'migration_pause_model': 'unavailable; total transfer duration is not measured pause',
        'no_random_failure_sampling': True,
        'fit_data': source_episodes,
        'fit_cohort_groups': [selected['base_trace_hash']],
        'limitations': ['one paired trace group; repeated decision/request samples are correlated',
            'single-stage physical summaries do not establish all instance/direction asymmetries',
            'native dispatch fit uses native route and state events; no future route is injected',
            'no failures observed is not a failure-rate estimate']}
    _atomic_json(saved / 'model.json', model)
    _atomic_json(saved / 'fit_report.json', {
        'status': 'fit_available' if service_samples else 'fit_insufficient',
        'model_sha256': _sha_file(saved / 'model.json'),
        'service_effective_interval_ms': {
            'n': len(fit_rows), 'mean': (math.fsum(r['effective_ms_per_token'] for r in fit_rows)/len(fit_rows)
                                          if fit_rows else None),
            'median': median if fit_rows else None,
            'min': min((r['effective_ms_per_token'] for r in fit_rows), default=None),
            'max': max((r['effective_ms_per_token'] for r in fit_rows), default=None)},
        'client_frame_interval_ms_per_token': {'n': len(frame_rows), 'median': frame_median},
        'migration_duration_ms': {'n': len(usable_migrations),
            'median': (float(sorted(r['duration_ms'] for r in usable_migrations)[len(usable_migrations)//2])
                       if usable_migrations else None),
            'pause_samples': 0},
        'bucket_vs_ridge': {'selection_rule': 'one-trace cluster; prefer single global median (lower complexity); ridge retained as diagnostic',
            'bucket_prediction_median': median, 'ridge_coefficients': ridge},
        'source_sample_rows': {'service': len(fit_rows), 'client_frames': len(frame_rows),
                               'migrations': len(migrations)},
        'independent_validation_not_seen_during_model_choice': True,
    })
    return {'status': 'fit_available' if service_samples else 'fit_insufficient',
            'model_path': str((saved / 'model.json').relative_to(run_dir)),
            'model_sha256': _sha_file(saved / 'model.json'),
            'fit_report_path': str((saved / 'fit_report.json').relative_to(run_dir)),
            'service_fit_status': service_fit_status, 'migration_fit_status': migration_status,
            'service_samples': service_samples, 'migration_samples': len(usable_migrations),
            'source_episodes': source_episodes}


def _fit_hotspot_threshold(run_dir, matrix):
    observations = []
    for env_row in matrix:
        if env_row.get('environment_id') == 'NATIVE' or not env_row.get('selection_passed'):
            continue
        selected = env_row['selection']
        for policy in ('Stay', 'Balanced'):
            episode_id = selected['episodes'][policy]['episode_id']
            decisions = _read_jsonl(Path(run_dir) / 'runs' / episode_id / 'decisions.jsonl')
            for row in decisions:
                states = row['snapshot']['instances']
                max_kv = max(float(state.get('used_blocks', 0)) /
                             max(1., float(state.get('total_blocks', 1))) for state in states)
                congested = max(float(state.get('waiting') or 0) for state in states) > 0
                observations.append({'environment_id': env_row['environment_id'],
                    'episode_id': episode_id, 'trace_group': selected['base_trace_hash'],
                    'max_kv_fraction': max_kv, 'queue_congested': congested})
    positives = sum(row['queue_congested'] for row in observations)
    negatives = len(observations) - positives
    if not observations or not positives or not negatives:
        return {'status': 'unidentifiable', 'hotspot_threshold': None,
            'rule': 'maximize Youden J for max(waiting)>0 vs max per-instance KV fraction; tie chooses largest threshold',
            'observations': len(observations), 'positive_queue_snapshots': positives,
            'negative_queue_snapshots': negatives,
            'reason': 'both congestion labels are required; no default threshold substituted',
            'trace_groups': sorted({row['trace_group'] for row in observations})}
    values = sorted({min(.999999, max(0., row['max_kv_fraction'])) for row in observations})
    candidates = sorted(set([0.] + values + [min(.999999, (values[-1]+1.)/2)]))
    scores = []
    for threshold in candidates:
        tp = sum(row['queue_congested'] and row['max_kv_fraction'] >= threshold for row in observations)
        fp = sum((not row['queue_congested']) and row['max_kv_fraction'] >= threshold for row in observations)
        sensitivity, specificity = tp / positives, (negatives-fp) / negatives
        scores.append({'threshold': threshold, 'sensitivity': sensitivity,
                       'specificity': specificity, 'youden_j': sensitivity+specificity-1.})
    best = max(scores, key=lambda row: (row['youden_j'], row['threshold']))
    compatible = [row['threshold'] for row in scores
                  if abs(row['youden_j']-best['youden_j']) <= 1e-12]
    return {'status': 'frozen', 'hotspot_threshold': best['threshold'],
        'rule': 'maximize Youden J for max(waiting)>0 vs max per-instance KV fraction; tie chooses largest threshold',
        'best_point': best, 'equal_score_threshold_range': [min(compatible), max(compatible)],
        'candidate_count': len(scores), 'observations': len(observations),
        'positive_queue_snapshots': positives, 'negative_queue_snapshots': negatives,
        'trace_groups': sorted({row['trace_group'] for row in observations}),
        'independent_cluster_count': len({row['trace_group'] for row in observations}),
        'uncertainty': 'decision snapshots are autocorrelated within one or few trace groups; cluster-level threshold uncertainty is not identified'}


def fit(run_dir: Path):
    run_dir = Path(run_dir).resolve()
    plan = validate_plan(run_dir)
    selection = _json(run_dir / 'selection_summary.json')
    if selection.get('status') not in ('complete', 'partial'):
        raise RuntimeError('selection phase has not been sealed')
    if (run_dir / 'model_registry.json').exists():
        raise FileExistsError('models are already frozen; this phase cannot be refit in place')
    start = time.monotonic(); start_wall = time.time()
    matrix = _json(run_dir / 'environment_matrix.json')
    old_model_path = ROOT / 'benchmark/results/p2p3_q15b_20261104T135200Z/calibration/model.json'
    prior_model = _json(old_model_path) if old_model_path.exists() else None
    registry_rows = []
    for env_row, matrix_row in zip(selection['environments'], matrix):
        if env_row.get('fit_status') not in ('eligible_for_unified_fit', 'eligible_for_test_only_fit'):
            registry_rows.append({'environment_id': env_row['environment_id'],
                'role': env_row['role'], 'selection_passed': bool(env_row.get('selection_passed')),
                'fit_status': 'unavailable', 'reason': env_row.get('status')})
            continue
        result = _fit_one_environment(run_dir, env_row, matrix_row, prior_model)
        registry_rows.append({'environment_id': env_row['environment_id'],
            'role': env_row['role'], 'routing_mode': env_row['routing_mode'],
            'selection_passed': bool(env_row.get('selection_passed')),
            'selected_candidate_id': env_row.get('selected_candidate_id'), **result})

    hotspot = _fit_hotspot_threshold(run_dir, selection['environments'])
    reward_config = _json(ROOT / 'rl_scheduler/configs/mig_ppo_v2_reward.json')
    reward_config['hotspot_threshold'] = hotspot.get('hotspot_threshold')
    reward_frozen = {'schema': 'mig-ppo-v2-reward-freeze-v1',
        'status': 'frozen' if hotspot.get('status') == 'frozen' else 'reward_threshold_pending',
        'parameters': reward_config, 'hotspot_threshold_calibration': hotspot,
        'selection_sha256': selection['selection_sha256'],
        'threshold_data_role': 'selected artificial-hotspot fit pairs only',
        'validation_data_used': False, 'native_test_data_used': False,
        'frozen_before_independent_validation': True,
        'normalizer_note': 'fixed shared feature scales; reward reference scales do not auto-scale with N'}
    _atomic_json(run_dir / 'reward_frozen.json', reward_frozen)
    from rl_scheduler.cohort_ppo import normalizer_definition
    normalizer = normalizer_definition()
    normalizer_frozen = {'schema': normalizer['schema'], 'definition': normalizer,
        'instance_feature_count': len(normalizer['instance_features']),
        'pair_feature_count': len(normalizer['pair_features']),
        'global_feature_count': len(normalizer['global_features']),
        'action_count': 13, 'instance_count': 4,
        'normalizer_sha256': _canonical_hash(normalizer),
        'reward_scale_reference': {key: reward_config[key] for key in
            ('n_ref', 'q_ref', 'q_max_ref', 'failure_ref')},
        'request_count_scaling': 'none'}
    _atomic_json(run_dir / 'normalizer_frozen.json', normalizer_frozen)

    reward_reconstruction = {'schema': 'mig-ppo-v2-reward-reconstruction-v1',
        'status': 'blocked_threshold_pending', 'episodes': {},
        'rule': 'reconstruct from actual request token frames, saved snapshots, actions, receipts, and true terminal boundary'}
    if reward_frozen['status'] == 'frozen':
        from rl_scheduler.cohort_reward import MigPPOV2Reward
        reward = MigPPOV2Reward.from_state_dict({'schema': 'mig-ppo-v2', **reward_config})
        reward_reconstruction['status'] = 'complete'
        for env_row in selection['environments']:
            if not env_row.get('selection_passed'):
                continue
            entry = {}
            for policy in ('Stay', 'Balanced'):
                episode_id = env_row['selection']['episodes'][policy]['episode_id']
                _, episode, decisions, _requests, _events = _episode_artifacts(run_dir, episode_id)
                end = max(row['client_terminal_ns'] for row in episode['requests'])
                eligible = [row for row in decisions
                            if row.get('observation_decision_ns', row['monotonic_ns']) < end]
                times = [int(row.get('observation_decision_ns', row['monotonic_ns'])) for row in eligible]
                result = reward.reconstruct(episode, times, eligible)
                if not math.isclose(result['return'], math.fsum(row['reward'] for row in result['windows']),
                                    rel_tol=0., abs_tol=1e-10):
                    raise RuntimeError('V2 reward failed exact reconstruction for ' + episode_id)
                path = run_dir / 'calibration' / env_row['environment_id'] / f'reward_{policy.lower()}.json'
                _atomic_json(path, result)
                entry[policy] = {'path': str(path.relative_to(run_dir)),
                    'sha256': _sha_file(path), 'return': result['return'],
                    'window_count': len(result['windows']),
                    'terminal_potential': result['terminal_potential'],
                    'max_gamma_identity_error': max((abs(row['gamma_t'] - reward.gamma_per_sec**row['dt_sec'])
                                                      for row in result['windows']), default=0.),
                    'max_lambda_identity_error': max((abs(row['lambda_t'] - reward.lambda_per_sec**row['dt_sec'])
                                                       for row in result['windows']), default=0.)}
            reward_reconstruction['episodes'][env_row['environment_id']] = entry
    _atomic_json(run_dir / 'reward_reconstruction.json', reward_reconstruction)

    source_index = _json(run_dir / 'source_index.json')
    registry = {'schema': 'mig-ppo-v2-environment-registry-v1',
        'models_frozen': True, 'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'selection_sha256': selection['selection_sha256'],
        'plan_freeze_sha256': plan['freeze_sha256'],
        'code_hashes': plan['code_hashes'], 'runtime_config_sha256': plan['runtime_config_sha256'],
        'old_q15b_model': {'path': str(old_model_path.relative_to(ROOT)),
            'sha256': _sha_file(old_model_path) if old_model_path.exists() else None,
            'used_as_prior_when_local_samples_missing': True,
            'independent_validation_claim': False},
        'environment_models': registry_rows,
        'accepted_positive_gain_environments': [row['environment_id'] for row in registry_rows
            if row.get('selection_passed') and row.get('fit_status') == 'fit_available'],
        'native_role': 'test_only; excluded from PPO pretraining',
        'reward_threshold_status': reward_frozen['status'],
        'reward_frozen_sha256': _sha_file(run_dir / 'reward_frozen.json'),
        'normalizer_sha256': normalizer_frozen['normalizer_sha256'],
        'selection_fit_trace_groups': source_index.get('selection_base_trace_groups'),
        'independent_validation_seed': plan['validation_seed'],
        'independent_validation_materialized_after_freeze': True,
        'unavailable_support': [],
        'status': 'frozen_with_limitations'}
    _atomic_json(run_dir / 'model_registry.json', registry)
    budget = _json(run_dir / 'budget.json')
    budget['cpu_elapsed_sec'] = float(budget.get('cpu_elapsed_sec', 0.)) + (time.monotonic()-start)
    budget.setdefault('phases', []).append({'stage': 'unified_cpu_fit',
        'elapsed_sec': time.monotonic()-start, 'started_utc': datetime.fromtimestamp(start_wall, timezone.utc).isoformat(),
        'ended_utc': datetime.now(timezone.utc).isoformat()})
    _atomic_json(run_dir / 'budget.json', budget)
    if budget['cpu_elapsed_sec'] > 14400.:
        registry['status'] = 'cpu_budget_exceeded'
        _atomic_json(run_dir / 'model_registry.json', registry)
    return {'status': 'fit_complete', 'environment_models': len(registry_rows),
        'reward_threshold_status': reward_frozen['status'],
        'models_frozen': True, 'cpu_elapsed_sec': budget['cpu_elapsed_sec']}


def validate_simulation(run_dir: Path):
    """Run CPU closed-loop Stay/Balanced validation on the independent real traces."""
    from rl_scheduler.q15b_sim import Q15BCohortSim
    run_dir = Path(run_dir).resolve()
    plan = validate_plan(run_dir)
    registry = _json(run_dir / 'model_registry.json')
    selection = _json(run_dir / 'selection_summary.json')
    models = {row['environment_id']: row for row in registry['environment_models']}
    previous = _json(run_dir / 'sim_validation_summary.json') if (run_dir / 'sim_validation_summary.json').exists() else None
    if previous:
        raise FileExistsError('simulation validation is write-once after model freeze')
    started = time.monotonic()
    results = []
    reward_frozen = _json(run_dir / 'reward_frozen.json')
    reward = None
    if reward_frozen.get('status') == 'frozen':
        from rl_scheduler.cohort_reward import MigPPOV2Reward
        reward = MigPPOV2Reward.from_state_dict({'schema': 'mig-ppo-v2', **reward_frozen['parameters']})
    matrix_by_id = {row['environment_id']: row for row in _json(run_dir / 'environment_matrix.json')}
    for env_row in selection['environments']:
        environment_id = env_row['environment_id']
        model_row = models.get(environment_id, {})
        validation = env_row.get('validation') or {}
        if (model_row.get('status') != 'fit_available' or validation.get('status') != 'complete'
                or not matrix_by_id[environment_id].get('candidates')):
            results.append({'environment_id': environment_id,
                'status': 'not_run', 'physical_fit_passed': False,
                'reason': 'requires a measured physical model and a complete independent validation pair'})
            continue
        candidate = next((item for item in matrix_by_id[environment_id]['candidates']
                          if item['candidate_id'] == env_row.get('selected_candidate_id')), None)
        if not candidate or not candidate.get('validation_trace_path'):
            results.append({'environment_id': environment_id, 'status': 'not_run',
                            'physical_fit_passed': False, 'reason': 'validation trace index is absent'})
            continue
        trace = _json(run_dir / candidate['validation_trace_path'])
        model_path = run_dir / model_row['model_path']
        model = _json(model_path)
        pair = {'environment_id': environment_id, 'status': 'running',
            'validation_base_trace_hash': candidate['validation_base_trace_hash'],
            'fit_base_trace_hash': candidate['base_trace_hash'],
            'validation_trace_sha256': candidate['validation_full_trace_hash'],
            'frozen_model_sha256': model_row['model_sha256'], 'policies': {}}
        failure = None
        for policy in ('Stay', 'Balanced'):
            try:
                out = run_dir / 'sim_validation' / environment_id / policy.lower()
                sim = Q15BCohortSim(model, trace, seed=plan['validation_seed'], output_dir=out)
                summary, decisions, events = sim.run(policy, T_ref=1., reward_definition=reward,
                    episode_horizon_sec=EPISODE_HORIZON_SEC)
                simulated = _metrics(summary)
                actual_item = validation.get('episodes', {}).get(policy, {})
                actual_id = actual_item.get('episode_id')
                actual = _metrics(_json(run_dir / 'runs' / actual_id / 'episode.json')) if actual_id else None
                rel_c = (simulated['C_ms']-actual['C_ms'])/actual['C_ms'] if actual and actual['C_ms'] else None
                pair['policies'][policy] = {'status': summary.get('status'),
                    'simulated_metrics': simulated, 'real_metrics': actual,
                    'relative_C_error': rel_c, 'decision_count': len(decisions),
                    'migration_event_count': len(events), 'reward_reconstruction':
                        _json(out / 'reward.json') if reward is not None else None,
                    'trace_route_mode': summary.get('routing_mode'),
                    'simulated_native_dispatch': summary.get('routing_mode') == 'native'}
            except Exception as exc:
                failure = {'policy': policy, 'error': type(exc).__name__ + ': ' + str(exc),
                           'traceback': traceback.format_exc()}
                break
        if failure:
            pair.update({'status': 'failed', 'failure': failure,
                'physical_fit_passed': False, 'closed_loop_passed': False})
        else:
            stay = pair['policies']['Stay']; balanced = pair['policies']['Balanced']
            real_stay = stay.get('real_metrics'); real_balanced = balanced.get('real_metrics')
            sim_stay = stay.get('simulated_metrics'); sim_balanced = balanced.get('simulated_metrics')
            real_gain = ((real_stay['C_ms']-real_balanced['C_ms'])/real_stay['C_ms']
                         if real_stay and real_stay['C_ms'] else None)
            sim_gain = ((sim_stay['C_ms']-sim_balanced['C_ms'])/sim_stay['C_ms']
                        if sim_stay and sim_stay['C_ms'] else None)
            gain_error = abs(sim_gain-real_gain) if sim_gain is not None and real_gain is not None else None
            c_errors = [abs(row['relative_C_error']) for row in (stay, balanced)
                        if row.get('relative_C_error') is not None]
            # Full state-curve, migration-pause, reward-window and action-mask
            # tolerances remain separately required before declaring readiness.
            physical_pass = (len(c_errors) == 2 and all(error <= .20 for error in c_errors)
                             and gain_error is not None and gain_error <= .10)
            pair.update({'status': 'complete', 'real_gain_C': real_gain,
                'simulated_gain_C': sim_gain, 'absolute_gain_C_error': gain_error,
                'physical_fit_passed': physical_pass,
                'closed_loop_passed': physical_pass,
                'sim_pretraining_ready': bool(physical_pass and environment_id != 'NATIVE'
                    and env_row.get('selection_passed')),
                'sim_evaluation_ready': bool(physical_pass and environment_id == 'NATIVE'),
                'readiness_scope': 'independent_trace' if physical_pass else 'fit_only_or_failed',
                'not_yet_evaluated': ['outstanding_curve_error', 'per_instance_waiting_KV_error',
                    'new_reward_pressure_component_error', 'migration_pause_and_tail_error',
                    'full observation/action-mask tolerance contract']})
        results.append(pair)
        _atomic_json(run_dir / 'sim_validation' / environment_id / 'validation.json', pair)

    elapsed = time.monotonic()-started
    result = {'schema': 'mig-ppo-v2-simulation-validation-v1',
        'status': 'complete_with_explicit_pending_checks',
        'run_id': plan['run_id'], 'model_registry_sha256': _sha_file(run_dir / 'model_registry.json'),
        'reward_frozen_sha256': _sha_file(run_dir / 'reward_frozen.json'),
        'normalizer_sha256': _json(run_dir / 'normalizer_frozen.json')['normalizer_sha256'],
        'validation_base_trace_groups': sorted({r.get('validation_base_trace_hash') for r in results
                                                if r.get('validation_base_trace_hash')}),
        'fit_trace_groups': sorted({r.get('fit_base_trace_hash') for r in results
                                    if r.get('fit_base_trace_hash')}),
        'environments': results, 'cpu_elapsed_sec': elapsed,
        'no_formal_ppo_training_or_real_update': True}
    _atomic_json(run_dir / 'sim_validation_summary.json', result)
    budget = _json(run_dir / 'budget.json')
    budget['cpu_elapsed_sec'] = float(budget.get('cpu_elapsed_sec', 0.)) + elapsed
    budget.setdefault('phases', []).append({'stage': 'cpu_sim_validation', 'elapsed_sec': elapsed,
        'ended_utc': datetime.now(timezone.utc).isoformat()})
    _atomic_json(run_dir / 'budget.json', budget)
    return {'status': result['status'], 'environment_count': len(results),
            'completed': sum(row['status'] == 'complete' for row in results),
            'cpu_elapsed_sec': elapsed}


def write_delivery(run_dir: Path):
    """Materialize a complete 13-row report and immutable handoff references."""
    import csv
    run_dir = Path(run_dir).resolve()
    plan = validate_plan(run_dir)
    matrix = _json(run_dir / 'environment_matrix.json')
    selection = _json(run_dir / 'selection_summary.json')
    registry = _json(run_dir / 'model_registry.json') if (run_dir / 'model_registry.json').exists() else {}
    sim_summary = _json(run_dir / 'sim_validation_summary.json') if (run_dir / 'sim_validation_summary.json').exists() else {'environments': []}
    model_by_id = {row['environment_id']: row for row in registry.get('environment_models', [])}
    sim_by_id = {row['environment_id']: row for row in sim_summary.get('environments', [])}
    fields = ['environment_id', 'role', 'hotspot_instance_index', 'actual_hotspot_fraction',
              'routing_mode', 'actual_qps', 'status', 'candidate_id', 'candidate_branch',
              'stay_C_ms', 'balanced_C_ms', 'gain_C', 'balanced_successful_migrations',
              'balanced_distinct_migrated_requests', 'selected_for_positive_gain',
              'fit_status', 'independent_validation_status', 'physical_fit_passed',
              'sim_pretraining_ready', 'sim_evaluation_ready', 'evidence_path']
    out_rows = []
    for row in matrix:
        selected = row.get('selection') or {}
        candidate = next((c for c in row.get('candidates', [])
                          if c['candidate_id'] == row.get('selected_candidate_id')), {})
        model = model_by_id.get(row['environment_id'], {})
        sim = sim_by_id.get(row['environment_id'], {})
        out_rows.append({'environment_id': row['environment_id'], 'role': row['role'],
            'hotspot_instance_index': (None if row.get('hotspot_instance_index') is None else row['hotspot_instance_index']+1),
            'actual_hotspot_fraction': candidate.get('actual_hotspot_fraction'),
            'routing_mode': row['routing_mode'], 'actual_qps': candidate.get('target_qps'),
            'status': row.get('status'), 'candidate_id': row.get('selected_candidate_id'),
            'candidate_branch': candidate.get('branch'), 'stay_C_ms': selected.get('stay_C_ms'),
            'balanced_C_ms': selected.get('balanced_C_ms'), 'gain_C': selected.get('gain_C'),
            'balanced_successful_migrations': selected.get('balanced_successful_migrations'),
            'balanced_distinct_migrated_requests': selected.get('balanced_distinct_migrated_requests'),
            'selected_for_positive_gain': bool(row.get('selection_passed')),
            'fit_status': model.get('status', row.get('fit_status')),
            'independent_validation_status': (row.get('independent_validation_status') or 'pending'),
            'physical_fit_passed': sim.get('physical_fit_passed', False),
            'sim_pretraining_ready': sim.get('sim_pretraining_ready', False),
            'sim_evaluation_ready': sim.get('sim_evaluation_ready', False),
            'evidence_path': f"comparisons/{row.get('selected_candidate_id')}.json" if row.get('selected_candidate_id') else None})
    with (run_dir / 'results.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(out_rows)
    lines = ['# Mig-PPO V2 多环境筛选与拟合报告', '',
        f"Run `{plan['run_id']}`；选择阶段：`{selection['status']}`；CPU 仿真验收：`{sim_summary.get('status', 'not_run')}`。", '',
        '单个配对通过仅表示该冻结 trace 上 `gain_C > 0` 且有成功迁移证据，不代表统计显著或跨 seed 稳健。', '',
        '| 场景 | 热点位置 / 实际占比 | 路由 | QPS | Stay C | Balanced C | gain_C | 迁移成功 / 请求 | 选择状态 | 拟合 | 独立验证 / 仿真 |',
        '|---|---:|---|---:|---:|---:|---:|---:|---|---|---|']
    for row in out_rows:
        loc = '—' if row['hotspot_instance_index'] is None else str(row['hotspot_instance_index'])
        frac = '—' if row['actual_hotspot_fraction'] is None else f"{row['actual_hotspot_fraction']:.3%}"
        fmt = lambda x: '—' if x is None else f'{float(x):.3f}'
        lines.append(f"| {row['environment_id']} | {loc} / {frac} | {row['routing_mode']} | {fmt(row['actual_qps'])} | {fmt(row['stay_C_ms'])} | {fmt(row['balanced_C_ms'])} | {fmt(row['gain_C'])} | {row['balanced_successful_migrations'] or 0} / {row['balanced_distinct_migrated_requests'] or 0} | {row['status']} | {row['fit_status']} | {row['independent_validation_status']} / {'passed' if row['physical_fit_passed'] else 'pending or failed'} |")
    lines.extend(['', '## 适用范围与限制', '',
        '- `NATIVE` 保持 `test_only`；新请求路由通过本任务模拟分发器重新决策，不能把它作为 PPO 训练环境。',
        '- 物理拟合来源是一对开发 trace；重复决策样本并非独立实验。迁移暂停时间与失败概率没有足够样本时保持 unavailable。',
        '- 只有 `sim_pretraining_ready=true` 的人工热点环境可进入后续 PPO 仿真采样；本报告不会启动训练。',
        '- 完整状态曲线、奖励压力、接口容差及迁移尾部任一项未审计通过时，仿真 ready 保持 false。', '',
        '详见 `model_registry.json`、`reward_frozen.json`、`reward_reconstruction.json`、`sim_validation_summary.json`、`cleanup_verification.json`。',
        ''])
    (run_dir / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    accepted = [r['environment_id'] for r in out_rows if r['selected_for_positive_gain']]
    handoff = {'schema': 'mig-ppo-v2-multienv-handoff-v1', 'run_id': plan['run_id'],
        'plan_freeze_sha256': plan['freeze_sha256'], 'selection_sha256': selection['selection_sha256'],
        'scenario_count': 13, 'scenarios': out_rows, 'accepted_positive_gain_environments': accepted,
        'native_role': 'test_only', 'native_sim_evaluation_ready': sim_by_id.get('NATIVE', {}).get('sim_evaluation_ready', False),
        'old_q15b_alias_status': 'conditional prior only; this run refits where samples exist; 141104 is development data',
        'decision_interval_sec': DECISION_INTERVAL_SEC, 'migration_cooldown_sec': MIGRATION_COOLDOWN_SEC,
        'episode_horizon_sec': EPISODE_HORIZON_SEC, 'normalizer_path': 'normalizer_frozen.json',
        'reward_path': 'reward_frozen.json', 'model_registry_path': 'model_registry.json',
        'schema_and_action_shape': {'normalizer_path': 'normalizer_frozen.json', 'action_count': 13,
                                    'instance_count': 4, 'request_count_scaling': 'none'},
        'gpu_budget': _json(run_dir / 'budget.json'),
        'uncompleted_or_unready': [r['environment_id'] for r in out_rows
            if not (r['sim_pretraining_ready'] or r['sim_evaluation_ready'])],
        'simulator_load_command': f"python benchmark/run_ppo_sim.py multienv-validate --run-dir {run_dir}",
        'cpu_rollout_command': 'pending: use only after per-environment sim_pretraining_ready is true',
        'formal_ppo_training_started': False,
        'native_closed_loop_claim': 'only if native sim_evaluation_ready is true; otherwise not validated'}
    _atomic_json(run_dir / 'handoff_sim_ppo.json', handoff)
    (run_dir / 'reproduce.md').write_text(
        '# 复现命令\n\n'
        f"```bash\npython benchmark/run_ppo_real.py multienv-plan --run-dir {run_dir}\n"
        f"python benchmark/run_ppo_real.py multienv-collect --run-dir {run_dir} --stage selection\n"
        f"python benchmark/run_ppo_sim.py multienv-fit --run-dir {run_dir}\n"
        f"python benchmark/run_ppo_real.py multienv-collect --run-dir {run_dir} --stage validation\n"
        f"python benchmark/run_ppo_sim.py multienv-validate --run-dir {run_dir}\n"
        f"python benchmark/run_ppo_sim.py multienv-report --run-dir {run_dir}\n```\n",
        encoding='utf-8')
    (run_dir / 'next_step.md').write_text(
        '后续动作：先查看 `report.md` 与 `handoff_sim_ppo.json`，只对标记 `sim_pretraining_ready=true` 的人工热点环境准备仿真 PPO；本轮不自动启动训练。NATIVE 始终只用于独立测试。\n',
        encoding='utf-8')
    return {'status': 'report_written', 'environment_rows': len(out_rows),
            'accepted_positive_gain_environments': accepted,
            'report_path': str(run_dir / 'report.md'), 'handoff_path': str(run_dir / 'handoff_sim_ppo.json')}



