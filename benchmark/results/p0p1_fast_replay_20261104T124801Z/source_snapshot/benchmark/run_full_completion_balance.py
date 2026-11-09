#!/usr/bin/env python3
"""Run the versioned full-completion Balance-vs-Stay development search."""
from __future__ import annotations

import argparse
import asyncio
import gzip
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_scheduler.full_completion import compare_pair, make_trace, run_episode, runtime_dependencies_hash
from rl_scheduler.real_budget import atomic_json


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _supervisor(*args, check=True):
    return subprocess.run(['supervisorctl', *args], check=check, capture_output=True,
                          text=True, timeout=130.)


class ManagedFullCompletionService:
    """Install one temporary supervisor program that owns the experiment cluster."""
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        suffix = hashlib.sha256(str(self.run_dir.resolve()).encode()).hexdigest()[:10]
        self.name = 'full_completion_' + suffix
        self.wrapper = self.run_dir / 'services' / (self.name + '.sh')
        self.conf = self.run_dir / 'services' / (self.name + '.conf')
        self.installed = Path('/etc/supervisor/conf.d') / self.conf.name
        self.running = False

    def prepare(self):
        self.wrapper.parent.mkdir(parents=True, exist_ok=True)
        argv = [str(ROOT / '.venv/bin/python'), '-m', 'rl_scheduler.full_completion_service',
                '--root', str(ROOT), '--run-dir', str(self.run_dir),
                '--config', str((self.run_dir / 'runtime_config.yaml').resolve())]
        import shlex
        self.wrapper.write_text('#!/bin/bash\nset -e\n'
            '. /opt/supervisor-scripts/utils/logging.sh\n'
            '. /opt/supervisor-scripts/utils/environment.sh\n'
            'cd ' + shlex.quote(str(ROOT)) + '\nexec ' + shlex.join(argv) + '\n')
        self.wrapper.chmod(0o755)
        self.conf.write_text(f'''[program:{self.name}]
environment=PROC_NAME="%(program_name)s"
command={self.wrapper}
autostart=false
autorestart=false
stopasgroup=true
killasgroup=true
stopsignal=TERM
stopwaitsecs=120
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_events_enabled=true
stdout_logfile_maxbytes=0
''')
        if self.installed.exists() and self.installed.read_bytes() != self.conf.read_bytes():
            raise RuntimeError('conflicting supervisor service name already exists')
        self.installed.write_bytes(self.conf.read_bytes())
        _supervisor('reread')
        _supervisor('update', self.name)

    def is_running(self):
        result = _supervisor('status', self.name, check=False)
        return result.returncode == 0 and ' RUNNING ' in result.stdout

    def start(self):
        self.prepare()
        _supervisor('start', self.name)
        self.running = True

    def stop(self):
        if self.running:
            _supervisor('stop', self.name, check=False)
            self.running = False
        if self.installed.exists():
            self.installed.unlink()
            _supervisor('reread')
            _supervisor('update')


def _load_yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text())


def _init_run(args):
    source_matrix_path = ROOT / 'rl_scheduler/docs/full_completion_balance_v2/candidate_matrix.json'
    source_matrix = json.loads(source_matrix_path.read_text())
    requested_ids = set(args.candidate_ids or ())
    available_ids = {row['candidate_id'] for row in source_matrix['candidates']}
    unknown_ids = requested_ids - available_ids
    if unknown_ids:
        raise ValueError('unknown candidate IDs: ' + ', '.join(sorted(unknown_ids)))
    if args.reuse_lambda_ref_from is not None:
        lambda_path = args.reuse_lambda_ref_from.resolve() / 'lambda_ref.json'
        if not lambda_path.is_file():
            raise FileNotFoundError('lambda_ref.json not found in reuse source: ' + str(lambda_path))
    run_dir = args.output_dir or ROOT / 'benchmark/results' / (
        'full_completion_balance_v2_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'services').mkdir()
    (run_dir / 'runtime').mkdir()
    docs = ROOT / 'rl_scheduler/docs/full_completion_balance_v2'
    shutil.copy2(docs / 'experiment_spec.yaml', run_dir / 'experiment_spec.yaml')
    shutil.copy2(docs / 'candidate_matrix.json', run_dir / 'candidate_matrix.json')
    cfg = _load_yaml(ROOT / 'rl_scheduler/configs/real_pilot.yaml')
    available_cpus = len(os.sched_getaffinity(0))
    cfg['service']['ray_num_cpus'] = max(16, int(cfg['service']['ray_num_cpus']))
    if cfg['service']['ray_num_cpus'] > available_cpus:
        raise RuntimeError('Ray CPU requirement exceeds this process cpuset')
    ray_tag = hashlib.sha256(str(run_dir).encode()).hexdigest()[:10]
    cfg['service']['ray_temp_dir'] = str(Path('/tmp') / ('fc_' + ray_tag))
    import yaml
    server_config = _load_yaml(ROOT / 'rl_scheduler/configs/real_llumnix.yaml')
    server_config['SERVER']['REQUEST_OUTPUT_QUEUE_TYPE'] = args.queue_type
    server_config_path = run_dir / 'real_llumnix.yaml'
    server_config_path.write_text(yaml.safe_dump(server_config, sort_keys=False))
    server_argv = cfg['service']['server_argv']
    server_argv[server_argv.index('--config-file') + 1] = str(server_config_path)
    cfg['service']['request_output_queue_type'] = args.queue_type
    (run_dir / 'runtime_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    candidate_matrix = json.loads((run_dir / 'candidate_matrix.json').read_text())
    source_paths = [ROOT / path for path in (
        'rl_scheduler/full_completion.py', 'rl_scheduler/full_completion_service.py',
        'rl_scheduler/real_metrics.py', 'rl_scheduler/real_client.py',
        'rl_scheduler/real_opportunity.py',
        'rl_scheduler/real_opportunity_runner.py', 'rl_scheduler/real_runner.py',
        'rl_scheduler/real_bridge.py', 'rl_scheduler/real_observation.py',
        'llumnix/entrypoints/vllm/real_metrics_stream.py',
        'llumnix/metrics/causal_events.py', 'llumnix/backends/vllm/llm_engine.py',
        'llumnix/entrypoints/vllm/real_control_api.py', 'llumnix/real_control.py',
        'llumnix/backends/output_forwarder.py', 'llumnix/queue/ray_queue_server.py',
        'llumnix/queue/ray_queue_client.py',
        'benchmark/run_full_completion_balance.py', 'rl_scheduler/configs/real_pilot.yaml',
        'rl_scheduler/configs/real_llumnix.yaml',
        'rl_scheduler/docs/full_completion_balance_experiment_v2.md',
        'rl_scheduler/docs/full_completion_balance_v2/experiment_spec.yaml',
        'rl_scheduler/docs/full_completion_balance_v2/candidate_matrix.json')]
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
                                capture_output=True, text=True).stdout.strip()
    except subprocess.CalledProcessError:
        commit = None
    manifest = {
        'schema': 'full-completion-balance-run-v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'created_client_date': datetime.now(timezone.utc).astimezone(
            ZoneInfo('Asia/Shanghai')).date().isoformat(),
        'run_id': 'fc_' + hashlib.sha256(str(run_dir).encode()).hexdigest()[:16],
        'git_commit': commit, 'working_tree_dirty_at_start': True,
        'code_hashes': {str(path.relative_to(ROOT)): _sha(path) for path in source_paths},
        'config_hash': _sha(run_dir / 'runtime_config.yaml'),
        'candidate_matrix_hash': _sha(run_dir / 'candidate_matrix.json'),
        'full_completion_protocol_hash': _sha(run_dir / 'experiment_spec.yaml'),
        'candidate_count': candidate_matrix['candidate_count'],
        'selected_candidate_ids': sorted(requested_ids) if requested_ids else
            [row['candidate_id'] for row in candidate_matrix['candidates']],
        'lambda_ref_reuse_source': (str(args.reuse_lambda_ref_from.resolve())
                                    if args.reuse_lambda_ref_from else None),
        'request_output_queue_type': args.queue_type,
        'telemetry_sampling': {
            'client_and_engine_cumulative_output_counts': 'first_every_64_final',
            'real_metrics_api_queue_updates': 'first_every_64_final_before_queue_delivery',
            'high_frequency_iteration_diagnostics': 'first_every_256_by_process_and_event',
            'client_event_journal_writer': 'ordered_background_queue_with_durable_checkpoint_barriers',
            'client_sender_event_loops': 4,
            'client_sender_http_contexts': 'prepared_on_each_sender_loop_before_measured_release',
            'client_sender_task_launch': 'one_cross_thread_batch_launcher_per_sender_loop',
            'process_core_dumps_disabled': True,
        },
        'send_lag_threshold_gate': 'disabled_by_user; metrics_retained_and_reported',
        'control_transport': {'keepalive_timeout_sec': 3.0,
                              'request_total_timeout_sec': None,
                              'readonly_snapshot_transport_attempts': 3,
                              'reason': 'avoid stale idle connections and total-timeout false invalidation'},
        'artifact_compression': 'completed JSONL artifacts are gzip-compressed with verified uncompressed SHA-256',
        'runtime_llumnix_config_hash': _sha(server_config_path),
        'gpu_start_authorized_by_user_task': True,
        'external_ports_opened': False,
        'legacy_fixed_window_scorer_used': False,
    }
    atomic_json(run_dir / 'run_manifest.json', manifest)
    return run_dir, cfg, manifest, candidate_matrix


def _append_jsonl(path, row):
    import os
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _append_episode(path, row):
    clean = dict(row)
    clean.pop('requests', None)
    clean.pop('ecdf', None)
    _append_jsonl(path, clean)


def _compress_jsonl_artifacts(directory):
    """Compress completed JSONL artifacts losslessly and record source hashes."""
    directory = Path(directory)
    if not directory.exists():
        return []
    records = []
    for source in sorted(directory.rglob('*.jsonl')):
        target = source.with_name(source.name + '.gz')
        temporary = target.with_name(target.name + '.tmp')
        digest = hashlib.sha256()
        original_bytes = 0
        with source.open('rb') as stream, temporary.open('wb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1, mtime=0) as compressed:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
                    original_bytes += len(block)
                    compressed.write(block)
            raw.flush()
            os.fsync(raw.fileno())
        source_hash = digest.hexdigest()
        verified_hash = hashlib.sha256()
        verified_bytes = 0
        with gzip.open(temporary, 'rb') as compressed:
            for block in iter(lambda: compressed.read(1024 * 1024), b''):
                verified_hash.update(block)
                verified_bytes += len(block)
        if verified_hash.hexdigest() != source_hash or verified_bytes != original_bytes:
            temporary.unlink(missing_ok=True)
            raise RuntimeError('lossless JSONL compression verification failed: ' + str(source))
        os.replace(temporary, target)
        source.unlink()
        records.append({
            'source': str(source.relative_to(directory)),
            'compressed': str(target.relative_to(directory)),
            'original_bytes': original_bytes,
            'compressed_bytes': target.stat().st_size,
            'sha256_uncompressed': source_hash,
        })
    if records:
        atomic_json(directory / 'storage_compression_manifest.json', {
            'method': 'gzip level 1, verified by uncompressed SHA-256',
            'files': records,
        })
    return records


def _warmup_trace(tokenizer):
    trace = make_trace(candidate={'family': 'C0', 'profile': 'decode64', 'rho': 1.0},
        n=4, seed=81000, lambda_ref=1., tokenizer=tokenizer, synchronous=True,
        episode_id_prefix='warm')
    for row in trace:
        row['output_tokens'] = 8
        row['request_type'] = 'warmup_output8'
    return trace


def _calibration_trace(tokenizer, seed, profile='decode1536'):
    return make_trace(candidate={'family': 'C0', 'profile': profile, 'rho': 1.0},
        n=256, seed=seed, lambda_ref=1., tokenizer=tokenizer, synchronous=True,
        episode_id_prefix='cal_' + profile + '_')


async def run_search(args):
    import aiohttp
    from disturbance_exp.workload import load_local_tokenizer
    from rl_scheduler.full_completion import wait_service_ready

    run_dir, cfg, manifest, matrix = _init_run(args)
    run_id = manifest['run_id']
    service = ManagedFullCompletionService(run_dir)
    episodes_path = run_dir / 'episodes.jsonl'
    candidates_path = run_dir / 'candidate_ledger.jsonl'
    results = []
    mapping = None
    started_at = time.monotonic()
    try:
        tokenizer = load_local_tokenizer(cfg['service']['model_path'])
        if tokenizer is None:
            raise RuntimeError('local Qwen tokenizer could not be loaded')
        service.start()
        control_timeout = aiohttp.ClientTimeout(total=None, connect=5., sock_connect=5.)
        async with aiohttp.ClientSession(timeout=control_timeout,
            connector=aiohttp.TCPConnector(limit=4096, keepalive_timeout=3.0)) as session:
            await wait_service_ready(cfg, service.is_running)

            # Warmup is named, separately audited, and never enters candidate metrics.
            trace = _warmup_trace(tokenizer)
            atomic_json(run_dir / 'warmup_trace.json', trace)
            warmup = await run_episode(cfg=cfg, run_id=run_id, episode_id='warmup_000',
                policy='Stay', trace=trace, run_dir=run_dir, control_session=session,
                frozen_mapping=mapping, service_probe=service.is_running,
                stop_owned_service=service.stop)
            _compress_jsonl_artifacts(run_dir / 'runs' / 'warmup_000')
            _append_episode(episodes_path, warmup)
            results.append(warmup)
            if warmup.get('status') != 'passed':
                raise RuntimeError('warmup full-cohort audit failed: ' + str(warmup.get('issues')))
            mapping = warmup['frozen_mapping']

            if args.stage == 'smoke':
                probe_trace = make_trace(
                    candidate={'family': 'C0', 'profile': 'decode64', 'rho': 1.0},
                    n=32, seed=81001, lambda_ref=1., tokenizer=tokenizer,
                    synchronous=True, episode_id_prefix='qprobe_')
                for row in probe_trace:
                    row['output_tokens'] = 256
                    row['request_type'] = 'queue_probe_output256'
                atomic_json(run_dir / 'queue_probe_trace.json', probe_trace)
                probe = await run_episode(cfg=cfg, run_id=run_id,
                    episode_id='queue_probe_000', policy='Stay', trace=probe_trace,
                    run_dir=run_dir, control_session=session, frozen_mapping=mapping,
                    service_probe=service.is_running, stop_owned_service=service.stop)
                _compress_jsonl_artifacts(run_dir / 'runs' / 'queue_probe_000')
                _append_episode(episodes_path, probe)
                results.append(probe)
                if probe.get('status') != 'passed':
                    raise RuntimeError('queue probe full-cohort audit failed: '
                                       + str(probe.get('issues')))
                report = {
                    'schema': 'full-completion-balance-smoke-v1', 'status': 'passed',
                    'run_id': run_id, 'run_dir': str(run_dir),
                    'warmup_episode': 'warmup_000', 'warmup_cohort_size': len(trace),
                    'warmup_output_tokens': 8, 'primary_metrics': warmup['primary_metrics'],
                    'queue_probe_episode': 'queue_probe_000',
                    'queue_probe_cohort_size': len(probe_trace),
                    'queue_probe_output_tokens': 256,
                    'queue_probe_primary_metrics': probe['primary_metrics'],
                    'request_output_queue_type': args.queue_type,
                    'service_cleanup_record': 'service_cleanup.json',
                    'candidate_performance_evaluated': False,
                }
                atomic_json(run_dir / 'smoke_summary.json', report)
                print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
                return report

            # Reuse only a complete, audited calibration when continuing a search.
            # Otherwise, every profile gets its own finite-batch Stay reference.
            profiles = list(dict.fromkeys(row['profile'] for row in matrix['candidates']))
            if args.reuse_lambda_ref_from is not None:
                lambda_source = args.reuse_lambda_ref_from.resolve() / 'lambda_ref.json'
                lambda_document = json.loads(lambda_source.read_text())
                reference_by_profile = lambda_document.get('profiles', {})
                missing_profiles = set(profiles) - set(reference_by_profile)
                if missing_profiles:
                    raise ValueError('reused lambda_ref lacks profiles: '
                                     + ', '.join(sorted(missing_profiles)))
                shutil.copy2(lambda_source, run_dir / 'lambda_ref.json')
                atomic_json(run_dir / 'lambda_ref_reuse.json', {
                    'source_run_dir': str(args.reuse_lambda_ref_from.resolve()),
                    'source_file': str(lambda_source), 'sha256': _sha(lambda_source),
                    'profiles_used': profiles,
                    'reason': 'continuation search reuses the already passed 3/3 profile calibrations',
                })
            else:
                reference_by_profile = {}
                for profile_index, profile in enumerate(profiles):
                    reference_rates = []
                    for repeat, seed in enumerate((84001, 84002, 84003)):
                        trace = _calibration_trace(tokenizer, seed, profile)
                        trace_path = run_dir / f'calibration_{profile}_seed{seed}_trace.json'
                        atomic_json(trace_path, trace)
                        row = await run_episode(cfg=cfg, run_id=run_id,
                            episode_id=f'cal_{profile}_{seed}', policy='Stay', trace=trace,
                            run_dir=run_dir, control_session=session, frozen_mapping=mapping,
                            service_probe=service.is_running, stop_owned_service=service.stop)
                        _compress_jsonl_artifacts(run_dir / 'runs' / row['episode_id'])
                        row.update({'calibration_profile': profile, 'calibration_seed': seed,
                                    'calibration_repeat': repeat})
                        _append_episode(episodes_path, row)
                        results.append(row)
                        if row.get('status') != 'passed':
                            raise RuntimeError('full-cohort lambda_ref calibration failed for '
                                               + profile + ': ' + str(row.get('issues')))
                        rate = row['primary_metrics']['cohort_throughput_rps']
                        reference_rates.append(rate)
                        _append_jsonl(candidates_path, {'kind': 'lambda_ref_repeat',
                            'profile': profile, 'seed': seed, 'cohort_throughput_rps': rate})
                    reference_by_profile[profile] = {
                        'repeats_rps': reference_rates,
                        'lambda_ref_rps': sorted(reference_rates)[1],
                    }
                    print(json.dumps({'calibration_profile': profile,
                        'repeat_rps': reference_rates,
                        'lambda_ref_rps': reference_by_profile[profile]['lambda_ref_rps']},
                        sort_keys=True), flush=True)
                atomic_json(run_dir / 'lambda_ref.json', {
                    'schema': 'full-completion-lambda-ref-v1', 'profiles': reference_by_profile,
                    'unit': 'requests_per_sec',
                    'formula': 'per profile: median(256/(last_client_terminal-first_scheduled))',
                    'interpretation': 'finite uniform fixed-route synchronous cohort reference',
                })

            candidate_summaries = []
            seeds = (82001, 82002)
            requested_ids = set(args.candidate_ids or ())
            candidates = [(index, row) for index, row in enumerate(matrix['candidates'])
                         if not requested_ids or row['candidate_id'] in requested_ids]
            for candidate_index, candidate in candidates:
                lambda_ref = reference_by_profile[candidate['profile']]['lambda_ref_rps']
                pair_rows = []
                for seed_index, seed in enumerate(seeds):
                    trace = make_trace(candidate=candidate, n=128, seed=seed,
                        lambda_ref=lambda_ref, tokenizer=tokenizer,
                        episode_id_prefix=candidate['candidate_id'] + '_')
                    trace_path = run_dir / 'traces' / candidate['candidate_id'] / f'seed_{seed}.json'
                    atomic_json(trace_path, trace)
                    trace_hash = _sha(trace_path)
                    order = (('Stay', 'Balanced') if (candidate_index + seed_index) % 2 == 0
                             else ('Balanced', 'Stay'))
                    pair = {'kind': 'paired_candidate', 'candidate_id': candidate['candidate_id'],
                            'seed': seed, 'trace_sha256': trace_hash, 'policy_order': order,
                            'episodes': {}}
                    for policy in order:
                        episode_id = f"{candidate['candidate_id']}_{seed}_{policy.lower()}"
                        row = await run_episode(cfg=cfg, run_id=run_id,
                            episode_id=episode_id, policy=policy, trace=trace,
                            run_dir=run_dir, control_session=session, frozen_mapping=mapping,
                            service_probe=service.is_running, stop_owned_service=service.stop)
                        _compress_jsonl_artifacts(run_dir / 'runs' / episode_id)
                        row.update({'candidate_id': candidate['candidate_id'], 'seed': seed,
                                    'trace_sha256': trace_hash, 'rho': candidate['rho'],
                                    'family': candidate['family']})
                        _append_episode(episodes_path, row)
                        results.append(row)
                        pair['episodes'][policy] = row
                        mapping = row.get('frozen_mapping', mapping)
                    pair['comparison'] = compare_pair(pair['episodes']['Stay'],
                                                      pair['episodes']['Balanced'])
                    pair_rows.append(pair)
                    _append_jsonl(candidates_path, pair)

                gains = [row['comparison'] for row in pair_rows
                         if row['comparison']['status'] == 'passed']
                mean_gains = [row['mean_gain'] for row in gains]
                p99_gains = [row['p99_gain'] for row in gains]
                joint_gains = sorted(min(row['mean_gain'], row['p95_gain']) for row in gains)
                ranking_gain = (sum(joint_gains) / 2 if len(joint_gains) == 2 else
                                joint_gains[0] if joint_gains else None)
                cvar_gains = sorted(row['cvar95_gain'] for row in gains)
                median_cvar_gain = (sum(cvar_gains) / 2 if len(cvar_gains) == 2 else
                                    cvar_gains[0] if cvar_gains else None)
                candidate_summary = {
                    'kind': 'candidate_summary', 'candidate_id': candidate['candidate_id'],
                    'family': candidate['family'], 'rho': candidate['rho'],
                    'profile': candidate['profile'],
                    'hot_fraction': candidate.get('hot_fraction'),
                    'long_fraction': candidate.get('long_fraction'),
                    'burst_multiplier': candidate.get('burst_multiplier'),
                    'lambda_ref': lambda_ref,
                    'valid_pairs': len(gains), 'planned_pairs': len(seeds),
                    'mean_gain_by_seed': mean_gains, 'p99_gain_by_seed': p99_gains,
                    'median_mean_gain': (sum(sorted(mean_gains)[len(mean_gains)//2-1:
                        len(mean_gains)//2+1]) / 2 if len(mean_gains) == 2 else
                        mean_gains[0] if mean_gains else None),
                    'median_p99_gain': (sum(sorted(p99_gains)[len(p99_gains)//2-1:
                        len(p99_gains)//2+1]) / 2 if len(p99_gains) == 2 else
                        p99_gains[0] if p99_gains else None),
                    'median_p95_gain': (sum(sorted(row['p95_gain'] for row in gains)) / 2
                        if len(gains) == 2 else gains[0]['p95_gain'] if gains else None),
                    'median_cvar95_gain': median_cvar_gain,
                    'exploratory_win_both_metrics': bool(len(gains) == len(seeds)
                        and all(row['mean_gain'] > 0 and row['p99_gain'] > 0 for row in gains)),
                    'ranking_gain_min_mean_p95': ranking_gain if len(gains) == len(seeds) else None,
                    'independent_validation': False,
                }
                candidate_summaries.append(candidate_summary)
                _append_jsonl(candidates_path, candidate_summary)
                print(json.dumps({'candidate_complete': candidate['candidate_id'],
                    'mean_gain_by_seed': mean_gains, 'p99_gain_by_seed': p99_gains,
                    'exploratory_win_both_metrics': candidate_summary['exploratory_win_both_metrics']},
                    sort_keys=True), flush=True)

        wins = [row for row in candidate_summaries if row['exploratory_win_both_metrics']]
        ranked_candidates = sorted(candidate_summaries, key=lambda row: (
            row['ranking_gain_min_mean_p95'] is None,
            -(row['ranking_gain_min_mean_p95'] if row['ranking_gain_min_mean_p95'] is not None else 0.),
            -(row['median_cvar95_gain'] if row['median_cvar95_gain'] is not None else 0.),
            row['candidate_id']))
        report = {
            'schema': 'full-completion-balance-search-v1', 'status': 'screened',
            'stage': 'coarse_screen_selected_candidates', 'run_id': run_id, 'run_dir': str(run_dir),
            'selected_candidate_ids': [row['candidate_id'] for _, row in candidates],
            'lambda_ref_reuse_source': (str(args.reuse_lambda_ref_from.resolve())
                                        if args.reuse_lambda_ref_from else None),
            'lambda_ref_by_profile': reference_by_profile,
            'candidate_summaries': candidate_summaries,
            'ranked_candidates': ranked_candidates,
            'exploratory_wins_both_mean_and_p99': wins,
            'found_multiple_exploratory_scenarios': len(wins) >= 2,
            'independent_validation': False, 'episode_count': len(results),
            'elapsed_sec': time.monotonic() - started_at,
            'source_hash': runtime_dependencies_hash([
                ROOT / 'rl_scheduler/full_completion.py', ROOT / 'rl_scheduler/real_metrics.py',
                ROOT / 'rl_scheduler/real_client.py', ROOT / 'rl_scheduler/real_opportunity.py',
                ROOT / 'rl_scheduler/real_bridge.py',
                ROOT / 'llumnix/entrypoints/vllm/real_control_api.py', ROOT / 'llumnix/real_control.py',
                ROOT / 'llumnix/entrypoints/vllm/real_metrics_stream.py',
                ROOT / 'llumnix/metrics/causal_events.py', ROOT / 'llumnix/backends/vllm/llm_engine.py',
                ROOT / 'llumnix/backends/output_forwarder.py', ROOT / 'llumnix/queue/ray_queue_server.py',
                ROOT / 'llumnix/queue/ray_queue_client.py',
                ROOT / 'benchmark/run_full_completion_balance.py',
                ROOT / 'rl_scheduler/real_runner.py']),
        }
        atomic_json(run_dir / 'screening_summary.json', report)
        print(json.dumps({'status': report['status'], 'run_dir': str(run_dir),
            'found_multiple_exploratory_scenarios': report['found_multiple_exploratory_scenarios'],
            'candidate_summaries': candidate_summaries}, ensure_ascii=False, indent=2), flush=True)
        return report
    except BaseException as exc:
        atomic_json(run_dir / 'run_failure.json', {
            'status': 'incomplete', 'reason': type(exc).__name__ + ': ' + str(exc),
            'elapsed_sec': time.monotonic() - started_at,
        })
        raise
    finally:
        try:
            service.stop()
            _compress_jsonl_artifacts(run_dir / 'causal_events')
            atomic_json(run_dir / 'service_cleanup.json', {
                'status': 'passed', 'supervisor_program_removed': not service.installed.exists(),
                'stopped_utc': datetime.now(timezone.utc).isoformat(),
            })
        except BaseException as exc:
            atomic_json(run_dir / 'service_cleanup.json', {
                'status': 'incomplete', 'reason': type(exc).__name__ + ': ' + str(exc),
                'supervisor_program_removed': not service.installed.exists(),
            })
            raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('smoke', 'all'), default='all')
    parser.add_argument('--queue-type', choices=('rayqueue', 'zmq'), default='rayqueue')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--candidate-ids', nargs='+',
                        help='optional subset from the frozen candidate matrix')
    parser.add_argument('--reuse-lambda-ref-from', type=Path,
                        help='reuse a prior run with passed 3/3 per-profile calibration')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        asyncio.run(run_search(args))
        return 0
    except KeyboardInterrupt:
        print('Interrupted by user; owned service cleanup is running.', file=sys.stderr, flush=True)
        return 130
    except BaseException as exc:
        print(json.dumps({'status': 'incomplete',
            'reason': type(exc).__name__ + ': ' + str(exc)}, ensure_ascii=False),
            file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
