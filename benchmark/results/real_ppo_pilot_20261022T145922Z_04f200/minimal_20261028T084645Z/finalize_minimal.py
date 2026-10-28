"""Verify saved live evidence on CPU, then write the amended 05/06 handoffs.

Does not start services or advance an experiment clock. Original full gates and
protocols remain unchanged. Replaying this file verifies the same saved run.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

CASE = Path(__file__).resolve().parent
RUN = CASE.parent
ROOT = CASE.parents[3]
sys.path.insert(0, str(ROOT))
from rl_scheduler.real_bridge import digest
from rl_scheduler.real_metrics import Event, Checkpoint, RequestLedger, _percentile
from rl_scheduler.real_opportunity_runner import reconcile_migrations
from rl_scheduler.real_reward import RewardScales, reconstruct_rewards


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def command(argv):
    value = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=False)
    # supervisorctl status uses 3 when any installed service is stopped/exited.
    allowed = (0, 3) if argv == ['supervisorctl', 'status'] else (0,)
    assert value.returncode in allowed, (argv, value.returncode, value.stderr)
    return {'argv': argv, 'exit_code': value.returncode, 'stdout': value.stdout}


def main():
    now = datetime.now(timezone.utc).isoformat()
    prefix = CASE.name
    baseline = read(CASE/'baseline.json')
    protocol = read(CASE/'protocol.json')
    result = read(CASE/'result.json')
    first = read(CASE/'attempt_01/result.json')
    assert result['status'] == 'passed'
    assert result['protocol_sha256'] == sha(CASE/'protocol.json')
    assert sha(CASE/'run_minimal.py') == protocol['code_sha256']
    assert sha(CASE/'runtime.yaml') == protocol['runtime_sha256']
    for name, value in baseline['source_hashes'].items():
        assert sha(ROOT/name) == value, name
    for name, value in baseline['root_config_hashes'].items():
        assert sha(ROOT/name) == value, name
    assert sha(RUN/'experiment_clock.json') == baseline['parent_clock_sha256']
    for name, value in baseline['parent_inputs'].items():
        assert sha(RUN/name) == value, name
    for name, value in protocol['trace_hashes'].items():
        assert sha(CASE/'traces'/name) == value, name
    spec = importlib.util.spec_from_file_location('original_gate_validator', ROOT/'benchmark/run_real_ppo.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    old_gates = ['reviews/bridge_review.json', 'gates/p1_offline_portfix_02.json',
                 'gates/p2_offline_portfix_02.json', 'gates/p3_ready.json']
    for name in old_gates:
        cli.validate_gate(RUN/name, RUN)

    scales = RewardScales(**read(CASE/'mini_reward_scales.json')['used'])
    summaries = []
    for episode in result['episodes']:
        directory = CASE/episode['episode_directory']
        manifest = read(directory/'producer_manifest.json')
        ledger = RequestLedger(**manifest)
        ledger.ingest(Event(**e) for e in rows(directory/'events.jsonl'))
        for cp in read(directory/'checkpoints.json'):
            ledger.acknowledge(Checkpoint(**cp))
        reward_input = read(directory/'reward_inputs.json')
        audit = ledger.audit(through_ns=reward_input['cleanup_through_ns'], require_cleanup=True)
        assert audit['status'] == 'passed', audit
        assert audit == read(directory/'audit.json')
        score = ledger.freeze_score()
        assert score == read(directory/'window_metrics.json')
        assert score['fixed'] == episode['score']
        assert score['fixed']['completed'] == score['fixed']['offered']
        raw_requests, problems = ledger._rows()
        assert not problems and json.loads(json.dumps(raw_requests)) == rows(directory/'requests.jsonl')
        assert all(r['outcome'] == 'success' for r in raw_requests)
        assert all(cp.pending_delivery == 0 and cp.durable for cp in ledger.checkpoints.values())
        events = [e for e in ledger.events.values()
                  if ledger.start_ns <= e.occurred_ns <= reward_input['cleanup_through_ns']]
        migration = reconcile_migrations(rows(directory/'decisions.jsonl'), episode['receipts'], events)
        assert migration == rows(directory/'migrations.jsonl') == reward_input['migrations']
        rewards = reconstruct_rewards(ledger, reward_input['edges_ns'], migration, scales=scales,
                                      cleanup_through_ns=reward_input['cleanup_through_ns'], valid_episode=True)
        assert rewards == rows(directory/'reward_windows.jsonl')
        assert all(math.isfinite(r['reward']) for r in rewards)
        assert sum(r['reward'] for r in rewards) == episode['diagnostic_reward_sum']
        assert read(directory/'route_audit.json')['status'] == 'passed'
        summary = {k: score['fixed'][k] for k in ['offered', 'sent', 'completed', 'slo_completed',
                    'client_output_tokens', 'ttft_p95_ms', 'tpot_p95_ms']}
        summary.update(label=episode['label'], audit_status='passed', producer_count=len(ledger.sources),
                       decision_latency_p95_ms=_percentile(episode['decision_latency_ms'], .95),
                       reward_windows=len(rewards), diagnostic_reward_sum=episode['diagnostic_reward_sum'],
                       replayed_from_raw_events=True)
        if episode['label'] == 'migration':
            receipt = episode['duplicate_proof']
            assert receipt == read(directory/'duplicate_receipt.json')
            assert receipt['terminal_status'] == 'succeeded' and receipt['copied_blocks'] > 0
            assert receipt['actual_request_id'] == receipt['expected_candidate_id']
            assert len(migration) == 1 and migration[0]['migration_id'] == receipt['migration_id']
            assert migration[0]['attempt_ns'] is not None
            assert migration[0]['actual_request_id'] == receipt['actual_request_id']
            original = episode['receipts'][receipt['decision_id']]
            assert original['migration_id'] == receipt['migration_id']
            assert original['terminal_status'] == 'succeeded'
            rejection = read(directory/'explicit_rejection.json')
            assert rejection['terminal_status'] == 'rejected' and rejection['migration_id'] is None
            summary.update(migration_id=receipt['migration_id'], actual_request_id=receipt['actual_request_id'],
                           copied_blocks=receipt['copied_blocks'], successful_migrations=1,
                           original_coordinator_attempts=1, duplicate_idempotency='passed',
                           stale_candidate_rejection=rejection['rejection_reason'])
        else:
            assert not migration
            summary['original_coordinator_attempts'] = 0
        summaries.append(summary)
    assert sum(e['completed'] for e in summaries) == 3
    assert result['cleanup']['owned_processes_remaining'] == first['cleanup']['owned_processes_remaining'] == 0
    assert result['budget_status_at_stop']['elapsed_sec'] <= 900
    assert result['budget_status_at_stop']['phase_usage']['P4'] == 0
    assert result['budget_status_at_stop']['phase_usage']['P5'] == 0
    assert result['PPO_updates'] == 0
    import psutil
    assert not [p for p in psutil.process_iter(['name']) if p.info['name'] in ['raylet', 'gcs_server']]
    inventory = {'verified_at_utc': now,
                 'gpu': command(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader']),
                 'services': command(['supervisorctl', 'status'])}
    assert len(inventory['gpu']['stdout'].splitlines()) == 4
    assert all(int(line.split(',')[1].split()[0]) <= 10 for line in inventory['gpu']['stdout'].splitlines())
    for service in ['real_ppo_252b0d3795', 'real_ppo_guard_252b0d3795']:
        assert any(line.split()[0] == service and line.split()[1] == 'STOPPED'
                   for line in inventory['services']['stdout'].splitlines())
    write(CASE/'cleanup_verified.json', inventory)
    shutil.copyfile('/var/log/portal/real_ppo_252b0d3795.log', CASE/'service.log')

    budget = {'scope': 'closed supplemental minimal verification; CPU documentation after stop excluded',
              'cap_sec': 900, 'elapsed_sec_at_service_stop': result['budget_status_at_stop']['elapsed_sec'],
              'unused_sec_at_stop': result['budget_status_at_stop']['remaining']['P0_P3_sec'],
              'attempt_01_sec': first['experiment_wall_sec'], 'retry_sec': result['experiment_wall_sec'],
              'service_attempt_wall_sec_sum': first['experiment_wall_sec'] + result['experiment_wall_sec'],
              'four_GPU_hour_equivalent_upper_bound': 4*(first['experiment_wall_sec']+result['experiment_wall_sec'])/3600,
              'parent_clock_unchanged': True, 'parent_clock_sha256': baseline['parent_clock_sha256'],
              'parent_last_recorded_elapsed_sec': protocol['budget_amendment']['parent_last_recorded_elapsed_sec'],
              'parent_remaining_not_recomputed_or_reset': True, 'P4_P5_allocations_unchanged': True,
              'automatic_reuse_or_rerun_authorized': False}
    write(CASE/'budget_closure.json', budget)
    checks = ['current_sources_configs_and_prior_gates_bound', 'four_GPU_real_model_startup',
              'three_normal_HTTP_completions', 'durable_gap_free_all_producer_ledger',
              'fixed_initial_routing_from_engine_admission', 'actual_migration_with_matching_ids_and_KV_blocks',
              'duplicate_RPC_single_coordinator_attempt', 'stale_candidate_rejected_without_dispatch',
              'real_observation_and_reward_replayed_from_raw_events', 'owned_processes_stopped_GPU_released',
              'supplemental_900s_cap_parent_clock_preserved']
    verification = {'status': 'passed', 'scope': 'amended minimal feasibility only', 'verified_at_utc': now,
                    'episodes': summaries, 'checks': [{'name': n, 'status': 'passed'} for n in checks],
                    'old_gates_hash_valid': old_gates, 'production_reference_scales_measured': False,
                    'original_100ms_latency_gate': 'not accepted; observed values retained',
                    'full_original_05_accepted': False, 'P3_opportunity_gate_passed': False, 'PPO_updates': 0}
    write(CASE/'verification.json', verification)
    argv = [str(ROOT/'.venv/bin/python'), '-u', str(CASE/'run_minimal.py')]
    commands = {'scope': 'actual execution record; completed, do not automatically rerun', 'cwd': str(ROOT),
                'preflight': {'CPU_test_exit_code': 0, 'CPU_test_count': 4, 'log': prefix+'/cpu_preflight_tests.log',
                              'initial_dry_run': {'argv': argv+['--dry-run'], 'exit_code': 0,
                                                  'GPU_started': False, 'protocol_revision': 1}},
                'attempts': [{'argv': argv, 'exit_code': 1, 'runtime': prefix+'/attempt_01/runtime.yaml',
                              'runner_sha256': sha(CASE/'attempt_01/run_minimal.py'),
                              'result': prefix+'/attempt_01/result.json', 'reason': 'diagnosed CPU booking deadlock; interrupted and cleaned'},
                             {'argv': argv, 'exit_code': 0, 'runtime': prefix+'/runtime.yaml',
                              'runner_sha256': sha(CASE/'run_minimal.py'), 'result': prefix+'/result.json',
                              'stdout_log': prefix+'/live_execution.log'}],
                'effective_runtime': prefix+'/runtime.yaml', 'ray_num_cpus': 16, 'ray_port': 20000,
                'protocol': prefix+'/protocol.json', 'raw_evidence_replay': [str(ROOT/'.venv/bin/python'), str(CASE/'finalize_minimal.py')],
                'original_05_07_command_manifests_unchanged': True}
    write(RUN/'commands/05_06_minimal.json', commands)
    sources = dict(baseline['source_hashes'])
    for p in [CASE/'run_minimal.py', Path(__file__)]:
        sources[str(p.relative_to(ROOT))] = sha(p)
    configs = dict(baseline['root_config_hashes'])
    for p in [CASE/'protocol.json', CASE/'runtime.yaml', CASE/'budget.json', RUN/'commands/05_06_minimal.json', *sorted((CASE/'traces').glob('*.json'))]:
        configs[str(p.relative_to(RUN))] = sha(p)
    evidence = {}
    for directory in [CASE/'runs', CASE/'causal_events', CASE/'attempt_01', CASE/'prior']:
        for p in sorted(directory.rglob('*')):
            if p.is_file(): evidence[str(p.relative_to(RUN))] = sha(p)
    for name in ['baseline.json', 'result.json', 'ready.json', 'verification.json', 'cleanup_verified.json',
                 'mini_reward_scales.json', 'cpu_preflight_tests.log', 'budget_closure.json', 'service.log', 'live_execution.log']:
        evidence[prefix+'/'+name] = sha(CASE/name)
    for name in old_gates:
        evidence[name] = sha(RUN/name)
    gate = {'stage': '05_06_minimal_feasibility', 'status': 'passed', 'scope': verification['scope'],
            'created_at_utc': now, 'checks': verification['checks'], 'source_hashes': sources,
            'source_hash': digest(sources), 'config_hashes': configs, 'config_hash': digest(configs),
            'evidence_hashes': evidence, 'result_path': prefix+'/result.json',
            'original_p1_p2_live_acceptance': False, 'opportunity_gate_passed': False,
            'PPO_training_ready': False, 'not_claimed': protocol['not_claimed']}
    gate_path = RUN/'gates/minimal_feasibility.json'
    write(gate_path, gate)
    cli.validate_gate(gate_path, RUN)

    limits = [*protocol['not_claimed'],
              'Active decision latency exceeds original 100ms P95 threshold; this is not a latency pass.',
              'Original runtime declares 12 Ray CPUs and cannot place all four instances; use the proven derived 16-CPU runtime for any future reviewed launch.',
              'Three requests establish engineering feasibility only; no throughput comparison or PPO advantage is inferred.']
    common = {'status': 'passed', 'status_scope': 'user-amended minimal feasibility only', 'updated_at_utc': now,
              'run_id': RUN.name, 'run_dir': str(RUN), 'user_amendment': prefix+'/protocol.json',
              'minimal_gate': 'gates/minimal_feasibility.json', 'minimal_gate_sha256': sha(gate_path),
              'source_hashes': sources, 'config_hashes': configs, 'evidence_hashes': evidence,
              'actual_commands_and_exit_codes': commands, 'budget': budget, 'limitations': limits,
              'next_step_executable': False, 'next_GPU_step_executable': False,
              'next_step': 'Stop after this amended 05/06 delivery; no automatic 07, training, or final evaluation.',
              'effective_runtime': prefix+'/runtime.yaml', 'changed_files': [prefix+'/', 'gates/minimal_feasibility.json',
              'commands/05_06_minimal.json', 'handoff/05.md', 'handoff/05.json', 'handoff/06.md', 'handoff/06.json',
              'rl_scheduler/runs/real_ppo_handoff/active_run.json']}
    h5 = dict(common, step='05', stage='05_minimal_live_feasibility', original_full_smoke_status='incomplete',
              gpu_started=True, ray_started=True, real_requests=3, real_successful_migrations=1,
              live_evidence=summaries, original_live_gates_created=False,
              prior_handoff=prefix+'/prior/handoff/05.json')
    h6 = dict(common, step='06', stage='06_CPU_preparation_and_minimal_live_path',
              CPU_preparation_status='passed', prior_CPU_test_count=58, current_targeted_CPU_test_count=4,
              prior_preparation_handoff=prefix+'/prior/handoff/06.json', p3_ready_gate='gates/p3_ready.json',
              p3_ready_gate_sha256=sha(RUN/'gates/p3_ready.json'),
              minimal_observation_reward_replay='passed', production_C_ref=None, production_N_ref=None,
              diagnostic_scales=read(CASE/'mini_reward_scales.json'), live_evidence=summaries,
              opportunity_experiments_run=0, opportunity_gate_status='incomplete', PPO_updates=0,
              frozen_P3_artifacts_unchanged=['opportunity_spec.yaml', 'commands/07_opportunity.json',
                                            'reward_initial.yaml', 'final_burst_spec.yaml'])
    write(RUN/'handoff/05.json', h5)
    write(RUN/'handoff/06.json', h6)
    active = next(e for e in summaries if e['label'] == 'migration')
    latency = active['decision_latency_p95_ms']
    shared = f'''
用户本轮明确要求简化 05/06，以最短流程验证可行性。本次替代协议见 `{prefix}/protocol.json`；原始交接保存在 `{prefix}/prior/handoff/`。完整旧 smoke 和机会实验的协议、命令、证据均保留。

实际流程为一次有效集群启动、10 秒 Stay 基线（1 请求）和 30 秒主动迁移（2 请求），随后停服。四张 RTX 3060、Qwen2-0.5B、TP=1、Ray 20000、API 18080 及迁移安全约束保持原值。首次启动因 Ray 仅声明 12 个逻辑 CPU，无法同时容纳四个 3-CPU placement group 和 Manager/Scaler，诊断后停止；失败记录在 `{prefix}/attempt_01/`。仅派生运行配置改为 16 CPU，限定一次重试后成功。根配置尚为 12，未来启动应采用本次已验证的 `{prefix}/runtime.yaml` 并重新核对前置。

3/3 请求正常完成，客户端累计 544 个输出 token。两回合原始事件重放审计均通过，每回合 7 个生产者检查点持久化、pending_delivery=0、无序列缺口，真实初始路由与引擎接纳匹配，清理确认 KV 回归空闲基线。1 次原协调器迁移成功，实际请求 ID 与候选 ID 一致，复制 19 blocks；同 decision_id 重复 RPC 返回同一成功终态且未增加协调器尝试，过期候选明确拒绝（SR_candidate_changed）。

本次服务窗口（含首次失败、修正间隔及清理）为 {budget['elapsed_sec_at_service_stop']:.3f} 秒，低于本轮补充上限 900 秒；两次服务尝试累计 {budget['service_attempt_wall_sec_sum']:.3f} 秒。原实验时钟与预算字节未改，P4/P5 配额未挪用。停止时尚余 {budget['unused_sec_at_stop']:.3f} 秒仅作本次封账记录，不自动用于下一实验。四卡均已回到 1 MiB，本任务服务 STOPPED，Ray 无残留，管理服务保持运行。

最小可行性 gate：`gates/minimal_feasibility.json`。源码、有效配置、原始日志与台账均绑定 SHA256；命令及退出码见 `commands/05_06_minimal.json`。4 项本次针对性 CPU 测试通过；交付前逐事件重放了审计、迁移对账和奖励，并与已保存结果逐项比较。旧 04 review、P1/P2 offline 和 P3 CPU gate 的当前哈希仍有效。

验收范围限制：原 10 次终态/5 次成功、独立 shadow、OFF/ON 开销对照与机会实验均未完成，未生成旧 `p1_live`/`p2_live` passed gate。主动阶段决策 P95 为 {latency:.3f} ms，超过原 100 ms 门槛，功能可行不代表原延迟验收通过。没有运行 PPO 或最终评估，不推断性能收益。完成本次 05/06 交付后停止，不自动进入 07。
'''
    (RUN/'handoff/05.md').write_text('# 05：最小真机可行性 passed；原完整 smoke incomplete\n'+shared)
    (RUN/'handoff/06.md').write_text('# 06：CPU 准备及最小真实观测/奖励链路 passed\n'+
        '\n保留此前 58 项 CPU 测试与 P3 协议准备通过的结论；`gates/p3_ready.json` 仅表示 CPU 准备，不表示机会实验通过。\n'+
        '\n本次使用真实客户端累计 token、生命周期事件、原协调器迁移起点和已确认终态，成功重建 38 个有限窗口奖励（基线 10、主动 28）；从磁盘原始事件重算与已保存结果一致。诊断尺度由本次极小基线测得后按既有 floor=1 规则得到 C_ref=N_ref=1，仅用于链路检查，原 reward_initial.yaml 的正式 C_ref/N_ref 仍为 null。负的主动回合奖励也原样保留；不能将两种不同负载的回合比较为策略收益。\n'+
        '\nQPS 3/6/9 健康选档、6 个探测及 12 个 Stay/Balanced 配对未运行，健康档位及正式奖励尺度尚未实测，opportunity gate incomplete。原 opportunity_spec.yaml、commands/07_opportunity.json、reward_initial.yaml、final_burst_spec.yaml 不变，原完整 07 清单不可据此自动放行。\n'+shared)
    pointer_path = ROOT/'rl_scheduler/runs/real_ppo_handoff/active_run.json'
    pointer = read(pointer_path)
    pointer.update(updated_at_utc=now, active_stage='05_06_minimal_feasibility_complete', last_completed_step='06_amended_minimal_scope',
                   last_attempted_step='05_06_minimal_live_verification', smoke_status='minimal_feasibility_passed_full_smoke_incomplete',
                   minimal_gate='gates/minimal_feasibility.json', minimal_gate_sha256=sha(gate_path),
                   minimal_protocol=prefix+'/protocol.json', effective_runtime=prefix+'/runtime.yaml',
                   command_manifest='commands/05_06_minimal.json', next_step='stopped_after_requested_05_06_update',
                   next_GPU_step_executable=False, budget_preflight_reason='Supplemental minimal verification closed within 900s; original parent clock preserved; no automatic next-stage authorization.',
                   last_handoff=str(RUN/'handoff/06.json'), latest_handoffs=[str(RUN/'handoff/05.json'), str(RUN/'handoff/06.json')])
    write(pointer_path, pointer)
    for path in [gate_path, *[RUN/n for n in old_gates]]:
        cli.validate_gate(path, RUN)
    print(json.dumps({'status': 'passed', 'episodes': summaries, 'elapsed_sec': budget['elapsed_sec_at_service_stop'],
                      'gate': str(gate_path), 'updated_handoffs': ['05', '06'], 'PPO_updates': 0}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
