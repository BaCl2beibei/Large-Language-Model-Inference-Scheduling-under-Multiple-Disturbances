"""Reproduce P1 CPU checks and the hand-computed exported example; no GPU imports."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_scheduler.tests.test_real_metrics import Fixture


def main():
    commands = []
    for label, cmd in [
        ('unit_and_http', [str(ROOT / '.venv/bin/python'), '-m', 'unittest',
                           'rl_scheduler.tests.test_real_metrics', '-v']),
        ('compile', [str(ROOT / '.venv/bin/python'), '-m', 'py_compile',
                     'rl_scheduler/real_metrics.py', 'rl_scheduler/real_client.py',
                     'rl_scheduler/tests/test_real_metrics.py', 'llumnix/metrics/causal_events.py',
                     'llumnix/entrypoints/vllm/real_metrics_stream.py',
                     'llumnix/entrypoints/vllm/api_server.py']),
        ('diff_check', ['git', 'diff', '--check']),
        ('gpu_inventory_only', ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                '--format=csv,noheader']),
    ]:
        started = time.monotonic()
        result = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        path = RUN / 'p1_evidence' / (label + '.log')
        path.write_bytes(result.stdout)
        commands.append({'command': cmd, 'exit_code': result.returncode,
                         'wall_time_sec': time.monotonic() - started,
                         'log': str(path.relative_to(RUN)),
                         'log_sha256': hashlib.sha256(result.stdout).hexdigest()})
    f = Fixture()
    f.plan_send('a', 0); f.output(1, 1); f.output(1.3, 4); f.terminal(2)
    f.plan_send('b', 1); f.output(2, 1, 'b'); f.terminal(4, 'b')
    f.plan_send('c', 2); f.output(3, 2, 'c')
    f.add('planned', 0, 'not_sent', scheduled_ns=3_000_000_000)
    f.ack()
    frozen = f.ledger.freeze_score()
    f.output(9, 4, 'c'); f.terminal(10, 'c'); f.ack(12)
    result = f.ledger.export(RUN / 'p1_evidence/hand_calculated', cleanup_through_ns=12_000_000_000)
    checks = {
        'all_commands_passed': all(row['exit_code'] == 0 for row in commands),
        'no_gpu_compute_processes_at_final_inspection': (RUN/'p1_evidence/gpu_inventory_only.log').read_text().strip() == '',
        'completed': frozen['fixed']['completed'] == 2,
        'offered': frozen['fixed']['offered'] == 4,
        'completion_rate': frozen['fixed']['completion_rate'] == .5,
        'slo_goodput_rps': frozen['fixed']['slo_goodput_rps'] == .25,
        'visible_tokens': frozen['fixed']['client_output_tokens'] == 7,
        'outstanding_integral': frozen['fixed']['outstanding_request_seconds'] == 11,
        'gap_integral': frozen['fixed']['output_gap_request_seconds'] == 5,
        'frozen_score_unchanged': f.ledger.freeze_score() == frozen,
        'cleanup_passed': result['cleanup_status'] == 'passed',
    }
    (RUN/'p1_evidence/commands.json').write_text(json.dumps(commands, indent=2)+'\n')
    summary = {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
               'cpu_tests': 28, 'independent_live_tests': 0, 'gpu_experiments': 0}
    (RUN/'p1_evidence/verification.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
