"""One-invocation budget override for the explicitly authorized P3 experiment.

The persistent experiment clock and original phase budgets are still updated and
never rewritten. Only this invocation sees a monotonic execution lease in place
of the cumulative-budget gate. The lease also lets the owned watchdog stop a
stalled invocation without affecting other services.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import time

from .real_budget import BudgetExhausted, BudgetLedger, atomic_json
from .real_service import ManagedService


def _update_json(path, update):
    path = Path(path)
    with path.with_suffix(path.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = json.loads(path.read_text())
        update(value)
        atomic_json(path, value)
        fcntl.flock(lock, fcntl.LOCK_UN)


def create_lease(path, *, run_id, invocation_id, max_runtime_sec):
    path = Path(path)
    if path.exists():
        raise FileExistsError('budget override lease already exists')
    if not isinstance(max_runtime_sec, (int, float)) or max_runtime_sec <= 0:
        raise ValueError('positive invocation lease required')
    now_mono = time.monotonic_ns()
    now_wall = time.time()
    value = {
        'schema': 'real-opportunity-budget-lease-v1',
        'run_id': run_id,
        'invocation_id': invocation_id,
        'created_wall': now_wall,
        'created_monotonic_ns': now_mono,
        'max_runtime_sec': float(max_runtime_sec),
        'deadline_monotonic_ns': now_mono + int(max_runtime_sec * 1e9),
        'closed_wall': None,
        'close_reason': None,
    }
    atomic_json(path, value)
    return value


def close_lease(path, reason):
    def close(value):
        if value.get('closed_wall') is None:
            value['closed_wall'] = time.time()
            value['close_reason'] = str(reason)
    _update_json(path, close)


def continue_lease(original_path, continuation_path, *, run_id, invocation_id,
                   max_runtime_sec=None, allow_extend=False):
    """Open a bounded, explicit manual continuation of one P3 invocation.

    By default the original absolute deadline is retained. A separately
    authorized continuation may request one bounded runtime window; the
    previous lease remains immutable and the child records its exact extension.
    """
    original_path=Path(original_path);continuation_path=Path(continuation_path)
    original=json.loads(original_path.read_text())
    if original.get('run_id')!=run_id or original.get('invocation_id')!=invocation_id:
        raise RuntimeError('continuation does not match the original authorized invocation')
    if original.get('closed_wall') is None:
        raise RuntimeError('only a cleaned, closed invocation can be continued')
    now_mono=time.monotonic_ns();now_wall=time.time()
    remaining=(int(original['deadline_monotonic_ns'])-now_mono)/1e9
    if remaining<=0 and not allow_extend:
        raise BudgetExhausted('original invocation lease deadline has passed')
    if continuation_path.exists():
        raise FileExistsError('continuation lease already exists')
    requested=float(max_runtime_sec) if max_runtime_sec is not None else remaining
    if requested<=0:
        raise ValueError('positive continuation lease required')
    if requested>remaining and not allow_extend:
        raise BudgetExhausted('continuation would extend the original invocation deadline')
    deadline=now_mono+int(requested*1e9) if requested>remaining else int(original['deadline_monotonic_ns'])
    value={**original,'created_wall':now_wall,'created_monotonic_ns':now_mono,
        'max_runtime_sec':(deadline-now_mono)/1e9,'closed_wall':None,'close_reason':None,
        'continuation_of':str(original_path.resolve()),
        'original_deadline_monotonic_ns':int(original['deadline_monotonic_ns']),
        'deadline_monotonic_ns':deadline,
        'deadline_extended_sec':max(0.,(deadline-int(original['deadline_monotonic_ns']))/1e9)}
    atomic_json(continuation_path,value)
    return value


class OpportunityBudgetLedger:
    """P3-only ledger view: preserve cumulative accounting, gate on this lease."""

    def __init__(self, run_dir, lease_path):
        self.raw = BudgetLedger(run_dir)
        self.run_dir = self.raw.run_dir
        self.input = self.raw.input
        self.lease_path = Path(lease_path)
        lease = json.loads(self.lease_path.read_text())
        if lease.get('run_id') != self.input['run_id']:
            raise RuntimeError('opportunity lease run ID mismatch')

    def _lease_remaining(self):
        lease = json.loads(self.lease_path.read_text())
        if lease.get('closed_wall') is not None:
            return 0., lease
        return max(0., (lease['deadline_monotonic_ns'] - time.monotonic_ns()) / 1e9), lease

    def status(self):
        raw = self.raw.status()  # advances persistent wall/monotonic accounting
        remaining, lease = self._lease_remaining()
        raw['cumulative_budget_remaining'] = dict(raw['remaining'])
        raw['remaining'] = {
            'total_experiment_wall_time_sec': remaining,
            'P0_P3_sec': remaining,
            'P4_sec': remaining,
            'P5_sec': remaining,
        }
        raw['exhausted'] = remaining <= 0.
        raw['budget_override'] = {
            'scope': 'single_07_opportunity_invocation',
            'invocation_id': lease['invocation_id'],
            'lease_remaining_sec': remaining,
            'persistent_budget_modified': False,
        }
        return raw

    def activate(self, phase='P0_P3', *, required_sec=60.):
        if phase not in ('P0_P3', 'P4', 'P5'):
            raise ValueError('unknown phase')
        remaining, _ = self._lease_remaining()
        if remaining < required_sec:
            raise BudgetExhausted('opportunity invocation lease reserve insufficient')
        with self.raw._locked() as state:
            if state['first_started_wall'] is None:
                now = self.raw.wall()
                state.update(first_started_wall=now, origin_mono=self.raw.mono(),
                             last_wall=now, last_mono=self.raw.mono(), boot_id=self.raw.boot_id)
            state['phase'] = phase
        return self.status()

    def require(self, required_sec, phase='P0_P3'):
        status = self.status()
        if status['remaining']['total_experiment_wall_time_sec'] < required_sec:
            raise BudgetExhausted('opportunity invocation lease reserve insufficient')
        return status

    def expire(self):
        def expire_lease(value):
            if value.get('closed_wall') is None:
                value['closed_wall'] = time.time()
                value['close_reason'] = 'invocation_lease_reserve_reached'
        _update_json(self.lease_path, expire_lease)
        self.raw.status()


class OpportunityManagedService(ManagedService):
    """Same owned-service lifecycle, with a run-specific P3 lease entrypoint."""

    def __init__(self, root, run_dir, config_path, *, invocation_id, lease_path, artifact_dir=None, service_tag=None):
        super().__init__(root, run_dir, config_path)
        suffix = hashlib.sha256((self.run_id + ':07:' + (service_tag or invocation_id)).encode()).hexdigest()[:10]
        self.server_name = 'real_ppo_op_' + suffix
        self.guard_name = 'real_ppo_op_guard_' + suffix
        self.invocation_id = invocation_id
        self.lease_path = Path(lease_path).resolve()
        import yaml
        original = yaml.safe_load(self.config_path.read_text())
        original_cpus = int(original['service']['ray_num_cpus'])
        available_cpus = len(os.sched_getaffinity(0))
        # Ray placement needs 3 CPU per each of the four fixed Llumlet actors,
        # plus CPU for the manager/scaler actors. 14 is the observed minimum;
        # 16 leaves two scheduling slots and fits this instance's 80-CPU cpuset.
        effective_cpus = max(original_cpus, 16)
        if effective_cpus > available_cpus:
            raise RuntimeError(f'P3 Ray CPU override {effective_cpus} exceeds cpuset {available_cpus}')
        runtime_base=Path(artifact_dir) if artifact_dir is not None else self.run_dir/'opportunity_runs'/invocation_id
        runtime = runtime_base / 'runtime_override.yaml'
        if runtime.exists():
            raise FileExistsError('opportunity runtime override already exists')
        original['service']['ray_num_cpus'] = effective_cpus
        runtime.write_text(yaml.safe_dump(original, sort_keys=False))
        self.config_path = runtime.resolve()
        self.runtime_override = {
            'path': str(runtime.relative_to(self.run_dir)),
            'original_ray_num_cpus': original_cpus,
            'effective_ray_num_cpus': effective_cpus,
            'available_cpuset_cpus': available_cpus,
            'reason': 'four fixed actors each reserve 3 Ray CPUs; 2 additional CPUs are used by the manager/scaler',
            'workload_or_protocol_changed': False,
        }

    def prepare(self):
        directory = self.run_dir / 'services'
        directory.mkdir(parents=True, exist_ok=True)
        files = []
        for name, role in ((self.server_name, 'server'), (self.guard_name, 'watchdog')):
            argv = [str(self.root / '.venv/bin/python'), '-m', 'rl_scheduler.real_opportunity_service',
                    '--role', role, '--root', str(self.root), '--run-dir', str(self.run_dir),
                    '--config', str(self.config_path), '--lease', str(self.lease_path),
                    '--server-name', self.server_name, '--guard-name', self.guard_name]
            wrapper = directory / (name + '.sh')
            wrapper.write_text('#!/bin/bash\nset -e\n. /opt/supervisor-scripts/utils/logging.sh\n'
                               '. /opt/supervisor-scripts/utils/environment.sh\ncd '
                               + shlex.quote(str(self.root)) + '\nexec ' + shlex.join(argv) + '\n')
            wrapper.chmod(0o755)
            conf = directory / (name + '.conf')
            conf.write_text(f'''[program:{name}]
environment=PROC_NAME="%(program_name)s"
command={wrapper}
autostart=false
autorestart=false
stopasgroup=true
killasgroup=true
stopsignal=TERM
stopwaitsecs=60
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_events_enabled=true
stdout_logfile_maxbytes=0
''')
            files.extend((str(wrapper), str(conf)))
        return files
