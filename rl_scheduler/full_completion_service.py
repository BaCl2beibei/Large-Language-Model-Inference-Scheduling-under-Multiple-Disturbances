"""Foreground owner for the full-cohort experiment's private Llumnix cluster.

Supervisor owns this process; this process owns Ray and the API server.  It has
no experiment or request completion deadline.  Shutdown is signal driven and
only recorded child identities are stopped.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import shutil

from .real_budget import atomic_json
from .real_service import OwnedProcesses


def _owned_ray_processes(ray_dir: Path):
    import psutil
    matches = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if (proc.info['name'] in ('raylet', 'gcs_server') and
                any(str(ray_dir) in str(part) for part in (proc.info['cmdline'] or []))):
            matches.append(proc.info['pid'])
    return matches


def _ignore_runtime_sockets(directory, names):
    """Ray owns live Unix sockets that cannot be copied into run artifacts."""
    ignored = []
    for name in names:
        path = Path(directory) / name
        try:
            if path.is_socket() or path.is_fifo():
                ignored.append(name)
        except OSError:
            ignored.append(name)
    return ignored


def run_owner(root: Path, run_dir: Path, config_path: Path):
    import psutil
    import yaml
    from llumnix.metrics.causal_events import CAUSAL_DIAGNOSTIC_SAMPLE_INTERVAL

    # Ray proxy actors were producing multi-hundred-MB core files in the
    # Vast-managed data area when the private cluster shut down. Keep these
    # experiment processes from creating host-side core dumps.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    cfg = yaml.safe_load(config_path.read_text())
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    lock_path = run_dir.parent / '.real_ppo_cluster.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner = OwnedProcesses(run_dir / 'owned_processes.json')
    ray_dir = Path(cfg['service']['ray_temp_dir'])
    ray_attempted = False
    child = None
    try:
        if any(proc.info['name'] in ('raylet', 'gcs_server')
               for proc in psutil.process_iter(['name'])):
            raise RuntimeError('Ray was running before the full-cohort service')
        gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                             check=True, capture_output=True, text=True, timeout=5.)
        if gpu.stdout.strip():
            raise RuntimeError('GPU already used by another process')
        if ray_dir.exists():
            # A stopped selection service can leave Ray's private temp tree
            # behind. Preserve it under a timestamped sibling and start a
            # clean tree at the frozen config path for a later stage.
            archive_dir = ray_dir.with_name(ray_dir.name + '_archived_' + str(time.time_ns()))
            ray_dir.rename(archive_dir)
            atomic_json(run_dir / 'runtime' / 'ray_temp_archive.json', {
                'previous_ray_temp_dir': str(archive_dir),
                'active_processes_checked': True,
                'archived_utc_epoch_ns': time.time_ns(),
                'reason': 'stale private Ray temp directory from an already stopped owned service',
            })
        ray_dir.mkdir(parents=True, exist_ok=False)
        causal_dir = run_dir / 'causal_events' / ('service_' + str(time.time_ns()))
        causal_dir.mkdir(parents=True, exist_ok=False)
        env = os.environ.copy()
        env.update(cfg['service']['env'])
        env['PATH'] = str(root / '.venv/bin') + ':' + env.get('PATH', '')
        env['LLUMNIX_CAUSAL_TRACE_DIR'] = str(causal_dir)
        env['LLUMNIX_CAUSAL_SAMPLE_HIGH_FREQ'] = '1'
        env['PYTHONUNBUFFERED'] = '1'
        atomic_json(run_dir / 'service_session.json', {
            'schema': 'full-completion-service-v1',
            'causal_dir': str(causal_dir), 'ray_temp_dir': str(ray_dir),
            'high_frequency_diagnostic_sample_interval': CAUSAL_DIAGNOSTIC_SAMPLE_INTERVAL,
            'process_core_dumps_disabled': True,
            'started_wall': time.time(), 'pid': os.getpid(),
        })
        ray_cmd = [str(root / '.venv/bin/ray'), 'start', '--head',
                   '--node-ip-address=127.0.0.1', '--port=' + str(cfg['service']['ray_port']),
                   '--temp-dir=' + str(ray_dir), '--num-gpus=4',
                   '--num-cpus=' + str(cfg['service']['ray_num_cpus']),
                   '--include-dashboard=false', '--disable-usage-stats']
        ray_attempted = True
        subprocess.run(ray_cmd, env=env, cwd=root, check=True)
        for pid in _owned_ray_processes(ray_dir):
            owner.register(pid, 'owned_ray_head')
        if not owner.path.exists():
            raise RuntimeError('could not establish Ray process ownership')
        if stopping:
            raise RuntimeError('service startup interrupted')
        # Keep the API/server traceback alongside Ray's worker logs. This is
        # especially useful when a remote actor error is truncated in an HTTP
        # response while the invocation itself is running unattended.
        server_log = (run_dir / 'api_server.log').open('ab', buffering=0)
        child = subprocess.Popen(cfg['service']['server_argv'], env=env, cwd=root,
                                 stdout=server_log, stderr=subprocess.STDOUT)
        owner.register(child.pid, 'owned_api')
        atomic_json(run_dir / 'service_process.json', {
            'pid': child.pid, 'argv': cfg['service']['server_argv'],
            'started_wall': time.time(), 'causal_dir': str(causal_dir),
            'ray_temp_dir': str(ray_dir), 'config': str(config_path),
        })
        while child.poll() is None and not stopping:
            owner.capture_children()
            time.sleep(.25)
        if child.poll() not in (None, 0) and not stopping:
            raise RuntimeError('API process exited with status ' + str(child.returncode))
    finally:
        cleanup_error = None
        if ray_attempted:
            for pid in _owned_ray_processes(ray_dir):
                try:
                    owner.register(pid, 'owned_ray_head')
                except RuntimeError:
                    pass
        try:
            owner.stop(timeout=20.)
        except BaseException as exc:
            cleanup_error = type(exc).__name__ + ': ' + str(exc)
        if child is not None and child.poll() is None:
            try:
                child.wait(timeout=5.)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if 'server_log' in locals():
            server_log.close()
        if ray_dir.exists():
            try:
                shutil.copytree(ray_dir, run_dir / 'runtime/ray', dirs_exist_ok=True,
                                ignore=_ignore_runtime_sockets)
            except BaseException as exc:
                cleanup_error = cleanup_error or (type(exc).__name__ + ': ' + str(exc))
        atomic_json(run_dir / 'service_stopped.json', {
            'stopped_wall': time.time(), 'signal_or_child_exit': True,
            'cleanup_error': cleanup_error,
        })
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        if cleanup_error:
            raise RuntimeError('owned service cleanup incomplete: ' + cleanup_error)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    run_owner(args.root.resolve(), args.run_dir.resolve(), args.config.resolve())


if __name__ == '__main__':
    main()
