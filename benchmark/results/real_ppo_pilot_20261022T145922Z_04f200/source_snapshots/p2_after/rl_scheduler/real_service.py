"""Supervisor-managed private serving and independent persistent budget watchdog.

Preparing these files is CPU-only. install/start are invoked only by a later
explicit live runner, after gate and ownership checks. No global ray stop/pkill.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from .real_budget import BudgetLedger, atomic_json

PROTECTED={'caddy','instance_portal','tunnel_manager','jupyter','cron','tensorboard','syncthing','pyworker'}


def process_identity(pid):
    import psutil
    try:
        p=psutil.Process(pid)
        return {'pid':p.pid,'created':p.create_time()}
    except psutil.NoSuchProcess:return None


def identity_alive(record):
    if not record:return False
    return process_identity(record['pid'])==record


class OwnedProcesses:
    def __init__(self,path):self.path=Path(path)

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            data=json.loads(self.path.read_text()) if self.path.exists() else {'processes':[]}
            try:yield data
            finally:
                atomic_json(self.path,data)
                fcntl.flock(lock,fcntl.LOCK_UN)

    def register(self,pid,role):
        identity=process_identity(pid)
        if identity is None:raise RuntimeError('owned process disappeared')
        with self._locked() as data:
            record={'identity':identity,'role':role}
            if record not in data['processes']:data['processes'].append(record)

    def capture_children(self):
        import psutil
        with self._locked() as data:
            additions=[]
            for record in data['processes']:
                if identity_alive(record['identity']):
                    try:
                        for child in psutil.Process(record['identity']['pid']).children(recursive=True):
                            identity=process_identity(child.pid)
                            if identity:additions.append({'identity':identity,'role':'descendant'})
                    except psutil.NoSuchProcess:pass
            by_identity={(r['identity']['pid'],r['identity']['created']):r for r in data['processes']+additions}
            data['processes']=list(by_identity.values())
        return data

    def stop(self,timeout=20.):
        import psutil
        data=self.capture_children();owned=[]
        for record in data['processes']:
            identity=record['identity']
            if identity_alive(identity) and identity['pid']!=os.getpid():
                proc=psutil.Process(identity['pid'])
                if proc.name() in ('supervisord','sshd','caddy'):
                    raise RuntimeError('protected management process in ownership ledger')
                owned.append(proc)
                try:proc.terminate()
                except psutil.NoSuchProcess:pass
        _,alive=psutil.wait_procs(owned,timeout=timeout)
        for proc in alive:
            identity=next((r['identity'] for r in data['processes'] if r['identity']['pid']==proc.pid),None)
            if identity_alive(identity):
                try:proc.kill()
                except psutil.NoSuchProcess:pass
        _,alive=psutil.wait_procs(alive,timeout=2.)
        if alive:raise RuntimeError('owned process cleanup not confirmed')


def service_names(run_id):
    import hashlib
    suffix=hashlib.sha256(run_id.encode()).hexdigest()[:10]
    return 'real_ppo_'+suffix, 'real_ppo_guard_'+suffix


class ManagedService:
    def __init__(self,root,run_dir,config_path,*,command_runner=subprocess.run,supervisor_dir='/etc/supervisor/conf.d'):
        self.root,self.run_dir,self.config_path=Path(root),Path(run_dir),Path(config_path)
        self.run=command_runner
        self.supervisor_dir=Path(supervisor_dir)
        self.run_id=json.loads((self.run_dir/'budget.json').read_text())['run_id']
        self.server_name,self.guard_name=service_names(self.run_id)
        self.owner=OwnedProcesses(self.run_dir/'owned_processes.json')

    def prepare(self):
        directory=self.run_dir/'services';directory.mkdir(parents=True,exist_ok=True)
        files=[]
        for name,role in [(self.server_name,'server'),(self.guard_name,'watchdog')]:
            argv=[str(self.root/'.venv/bin/python'),'-m','rl_scheduler.real_service','--role',role,
                  '--root',str(self.root),'--run-dir',str(self.run_dir),'--config',str(self.config_path)]
            import shlex
            wrapper=directory/(name+'.sh')
            wrapper.write_text('#!/bin/bash\nset -e\n. /opt/supervisor-scripts/utils/logging.sh\n. /opt/supervisor-scripts/utils/environment.sh\ncd '+shlex.quote(str(self.root))+'\nexec '+shlex.join(argv)+'\n')
            wrapper.chmod(0o755)
            conf=directory/(name+'.conf')
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
            files.extend([str(wrapper),str(conf)])
        return files

    def _supervisor(self,*argv,check=True,timeout_sec=65.):
        if any(name in PROTECTED for name in argv):raise RuntimeError('protected supervisor service')
        return self.run(['supervisorctl',*argv],check=check,capture_output=True,text=True,timeout=timeout_sec)

    def start(self):
        # Do not adopt any existing cluster. Foreign Ray and GPU processes are blockers.
        import psutil
        if any(p.info['name'] in ('raylet','gcs_server') for p in psutil.process_iter(['name'])):
            raise RuntimeError('foreign or unclean Ray cluster; refuse to adopt/stop')
        gpu=self.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
                     check=True,capture_output=True,text=True,timeout=5.)
        if gpu.stdout.strip():raise RuntimeError('GPU already used by another process')
        self.prepare()
        for name in (self.server_name,self.guard_name):
            source=self.run_dir/'services'/(name+'.conf');target=self.supervisor_dir/source.name
            if target.exists() and target.read_bytes()!=source.read_bytes():raise RuntimeError('conflicting supervisor owner')
            target.write_bytes(source.read_bytes())
        self._supervisor('reread');self._supervisor('update',self.server_name,self.guard_name)
        self._supervisor('start',self.guard_name)
        try:self._supervisor('start',self.server_name)
        except BaseException:
            self.stop();raise

    def stop(self):
        # Names derive exclusively from this run ID; never touch management/foreign names.
        self._supervisor('stop',self.server_name,check=False)
        self.owner.stop(timeout=15.)
        self._supervisor('stop',self.guard_name,check=False)

    def assert_running_owned(self):
        if not self.owner.path.exists():raise RuntimeError('no owned service ledger')
        records=json.loads(self.owner.path.read_text())['processes']
        if not any(r['role']=='owned_api' and identity_alive(r['identity']) for r in records):
            raise RuntimeError('no living API process owned by this run')


def _read_config(path):
    import yaml
    return yaml.safe_load(Path(path).read_text())


def run_server(args):
    import psutil
    cfg=_read_config(args.config)
    run_dir=Path(args.run_dir)
    budget=BudgetLedger(run_dir)
    budget.activate('P0_P3',required_sec=cfg['service']['startup_timeout_sec']+60)
    owner=OwnedProcesses(run_dir/'owned_processes.json')
    stopping=False
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    # Exclusion is held for the entire service lifetime, preventing two task clusters.
    lock=(run_dir.parent/'.real_ppo_cluster.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ray_attempted=False
    try:
        if any(p.info['name'] in ('raylet','gcs_server') for p in psutil.process_iter(['name'])):
            raise RuntimeError('Ray was already running before this owned service')
        ray_dir=Path(cfg['service']['ray_temp_dir'])
        env=os.environ.copy();env.update(cfg['service']['env'])
        env['PATH']=str(Path(args.root)/'.venv/bin')+':'+env.get('PATH','')
        causal_dir=run_dir/'causal_events'/('service_'+str(time.time_ns()))
        causal_dir.mkdir(parents=True,exist_ok=False)
        env['LLUMNIX_CAUSAL_TRACE_DIR']=str(causal_dir)
        atomic_json(run_dir/'service_session.json',{'run_id':budget.input['run_id'],
                    'causal_dir':str(causal_dir),'ray_temp_dir':str(ray_dir),'started_wall':time.time()})
        ray_cmd=[str(Path(args.root)/'.venv/bin/ray'),'start','--head','--node-ip-address=127.0.0.1',
                 '--port='+str(cfg['service']['ray_port']),'--temp-dir='+str(ray_dir),
                 '--num-gpus=4','--num-cpus='+str(cfg['service']['ray_num_cpus']),
                 '--include-dashboard=false','--disable-usage-stats']
        ray_attempted=True
        subprocess.run(ray_cmd,env=env,check=True,timeout=60.)
        # Ray daemons detach; bind ownership only to process command lines bearing this session path.
        for proc in psutil.process_iter(['pid','name','cmdline']):
            if proc.info['name'] in ('raylet','gcs_server') and any(str(ray_dir) in str(x) for x in proc.info['cmdline'] or []):
                owner.register(proc.pid,'owned_ray_head')
        if not owner.path.exists():raise RuntimeError('could not establish owned Ray daemon identities')
        if stopping:raise RuntimeError('service startup interrupted')
        budget.require(60.)
        child=subprocess.Popen(cfg['service']['server_argv'],env=env,cwd=args.root)
        owner.register(child.pid,'owned_api')
        while child.poll() is None and not stopping:
            owner.capture_children()
            status=budget.status()
            if min(status['remaining']['total_experiment_wall_time_sec'],status['remaining']['P0_P3_sec'])<=60:
                stopping=True;budget.expire()
            time.sleep(.25)
    finally:
        # A failed/timeout ray-start may have detached partial daemons too.
        ray_dir=Path(cfg['service']['ray_temp_dir'])
        if ray_attempted:
            for proc in psutil.process_iter(['pid','name','cmdline']):
                if proc.info['name'] in ('raylet','gcs_server') and any(str(ray_dir) in str(x) for x in proc.info['cmdline'] or []):
                    owner.register(proc.pid,'owned_ray_head')
        owner.stop(timeout=20.)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def run_watchdog(args):
    budget=BudgetLedger(args.run_dir)
    service=ManagedService(args.root,args.run_dir,args.config)
    running=True
    def stop(signum,frame):
        nonlocal running
        running=False
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    while running:
        status=budget.status()
        if status['started'] and (status['exhausted'] or min(status['remaining']['total_experiment_wall_time_sec'],status['remaining']['P0_P3_sec'])<=60):
            budget.expire()
            try:
                service._supervisor('stop',service.server_name,check=False,timeout_sec=40.)
            except subprocess.TimeoutExpired:
                pass
            service.owner.stop(timeout=10.)
            break
        time.sleep(.25)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--role',choices=['server','watchdog'],required=True)
    for name in ('root','run-dir','config'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    return run_server(args) if args.role=='server' else run_watchdog(args)


if __name__=='__main__':main()
