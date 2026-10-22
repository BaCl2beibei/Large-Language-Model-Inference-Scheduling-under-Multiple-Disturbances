"""Flocked, fsynced budget clock shared by runner and independent supervisor guard.

The original P0 budget remains immutable input. After first service activation
wall time includes startup, cleanup, failures and handoff waiting; no restart can
reset the origin. Monotonic elapsed is used on the same boot to resist wall jumps.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    descriptor=os.open(str(path.parent),os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


class BudgetExhausted(RuntimeError):pass


class BudgetLedger:
    def __init__(self,run_dir,*,wall=time.time,mono=time.monotonic,boot_id=None):
        self.run_dir=Path(run_dir);self.path=self.run_dir/'experiment_clock.json'
        self.wall,self.mono=wall,mono
        self.boot_id=boot_id or Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        self.input=json.loads((self.run_dir/'budget.json').read_text())

    @contextmanager
    def _locked(self):
        self.run_dir.mkdir(parents=True,exist_ok=True)
        with (self.run_dir/'experiment_clock.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if self.path.exists():state=json.loads(self.path.read_text())
            else:
                state={'schema':1,'run_id':self.input['run_id'],'first_started_wall':None,
                       'origin_mono':None,'boot_id':self.boot_id,'last_wall':None,'last_mono':None,
                       'elapsed_sec':float(self.input['usage']['task_experiment_wall_time_sec']),
                       'phase':None,'phase_usage':{'P0_P3':0.,'P4':float(self.input['usage']['ppo_training_wall_time_sec']),'P5':0.},
                       'exhausted':False}
            if state['run_id']!=self.input['run_id']:raise RuntimeError('budget run ID mismatch')
            if state['first_started_wall'] is not None:
                if state['boot_id']==self.boot_id:
                    delta=max(0.,self.mono()-state['last_mono'],self.wall()-state['last_wall'])
                else:
                    delta=max(0.,self.wall()-state['last_wall'])
                    if self.wall()<state['last_wall']:raise RuntimeError('clock moved backwards across boot')
                state['elapsed_sec']+=delta
                if state['phase']:state['phase_usage'][state['phase']]+=delta
                state['boot_id']=self.boot_id;state['last_wall']=self.wall();state['last_mono']=self.mono()
            try:
                yield state
            finally:
                atomic_json(self.path,state)
                fcntl.flock(lock,fcntl.LOCK_UN)

    def _remaining(self,state):
        caps=self.input['phase_caps_sec']
        phase_caps={'P0_P3':caps['P0_P3_including_GPU_checks_and_opportunity'],
                    'P4':caps['P4_PPO_training'],'P5':caps['P5_final_evaluation_reserved']}
        total=max(0.,self.input['total_experiment_wall_time_sec']-state['elapsed_sec'])
        result={'total_experiment_wall_time_sec':total}
        result.update({name+'_sec':max(0.,phase_caps[name]-state['phase_usage'][name]) for name in phase_caps})
        return result

    def status(self):
        with self._locked() as state:
            remaining=self._remaining(state)
            return {'started':state['first_started_wall'] is not None,'elapsed_sec':state['elapsed_sec'],
                    'phase_usage':dict(state['phase_usage']),'remaining':remaining,'exhausted':state['exhausted']}

    def activate(self,phase='P0_P3',*,required_sec=60.):
        if phase not in ('P0_P3','P4','P5'):raise ValueError('unknown phase')
        with self._locked() as state:
            remaining=self._remaining(state)
            if state['exhausted'] or min(remaining['total_experiment_wall_time_sec'],remaining[phase+'_sec'])<required_sec:
                state['exhausted']=True
                raise BudgetExhausted('not enough budget including cleanup reserve')
            if state['first_started_wall'] is None:
                now=self.wall();state.update(first_started_wall=now,origin_mono=self.mono(),
                                              last_wall=now,last_mono=self.mono(),boot_id=self.boot_id)
            state['phase']=phase
            return self._remaining(state)

    def require(self,required_sec,phase='P0_P3'):
        status=self.status()
        if status['exhausted'] or min(status['remaining']['total_experiment_wall_time_sec'],status['remaining'][phase+'_sec'])<required_sec:
            raise BudgetExhausted('budget reserve insufficient')
        return status

    def expire(self):
        with self._locked() as state:state['exhausted']=True
