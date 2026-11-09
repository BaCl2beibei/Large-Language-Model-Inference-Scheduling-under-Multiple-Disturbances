#!/usr/bin/env python3
"""Complete the missing 1024/1024 same-seed four-policy group."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import aiohttp
import psutil
ROOT=Path(__file__).resolve().parents[3]
RUN_DIR=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from rl_scheduler.real_env import RealEnvConfig
from rl_scheduler.real_budget import atomic_json
from rl_scheduler.real_bridge import digest
import rl_scheduler.migration_scan_runner as scan_runner
import benchmark.run_migration_opportunity_scan as runner

SCENARIO="i1024_o1024_q3_s192_b4096"
SEED=91000
POLICIES=("Stay","Balanced-L","Balanced-M","Balanced-H")

_original_timeout=aiohttp.ClientTimeout
def _window_aligned_timeout(*args,**kwargs):
    if kwargs.get("total")==360.0:
        kwargs=dict(kwargs)
        kwargs.update(total=None,connect=10.0,sock_connect=10.0,sock_read=None)
    return _original_timeout(*args,**kwargs)
aiohttp.ClientTimeout=_window_aligned_timeout
_original_episode=scan_runner.episode
async def _full_scoring_window(*,episode_id,**kwargs):
    if episode_id!="warmup":
        kwargs["horizons"]=RealEnvConfig(arrival_sec=120.0,drain_sec=480.0)
    return await _original_episode(episode_id=episode_id,**kwargs)
scan_runner.episode=_full_scoring_window

def _schedule(cases,seeds,thresholds):
    matches=[case for case in cases if case["scenario"]==SCENARIO]
    if len(matches)!=1:
        raise RuntimeError(f"expected exactly one frozen case {SCENARIO}; found {len(matches)}")
    case=matches[0]
    return [dict(case=case,seed=SEED,policy=policy,
                 threshold=None if policy=="Stay" else thresholds[SCENARIO][policy])
            for policy in POLICIES]
runner.paired_schedule=_schedule

def _owned_alive(service_dir):
    p=service_dir/"owned_processes.json"
    if not p.exists():
        return []
    rows=json.loads(p.read_text()).get("processes",[])
    alive=[]
    for row in rows:
        ident=row["identity"]
        try:
            proc=psutil.Process(int(ident["pid"]))
            if abs(proc.create_time()-float(ident["created"]))<3:
                alive.append({"pid":ident["pid"],"role":row.get("role")})
        except psutil.NoSuchProcess:
            pass
    return alive

def _gpu_compute_processes():
    p=subprocess.run(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader"],capture_output=True,text=True)
    if p.returncode:
        return ["nvidia-smi error: "+p.stderr.strip()]
    return [line.strip() for line in p.stdout.splitlines() if line.strip()]

def _complete_group():
    ledger=RUN_DIR/"completed_runs.jsonl"
    records=[json.loads(x) for x in ledger.read_text().splitlines() if x.strip()] if ledger.exists() else []
    rows={}
    for row in records:
        if row.get("phase")=="paired" and row.get("case",{}).get("scenario")==SCENARIO and row.get("seed")==SEED:
            rows[row["policy"]]=row
    if set(rows)!=set(POLICIES):
        return False,{"found":sorted(rows),"missing":sorted(set(POLICIES)-set(rows))}
    checks={}
    hashes=set()
    gpu_sets=set()
    for policy,row in rows.items():
        summary_path=RUN_DIR/row["summary"]
        if not summary_path.exists(): return False,{"missing_summary":str(summary_path)}
        summary=json.loads(summary_path.read_text())
        gpu_path=summary_path.parent/"gpu_mapping.json"
        if not gpu_path.exists(): return False,{"missing_gpu_mapping":str(gpu_path)}
        gpu_map=json.loads(gpu_path.read_text())
        checks[policy]={"status":summary.get("status"),"routes":summary.get("routes",{}).get("status"),"submission":summary.get("submission",{}).get("status"),"window_sec":summary.get("window",{}).get("end",0)-summary.get("window",{}).get("start",0)}
        if summary.get("status")!="passed" or summary.get("routes",{}).get("status")!="passed" or summary.get("submission",{}).get("status")!="passed" or abs(checks[policy]["window_sec"]-600)>2.5:
            return False,{"invalid_policy":policy,"checks":checks[policy]}
        hashes.add(summary.get("trace_hash"))
        gpu_sets.add(tuple(sorted(gpu_map.values())))
    detail={"checks":checks,"trace_hashes":sorted(str(x) for x in hashes),"gpu_sets":len(gpu_sets)}
    return len(hashes)==1 and len(gpu_sets)==1 and len(next(iter(gpu_sets)))==4,detail

parser=argparse.ArgumentParser()
parser.add_argument("--max-seconds",type=float,default=86400.0)
args=parser.parse_args()
before={p.name for p in (RUN_DIR/"raw_runs").glob("paired_*") if p.is_dir()}
sys.argv=["run_migration_opportunity_scan.py","paired","--run-dir",str(RUN_DIR),"--max-seconds",str(args.max_seconds)]
print(f"resume_missing_group scenario={SCENARIO} seed={SEED} policies={','.join(POLICIES)} thresholds=stay_frozen measured_window=120+480 client_total_timeout=None",flush=True)
error=None
try:
    runner.main()
except BaseException as exc:
    error=exc
    print(f"runner_exception={type(exc).__name__}:{exc}",flush=True)

ok,group_detail=_complete_group()
if not ok:
    atomic_json(RUN_DIR/"missing_1024_o1024_completion_failure.json",{"runner_exception":None if error is None else type(error).__name__+":"+str(error),"group_detail":group_detail})
    if error is not None: raise error
    raise RuntimeError("paired group did not pass final audits: "+str(group_detail))
new_dirs=[p for p in (RUN_DIR/"raw_runs").glob("paired_*") if p.is_dir() and p.name not in before]
cleanup_records=[]
cleanup_ok=True
for service_dir in new_dirs:
    deadline=time.monotonic()+300
    alive=_owned_alive(service_dir)
    gpu=_gpu_compute_processes()
    while (alive or gpu) and time.monotonic()<deadline:
        time.sleep(5)
        alive=_owned_alive(service_dir)
        gpu=_gpu_compute_processes()
    confs=sorted((service_dir/"services").glob("full_completion_*.conf"))
    svc_name=confs[0].stem if confs else None
    svc=subprocess.run(["supervisorctl","status",svc_name],capture_output=True,text=True) if svc_name else None
    absent=bool(svc and svc.returncode!=0 and "no such process" in (svc.stdout+svc.stderr).lower())
    final={"checked_unix":time.time(),"owned_alive":alive,"gpu_compute_processes":gpu,"supervisor_service":svc_name,"supervisor_status":None if svc is None else {"returncode":svc.returncode,"stdout":svc.stdout.strip(),"stderr":svc.stderr.strip()},"supervisor_service_absent":absent,"cleanup_finally_confirmed":not alive and not gpu and absent}
    atomic_json(service_dir/"service_cleanup_final_recheck.json",final)
    cleanup_records.append(str((service_dir/"service_cleanup_final_recheck.json").relative_to(RUN_DIR)))
    cleanup_ok=cleanup_ok and final["cleanup_finally_confirmed"]

if not cleanup_ok:
    raise RuntimeError("all episode audits passed but owned service cleanup remains unconfirmed; details in service_cleanup_final_recheck.json")
status={"status":"single_seed_missing_group_completed","scenario":SCENARIO,"seed":SEED,"policies":list(POLICIES),"thresholds":json.loads((RUN_DIR/"threshold_selection.json").read_text())["selection"][SCENARIO]["thresholds"],"episodes_passed":True,"same_trace_and_gpu_set":True,"delayed_cleanup_rechecks":cleanup_records,"runner_cleanup_exception_resolved":None if error is None else type(error).__name__+":"+str(error)}
atomic_json(RUN_DIR/"paired_status.json",status)
subprocess.run([sys.executable,str(RUN_DIR/"build_paired_report.py")],check=True)
print(json.dumps({"status":status,"group_audit":group_detail},ensure_ascii=False),flush=True)
