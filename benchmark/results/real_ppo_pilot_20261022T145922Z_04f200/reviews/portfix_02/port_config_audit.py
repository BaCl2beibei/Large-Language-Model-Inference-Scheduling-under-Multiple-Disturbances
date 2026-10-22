from pathlib import Path
import hashlib,json,subprocess,yaml
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[5]
RUN=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(argv):
 r=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True)
 return {'argv':argv,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
pilot_path=ROOT/'rl_scheduler/configs/real_pilot.yaml';ll_path=ROOT/'rl_scheduler/configs/real_llumnix.yaml'
pilot=yaml.safe_load(pilot_path.read_text());ll=yaml.safe_load(ll_path.read_text())
old_failure=(RUN/'reviews/smoke05/telemetry_server_supervisor.log').read_text()
ss=run(['ss','-ltnp']);gpu=run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])
supervisor=run(['supervisorctl','status','real_ppo_fcda2e237b','real_ppo_guard_fcda2e237b'])
ray_process=run(['bash','-lc',"ps -eo pid,args | rg '[r]ay start|[r]aylet' || true"])
port=pilot['service']['ray_port'];llport=ll['SERVER']['RAY_CLUSTER_PORT']
checks={
 'pilot_and_llumnix_ray_ports_match':port==llport,
 'both_ray_ports_are_20000':port==llport==20000,
 'port_outside_observed_worker_range':not (10002<=port<=19999),
 'prior_failure_log_proves_collision_cause':all(x in old_failure for x in ["worker_ports': '9998 ports from 10002 to 19999'", "gcs_server': 16379"]),
 'port_20000_not_listening':not any(':'+str(port)+' ' in line or ':'+str(port)+'\n' in line for line in ss['stdout'].splitlines()),
 'task_services_not_running':'EXITED' in supervisor['stdout'] and 'STOPPED' in supervisor['stdout'],
 'no_gpu_compute_processes':gpu['exit_code']==0 and not gpu['stdout'].strip(),
 'no_ray_head_or_raylet_process':ray_process['exit_code']==0 and not ray_process['stdout'].strip(),
}
# Normalize port-listener check: ss prints LISTEN rows only for bound sockets.
checks['port_20000_not_listening']=not any(line.split()[3].rsplit(':',1)[-1]==str(port) for line in ss['stdout'].splitlines()[1:] if len(line.split())>3)
value={'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'passed' if all(checks.values()) else 'failed',
 'ray_port':port,'worker_port_range_observed':'10002-19999','pilot_config_sha256':sha(pilot_path),'llumnix_config_sha256':sha(ll_path),
 'checks':checks,'commands':{'ss':ss,'nvidia_smi_compute_apps':gpu,'supervisor_owned_services':supervisor,'ray_process_inventory':ray_process},
 'prior_failure_log':'reviews/smoke05/telemetry_server_supervisor.log','prior_failure_log_sha256':sha(RUN/'reviews/smoke05/telemetry_server_supervisor.log')}
out=RUN/'reviews/portfix_02/port_config_audit.json';out.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':value['status'],'checks':checks,'pilot_sha256':value['pilot_config_sha256'],'llumnix_sha256':value['llumnix_config_sha256'],'ss_exit':ss['exit_code'],'gpu_inventory_exit':gpu['exit_code'],'supervisor_exit':supervisor['exit_code'],'ray_process_inventory_exit':ray_process['exit_code']},ensure_ascii=False,indent=2))
if not all(checks.values()):raise SystemExit(1)
