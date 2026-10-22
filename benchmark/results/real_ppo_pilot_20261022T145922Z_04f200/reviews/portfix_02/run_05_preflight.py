from pathlib import Path
import json,os,subprocess,time,hashlib
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[5];RUN=Path(__file__).resolve().parents[2]
manifest=json.loads((RUN/'commands/05_smoke.json').read_text())
rows=[]
for entry in manifest['commands']:
 argv=entry['validation_argv'];started=time.monotonic()
 result=subprocess.run(argv,cwd=ROOT,env={**os.environ,**entry.get('env',{})},capture_output=True,text=True)
 elapsed=time.monotonic()-started
 label=entry['label'];out=RUN/'reviews/portfix_02'/f'05_dryrun_{label}.stdout.log';err=RUN/'reviews/portfix_02'/f'05_dryrun_{label}.stderr.log'
 out.write_text(result.stdout);err.write_text(result.stderr)
 plan=json.loads(result.stdout) if result.stdout.strip() else None
 rows.append({'label':label,'argv':argv,'exit_code':result.returncode,'elapsed_sec':elapsed,
  'ready_for_live':plan.get('ready_for_live') if plan else None,
  'blocked_reasons':plan.get('blocked_reasons') if plan else None,
  'estimated_max_sec':plan.get('estimated_max_sec') if plan else None,
  'remaining_budget':plan.get('remaining_budget') if plan else None,
  'stdout_log':str(out.relative_to(RUN)),'stdout_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
  'stderr_log':str(err.relative_to(RUN)),'stderr_sha256':hashlib.sha256(err.read_bytes()).hexdigest(),
  'GPU_will_start':plan.get('GPU_will_start') if plan else None})
status=subprocess.run([str(ROOT/'.venv/bin/python'),'-c',
 'from pathlib import Path; import sys; sys.path.insert(0,sys.argv[1]); from rl_scheduler.real_budget import BudgetLedger; import json; print(json.dumps(BudgetLedger(sys.argv[2]).status()))',str(ROOT),str(RUN)],
 cwd=ROOT,capture_output=True,text=True)
assert status.returncode==0,status.stderr
budget=json.loads(status.stdout)
full_fits=budget['remaining']['P0_P3_sec']>=manifest['planned_total_max_sec'] and budget['remaining']['total_experiment_wall_time_sec']>=manifest['planned_total_max_sec']
value={'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'incomplete' if not full_fits else 'passed',
 'scope':'05 command/hash/order dry-run only; no GPU or Ray startup', 'results':rows,
 'planned_total_max_sec':manifest['planned_total_max_sec'],'remaining_budget':budget['remaining'],
 'full_sequence_max_fits_current_budget':full_fits,'GPU_started':False,
 'live_authorized_by_this_preflight':bool(full_fits and rows and rows[0]['ready_for_live'] is True),
 'reason':None if full_fits else 'the frozen 6060-second maximum exceeds current persistent P0_P3 budget'}
out=RUN/'reviews/portfix_02/05_preflight_dryruns.json';out.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':value['status'],'full_sequence_max_fits_current_budget':full_fits,'remaining_budget':budget['remaining'],
 'results':[{'label':r['label'],'exit_code':r['exit_code'],'ready_for_live':r['ready_for_live'],'blocked_reasons':r['blocked_reasons'],'GPU_will_start':r['GPU_will_start']} for r in rows]},ensure_ascii=False,indent=2))
if any(r['exit_code']!=0 for r in rows):raise SystemExit(1)
