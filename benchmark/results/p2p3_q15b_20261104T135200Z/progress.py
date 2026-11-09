from pathlib import Path
import json,time
b=Path(__file__).resolve().parent
s=json.load(open(b/'run_state.json'));print('state',s['status'],'GPU elapsed min',round((s.get('ended_wall',time.time())-s['started_wall'])/60,2))
for p in sorted((b/'runs').glob('*/decisions.jsonl')):
 if (p.parent/'episode.json').exists():
  e=json.load(open(p.parent/'episode.json'));print(p.parent.name,e['status'],'completed',e['measured_request_count'])
 else:
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-100000));ls=f.read().splitlines()
  if ls:
   d=json.loads(ls[-1]);print(p.parent.name,'active completed',d['completed_requests'],'tokens',d['actual_output_tokens'],'last decision age',round((time.monotonic_ns()-d['observation_decision_ns'])/1e9,2))
for name in ('real_collection_report.json','real_update_report.json'):
 if (b/name).exists():
  a=json.load(open(b/name));print(name,a.get('status'),'complete',a.get('complete_cohorts'),'eligible',a.get('total_actor_eligible'),'success',a.get('verified_successful_kv_migrations'))
