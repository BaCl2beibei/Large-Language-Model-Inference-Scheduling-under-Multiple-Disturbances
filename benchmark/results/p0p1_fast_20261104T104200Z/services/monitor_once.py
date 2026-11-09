"""Read-only, compact progress snapshot. Does not restart or alter any service."""
import json,time
from pathlib import Path
b=Path(__file__).resolve().parents[1]
state=json.load(open(b/'run_state.json'))
print('state',state['status'],'elapsed_min',round((state.get('ended_wall',time.time())-state['started_wall'])/60,2))
for p in sorted((b/'runs').glob('*/decisions.jsonl')):
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-100000));data=f.read().splitlines()
  r=json.loads(data[-1])
 e=p.parent/'episode.json'
 if e.exists():
  summary=json.load(open(e));m=(summary.get('primary_metrics') or {}).get('client_e2e_ms',{})
  print(p.parent.name,summary['status'],'mean_ms',round(m.get('mean',0),1),'cost_ms',round(m.get('mean',0)+m.get('cvar95',0),1))
 else:print(p.parent.name,'active','complete',r['completed_requests'],'tokens',r['actual_output_tokens'],'last_decision_age_sec',round((time.monotonic_ns()-r['monotonic_ns'])/1e9,1))
if (b/'comparison.json').exists():
 a=json.load(open(b/'comparison.json'));print('P0',a['p0']);print('candidates',a['candidates'])
