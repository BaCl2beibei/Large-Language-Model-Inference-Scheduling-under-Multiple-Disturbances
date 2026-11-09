"""Rebuild explicit incomplete evidence after the collector was killed by chat interruption.
No GPU service is started and no missing client terminal is inferred from server output.
"""
import json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from benchmark.run_p0p1_fast import read,rows,environment,analyze
from rl_scheduler.real_budget import atomic_json
from rl_scheduler.real_metrics import CausalEventAdapter
from rl_scheduler.real_runner import server_request_id
import psutil
b=Path(__file__).resolve().parent;name='02_Q15-A_141101_Stay-A';d=b/'runs'/name
trace=read(d/'trace.json');client=rows(d/'client_events.jsonl');ds=rows(d/'decisions.jsonl')
requests={r['request_id']:dict(request_id=r['request_id'],sent_ns=None,scheduled_ns=None,client_first_ns=None,client_terminal_ns=None,
          actual_output_tokens=0,outcome=None,expected_input_tokens=r['input_tokens'],expected_output_tokens=r['output_tokens']) for r in trace}
for e in client:
 r=requests.get(e.get('request_id'))
 if r is None:continue
 kind=e['kind'];at=e['occurred_ns']
 if kind=='planned':r['scheduled_ns']=e['data']['scheduled_ns']
 if kind=='sent':r['sent_ns']=at
 if kind=='client_first':r['client_first_ns']=at
 if kind=='client_output':r['actual_output_tokens']=max(r['actual_output_tokens'],e['data']['cumulative_tokens'])
 if kind=='client_terminal':r.update(client_terminal_ns=at,outcome=e['data'].get('outcome'))
(d/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in requests.values()))
adapter=CausalEventAdapter(b.name,name,{server_request_id(b.name,name,r['request_id']):r['request_id'] for r in trace})
server=[]
for p in sorted((b/'causal_events').rglob('events*.jsonl')):
 for e in rows(p):
  if (e.get('run_id'),e.get('episode_id'))==(b.name,name) or (e.get('server_request_id') or e.get('request_id')) in adapter.request_map:
   server.append(adapter.adapt(e,received_ns=time.monotonic_ns()).__dict__)
(d/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in [*client,*server]))
(d/'migrations.jsonl').write_text('');(d/'ecdf.jsonl').write_text('')
complete=sum(r['outcome']=='success' and r['actual_output_tokens']==1536 for r in requests.values())
issues=['user_interrupted_collector_process','missing_complete_client_cohort','controller_lease_expired_after_collector_exit']
summary=dict(schema='full-completion-episode-v1',run_id=b.name,episode_id=name,policy='Stay',status='aborted_user_interruption',
             cohort_complete=False,planned_request_count=128,measured_request_count=complete,primary_metrics=None,
             diagnostic_partial={'completed_successes':complete,'sent':sum(r['sent_ns'] is not None for r in requests.values()),
                                 'observed_client_tokens':sum(r['actual_output_tokens'] for r in requests.values())},
             issues=issues,control_issues=issues,trace_sha256=read(b/'plan.json')['trace_hashes']['Q15-A_141101'],
             frozen_mapping=ds[0]['instance_mapping'],route_audit_passed=False,route_audit_status='not_completed',
             migration_receipts={},actual_migration_attempts=0,balanced_threshold=read(b/'plan.json')['balanced_threshold'],
             client_terminal_count=complete,requests=list(requests.values()),ecdf=[],
             server_replay_note='post-interruption received_ns is reconstruction time; never used as online observation or performance completion')
atomic_json(d/'episode.json',summary)
atomic_json(d/'execution_audit.json',{'status':'incomplete','issues':issues,'reconciled_migrations':[]})
(d/'event_cursor.json').write_text(json.dumps({'status':'unavailable_collector_terminated','sources_preserved':True})+'\n')
# Idempotent append: preserve completed warmup and append one aborted formal round.
ledger=rows(b/'episode_ledger.jsonl')
if not any(e['episode_id']==name for e in ledger):
 with (b/'episode_ledger.jsonl').open('a') as f:f.write(json.dumps(dict(episode_id=name,candidate='Q15-A',seed=141101,mode='Stay-A',wall_sec=(ds[-1]['monotonic_ns']-ds[0]['monotonic_ns'])/1e9))+'\n')
plan=read(b/'plan.json');state=read(b/'run_state.json');now=time.time()
state.update(status='incomplete_user_interruption',reason='collector killed by accidental chat interruption; no second repair/restart; missing full P0/P1',
             ended_wall=now,elapsed_sec=now-plan['budget_origin_wall'],attempt_elapsed_sec=now-state['started_wall'],formal_rounds=2,
             monitoring_stopped=True,rounds_not_run=['PPO-STAY','Stay-B','Balanced seed1','Balanced seed2','Stay seed2','Q15-B'])
atomic_json(b/'run_state.json',state)
env=read(b/'environment_audit.json');env['after_cleanup']=environment();env['service_stopped']=read(b/'service_stopped.json')
live=[]
for r in read(b/'owned_processes.json')['processes']:
 try:
  p=psutil.Process(r['identity']['pid'])
  if p.create_time()==r['identity']['created']:live.append({'pid':p.pid,'status':p.status()})
 except psutil.NoSuchProcess:pass
cleanup={'owned_identities_still_present':live,'verified_wall':now,'gpu_processes':env['after_cleanup']['gpu_processes'],
         'note':'initial cleanup confirmation warning retained in service_stopped.json; subsequent identity audit finds no residual owned process'}
atomic_json(b/'cleanup_verification.json',cleanup);env['final_cleanup_verification']=cleanup
text=(b/'api_server.log').read_text(errors='replace')
line=next(l for l in text.splitlines() if 'engine_args:' in l)
env['effective_engine_parameters']={'evidence':'api_server.log engine_args','raw':line,'batch_tokens_2112_confirmed':'max_num_batched_tokens=2112' in line,
                                  'model_and_tokenizer_same_15b_confirmed':line.count("Qwen2-1.5B")>=2,'four_capacities':[x['total_blocks'] for x in plan['capacity_snapshot']['instances']]}
atomic_json(b/'environment_audit.json',env)
result=analyze(b)
# Raw source and replay remain distinct; no performance result is manufactured.
print(json.dumps({'status':result['status'],'observed_client_completions':complete,'cleanup_live_owned':live,'elapsed_sec':state['elapsed_sec']}))
