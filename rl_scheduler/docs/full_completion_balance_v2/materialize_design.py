"""CPU-only design generation/validation; no model, traffic, service or lease."""
import hashlib,json
from pathlib import Path
import yaml
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def write(name,value):(HERE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
rows=[]
for rho in [.35,.55,.75]:
 for hot in [.7,.9]:rows.append(dict(family='S1',rho=rho,profile='decode1536',hot_fraction=hot,routing='fixed_seeded_hotspot',role='search'))
for rho in [.55,.75]:
 for frac in [.1,.2]:rows.append(dict(family='S2',rho=rho,profile='bimodal_'+str(frac),long_fraction=frac,long_hot_share=.8,routing='equal_request_counts_correlated_lengths',role='search'))
for rho in [.55,.75]:
 for burst in [2,4]:rows.append(dict(family='S3',rho=rho,profile='decode512',hot_fraction=.8,burst_multiplier=burst,routing='rotating_fixed_hotspot',role='search'))
rows.extend([dict(family='C0',rho=.55,profile='decode1536',routing='uniform_fixed',role='negative_control'),dict(family='C1',rho=.55,profile='decode64',hot_fraction=.9,routing='fixed_seeded_hotspot',role='negative_control')])
for i,row in enumerate(rows,1):row.update(candidate_id=f'{row["family"]}_{i:02d}',qps=None,qps_rule='rho * measured_lambda_ref[profile]',status='proposed_not_run')
write('candidate_matrix.json',dict(schema='balance-full-completion-candidates-v2',candidate_count=len(rows),candidates=rows))
spec=yaml.safe_load((HERE/'experiment_spec.yaml').read_text())
assert len(rows)==16 and len({r['candidate_id'] for r in rows})==16
seeds=[spec['search']['screening']['seeds'],spec['search']['confirmation']['seeds'],spec['validation']['seeds']]
assert len(set(sum(seeds,[])))==sum(map(len,seeds))
assert not set(sum(seeds,[]))&{1213,1319,2101,2102,2103,5101,5102,5103,7101,7102,7103}
for profile in spec['profiles'].values():
 assert profile['input_tokens']+max(profile.get('output_tokens',0),profile.get('long_output_tokens',0))<=2112
for n in [128,2048]:
 assert n%4==0
 for f in [.1,.2]:assert int(int(n*f+.5)*.8+.5)<=n//4
assert spec['completion']['completion_time_limit_sec'] is None
assert spec['completion']['experiment_wall_time_limit_sec'] is None
assert spec['search']['screening']['planned_episodes']==len(rows)*len(seeds[0])*2
assert spec['validation']['planned_episodes']==len(spec['validation']['scenes'])*len(seeds[2])*2
write('design_checks.json',dict(status='passed',scope='design_structure_only',checks=['16 unique candidate IDs','disjoint development/validation seeds, distinct from named historical seeds','token budgets fit current context limit','rounded S2 route quotas feasible','no request or experiment completion time cap','planned episode counts consistent','YAML parsed'],business_implementation_tests=False,live_run=False))
files=[HERE.parent/'full_completion_balance_experiment_v2.md',*[p for p in sorted(HERE.iterdir()) if p.is_file() and p.name!='design_manifest.json']]
write('design_manifest.json',dict(status='design_ready',local_date='2026-10-28',timezone='Asia/Shanghai',files={str(p.relative_to(ROOT)):sha(p) for p in files},historical_source_status='read_and_analyzed_earlier_in_session_then_original_run_and_pointer_disappeared',historical_input_hashes=None,actual_design_command={'argv':['.venv/bin/python',str(Path(__file__).resolve().relative_to(ROOT))],'exit_code':0},GPU_started=False,business_implementation_changed=False,historical_files_deleted_by_this_task=False,performance_superiority_established=False))
print('Design checks passed; 16 proposed candidates; no live experiment executed.')
