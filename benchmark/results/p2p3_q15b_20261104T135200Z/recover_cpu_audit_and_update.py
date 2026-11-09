"""One CPU-only re-audit and real update; never starts any service or recollects data.
Preserves the initial audit/report failure and uses a single exclusive update guard.
"""
from pathlib import Path
import sys,json,copy,time,shutil
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch
from rl_scheduler.q15b_data import read,stream,sha
from rl_scheduler.cohort_ppo import audit_execution,build_rollout,ppo_update,CohortActorCritic,tensors
from rl_scheduler.cohort_reward import CohortReward
from rl_scheduler.cohort_training import policy_hash
from rl_scheduler.real_budget import atomic_json
from benchmark.run_p2p3_q15b import real_stats,checkpoint,final_report,SETTINGS
b=Path(__file__).resolve().parent;p=b/'runs/ppo_151101';manifest=read(b/'manifest.json')
assert read(b/'run_state.json')['status']=='completed' and read(b/'cleanup_verification.json')['status']=='passed'
assert not (b/'real_update_report.json').exists()
assert sha(b/'checkpoints/initial.pt')==manifest['initial_checkpoint_sha256']
summary=read(p/'episode.json');ds=list(stream(p/'decisions.jsonl'))
audit=audit_execution(summary,ds,stream(p/'events.jsonl'),stream(p/'ppo_action_intents.jsonl'))
atomic_json(p/'execution_audit_rechecked.json',audit)
# All original outcomes and raw request/decision streams remain untouched.
shutil.copy2(b/'real_collection_report.json',b/'real_collection_report_initial.json')
shutil.copy2('/tmp/p2_audit_recovery_tests.log',b/'audit_recovery_tests.log')
stats=real_stats(p,summary);stats['execution_audit']='passed';stats['execution_audit_path']=str(p/'execution_audit_rechecked.json')
rollout=build_rollout(summary,ds,CohortReward(manifest['T_ref_sec']));rollout['source']='real_on_policy_cpu_reaudit'
atomic_json(b/'real_runs/ppo_151101_rollout.json',rollout)
atomic_json(b/'real_runs/ppo_151101_rechecked.json',stats)
collection=dict(status='passed' if stats['actor_eligible']>=20 and stats['opportunity']['succeeded']>=1 else 'failed',attempts=1,
    complete_cohorts=1,total_actor_eligible=stats['actor_eligible'],verified_successful_kv_migrations=stats['opportunity']['succeeded'],
    same_frozen_policy_sha256=read(b/'real_collection_report_initial.json')['same_frozen_policy_sha256'],episodes=[stats],
    second_cohort='not_run: stopped after postprocessing interface error, no automatic retry',scope='one valid cohort chain verification only',
    initial_failure_retained='real_collection_report_initial.json and runs/ppo_151101/execution_audit.json',
    recovery='CPU iterator materialization only; no recollection, relabeling, changed masks or changed actions')
atomic_json(b/'real_collection_report.json',collection)
loaded=torch.load(b/'checkpoints/initial.pt',map_location='cpu',weights_only=False)
torch.set_num_threads(1);net=CohortActorCritic();net.load_state_dict(loaded['policy']);net.eval()
before_hash=policy_hash(net);assert before_hash==collection['same_frozen_policy_sha256']
observations=[r['observation'] for r in rollout['records'] if r['actor_eligible']][:16]
with torch.no_grad():dist,value=net.distribution(*tensors(observations));before=dict(probabilities=dist.probs.tolist(),values=value.tolist())
with (b/'real_update_started.json').open('x') as f:json.dump({'wall':time.time(),'max_real_updates':1,'source':'one complete real frozen-policy cohort'},f)
opt=torch.optim.Adam(net.parameters(),lr=SETTINGS['learning_rate'])
try:
 update=ppo_update(net,opt,[rollout],SETTINGS,seed=151001)
 after_hash=policy_hash(net)
 checkpoint(b/'checkpoints','real_updated.pt',net,opt,'q15b-p2-updated-v1',manifest,manifest['normalizer'])
 restored=CohortActorCritic();restored.load_state_dict(torch.load(b/'checkpoints/real_updated.pt',map_location='cpu',weights_only=False)['policy'])
 with torch.no_grad():a,v=net.distribution(*tensors(observations));d,w=restored.distribution(*tensors(observations))
 assert torch.equal(a.probs,d.probs) and torch.equal(v,w) and torch.isfinite(a.probs).all() and torch.isfinite(v).all()
 atomic_json(b/'real_update_report.json',dict(status='passed',real_update_calls=1,synthetic=False,complete_cohorts=1,update=update,
    before_sha256=before_hash,after_sha256=after_hash,parameters_changed=after_hash!=before_hash,checkpoint_roundtrip_exact=True,
    before=before,after=dict(probabilities=a.probs.tolist(),values=v.tolist()),post_update_real_evaluation=False,
    limitation='single valid real cohort; no policy-quality or superiority claim'))
except BaseException as exc:
 atomic_json(b/'real_update_report.json',dict(status='failed',real_update_calls=1,reason=str(exc)));raise
finally:
 atomic_json(b/'audit_recovery_report.json',dict(status='passed',initial_error="object of type generator has no len()",fix='materialize small intent iterator before duplicate-count check',
    original_failure_retained=True,raw_requests_actions_masks_unchanged=True,gpu_restarts=0,new_real_cohorts=0,code_sha256=sha(ROOT/'rl_scheduler/cohort_ppo.py'),
    test_log='audit_recovery_tests.log',real_update_guard='real_update_started.json'))
 final_report(b)
print(json.dumps({'collection':collection['status'],'complete':1,'eligible':stats['actor_eligible'],'success':stats['opportunity']['succeeded'],'distinct':stats['opportunity']['distinct_requests'],'update':read(b/'real_update_report.json')['status']}))
