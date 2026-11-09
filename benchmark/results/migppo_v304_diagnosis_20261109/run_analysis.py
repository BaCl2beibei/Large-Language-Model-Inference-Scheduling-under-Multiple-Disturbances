"""Read-only V304 diagnosis and CPU same-arrival counterchecks; no live RPCs."""
from pathlib import Path
import collections
import copy
import gzip
import json
import random
import statistics
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import torch
from benchmark.ppo_local import episode_metrics,read,sha
from rl_scheduler.cohort_ppo import CohortActorCritic,FrozenController,normalizer_definition,policy_hash
from rl_scheduler.cohort_reward import MigPPOV2Reward
from rl_scheduler.q15b_sim import Q15BCohortSim
from rl_scheduler.portable_io import atomic_json

OUT=Path(__file__).resolve().parent
PILOT=ROOT/'benchmark/results/migppo_v304_real13_pilot_20261109'
EXPORT=ROOT/'benchmark/results/migppo_local_001_best_v304'
FROZEN=ROOT/'benchmark/results/mig_ppo_v2_native_cpu_20261106T113030Z/mig_ppo_v2_cpu'

def rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return [json.loads(l) for l in f]

def behavior(decisions,migrations):
    eligible=[d for d in decisions if d['actor_eligible']]
    moves=[m for m in migrations if m.get('terminal_status')=='succeeded']
    counts=collections.Counter(m['actual_request_id'] for m in moves)
    return dict(decisions=len(decisions),eligible=len(eligible),forced_stay=len(decisions)-len(eligible),
        sampled_migrations=sum(d['sampled_action']>0 for d in eligible),
        voluntary_stay=sum(d['sampled_action']==0 for d in eligible),
        mean_migration_probability=statistics.mean(d['ppo']['migration_probability'] for d in eligible) if decisions[0].get('ppo') else None,
        successful_migrations=len(moves),distinct_requests=len(counts),
        repeat_migrations=sum(v-1 for v in counts.values()),max_migrations_per_request=max(counts.values(),default=0))

def main():
    torch.set_num_threads(2)
    saved=torch.load(EXPORT/'policy.pt',map_location='cpu',weights_only=False)
    policy=CohortActorCritic(64).cpu();policy.load_state_dict(saved['policy']);policy.eval()
    assert policy_hash(policy)==saved['policy_sha256']
    reward=MigPPOV2Reward.from_state_dict(read(EXPORT/'reward_definition.json'))
    envs={e['env_id']:e for e in read(EXPORT/'environments.json')['environments']}
    evaluation=read(EXPORT/'development_evaluation.json')
    summary=read(PILOT/'external_policy_full_pilot_summary.json')
    report=dict(scope='diagnosis only; no refit/training/real experiments',policy_sha256=saved['policy_sha256'],
        real_source=str(PILOT.relative_to(ROOT)),sim_source=str(EXPORT.relative_to(ROOT)),
        real_native_ppo_valid=False,training_coverage_available=False,real={},simulation_same_trace={})
    for eid in ('NATIVE','HOT2-H'):
        report['real'][eid]={}
        trace=read(next((PILOT/'traces/v304_real_eval'/eid).glob('*.json')))
        for name in ('Stay','Balanced','PPO'):
            case=PILOT/'runs'/f'v304_{eid}_{name.lower()}'
            ep=read(case/'episode.json');decs=rows(case/'decisions.jsonl.gz');ms=rows(case/'migrations.jsonl.gz')
            snapshot_stats=[]
            for d in decs:
                states=d['snapshot']['instances']
                snapshot_stats.extend(s['last_step_ms'] for s in states if s['running'] and s.get('last_step_ms'))
            rec=dict(summary['environments'][eid][name],behavior=behavior(decs,ms),
                median_request_tpot_ms=statistics.median(r['tpot_ms'] for r in ep['requests']),
                median_sampled_active_engine_step_ms=statistics.median(snapshot_stats),
                observed_initial_route_counts=dict(collections.Counter(ep['observed_initial_routes'].values())))
            engine_events=[e.get('data',{}) for e in rows(case/'events.jsonl.gz')
                           if e.get('data',{}).get('event')=='llm_engine_step_completed']
            rec['sampled_step_diagnostics']={}
            for key in ('engine_step_wall_ms','executor_wall_ms'):
                vv=[e[key] for e in engine_events if e.get(key) is not None]
                rec['sampled_step_diagnostics'][key]=dict(median=statistics.median(vv),n=len(vv)) if vv else None
            if name=='PPO':
                eligible=[d for d in decs if d['actor_eligible']]
                from rl_scheduler.cohort_ppo import tensors
                with torch.no_grad():dist,_=policy.distribution(*tensors([d['ppo']['observation'] for d in eligible]))
                probs=dist.probs
                rec['frozen_policy_recheck']=dict(eligible_states=len(eligible),
                    greedy_stay_count=int((probs.argmax(-1)==0).sum()),mean_stay_probability=float(probs[:,0].mean()),
                    mean_categorical_entropy=float(dist.entropy().mean()))
            # Respect the original audit: invalid Native PPO has no valid V2 return.
            try:
                end=max(r['client_terminal_ns'] for r in ep['requests'])
                selected=[d for d in decs if d['observation_decision_ns']<end]
                audit=reward.reconstruct(ep,[d['observation_decision_ns'] for d in selected],selected)
                rec['reward_components']={k:sum(w['components'][k] for w in audit['windows']) for k in audit['windows'][0]['components']}
                rec['V2_return']=audit['return']
            except ValueError as exc:rec['reward_unavailable_reason']=str(exc)
            report['real'][eid][name]=rec
            model=read(FROZEN/envs[eid]['model_path'])
            seed=summary['environments'][eid]['PPO']['controller_seed']
            torch.manual_seed(seed)
            sim=Q15BCohortSim(model,copy.deepcopy(trace),seed=0,output_dir=OUT/'same_trace'/eid/name,
                minimal_logging=True,finalize_deadline=True,wall_time_limit_sec=120.)
            try:
                controller=FrozenController(policy,normalizer_definition(),304) if name=='PPO' else None
                s,d,m=sim.run(name,1.,controller=controller,reward_definition=reward,episode_horizon_sec=200.)
                audit=read(sim.output_dir/'reward.json');metrics=episode_metrics(s,d,m,audit)
                metrics['model_sha256']=sha(FROZEN/envs[eid]['model_path'])
                metrics['C_error_vs_measured']=abs(metrics['C_ms']-rec['C_ms'])/rec['C_ms']
                metrics['reward_components']={k:sum(w['components'][k] for w in audit['windows']) for k in audit['windows'][0]['components']}
                report['simulation_same_trace'].setdefault(eid,{})[name]=metrics
                print(eid,name,'sim C',round(metrics['C_ms'],1),'real C',round(rec['C_ms'],1),'moves',metrics['migration_count'],flush=True)
            finally:sim.event_file.close()
        if eid=='NATIVE':
            ep=read(PILOT/'runs/v304_NATIVE_stay/episode.json');fixed=copy.deepcopy(trace)
            ids=[r[0] for r in ep['frozen_mapping']]
            for row in fixed:
                row['initial_instance_index']=ids.index(ep['observed_initial_routes'][row['request_id']]);row['routing_mode']='fixed'
            sim=Q15BCohortSim(read(FROZEN/envs[eid]['model_path']),fixed,seed=0,
                output_dir=OUT/'same_trace/NATIVE/Stay_observed_routes',minimal_logging=True,finalize_deadline=True)
            try:
                s,d,m=sim.run('Stay',1.,reward_definition=reward,episode_horizon_sec=200.)
                report['native_stay_observed_routes_diagnostic']=episode_metrics(s,d,m,read(sim.output_dir/'reward.json'))
            finally:sim.event_file.close()
    report['simulated_behavior']={}
    for eid in envs:
        cases=[c for c in evaluation['cases'] if c['case']['env_id']==eid]
        rr=[r for c in cases for r in c['action_seed_results']]
        report['simulated_behavior'][eid]={k:statistics.mean(r[k] for r in rr) for k in
            ('migration_count','distinct_migrated_requests','repeated_migrations','reverse_migrations',
             'eligible_decisions','forced_stay_decisions','voluntary_stay_decisions','conditional_migration_rate')}
    rng=random.Random(133701);counts=collections.Counter(rng.choice(list(envs)) for _ in range(304*4))
    report['default_sampler_reconstruction']=dict(assumption='unchanged default environment seed 133701; four episodes/update; no missing/custom history',
        observed_training_history=False,conditional_counts=dict(counts),native_expected=304*4/13)
    atomic_json(OUT/'diagnosis.json',report)
    print('Saved',OUT/'diagnosis.json',flush=True)

if __name__=='__main__':main()
