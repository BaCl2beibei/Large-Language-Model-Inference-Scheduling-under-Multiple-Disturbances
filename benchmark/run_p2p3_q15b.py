#!/usr/bin/env python3
"""Internal Q15-B P2/P3 implementation. Use run_ppo_sim.py / run_ppo_real.py as entrypoints."""
from __future__ import annotations
import argparse,asyncio,copy,json,hashlib,math,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from rl_scheduler.q15b_data import ROOT,REPLAY,DEVELOPMENT,read,stream,sha,sources,extract,fit_models,episode_origin
from rl_scheduler.real_budget import atomic_json
from rl_scheduler.real_bridge import digest,RealBridge
from rl_scheduler.cohort_ppo import (CohortActorCritic,FrozenController,normalizer_definition,tensors,causal_observation,
                                     ppo_update,build_rollout,audit_execution,action_classification,policy_hash,save_checkpoint)
from rl_scheduler.cohort_reward import MigPPOV2Reward
from rl_scheduler.full_completion import run_episode,make_trace,wait_service_ready,summarize_values
from rl_scheduler.ppo_runtime import PPO_DECISION_INTERVAL_SEC, PPO_MIGRATION_COOLDOWN_SEC
from benchmark.run_full_completion_balance import ManagedFullCompletionService,_warmup_trace
from benchmark.run_p0p1_fast import validate_config,environment,contains_seed

SETTINGS=dict(actor_sample_mode='choice_only',update_epochs=2,minibatch_size=64,clip_epsilon=.2,value_coefficient=.5,
              entropy_coefficient=.01,max_grad_norm=.5,target_kl=.03,learning_rate=.0003,
              gamma_per_sec=.99,lambda_per_sec=.97,decision_interval_sec=PPO_DECISION_INTERVAL_SEC,
              migration_cooldown_sec=PPO_MIGRATION_COOLDOWN_SEC)


def load_v2_manifest(out):
    manifest=read(Path(out)/'manifest.json')
    if manifest.get('schema') != 'q15b-mig-ppo-v2' or manifest.get('normalizer',{}).get('schema') != 'mig-ppo-v2':
        raise ValueError('old PPO state/reward schema; prepare a fresh Mig-PPO V2 run directory')
    reward=MigPPOV2Reward.from_state_dict(manifest['reward'])
    for key in ('actor_sample_mode', 'decision_interval_sec', 'migration_cooldown_sec'):
        if manifest['settings'].get(key) != SETTINGS[key]:
            raise ValueError('PPO sampling/update settings changed; prepare a fresh run directory')
    if (manifest['settings']['gamma_per_sec'] != reward.gamma_per_sec
            or manifest['settings']['lambda_per_sec'] != reward.lambda_per_sec):
        raise ValueError('reward shaping and GAE discount definitions differ')
    return manifest


def checkpoint(out,name,net,opt,version,manifest,normalizer):
    save_checkpoint(out/name,net,opt,next_index=0,version=version,freeze_sha256=digest(manifest),
                    config={**manifest['settings'],'scope':'bounded P2/P3 chain check'},normalizer=normalizer)


def offline_observations(path,T_ref,reward_definition=None):
    """Incremental client-journal projection at saved observation time; future records excluded."""
    from rl_scheduler.real_observation import RealObservationBuilder
    from rl_scheduler.real_opportunity_runner import OpportunityRequestLedger
    from rl_scheduler.real_metrics import Event
    path=Path(path);trace=read(path/'trace.json');e=read(path/'episode.json');origin=episode_origin(path,trace)
    it=iter(stream(path/'client_events.jsonl'));pending=next(it,None)
    ledger=OpportunityRequestLedger(e['run_id'],e['episode_id'],start_ns=origin,arrival_end_ns=origin+round(max(r['scheduled_arrival_sec'] for r in trace)*1e9),
        score_deadline_ns=None,full_completion=True,expected_sources=[pending['source']],expected_request_ids=[r['request_id'] for r in trace],sequence_baselines={})
    thresholds=({
        'ttft_slo_ms': reward_definition.ttft_ref_sec*1000,
        'tpot_slo_ms': reward_definition.token_gap_ref_sec*1000} if reward_definition else {})
    historical_builder=RealObservationBuilder(e['frozen_mapping'], **thresholds)
    builder=RealObservationBuilder(e['frozen_mapping'], cooldown_sec=PPO_MIGRATION_COOLDOWN_SEC, **thresholds)
    for d in stream(path/'decisions.jsonl'):
        at=d.get('observation_decision_ns')
        if at is None:continue # never reconstruct unknown historical timestamp as exact
        while pending and pending['received_ns']<=at:
            ledger.ingest([Event(**pending)]);pending=next(it,None)
        kwargs=dict(now_ns=at,ledger=ledger,episode_start_ns=origin,arrival_end_ns=ledger.arrival_end_ns,deadline_ns=origin+3_600_000_000_000)
        historical=historical_builder.build(d['snapshot'],**kwargs)
        mismatch=historical['action_mask']!=d['action_mask']
        if mismatch:raise ValueError('offline raw mask reconstruction differs from saved mask')
        # Historical behavior is audited above; the new mask is an offline
        # exploration diagnostic, never relabeled historical on-policy data.
        raw=builder.build(d['snapshot'],**kwargs)
        yield raw,origin


def prepare(out, *, reward_config, trace_seeds=(161101,161102), policy_seed=161001):
    import yaml
    from tokenizers import Tokenizer
    reward=MigPPOV2Reward.from_state_dict(reward_config)
    cpu_start=time.time();out.mkdir(parents=True,exist_ok=False)
    for sub in ('calibration','sim_validation','real_runs','runtime','services','checkpoints','traces'):(out/sub).mkdir()
    cfg=yaml.safe_load((REPLAY/'runtime_config.yaml').read_text());srv=yaml.safe_load((REPLAY/'real_llumnix.yaml').read_text())
    cfg['service']['ray_temp_dir']='/tmp/p23_'+hashlib.sha256(str(out).encode()).hexdigest()[:10]
    cfg['service']['server_argv'][cfg['service']['server_argv'].index('--config-file')+1]=str(out/'real_llumnix.yaml')
    (out/'runtime_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));(out/'real_llumnix.yaml').write_text(yaml.safe_dump(srv,sort_keys=False));validate_config(cfg,srv)
    if len(trace_seeds) != 2 or len(set([*trace_seeds,policy_seed])) != 3:
        raise ValueError('two distinct trace seeds and a separate policy seed required')
    seedset=set([*trace_seeds,policy_seed])
    for p in (ROOT/'benchmark/results').rglob('*'):
        if not p.is_file() or out in p.parents:continue
        if (p.name in ('manifest.json','plan.json','run_manifest.json') or 'ledger' in p.name) and p.suffix in ('.json','.jsonl'):
            for row in stream(p) if p.suffix=='.jsonl' else [read(p)]:
                if contains_seed(row,seedset):raise ValueError('new seed conflict: '+str(p))
    reqs=list(stream(REPLAY/'runs/02_Q15-B_141104_Stay/requests.jsonl'))
    T_ref=math.fsum((r['client_terminal_ns']-r['sent_ns'])/1e9 for r in reqs)/128
    manifest=dict(schema='q15b-mig-ppo-v2',created_utc=datetime.now(timezone.utc).isoformat(),cpu_started_wall=cpu_start,
        accepted_decision='rl_scheduler/docs/research_implementation_plan_v2.md opening Q15-B decision',model=cfg['service']['model_path'],
        sources=sources(),T_ref_sec=T_ref,T_ref_source=str(REPLAY/'runs/02_Q15-B_141104_Stay/requests.jsonl'),
        units=dict(latency='seconds in reward, milliseconds in metrics',blocks='native KV blocks',clock='CLOCK_MONOTONIC nanoseconds'),
        trace_seeds=list(trace_seeds),policy_seed=policy_seed,policy_version='q15b-mig-ppo-v2-init',normalizer=normalizer_definition(),settings=dict(SETTINGS,gamma_per_sec=reward.gamma_per_sec,lambda_per_sec=reward.lambda_per_sec),
        reward=reward.state_dict(),episode_horizon_sec=3600.,
        budget=dict(cpu_soft_sec=5400,max_model_revisions=2,max_real_attempts=2,max_real_updates=1,gpu_target_sec=1800,gpu_hard_sec=2100,startup_sec=480,cleanup_reserve_sec=480),
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [REPLAY/'plan.json',REPLAY/'runtime_config.yaml',REPLAY/'comparison.json',ROOT/'rl_scheduler/docs/p2p3_q15b_execution_prompt.md']},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),initial_diff_summary=subprocess.check_output(['git','diff','--stat'],text=True),
        initial_bias_rule='retain log(12) unless mean conditional migration probability <.01; only then explicit bias bisection target .45 within preregistered .3-.6; no real-time forced actions',
        p3_thresholds=dict(latency_error=.2,outstanding_curve_error=.2,migration_count_error_fraction=.2,migration_count_error_minimum=5,migration_median_error=.2,migration_p95_error=.3,gain_percentage_point_error=.1),
        no_extra_baselines=True)
    atomic_json(out/'manifest.json',manifest)
    atomic_json(out/'data_split.json',{'fit_group':{'trace_seed':141103,'candidate':'Q15-B','episodes':[x for x in sources() if x['split']=='fit']},
        'development_validation_group':{'trace_seed':141104,'all_attempts_same_group':True,'formal_pair_same_attempt_only':str(REPLAY)},
        'negative_diagnostic':'Q15-A seed141103 is not a training candidate','incomplete':'fault analysis only, excluded from fit labels and on-policy update',
        'new_real_seeds':list(trace_seeds),'selection_bias':'all historical data are development data, not blind test','no_per_request_or_per_second_random_split':True})
    (out/'protocol.md').write_text('''# Frozen P2/P3 protocol\n\nUse accepted Qwen2-1.5B Q15-B only; no candidate search. Historical 141103 fit; all 141104 attempts one development validation group. Full replay pair only. Mig-PPO V2: time/queue/hotspot/gated KV-balance reward; parameters frozen in manifest.reward. Real-time gamma and lambda are frozen per second. T_ref is retained only for historical latency audits. All decisions contribute time-discounted GAE; actor normalization/loss/entropy/KL/clip only eligible, critic all. Preserve voluntary Stay. Default actor mode is choice_only. PPO decision interval=1.5s; migration cooldown=0s in observation, simulator and Manager. Forced Stay remains in critic and GAE only.\n\nP3 hard checks: 128 unique complete requests; causal ownership; valid KV; identical real actor/normalizer/action/reward interface; no future state correction. Validation mean/P95/cost errors <=20%, outstanding-curve normalized absolute area error <=20%, migration count absolute error <=max(5,20% observed), transfer median error <=20%, P95 <=30% where available, positive mean/cost gain directions and gain differences <=10 percentage points. Q15-A negative sign diagnostic reported separately. No hardcoded gain multiplier. Zero to two documented revisions only. Data used for model selection remain development validation. Replay actions and closed-loop controls explicitly separate.\n\nReal sampling only after CPU checks pass and sim model/initial checkpoint are frozen: max2 complete128-request attempts, same frozen policy, zero intervening update, max1 real PPO update. Incomplete/measurement-only/mixed-policy data excluded. At least1 complete cohort,20 eligible decisions,1 verified successful KV migration. No Stay/Balanced or post-update real evaluation. GPU target30min,hard35min including cleanup; before each formal round reserve estimated full round +480s; startup<=480s; no progress120s. Independent supervisor watchdog owns only this batch.\n\nInitial exploration: retain existing bias unless conditional mean migration probability <1%; if necessary explicitly adjust only new initial policy toward mean .45 (.3-.6). Otherwise retain original bias even if below .3; not a profit predictor.\n''')
    data,moves=extract(out,T_ref);model=fit_models(out,data,moves)
    return finish_prepare(out)


def finish_prepare(out):
    import yaml
    from tokenizers import Tokenizer
    manifest=load_v2_manifest(out);cpu_start=manifest['cpu_started_wall'];T_ref=manifest['T_ref_sec']
    cfg=yaml.safe_load((out/'runtime_config.yaml').read_text());model=read(out/'calibration/model.json')
    reward=MigPPOV2Reward.from_state_dict(manifest['reward'])
    torch.set_num_threads(1);torch.manual_seed(manifest['policy_seed']);net=CohortActorCritic().cpu().eval();normalizer=manifest['normalizer']
    raw_samples=[];observations=[];prob=[];counts=0
    with (out/'diagnostic_raw.jsonl').open('w') as f:
        for spec in sources():
            if spec['split'] not in ('fit','development_validation') or spec['policy']!='Stay':continue
            for raw,origin in offline_observations(spec['path'],T_ref,reward):
                obs=causal_observation(raw,origin,normalizer)
                if any(obs['action_mask'][1:]):
                    with torch.no_grad():dist,_=net.distribution(*tensors([obs]))
                    prob.append(float(1-dist.probs[0,0]));counts+=1;observations.append(obs)
                    if len(raw_samples)<32:
                        f.write(json.dumps({'raw':raw,'origin':origin})+'\n');raw_samples.append((raw,origin))
    before=float(net.stay.bias.item());original=float(np_mean(prob))
    if not prob:raise RuntimeError('no historical choice opportunities; cannot fix with bias')
    if original<.01:
        lo,hi=-12.,12.
        for _ in range(30):
            mid=(lo+hi)/2
            with torch.no_grad():
                net.stay.bias.fill_(mid);dist,_=net.distribution(*tensors(observations));p=float((1-dist.probs[:,0]).mean())
            if p>.45:lo=mid
            else:hi=mid
        with torch.no_grad():net.stay.bias.fill_((lo+hi)/2)
    with torch.no_grad():dist,value=net.distribution(*tensors(observations));mean=float((1-dist.probs[:,0]).mean())
    manifest['initial_exploration']=dict(original_stay_bias=before,frozen_stay_bias=float(net.stay.bias.item()),original_conditional_move_probability=original,
        frozen_conditional_move_probability=mean,eligible_observations=counts,changed=original<.01,scope='offline distribution diagnostic only; historical records never PPO samples')
    tokenizer=Tokenizer.from_file(str(Path(cfg['service']['model_path'])/'tokenizer.json'))
    manifest['trace_hashes']={}
    for seed in manifest['trace_seeds']:
        trace=make_trace(candidate={'family':'S1','profile':'decode1536','rho':1.,'hot_fraction':.9},n=128,seed=seed,lambda_ref=.6,tokenizer=tokenizer,episode_id_prefix=f'p2_{seed}_')
        assert all(len(tokenizer.encode(r['prompt']).ids)==128 for r in trace)
        atomic_json(out/'traces'/f'{seed}.json',trace);manifest['trace_hashes'][str(seed)]=digest(trace)
    atomic_json(out/'manifest.json',manifest)
    opt=torch.optim.Adam(net.parameters(),lr=SETTINGS['learning_rate']);checkpoint(out/'checkpoints','initial.pt',net,opt,manifest['policy_version'],manifest,normalizer)
    atomic_json(out/'initial_exploration.json',manifest['initial_exploration'])
    simulation_validation(out,model,net,manifest)
    tests=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','pytest','-q','rl_scheduler/tests/test_q15b_choice_ppo.py','rl_scheduler/tests/test_cohort_ppo.py',
        'rl_scheduler/tests/test_p0p1_fast.py','rl_scheduler/tests/test_env.py',
        'rl_scheduler/tests/test_mig_ppo_v2.py','rl_scheduler/tests/test_cohort_ppo_http.py'],cwd=ROOT,capture_output=True,text=True)
    (out/'cpu_validation.log').write_text(tests.stdout+tests.stderr)
    report=dict(p2_cpu_contract='passed' if tests.returncode==0 else 'failed',tests_exit_code=tests.returncode,
                historical_reward_identity=read(out/'historical_reward_identity.json'),synthetic_only=True,real_update_not_yet_run=True)
    atomic_json(out/'ppo_contract_report.json',report)
    manifest['frozen_before_real_wall']=time.time();manifest['cpu_elapsed_sec']=time.time()-cpu_start
    manifest['code_hashes']={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'benchmark/run_ppo_sim.py',ROOT/'benchmark/run_ppo_real.py',*sorted((ROOT/'rl_scheduler').glob('*.py')),ROOT/'benchmark/run_full_completion_balance.py',ROOT/'benchmark/run_p0p1_fast.py']}
    manifest['frozen_model_sha256']=sha(out/'calibration/model.json');manifest['initial_checkpoint_sha256']=sha(out/'checkpoints/initial.pt')
    manifest['runtime_config_sha256']=sha(out/'runtime_config.yaml');atomic_json(out/'manifest.json',manifest)
    final_report(out)
    if tests.returncode:raise RuntimeError('P2 CPU contract tests failed; no real launch')
    return manifest


def np_mean(values):return math.fsum(values)/len(values)


def curve_error(real_requests,sim_requests):
    origin=min(r['scheduled_ns'] for r in real_requests)
    real=[((r['sent_ns']-origin)/1e9,(r['client_terminal_ns']-origin)/1e9) for r in real_requests]
    sim=[(r['sent_ns']/1e9,r['client_terminal_ns']/1e9) for r in sim_requests]
    # Sim starts at trace time zero; real origin is first scheduled arrival, restore this offset below in caller.
    events=[]
    for a,b in real:events.extend([(a,1,0),(b,-1,0)])
    for a,b in sim:events.extend([(a,0,1),(b,0,-1)])
    events.sort();nr=ns=0;last=0.;error=area=0.
    for t,dr,ds in events:error+=abs(nr-ns)*(t-last);area+=nr*(t-last);nr+=dr;ns+=ds;last=t
    return error/area if area else None


def simulation_validation(out,model,net,manifest):
    from rl_scheduler.q15b_sim import Q15BCohortSim
    out=Path(out);T_ref=manifest['T_ref_sec'];reward=MigPPOV2Reward.from_state_dict(manifest['reward']);results=[];sim_by={};real_by={}
    specs=[s for s in sources() if s['split']=='development_validation']
    # Also diagnose the negative case without changing fitting candidate or data split.
    specs += [s for s in sources() if s['split']=='negative_diagnostic' and ('Stay-A' in s['episode_id'] or s['policy']=='Balanced')]
    for spec in specs:
        path=Path(spec['path']);trace=read(path/'trace.json');name=spec['candidate']+'_'+spec['policy'];real=read(path/'episode.json')
        env=Q15BCohortSim(model,trace,spec['seed'],out/'sim_validation'/('closed_'+name));summary,ds,migs=env.run(spec['policy'],T_ref,reward_definition=reward,episode_horizon_sec=manifest['episode_horizon_sec'])
        actual=real['primary_metrics']['client_e2e_ms'];pred=summary['primary_metrics']['client_e2e_ms']
        errors={k:abs(pred[k]-actual[k])/actual[k] for k in ('mean','p95')};errors['cost']=abs(pred['mean']+pred['cvar95']-actual['mean']-actual['cvar95'])/(actual['mean']+actual['cvar95'])
        # Align both traces to their first scheduled arrival without changing lifetimes.
        offset=min(r['scheduled_arrival_sec'] for r in trace)
        shifted=[dict(r,sent_ns=r['sent_ns']-round(offset*1e9),client_terminal_ns=r['client_terminal_ns']-round(offset*1e9)) for r in summary['requests']]
        area_error=curve_error(real['requests'],shifted)
        observed=len({r['migration_id'] for r in real['migration_receipts'].values() if r.get('terminal_status')=='succeeded'})
        succeeded=sum(m['status']=='success' for m in migs);count_error=abs(succeeded-observed)
        row=dict(name=name,errors=errors,outstanding_curve_error=area_error,actual_migrations=observed,sim_migrations=succeeded,
            migration_count_error=count_error,hard_checks='passed',mode='closed_loop',simulated=pred,observed=actual,
            numerical_passed=all(v<=.2 for v in errors.values()) and area_error<=.2 and count_error<=max(5,observed*.2))
        results.append(row);sim_by[name]=pred;real_by[name]=actual
    gains={}
    for candidate in ('Q15-B','Q15-A'):
        sm=sim_by[candidate+'_Stay'];bm=sim_by[candidate+'_Balanced'];sr=real_by[candidate+'_Stay'];br=real_by[candidate+'_Balanced']
        sg={k:((sm['mean']+sm['cvar95'])-(bm['mean']+bm['cvar95']))/(sm['mean']+sm['cvar95']) if k=='cost' else (sm[k]-bm[k])/sm[k] for k in ('mean','cost')}
        rg={k:((sr['mean']+sr['cvar95'])-(br['mean']+br['cvar95']))/(sr['mean']+sr['cvar95']) if k=='cost' else (sr[k]-br[k])/sr[k] for k in ('mean','cost')}
        gains[candidate]=dict(sim=sg,real=rg,difference={k:abs(sg[k]-rg[k]) for k in sg},sign_match={k:sg[k]*rg[k]>0 for k in sg})
    fit=read(out/'calibration/fit_report.json')
    passed=all(r['numerical_passed'] for r in results if r['name'].startswith('Q15-B')) and all(v>0 for v in gains['Q15-B']['sim'].values()) and all(v<=.1 for v in gains['Q15-B']['difference'].values()) and all(v['passed'] for v in fit['migration_errors'].values())
    atomic_json(out/'sim_validation/closed_loop_report.json',dict(status='passed' if passed else 'failed',results=results,gains=gains,model_revisions=0,
        scope='Q15-B only; Q15-A negative sign diagnostic separately reported',thresholds_frozen=True))
    # Given-action replay is not closed loop; logged source/target translated to current simulated instance mapping.
    spec=next(s for s in specs if s['candidate']=='Q15-B' and s['policy']=='Balanced');path=Path(spec['path']);real=read(path/'episode.json');trace=read(path/'trace.json');origin=episode_origin(path,trace)
    mapping={v[0]:f'instance_{i}' for i,v in enumerate(real['frozen_mapping'])};actions=[]
    for d in stream(path/'decisions.jsonl'):
        receipt=real['migration_receipts'].get(d['decision_id'])
        if receipt and receipt.get('migration_start_ns'):
            logical=next((r['request_id'] for r in trace if receipt.get('actual_request_id')==__import__('rl_scheduler.real_runner',fromlist=['server_request_id']).server_request_id(real['run_id'],real['episode_id'],r['request_id'])),None)
            if logical:actions.append(dict(migration_id=receipt['migration_id'],migration_time_sec=(receipt['migration_start_ns']-origin)/1e9,
                source_instance=mapping[d['src_id']],target_instance=mapping[d['dst_id']],request_id=logical,blocks=receipt.get('copied_blocks'),
                cost_ms=(receipt['terminal_ns']-receipt['migration_start_ns'])/1e6,status='FINISHED' if receipt['terminal_status']=='succeeded' else receipt['terminal_status']))
    env=Q15BCohortSim(model,trace,141104,out/'sim_validation/given_action_replay');env.run('Stay',T_ref,replay_actions=actions,reward_definition=reward,episode_horizon_sec=manifest['episode_horizon_sec'])
    atomic_json(out/'sim_validation/replay_report.json',dict(mode='given_logged_actions_not_closed_loop',attempts=len(actions),results=env.migration_replay_results))
    # Same 13-action CPU policy and exact reward; synthetic simulation only, separate from real update budget.
    env=Q15BCohortSim(model,trace,141104,out/'sim_validation/ppo_chain');policy=copy.deepcopy(net);controller=FrozenController(policy,manifest['normalizer'],manifest['policy_version'])
    summary,ds,_=env.run('PPO',T_ref,controller=controller,reward_definition=reward,episode_horizon_sec=manifest['episode_horizon_sec']);rollout=build_rollout(summary,ds,reward)
    opt=torch.optim.Adam(policy.parameters(),lr=SETTINGS['learning_rate']);update=ppo_update(policy,opt,[rollout],SETTINGS,seed=manifest['policy_seed'])
    checkpoint(out/'checkpoints','simulation_only.pt',policy,opt,'simulation-only-updated',manifest,manifest['normalizer'])
    restored=CohortActorCritic();restored.load_state_dict(torch.load(out/'checkpoints/simulation_only.pt',map_location='cpu',weights_only=False)['policy'])
    obs=rollout['records'][0]['observation']
    with torch.no_grad():a,v=policy.distribution(*tensors([obs]));b,w=restored.distribution(*tensors([obs]))
    assert torch.equal(a.probs,b.probs) and torch.equal(v,w)
    atomic_json(out/'sim_validation/ppo_contract.json',dict(status='passed',synthetic_only=True,not_real_on_policy_update=True,checkpoint_roundtrip_exact=True,update=update))


def final_report(out):
    out=Path(out);cpu=read(out/'ppo_contract_report.json') if (out/'ppo_contract_report.json').exists() else {}
    sim=read(out/'sim_validation/closed_loop_report.json') if (out/'sim_validation/closed_loop_report.json').exists() else {}
    real=read(out/'real_collection_report.json') if (out/'real_collection_report.json').exists() else {}
    upd=read(out/'real_update_report.json') if (out/'real_update_report.json').exists() else {}
    states=dict(p2_cpu_contract=cpu.get('p2_cpu_contract','not_run'),p2_real_collection=real.get('status','not_run'),p2_update=upd.get('status','not_run'),
        p3_data_model='passed' if (out/'calibration/fit_report.json').exists() else 'not_run',p3_closed_loop=sim.get('status','not_run'))
    p2=all(states[k]=='passed' for k in ('p2_cpu_contract','p2_real_collection','p2_update'));p3=states['p3_data_model']=='passed' and states['p3_closed_loop']=='passed'
    states['ready_for_real_training']='passed' if p2 else 'not_run' if not real else 'failed';states['ready_for_sim_pretraining']='passed' if p3 else 'failed' if sim else 'not_run'
    atomic_json(out/'status_summary.json',dict(P2='passed' if p2 else 'not_run' if not real else 'failed',P3='passed' if p3 else sim.get('status','not_run'),items=states))
    lines=['# Q15-B P2/P3 首轮交付','','| 项目 | 状态 | 证据 |','|---|---|---|']
    evidences=['ppo_contract_report.json','real_collection_report.json','real_update_report.json','calibration/fit_report.json','sim_validation/closed_loop_report.json','status_summary.json','status_summary.json']
    lines += [f'| {k} | {v} | {p} |' for (k,v),p in zip(states.items(),evidences)]
    lines += ['', '固定 Qwen2-1.5B / 已接受 Q15-B；历史数据仅用于开发校准和分布诊断。合成 PPO 更新与真实更新严格区分。',
              'P3 未通过时不宣传模拟预训练可用；本轮不要求 PPO 优于 Balanced，也没有额外真实评测。',
              '模型采用快照累计输出窗口拟合有效 token cadence，沿用原离散事件推进。迁移总耗时可观测，单请求服务暂停无法分离，使用全传输时长保守上界而非伪造测量。',
              '全部验证数值及 Q15-A 符号诊断见 sim_validation/closed_loop_report.json；没有写死迁移收益加成，未进行无限修正。',
              f'真实采样：`{json.dumps(real,ensure_ascii=False)}`',f'真实更新：`{json.dumps(upd,ensure_ascii=False)}`']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    (out/'next_step.md').write_text('先读取 status_summary.json、report.md 和 sim_validation/closed_loop_report.json。若 P3 失败，优先检查有效 token cadence 是否区分迁移数据面占用与批次服务干扰；不要用迁移收益乘数凑通过。P2 若通过可另行申请下一轮小预算真实训练，本轮不自动扩大。\n')
    return states


def stream_list(p):return list(stream(p)) if Path(p).exists() else []


def real_stats(path,summary):
    from benchmark.run_p0p1_fast import opportunity_stats
    ds=stream_list(path/'decisions.jsonl');end=max((r['client_terminal_ns'] for r in summary.get('requests',[]) if r.get('client_terminal_ns')),default=0)
    chosen=[d for d in ds if d.get('ppo') and d['ppo']['decision_ns']<end]
    classes={name:sum(d['ppo']['action_class']==name for d in chosen) for name in ('forced_stay','voluntary_stay','migration_attempt')}
    eligible=[d for d in chosen if d['ppo']['actor_eligible']]
    if summary['status']=='passed':
        o=opportunity_stats(ds,summary['migration_receipts'],summary['requests'],n=128,threshold=1070.625)
    else:o={}
    probs=[d['ppo']['migration_probability'] for d in eligible]
    return dict(episode_id=summary['episode_id'],status=summary['status'],completed=summary.get('measured_request_count'),classes=classes,
        actor_eligible=len(eligible),conditional_migration_probability=np_mean(probs) if probs else None,
        conditional_migration_selection=classes['migration_attempt']/len(eligible) if eligible else None,opportunity=o,
        send_lag_ms=(summary.get('primary_metrics') or {}).get('send_lag_ms'),
        cpu_forward_ms=summarize_values([d['ppo']['cpu_forward_ms'] for d in chosen]),cpu_conversion_ms=summarize_values([d['ppo']['cpu_conversion_ms'] for d in chosen]),
        control_work_ms=summarize_values([d['control_work_ms'] for d in chosen]),
        rejection_reasons=dict(__import__('collections').Counter(r.get('rejection_reason') for r in summary.get('migration_receipts',{}).values() if r.get('terminal_status')=='rejected')))


async def real_run(out):
    import aiohttp,yaml
    from tokenizers import Tokenizer
    from rl_scheduler.migration_scan_runner import gpu_mapping
    out=Path(out);manifest=load_v2_manifest(out)
    if (out/'gpu_budget.json').exists():raise RuntimeError('attempt directory already started; no implicit resume/retry')
    if read(out/'ppo_contract_report.json')['p2_cpu_contract']!='passed':raise RuntimeError('CPU contract not passed')
    for rel,h in manifest['code_hashes'].items():
        if sha(ROOT/rel)!=h:raise RuntimeError('code changed after freeze: '+rel)
    if sha(out/'checkpoints/initial.pt')!=manifest['initial_checkpoint_sha256'] or sha(out/'calibration/model.json')!=manifest['frozen_model_sha256']:
        raise RuntimeError('frozen model or checkpoint changed')
    cfg=yaml.safe_load((out/'runtime_config.yaml').read_text());srv=yaml.safe_load((out/'real_llumnix.yaml').read_text());validate_config(cfg,srv)
    if sha(out/'runtime_config.yaml')!=manifest['runtime_config_sha256']:raise RuntimeError('runtime configuration changed')
    torch.set_num_threads(1);loaded=torch.load(out/'checkpoints/initial.pt',map_location='cpu',weights_only=False)
    net=CohortActorCritic();net.load_state_dict(loaded['policy']);net.eval();initial_hash=policy_hash(net)
    controller=FrozenController(net,manifest['normalizer'],manifest['policy_version']);reward=MigPPOV2Reward.from_state_dict(manifest['reward'])
    service=ManagedFullCompletionService(out);started=time.monotonic();wall=time.time()
    budget={'started_wall':wall,'hard_deadline_wall':wall+2100,'stop_service_by_wall':wall+2040,'max_attempts':2,'max_updates':1}
    atomic_json(out/'gpu_budget.json',budget);state={'status':'running','started_wall':wall};atomic_json(out/'run_state.json',state)
    env={'before':environment()};atomic_json(out/'environment_audit.json',env)
    stop=asyncio.Event();loop=asyncio.get_running_loop()
    for sig in (signal.SIGTERM,signal.SIGINT):loop.add_signal_handler(sig,stop.set)
    async def signal_stop():
        await stop.wait();await asyncio.to_thread(service.stop)
    stop_task=asyncio.create_task(signal_stop());rollouts=[];statistics=[];attempts=0;durations=[];mapping=None
    try:
        await asyncio.to_thread(service.start)
        await asyncio.wait_for(wait_service_ready(cfg,service.is_running),timeout=max(.1,480-(time.monotonic()-started)))
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None,connect=5,sock_connect=5),connector=aiohttp.TCPConnector(keepalive_timeout=3.)) as session:
            kwargs=dict(cfg=cfg,run_id=out.name,run_dir=out,control_session=session,service_probe=service.is_running,
                        stop_owned_service=service.stop,no_progress_timeout_sec=120.,hard_stop_monotonic=started+2040,
                        episode_horizon_sec=manifest['episode_horizon_sec'],
                        ttft_slo_ms=reward.ttft_ref_sec*1000,tpot_slo_ms=reward.token_gap_ref_sec*1000)
            tokenizer=Tokenizer.from_file(str(Path(cfg['service']['model_path'])/'tokenizer.json'))
            warm=await run_episode(episode_id='warmup',policy='Stay',trace=_warmup_trace(tokenizer),**kwargs)
            if warm['status']!='passed':raise RuntimeError('warmup audit failure')
            mapping=warm['frozen_mapping'];bridge=RealBridge(cfg['service']['base_url'],session);snap=await bridge.snapshot()
            if any(s['total_blocks']!=8565 for s in snap['instances']):raise RuntimeError('real block capacity differs from accepted environment; stop instead of guessing')
            env['capacity_snapshot']=snap;env['worker_flush']=await bridge.flush();env['instance_gpu_uuid']=gpu_mapping(env['worker_flush'])
            env['after_warmup']=environment();atomic_json(out/'environment_audit.json',env)
            for seed in manifest['trace_seeds']:
                remaining=1800-(time.monotonic()-started);estimate=max(durations,default=480.)+30.
                if remaining<estimate+480:
                    state['collection_stop']='insufficient full-round estimate plus eight-minute cleanup reserve';break
                if stop.is_set():raise RuntimeError('user stop')
                attempts+=1;name=f'ppo_{seed}';trace=read(out/'traces'/f'{seed}.json')
                if digest(trace)!=manifest['trace_hashes'][str(seed)]:raise RuntimeError('trace changed')
                with (out/'real_attempts.jsonl').open('a') as f:
                    f.write(json.dumps(dict(attempt=attempts,episode_id=name,seed=seed,started_wall=time.time(),policy_sha256=initial_hash))+'\n');f.flush();os.fsync(f.fileno())
                print(json.dumps({'event':'real_start','episode':name,'elapsed_sec':time.monotonic()-started}),flush=True)
                torch.manual_seed(seed);before=time.monotonic()
                e=await run_episode(episode_id=name,policy='PPO',trace=trace,frozen_mapping=mapping,decision_controller=controller,
                                    controller_version=manifest['policy_version'],balanced_threshold=1070.625,
                                    allow_request_failures=True,decision_interval_sec=manifest['settings']['decision_interval_sec'],**kwargs)
                durations.append(time.monotonic()-before);path=out/'runs'/name
                summary=read(path/'episode.json');ds=stream_list(path/'decisions.jsonl')
                try:
                    audit=audit_execution(summary,ds,stream(path/'events.jsonl'),stream(path/'ppo_action_intents.jsonl'))
                    if policy_hash(net)!=initial_hash:raise ValueError('policy changed between collections')
                except Exception as exc:audit={'status':'failed','error':str(exc)}
                atomic_json(path/'execution_audit.json',audit)
                (path/'migrations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in audit.get('reconciled_migrations',[])))
                stats=real_stats(path,summary);stats['execution_audit']=audit['status'];statistics.append(stats)
                atomic_json(out/'real_runs'/f'{name}.json',dict(raw_directory=str(path),statistics=stats,initial_checkpoint='../checkpoints/initial.pt'))
                print(json.dumps({'event':'real_end','episode':name,'status':e['status'],'audit':audit['status'],'eligible':stats['actor_eligible'],'duration_sec':durations[-1]}),flush=True)
                if e['status']!='passed' or audit['status']!='passed':
                    state['collection_stop']='audit failure; failed attempt consumed, no automatic retry';break
                rollout=build_rollout(summary,ds,reward);rollout['source']='real_on_policy';rollouts.append(rollout)
                atomic_json(out/'real_runs'/f'{name}_rollout.json',rollout)
        total_eligible=sum(s['actor_eligible'] for s in statistics if s['execution_audit']=='passed')
        total_success=sum(s['opportunity'].get('succeeded',0) for s in statistics if s['execution_audit']=='passed')
        collection=dict(status='passed' if rollouts and total_eligible>=20 and total_success>=1 else 'failed',attempts=attempts,
            complete_cohorts=len(rollouts),total_actor_eligible=total_eligible,verified_successful_kv_migrations=total_success,
            same_frozen_policy_sha256=initial_hash,episodes=statistics,scope='pipeline coverage only, not policy quality')
        atomic_json(out/'real_collection_report.json',collection)
        if rollouts:
            # Stop GPU service before CPU optimizer work, minimizing GPU occupancy.
            await asyncio.to_thread(service.stop)
            opt=torch.optim.Adam(net.parameters(),lr=SETTINGS['learning_rate']);obs=rollouts[0]['records'][0]['observation']
            diagnostics=[]
            for r in rollouts[0]['records']:
                if r['actor_eligible'] and len(diagnostics)<16:diagnostics.append(r['observation'])
            if not diagnostics:diagnostics=[obs]
            with torch.no_grad():dist,value=net.distribution(*tensors(diagnostics));pre=dict(probabilities=dist.probs.tolist(),values=value.tolist())
            update=ppo_update(net,opt,rollouts,SETTINGS,seed=manifest['policy_seed']);after=policy_hash(net)
            checkpoint(out/'checkpoints','real_updated.pt',net,opt,'q15b-p2-updated-v1',manifest,manifest['normalizer'])
            restored=CohortActorCritic();restored.load_state_dict(torch.load(out/'checkpoints/real_updated.pt',map_location='cpu',weights_only=False)['policy'])
            with torch.no_grad():
                a,v=net.distribution(*tensors(diagnostics));b,w=restored.distribution(*tensors(diagnostics))
            assert torch.equal(a.probs,b.probs) and torch.equal(v,w) and torch.isfinite(a.probs).all() and torch.isfinite(v).all()
            atomic_json(out/'real_update_report.json',dict(status='passed',real_update_calls=1,synthetic=False,update=update,
                before_sha256=initial_hash,after_sha256=after,parameters_changed=after!=initial_hash,checkpoint_roundtrip_exact=True,
                before=pre,after=dict(probabilities=a.probs.tolist(),values=v.tolist()),post_update_real_evaluation=False))
        state['status']='completed'
    except BaseException as exc:
        state.update(status='failed',reason=type(exc).__name__+': '+str(exc));(out/'real_error.txt').write_text(traceback.format_exc())
        if not (out/'real_collection_report.json').exists():atomic_json(out/'real_collection_report.json',dict(status='failed',attempts=attempts,episodes=statistics,reason=str(exc)))
    finally:
        stop_task.cancel();await asyncio.gather(stop_task,return_exceptions=True)
        await asyncio.to_thread(service.stop)
        env['after_cleanup']=environment();env['service_stopped']=read(out/'service_stopped.json') if (out/'service_stopped.json').exists() else None
        state.update(ended_wall=time.time(),gpu_elapsed_sec=time.monotonic()-started,attempts=attempts,durations_sec=durations)
        atomic_json(out/'environment_audit.json',env);atomic_json(out/'run_state.json',state)
        import psutil
        live=[]
        if (out/'owned_processes.json').exists():
            for r in read(out/'owned_processes.json')['processes']:
                try:
                    p=psutil.Process(r['identity']['pid'])
                    if p.create_time()==r['identity']['created']:live.append({'pid':p.pid,'status':p.status()})
                except psutil.NoSuchProcess:pass
        atomic_json(out/'cleanup_verification.json',dict(status='passed' if not live and not env['after_cleanup']['gpu_processes']['stdout'].strip().splitlines()[1:] else 'failed',
            owned_residuals=live,after_cleanup=env['after_cleanup'],gpu_elapsed_sec=state['gpu_elapsed_sec'],within_hard_budget=state['gpu_elapsed_sec']<=2100))
        final_report(out);print(json.dumps(state),flush=True)


def watchdog(out):
    """Independent supervisor process; owns only the named batch services."""
    out=Path(out);idle_started=time.time()
    while True:
        if (out/'run_state.json').exists() and read(out/'run_state.json')['status']!='running':return
        if (out/'gpu_budget.json').exists():
            budget=read(out/'gpu_budget.json')
            if time.time()>=budget['stop_service_by_wall']:
                atomic_json(out/'watchdog_stop.json',dict(reason='35min hard deadline with 60s reserved cleanup',wall=time.time()))
                service=ManagedFullCompletionService(out);service.running=service.is_running();service.stop()
                name=read(out/'controller_supervisor.json')['controller']
                subprocess.run(['supervisorctl','stop',name],timeout=180,check=False);return
        elif time.time()-idle_started>480:return
        time.sleep(1.)


def launch(out,entry_script=None):
    out=Path(out)
    load_v2_manifest(out)
    if read(out/'ppo_contract_report.json')['p2_cpu_contract']!='passed' or (out/'gpu_budget.json').exists():raise RuntimeError('not launchable')
    prefix='p23_'+hashlib.sha256(str(out).encode()).hexdigest()[:10];names={role:prefix+'_'+role for role in ('controller','watchdog')}
    import shlex
    entry_script=Path(entry_script or __file__).resolve()
    for role in names:
        name=names[role];wrapper=out/'services'/f'{name}.sh';conf=out/'services'/f'{name}.conf'
        cmd=[str(ROOT/'.venv/bin/python'),'-u',str(entry_script),'run' if role=='controller' else 'watchdog','--run-dir',str(out)]
        wrapper.write_text('#!/bin/bash\nset -e\n. /opt/supervisor-scripts/utils/logging.sh\n. /opt/supervisor-scripts/utils/environment.sh\ncd '+shlex.quote(str(ROOT))+'\nexec '+shlex.join(cmd)+' >> '+shlex.quote(str(out/(role+'.log')))+' 2>&1\n');wrapper.chmod(0o755)
        conf.write_text(f'[program:{name}]\nenvironment=PROC_NAME="%(program_name)s"\ncommand={wrapper}\nautostart=false\nautorestart=false\nstopasgroup=true\nkillasgroup=true\nstopsignal=TERM\nstopwaitsecs=180\nstdout_logfile=/dev/stdout\nredirect_stderr=true\nstdout_logfile_maxbytes=0\n')
        installed=Path('/etc/supervisor/conf.d')/conf.name
        if installed.exists():raise RuntimeError('supervisor name conflict')
        installed.write_bytes(conf.read_bytes())
    atomic_json(out/'controller_supervisor.json',names)
    for args in [('reread',),('update',names['watchdog']),('update',names['controller']),('start',names['watchdog']),('start',names['controller'])]:
        subprocess.run(['supervisorctl',*args],check=True,capture_output=True,text=True,timeout=30)
    print(json.dumps(names))


def verify_delivery(out):
    """Read-only CPU verification of sealed real data and frozen/updated checkpoints."""
    torch.set_num_threads(1)
    manifest=load_v2_manifest(out);report=read(out/'real_update_report.json')
    initial=torch.load(out/'checkpoints/initial.pt',map_location='cpu',weights_only=False)
    updated=torch.load(out/'checkpoints/real_updated.pt',map_location='cpu',weights_only=False)
    a=CohortActorCritic();a.load_state_dict(initial['policy']);a.eval()
    b=CohortActorCritic();b.load_state_dict(updated['policy']);b.eval()
    assert policy_hash(a)==report['before_sha256'] and policy_hash(b)==report['after_sha256']
    first_episode='ppo_'+str(manifest['trace_seeds'][0])
    rollout=read(out/'real_runs'/f'{first_episode}_rollout.json');records=rollout['records']
    with torch.no_grad():
        logp,_,_=a.evaluate(tensors([r['observation'] for r in records]),torch.tensor([r['action'] for r in records],dtype=torch.long))
        saved=torch.tensor([r['old_logp'] for r in records])
        assert torch.allclose(logp,saved,atol=2e-6,rtol=1e-6)
        dist,value=b.distribution(*tensors([r['observation'] for r in records]))
        assert torch.isfinite(dist.probs).all() and torch.isfinite(value).all()
    summary=read(out/'runs'/first_episode/'episode.json')
    decisions=stream_list(out/'runs'/first_episode/'decisions.jsonl')
    selected=[d for d in decisions if d.get('ppo') and d['ppo']['decision_ns'] in {r['decision_ns'] for r in records}]
    reward=MigPPOV2Reward.from_state_dict(manifest['reward']).reconstruct(summary,[r['decision_ns'] for r in records],selected)
    assert math.isclose(reward['return'],rollout['reward_audit']['return'],abs_tol=1e-11)
    assert report['real_update_calls']==1 and read(out/'cleanup_verification.json')['status']=='passed'
    return dict(status='passed',cpu_only=True,behavior_logp_max_abs_error=float((logp-saved).abs().max()),
                reward_reconstruction_error=abs(reward['return']-rollout['reward_audit']['return']),real_updates=1,transitions=len(records),
                actor_eligible=sum(r['actor_eligible'] for r in records),finite_updated_outputs=True,
                scope='accepted Q15-B only; neither policy quality nor general simulator validation')


def main():
    argparse.ArgumentParser(description=__doc__).error(
        'This is an internal implementation module. Use benchmark/run_ppo_sim.py '
        'for simulator/CPU checks or benchmark/run_ppo_real.py for the real run.'
    )

if __name__=='__main__':main()
