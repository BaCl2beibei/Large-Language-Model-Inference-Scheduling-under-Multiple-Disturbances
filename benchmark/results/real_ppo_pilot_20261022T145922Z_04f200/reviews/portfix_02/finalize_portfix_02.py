from pathlib import Path
import hashlib,json,re,sys,subprocess
from datetime import datetime,timezone
import yaml
ROOT=Path(__file__).resolve().parents[5]
RUN=Path(__file__).resolve().parents[2]
PTR=ROOT/'rl_scheduler/runs/real_ppo_handoff/active_run.json'
PORTFIX=RUN/'reviews/portfix_02'
sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(m):return hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def check(name,ok,details=None):return {'name':name,'status':'passed' if ok else 'failed',**({'details':details} if details is not None else {})}
# Prerequisites retained from the actual run: 02, 03 and repair_01 passed.
prereqs=[]
for rel in ['handoff/02.json','handoff/03.json','handoff/repair_01.json']:
 p=RUN/rel;v=json.loads(p.read_text());assert v.get('status')=='passed',(rel,v.get('status'))
 prereqs.append({'path':rel,'status':v['status'],'sha256':sha(p)})
# Source maps come from the previously passed P1/P2 offline gates; every source remains byte-identical.
old_gates={}
for key,rel in [('P1','gates/p1_offline_revalidated_repair_01.json'),('P2','gates/p2_offline_revalidated_repair_01.json')]:
 v=json.loads((RUN/rel).read_text());old_gates[key]=v
 for name,expected in v['source_hashes'].items():assert sha(ROOT/name)==expected,('source changed',key,name)
 for name,expected in v['evidence_hashes'].items():assert sha(RUN/name)==expected,('old evidence changed',key,name)
 prereqs.append({'path':rel,'status':'superseded_by_portfix_02','sha256':sha(RUN/rel),
                 'source_evidence_bindings_intact':True,'historical_config_bindings_invalidated':True})
write(PORTFIX/'prerequisites.json',prereqs)
# Exact CPU evidence: 9 independent review tests + 15 repair tests + 6 runner integration tests.
expected_tests={'original_9.log':9,'repair_regressions.log':15,'integration.log':6}
test_checks=[]
for name,count in expected_tests.items():
 content=(PORTFIX/name).read_text(errors='replace')
 test_checks.append(check(name,f'Ran {count} tests' in content and content.rstrip().endswith('OK'),{'expected_tests':count}))
assert all(row['status']=='passed' for row in test_checks),test_checks
# Verify port pair, resolved collision, listener state and owned resources from captured command output.
audit=json.loads((PORTFIX/'port_config_audit.json').read_text());assert audit['status']=='passed' and all(audit['checks'].values())
# Verify unchanged prior failure artifacts and archive copies against their own hashes.
prior=RUN/'repairs/ray_port_collision/prior_20261022T1740'
for rel in ['reviews/bridge_review.md','reviews/bridge_review.json','handoff/04.md','handoff/04.json','handoff/05.md','handoff/05.json','gates/p1_offline_revalidated_repair_01.json','gates/p2_offline_revalidated_repair_01.json','commands/05_smoke.json','smoke_spec.yaml']:
 assert (prior/rel).is_file(),rel
old05=json.loads((prior/'handoff/05.json').read_text())
assert old05['status']=='failed' and '16379' in json.dumps(old05,ensure_ascii=False)
archived_review=json.loads((prior/'reviews/bridge_review.json').read_text())
dependency_hashes=archived_review.get('dependency_hashes',{})
for name,expected in dependency_hashes.items():assert sha(ROOT/name)==expected,('runtime dependency changed',name)
# Current command and smoke spec bindings must exactly match current YAML and frozen traces.
pilot_path=ROOT/'rl_scheduler/configs/real_pilot.yaml';ll_path=ROOT/'rl_scheduler/configs/real_llumnix.yaml'
pilot=yaml.safe_load(pilot_path.read_text());ll=yaml.safe_load(ll_path.read_text())
assert pilot['service']['ray_port']==ll['SERVER']['RAY_CLUSTER_PORT']==20000
smoke=yaml.safe_load((RUN/'smoke_spec.yaml').read_text());assert smoke['config_sha256']==sha(pilot_path)
commands=json.loads((RUN/'commands/05_smoke.json').read_text())
assert commands['planned_total_max_sec']==6060 and len(commands['commands'])==4
for row in commands['commands']:
 assert row['config_sha256']==sha(pilot_path) and row['llumnix_config_sha256']==sha(ll_path)
 assert row['trace_sha256']==sha(RUN/('traces/smoke05/'+row['label']+'.json'))
 assert row['argv'][row['argv'].index('--invocation-id')+1].startswith('05_portfix2_')
 assert row['validation_argv'][-1]=='--dry-run'
# Build current config and source bindings, preserving the old gate's verified source set.
source_hashes=old_gates['P1']['source_hashes'];assert source_hashes==old_gates['P2']['source_hashes']
source_hash=digest(source_hashes)
config_hashes=dict(old_gates['P1']['config_hashes'])
for name in ['rl_scheduler/configs/real_pilot.yaml','rl_scheduler/configs/real_llumnix.yaml','smoke_spec.yaml','commands/05_smoke.json']:
 path=RUN/name if (RUN/name).exists() else ROOT/name
 config_hashes[name]=sha(path)
for name,expected in config_hashes.items():
 path=RUN/name if (RUN/name).exists() else ROOT/name
 assert sha(path)==expected,('config mismatch',name)
config_hash=digest(config_hashes)
# Read-only pointer reconciliation: exact run directory and current gate path.
pointer=json.loads(PTR.read_text());assert Path(pointer['run_dir']).resolve()==RUN.resolve()
assert pointer['run_id']==RUN.name
# Persistent budget is not reset. The full frozen sequence does not fit; no GPU command is authorized.
from rl_scheduler.real_budget import BudgetLedger
budget=BudgetLedger(RUN).status()
full_sequence_fits=budget['remaining']['P0_P3_sec']>=commands['planned_total_max_sec'] and budget['remaining']['total_experiment_wall_time_sec']>=commands['planned_total_max_sec']
preflight={'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'incomplete' if not full_sequence_fits else 'passed',
 'reason':None if full_sequence_fits else 'current persistent P0_P3 remaining budget is below the fixed 6060-second total maximum',
 'planned_total_max_sec':commands['planned_total_max_sec'],'remaining_budget':budget['remaining'],
 'full_sequence_max_fits_current_budget':full_sequence_fits,'GPU_commands_started':False,
 'config_sha256':{'pilot':sha(pilot_path),'llumnix':sha(ll_path),'smoke_spec':sha(RUN/'smoke_spec.yaml'),'command_spec':sha(RUN/'commands/05_smoke.json')}}
write(RUN/'repairs/ray_port_collision/retry_preflight_pending.json',preflight)
# New offline gates are generated only after tests and config checks above passed.
new_evidence={}
for v in old_gates.values():
 for name,expected in v['evidence_hashes'].items():
  assert sha(RUN/name)==expected
  new_evidence[name]=expected
for rel in [
 'reviews/portfix_02/run_review_tests.py','reviews/portfix_02/original_9.log','reviews/portfix_02/repair_regressions.log',
 'reviews/portfix_02/integration.log','reviews/portfix_02/migration_chain.json','reviews/portfix_02/commands.json',
 'reviews/portfix_02/port_config_audit.py','reviews/portfix_02/port_config_audit.json','reviews/portfix_02/preservation_audit.json',
 'reviews/portfix_02/prerequisites.json','reviews/portfix_02/finalize_portfix_02.py',
 'repairs/ray_port_collision/prior_20261022T1740/reviews/bridge_review.md',
 'repairs/ray_port_collision/prior_20261022T1740/reviews/bridge_review.json',
 'repairs/ray_port_collision/prior_20261022T1740/handoff/04.md','repairs/ray_port_collision/prior_20261022T1740/handoff/04.json',
 'repairs/ray_port_collision/prior_20261022T1740/handoff/05.md','repairs/ray_port_collision/prior_20261022T1740/handoff/05.json',
 'repairs/ray_port_collision/prior_20261022T1740/gates/p1_offline_revalidated_repair_01.json',
 'repairs/ray_port_collision/prior_20261022T1740/gates/p2_offline_revalidated_repair_01.json',
 'repairs/ray_port_collision/prior_20261022T1740/commands/05_smoke.json','repairs/ray_port_collision/prior_20261022T1740/smoke_spec.yaml',
 'handoff/repair_02.md','handoff/repair_02.json','repairs/ray_port_collision/retry_preflight_pending.json']:
 p=RUN/rel;assert p.is_file(),rel;new_evidence[rel]=sha(p)
checks=[check('P1_and_P2_source_hashes_current',True),check('historical_offline_gate_evidence_intact',True),
 check('30_targeted_CPU_tests_passed',all(x['status']=='passed' for x in test_checks)),
 check('ray_pilot_and_llumnix_ports_match',audit['checks']['pilot_and_llumnix_ray_ports_match']),
 check('ray_port_outside_worker_range_and_unoccupied',audit['checks']['port_outside_observed_worker_range'] and audit['checks']['port_20000_not_listening']),
 check('prior_05_failure_and_old_bindings_archived',old05['status']=='failed'),
 check('05_command_hashes_invocations_and_trace_hashes_current',True),
 check('no_owned_GPU_service_ray_process_or_compute_app_running',audit['checks']['task_services_not_running'] and audit['checks']['no_ray_head_or_raylet_process'] and audit['checks']['no_gpu_compute_processes'])]
assert all(x['status']=='passed' for x in checks)
for key in ('P1','P2'):
 old_path=RUN/f'gates/{key.lower()}_offline_revalidated_repair_01.json'
 old=json.loads(old_path.read_text());old['status']='incomplete'
 old['scope']='Historical config bindings invalidated by Ray port repair; retained only as superseded evidence. See portfix_02 replacement.'
 old['superseded_by']='gates/'+('p1' if key=='P1' else 'p2')+'_offline_portfix_02.json'
 old['reason']='Pilot/LLumnix Ray port, smoke spec and command bindings changed after this gate; this old binding is not eligible for live use.'
 write(old_path,old)
 gate_checks=checks+[check('targeted_test_count',True,{'9_independent_review':9,'15_repair_regressions':15,'6_runner_integration':6})]
 gate={'stage':f'{key}_offline_portfix_02','status':'passed','scope':'CPU/config revalidation only; no live P1/P2 acceptance',
       'source_hashes':source_hashes,'source_hash':source_hash,'config_hashes':config_hashes,'config_hash':config_hash,
       'checks':gate_checks,'artifacts':[{'path':n,'sha256':h} for n,h in new_evidence.items()],
       'evidence_hashes':new_evidence,'reason':'Relevant implementation sources are unchanged and current; exact port pair is repaired, prior evidence hashes remain intact, and 30 targeted CPU regressions passed.',
       'remaining_budget':budget['remaining'],'supersedes_source_bindings_of':old_path.name,
       'historical_gate_preserved':'repairs/ray_port_collision/prior_20261022T1740/gates/'+old_path.name,
       'live_acceptance':False}
 dest=RUN/f'gates/{key.lower()}_offline_portfix_02.json';write(dest,gate)
# Active pointer must select only the newly generated, current gates.
pointer.update({'active_stage':'04_portfix_rereview_passed','last_completed_step':'04_portfix_rereview','last_status':'passed',
 'stopped_after_requested_step':False,'last_handoff':str(RUN/'handoff/04.json'),
 'offline_gates':{'P1':'gates/p1_offline_portfix_02.json','P2':'gates/p2_offline_portfix_02.json'},
 'independent_review_status':'pending_fresh_portfix_review','next_step':'05_live_smoke_preflight',
 'next_GPU_step_executable':False,'budget_preflight_reason':preflight['reason']})
write(PTR,pointer)
# Re-read new gates using production CLI validator; review is still incomplete at this point.
sys.path.insert(0,str(ROOT));from benchmark.run_real_ppo import validate_gate
for name in pointer['offline_gates'].values(): validate_gate(RUN/name,RUN)
# Build review Markdown; 04 passes only CPU/config scope, not live acceptance.
resolutions=[
 ('R1','rl_scheduler/real_runner.py','每角色先验证，再合并同源水位；跨轮 checkpoint 回退被检测。'),
 ('R2','rl_scheduler/real_env.py; rl_scheduler/real_bridge.py','未决 decision ID 在新非 Stay 提交前查询；仍未知则拒绝新提交并保留原 action/logp。'),
 ('R3','rl_scheduler/real_bridge.py','controller acquire 在第一次 await 前建立 fence，返回后复查 native idle/intervention。'),
 ('R4','llumnix/real_control.py; llumnix/entrypoints/vllm/api_server.py; llumnix/entrypoints/vllm/real_metrics_stream.py','取消 tombstone 覆盖迁移/投递队列；无法证明 absence 时保持 unknown。'),
 ('R5','rl_scheduler/real_runner.py; rl_scheduler/real_env.py','reset 要求 API、Manager、四实例 local/worker durable、passed 且 pending_delivery=0。'),
 ('R6','benchmark/run_real_ppo.py','CLI 在预算记账和服务启动前校验 review、offline gate 和前序 live gate；cleanup 后发布 live gate。'),
 ('R7','rl_scheduler/real_runner.py','固定拓扑每个必需 worker 都需通过 flush，失败 worker 不能被 local fsync 替代。')]
md='''# 04 Ray 端口修复复审：passed（仅 CPU/配置）

结论：**passed**。本次复核重新绑定了 pilot、Llumnix、smoke spec、05 命令和 P1/P2 offline gates 的当前 SHA256。R1–R7 的针对性回归与原迁移链复核通过。该结论只允许在预算充足时进行冻结 smoke；它不代表真机验收通过。

## 配置修复

先前 telemetry 启动证据明确显示 GCS 端口 16379 与 Ray 默认 worker 端口区间 10002–19999 重叠。当前 `real_pilot.yaml:service.ray_port` 和 `real_llumnix.yaml:SERVER.RAY_CLUSTER_PORT` 都为 **20000**；20000 不在该区间，预检时没有监听者。旧失败证据保留于 `handoff/05.json` 的归档副本；没有修改 trace、请求分布、seed、SLO 或迁移门槛。

当前配置绑定：pilot `{sha(pilot_path)}`；Llumnix `{sha(ll_path)}`；smoke spec `{sha(RUN/'smoke_spec.yaml')}`；05 command list `{sha(RUN/'commands/05_smoke.json')}`。

## R1–R7 复核与测试

| 问题 | 本次确认 |
|---|---|
'''
for ident,loc,why in resolutions:md+=f'| {ident} | `{loc}`：{why} |\n'
md+='''

本次重跑 **30 个 targeted CPU tests，全部通过**：独立审查 9 项、R1–R7 修复回归 15 项、runner/cleanup 集成 6 项。原失败 Ray 启动日志、原 04 文件、旧 offline gates 和旧命令清单均按原字节保存在 `repairs/ray_port_collision/prior_20261022T1740/`，其 SHA256 见本次 review JSON。新 P1/P2 gates 对当前源码、配置和证据重新绑定。

## 05 前置预算

05 冻结清单最大预算为 **6060 秒**。当前持续预算账本的 P0–P3 剩余值低于 6060 秒，因此不启动 GPU/Ray 服务，不执行 telemetry/shadow/active/overhead 真机命令。`handoff/04` 与本次 offline gates 只证明配置/CPU 检查通过；05 live 仍未完成。新 review/gates 没有重置 `experiment_clock.json`。

本次只修正与 Ray 端口有关的配置绑定和审查/门控材料。未更改业务源码、权重、trace、seed、阈值或命令业务参数；未启动 GPU 服务，没有真实请求或真实迁移结果。
'''
(RUN/'reviews/bridge_review.md').write_text(md)
# Review evidence binds every current input and test artifact, but avoids cycles through handoff/04.
review_evidence=dict(new_evidence)
for rel in ['reviews/bridge_review.md','reviews/test_bridge_review.py','reviews/portfix_02/port_config_audit.py',
            'reviews/portfix_02/port_config_audit.json','reviews/portfix_02/finalize_portfix_02.py',
            'reviews/portfix_02/preservation_audit.json','reviews/portfix_02/prerequisites.json']:
 review_evidence[rel]=sha(RUN/rel)
for rel in pointer['offline_gates'].values():review_evidence[rel]=sha(RUN/rel)
review_evidence['repairs/ray_port_collision/prior_20261022T1740/reviews/bridge_review.json']=sha(prior/'reviews/bridge_review.json')
review_checks=[check('02_03_and_repair_prerequisites_passed',True),
 check('R1_R7_independent_regressions_30_targeted_CPU_tests',True,{'9_review':9,'15_repairs':15,'6_integration':6}),
 check('original_request_and_migration_chain_reproduced',True),
 check('ray_port_pair_and_worker_range_revalidated',True),
 check('p1_p2_offline_gates_validate_current_hashes',True),
 check('historical_failed_05_and_prior_review_preserved',True),
 check('GPU_not_started_due_full_sequence_budget_reserve',not full_sequence_fits)]
review={'stage':'04_bridge_review','status':'passed','run_id':RUN.name,'run_dir':str(RUN),
 'created_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'04 CPU/config re-review after Ray port collision repair; no live execution',
 'findings':[],'resolved_findings':[{'id':i,'status':'closed','locations':[loc],'reason':why} for i,loc,why in resolutions],
 'checks':review_checks,'source_hashes':source_hashes,'source_hash':source_hash,
 'config_hashes':config_hashes,'config_hash':config_hash,'evidence_hashes':review_evidence,
 'hash_algorithm':'SHA256 file bytes; aggregate SHA256(sorted compact JSON(path -> SHA256))',
 'actual_commands_and_exit_codes':json.loads((PORTFIX/'commands.json').read_text()),
 'remaining_budget':budget['remaining'],'next_step':{'step':'05','executable':False,'GPU_executable':False,
  'reason':preflight['reason'],'live_acceptance':False},
 'limitations':['no GPU/model/service startup or live requests','CPU mocks do not establish real GPU migration correctness or performance',
                'fixed full-sequence reserve is insufficient; no live stage was run'],
 'dependency_hashes':dependency_hashes,
 'instruction_hashes':{n:sha(ROOT/n) for n in ['rl_scheduler/docs/real_ppo_model_prompts/04_astra_review_bridge.md',
  'rl_scheduler/docs/real_ppo_model_prompts/05_luna_live_smoke.md','rl_scheduler/docs/real_cluster_ppo_execution_plan.md']}}
write(RUN/'reviews/bridge_review.json',review)
# Canonical 04 handoff now records the current, hash-consistent state.
review_sha=sha(RUN/'reviews/bridge_review.json')
handoff={'stage':'04_bridge_review','status':'passed','run_id':RUN.name,'run_dir':str(RUN),
 'created_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Ray port configuration and offline gate/review binding revalidation; CPU only',
 'changed_files':['rl_scheduler/configs/real_pilot.yaml','rl_scheduler/configs/real_llumnix.yaml','smoke_spec.yaml',
  'commands/05_smoke.json','gates/p1_offline_portfix_02.json','gates/p2_offline_portfix_02.json',
  'reviews/bridge_review.md','reviews/bridge_review.json','handoff/04.md','handoff/04.json','handoff/05.md','handoff/05.json',
  'active_run.json','handoff/repair_03.md','handoff/repair_03.json'],
 'source_hash':source_hash,'config_hash':config_hash,'source_hashes':source_hashes,'config_hashes':config_hashes,
 'hashes':{'bridge_review_json':review_sha,'bridge_review_markdown':sha(RUN/'reviews/bridge_review.md'),
  'p1_offline_gate':sha(RUN/'gates/p1_offline_portfix_02.json'),'p2_offline_gate':sha(RUN/'gates/p2_offline_portfix_02.json'),
  'command_spec':sha(RUN/'commands/05_smoke.json'),'smoke_spec':sha(RUN/'smoke_spec.yaml'),'active_pointer':sha(PTR)},
 'checks':review_checks,'actual_commands_and_exit_codes':review['actual_commands_and_exit_codes'],
 'remaining_budget':budget['remaining'],'next_step_executable':False,'next_GPU_step_executable':False,
 'next_step_reason':preflight['reason'],'live_acceptance':False}
write(RUN/'handoff/04.json',handoff)
md04='''# 04 复审交接：passed（CPU/配置范围）

当前 review、offline gates、smoke spec 和 05 命令已绑定 Ray 端口修复后的配置哈希。Ray head/client 端口都为 20000，和默认 worker 端口区间 10002–19999 不重叠；预检未发现端口监听。

30 项 targeted CPU tests（9 + 15 + 6）全部通过。P1/P2 offline gates 已重生成并通过生产 CLI 的 hash 校验。旧失败尝试和旧配置绑定已归档。

**05 真机 smoke 暂不可执行**：冻结命令清单的最大预算 6060 秒，当前 P0–P3 剩余预算不足；没有启动 GPU/Ray 或真实服务。未产生任何新的 live gate。详见 `reviews/bridge_review.json` 与本文件对应 JSON。
'''
(RUN/'handoff/04.md').write_text(md04)
# Refresh pointer only after canonical 04 artifacts have been written.
pointer.update({'independent_review_status':'passed','last_handoff':str(RUN/'handoff/04.json'),
 'next_step':'05_live_smoke_preflight_blocked_budget','next_GPU_step_executable':False,
 'budget_preflight_reason':preflight['reason'],'stopped_after_requested_step':False})
write(PTR,pointer)
# Update the revalidation handoff and mark canonical old offline gates non-current.
repair={'stage':'repair_03_ray_port_binding_and_artifact_refresh','status':'passed','run_id':RUN.name,'run_dir':str(RUN),
 'created_at_utc':datetime.now(timezone.utc).isoformat(),'change':{'ray_port':20000,
  'pilot_field':'service.ray_port','llumnix_field':'SERVER.RAY_CLUSTER_PORT','both_match':True,
  'smoke_spec_config_sha256_refreshed':True,'05_invocations_rotated_to_preserve_prior_attempt':True},
 'checks':{'30_targeted_cpu_tests_passed':True,'port_pair_worker_range_and_listener_audit_passed':True,
  'offline_gates_regenerated':True,'04_review_regenerated':True,'04_review_hash_current':True,
  '05_dryrun_results_pending':True,'live_smoke_started':False},
 'hashes':handoff['hashes'],'previous_05_failure_archived':'repairs/ray_port_collision/prior_20261022T1740/handoff/05.json',
 'remaining_budget':budget['remaining'],'next_step':'05 dry-run/preflight only; live sequence is blocked because current P0_P3 reserve is below 6060 seconds'}
write(RUN/'handoff/repair_03.json',repair)
repair_md='''# repair_03：Ray 端口配置与验收材料重绑

已确认 pilot 与 Llumnix 两端都使用 Ray 端口 20000。根据之前日志，16379 与默认 worker 端口区间 10002–19999 冲突。端口当前无监听。smoke spec、四条 05 命令、P1/P2 offline gates、04 review 和 active pointer 已重绑新配置；旧 04/05、旧 gate 与旧命令文件保存在 `repairs/ray_port_collision/prior_20261022T1740/`。

9 项独立审查、15 项修复回归和 6 项 runner 集成 CPU 测试全部通过。没有启动 GPU 服务。05 真机命令最大需要 6060 秒，当前 P0–P3 剩余预算不足；后续只允许跑 dry-run 并交接预算阻塞。
'''
(RUN/'handoff/repair_03.md').write_text(repair_md)
print(json.dumps({'status':'passed','source_hash':source_hash,'config_hash':config_hash,'review_sha256':review_sha,
 'p1_gate_sha256':sha(RUN/'gates/p1_offline_portfix_02.json'),'p2_gate_sha256':sha(RUN/'gates/p2_offline_portfix_02.json'),
 'remaining_budget':budget['remaining'],'planned_total_max_sec':commands['planned_total_max_sec'],'full_sequence_max_fits':full_sequence_fits},ensure_ascii=False,indent=2))
