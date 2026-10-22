from pathlib import Path
import json,hashlib,sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[5];RUN=Path(__file__).resolve().parents[2];PTR=ROOT/'rl_scheduler/runs/real_ppo_handoff/active_run.json'
sys.path.insert(0,str(ROOT))
from rl_scheduler.real_budget import BudgetLedger
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
pre=json.loads((RUN/'reviews/portfix_02/05_preflight_dryruns.json').read_text())
cmd=json.loads((RUN/'commands/05_smoke.json').read_text())
review=json.loads((RUN/'reviews/bridge_review.json').read_text())
assert review['status']=='passed' and pre['scope'].startswith('05 command/hash/order dry-run')
assert all(row['exit_code']==0 and row['GPU_will_start'] is False for row in pre['results'])
assert pre['results'][0]['ready_for_live'] is True
assert all(row['ready_for_live'] is False for row in pre['results'][1:])
status=BudgetLedger(RUN).status();required=cmd['planned_total_max_sec']
fits=status['remaining']['P0_P3_sec']>=required and status['remaining']['total_experiment_wall_time_sec']>=required
assert not fits and not pre['GPU_started']
run_names=['p1_live.json','p2_shadow_live.json','p2_active_live.json','p2_live.json']
live_present={name:(RUN/'gates'/name).exists() for name in run_names}
assert not any(live_present.values())
commands=[{'label':r['label'],'argv':r['argv'],'exit_code':r['exit_code'],'ready_for_live':r['ready_for_live'],
 'blocked_reasons':r['blocked_reasons'],'GPU_will_start':r['GPU_will_start'],'stdout_log':r['stdout_log'],
 'stderr_log':r['stderr_log']} for r in pre['results']]
review_sha=sha(RUN/'reviews/bridge_review.json')
hashes={'bridge_review_json':review_sha,'bridge_review_markdown':sha(RUN/'reviews/bridge_review.md'),
 'source_hash':review['source_hash'],'config_hash':review['config_hash'],
 'pilot_config':sha(ROOT/'rl_scheduler/configs/real_pilot.yaml'),'llumnix_config':sha(ROOT/'rl_scheduler/configs/real_llumnix.yaml'),
 'smoke_spec':sha(RUN/'smoke_spec.yaml'),'command_spec':sha(RUN/'commands/05_smoke.json'),
 'p1_offline_gate':sha(RUN/'gates/p1_offline_portfix_02.json'),'p2_offline_gate':sha(RUN/'gates/p2_offline_portfix_02.json'),
 'dryrun_preflight':sha(RUN/'reviews/portfix_02/05_preflight_dryruns.json'),
 'prior_failed_handoff':sha(RUN/'repairs/ray_port_collision/prior_20261022T1740/handoff/05.json')}
value={'stage':'05_live_smoke','status':'incomplete','run_id':RUN.name,'run_dir':str(RUN),
 'created_at_utc':datetime.now(timezone.utc).isoformat(),
 'scope':'修正后的 05 固定命令清单完成四条 dry-run；没有执行真实 telemetry/shadow/active/overhead。',
 'repair_summary':'Ray head 与 Llumnix client 均配置 20000，已离开默认 worker 端口 10002–19999；旧失败证据保留。',
 'preconditions':{'04_review_passed':True,'current_hash_bindings_match':True,'P1_telemetry_dryrun_ready':True,
  'shadow_active_overhead_blocked_until_p1_live':True,'GPU_idle':True,'task_services_stopped':True},
 'budget':{'remaining':status['remaining'],'planned_total_max_sec':required,
  'full_sequence_max_fits':False,'live_start_allowed':False,
  'reason':'当前持续 P0–P3 剩余预算低于固定清单 6060 秒最大值；不启动一个会挤占后续冻结步骤预算的真机序列。'},
 'actual_commands_and_exit_codes':commands,'live_gates_present':live_present,
 'gpu_started':False,'ray_started':False,'real_requests':0,'real_migrations':0,
 'source_hashes':{'review':review['source_hash']},'config_hashes':review['config_hashes'],
 'hashes':hashes,'changed_files':['smoke_spec.yaml','commands/05_smoke.json','reviews/bridge_review.json','reviews/bridge_review.md',
  'gates/p1_offline_portfix_02.json','gates/p2_offline_portfix_02.json','handoff/04.json','handoff/04.md','handoff/05.json','handoff/05.md',
  'handoff/repair_03.json','handoff/repair_03.md','active_run.json','repairs/ray_port_collision/'],
 'limitations':['dry-run 证明 hash、配置、Trace 与依赖门控可解析，不证明 Ray/GPU 启动或模型推理成功',
  'Telemetry 的固定 live 命令未启动；P1/P2 live gates 均不存在。'],
 'next_step_executable':False,'next_GPU_step_executable':False,
 'next_step':'本 run 的 P0–P3 预算不足；不得重置账本或缩短冻结命令。交付预算阻塞状态。'}
write(RUN/'handoff/05.json',value)
md='''# 05 smoke 预检交接：incomplete

修正后的四条固定命令均完成 `--dry-run`，退出码全为 0。Telemetry 在单阶段上满足 CLI 前置；shadow、active、overhead 因尚无 `p1_live.json` 按依赖顺序被正确阻断。四条都没有启动 GPU。

当前持续预算仅剩 **{p0:.1f} 秒 P0–P3**，低于冻结清单 **6060 秒**最大需求。因此没有执行真实 telemetry 或后续阶段，也没有生成任何 live gate。旧 telemetry 失败仍保存在归档中；本次没有重试真实命令。

Ray 配置已统一到 20000，旧端口冲突修复后的配置哈希已重新绑定到 04 review、P1/P2 offline gates 和 05 命令。当前结果只证明离线配置/CPU 回归及命令预检，不是实际真机验收。

详细 dry-run 命令、退出码、阻断原因、配置哈希与剩余预算见本文件对应 JSON。'''.format(p0=status['remaining']['P0_P3_sec'])
(RUN/'handoff/05.md').write_text(md)
pointer=json.loads(PTR.read_text())
pointer.update({'active_stage':'05_live_smoke_preflight_incomplete','last_completed_step':'04_portfix_rereview',
 'last_attempted_step':'05_live_smoke_preflight','last_status':'incomplete','stopped_after_requested_step':True,
 'last_handoff':str(RUN/'handoff/05.json'),'independent_review_status':'passed',
 'offline_gates':{'P1':'gates/p1_offline_portfix_02.json','P2':'gates/p2_offline_portfix_02.json'},
 'next_step':'budget_blocked_no_live_retry_in_current_run','next_GPU_step_executable':False,
 'budget_preflight_reason':value['budget']['reason']})
write(PTR,pointer)
# Finalize hashes in the earlier handoffs after the pointer update; no back-reference from pointer.
for rel in ['handoff/04.json','handoff/repair_03.json']:
 p=RUN/rel;d=json.loads(p.read_text());d.setdefault('hashes',{})['active_pointer']=sha(PTR)
 if rel=='handoff/repair_03.json':
  d['remaining_budget']=status['remaining'];d['five_preflight_sha256']=hashes['dryrun_preflight']
 write(p,d)
print(json.dumps({'status':'incomplete','P0_P3_remaining_sec':status['remaining']['P0_P3_sec'],
 'planned_total_max_sec':required,'dryrun_exit_codes':[x['exit_code'] for x in commands],
 'telemetry_ready':pre['results'][0]['ready_for_live'],'live_started':False,
 'handoff_05_sha256':sha(RUN/'handoff/05.json'),'active_pointer_sha256':sha(PTR)},ensure_ascii=False,indent=2))
