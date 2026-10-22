import hashlib,json,pathlib,subprocess
from datetime import datetime,timezone
ROOT=pathlib.Path(__file__).resolve().parents[5]
RUN=pathlib.Path(__file__).resolve().parents[2]
HERE=pathlib.Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(m):return hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
ptr=json.loads((ROOT/'rl_scheduler/runs/real_ppo_handoff/active_run.json').read_text())
assert pathlib.Path(ptr['run_dir'])==RUN
prereqs=[]
for n in ['handoff/02.json','handoff/03.json']:
 v=json.loads((RUN/n).read_text());assert v['status']=='passed'
 prereqs.append(dict(path=n,status=v['status'],sha256=sha(RUN/n),binding='historical; affected source bindings superseded by repair_01'))
for n in ['handoff/repair_01.json',*ptr['offline_gates'].values()]:
 v=json.loads((RUN/n).read_text());assert v['status']=='passed';count=0
 for field in ['source_hashes','config_hashes','evidence_hashes']:
  for name,expected in v[field].items():
   p=(ROOT/name) if field=='source_hashes' else (RUN/name if (RUN/name).exists() else ROOT/name)
   assert sha(p)==expected,(n,field,name);count+=1
 assert digest(v['source_hashes'])==v['source_hash']
 prereqs.append(dict(path=n,status='passed',sha256=sha(RUN/n),verified_bindings=count))
write(HERE/'prerequisites.json',prereqs)
old=json.loads((HERE/'preserved_hashes.json').read_text())
assert all(sha(ROOT/n)==h for n,h in old.items())
before=set((HERE/'git_status_before.bin').read_bytes().split(b'\0'))
after=subprocess.check_output(['git','status','--porcelain=v1','-z'],cwd=ROOT)
(HERE/'git_status_after.bin').write_bytes(after)
assert before<=set(after.split(b'\0'))
write(HERE/'preservation.json',dict(status='passed',source_config_pointer_clock_hashes_unchanged=len(old),prior_git_entries_retained=True,business_implementation_edits=[],GPU_services_started=False,models_loaded=False,experiment_clock_started=False))
commands=json.loads((HERE/'commands.json').read_text());assert all(c['exit_code']==0 for c in commands)
gate=json.loads((RUN/ptr['offline_gates']['P2']).read_text())
resolutions=[
 ('R1',['rl_scheduler/real_runner.py:24','rl_scheduler/real_runner.py:55'],'每个角色先验证，随后按 source 合并同轮可比较水位；跨轮 checkpoint 回退检测保留。'),
 ('R2',['rl_scheduler/real_env.py:121','rl_scheduler/real_bridge.py:335','rl_scheduler/real_bridge.py:356'],'Env 与 bridge 跟踪未决 ID；新非 Stay 提交前查询原 ID，仍未知则拒绝新提交；保留原 action/logp。'),
 ('R3',['rl_scheduler/real_bridge.py:96','rl_scheduler/real_bridge.py:130'],'第一次 await 前设置 acquiring fence；返回后复查 native idle 和 intervention，异常回收 acquiring。'),
 ('R4',['llumnix/real_control.py:14','llumnix/real_control.py:235','llumnix/entrypoints/vllm/api_server.py:69','llumnix/entrypoints/vllm/real_metrics_stream.py:82'],'Manager/四实例设置取消 tombstone，核查 dispatch、迁移、运行/等待/暂停/预分配队列、epoch 与投递；API 清空队列后关闭流；未证明的取消保持 unknown。'),
 ('R5',['rl_scheduler/real_runner.py:55','rl_scheduler/real_runner.py:99','rl_scheduler/real_env.py:72'],'reset 显式要求 API/Manager/四实例 local/worker 的 durable、passed、pending_delivery=0；prepare 有界。'),
 ('R6',['benchmark/run_real_ppo.py:66','benchmark/run_real_ppo.py:99','benchmark/run_real_ppo.py:147','benchmark/run_real_ppo.py:219'],'live plan 在预算写入和服务启动前验证当前独立审查及冻结顺序的前序 live gate；清理本任务服务之后才发布结果 gate。'),
 ('R7',['rl_scheduler/real_runner.py:24','rl_scheduler/real_runner.py:55'],'固定拓扑的必需角色先校验，失败 worker 即使缺少 source/durable 也被阻断，不能被 local fsync 替代。')]
md='''# 04 修复复审：passed

结论：**passed，原 R1–R7 已关闭；本轮未发现新的阻断问题。** 仅允许下一步骤按冻结协议执行 05 smoke；不表示真机验收通过，不自动启动 GPU、P3 或 PPO。

本轮依据 04 prompt 进行独立源码审查，核验 active_run、历史 02/03 passed、repair_01 交接和两个 repair_01 重验 gate。三个当前材料共 243 个源码/配置/证据绑定均匹配（85 + 79 + 79）。旧 failed 审查和交接逐字节归档至 `reviews/rereview_01/prior_04/`，原反例代码、失败日志和迁移链证据保留。

## R1–R7 复核

| 问题 | 当前源码 | 结论与实现依据 |
|---|---|---|
'''
for id,loc,why in resolutions:md+=f'| {id} | '+ '<br>'.join('`'+p+'`' for p in loc)+f' | 已关闭。{why} |\n'
md+='''
## 独立链路与验证

本轮实际运行 **30 个针对性 CPU 测试，全部通过**：原独立审查 9 项、修复回归 15 项、runner 集成及清理超时 6 项。修复报告中的 78 项全量测试只作为经 hash 核验的前置证据，没有冒称本轮又运行 78 项。

- 请求生命周期：客户端 send/receive 时钟用于 TTFT/TPOT 和评分；内部 engine/API/queue 时间仅用于归因。测试覆盖截止前/后的完成与 cleanup 对账，迟到完成不回填历史成功。完整输出流与 sampled causal 事件仍分离；缺失终态/producer 不得通过。
- 迁移链：陈旧候选在原仲裁入口被拒绝；有效候选沿 `ControlArbiter -> controlled_migrate -> MigrationCoordinator._migrate_out_one_request -> destination commit -> source free/unpause` 返回同一 decision/migration ID 的 succeeded。实际执行原 coordinator 方法体，RPC、KV 与引擎为 CPU mock，证据为本轮 `migration_chain.json`。
- 取消链：补审 tombstone 对晚到 generate/preallocate/receive/commit 的拦截，四实例 absence proof 的 epoch/角色稳定性，engine step/output barrier，API 队列和流结束顺序。未决 dispatch/迁移/投递/RPC 不被记作取消成功。已运行相关修复测试；并非实际 GPU 迁移取消验收。
- 观测与动作：采样时 observation/mask/action/logp 不回写，normalizer 在回合内冻结，UUID/epoch mapping 变化使回合失效。接管期间普通控制器受 acquiring fence 限制；故障介入使回合无效。
- 清理与资源：finish 共用 60 秒截止，未知终态、清理超时或未确认释放使回合 invalid；共享预算不因进程切换重置。CLI 的 finally 保留本任务服务清理，所有权限制未被移除；watchdog CPU 测试通过。
- 四条冻结命令的实际 `--dry-run` 均退出 0，写入本次审查前全部明确被旧 failed review 阻断，预算未写入。新的 passed 文件落盘后将另存一次只读门控核验至交接证据，不执行 live 命令。

证据：`reviews/rereview_01/original_9.log`、`repair_regressions.log`、`integration.log`、`migration_chain.json`、`commands.json`、`prerequisites.json`、`preservation.json`。源码与 smoke 配置逐文件及聚合 SHA256 见 `bridge_review.json`。

## 范围、预算与交接

仅写 reviews/handoff 和审查脚本；业务源码、配置、前序 gate、active pointer、实验时钟和既有 Git 状态均保留。没有加载模型、启动 GPU 服务、修改测试 seed/阈值、派生代理、提交或推送。本轮 GPU 实验时间为 0；实验时钟尚未启动。剩余总预算 28800 秒，P0–P3 7200 秒、P4 7200 秒、P5 14400 秒。

后续可执行范围仅为 05：依次 telemetry → shadow → active → overhead；后序阶段仍依赖真实前序 live gate。本步停止，实际性能、成功迁移次数、overhead 和 GPU 清理效果仍须 05 验证。active pointer 内的 failed_pending_rerun 是修复交接时的旧描述，本步依照只写 reviews/handoff 的范围保留；本次结论见 `handoff/04.json`，CLI 直接校验本次 `reviews/bridge_review.json`。
'''
(RUN/'reviews/bridge_review.md').write_text(md)
evidence={str(p.relative_to(RUN)):sha(p) for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
evidence.update({'reviews/test_bridge_review.py':sha(RUN/'reviews/test_bridge_review.py'),'reviews/bridge_review.md':sha(RUN/'reviews/bridge_review.md')})
for n in ['handoff/repair_01.json',*ptr['offline_gates'].values()]:evidence[n]=sha(RUN/n)
checks=[dict(name=n,status='passed') for n in ['02_03_and_repair_prerequisites_hashes','R1_R7_independent_regressions','30_targeted_CPU_tests','independent_request_and_original_migration_chain','four_frozen_CLI_dry_runs_block_failed_review','business_config_pointer_clock_and_prior_state_preserved']]
review=dict(stage='04_bridge_review',status='passed',run_id=RUN.name,run_dir=str(RUN),created_at_utc=datetime.now(timezone.utc).isoformat(),scope='P1/P2 independent repair rereview, CPU only',findings=[],resolved_findings=[dict(id=id,status='closed',locations=loc,reason=why) for id,loc,why in resolutions],checks=checks,source_hashes=gate['source_hashes'],source_hash=gate['source_hash'],config_hashes=gate['config_hashes'],config_hash=gate['config_hash'],evidence_hashes=evidence,hash_algorithm='SHA256 bytes; aggregate SHA256(sorted compact JSON(path -> file SHA256))',actual_commands_and_exit_codes=commands,remaining_budget=dict(total_experiment_wall_time_sec=28800.,P0_P3_sec=7200.,P4_sec=7200.,P5_sec=14400.),next_step=dict(step='05',executable=True,scope='frozen sequential smoke only; no GPU execution in this review',live_acceptance=False),limitations=['CPU actors/KV/engine mocks do not establish real GPU migration correctness or performance','Unknown/cancellation timeout conservatively invalidates episode; no threshold relaxation','active pointer metadata retained under 04 write scope; current review/handoff verdict supersedes prior failed review'])
review['dependency_hashes']={n:sha(ROOT/n) for n in ['.venv/lib/python3.10/site-packages/vllm/envs.py','.venv/lib/python3.10/site-packages/vllm/config.py','.venv/lib/python3.10/site-packages/vllm/engine/async_llm_engine.py']}
review['instruction_hashes']={n:sha(ROOT/n) for n in ['rl_scheduler/docs/real_ppo_model_prompts/04_astra_review_bridge.md','rl_scheduler/docs/real_cluster_ppo_execution_plan.md']}
write(RUN/'reviews/bridge_review.json',review)
print('published passed review; 30 CPU tests; unchanged implementation/config/pointer/clock')
