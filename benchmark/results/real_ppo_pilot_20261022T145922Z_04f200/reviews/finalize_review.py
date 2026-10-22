"""Write only step-04 review/handoff artifacts; never modify earlier gates."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parents[1]
REV = RUN/'reviews'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
canonical = lambda obj: hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
before = json.loads((REV/'prerequisites.json').read_text())
gate = json.loads((RUN/'gates/p2_offline.json').read_text())

findings = [
    dict(id='R1', severity='P1/high', blocking=True,
         title='同进程 driver worker / Llumlet 的刷盘回执导致确定性的 checkpoint 回退',
         locations=['rl_scheduler/real_runner.py:24', 'rl_scheduler/real_runner.py:155',
                    'llumnix/llumlet/llumlet.py:293', 'llumnix/backends/vllm/executor.py:97',
                    'llumnix/backends/vllm/executor.py:325', 'rl_scheduler/real_metrics.py:237'],
         trigger='非 SPMD 的 TP=1 路径：driver_worker 是 Llumlet 进程内对象，二者调用同一进程的 causal_events.flush。先 flush worker，再 flush local；返回字典却先遍历 local，再遍历 workers。当前显式 smoke 配置 TP=1，未设置 SPMD；当前进程该开关未设置，安装版本默认 False。',
         consequence='同一 source 的 through_ns 从较新的 local 回退到旧 worker，台账记录 sticky checkpoint_regression。即使请求全完成、投递为零，cleanup audit 仍 incomplete，首条 telemetry smoke 就无法正常验收。此为源码与 CPU 复现结论，不冒称已运行真机。',
         reproduction='test_R1_same_process_worker_local_flush_must_not_regress：local through=12、worker through=11、source 相同、sequence=0，实际 incomplete / checkpoint_regression:same-process。',
         minimum_fix='按 producer identity 合并同一轮多角色 ack，保留最新有效水位并分别验证各角色的 pending/durable 状态；不要关闭一般性的回退检测。新增真实 TP=1 同进程布局和独立 worker 布局的 CPU 测试。'),
    dict(id='R2', severity='P1/high', blocking=True,
         title='提交结果未知后，环境继续用新 decision_id 提交动作而未先查询旧动作',
         locations=['rl_scheduler/real_env.py:142', 'rl_scheduler/real_env.py:146',
                    'rl_scheduler/real_env.py:173', 'rl_scheduler/real_bridge.py:326'],
         trigger='submit_migration 超时/连接丢失返回 receipt_status=unknown；后续 snapshot 仍有合法动作，例如请求未抵达 Manager，或已经执行完成且稍后冷却结束。',
         consequence='RealEnv.step 不检查 unresolved receipt；只在 finish 查询。反例产生 submit(run:ep:0) → submit(run:ep:1)，中间没有 query。Manager 已知的占槽仍能防并发，但不能替代客户端对未确认投递的决策追踪，违反先查询、未知即停的协议。',
         reproduction='test_R2_unknown_submit_must_query_before_new_decision；使用实际 RealEnv.step 和明确不确定回执，连续两次 step 即失败。',
         minimum_fix='在 bridge/env 持久保存 unresolved decision fence；任何新非 Stay 动作前按原 decision_id 查询，未确定时停止采样并进入有界清理。保留原 action/logp，不能把 unknown 改写成拒绝或 Stay。'),
    dict(id='R3', severity='P2/medium', blocking=True,
         title='控制权获取的 await 窗口未阻挡普通控制器',
         locations=['rl_scheduler/real_bridge.py:94', 'rl_scheduler/real_bridge.py:129',
                    'rl_scheduler/real_bridge.py:135', 'llumnix/manager.py:292'],
         trigger='acquire_controller 在 native_idle 检查后等待 state_provider；普通 Manager._migrate 不持 arbiter 锁，此时 lease/slot 仍为空，可使用已有 instance_info 发起 native migration。状态 RPC 若已采样但回包尚未到达，仍会返回旧的 idle 状态。',
         consequence='原生迁移已经进入而租约仍授予，且该次 native 动作发生时 ordinary_blocked=False，不会记录 intervention。并非声称空载 smoke 必然触发；这是接管协议的竞态缺口。',
         reproduction='test_R3_controller_acquisition_must_fence_native_during_await：通过两个 asyncio.Event 控制先采样、再 native 入场、再返回状态的顺序，租约仍成功返回。',
         minimum_fix='第一次 await 前设置 acquisition fence，并让普通调度器遵守；异常时正确回收 fence，授予前重新确认 native task/fault 状态。failover/pre-stop 仍保留，但必须记录 intervention 并使回合失效。'),
    dict(id='R4', severity='P1/high', blocking=True,
         title='迁移后首输出尚未到达时，取消确认只检查旧实例，可能虚报已取消',
         locations=['llumnix/entrypoints/vllm/api_server.py:69', 'llumnix/entrypoints/vllm/client.py:206',
                    'llumnix/entrypoints/client.py:127', 'llumnix/entrypoints/vllm/real_metrics_stream.py:80',
                    'llumnix/entrypoints/vllm/real_metrics_stream.py:122'],
         trigger='请求已在目标提交，但 API 尚未收到目标第一条输出；request_instances 仍指向旧源。abort 源实例对不存在的请求是无异常 no-op。源暂停/Manager 延迟 dispatch 也需要同类取消 fencing。',
         consequence='_real_metrics_abort_confirmed 只等待旧实例 abort RPC，然后清掉 client stream 并返回 True。stream 因 cancel 标志退出可被标记 cancelled，路由据此输出 cancelled_confirmed；目标请求实际仍可能运行。后续队列等待可能发现或等到它自然完成，但这不能追认先前虚假的取消证明。',
         reproduction='test_R4_abort_ack_must_cover_migrated_owner_before_output：执行真实 API callback 函数体，source abort 无操作，目标仍有 r，返回 confirmed=True 并清除客户端状态。RPC/目标状态为最小 CPU 模型，未执行真实迁移取消。',
         minimum_fix='由可权威追踪 dispatch/迁移状态的控制面执行取消 fence；确认源、目标、暂停/预分配及未完成 dispatch 均不能再产生该请求，并等待投递/流结束，再报告 confirmed。不能仅凭客户端最近输出位置或无异常 abort 返回值确认；未知保留 unknown。'),
    dict(id='R5', severity='P1/high', blocking=True,
         title='reset 忽略 API 侧 pending delivery / 活跃流',
         locations=['rl_scheduler/real_runner.py:59', 'rl_scheduler/real_runner.py:69',
                    'llumnix/entrypoints/vllm/real_control_api.py:20', 'rl_scheduler/real_env.py:72'],
         trigger='四实例快照的 running/waiting/pending_delivery 均零，但 API flush 返回 pending_delivery=1（队列、request_stream 或旧 HTTP handler 尚未清空）。',
         consequence='prepare 仅检查 status/durable，照常建立新 producer inventory 并开始新回合；“引擎空闲但仍有投递”的污染路径没有被 reset 阻止。',
         reproduction='test_R5_reset_must_reject_pending_API_delivery：使用实际 LiveEpisodeIO.prepare，API ack pending=1，预期拒绝，实际未抛错。',
         minimum_fix='reset 的集群空闲契约涵盖 API/Manager/worker 全部 pending 状态；非零或缺失均阻断 begin，且等待/清理使用同一有界截止。结合角色 ack 校验修复，不能只修改测试中的快照。'),
    dict(id='R6', severity='P1/high', blocking=True,
         title='真实执行计划未检查独立审查 gate 和 live 阶段依赖',
         locations=['benchmark/run_real_ppo.py:55', 'benchmark/run_real_ppo.py:68',
                    'benchmark/run_real_ppo.py:116'],
         trigger='调用现有非 dry-run telemetry/bridge 命令；02/03 offline gate passed，但 reviews/bridge_review.json 缺失或 failed。',
         consequence='plan 仍返回 GPU_will_start=True，随后 main 能调用 service.start。代码未读取审查结果；active/overhead 也未校验前序 live gate。人工按 05 prompt 检查可以避免本次误执行，但不满足编排器自己核验依赖及 hash 的合同。',
         reproduction='test_R6_live_plan_requires_review_gate：真实 plan、现有源码/config/trace/gate，仅用只读 budget stub 避免改共享时钟；缺失 passed review 仍生成 live intent。没有调用 service.start。',
         minimum_fix='live intent 生成前校验 hash 绑定的 step-04 审查结论；按冻结顺序验证相应 live 结果和依赖。缺失、failed、incomplete、相关 hash 变化都拒绝；dry-run 明确列出阻塞项。审查失败文件不得被自动覆盖成 passed。'),
    dict(id='R7', severity='P1/high', blocking=True,
         title='失败的 worker flush 回执被 flatten_acks 静默丢弃',
         locations=['rl_scheduler/real_runner.py:24', 'rl_scheduler/real_runner.py:66',
                    'llumnix/backends/vllm/worker.py:59'],
         trigger='MigrationWorker.real_control_flush 返回真实代码中的 {status: incomplete, reason: pending_cuda_telemetry}，其中没有 source/durable；workers 列表本身非空。',
         consequence='flatten_acks 递归到字符串后不产出记录，prepare 仅验证剩余成功 ack，并从它们反推 expected_sources。失败 producer/角色的存在被隐去，初次 inventory 可以错误通过。即使 local/worker 同进程，local fsync 也不能代替尚未结束的 worker CUDA telemetry queue barrier。',
         reproduction='test_R7_failed_worker_flush_must_not_disappear_from_inventory：一个 worker incomplete，其余 ack passed，真实 prepare 未拒绝。',
         minimum_fix='先验证完整响应树的每个必需角色/worker 的状态和结构，再展平成功的 producer ack；任一失败/未知/缺字段须显式保留并阻断。期望 inventory 应独立于本次成功回执建立；覆盖 pending_cuda_telemetry、writer_closed、tracing_disabled 等失败形状。'),
]

source_paths = set(gate['source_hashes']) | {
    'llumnix/backends/vllm/executor.py', 'llumnix/backends/vllm/scheduler.py',
    'llumnix/entrypoints/vllm/client.py', 'llumnix/entrypoints/client.py',
    'llumnix/llumlet/local_migration_scheduler.py', 'llumnix/llumlet/request.py',
    'llumnix/queue/zmq_server.py', 'llumnix/queue/zmq_client.py',
    'llumnix/global_scheduler/migration_policy.py', 'llumnix/global_scheduler/migration_filter.py',
    'llumnix/config/default.py', 'llumnix/utils.py', 'llumnix/constants.py',
}
source_hashes = {str(path): sha(ROOT/path) for path in sorted(source_paths)}
config_paths = list(gate['config_hashes']) + ['commands/05_smoke.json', 'traces/smoke05/manifest.json']
config_paths += ['traces/smoke05/'+name+'.json' for name in ('telemetry', 'shadow', 'active', 'overhead')]
config_paths += [str(p.relative_to(RUN)) for p in sorted((RUN/'services').glob('*')) if p.is_file()]
config_hashes = {name: sha(RUN/name if (RUN/name).is_file() else ROOT/name) for name in sorted(config_paths)}
evidence_names = ['prerequisites.json', 'commands.json', 'counterexamples_initial.log', 'counterexamples.log',
                  'failed_worker_flush.log', 'targeted_regressions.log', 'smoke_cli.json', 'smoke_cli.log',
                  'migration_chain.json', 'gpu_process_inventory.log', 'diff_check.log',
                  'test_bridge_review.py', 'verify_smoke_cli.py', 'finalize_review.py', 'git_status_before.bin']
evidence_hashes = {'reviews/'+name: sha(REV/name) for name in evidence_names}
current = {name: sha(ROOT/name) for name in before['business_hashes_before']}
assert current == before['business_hashes_before'], 'Business code changed during review'
assert sha(RUN/'experiment_clock.json') == before['experiment_clock_sha256'], 'Budget clock changed'
assert all(json.loads((RUN/p).read_text())['status'] == 'passed' for p in
           ('handoff/02.json', 'handoff/03.json', 'gates/p1_offline_revalidated_03.json', 'gates/p2_offline.json'))
remaining = {'total_experiment_wall_time_sec': 28800., 'P0_P3_sec': 7200., 'P4_sec': 7200., 'P5_sec': 14400.}
commands = json.loads((REV/'commands.json').read_text())
commands.append({'argv': ['.venv/bin/python', str(Path(__file__).relative_to(ROOT))],
                 'exit_code': 0, 'type': 'artifact generation; all preservation assertions passed before writing verdict'})
checks = [
    dict(name='02_03_current_prerequisites_and_hashes', status='passed', evidence='reviews/prerequisites.json'),
    dict(name='independent_request_and_migration_chain', status='passed', evidence='reviews/counterexamples.log'),
    dict(name='16_targeted_CPU_regressions', status='passed', evidence='reviews/targeted_regressions.log'),
    dict(name='four_frozen_CLI_dry_runs_readonly_budget', status='passed', evidence='reviews/smoke_cli.json'),
    dict(name='protocol_counterexamples_R1_R7', status='failed', evidence=['reviews/counterexamples.log', 'reviews/failed_worker_flush.log']),
    dict(name='business_code_and_budget_preserved', status='passed'),
]
report = dict(stage='04_bridge_review', status='failed', run_id=RUN.name, run_dir=str(RUN),
    created_at_utc=datetime.now(timezone.utc).isoformat(), scope='P1/P2 independent CPU review only',
    reviewer_model_note='Executed current session; no model-switch assertion and no subagents.',
    reason='7 reproducible protocol defects; return to Sol for implementation repair, then repeat independent review. Smoke not permitted.',
    findings=findings, checks=checks, source_hashes=source_hashes, source_hash=canonical(source_hashes),
    config_hashes=config_hashes, config_hash=canonical(config_hashes), evidence_hashes=evidence_hashes,
    hash_algorithm='SHA256 bytes; aggregate SHA256(sorted compact JSON(path -> file SHA256))',
    dependency_hashes={'.venv/lib/python3.10/site-packages/vllm/envs.py': sha(ROOT/'.venv/lib/python3.10/site-packages/vllm/envs.py')},
    instruction_hashes={p: sha(ROOT/p) for p in ['rl_scheduler/docs/real_ppo_model_prompts/04_astra_review_bridge.md',
                                               'rl_scheduler/docs/real_cluster_ppo_execution_plan.md']},
    actual_commands_and_exit_codes=commands,
    tests={'independent_cases_executed': 9, 'independent_passed': 2, 'independent_failed': 7,
           'targeted_regressions_passed': 16, 'CLI_dry_runs_passed': 4,
           'execution_note': 'First 8 independent cases in counterexamples.log; added R7 tested separately. Initial 7-case log retained. No production edits or threshold relaxation.'},
    accounting={'GPU_services_started': [], 'GPU_experiment_wall_sec_this_step': 0, 'GPU_hours_this_step': 0,
                'real_migration_successes_this_step': 0, 'PPO_updates': 0, 'experiment_clock_started': False,
                'measured_CPU_wall_sec_since_prerequisite_artifact': time.monotonic()-before['recorded_monotonic'],
                'CPU_measurement_limitation': 'Earlier reading/source inspection time not instrumented; this is not total turn duration.',
                'subprocess_verification_wall_sec': sum(c.get('elapsed_sec', 0) for c in commands)},
    remaining_budget=remaining, next_step_executable={'Sol_CPU_fix': True, '05_GPU_smoke': False, 'automatic_next_stage': False},
    business_implementation_modified=False, prior_gates_modified=False, active_pointer_modified=False,
    unresolved=['R1–R7 require Sol fixes and scoped revalidation; prior offline gates are historical and do not override this failed review.',
                'No real Ray/GPU transport, cancellation, migration success count, CPU latency or paired overhead acceptance was performed.',
                '60-second cleanup and persistent budget/watchdog passed targeted CPU tests; real GPU process teardown still requires a later authorized smoke after review passes.',
                'The instance workspace is not a persistent volume per P0; no external backup/upload performed.'],
    artifacts=['reviews/bridge_review.md', 'reviews/bridge_review.json', 'handoff/04.md', 'handoff/04.json'] + list(evidence_hashes))

lines = ['# 04 独立审查：failed', '',
    '结论：**failed，禁止按当前实现进入 05 GPU smoke，交回 Sol 修复。** 本结论只涉及 P1/P2 CPU 审查；没有真机验收、PPO 更新或优于基线的结论。', '',
    '已完整读取 04 prompt 和实例规范，阅读主任务书总体约束、阶段依赖及相关实现，核验 active_run 指向当前目录。02/03 均 passed；03 当前 P1 重验 gate 与 P2 gate 的源码、配置和证据逐项 hash 均匹配。02 旧 API/共享文档 hash 的变化由 03 重验材料解释，未直接将旧 hash 当成当前验收。', '',
    '执行边界：仅当前实例、只读业务代码、只新增 reviews/handoff 和审查测试；无子代理、无模型加载、无 GPU 服务、无训练、无提交推送。未修改 active pointer 或前序 gates。', '',
    '## 阻断问题', '']
for f in findings:
    lines += [f"### {f['id']} — {f['severity']} — {f['title']}", '',
              '源码：' + '；'.join('`'+x+'`' for x in f['locations']) + '。', '',
              '**触发条件：** '+f['trigger'], '', '**后果：** '+f['consequence'], '',
              '**CPU 证据：** '+f['reproduction'], '', '**最小修复要求：** '+f['minimum_fix'], '']
lines += ['## 独立链路追踪与七项范围覆盖', '',
    '请求链：planned/sent → API accepted → engine_first → api_first → 客户端累计输出 1→4 → server_terminal → client_terminal。独立案例中 a 的客户端 TTFT=1000ms、TPOT=100ms、窗内完成=1、token=4；b 的服务端在 7 秒完成但客户端截止 8 秒仍未完成，仍保留 outstanding，合计驻留积分 10 请求秒。b 在 9/10 秒收到输出/终态不改变冻结分数。发生在 1.3 秒、2 秒才收到的事件不进入 1.5 秒观测。此测试通过。', '',
    '迁移链：ManagerControlMixin.dispatch → Llumlet.migrate_out(expected_candidate_id=...) → controlled_migrate → 原 MigrationCoordinator._migrate_out_one_request → target commit ack → 释放源 → 终态回执。独立 CPU 测试执行 arbiter、adapter 和原 coordinator 方法体，只有 RPC/KV transfer 替换为 mock：陈旧 candidate 明确拒绝且零 dispatch；正确 candidate 产生唯一 migration_id，原 coordinator start/commit/resume/completed 事件一致；actual_request_id=request-0、copied_blocks=20、终态 succeeded、重复 decision_id 仅执行一次。证据 `reviews/migration_chain.json`。这是 CPU 链路验证，真实成功迁移数仍为 0。', '',
    '| 指引检查项 | 结果与边界 |',
    '| --- | --- |',
    '| 1. 客户端/内部时钟、迟到、未完成 | 独立生命周期与边界回归通过；使用 monotonic，客户端主指标独立于 API/engine；取消确认仍受 R4 阻断。 |',
    '| 2. 完整输出、缺事件/终态 | 累计 token 正增量、抽样仅诊断、未知/缺终态/序列缺口回归通过；刷盘真实布局 R1、失败 ack 隐去 R7 阻断完整性。 |',
    '| 3. 原迁移路径与控制权 | 复用原 coordinator，租约建立后 routine 被抑制、failover 保留；接管 await 竞态 R3 未解决。 |',
    '| 4. 候选/容量/原子槽/幂等/终态 | 正常原子提交、幂等及独立迁移关联通过；R2 未查询就发新动作、R4 取消证明不足。 |',
    '| 5. 在线观测、UUID、normalizer | received_ns 过滤、历史观测副本、epoch 映射、显式 missing mask 和冻结命名特征 normalizer 已检查，相关 CPU 测试通过。未输入未来 trace/实际最终长度或仿真预测成本。 |',
    '| 6. reset/60 秒清理/总预算 | timeout 使回合 invalid、持久 flock/fsync 时钟和 watchdog CPU 检查通过；R5/R7 允许不干净 reset，R4 不能证明已取消。实际 GPU 停服仍未验证。 |',
    '| 7. CLI/自动释放 | 四个冻结 argv 均可解析并通过受只读 budget stub 约束的真实 CLI dry-run；manage-services 的 finally/owned teardown/watchdog 路径存在，owned 清理保留无关进程测试通过。R6 未强制审查及 live 依赖。 |', '',
    '## 测试、证据与复现', '',
    '- `reviews/counterexamples.log`：8 项独立测试，2 passed / 6 failed，退出码 1。随后新增 R7 单独运行，`reviews/failed_worker_flush.log`：1 failed，退出码 1。合计 9 个独立案例，7 个失败即上述缺陷。',
    '- `reviews/targeted_regressions.log`：16 项针对性 CPU 回归 passed，退出码 0。没有反复运行 63 项全套。',
    '- `reviews/smoke_cli.json`：四个冻结 validation_argv 的 CLI main dry-run 均退出 0；仅替换 BudgetLedger 的写入式读取为当前已记录余额的只读检查。未运行 live argv、未写共享时钟。',
    '- 初次反例日志也保留在 `reviews/counterexamples_initial.log`。随后审查测试替换 import target，避免通过 llumnix.__init__ 导入完整运行栈，并加入原 coordinator 链；没有修改业务实现。',
    '- `git diff --check` 退出 0；只读 `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader` 退出 0、输出为空。', '',
    '从仓库根目录复现新增测试（预期当前实现退出 1）：', '',
    '```bash',
    'CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .venv/bin/python '+str((REV/'test_bridge_review.py').relative_to(ROOT)),
    '```', '',
    '完整实际命令及退出码保存在 `reviews/commands.json` 和 JSON 报告。业务源码逐文件 hash、smoke 配置/trace/commands/services hash 与测试日志 hash 均在 `reviews/bridge_review.json`；aggregate 算法为 sorted compact JSON 的 SHA256。', '',
    '## 预算与交回要求', '',
    '本步 GPU 实验墙钟和 GPU 小时均为 0；首次实验时钟未启动且其字节 hash 未变。剩余总预算 28800 秒，P0–P3 7200 秒、P4 7200 秒、P5 14400 秒。CPU 验证时间单独记录；从 prerequisites 落盘前的阅读/审查时间未插桩，不冒称精确总耗时。', '',
    'Sol 只修复上述实现和必要测试，保留 seed、SLO、容量/冷却/成功次数/延迟/开销门槛及失败证据；受影响的 P1/P2 gate 需重新验收并更新相关 hash，再交回独立审查。所有阻断关闭且审查 passed 前，05 smoke 不可执行；不继续 P3/PPO。', '',
    '本步交接：`handoff/04.md`、`handoff/04.json`。没有改业务实现、清理他人服务、更新 active pointer、提交或推送。']
(REV/'bridge_review.md').write_text('\n'.join(lines)+'\n')
(REV/'bridge_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
handoff = {**report, 'step': '04', 'status': 'failed',
           'review_path': 'reviews/bridge_review.json', 'review_sha256': sha(REV/'bridge_review.json'),
           'review_markdown_sha256': sha(REV/'bridge_review.md'),
           'changed_files': sorted(list(evidence_hashes) + ['reviews/bridge_review.md', 'reviews/bridge_review.json',
                                                          'handoff/04.md', 'handoff/04.json']),
           'stop': 'Requested review complete; return to Sol for CPU fixes. No automatic next step.'}
(RUN/'handoff/04.json').write_text(json.dumps(handoff, ensure_ascii=False, indent=2)+'\n')
(RUN/'handoff/04.md').write_text('''# 04 — P1/P2 独立审查

状态：**failed**。02/03 当前 gate/hash 前置核验通过，但独立审查复现 **7 个阻断问题**，因此 **05 GPU smoke 不可执行**；交回 Sol 修复并重验，本步停止。

问题：R1 同进程 worker/local 刷盘水位回退；R2 unknown 回执未查询即提交新 decision；R3 接管 await 期间普通迁移未被隔离；R4 迁移后首输出前可能虚报取消成功；R5 reset 忽略 API 待投递；R6 CLI 未强制审查/live 依赖；R7 失败 worker ack 被静默丢弃。严重性、源码行、触发条件、后果和最小修复要求见 [审查报告](../reviews/bridge_review.md)。

独立新增 9 个 CPU 案例：2 通过、7 按协议断言失败（退出码 1），失败均保留为反例；16 项针对性既有回归通过（退出码 0）；四条冻结 CLI 参数通过 dry-run 检查（只读 budget stub，无服务启动）。完成一条请求生命周期与一条迁移拒绝/成功链追踪，原 coordinator CPU 成功不计作真实迁移。完整命令、源码/配置/证据 SHA256 在 [机器可读交接](04.json) 和 [JSON 审查](../reviews/bridge_review.json)。

没有修改业务代码、前序 gate、active pointer、模型、配置或 seed；没有提交/推送、子代理、GPU 服务、PPO 或下一步实验。GPU 实验墙钟/小时均为 0，真实成功迁移 0，PPO 更新 0，基线比较未运行。当前 GPU 进程清单为空；共享 experiment_clock 字节未变、首次实验时钟未启动。

预算余额：总 28800 秒，P0–P3 7200 秒，P4 7200 秒，P5 14400 秒。CPU 检查计时独立，先前阅读时间未插桩。

下一步：仅 Sol CPU 修复 → 受影响 gate 重验/hash 更新 → 独立复审。保留现有阈值与失败证据；审查通过前不得进入 smoke。
''')
print(json.dumps({'status': report['status'], 'findings': len(findings),
                  'source_hash': report['source_hash'], 'config_hash': report['config_hash'],
                  'review': str(REV/'bridge_review.json'), 'handoff': str(RUN/'handoff/04.json')}, ensure_ascii=False))
