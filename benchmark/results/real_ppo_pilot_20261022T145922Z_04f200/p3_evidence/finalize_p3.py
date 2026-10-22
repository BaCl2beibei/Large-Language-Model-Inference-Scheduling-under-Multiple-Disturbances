"""Package step 06 CPU preparation under the explicit user smoke-skip exception.

Does not execute 05/07, mutate budget inputs/clock, or rewrite bridge acceptance.
"""
from datetime import datetime,timezone
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from unittest.mock import Mock,patch
import yaml

ROOT=Path(__file__).resolve().parents[4];RUN=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from rl_scheduler.real_bridge import digest
from rl_scheduler.real_budget import atomic_json
from rl_scheduler.real_opportunity import readonly_budget,maximum_duration,load_frozen_trace


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    os.chdir(ROOT)
    if (RUN/'handoff/06.json').exists():raise RuntimeError('step 06 already saved; preserve existing handoff')
    baseline=json.loads((HERE/'baseline.json').read_text());spec=yaml.safe_load((RUN/'opportunity_spec.yaml').read_text())
    runtime=yaml.safe_load((ROOT/spec['runtime_config']).read_text())
    for name,value in baseline['inputs'].items():assert sha(RUN/name)==value,name
    for name,value in baseline['source_hashes'].items():assert sha(ROOT/name)==value,name
    assert not (RUN/'gates/p1_live.json').exists() and not (RUN/'gates/p2_live.json').exists()
    assert spec['reward']['C_ref'] is None and spec['reward']['N_ref'] is None
    sources=['benchmark/run_real_opportunity.py','rl_scheduler/real_reward.py','rl_scheduler/real_opportunity.py',
        'rl_scheduler/real_opportunity_runner.py','rl_scheduler/tests/test_real_reward.py','rl_scheduler/tests/test_real_opportunity.py']
    source_map={name:sha(ROOT/name) for name in sources}
    commands=[]
    # These test commands were already executed in this turn; preserve their exact logs.
    for modules,log,count in [(['rl_scheduler.tests.test_real_reward','rl_scheduler.tests.test_real_opportunity'],'tests_initial.log',25),
        (['rl_scheduler.tests.test_real_opportunity.OrchestrationTests.test_actual_episode_over_CPU_HTTP_reconciles_normal_and_skew_routes'],'test_HTTP_episode.log',1),
        (['rl_scheduler.tests.test_real_reward','rl_scheduler.tests.test_real_opportunity','rl_scheduler.tests.test_real_metrics',
          'rl_scheduler.tests.test_real_runner.LiveFixtureTests.test_complete_CPU_HTTP_episode_through_real_env_IO_and_ledger'],'tests_final.log',58)]:
        text=(HERE/log).read_text();assert re.search(r'Ran '+str(count)+r' tests? in',text) and text.rstrip().endswith('OK')
        commands.append({'argv':[str(ROOT/'.venv/bin/python'),'-m','unittest',*modules,'-v'],
            'env':{'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},'exit_code':0,'test_count':count,
            'log':str((HERE/log).relative_to(RUN)),'evidence':'executed earlier in this step; not redundantly rerun by packaging'})
    def run(argv,log,required=True):
        result=subprocess.run(argv,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (HERE/log).write_text(result.stdout)
        commands.append({'argv':argv,'env':{'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},
            'exit_code':result.returncode,'log':str((HERE/log).relative_to(RUN)),'required':required})
        atomic_json(HERE/'commands.json',commands)
        if required and result.returncode:raise RuntimeError('required CPU verification failed: '+log)
        return result
    run([str(ROOT/'.venv/bin/python'),'-c',
        'import pathlib,sys; [compile(pathlib.Path(p).read_text(),p,"exec") for p in sys.argv[1:]]; print("Compiled",len(sys.argv)-1,"sources without bytecode writes")',
        *sources,str(HERE/'prepare_protocol.py'),str(Path(__file__).resolve())],'compile.log')
    run(['git','diff','--check'],'diff_check.log')
    for name in sources:
        assert all(line.rstrip()==line for line in (ROOT/name).read_text().splitlines()),name
    gpu=run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],'gpu_inventory.log')
    assert not gpu.stdout.strip(),'GPU compute process observed; no service action taken'
    supervisor=run(['supervisorctl','status'],'supervisor_inventory.log',required=False)
    assert 'real_ppo_' not in '\n'.join(line for line in supervisor.stdout.splitlines() if 'RUNNING' in line)
    before_status=(HERE/'git_status_before.bin').read_bytes().split(b'\0')
    after_status=subprocess.check_output(['git','status','--porcelain=v1','-z','--untracked-files=all'],cwd=ROOT)
    (HERE/'git_status_after.bin').write_bytes(after_status)
    status_transitions=[]
    for record in before_status:
        if not record or record in after_status.split(b'\0'):continue
        name=record[3:].decode();path=ROOT/name
        # These three files were created by this step before the status snapshot.
        # The workspace indexed them afterwards; all preexisting entries still match.
        own=HERE/'before'
        assert path.is_relative_to(own),name
        relative=path.relative_to(own)
        counterpart=ROOT/relative if str(relative).startswith('rl_scheduler/') else RUN/relative
        assert path.read_bytes()==counterpart.read_bytes(),name
        tracked=subprocess.run(['git','ls-files','--error-unmatch',name],cwd=ROOT,capture_output=True)
        assert tracked.returncode==0,name
        status_transitions.append({'path':name,'classification':'own baseline snapshot became tracked, bytes identical','sha256':sha(path)})
    atomic_json(HERE/'git_status_preservation.json',{'status':'passed','all_preexisting_status_entries_retained':True,
        'own_snapshot_status_transitions':status_transitions})
    for entry in baseline['source_inventory']:assert sha(ROOT/entry['path'])==entry['sha256'],entry['path']
    snapshot_hashes={};patches=[]
    for name,value in source_map.items():
        target=RUN/'source_snapshots/p3_after'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        snapshot_hashes[str(target.relative_to(RUN))]=value
        patches.extend(difflib.unified_diff([], (ROOT/name).read_text().splitlines(True),fromfile='/dev/null',tofile='p3_after/'+name))
    (HERE/'step06_source.patch').write_text(''.join(patches))
    atomic_json(RUN/'source_manifest_p3.json',{'stage':'P3_CPU_preparation','source_hashes':source_map,'source_hash':digest(source_map),
        'snapshots':snapshot_hashes,'bridge_source_hashes_unchanged':baseline['source_hashes'],
        'historical_manifest_preserved':'source_manifest_after.json','note':'phase-specific source binding; no bridge revalidation caused by additive P3 modules'})
    # Validate every frozen trace and its deterministic recipe without running requests.
    checked=[]
    for qps,seed in zip(spec['probe']['qps'],spec['probe']['seeds']):
        rows=load_frozen_trace(RUN,spec,seed=seed,qps=qps,phase='probe');checked.append({'phase':'probe','qps':qps,'seed':seed,'offered':len(rows),'materialized_hash':digest(rows)})
    for qps in spec['probe']['qps']:
        for seed in spec['opportunity']['seeds']:
            rows=load_frozen_trace(RUN,spec,seed=seed,qps=qps,phase='opportunity');checked.append({'phase':'opportunity','qps':qps,'seed':seed,'offered':len(rows),'materialized_hash':digest(rows)})
    atomic_json(HERE/'trace_validation.json',{'status':'passed','traces':checked,'final_burst':'rule and seeds frozen; no final policy results loaded'})
    code='''from pathlib import Path\nimport json,importlib.util,yaml\nfrom unittest.mock import Mock,patch\nroot=Path.cwd();run=root/"'''+str(RUN.relative_to(ROOT))+'''"\ns=importlib.util.spec_from_file_location("p3_plan_cpu",root/"benchmark/run_real_opportunity.py");m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\nc=json.loads((run/"commands/07_opportunity.json").read_text())["commands"][0]\nargs=m.parse_args(c["argv"][2:]);cfg=yaml.safe_load(args.config.read_text());spec=yaml.safe_load(args.opportunity_spec.read_text())\ntry:m.plan(args,spec,cfg)\nexcept RuntimeError as exc:print(json.dumps({"status":"blocked_before_budget_write_or_service","reason":str(exc)}))\nelse:raise AssertionError("CPU preparation must not launch a budget-insufficient experiment")\n'''
    # Document interfaces and inherited evidence limits; keep old implementation/gates untouched.
    implementation='''# Step 06 implementation and frozen protocol\n\n用户明确要求跳过 smoke，直接执行 06。本步因此完成 CPU 准备；不因 05 缺失而拒绝编码，也不把 05、p1_live 或 p2_live 改成 passed。例外原文、run_id、范围在 p3_evidence/user_override_06.json；07 的独立入口会保留此例外，而其他审计、预算、SLO、容量和冷却规则继续生效。当前无 GPU 执行授权。\n\n新增文件对应关系：\n\n- rl_scheduler/real_reward.py：按完整客户端事件重建好请求计数、outstanding/first/gap 三项时间积分、原 coordinator 尝试、失败/中止/明确拒绝、截止 tail。窗口 [lo,hi)，最后包含截止点一次；不裁剪。只有 scoring+cleanup 审计通过、回合 valid、动作终态明确、C_ref/N_ref 已测得时才输出 finalised reward。原 obs/action/logp 不回填。\n- rl_scheduler/real_opportunity.py：6 个短探测、健康阈值、固定选档、12 回合配对顺序、constrained Balanced、主指标/错误/截尾/零分母、机会 gate、trace 与预算只读检查。\n- rl_scheduler/real_opportunity_runner.py：使用原 LiveEpisodeIO、RealEnv、RealBridge 和原 KV 执行链；全程一个受监管服务，顺序执行，不派生代理。按 engine admission 审计实际初始实例，倾斜还要求 Manager 声明与预注册路由一致。首次审计失败保存 interrupted/gate 并停止，无自动重试。\n- benchmark/run_real_opportunity.py：独立 P3 CLI；主任务书的 run_real_ppo.py --stage opportunity 仍明确拒绝。保留旧入口源码避免破坏当前 hash 绑定的 04/P1/P2，07 清单调用新入口。dry-run 无启动、无预算写入；live 需当前 P3/bridge/config 绑定及完整最大时长。服务 finally 只清理本任务 owner。\n\n冻结规则：\n\n- 探测 QPS 3/6/9，每档 A/B 各一次 Stay，共 6 回合，30s 到达+60s 排空，分别用 7101/7102/7103。健康要求 offered 全 sent、completion >=99%、nonmanual error <=1%、first censor <=1%、send lateness P95 <=1s，全部请求/动作/清理/路由审计通过。\n- A 选健康正常档位的低侧中位数；B 选健康倾斜档位中，到达最后三分之一至少半数样本存在等待队列的最高档位。不存在则停止 incomplete，不挑新负载。\n- 机会 A/B 各 2101/2102/2103，Stay/Balanced 配对共 12 回合，120+120s 固定窗口，60s 清理。按 scene_index+seed_index 交替政策顺序，同一 scene/seed 对共享完整 trace、sampling、外生路由规则、窗口与实例 epoch 映射。\n- 输入/输出预算 256/512/1048 的 9 个组合等概率，保存本地 tokenizer 生成的 synthetic prompt pool，输入预算与实际编码数分别保存。输出预算不是实现长度。12 个候选/探测 trace 文件覆盖全部预注册负载，不截短 offered，记录 hash。\n- Balanced 使用实测 remaining-capacity/request、native 剩余容量阈值方向及 Balanced 排序/改善且不过冲的判据；迁移 delta 用实际 SR blocks。没有沿用不可见的 native killed-request override，版本及差异写在协议里。它不是预测剩余输出长度，也没有迁移时间/净收益拟合。一个槽、5s 终态冷却、2s 状态年龄、5%/16-block reserve 仍由原 bridge/env/executor 执行。\n- 主指标正常完成且 TTFT<=2000ms、请求 TPOT<=150ms 的数量/240s；单 token 只考核 TTFT。错误按固定截止客户端 timeout/service_error/all offered；手动取消另列；unknown 使审计 invalid。另列首输出截尾、终态截尾、未发送比例、完成率及绝对差值。\n- 机会 gate 要求至少一个场景 2/3 seed 提升 >=10%，且该场景所有 seed 非手动错误率增加均 <=1pp。使用整数计数比较门槛，不加 epsilon 放宽。Stay goodput=0 仅报绝对量，不自动通过；两场景都双方零则 incomplete。完整矩阵/trace/路由规则/窗口/映射不符一律 incomplete。\n- 奖励初始 good/hold/first/gap/attempt/bad/tail 权重 1/.2/.5/.2/.02/.2/2；C_ref、N_ref 当前 null。仅未来健康 Stay 机会回合按 pooled completed/seconds、outstanding-request-seconds/seconds 求值并分别 floor 1，保留原值、数据 ID 和不健康排除项。probe/Balanced/dev/final 不用于尺度拟合。记录分量占比与 reward 增但主指标降的配对，不自动调权；训练前最多一次有记录修订。\n- final burst 5101/5102/5103 规则现在冻结：基准 QPS 取探测选定 A，0–30/45–90/105–120s 为 1x，30–45/90–105s 为 3x，Poisson、同样九组合/采样/正常路由。机会选档后、任何最终策略表现可见前生成并 hash。现在没有查看或运行最终策略。\n\n客户端是否足够：现有 NUL JSON parser、累计正增量与 first/last/terminal 事件可支持请求级 TTFT/TPOT 和驻留/停顿代理积分；28 个原 P1 CPU 测试及新增 A/B HTTP 端到端验证通过。chunk 不是 token，不报告精确 ITL。没有真机样本，因此不能声称真机输出/迁移/取消链已验收；07 遇到缺事件、路由不符或 unknown 即停止，不能把未验证当成功。\n\n本步不改任何既有业务源码、配置、门槛、seed、04 review、P1/P2 offline gate、05 失败/预检材料或预算。阶段哈希独立，source_manifest_p3.json 引用旧桥接输入，原 source_manifest_after.json 保留。\n'''
    implementation+='\n包装核验首轮因 3 个本步新建初始副本后来成为已跟踪 clean 文件，严格 Git status 条目检查退出 1；失败材料保存在 p3_evidence/packaging_attempt_01/。修正仅针对本步副本，并逐字节与原文件比对，所有原有用户状态与源码均保留；58 项业务测试没有失败或被重跑掩盖。\n'
    (HERE/'implementation_06.md').write_text(implementation)
    (HERE/'reproduction.md').write_text('''CPU 复现（仓库根目录，不启动服务）：\n\n```sh\nCUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest rl_scheduler.tests.test_real_reward rl_scheduler.tests.test_real_opportunity rl_scheduler.tests.test_real_metrics rl_scheduler.tests.test_real_runner.LiveFixtureTests.test_complete_CPU_HTTP_episode_through_real_env_IO_and_ledger -v\n```\n\n07 的实际 parser/dry-run argv 在 commands/07_opportunity.json。不要在 06 执行 live argv。准备脚本会在已有 p3_ready 或实验后拒绝重新冻结；复跑测试改变日志时应重新核验 gate hash，而非手改 status。新 P3 CLI 与旧 run_real_ppo.py 分开，旧入口仍不支持 opportunity。\n''')
    manifest=json.loads((RUN/'traces/opportunity07/manifest.json').read_text())
    binding={'episode_budget_max_sec':maximum_duration(spec,runtime),'remaining_readonly':readonly_budget(RUN),
        'GPU_services_started':[],'GPU_experiment_wall_sec_this_step':0,'PPO_updates':0,'real_migrations_this_step':0,
        'clock_file_sha256':sha(RUN/'experiment_clock.json'),'CPU_wall_sec_since_baseline':
            (datetime.now(timezone.utc)-datetime.fromisoformat(baseline['started_utc'])).total_seconds(),
        'note':'CPU wall since snapshot only; earlier reading not instrumented. Existing clock continues since prior GPU attempt; this step does not alter it.'}
    atomic_json(HERE/'accounting.json',binding)
    doc=json.loads((RUN/'commands/07_opportunity.json').read_text());item=doc['commands'][0]
    assert all(sha(ROOT/name)==value for name,value in doc['source_hashes'].items())
    item['validation_exit_code']=0;item['validation_status']='CPU parser/dry-run; live budget may block; no GPU execution'
    atomic_json(RUN/'commands/07_opportunity.json',doc)
    # Create the CPU preparation gate. Missing live gates are an explicit user
    # exception, not checks falsely marked as passed.
    p3spec=importlib.util.spec_from_file_location('p3_cli_packaging',ROOT/'benchmark/run_real_opportunity.py')
    cli=importlib.util.module_from_spec(p3spec);p3spec.loader.exec_module(cli)
    args=cli.parse_args(item['validation_argv'][2:]);dependencies=cli.validate_inputs(args,spec,runtime,require_ready=False)
    configs={name:sha(RUN/name) for name in ['opportunity_spec.yaml','reward_initial.yaml','final_burst_spec.yaml',
        'commands/07_opportunity.json','p3_evidence/user_override_06.json','p3_evidence/normalizer_frozen.json',
        'traces/opportunity07/manifest.json','traces/opportunity07/prompt_pool.json']}
    configs.update({str(path.relative_to(RUN)):sha(path) for path in (RUN/'traces/opportunity07').glob('*.json')})
    configs.update({name:sha(ROOT/name) for name in ['rl_scheduler/configs/real_pilot.yaml','rl_scheduler/configs/real_llumnix.yaml']})
    configs.update({'budget.json':sha(RUN/'budget.json'),'experiment_spec.yaml':sha(RUN/'experiment_spec.yaml')})
    verification={'status':'passed','tests':{'new_reward_and_opportunity_tests':29,'existing_metrics_tests':28,'existing_HTTP_fixture':1,'total':58},
        'checks':{'hand_calculated_reward_and_time_integrals':True,'closed_deadline_partition_no_double_count':True,
            'unknown_incomplete_pending_rejected':True,'fixed_probe_selection_and_12_pair_gate':True,
            'normal_and_skew_real_CPU_HTTP_episode':True,'frozen_traces_and_final_burst_rule':True,
            'unmeasured_scales_remain_null':True,'all_bridge_sources_and_offline_review_hashes_preserved':True,
            'original_budget_inputs_and_clock_bytes_preserved':True,'prior_git_status_and_source_inventory_preserved':True,
            'compilation_and_whitespace':True,'explicit_user_smoke_skip_recorded_without_live_claim':True},
        'source_hashes':source_map,'source_hash':digest(source_map),'actual_commands_and_exit_codes':commands}
    atomic_json(HERE/'verification.json',verification)
    evidence_names=['baseline.json','verification.json','commands.json','tests_final.log','test_HTTP_episode.log','compile.log','diff_check.log',
        'gpu_inventory.log','supervisor_inventory.log','seed_collision_check.json','trace_validation.json','implementation_06.md','reproduction.md',
        'prepare_protocol.py','finalize_p3.py','accounting.json','step06_source.patch','git_status_preservation.json']
    evidence={str((HERE/name).relative_to(RUN)):sha(HERE/name) for name in evidence_names}
    evidence['source_manifest_p3.json']=sha(RUN/'source_manifest_p3.json')
    gate={'stage':'P3_CPU_preparation','status':'passed','scope':'06 code/protocol/CPU preparation only; user explicitly skipped 05 smoke',
        'source_hashes':source_map,'source_hash':digest(source_map),'config_hashes':configs,'config_hash':digest(configs),
        'checks':[{'name':name,'status':'passed'} for name in verification['checks']], 'evidence_hashes':evidence,
        'dependency_hashes':dependencies,'artifacts':['opportunity_spec.yaml','commands/07_opportunity.json','reward_initial.yaml','final_burst_spec.yaml'],
        'reason':'CPU preparation complete; p1_live/p2_live are absent and unverified under explicit user prerequisite exception',
        'remaining_budget':readonly_budget(RUN)['remaining'],'live_execution_ready':False,'PPO_training_ready':False,
        'unverified_prerequisites':['gates/p1_live.json','gates/p2_live.json'],'next_step_not_executed':'07'}
    atomic_json(RUN/'gates/p3_ready.json',gate)
    dry=run(item['validation_argv'],'dry_run_final.log')
    intent=json.loads(dry.stdout)
    assert dry.returncode==0 and intent['GPU_will_start'] is False
    assert intent['prior_smoke']=='skipped_by_user_not_verified'
    assert all('budget insufficient' in reason for reason in intent['blocked_reasons']),intent['blocked_reasons']
    assert intent['C_ref'] is None and intent['N_ref'] is None
    # Verify non-dry planning rejects the same budget before importing services or
    # making any clock write. No live CLI invocation or service.start is performed.
    args.dry_run=False
    try:cli.plan(args,spec,runtime)
    except RuntimeError as exc:
        assert 'budget insufficient' in str(exc)
        live_plan={'status':'blocked_before_service_or_budget_write','reason':str(exc)}
    else:raise AssertionError('expected current budget to block full opportunity sequence')
    atomic_json(HERE/'cli_validation.json',{'status':'passed','actual_dry_run':commands[-1],
        'dry_run_intent':intent,'live_plan_only':live_plan,'GPU_execution':'not_run'})
    # Bind the dry-run and final command log after they exist; the intent does not
    # embed this gate's own hash, so there is no circular evidence binding.
    verification['actual_commands_and_exit_codes']=commands
    atomic_json(HERE/'verification.json',verification)
    gate['evidence_hashes'].update({str((HERE/name).relative_to(RUN)):sha(HERE/name)
        for name in ['verification.json','commands.json','cli_validation.json','dry_run_final.log']})
    atomic_json(RUN/'gates/p3_ready.json',gate)
    cli.validate_inputs(args,spec,runtime)
    for name,value in baseline['inputs'].items():assert sha(RUN/name)==value,name
    for name,value in baseline['source_hashes'].items():assert sha(ROOT/name)==value,name
    pointer_path=ROOT/'rl_scheduler/runs/real_ppo_handoff/active_run.json';pointer=json.loads(pointer_path.read_text())
    pointer.update(active_stage='P3_CPU_prepared_smoke_skipped_by_user',last_completed_step='06',last_status='passed',
        last_handoff=str(RUN/'handoff/06.json'),last_attempted_step='06_CPU_preparation',next_step='07_opportunity_not_started_budget_insufficient',
        next_GPU_step_executable=False,smoke_status='skipped_by_explicit_user_instruction_unverified',
        smoke_skip_evidence=str(RUN/'p3_evidence/user_override_06.json'),p3_ready_gate='gates/p3_ready.json',
        budget_preflight_reason=live_plan['reason'],stopped_after_requested_step=True)
    atomic_json(pointer_path,pointer)
    handoff={'step':'06','status':'passed','status_scope':'CPU code and frozen protocol only, under explicit user smoke-skip exception',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'run_id':pointer['run_id'],'run_dir':str(RUN),
        'user_override':{'smoke':'skipped_by_user','proof':'p3_evidence/user_override_06.json','P1_P2_live':'unverified, no passed gate created'},
        'changed_source_files':sources,'source_hashes':source_map,'source_hash':digest(source_map),
        'config_hashes':configs,'config_hash':digest(configs),
        'evidence_hashes':{**gate['evidence_hashes'],'gates/p3_ready.json':sha(RUN/'gates/p3_ready.json')},
        'actual_commands_and_exit_codes':commands,'tests':verification['tests'],'CPU_verification_status':'passed',
        'artifacts':['opportunity_spec.yaml','commands/07_opportunity.json','reward_initial.yaml','final_burst_spec.yaml',
                     'source_manifest_p3.json','p3_evidence/implementation_06.md','gates/p3_ready.json'],
        'phase_hash_separation':'Only additive P3 modules bound here. Existing bridge/review/offline gates and source_manifest_after.json retained byte-for-byte.',
        'accounting':binding,'remaining_budget':readonly_budget(RUN)['remaining'],
        'unresolved':['Real P1/P2 smoke was explicitly skipped; real cancellation/migration/latency not accepted by this step',
            'C_ref/N_ref await healthy Stay opportunity samples; no observed reward scale or opportunity gain yet',
            'Current original P0-P3 budget insufficient for frozen 5760s opportunity maximum; no budget revision authorized or performed'],
        'next_step_executable':False,'next_step':'07 only on later execution instruction and sufficient valid budget',
        'GPU_services_started':[],'GPU_experiment_wall_sec_this_step':0,'real_migrations_this_step':0,'PPO_updates':0,'stop':True}
    atomic_json(RUN/'handoff/06.json',handoff)
    (RUN/'handoff/06.md').write_text('''# 06：CPU 准备 passed（按用户要求跳过 05）\n\n用户原文“跳过smoke，直接执行06_sol_p3_prepare.md”已记录。没有把缺失的 p1_live/p2_live 写为 passed，也未覆盖 05 incomplete 或原失败证据。06 的 live 前置按该明确指示豁免，完成本步代码/协议准备。\n\n新增 real_reward.py、real_opportunity.py、real_opportunity_runner.py 及独立 benchmark/run_real_opportunity.py；58 项 CPU 测试通过，包含 29 项新奖励/协议/编排测试、28 项原 P1 测试和原 HTTP 集成。新增 HTTP 回合同时核验正常/倾斜真实初始路由、实际累计 token、完整台账和清理。编译、空白与 hash 检查通过。\n\n已冻结 opportunity_spec.yaml、commands/07_opportunity.json、reward_initial.yaml、final_burst_spec.yaml、12 个候选/探测 trace 文件和 normalizer。QPS 3/6/9 固定健康选档，6 个探测后 12 个 Stay/Balanced 配对，所有阈值/顺序/路由/零分母规则见协议。C_ref/N_ref 仍 null，必须从未来健康 Stay 机会数据实测后才能重建奖励。没有更换既有 seed 或放宽阈值。\n\n独立 P3 入口保留已审查桥接源码不动；旧 run_real_ppo.py --stage opportunity 仍拒绝，07 清单使用新入口。P1/P2 offline gate 与 04 review 当前 hash 仍有效。gates/p3_ready.json 的 passed 仅表示本步 CPU 准备完成，不表示真机或机会门槛通过。\n\n07 最大预留 5760 秒（一次启动和停服、6 个短探测、12 个固定回合及每回合 60 秒清理），当前原预算不足。实际 dry-run 退出 0，唯一阻塞为预算；live plan 在服务/预算写入前拒绝。没有修改预算或重置时钟。\n\n本步没有启动 GPU/Ray/模型、机会实验或 PPO，也没有自动进入 07。源码/配置/证据 hash、命令和剩余预算见 06.json，详细实现与 CPU 复现见 p3_evidence/implementation_06.md、reproduction.md。\n''')
    final={'status':'passed','CPU_only':True,'handoff_sha256':sha(RUN/'handoff/06.json'),
        'p3_ready_sha256':sha(RUN/'gates/p3_ready.json'),
        'all_prior_inputs_preserved':all(sha(RUN/name)==value for name,value in baseline['inputs'].items()),
        'all_prior_bridge_source_bindings_preserved':all(sha(ROOT/name)==value for name,value in baseline['source_hashes'].items()),
        'clock_unchanged':sha(RUN/'experiment_clock.json')==baseline['inputs']['experiment_clock.json'],
        'no_live_gate_created':not (RUN/'gates/p1_live.json').exists() and not (RUN/'gates/p2_live.json').exists(),
        'no_opportunity_attempt_directory_created':not (RUN/'opportunity_runs').exists(),
        'current_gate_bindings_validated':True,'dry_run_exit_code':dry.returncode,'GPU_experiment_started':False}
    atomic_json(HERE/'final_validation.json',final)
    print(json.dumps({'status':'passed_CPU_preparation','tests':58,'gate':str(RUN/'gates/p3_ready.json'),
        'handoff':str(RUN/'handoff/06.json'),'07_started':False,'remaining_budget':handoff['remaining_budget']},indent=2))


if __name__=='__main__':main()
