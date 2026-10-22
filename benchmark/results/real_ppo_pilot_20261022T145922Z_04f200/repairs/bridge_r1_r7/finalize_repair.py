"""Bind CPU repair evidence; leave step-04 failed and all live stages blocked."""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[5];HERE=Path(__file__).resolve().parent;RUN=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from rl_scheduler.real_bridge import digest
from rl_scheduler.real_budget import atomic_json


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    os.chdir(ROOT)
    baseline=json.loads((HERE/'baseline.json').read_text())
    verified=json.loads((HERE/'verification.json').read_text())
    assert verified['status']=='passed' and all(verified['checks'].values())
    sources=verified['source_hashes']
    assert all(sha(ROOT/name)==expected for name,expected in sources.items())
    assert all(sha(RUN/name)==expected for name,expected in baseline['review_files'].items())
    assert sha(RUN/'experiment_clock.json')==baseline['clock_sha256']
    old_review=json.loads((RUN/'reviews/bridge_review.json').read_text())
    assert old_review['status']=='failed'
    # Change bindings only; preserve every frozen invocation, workload and bound.
    command_path=RUN/'commands/05_smoke.json'
    command_before=HERE/'before/commands_05_smoke.json'
    if not command_before.exists():shutil.copyfile(command_path,command_before)
    old_commands=json.loads(command_before.read_text());commands=deepcopy(old_commands)
    commands['source_hashes']={name:value for name,value in sources.items() if not name.startswith('rl_scheduler/tests/')}
    commands['source_hash']=digest(commands['source_hashes'])
    commands['repair_01_note']='Bindings updated after CPU repair; original frozen argv/env/seeds/durations unchanged. Independent review is failed; do not run live.'
    assert commands['commands']==old_commands['commands']
    atomic_json(command_path,commands)
    preserved_configs={name:value for name,value in old_review['config_hashes'].items() if name!='commands/05_smoke.json'}
    for name,value in preserved_configs.items():
        p=RUN/name if (RUN/name).exists() else ROOT/name
        assert sha(p)==value,name
    configs=dict(preserved_configs,**{'commands/05_smoke.json':sha(command_path)})
    prior_manifest=RUN/'source_manifest_after.json'
    if not (HERE/'before/source_manifest_after.json').exists():shutil.copyfile(prior_manifest,HERE/'before/source_manifest_after.json')
    old_manifest=json.loads((HERE/'before/source_manifest_after.json').read_text())
    inventory={entry['path']:entry for entry in old_manifest['source_inventory']}
    for name,value in sources.items():inventory[name]={'path':name,'sha256':value,'bytes':(ROOT/name).stat().st_size}
    manifest={'stage':'P1_P2_offline_repair_01','source_hashes':sources,'source_hash':digest(sources),
              'source_inventory':sorted(inventory.values(),key=lambda e:e['path']),
              'prior_manifest':'repairs/bridge_r1_r7/before/source_manifest_after.json',
              'changed_files':verified['changed_files'],'independent_review':'failed_pending_rerun'}
    atomic_json(prior_manifest,manifest)
    report='''# Repair 01：R1–R7 CPU 修复\n\n本次只修复 04 failed 审查确认的七项问题；没有启动 GPU/Ray/模型、PPO、机会实验或下一阶段。原 reviews/*、handoff/04.* 及旧 gates 保持原字节。\n\n| 问题 | 最小根因与修改 | 复现与新增覆盖 |\n| --- | --- | --- |\n| R1 | 同进程多角色回执遍历顺序产生水位倒退。逐角色验证后，同轮按 producer identity 合并可比较的最大 sequence/through 水位；台账跨轮回退检测不变。 | 原 R1、独立 worker / 同进程、跨轮 regression。 |\n| R2 | unknown 后环境仍提交新 ID。env 与 transport 保存 unresolved fence；新非 Stay 前按旧 ID 查询，未确定即停止采样并有界清理。原 action/logp/unknown 保留，决策日志逐条 fsync。 | 原 R2；直接 transport、环境退出清理、已知终态后才允许新提交。 |\n| R3 | 获取租约时 RPC await 窗口允许 routine 入场。首次 await 前 acquisition fence，授予前复核 native task/fault/intervention；异常 finally 释放 acquisition fence。 | 原 R3；pre-stop intervention、provider 异常。 |\n| R4 | API 最近输出路由不能证明迁移后 owner。Manager 在 dispatch 端及全部四实例安装永久每 epoch tombstone；晚到 generate / reserve / prealloc / receive / commit 不能创建请求。跟踪 dispatch、入站传输、generating 和受控迁移；在原 engine step/output-processing barrier 后 abort 并检查 running/waiting/swapped、paused、preallocated、migration sets / pending、request_ids 和 delivery。Manager 再核对四实例 ID / epoch / 健康；API 等队列清空、HTTP stream done 后才 confirmed。失联、重建、未结束操作或未知迁移保持 unknown。real-* 请求不旁路 Manager 重派。保留实际 engine 生产事件，fence 后不再投递该请求的新输出。 | 原 R4；目标已运行而源无请求、暂停/预分配、晚到 dispatch/commit、in-flight engine barrier、RPC 不确定、epoch 变化、投递未空、unproven server outcome。 |\n| R5 | reset 只检查引擎快照。固定四实例 TP1 角色契约要求 API / Manager / local / worker 的 durable passed 及显式 pending=0；prepare 有 cleanup_sec 超时，失败立即阻断 begin。Manager pending 也包含 dispatch task。 | 原 R5；各角色 pending 缺失/非零。 |\n| R6 | CLI 未强制独立审查与 live 顺序。live plan 在服务/预算操作前检查 hash 绑定的 04 passed；telemetry → shadow → active → overhead 的 gate、结果与传递依赖 hash 逐项核验。真实结果和 owned cleanup 后由程序产出 p1_live / p2_shadow_live / p2_active_live / p2_live，已有尝试不覆盖。dry-run 只读时钟并列出阻塞。 | 原 R6；review 缺失/failed/incomplete/stale、前序缺失、结果 hash 改变及顺序。 |\n| R7 | 失败 worker 无 source/durable 被递归丢弃。先依据固定四实例 TP1 角色数量建立必需角色，再验证完整结构及每个 barrier，最后才合并 producer；失败回执落盘保留。 | 原 R7；pending_cuda_telemetry、writer_closed、tracing_disabled、缺失/多余 worker。 |\n\nCPU：78 项回归 + 原 9 项独立审查案例通过（含原 7 项反例及请求/原 coordinator 成功链）。原审查测试源码未改；唯一会写 reviews/migration_chain.json 的测试输出通过 Path.write_text 包装重定向到 repairs/bridge_r1_r7/review_migration_chain_after.json，原失败证据逐字节校验。\n\nverify_cpu.py 首次包装执行退出 1：将 supervisorctl status 的退出 3 当成必需成功检查。原输出显示既有 jupyter/pyworker EXITED；78+9 测试、编译及 diff 均通过。该包装失败单独保存 verification_attempt_01.json / commands_attempt_01.json；修正为只读清单后复用已完成测试证据完成核验，没有操作服务、重跑实验或隐藏失败。\n\n源码差异以 repair_01.patch 为准（相对独立审查输入，保留既有修改）。before/after 保存相关源码；source_manifest_after.json 已更新并保留先前副本。commands/05_smoke.json 仅更新源码绑定和说明，四个 argv/env/config/seed/trace/时长不变；原版在 before/commands_05_smoke.json，旧审查对应该版。\n\nCPU gate 使用 gates/p1_offline_revalidated_repair_01.json、gates/p2_offline_revalidated_repair_01.json；旧 gates 不改，因行为源码改变只作为历史材料。下一步是 04 独立重审。reviews/bridge_review.json 仍 failed，05 不可执行；尚无任何真机验收，p1_live / p2_live 尚未生成。\n'''
    (HERE/'report.md').write_text(report)
    (HERE/'reproduction.md').write_text('''从仓库根目录执行（CPU，无服务启动）：\n\n```sh\nCUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .venv/bin/python benchmark/results/real_ppo_pilot_20261022T145922Z_04f200/repairs/bridge_r1_r7/verify_cpu.py\n```\n\n复跑会改变本目录日志及证据 hash；必须核验新结果并重新签发 CPU gate，再交独立审查。finalize_repair.py 会更新本次 CPU gate、共享指针、源码 manifest 和命令 hash，不能用于将 failed 独立审查改为 passed。\n''')
    common_evidence={str(p.relative_to(RUN)):sha(p) for p in HERE.iterdir() if p.is_file() and p.name not in ('cli_validation.json','final_validation.json')}
    common_evidence['source_manifest_after.json']=sha(prior_manifest)
    gates={}
    for phase in ('P1','P2'):
        name='gates/'+phase.lower()+'_offline_revalidated_repair_01.json'
        gate={'stage':phase+'_offline_revalidated_repair_01','status':'passed',
            'scope':'CPU/mock repair only; independent 04 review remains failed; live not authorized',
            'source_hashes':sources,'source_hash':digest(sources),'config_hashes':configs,'config_hash':digest(configs),
            'checks':[{'name':name,'status':'passed'} for name,value in verified['checks'].items() if value],
            'evidence_hashes':common_evidence,'supersedes_source_bindings_of':
                'gates/p1_offline_revalidated_03.json' if phase=='P1' else 'gates/p2_offline.json',
            'historical_gate_preserved':True,'independent_review_status':'failed_pending_rerun'}
        atomic_json(RUN/name,gate);gates[phase]=name
    pointer_path=ROOT/'rl_scheduler/runs/real_ppo_handoff/active_run.json'
    if not (HERE/'before/active_run.json').exists():shutil.copyfile(pointer_path,HERE/'before/active_run.json')
    pointer=json.loads(pointer_path.read_text())
    assert Path(pointer['run_dir']).resolve()==RUN
    pointer.update(active_stage='P2_offline_repair_review_pending',last_completed_step='repair_01',
        last_status='passed',last_handoff=str(RUN/'handoff/repair_01.json'),offline_gates=gates,
        independent_review_status='failed_pending_rerun',next_step='04_independent_re_review',
        next_GPU_step_executable=False,stopped_after_requested_step=True)
    atomic_json(pointer_path,pointer)
    # Real CLI dry-runs: no BudgetLedger stub and no clock mutation.
    cli_runs=[]
    for item in commands['commands']:
        env=dict(os.environ,**item['env'],CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
        result=subprocess.run(item['validation_argv'],cwd=ROOT,env=env,capture_output=True,text=True)
        (HERE/('dry_run_'+item['label']+'.log')).write_text(result.stdout+result.stderr)
        intent=json.loads(result.stdout)
        assert result.returncode==0 and intent['GPU_will_start'] is False and intent['ready_for_live'] is False
        assert len(intent['blocked_reasons'])==1 and 'gate not passed:' in intent['blocked_reasons'][0]
        assert 'reviews/bridge_review.json' in intent['blocked_reasons'][0]
        cli_runs.append({'argv':item['validation_argv'],'env':item['env'],'exit_code':result.returncode,
                         'blocked_reasons':intent['blocked_reasons'],'GPU_will_start':False})
    spec=importlib.util.spec_from_file_location('repair_cli',ROOT/'benchmark/run_real_ppo.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    for name in gates.values():cli.validate_gate(RUN/name,RUN)
    import yaml
    cfg=yaml.safe_load((ROOT/'rl_scheduler/configs/real_pilot.yaml').read_text())
    live_plan_rejections=[]
    for item in commands['commands']:
        args=cli.parse_args(item['argv'][2:])
        try:cli.plan(args,cfg)
        except RuntimeError as exc:
            assert 'reviews/bridge_review.json' in str(exc)
            live_plan_rejections.append({'label':item['label'],'status':'blocked_before_budget_or_service','reason':str(exc)})
        else:raise AssertionError('live plan incorrectly admitted')
    atomic_json(HERE/'cli_validation.json',{'status':'passed','dry_runs':cli_runs,'live_plan_rejections':live_plan_rejections,
        'no_GPU_or_service_start':True,'shared_budget_clock_unchanged':sha(RUN/'experiment_clock.json')==baseline['clock_sha256']})
    assert sha(RUN/'experiment_clock.json')==baseline['clock_sha256']
    assert all(sha(RUN/name)==expected for name,expected in baseline['review_files'].items())
    evidence={str(p.relative_to(RUN)):sha(p) for p in HERE.iterdir() if p.is_file() and p.name!='final_validation.json'}
    for name in gates.values():evidence[name]=sha(RUN/name)
    handoff={'step':'repair_01','status':'passed','status_scope':'CPU implementation repair only; independent review pending',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'run_id':pointer['run_id'],'run_dir':str(RUN),
        'scope':'Fix confirmed R1-R7 and required CPU tests; no GPU/PPO or next stage',
        'changed_files':verified['changed_files'],'source_hashes':sources,'source_hash':digest(sources),
        'config_hashes':configs,'config_hash':digest(configs),'evidence_hashes':evidence,
        'CPU_tests':{'unit_regressions':78,'original_independent_review_cases':9,'status':'passed'},
        'actual_commands_and_exit_codes':verified['actual_commands_and_exit_codes']+cli_runs,
        'verification_wrapper_failure':'repairs/bridge_r1_r7/verification_attempt_01.json',
        'review_status':'failed_pending_independent_rerun','preserved_failed_review_sha256':baseline['review_sha256'],
        'revalidated_offline_gates':gates,'gates_requiring_independent_revalidation':['reviews/bridge_review.json'],
        'live_gates_not_run':['gates/p1_live.json','gates/p2_shadow_live.json','gates/p2_active_live.json','gates/p2_live.json'],
        'accounting':{'GPU_services_started':[],'GPU_experiment_wall_sec':0,'GPU_hours':0,'real_migrations':0,'PPO_updates':0,
            'CPU_wall_sec_since_snapshot':(datetime.now(timezone.utc)-datetime.fromisoformat(baseline['started_utc'])).total_seconds(),
            'CPU_time_note':'wall time since baseline snapshot; earlier document reading not instrumented',
            'experiment_clock_started':False,'experiment_clock_sha256':baseline['clock_sha256']},
        'remaining_budget':{'total_experiment_wall_time_sec':28800,'P0_P3_sec':7200,'P4_sec':7200,'P5_sec':14400},
        'limitations':['CPU actor/KV/engine fixtures only; no live cancellation or migration acceptance claim',
            'Unknown operations retain fences and invalidate cleanup; no forced unknown-to-cancelled conversion',
            'Independent step-04 failed review remains unchanged; repeat independent review before any live run'],
        'next_step':'04 independent re-review','next_step_executable':{'04_independent_CPU_review':True,'05_GPU_smoke':False},
        'stop':True}
    atomic_json(RUN/'handoff/repair_01.json',handoff)
    (RUN/'handoff/repair_01.md').write_text('''# Repair 01：CPU passed，独立审查待重验\n\n已修复 R1–R7；78 项 CPU 回归及原 9 项独立审查案例通过，编译和 diff 检查通过。详细根因、修改、测试与边界见 `repairs/bridge_r1_r7/report.md`，准确源码差异见 `repair_01.patch`。\n\n原 failed 审查、04 交接、旧 gate 和失败反例均保留。当前 CPU gate 为 `gates/p1_offline_revalidated_repair_01.json`、`gates/p2_offline_revalidated_repair_01.json`，共享指针已引用。命令仅更新源码 hash，冻结 argv/env/seed/trace/时长及所有验收阈值不变。\n\n四条真实 CLI dry-run 均退出 0、GPU_will_start=false，唯一阻塞均为 failed 独立审查；四个 live plan 均在预算/服务操作之前拒绝。未启动模型、Ray、GPU 服务、PPO 或下一阶段，实验时钟字节未变，总剩余 28800 秒（P0–P3 7200、P4 7200、P5 14400）。\n\n一次核验包装脚本误将既有 EXITED 服务导致的 supervisorctl status 退出 3 当作失败；失败记录已保存，修正为只读清单后复用既有成功测试结果完成核验。未触碰管理服务。\n\n下一步交给独立 04 CPU 审查。审查状态仍 failed；05 GPU smoke 不可执行。交接详情、源码/配置/证据 hash 与实际命令/退出码见 `repair_01.json`。完成本次限定修复并停止。\n''')
    final={'status':'passed','scope':'CPU repair packaging, not independent review or live gate',
        'source_bindings_current':all(sha(ROOT/name)==value for name,value in sources.items()),
        'original_review_bytes_preserved':all(sha(RUN/name)==value for name,value in baseline['review_files'].items()),
        'budget_clock_unchanged':sha(RUN/'experiment_clock.json')==baseline['clock_sha256'],
        'old_config_and_trace_bindings_preserved':True,'all_four_CLI_dry_runs_blocked_by_failed_review':True,
        'all_four_live_plans_rejected_before_budget_or_service':True,
        'handoff_sha256':sha(RUN/'handoff/repair_01.json'),'no_live_gate_created':all(not (RUN/name).exists() for name in handoff['live_gates_not_run'])}
    assert all(value is True for key,value in final.items() if key not in ('status','scope','handoff_sha256'))
    atomic_json(HERE/'final_validation.json',final)
    print(json.dumps({'status':'passed_CPU_only','handoff':str(RUN/'handoff/repair_01.json'),
                      'independent_review':'failed_pending_rerun','GPU_smoke_executable':False},indent=2))


if __name__=='__main__':main()
