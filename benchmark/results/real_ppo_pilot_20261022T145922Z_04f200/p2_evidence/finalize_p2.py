"""Bind step-03 CPU evidence and save the bounded next-step review handoff."""
from datetime import datetime,timezone
import difflib
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import yaml

ROOT=Path(__file__).resolve().parents[4];RUN=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rl_scheduler.real_bridge import digest
from rl_scheduler.real_budget import BudgetLedger,atomic_json


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    verification=json.loads((RUN/'p2_evidence/verification.json').read_text())
    assert verification['status']=='passed' and all(verification['checks'].values())
    source_map=verification['source_hashes']
    assert all(sha(ROOT/name)==expected for name,expected in source_map.items())
    prior_gate=json.loads((RUN/'gates/p1_offline.json').read_text())
    prior_manifest=json.loads((RUN/'source_manifest_after.json').read_text())
    if prior_manifest.get('stage')!='P2_offline':
        shutil.copyfile(RUN/'source_manifest_after.json',RUN/'p2_evidence/source_manifest_after_02.json')
    impl=(RUN/'implementation.md').read_bytes()
    old_impl=impl.split(b'\n## P2 / step 03',1)[0]
    assert hashlib.sha256(old_impl).hexdigest()==prior_gate['evidence_hashes']['implementation.md']
    (RUN/'p2_evidence/implementation_02.md').write_bytes(old_impl)
    assert sha(RUN/'p2_evidence/source_manifest_after_02.json')==prior_gate['evidence_hashes']['source_manifest_after.json']
    original_after=json.loads((RUN/'p2_evidence/source_manifest_after_02.json').read_text())
    changed_allowed=set(source_map)
    preserved=[entry for entry in original_after['source_inventory'] if entry['path'] not in changed_allowed]
    assert all(sha(ROOT/e['path'])==e['sha256'] for e in preserved)
    prior_status=(RUN/'p2_evidence/git_status_before.bin').read_bytes().split(b'\0')
    current=subprocess.check_output(['git','status','--porcelain=v1','-z','--untracked-files=all'],cwd=ROOT)
    assert all(record in current.split(b'\0') for record in prior_status if record)
    (RUN/'p2_evidence/git_status_after.bin').write_bytes(current)
    before=json.loads((RUN/'p2_evidence/source_before.json').read_text())
    diff=[]
    for name in sorted(source_map):
        old_path=RUN/'source_snapshots/p2_before'/name
        if old_path.exists():old=old_path.read_text().splitlines(True)
        elif name not in {e['path'] for e in original_after['source_inventory']}:old=[]
        else:continue
        new=(ROOT/name).read_text().splitlines(True)
        diff.extend(difflib.unified_diff(old,new,fromfile='p2_before/'+name,tofile='p2_after/'+name))
    (RUN/'p2_evidence/step03_source.patch').write_text(''.join(diff))
    snapshots={}
    for name,expected in source_map.items():
        target=RUN/'source_snapshots/p2_after'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target);assert sha(target)==expected
        snapshots[str(target.relative_to(RUN))]=expected
    inventory={entry['path']:entry for entry in original_after['source_inventory']}
    for name,expected in source_map.items():inventory[name]={'path':name,'sha256':expected,'bytes':(ROOT/name).stat().st_size}
    after={'stage':'P2_offline','source_hash':digest(source_map),'source_hashes':source_map,
           'source_inventory':list(sorted(inventory.values(),key=lambda x:x['path'])),
           'snapshots':snapshots,'prior_manifest':'p2_evidence/source_manifest_after_02.json',
           'preservation':{'all_prior_status_entries_retained':True,'untouched_prior_inventory_entries_verified':len(preserved)},
           'hash_algorithm':'SHA256 of sorted compact JSON(path -> file SHA256), UTF-8'}
    atomic_json(RUN/'source_manifest_after.json',after)
    budget=BudgetLedger(RUN).status();assert not budget['started'] and budget['elapsed_sec']==0
    started=json.loads((RUN/'p2_evidence/prerequisite_review.json').read_text())['started_monotonic']
    accounting={'CPU_wall_sec_since_prerequisite_review':time.monotonic()-started,
                'GPU_services_started':[],'GPU_experiment_wall_sec_this_step':0,'GPU_hours_this_step':0,
                'experiment_clock_started':False,'remaining_budget':budget['remaining'],
                'real_successful_migrations':0,'PPO_updates':0,'real_baseline_comparison':'not_run'}
    atomic_json(RUN/'p2_evidence/accounting.json',accounting)
    enforcement={'status':'implemented_and_CPU_mock_verified','live_GPU_deadline_validation':'not_run',
                 'historical_budget_input':'budget.json','authoritative_mutable_clock':'experiment_clock.json',
                 'supersedes':'P0 budget_expiry_enforcement implementation-status note; caps and origin are not reset',
                 'source_hashes':{name:source_map[name] for name in ('rl_scheduler/real_budget.py','rl_scheduler/real_service.py')},
                 'tests':'p2_evidence/tests.log','current_clock_started':False,'remaining_budget':budget['remaining']}
    atomic_json(RUN/'budget_enforcement.json',enforcement)
    cfg=yaml.safe_load((ROOT/'rl_scheduler/configs/real_pilot.yaml').read_text())
    argument_sources=[ROOT/'llumnix/arg_utils.py',ROOT/'llumnix/entrypoints/vllm/api_server.py',
                      ROOT/'.venv/lib/python3.10/site-packages/vllm/engine/arg_utils.py']
    texts='\n'.join(path.read_text() for path in argument_sources)
    flags=[value for value in cfg['service']['server_argv'] if value.startswith('--')]
    assert all(flag in texts for flag in flags)
    atomic_json(RUN/'p2_evidence/service_argument_check.json',{'status':'passed_static_CPU_source_check',
                'flags':flags,'sources':{str(path):sha(path) for path in argument_sources},
                'actual_server_launch':'not_run'})
    atomic_json(RUN/'p2_evidence/python_runtime.json',{'executable':sys.executable,
                'distributions':{name:metadata.version(name) for name in ('llumnix','vllm','ray','torch','aiohttp')},
                'CUDA_or_driver_changed':False,'packages_installed':[]})
    repro=(RUN/'reproduction.md').read_text()
    if '## Step 03 CPU reproduction' not in repro:
        (RUN/'reproduction.md').write_text(repro+'''\n## Step 03 CPU reproduction\n\n```sh\n.venv/bin/python benchmark/results/real_ppo_pilot_20261022T145922Z_04f200/p2_evidence/verify_p2.py\n```\n\nThis reruns 63 CPU tests, compilation, whitespace checks, the minimal trace-contract reproduction and read-only GPU/supervisor inventories. It does not start a service, GPU experiment, simulation fitting or PPO. Saved smoke commands are review artifacts under commands/05_smoke.json; do not execute them in step 03. CPU --dry-run verification used all four prepared argv arrays with --dry-run appended; see p2_evidence/cli_validation_commands.json. implementation.md and smoke_spec.yaml describe contracts, boundaries and unverified live paths.\n\nRe-running verification changes logs and hashes. Recheck source/config/evidence and actual exit codes before issuing a fresh gate. Do not silently hand-edit a gate status.\n''')
    command_doc=json.loads((RUN/'commands/05_smoke.json').read_text())
    assert all(source_map[name]==expected for name,expected in command_doc['source_hashes'].items())
    config_paths={'experiment_spec.yaml':RUN/'experiment_spec.yaml','budget.json':RUN/'budget.json',
                  'smoke_spec.yaml':RUN/'smoke_spec.yaml',
                  'rl_scheduler/configs/real_pilot.yaml':ROOT/'rl_scheduler/configs/real_pilot.yaml',
                  'rl_scheduler/configs/real_llumnix.yaml':ROOT/'rl_scheduler/configs/real_llumnix.yaml'}
    config_map={name:sha(path) for name,path in config_paths.items()}
    evidence_names=['p2_evidence/prerequisite_review.json','p2_evidence/verification.json',
                    'p2_evidence/commands.json','p2_evidence/tests.log','p2_evidence/compile.log',
                    'p2_evidence/diff_check.log','p2_evidence/gpu_process_inventory.log',
                    'p2_evidence/p1_revalidation.log','gates/p1_offline_revalidated_03.json',
                    'p2_evidence/cli_validation_commands.json','p2_evidence/smoke_seed_search.json',
                    'p2_evidence/service_preparation.json','p2_evidence/service_argument_check.json',
                    'p2_evidence/trace_generation_initial_failure.json','p2_evidence/repro_trace_contract.log',
                    'p2_evidence/step03_source.patch','p2_evidence/accounting.json','budget_enforcement.json',
                    'source_manifest_after.json','commands/05_smoke.json',
                    'p2_evidence/verify_p2.py','p2_evidence/prepare_smoke.py','p2_evidence/finalize_p2.py']
    evidence_names += [str(path.relative_to(RUN)) for path in sorted((RUN/'traces/smoke05').glob('*.json'))]
    evidence_names += [str(path.relative_to(RUN)) for path in sorted((RUN/'services').glob('*'))]
    evidence={name:sha(RUN/name) for name in evidence_names}
    gate={'stage':'P2_offline','status':'passed','run_id':json.loads((RUN/'budget.json').read_text())['run_id'],
          'created_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Step 03 CPU/mock implementation only',
          'source_hash':digest(source_map),'source_hashes':source_map,
          'config_hash':digest(config_map),'config_hashes':config_map,'evidence_hashes':evidence,
          'hash_algorithm':after['hash_algorithm'],
          'checks':[{'name':name,'status':'passed','evidence':path} for name,path in [
              ('02_and_P1_actual_hash_review','p2_evidence/prerequisite_review.json'),
              ('P1_current_source_revalidated_28_tests','gates/p1_offline_revalidated_03.json'),
              ('63_CPU_concurrency_lifecycle_HTTP_budget_and_ownership_tests','p2_evidence/tests.log'),
              ('compile_and_diff_check','p2_evidence/commands.json'),
              ('four_structured_CLI_dry_runs_and_frozen_traces','p2_evidence/cli_validation_commands.json'),
              ('P0_P3_budget_fit_no_GPU_clock_start','p2_evidence/verification.json'),
              ('prior_changes_and_deletions_preserved','source_manifest_after.json')]],
          'artifacts':evidence_names+['implementation.md','handoff/03.md','handoff/03.json'],
          'reason':'Required control/finite-episode/telemetry/budget contracts pass CPU/mock tests; live Ray/GPU correctness, latency, migration success and overhead remain unverified.',
          'remaining_budget':budget['remaining'],'live_status':'not_run','GPU_experiments':0,
          'next_step_executable':{'04_Astra_CPU_review':True,'automatic_smoke':False},
          'no_PPO_or_opportunity_implementation':True}
    atomic_json(RUN/'gates/p2_offline.json',gate)
    pointer=ROOT/'rl_scheduler/runs/real_ppo_handoff/active_run.json'
    p=json.loads(pointer.read_text());assert Path(p['run_dir']).resolve()==RUN.resolve()
    p.update(active_stage='P2_offline',last_completed_step='03',last_status='passed',
             stopped_after_requested_step=True,last_handoff=str(RUN/'handoff/03.json'))
    atomic_json(pointer,p)
    commands=json.loads((RUN/'p2_evidence/commands.json').read_text())+json.loads((RUN/'p2_evidence/cli_validation_commands.json').read_text())
    commands += [{'command':'.venv/bin/python inline first CPU smoke trace preparation','exit_code':1,
                  'error':'KeyError: arrival','resolution':'required arrival=poisson fixed, regression test added; failure evidence retained'},
                 {'command':'.venv/bin/python inline corrected four frozen trace preparation','exit_code':0},
                 {'command':'.venv/bin/python '+str(RUN/'p2_evidence/prepare_smoke.py'),'exit_code':0},
                 {'command':'.venv/bin/python '+str(RUN/'p2_evidence/verify_p2.py'),'exit_code':0}]
    changed=[name for name in source_map if name not in before or source_map[name]!=before.get(name)]
    unresolved=['Full Llumnix/vLLM/Ray runtime imports, actor placement and RPC lifetimes are not GPU-validated.',
                'Real SR transfer/abort, worker fsync and end-to-end pending-delivery correctness require future smoke.',
                'Strict CPU decision P95<100ms and two paired overhead thresholds are implemented but not live measured.',
                'Supervisor watchdog and owned GPU-process teardown under failure require actual bounded smoke validation.',
                'First smoke conservatively waits fixed scoring deadline; optional early drain termination is not enabled.',
                'Serving process retains duplicate-ID tombstones and token high-water IDs until service shutdown; the bounded smoke lifecycle rotates services.',
                'PPO, opportunity and independent final comparison are unimplemented/refused; no superiority claim.',
                'Non-volume workspace still needs off-box backup before recycle/destroy; nothing uploaded.']
    handoff={'step':'03','stage':'P2_offline','status':'passed','run_id':p['run_id'],
             'scope':gate['scope'],'changed_source_files':changed,
             'configuration_files':['rl_scheduler/configs/real_pilot.yaml','rl_scheduler/configs/real_llumnix.yaml'],
             'artifact_files':gate['artifacts'],'source_hashes':source_map,'source_hash':gate['source_hash'],
             'config_hashes':config_map,'config_hash':gate['config_hash'],'evidence_hashes':evidence,
             'p2_gate_sha256':sha(RUN/'gates/p2_offline.json'),
             'P1_current_gate':'gates/p1_offline_revalidated_03.json','P1_current_gate_sha256':sha(RUN/'gates/p1_offline_revalidated_03.json'),
             'active_pointer_sha256':sha(pointer),'actual_commands_and_exit_codes':commands,
             'tests':{'CPU_tests_passed':63,'P1_regressions_passed':28,'temporary_localhost_HTTP_and_mock_actors':True,
                      'GPU_experiments':0,'real_migration_successes':0,'PPO_updates':0},
             'accounting':accounting,'remaining_budget':budget['remaining'],'unresolved':unresolved,
             'smoke_commands':{'path':'commands/05_smoke.json','sha256':sha(RUN/'commands/05_smoke.json'),
                               'status':'prepared_not_executed','commands':4,'episodes':7,'estimated_max_sec':6060},
             'next_step_executable':{'04_Astra_CPU_review':True,'05_live_requires_future_authorized_prompt_and_review':True,
                                     'automatic_next_stage':False},
             'stop':'All requested step-03 artifacts saved; no smoke or later stage started.'}
    atomic_json(RUN/'handoff/03.json',handoff)
    markdown=f'''# 03 — P2 动作桥接与真实环境\n\n状态：**passed，仅 P2 offline**。63 项 CPU/mock 测试通过，含 28 项 P1 回归和真实 localhost HTTP 的采集→台账→有限回合贯通测试。没有启动 GPU/Ray/推理服务，没有执行 smoke、PPO 或下一阶段。真实成功迁移数 0、PPO 更新数 0、真实基线比较未运行。\n\n交付接口：`real_bridge.py`、`real_env.py`、`real_observation.py`（独立冻结 normalizer）、`real_budget.py`、`real_service.py`、`real_runner.py` 与 `benchmark/run_real_ppo.py`；Manager/调度器/Llumlet/原 KV coordinator 的默认兼容分支；逐阶段 receipt、SR 执行复核、目标 epoch/容量复核、原子单槽、终态后冷却、租约退出、未知状态保留、完整投递/刷盘确认。源码文件及逐个 hash 见 JSON 交接和 `source_manifest_after.json`；相对本步开始的 diff：`p2_evidence/step03_source.patch`。既有修改、未跟踪文件和删除状态均保留；没有提交、推送或安装依赖。\n\n原 `p1_offline.json` 的 API 源码已发生相关改动，因此重新运行了全部 28 项 P1 CPU 测试；当前使用 `gates/p1_offline_revalidated_03.json`。原 gate、交接和准确的原 manifest/implementation 副本保留，不把旧 hash 冒充当前。\n\n主要命令/退出码：\n\n- `.venv/bin/python -m unittest rl_scheduler.tests.test_real_metrics rl_scheduler.tests.test_real_bridge rl_scheduler.tests.test_real_env rl_scheduler.tests.test_real_runner -v` → 0（63 tests）\n- `.venv/bin/python -m py_compile <26 个相关源码/测试文件>` → 0\n- `git diff --check` → 0\n- 四条 smoke argv 加 `--dry-run` → 全部 0，完整参数在 `p2_evidence/cli_validation_commands.json`\n- `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader` → 0，输出为空\n- 只读 `supervisorctl status` → 3，原有 Jupyter/pyworker 为 EXITED；没有改动/停止管理服务\n\nCPU trace 准备首次因缺少旧生成器必需的 `arrival` 字段退出 1；已修复、补回归测试并保留失败说明和最小复现。没有 GPU 失败尝试，也没有改 seed、SLO 或门槛求通过。完整命令、退出码和证据 hash 见 JSON 交接。\n\nGPU 实验时钟尚未启动。本步 GPU 墙钟/小时均为 0；CPU 墙钟约 {accounting['CPU_wall_sec_since_prerequisite_review']:.1f} 秒（从前置 hash 核查开始计）。剩余总实验预算 28800 秒；P0–P3 7200 秒，P4 7200 秒，P5 14400 秒。跨进程 flock/fsync 预算和独立 supervisor watchdog 已实现并通过 CPU/mock 测试；实际 GPU 故障下的停服仍未验证。原 `budget.json` 作为不可变 P0 输入保留，当前实现状态见 `budget_enforcement.json`，计时状态见 `experiment_clock.json`。\n\n审查材料：\n\n- `gates/p2_offline.json`（SHA256 `{handoff['p2_gate_sha256']}`）\n- `smoke_spec.yaml`\n- `commands/05_smoke.json`（仅准备，尚未执行）\n- `implementation.md` 的 P2 章节与 `reproduction.md`\n\n四条已检查命令包含低负载 telemetry、shadow、受约束主动 SR 和两组监控开销配对，共 7 回合，最长预计 6060 秒（含启动/清理/停服），在当前 P0–P3 7200 秒余额内。冻结 trace 请求数分别为 5/21/95/18；overhead 的 18-request trace 按 OFF/ON/ON/OFF 复用。主动门槛仍是至少 10 个明确终态、5 次成功，shadow 零真实迁移，CPU P95<100ms，开销退化不超过 5% throughput / 10% TTFT P95，均未真机验收。\n\n遗留：完整 Ray/vLLM 调用、真实迁移/取消、跨进程刷盘/投递、GPU 故障时 watchdog 与实际时延/开销都待后续 smoke。首版保守等待固定截止，不启用提前排空结束。身份/去重记录按有界服务生命周期保留；未实现 PPO/机会实验/独立评估。工作区非持久卷，未上传外部服务。\n\n下一步仅交给 Astra 做 04 CPU 审查；GPU 命令必须等未来明确授权的 smoke 步骤再执行。03 已停止，不自动进入后续阶段。\n'''
    (RUN/'handoff/03.md').write_text(markdown)
    # Read-only final gate consumption uses the same validator as the future runner.
    spec=importlib.util.spec_from_file_location('_final_real_cli',ROOT/'benchmark/run_real_ppo.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    cli.validate_gate(RUN/'gates/p2_offline.json',RUN)
    cli.validate_gate(RUN/'gates/p1_offline_revalidated_03.json',RUN)
    assert not any((RUN/'gates'/name).exists() for name in ('p1_live.json','p2_live.json'))
    assert all(sha(RUN/path)==expected for path,expected in snapshots.items())
    print(json.dumps({'status':'passed','CPU_tests':63,'source_files_bound':len(source_map),
                      'gate':str(RUN/'gates/p2_offline.json'),'handoff':str(RUN/'handoff/03.md'),
                      'prepared_smoke_commands':4,'GPU_started':False,'next_stage_started':False},indent=2))


if __name__=='__main__':main()
