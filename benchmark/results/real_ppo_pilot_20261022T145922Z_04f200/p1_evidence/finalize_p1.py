"""Save reviewable P1 handoff and hash bindings; never launches services."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    original = json.loads((RUN/'source_manifest_before.json').read_text())
    review = json.loads((RUN/'p1_evidence/p0_review.json').read_text())
    assert review['status']=='passed' and all(review['checks'].values())
    review['source_hash_derivation'] = 'SHA256 of JSON(path->SHA256), sorted keys, compact separators, UTF-8'
    review['independently_verified_p0_source_hash'] = canonical({x['path']:x['sha256'] for x in original['source_inventory']})
    review['p0_source_manifest_sha256'] = sha(RUN/'source_manifest_before.json')
    review['p0_gate_sha256'] = sha(RUN/'gates/p0.json')
    write(RUN/'p1_evidence/p0_review.json',review)
    verification = json.loads((RUN/'p1_evidence/verification.json').read_text())
    assert verification['status']=='passed' and all(verification['checks'].values())
    budget = json.loads((RUN/'budget.json').read_text())
    assert budget['usage']['budget_clock_started'] is False
    sources = ['rl_scheduler/real_metrics.py','rl_scheduler/real_client.py',
               'rl_scheduler/tests/test_real_metrics.py','llumnix/entrypoints/vllm/real_metrics_stream.py',
               'llumnix/entrypoints/vllm/api_server.py','llumnix/metrics/causal_events.py']
    source_map = {p:sha(ROOT/p) for p in sources}
    # Pre-existing changes outside this explicitly authorized source set must survive.
    unchanged = [x['path'] for x in original['source_inventory'] if x['path'] not in source_map]
    assert all(sha(ROOT/x['path'])==x['sha256'] for x in original['source_inventory'] if x['path'] in unchanged)
    before = (RUN/'p1_evidence/git_status_before.bin').read_bytes().split(b'\0')
    current = subprocess.check_output(['git','status','--porcelain=v1','-z','--untracked-files=all'],cwd=ROOT)
    current_rows = current.split(b'\0')
    assert all(row in current_rows for row in before if row)
    (RUN/'p1_evidence/git_status_after.bin').write_bytes(current)
    (RUN/'p1_evidence/authorized_source.diff').write_bytes(subprocess.check_output(
        ['git','diff','--binary','HEAD','--',*sources],cwd=ROOT))
    snapshot_files=[]
    for path in sources:
        target=RUN/'source_snapshots/p1_after'/path
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/path,target)
        assert sha(target)==source_map[path]
        snapshot_files.append(str(target.relative_to(RUN)))
    inventory = {x['path']:x for x in original['source_inventory']}
    for path in sources:
        inventory[path]={'path':path,'sha256':source_map[path],'bytes':(ROOT/path).stat().st_size}
    manifest={'stage':'P1_offline','head':original['head'],'scope':'P0 inventory plus authorized P1 source additions/changes',
              'source_inventory':list(sorted(inventory.values(),key=lambda x:x['path'])),
              'relevant_source_hashes':source_map,'source_hash':canonical(source_map),
              'hash_derivation':'SHA256(JSON(path->SHA256), sort_keys=True, separators=(comma,colon), UTF-8)',
              'source_snapshots':snapshot_files,'preservation':{'unchanged_p0_inventory_entries':len(unchanged),
              'preexisting_git_status_entries_retained':True,'no_commit_push_reset_clean':True}}
    write(RUN/'source_manifest_after.json',manifest)
    cpu_elapsed=time.monotonic()-review['cpu_started_monotonic']
    accounting={'stage':'02','cpu_wall_time_sec_since_review':cpu_elapsed,
                'gpu_services_started':[],'gpu_experiments':0,'task_gpu_hours_this_step':0,
                'experiment_budget_changed':False,'remaining_budget':budget['remaining'],
                'reason':'GPU experiment clock has never started for this run; CPU work recorded separately.'}
    write(RUN/'p1_evidence/accounting.json',accounting)
    pointer=ROOT/'rl_scheduler/runs/real_ppo_handoff/active_run.json'
    p=json.loads(pointer.read_text()); assert Path(p['run_dir']).resolve()==RUN.resolve()
    p.update({'active_stage':'P1_offline','last_completed_step':'02','last_status':'passed',
              'stopped_after_requested_step':True,'last_handoff':str(RUN/'handoff/02.json')})
    write(pointer,p)
    reproduction='''# P1 CPU reproduction\n\nFrom the project root, run:\n\n```sh\n.venv/bin/python benchmark/results/real_ppo_pilot_20261022T145922Z_04f200/p1_evidence/verify_p1.py\n```\n\nThis runs 28 CPU unit/HTTP/ASGI tests, compilation, whitespace checks and a read-only GPU process inventory; exports the hand-computed cohort. It does not start Ray, an inference server, PPO, or a GPU experiment. Generated timestamps/log hashes change on rerun; revalidate gate bindings before consumption. Do not regenerate a passed gate without inspecting every result and the source/config/evidence hashes.\n\nInterfaces, exact formulas, clock domains, required producer/offered manifests, cleanup limitations and unverified live paths are documented in implementation.md. Historic P0 evidence remains under its original gate; p1_evidence/p0_review.json provides this step's independent verification and canonical source-hash supplement.\n'''
    (RUN/'reproduction.md').write_text(reproduction)
    evidence_paths=['p1_evidence/p0_review.json','p1_evidence/verification.json','p1_evidence/commands.json',
                    'p1_evidence/accounting.json','p1_evidence/unit_and_http.log','p1_evidence/compile.log',
                    'p1_evidence/diff_check.log','p1_evidence/gpu_inventory_only.log',
                    'p1_evidence/verify_p1.py','p1_evidence/finalize_p1.py',
                    'source_manifest_after.json','implementation.md','reproduction.md']
    evidence_paths += [str(p.relative_to(RUN)) for p in sorted((RUN/'p1_evidence/hand_calculated').glob('*'))]
    evidence_map={path:sha(RUN/path) for path in evidence_paths}
    gate={'stage':'P1_offline','status':'passed','run_id':p['run_id'],
          'created_at_utc':datetime.now(timezone.utc).isoformat(),
          'scope':'CPU metrics/event accounting implementation only; no live or research-performance acceptance',
          'source_hash':canonical(source_map),'source_hashes':source_map,
          'config_hash':sha(RUN/'experiment_spec.yaml'),'evidence_hashes':evidence_map,
          'hash_derivation':manifest['hash_derivation'],
          'checks':[{'name':'P0_actual_file_review','status':'passed','evidence':'p1_evidence/p0_review.json'},
                    {'name':'28_substantive_CPU_tests','status':'passed','evidence':'p1_evidence/unit_and_http.log'},
                    {'name':'hand_calculated_metrics_and_immutable_score','status':'passed','evidence':'p1_evidence/verification.json'},
                    {'name':'source_and_preexisting_status_preservation','status':'passed','evidence':'source_manifest_after.json'},
                    {'name':'compile_and_diff_check','status':'passed','evidence':'p1_evidence/commands.json'}],
          'artifacts':evidence_paths+snapshot_files,
          'reason':'Hand-computed request/window accounting, receipt-time observations, full-count streaming, bounded cancellation and durable incremental telemetry pass CPU tests. All live paths remain unverified.',
          'remaining_budget':budget['remaining'],
          'next_step_executable':{'P2_CPU_only':True,'GPU':False},
          'live_status':'not_run','p1_live_gate_generated':False,
          'blocking_before_GPU':['persistent budget expiry enforcement absent',
                                'cluster-wide producer inventory/flush and pending-delivery confirmation not live validated',
                                'live telemetry/bridge acceptance and stage dependencies required']}
    write(RUN/'gates/p1_offline.json',gate)
    commands=json.loads((RUN/'p1_evidence/commands.json').read_text())
    commands += [{'command':'.venv/bin/python benchmark/results/real_ppo_pilot_20261022T145922Z_04f200/p1_evidence/verify_p1.py','exit_code':0},
                 {'command':'python3 inline independent P0 file/hash/status review','exit_code':0},
                 {'command':'rg --files -g AGENTS.md /workspace','exit_code':1,'interpretation':'no additional AGENTS.md file returned; explicit /workspace/AGENTS.md was read in full'}]
    handoff={'step':'02','status':'passed','stage':'P1_offline','run_id':p['run_id'],
             'scope':gate['scope'],'changed_source_files':sources,
             'changed_artifact_files':sorted(set(evidence_paths+snapshot_files+['gates/p1_offline.json','handoff/02.json','handoff/02.md'])),
             'pointer_updated':str(pointer),'source_hashes':source_map,'source_hash':gate['source_hash'],
             'config_hash':gate['config_hash'],'gate_sha256':sha(RUN/'gates/p1_offline.json'),
             'active_pointer_sha256':sha(pointer),'evidence_hashes':evidence_map,
             'actual_commands_and_exit_codes':commands,'tests':verification,
             'p0_review':{'status':'passed','checks':review['checks'],'supplement':review['source_hash_derivation']},
             'remaining_budget':budget['remaining'],'accounting':accounting,
             'unresolved':['No GPU/live P1 acceptance; GPU count 0, real migrations 0, PPO updates 0.',
                           'Persistent budget expiry enforcer remains absent; exact live sampling/protocol freeze pending.',
                           'Cluster-wide producer flush RPCs, pending-delivery counters and migrated-request cancellation require live validation.',
                           'New route per-request records retained for serving process lifetime; future bounded episode lifecycle needs safe release/rotation.',
                           'Existing internal failover/retry paths are unchanged; future runner must invalidate faulted/failover episodes.',
                           'Workspace is not a persistent volume; no off-box upload performed.'],
             'next_step_executable':{'P2_CPU_only':True,'GPU':False},
             'stop':'Requested step completed; no P2, p1_live, cluster, PPO or automatic goal started.'}
    write(RUN/'handoff/02.json',handoff)
    text=f'''# 02 — P1 请求台账与评分语义\n\n状态：**passed（仅 P1 offline）**。28 个实质性 CPU 测试、编译和 diff 检查通过；没有启动 GPU、集群、PPO 或下一阶段，没有生成 p1_live 通过结论。\n\n范围：真实事件/请求台账、客户端完整计数流、单调时钟发生与接收时间、截止冻结、独立取消确认、增量游标、序列完整性、fsync 水位和导出审计。源码修改：\n\n'''
    text+=''.join(f'- `{path}` — SHA256 `{digest}`\n' for path,digest in source_map.items())
    text+=f'''\nP0 实际复核：14 项全部通过。原 172 个源码清单、6 份未跟踪快照、压缩 diff、配置与证据均一致；P0 不明的 aggregate source_hash 推导已用明确的 canonical file-map hash 补充。原修改、未跟踪及删除状态保留；没有提交或推送。证据：`p1_evidence/p0_review.json` 和 `source_manifest_after.json`。\n\n主要命令及退出码：\n\n- `.venv/bin/python -m unittest rl_scheduler.tests.test_real_metrics -v` → 0（28 tests）\n- `.venv/bin/python -m py_compile <六个相关源码>` → 0\n- `git diff --check` → 0\n- `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader` → 0，输出为空\n- `.venv/bin/python {RUN.relative_to(ROOT)}/p1_evidence/verify_p1.py` → 0\n\n完整参数/退出码与日志 hash：`p1_evidence/commands.json`；复现说明：`reproduction.md`。手算案例：4 offered / 3 sent / 2 窗内完成 / 1 未发送，8 秒内完成率 0.5、goodput 0.25 rps、客户端可见 token 7、outstanding 积分 11 请求秒、gap 积分 5 请求秒；清理迟到完成 1，冻结分数完全不变。原始导出在 `p1_evidence/hand_calculated/`。\n\n离线 gate：`gates/p1_offline.json`；gate SHA256 `{handoff['gate_sha256']}`；source hash `{gate['source_hash']}`；config hash `{gate['config_hash']}`。gate 绑定相关源码、配置、测试日志和导出证据；JSON 交接保存完整 hash 和命令。\n\nCPU 编码/验证墙钟约 {cpu_elapsed:.1f} 秒（从本步 P0 复核开始计，不含前面的阅读准备）；GPU 实验墙钟与 GPU 小时均为 0。run 的首次 GPU 服务尚未启动，原预算没有重置：总剩余 28800 秒，P0–P3 7200 秒，P4 7200 秒，P5 14400 秒。CPU 计时独立保存在 `p1_evidence/accounting.json`。\n\n遗留：持久预算到期强制机制未实现；实际采样参数/协议冻结仍待完成；完整集群 producer 发现/刷盘 RPC、投递清空、实际流时序/开销、迁移中取消均未真机验证。新接口当前按服务进程保留身份记录；未来 runner 需安全释放/轮换回合记录。旧内部 failover/retry 路径没有替换，介入时未来 runner 必须判回合 invalid。工作区非持久卷，未上传外部服务。\n\n下一步：仅 P2 CPU 实现可在复核 hash 后接手；GPU 仍受上述阻塞和 live gate 约束。本次已停止，不自动进入 P2。\n'''
    (RUN/'handoff/02.md').write_text(text)
    # Final inspection is read-only, including every gate binding.
    assert all(sha(ROOT/path)==digest for path,digest in gate['source_hashes'].items())
    assert sha(RUN/'experiment_spec.yaml')==gate['config_hash']
    assert all(sha(RUN/path)==digest for path,digest in gate['evidence_hashes'].items())
    assert not (RUN/'gates/p1_live.json').exists()
    assert sha(RUN/'budget.json')==json.loads((RUN/'gates/p0.json').read_text())['captured_evidence_hashes']['budget_json_sha256']
    print(json.dumps({'status':'passed','gate':str(RUN/'gates/p1_offline.json'),
                      'handoff':str(RUN/'handoff/02.md'),'source_files':len(sources),
                      'tests':28,'source_hash':gate['source_hash'],'remaining_budget':budget['remaining'],
                      'stopped_after_step':'02'},indent=2))


if __name__ == '__main__':
    main()
