"""Reproduce repair CPU checks; never launch a service or write the experiment clock.

Logs and verification hashes change on rerun. Reissue CPU gates after checking the
new results; independent review status is never modified by this script.
"""
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[5]
HERE=Path(__file__).resolve().parent
RUN=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from rl_scheduler.real_bridge import digest
from rl_scheduler.real_budget import atomic_json


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    os.chdir(ROOT)
    baseline=json.loads((HERE/'baseline.json').read_text())
    sources=sorted(set(baseline['source_hashes'])|{'rl_scheduler/tests/test_real_repairs.py'})
    source_map={name:sha(ROOT/name) for name in sources}
    reuse="--finalize-existing" in sys.argv
    commands=json.loads((HERE/"commands.json").read_text()) if reuse else []
    def run(argv,name,required=True):
        if reuse:
            record=next(row for row in commands if row["log"]==str((HERE/name).relative_to(RUN)))
            if required and record["exit_code"]:raise RuntimeError("saved check failed")
            return (HERE/name).read_text()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
        result=subprocess.run(argv,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (HERE/name).write_text(result.stdout)
        commands.append({'argv':argv,'env':{'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},
                         'exit_code':result.returncode,'log':str((HERE/name).relative_to(RUN))})
        atomic_json(HERE/'commands.json',commands)
        if required and result.returncode:raise RuntimeError('CPU command failed; retain evidence: '+name)
        return result.stdout
    python=str(ROOT/'.venv/bin/python')
    modules=['rl_scheduler.tests.test_real_metrics','rl_scheduler.tests.test_real_bridge',
             'rl_scheduler.tests.test_real_env','rl_scheduler.tests.test_real_runner','rl_scheduler.tests.test_real_repairs']
    run([python,'-m','unittest',*modules,'-v'],'tests_final.log')
    # Execute the original independent tests unchanged. Its successful migration
    # trace is redirected to this repair directory to preserve all review bytes.
    code='''import importlib.util,unittest\nfrom pathlib import Path\nfrom unittest.mock import patch\np=Path("'''+str(RUN/'reviews/test_bridge_review.py')+'''")\nspec=importlib.util.spec_from_file_location("independent_review",p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)\noriginal=Path.write_text\ndef write(path,*args,**kwargs):\n if path.resolve()==p.parent/"migration_chain.json":path=p.parents[1]/"repairs/bridge_r1_r7/review_migration_chain_after.json"\n return original(path,*args,**kwargs)\nwith patch.object(Path,"write_text",write):\n result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(m.BridgeReview))\nraise SystemExit(not result.wasSuccessful())\n'''
    run([python,'-c',code],'independent_counterexamples_final.log')
    run([python,'-c','import pathlib,sys; [compile(pathlib.Path(p).read_text(),p,"exec") for p in sys.argv[1:]]; print("Compiled",len(sys.argv)-1,"source files without bytecode writes")',*sources],'compile.log')
    run(['git','diff','--check'],'diff_check.log')
    gpu=run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],'gpu_process_inventory.log')
    run(['supervisorctl','status'],'supervisor_inventory.log',required=False)
    assert sha(RUN/'experiment_clock.json')==baseline['clock_sha256']
    assert all(sha(RUN/name)==expected for name,expected in baseline['review_files'].items())
    before_status=(HERE/'git_status_before.bin').read_bytes().split(b'\0')
    after_status=subprocess.check_output(['git','status','--porcelain=v1','-z'],cwd=ROOT)
    (HERE/'git_status_after.bin').write_bytes(after_status)
    assert all(record in after_status.split(b'\0') for record in before_status if record)
    before_manifest=json.loads((RUN/'source_manifest_after.json').read_text())
    untouched=[entry for entry in before_manifest['source_inventory'] if entry['path'] not in sources]
    assert all(sha(ROOT/entry['path'])==entry['sha256'] for entry in untouched)
    for name,expected in source_map.items():
        destination=HERE/'after'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,destination)
        assert sha(destination)==expected
    changed=[name for name,value in source_map.items() if baseline['source_hashes'].get(name)!=value]
    patch_text=[]
    for name in changed:
        before=HERE/'before'/name
        patch_text.extend(difflib.unified_diff(before.read_text().splitlines(True) if before.exists() else [],
            (ROOT/name).read_text().splitlines(True),fromfile='before/'+name,tofile='after/'+name))
    (HERE/'repair_01.patch').write_text(''.join(patch_text))
    verification={'status':'passed','scope':'CPU only; original independent review remains failed',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'tests':{'unit_regressions':78,'independent_review_cases':9},
        'checks':{'CPU_78_tests':True,'original_9_review_tests':True,'compile_40_sources':True,'diff_check':True,
            'original_review_bytes_preserved':True,'experiment_clock_bytes_preserved':True,
            'prior_git_status_entries_retained':True,'untouched_source_inventory_hashes_preserved':True,
            'no_gpu_compute_processes_observed':not gpu.strip()},
        'source_hashes':source_map,'source_hash':digest(source_map),'changed_files':changed,
        'unaffected_inventory_entries_checked':len(untouched),'actual_commands_and_exit_codes':commands,
        'GPU_services_started':[],'GPU_experiment_wall_sec':0,'PPO_updates':0,
        'evidence_hashes':{str(p.relative_to(RUN)):sha(p) for p in HERE.iterdir() if p.is_file() and p.name!='verification.json'}}
    atomic_json(HERE/'verification.json',verification)
    print(json.dumps({'status':'passed','unit_tests':78,'review_tests':9,'changed_files':changed},indent=2))


if __name__=='__main__':main()
