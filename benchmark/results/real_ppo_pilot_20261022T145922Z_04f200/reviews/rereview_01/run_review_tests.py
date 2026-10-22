import importlib.util,json,os,subprocess,sys,time,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
HERE=Path(__file__).resolve().parent;RUN=HERE.parents[1]
commands=[]
started=time.monotonic()
spec=importlib.util.spec_from_file_location('original_review',RUN/'reviews/test_bridge_review.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
original=Path.write_text
def redirected(path,*args,**kwargs):
 if path.resolve()==(RUN/'reviews/migration_chain.json').resolve():path=HERE/'migration_chain.json'
 return original(path,*args,**kwargs)
with (HERE/'original_9.log').open('w') as log,patch.object(Path,'write_text',redirected):
 result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
commands.append(dict(command='original review suite via run_review_tests.py (migration_chain output redirected)',exit_code=0 if result.wasSuccessful() else 1,tests=result.testsRun,elapsed_sec=time.monotonic()-started,log='reviews/rereview_01/original_9.log'))
argv=[sys.executable,'-m','unittest','rl_scheduler.tests.test_real_repairs','-v'];started=time.monotonic()
with (HERE/'repair_regressions.log').open('w') as log:cp=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
commands.append(dict(argv=argv,exit_code=cp.returncode,elapsed_sec=time.monotonic()-started,log='reviews/rereview_01/repair_regressions.log'))
manifest=json.loads((RUN/'commands/05_smoke.json').read_text())
for entry in manifest['commands']:
 argv=entry['validation_argv'];started=time.monotonic();cp=subprocess.run(argv,capture_output=True,text=True,env={**os.environ,**entry.get('env',{})})
 name='dryrun_'+entry['label']+'.log';(HERE/name).write_text(cp.stdout+cp.stderr)
 commands.append(dict(argv=argv,exit_code=cp.returncode,elapsed_sec=time.monotonic()-started,log='reviews/rereview_01/'+name))
(HERE/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
print(json.dumps(commands,indent=2))
