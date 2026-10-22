"""Minimal CPU reproduction of the initial missing-arrival preparation error."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from disturbance_exp.workload import generate_trace
cfg={'experiment':{'seed':6101,'duration_sec':20},'workload':{'base_qps':.2,'input_tokens':16,
    'output_tokens':4,'max_model_len':2112},'limits':{'max_requests':20}}
try:
    generate_trace(cfg,tokenizer=None)
except KeyError as exc:
    assert str(exc)=="'arrival'"
    print('Reproduced missing arrival contract. Production caller now supplies arrival=poisson.')
else:
    raise AssertionError('expected original contract error')
