import asyncio
from copy import deepcopy
import inspect
import unittest
from unittest.mock import patch
from pathlib import Path
from benchmark.run_p0p1_fast import (MeasurementStay, can_start, opportunity_stats, p0_gate,
                                     pair_gate, candidate_gate, validate_config)
from rl_scheduler.full_completion import make_trace, run_episode
from rl_scheduler.real_bridge import digest
from rl_scheduler.cohort_ppo import build_rollout, normalizer_definition


def fixture():
    o=dict(N_move=0,succeeded=30,started=30,coverage=.2,success_per_minute=6.,success_per_started=1.,N_opp=100,opportunity_ratio=.5)
    stats=dict(mean=100.,p95=120.,cvar95=130.)
    return dict(status='passed',execution_audit={'status':'passed'},cohort_complete=True,measured_request_count=128,
                trace_sha256='same',frozen_mapping=['same'],opportunity=o,
                primary_metrics={k:dict(stats) for k in ('client_e2e_ms','ttft_ms','tpot_ms')})


class FastTest(unittest.TestCase):
    def test_trace_and_config(self):
        from tokenizers import Tokenizer
        import yaml
        root=Path(__file__).resolve().parents[2]
        tok=Tokenizer.from_file(str(root/'model/Qwen/Qwen2-1.5B/tokenizer.json'))
        kw=dict(candidate=dict(family='S1',profile='decode1536',rho=1.,hot_fraction=.9),n=128,seed=141101,lambda_ref=1.,tokenizer=tok)
        a,b=make_trace(**kw),make_trace(**kw)
        self.assertEqual(digest(a),digest(b))
        self.assertEqual(sum(r['initial_instance_index']==1 for r in a),115)
        self.assertTrue(all(len(tok.encode(r['prompt']).ids)==128 and r['output_tokens']==1536 for r in a))
        cfg=yaml.safe_load((root/'rl_scheduler/configs/real_pilot.yaml').read_text())
        srv=yaml.safe_load((root/'rl_scheduler/configs/real_llumnix.yaml').read_text())
        cfg['service']['ray_num_cpus']=16;cfg['service']['server_argv']+=['--max-num-batched-tokens','2112']
        srv['SERVER']['REQUEST_OUTPUT_QUEUE_TYPE']='rayqueue'
        validate_config(cfg,srv)
        cfg['service']['server_argv'][-1]='4096'
        with self.assertRaises(ValueError): validate_config(cfg,srv)

    def test_network_measurement_cannot_train(self):
        from rl_scheduler.real_observation import INSTANCE_FEATURES,PAIR_FEATURES,GLOBAL_FEATURES
        raw=dict(instance_feature_names=list(INSTANCE_FEATURES),pair_feature_names=list(PAIR_FEATURES),global_feature_names=list(GLOBAL_FEATURES),
                 instance_features=[[0.]*12]*4,instance_missing=[[False]*12]*4,pair_features=[[0.]*3]*12,pair_missing=[[False]*3]*12,
                 global_features=[0.]*11,global_missing=[False]*11,action_mask=[True]*13,decision_ns=1)
        c=MeasurementStay();r=c(raw,0)
        self.assertEqual(r['action'],0);self.assertEqual(r['executed_action'],0)
        self.assertIn('suggested_action',r);self.assertNotIn('old_logp',r)
        self.assertTrue(all(p.device.type=='cpu' for p in c.net.parameters()))
        with self.assertRaisesRegex(ValueError,'measurement-only'):
            build_rollout({},[{'ppo':r}],None)

    def test_counts_and_dedup(self):
        ds=[]
        for i,(mask,action) in enumerate([([True]+[False]*12,0),([True]*13,0),([True]*13,1)]):
            ds.append(dict(monotonic_ns=i+1,action_mask=mask,sampled_action=action,snapshot={'instances':[], 'mapping':[], 'busy':True} if i==0 else {},receipt={}))
        # Use a legal complete saved snapshot for reason derivation.
        ds[0]['snapshot']={'instances':[dict(instance_id=str(i)) for i in range(4)],'mapping':[[str(i),0] for i in range(4)],'busy':True}
        r=dict(migration_id='x',terminal_status='succeeded',actual_request_id='r',migration_start_ns=1)
        o=opportunity_stats(ds,{'a':r,'b':r},[dict(sent_ns=1,client_terminal_ns=60_000_000_001)],128)
        self.assertEqual((o['N_all'],o['N_opp'],o['N_forced'],o['N_voluntary'],o['N_move']),(3,2,1,1,1))
        self.assertEqual(o['succeeded'],1);self.assertEqual(o['coverage'],1/128)
        self.assertIsNone(opportunity_stats([],{},[])['move_per_opportunity'])

    def test_gates(self):
        a=fixture();p=fixture();b=fixture()
        for e in (a,p,b): e['opportunity']['succeeded']=0
        self.assertTrue(p0_gate([a,p,b])['passed'])
        p['primary_metrics']['client_e2e_ms']['mean']=150
        self.assertFalse(p0_gate([a,p,b])['passed'])
        bal=fixture()
        for k in bal['primary_metrics']:
            bal['primary_metrics'][k]={n:v*.9 for n,v in bal['primary_metrics'][k].items()}
        pair=pair_gate([a,b],bal)
        self.assertTrue(pair['passed']);self.assertTrue(candidate_gate({'passed':True},[pair,pair]))
        self.assertFalse(pair_gate([],bal)['passed'])
        bal['status']='failed';self.assertFalse(pair_gate([a],bal)['passed'])
        a['primary_metrics']['client_e2e_ms']['mean']=0;a['primary_metrics']['client_e2e_ms']['cvar95']=0
        self.assertFalse(pair_gate([a],fixture())['passed'])
        self.assertFalse(pair_gate([fixture()],fixture())['passed'])

    def test_progress_survives_rolling_window_eviction(self):
        from types import SimpleNamespace
        from rl_scheduler.full_completion import cumulative_output_progress
        ledger=SimpleNamespace(_online={'a': {'token_high': 128, 'token_increments': []}, 'b': {'token_high': 64, 'token_increments': []}})
        self.assertEqual(cumulative_output_progress(ledger),192)
        ledger._online['a']['token_high']=192
        self.assertEqual(cumulative_output_progress(ledger),256)

    def test_budget_and_threshold(self):
        self.assertTrue(can_start(0));self.assertFalse(can_start(2221));self.assertFalse(can_start(2100,[650]))
        self.assertEqual(inspect.signature(run_episode).parameters['balanced_threshold'].default,9000.)
        self.assertIn('threshold=balanced_threshold',inspect.getsource(run_episode))

    def test_service_cleanup_even_on_startup_failure(self):
        # Exercise actual outer run/finally with startup failure, no GPU service.
        import tempfile
        import yaml
        import benchmark.run_p0p1_fast as mod
        root=Path(__file__).resolve().parents[2]
        cfg=yaml.safe_load((root/'rl_scheduler/configs/real_pilot.yaml').read_text())
        srv=yaml.safe_load((root/'rl_scheduler/configs/real_llumnix.yaml').read_text())
        cfg['service']['ray_num_cpus']=16;cfg['service']['server_argv']+=['--max-num-batched-tokens','2112'];srv['SERVER']['REQUEST_OUTPUT_QUEUE_TYPE']='rayqueue'
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);(d/'plan.json').write_text('{"code_hashes":{},"config_sha256":"x"}')
            (d/'runtime_config.yaml').write_text(yaml.safe_dump(cfg));(d/'real_llumnix.yaml').write_text(yaml.safe_dump(srv))
            plan={'code_hashes':{},'config_sha256':__import__('hashlib').sha256((d/'runtime_config.yaml').read_bytes()).hexdigest()}
            (d/'plan.json').write_text(__import__('json').dumps(plan))
            with patch.object(mod,'environment',return_value={}),patch.object(mod,'analyze'),patch.object(mod,'ManagedFullCompletionService') as svc:
                svc.return_value.start.side_effect=RuntimeError('fixture startup failure')
                asyncio.run(mod.run(d));svc.return_value.stop.assert_called_once()

if __name__=='__main__': unittest.main()
