"""Hand-computed rewards from complete client ledgers, including cleanup censoring."""
from copy import deepcopy
from dataclasses import asdict
import math
import unittest

from rl_scheduler.real_metrics import NS
from rl_scheduler.real_reward import (RewardWeights,RewardScales,reconstruct_rewards,fit_reference_scales,reward_scale_diagnostics)
from rl_scheduler.tests.test_real_metrics import Fixture


class RewardTests(unittest.TestCase):
    def fixture(self):
        f=Fixture();f.plan_send('a',0);f.output(1,1,'a');f.output(1.3,4,'a');f.terminal(2,'a')
        f.plan_send('b',1);f.plan_send('c',2);f.output(3,2,'c');f.terminal(5,'c',outcome='service_error',http_status=500)
        f.add('planned',0,'not_sent',scheduled_ns=3*NS);f.ack(8)
        f.ledger.freeze_score();f.output(9,2,'b');f.terminal(10,'b');f.ack(10)
        return f

    def compute(self,f,edges=None,migrations=None,**kw):
        return reconstruct_rewards(f.ledger,edges or [0,2*NS,4*NS,8*NS],migrations or [],
            scales=kw.pop('scales',RewardScales(2.,2.,'hand-computed-fixture-only')),
            cleanup_through_ns=10*NS,valid_episode=kw.pop('valid_episode',True),**kw)

    def test_hand_calculated_counts_integrals_and_tail(self):
        f=self.fixture();saved=deepcopy(f.ledger.freeze_score())
        migrations=[{'decision_id':'d1','migration_id':'m1','attempt_ns':NS//2,'terminal_ns':2*NS,'terminal_status':'failed'},
                    {'decision_id':'d2','migration_id':None,'attempt_ns':None,'terminal_ns':4*NS,'terminal_status':'rejected'},
                    {'decision_id':'d3','migration_id':'m3','attempt_ns':7*NS,'terminal_ns':9*NS,'terminal_status':'succeeded'}]
        rows=self.compute(f,migrations=migrations)
        raw={key:sum(row['raw'][key] for row in rows) for key in rows[0]['raw']}
        self.assertEqual(raw,dict(good=1,hold=12.,first=5.,gap=1.,attempt=2,bad=2,tail=1))
        self.assertAlmostEqual(sum(r['reward'] for r in rows),-3.49)
        self.assertEqual([r['raw']['good'] for r in rows],[0,1,0])
        self.assertEqual([r['raw']['bad'] for r in rows],[0,1,1])
        self.assertEqual([r['raw']['tail'] for r in rows],[0,0,1])
        self.assertEqual(f.ledger.freeze_score(),saved)
        self.assertFalse(any(r['clipped'] for r in rows))

    def test_variable_windows_partition_same_reward_without_double_count(self):
        f=self.fixture();a=self.compute(f);b=self.compute(f,[0,NS//3,NS,2*NS,7*NS,8*NS])
        self.assertAlmostEqual(sum(row['reward'] for row in a),sum(row['reward'] for row in b))
        self.assertAlmostEqual(sum(row['dt_sec'] for row in b),8.)
        for key in a[0]['raw']:
            self.assertAlmostEqual(sum(row['raw'][key] for row in a),sum(row['raw'][key] for row in b))

    def test_deadline_completion_counted_once_and_single_token_TPOT_not_needed(self):
        f=Fixture();f.plan_send();f.output(1,1);f.terminal(8);f.ack(10)
        rows=self.compute(f)
        self.assertEqual(sum(r['raw']['good'] for r in rows),1)
        self.assertEqual(rows[-1]['raw']['tail'],0)
        self.assertIsNone(f.ledger.freeze_score()['fixed']['tpot_p95_ms'])

    def test_duplicate_cumulative_output_does_not_reset_gap_timer(self):
        f=Fixture();f.plan_send();f.output(1,1);f.output(3,1);f.terminal(5);f.ack(10)
        rows=self.compute(f)
        self.assertEqual(sum(r['raw']['gap'] for r in rows),3.)

    def test_pressure_not_saturated(self):
        f=Fixture()
        for i in range(10):f.plan_send(str(i));f.output(1,1,str(i));f.terminal(8,str(i))
        f.ack(10);row=self.compute(f,[0,8*NS],scales=RewardScales(1,1,'fixture'))[0]
        self.assertEqual(row['normalized']['hold'],80.)
        self.assertEqual(row['weighted']['hold'],-16.)

    def test_unknown_missing_flush_pending_and_invalid_episode_rejected(self):
        f=self.fixture()
        with self.assertRaises(ValueError):self.compute(f,valid_episode=False)
        with self.assertRaises(ValueError):self.compute(f,scales=None)
        for pending,durable in [(1,True),(None,True),(0,False)]:
            f=self.fixture();f.ack(10,pending=pending,durable=durable)
            with self.assertRaises(ValueError):self.compute(f)
        f=Fixture();f.plan_send();f.output(1,1);f.terminal(2,outcome='unknown');f.ack(10)
        with self.assertRaises(ValueError):self.compute(f)

    def test_ambiguous_migration_terminal_and_duplicate_attempt_rejected(self):
        f=self.fixture();record=dict(decision_id='d',migration_id='m',attempt_ns=NS,terminal_ns=2*NS,terminal_status='unknown')
        with self.assertRaises(ValueError):self.compute(f,migrations=[record])
        record['terminal_status']='succeeded'
        with self.assertRaises(ValueError):self.compute(f,migrations=[record,dict(record,decision_id='other')])
        record['attempt_ns']=None
        with self.assertRaises(ValueError):self.compute(f,migrations=[record])

    def test_partition_and_weights_validation(self):
        f=self.fixture()
        for edges in ([1,8*NS],[0,4*NS,4*NS,8*NS],[0,9*NS],[0.,8*NS]):
            with self.assertRaises(ValueError):self.compute(f,edges)
        for value in (float('nan'),float('inf'),-1.):
            with self.assertRaises(ValueError):RewardWeights(hold=value)
        with self.assertRaises(ValueError):RewardScales(0,1,'x')

    def test_scales_only_healthy_Stay_opportunity_not_final_or_Balanced(self):
        def episode(name,phase,policy,c,n,duration=240,status='passed',health=True):
            return dict(episode_id=name,phase=phase,policy=policy,status=status,health=health,
                metrics=dict(completed=c,outstanding_request_seconds=n,duration_sec=duration))
        rows=[episode('a','opportunity','Stay',480,960),episode('b','opportunity','Stay',240,480),
              episode('bad-health','opportunity','Stay',1000,999,health=False),
              episode('balanced','opportunity','Balanced',999999,999999),episode('final','final','Stay',999999,999999),
              episode('probe','probe','Stay',999999,999999)]
        scales,details=fit_reference_scales(rows,healthy=lambda row:row['health'])
        self.assertEqual((scales.C_ref,scales.N_ref),(1.5,3.))
        self.assertEqual(details['episode_ids'],['a','b']);self.assertEqual(details['excluded_unhealthy_Stay_ids'],['bad-health'])
        rows[0]['status']='incomplete'
        with self.assertRaises(ValueError):fit_reference_scales(rows,healthy=lambda row:True)
        with self.assertRaises(ValueError):fit_reference_scales([],healthy=lambda row:True)

    def test_measured_reference_floor_is_explicit(self):
        row={'episode_id':'tiny','phase':'opportunity','policy':'Stay','status':'passed',
             'metrics':{'duration_sec':240,'completed':1,'outstanding_request_seconds':20}}
        scales,evidence=fit_reference_scales([row],healthy=lambda row:True)
        self.assertEqual((scales.C_ref,scales.N_ref),(1,1));self.assertEqual(evidence['raw_C_ref_rps'],1/240)


if __name__=='__main__':unittest.main(verbosity=2)
