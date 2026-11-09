import copy
import math
import tempfile
import unittest
from pathlib import Path
import torch
from rl_scheduler.cohort_ppo import (CohortActorCritic,FrozenController,normalizer_definition,causal_observation,
                                     tensors,action_classification,ppo_update,build_rollout)
from rl_scheduler.cohort_training import policy_hash,save_checkpoint
from rl_scheduler.tests.test_cohort_ppo import observation,summary
from rl_scheduler.cohort_reward import CohortReward

SETTINGS=dict(actor_sample_mode='choice_only',update_epochs=1,minibatch_size=64,clip_epsilon=.2,
              value_coefficient=.5,entropy_coefficient=.01,max_grad_norm=.5,target_kl=.03)

class ChoiceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(151001);self.p=CohortActorCritic(16);self.norm=normalizer_definition(2.)

    def batch(self,patterns):
        c=FrozenController(self.p,self.norm,0);rs=[]
        for i,(legal,action,adv) in enumerate(patterns):
            raw=observation();raw['action_mask']=[True]+[j<legal for j in range(12)]
            saved=c(raw,0);saved['action']=action;saved.update(action_classification(raw['action_mask'],action))
            with torch.no_grad():lp,_,_=self.p.evaluate(tensors([saved['observation']]),torch.tensor([action]))
            saved.update(old_logp=lp.item(),return_target=1.+i,advantage=adv);rs.append(saved)
        return [dict(complete=True,policy_version=0,records=rs)]

    def test_forced_direct_actor_gradient_exact_zero(self):
        b=self.batch([(0,0,2.)]);r=b[0]['records'][0]
        lp,ent,value=self.p.evaluate(tensors([r['observation']]),torch.tensor([0]))
        self.assertEqual(lp.item(),0.);self.assertEqual(ent.item(),0.)
        (-lp.mean()*2).backward()
        self.assertTrue(all(p.grad is None or torch.count_nonzero(p.grad)==0 for p in self.p.parameters()))

    def test_voluntary_kept_and_forced_does_not_change_actor_normalization(self):
        base=self.batch([(1,0,-1.),(1,1,1.)]);extended=copy.deepcopy(base)
        extended[0]['records']+=self.batch([(0,0,1e5)])[0]['records']
        a=copy.deepcopy(self.p);b=copy.deepcopy(self.p);settings=dict(SETTINGS,value_coefficient=0.,entropy_coefficient=0.,max_grad_norm=100.)
        ra=ppo_update(a,torch.optim.SGD(a.parameters(),lr=.01),base,settings,seed=1)
        rb=ppo_update(b,torch.optim.SGD(b.parameters(),lr=.01),extended,settings,seed=1)
        self.assertEqual(ra['actor_eligible_count'],2);self.assertEqual(rb['actor_eligible_count'],2)
        self.assertEqual(rb['critic_sample_count'],3)
        for pa,pb in zip(a.parameters(),b.parameters()):self.assertTrue(torch.allclose(pa,pb,atol=1e-7))
        self.assertNotEqual(policy_hash(a),policy_hash(self.p))

    def test_all_forced_critic_updates_without_nan(self):
        batch=self.batch([(0,0,1.),(0,0,2.)]);old=self.p.stay.weight.detach().clone()
        r=ppo_update(self.p,torch.optim.Adam(self.p.parameters(),lr=.001),batch,dict(SETTINGS,minibatch_size=1),seed=1)
        self.assertEqual(r['actor_status'],'no_actor_update');self.assertEqual(r['actor_sample_count'],0)
        self.assertTrue(torch.equal(old,self.p.stay.weight))
        self.assertTrue(all(math.isfinite(m['loss']) for m in r['metrics']))

    def test_forced_time_stays_in_complete_return(self):
        c=FrozenController(self.p,self.norm,0);ds=[]
        for i,t in enumerate([0,3_000_000_000,6_000_000_000]):
            raw=observation();raw['decision_ns']=t;raw['action_mask']=[True]+[False]*12
            ds.append({'ppo':c(raw,0)})
        r=build_rollout(summary(),ds,CohortReward(2.))
        self.assertEqual(len(r['records']),3);self.assertAlmostEqual(r['records'][0]['return_target'],-5.25)
        self.assertTrue(all(x['action_class']=='forced_stay' for x in r['records']))
        self.assertTrue(r['records'][-1]['terminated']);self.assertFalse(r['records'][-1]['truncated'])

    def test_measurement_and_mixed_within_rollout_rejected(self):
        b=self.batch([(1,0,1.)]);b[0]['records'][0]['measurement_only']=True
        with self.assertRaises(ValueError):ppo_update(self.p,torch.optim.Adam(self.p.parameters()),b,SETTINGS,seed=1)
        b=self.batch([(1,0,1.)]);b[0]['records'][0]['policy_version']=2
        with self.assertRaises(ValueError):ppo_update(self.p,torch.optim.Adam(self.p.parameters()),b,SETTINGS,seed=1)

    def test_legacy_mode_and_checkpoint_predictions(self):
        batch=self.batch([(1,0,-1.),(1,1,1.)]);settings=dict(SETTINGS);settings.pop('actor_sample_mode')
        opt=torch.optim.Adam(self.p.parameters(),lr=.001);r=ppo_update(self.p,opt,batch,settings,seed=1)
        self.assertEqual(r['actor_sample_mode'],'all')
        obs=batch[0]['records'][0]['observation']
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'c.pt';save_checkpoint(p,self.p,opt,next_index=1,version=1,freeze_sha256='test',config={'scope':'synthetic'},normalizer=self.norm)
            saved=torch.load(p,weights_only=False);other=CohortActorCritic(16);other.load_state_dict(saved['policy'])
            with torch.no_grad():
                a,v=self.p.distribution(*tensors([obs]));b,w=other.distribution(*tensors([obs]))
            self.assertTrue(torch.equal(a.probs,b.probs));self.assertTrue(torch.equal(v,w))

if __name__=='__main__':unittest.main()
