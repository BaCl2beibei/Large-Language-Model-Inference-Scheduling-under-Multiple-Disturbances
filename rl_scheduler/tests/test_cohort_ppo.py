import copy
import math
from pathlib import Path
import tempfile
import unittest

import torch

from rl_scheduler.cohort_reward import CohortReward, complete_returns
from rl_scheduler.cohort_ppo import (CohortActorCritic, FrozenController, build_rollout,
    causal_observation, normalizer_definition, ppo_update, tensors, policy_hash, save_checkpoint)
from rl_scheduler.real_observation import RealObservationBuilder
from rl_scheduler.tests.test_real_bridge import Clock, states
from types import SimpleNamespace


SETTINGS = {'update_epochs': 3, 'minibatch_size': 4, 'clip_epsilon': .2,
    'value_coefficient': .5, 'entropy_coefficient': .01, 'max_grad_norm': .5, 'target_kl': .03}


def summary():
    return {'status': 'passed', 'cohort_complete': True, 'audit': {'status': 'passed'},
        'route_audit_passed': True, 'planned_request_count': 2, 'episode_id': 'test',
        'requests': [dict(request_id='a', sent_ns=1_000_000_000, client_terminal_ns=4_000_000_000,
                          actual_output_tokens=10, expected_output_tokens=10),
                     dict(request_id='b', sent_ns=2_000_000_000, client_terminal_ns=8_000_000_000,
                          actual_output_tokens=10, expected_output_tokens=10)]}


def observation(now=100_000_000_000):
    clock = Clock()
    measured = states(clock)
    snapshot = {'mapping': [[s['instance_id'], s['epoch']] for s in measured],
                'instances': measured, 'received_ns': now, 'version': 'test', 'busy': False}
    return RealObservationBuilder(snapshot['mapping']).build(snapshot, now_ns=now,
        ledger=SimpleNamespace(observe=lambda t: {'requests': []}), episode_start_ns=now-1_000_000_000,
        arrival_end_ns=now+1_000_000_000, deadline_ns=now+2_000_000_000)


class RewardTests(unittest.TestCase):
    def test_integral_and_terminal_tail_equal_exact_objective(self):
        result = CohortReward(2.).reconstruct(summary(), [0, 3_000_000_000, 6_000_000_000])
        self.assertEqual(result['mean_e2e_sec'], 4.5)
        self.assertEqual(result['cvar95_e2e_sec'], 6.)
        self.assertEqual(sum(w['outstanding_request_seconds'] for w in result['windows']), 9.)
        self.assertEqual(result['return'], -5.25)
        self.assertEqual([w['terminated'] for w in result['windows']], [False, False, True])
        self.assertEqual([w['tail_reward'] for w in result['windows']], [0., 0., -3.])

    def test_exact_fractional_tail_at_screening_N(self):
        s = summary()
        s['planned_request_count'] = 128
        s['requests'] = [dict(request_id=str(i), sent_ns=0, client_terminal_ns=(i+1)*1_000_000_000,
                             actual_output_tokens=1, expected_output_tokens=1) for i in range(128)]
        result = CohortReward(1.).reconstruct(s, [0])
        expected = (sum(range(123, 129))+.4*122)/6.4
        self.assertAlmostEqual(result['cvar95_e2e_sec'], expected)
        self.assertAlmostEqual(result['return'], -(64.5+expected))

    def test_predecision_incurred_area_counted_once(self):
        reward = CohortReward(2.)
        early = reward.reconstruct(summary(), [0, 3_000_000_000])
        late = reward.reconstruct(summary(), [2_500_000_000, 3_000_000_000])
        self.assertEqual(early['return'], late['return'])

    def test_incomplete_unknown_audit_and_length_mismatch_cannot_train(self):
        for key, value in [('status', 'incomplete'), ('cohort_complete', False),
                           ('audit', {'status': 'incomplete'}), ('route_audit_passed', False)]:
            s = summary()
            s[key] = value
            with self.assertRaises(ValueError):
                CohortReward(2.).reconstruct(s, [0])
        s = summary()
        s['requests'][0]['actual_output_tokens'] = 9
        with self.assertRaises(ValueError):
            CohortReward(2.).reconstruct(s, [0])

    def test_duplicate_ids_and_post_terminal_actions_rejected(self):
        s = summary()
        s['requests'][1]['request_id'] = 'a'
        with self.assertRaises(ValueError):
            CohortReward(2.).reconstruct(s, [0])
        with self.assertRaises(ValueError):
            CohortReward(2.).reconstruct(summary(), [0, 8_000_000_000])

    def test_exact_MC_targets_across_storage_boundary(self):
        self.assertEqual(complete_returns([-1., -2., -10.]), [-13., -12., -10.])
        # All rewards, including the terminal tail, affect earlier actions.
        self.assertEqual(complete_returns([-1., -2., -10.])[:2], [-13., -12.])
        with self.assertRaises(ValueError):
            CohortReward(1., gamma=.99)
        with self.assertRaises(ValueError):
            CohortReward(float('nan'))


class PPOTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)
        self.policy = CohortActorCritic(16)
        self.normalizer = normalizer_definition(2.)

    def test_requested_remaining_times_are_observed_without_future_request_metadata(self):
        a = observation()
        b = copy.deepcopy(a)
        b.update(future_length=1536, winner_label=True, scenario_id='S1_02', future_arrivals=[1, 2])
        self.assertEqual(causal_observation(a, 0, self.normalizer), causal_observation(b, 0, self.normalizer))
        for name in ['remaining_arrival_sec', 'remaining_total_sec']:
            b = copy.deepcopy(a)
            b['global_features'][b['global_feature_names'].index(name)] = 123.
            self.assertNotEqual(causal_observation(a, 0, self.normalizer), causal_observation(b, 0, self.normalizer))

    def test_all_masked_means_exact_Stay_logp_zero_and_entropy_zero(self):
        raw = observation()
        raw['action_mask'] = [True]+[False]*12
        obs = causal_observation(raw, 0, self.normalizer)
        dist, value = self.policy.distribution(*tensors([obs]))
        self.assertEqual(dist.sample().item(), 0)
        self.assertEqual(dist.log_prob(torch.tensor([0])).item(), 0.)
        self.assertEqual(dist.entropy().item(), 0.)
        self.assertTrue(torch.isfinite(value).all())
        with self.assertRaises(ValueError):
            self.policy.evaluate(tensors([obs]), torch.tensor([1]))

    def test_sampled_actions_obey_original_mask_and_reproduce_joint_logp(self):
        raw = observation()
        raw['action_mask'] = [True, True]+[False]*11
        controller = FrozenController(self.policy, self.normalizer, 0)
        for _ in range(30):
            saved = controller(raw, 0)
            self.assertIn(saved['action'], [0, 1])
            logp, _, _ = self.policy.evaluate(tensors([saved['observation']]), torch.tensor([saved['action']]))
            self.assertAlmostEqual(logp.item(), saved['old_logp'], places=6)

    def rollout(self):
        controller = FrozenController(self.policy, self.normalizer, 0)
        decisions = []
        for t in range(8):
            raw = observation()
            raw['decision_ns'] = t*1_000_000_000
            decisions.append({'ppo': controller(raw, 0)})
        return build_rollout(summary(), decisions, CohortReward(2.))

    def test_real_torch_clipped_PPO_changes_parameters_and_has_finite_losses(self):
        rollout = self.rollout()
        before = policy_hash(self.policy)
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=.0003)
        result = ppo_update(self.policy, optimizer, [rollout], SETTINGS, seed=1)
        self.assertNotEqual(policy_hash(self.policy), before)
        self.assertGreater(result['optimizer_steps'], 0)
        self.assertTrue(all(math.isfinite(m['loss']) for m in result['metrics']))
        self.assertTrue(result['behavior_logp_reproduced'])

    def test_old_logp_tamper_and_mixed_versions_stop_before_update(self):
        a = self.rollout()
        a['records'][0]['old_logp'] += .1
        before = policy_hash(self.policy)
        optimizer = torch.optim.Adam(self.policy.parameters())
        with self.assertRaisesRegex(ValueError, 'log-probability'):
            ppo_update(self.policy, optimizer, [a], SETTINGS, seed=1)
        self.assertEqual(policy_hash(self.policy), before)
        a = self.rollout()
        b = copy.deepcopy(a)
        b['policy_version'] = 1
        with self.assertRaises(ValueError):
            ppo_update(self.policy, optimizer, [a, b], SETTINGS, seed=1)

    def test_checkpoint_restores_policy_optimizer_and_protocol_identity(self):
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=.0003)
        ppo_update(self.policy, optimizer, [self.rollout()], SETTINGS, seed=1)
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'checkpoint.pt'
            save_checkpoint(p, self.policy, optimizer, next_index=2, version=1, freeze_sha256='frozen',
                            config={'test': True}, normalizer=self.normalizer)
            loaded = torch.load(p, weights_only=False)
            restored = CohortActorCritic(16)
            restored.load_state_dict(loaded['policy'])
            opt = torch.optim.Adam(restored.parameters())
            opt.load_state_dict(loaded['optimizer'])
            self.assertEqual(policy_hash(restored), policy_hash(self.policy))
            self.assertEqual(loaded['next_index'], 2)
            self.assertEqual(loaded['policy_version'], 1)
            self.assertTrue(opt.state)


if __name__ == '__main__':
    unittest.main()
