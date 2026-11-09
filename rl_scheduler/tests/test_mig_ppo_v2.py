"""Numerical V2 contracts: measured state, capped time integrals, shaping and GAE."""
import copy
import math
import unittest

import torch

from rl_scheduler.cohort_ppo import (CohortActorCritic, FrozenController, build_rollout,
                                    normalizer_definition, ppo_update)
from rl_scheduler.cohort_reward import MigPPOV2Reward, time_discounted_gae
from rl_scheduler.real_bridge import PAIRS
from rl_scheduler.real_observation import (GLOBAL_FEATURES, INSTANCE_FEATURES, PAIR_FEATURES,
                                           RealObservationBuilder)
from rl_scheduler.tests.test_cohort_ppo import observation, SETTINGS
from types import SimpleNamespace

NS = 1_000_000_000


def snapshot(at, queues=(0, 0, 0, 0), loads=(0., 0., 0., 0.)):
    return dict(received_ns=at, instances=[dict(instance_id=str(i), waiting=q,
        used_blocks=1000*u, total_blocks=1000) for i,(q,u) in enumerate(zip(queues, loads))])


def episode():
    return dict(status='passed', cohort_complete=True, audit={'status': 'passed'},
        route_audit_passed=True, planned_request_count=1, episode_id='v2',
        requests=[dict(request_id='r', sent_ns=0, client_first_ns=2*NS,
            token_increments=[(2*NS, 1), (3*NS, 1)], client_terminal_ns=4*NS,
            outcome='success', actual_output_tokens=2, expected_output_tokens=2)])


def decisions(times, queues=None, loads=None):
    return [dict(observation_decision_ns=t, snapshot=snapshot(t,
                queues[i] if queues else (0,)*4, loads[i] if loads else (0.,)*4))
            for i,t in enumerate(times)]


class V2Tests(unittest.TestCase):
    def setUp(self):
        self.reward = MigPPOV2Reward(hotspot_threshold=.75, n_ref=1.)

    def test_exact_state_schema_and_policy_dimensions(self):
        self.assertEqual(INSTANCE_FEATURES, ('running', 'waiting', 'head_wait_sec', 'used_fraction',
            'candidate_blocks', 'last_step_ms', 'tokens_per_sec_5s', 'waiting_slope_5s', 'used_fraction_slope_5s'))
        self.assertEqual(GLOBAL_FEATURES, ('remaining_arrival_sec', 'remaining_total_sec', 'outstanding',
            'overdue_first', 'output_stalled', 'client_tokens_per_sec_5s', 'busy', 'cooldown_remaining_sec'))
        self.assertEqual(PAIR_FEATURES, ('waiting_difference', 'remaining_kv_blocks_difference'))
        net = CohortActorCritic(16)
        record = FrozenController(net, normalizer_definition(), 'v2')(observation(), 0)
        obs = record['observation']
        self.assertEqual([len(x) for x in obs['instance_features']], [18]*4)
        self.assertEqual(len(obs['global_features']), 16)
        self.assertEqual([len(x) for x in obs['pair_features']], [4]*12)

    def test_capped_ramps_integrate_exactly_and_ignore_aggregate_TPOT(self):
        s = episode()
        s['requests'][0]['tpot_ms'] = 99999999.
        times = [0, NS, 3*NS]
        result = self.reward.reconstruct(s, times, decisions(times))
        # First wait: integral_0^2 t/2 = 1. Each one-second decode gap:
        # integral min(t/.15,2) = 1.7, weighted by kappa=.5.
        self.assertAlmostEqual(sum(w['D_integral'] for w in result['windows']), 2.7)
        self.assertAlmostEqual(result['return'], -.4*2.7)
        self.assertEqual([w['dt_sec'] for w in result['windows']], [1., 2., 1.])
        no_timeline = copy.deepcopy(s)
        del no_timeline['requests'][0]['token_increments']
        with self.assertRaisesRegex(ValueError, 'TPOT'):
            self.reward.reconstruct(no_timeline, times, decisions(times))
        s['requests'][0].update(client_first_ns=None, token_increments=[], actual_output_tokens=0,
                                outcome='timeout', client_terminal_ns=6*NS)
        result = self.reward.reconstruct(s, [0], decisions([0]))
        self.assertAlmostEqual(result['windows'][0]['D_integral'], 8.)
        self.assertEqual(result['windows'][0]['failure_count'], 1)
        self.assertAlmostEqual(result['return'], -.4*8-1/64)

    def test_snapshot_pressure_formulas_and_pressure_gated_balance(self):
        snap = snapshot(0, (0, 0, 64, 64), (0., 0., 1., 1.))
        p = self.reward.pressures(snap)
        self.assertEqual(p, dict(Q=.75, H=1., B=1., B_star=1.))
        low = self.reward.pressures(snapshot(0, loads=(0., 0., .5, .5)))
        self.assertEqual(low['H'], 0.)
        self.assertEqual(low['B'], .5)
        self.assertEqual(low['B_star'], 0.)
        snap['instances'][0]['reserved_blocks_not_in_used'] = 100
        snap['instances'][0]['reserved_blocks'] = 900  # Already in used: never add again.
        self.assertAlmostEqual(self.reward.pressures(snap)['B'], math.sqrt(.9075))
        snap['instances'][0]['used_blocks'] = 1001
        with self.assertRaisesRegex(ValueError, 'measurements'):
            self.reward.pressures(snap)

    def test_component_rewards_and_discounted_shaping_telescope_to_terminal_zero(self):
        times = [0, NS, 3*NS]
        ds = decisions(times, queues=[(64,)*4, (32,)*4, (0,)*4],
                       loads=[(1., 0., 0., 0.), (.8, .4, .4, .4), (.5,)*4])
        result = self.reward.reconstruct(episode(), times, ds)
        discounted_shaping, cumulative_discount = 0., 1.
        for w in result['windows']:
            p = w['pressures']
            base = -self.reward.alpha*w['D_integral']-w['dt_sec']*(.3*p['Q']+.2*p['H']+.1*p['B_star'])
            self.assertAlmostEqual(w['reward'], sum(w['components'].values())+w['migration_cost'])
            discounted_shaping += cumulative_discount*(w['reward']-base)
            cumulative_discount *= w['gamma_t']
        p0 = result['windows'][0]['pressures']
        self.assertAlmostEqual(discounted_shaping, .1*(.45*p0['Q']+.35*p0['H']+.2*p0['B_star']))
        self.assertEqual(result['windows'][-1]['next_pressures'], dict(Q=0., H=0., B_star=0.))

    def test_failure_on_boundary_is_charged_once_and_migration_failure_is_explicit(self):
        s = episode()
        failed = dict(request_id='failed', sent_ns=0, client_first_ns=None, token_increments=[],
                      client_terminal_ns=NS, outcome='service_error', actual_output_tokens=0, expected_output_tokens=2)
        s['requests'].append(failed)
        s['planned_request_count'] = 2
        times = [0, NS, 3*NS]
        ds = decisions(times)
        ds[0].update(sampled_action=1, decision_id='move')
        s['migration_receipts'] = {'move': dict(terminal_status='failed', terminal_ns=2*NS, migration_id='m')}
        result = self.reward.reconstruct(s, times, ds)
        self.assertEqual([w['failure_count'] for w in result['windows']], [0, 1, 0])
        self.assertEqual([w['invalid_migration_count'] for w in result['windows']], [0, 1, 0])
        self.assertEqual(result['windows'][1]['migration_cost'], -.01)

    def test_real_time_gae_uses_saved_values_and_irregular_dt(self):
        windows = [dict(reward=r, gamma_t=.99**dt, lambda_t=.97**dt,
                        terminated=i==2, truncated=False) for i,(r,dt) in enumerate(zip([1., 2., 3.], [.5, 2., 1.]))]
        values = [.4, .7, .9]
        advantages, targets = time_discounted_gae(windows, values)
        a2 = 3.-.9
        a1 = 2.+.99**2*.9-.7+(.99*.97)**2*a2
        a0 = 1.+.99**.5*.7-.4+(.99*.97)**.5*a1
        for got,want in zip(advantages, [a0, a1, a2]):
            self.assertAlmostEqual(got, want)
        for got,want in zip(targets, [a0+.4, a1+.7, 3.]):
            self.assertAlmostEqual(got, want)

    def test_raw_pair_differences_and_output_stall_threshold(self):
        from rl_scheduler.tests.test_real_env import snapshot as measured_snapshot
        from rl_scheduler.tests.test_real_bridge import Clock
        clock = Clock()
        snap = measured_snapshot(clock)
        snap['instances'][0].update(waiting=8, used_blocks=400, total_blocks=1000, reserved_blocks_not_in_used=100)
        snap['instances'][1].update(waiting=2, used_blocks=100, total_blocks=1000)
        now = clock()
        rows = [dict(sent_ns=now-3*NS, client_terminal_ns=None, client_first_ns=now-NS,
                     client_last_ns=now-200_000_000, token_increments=[])]
        builder = RealObservationBuilder(snap['mapping'])
        raw = builder.build(snap, now_ns=now, ledger=SimpleNamespace(observe=lambda t: {'requests': rows}),
                            episode_start_ns=now-4*NS, arrival_end_ns=now+5*NS, deadline_ns=now+50*NS)
        self.assertEqual(raw['pair_features'][PAIRS.index((0, 1))], [6., -400.])
        self.assertEqual(raw['pair_features'][PAIRS.index((1, 0))], [-6., 400.])
        self.assertEqual(raw['global_features'][GLOBAL_FEATURES.index('output_stalled')], 1.)
        self.assertEqual(raw['global_features'][GLOBAL_FEATURES.index('remaining_total_sec')], 50.)

    def test_v2_rollout_and_actual_optimizer_update(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        policy = CohortActorCritic(16)
        controller = FrozenController(policy, normalizer_definition(), 'v2')
        times = [0, NS, 3*NS]
        ds = decisions(times)
        for t,d in zip(times, ds):
            raw = observation()
            raw['decision_ns'] = t
            d['ppo'] = controller(raw, 0)
        rollout = build_rollout(episode(), ds, self.reward)
        self.assertEqual(rollout['reward_audit']['schema'], 'mig-ppo-v2')
        update = ppo_update(policy, torch.optim.Adam(policy.parameters(), lr=.0003), [rollout], SETTINGS, seed=1)
        self.assertGreater(update['optimizer_steps'], 0)
        self.assertTrue(all(math.isfinite(m['loss']) for m in update['metrics']))

    def test_forced_stay_keeps_its_full_1p5_second_gae_transition(self):
        from rl_scheduler.cohort_ppo import tensors
        torch.set_num_threads(1)
        torch.manual_seed(43)
        policy = CohortActorCritic(16)
        controller = FrozenController(policy, normalizer_definition(), 'v2')
        times = [0, 1_500_000_000, 3*NS]
        ds = decisions(times)
        for i,(t,d) in enumerate(zip(times, ds)):
            raw = observation()
            raw['decision_ns'] = t
            raw['action_mask'] = [True, i != 1]+[False]*11
            saved = controller(raw, 0)
            saved['action'] = 0  # Explicit voluntary / forced / voluntary Stay.
            with torch.no_grad():
                logp, _, _ = policy.evaluate(tensors([saved['observation']]), torch.tensor([0]))
            saved['old_logp'] = logp.item()
            d['ppo'] = saved
        rollout = build_rollout(episode(), ds, self.reward)
        records = rollout['records']
        self.assertEqual([r['action_class'] for r in records],
                         ['voluntary_stay', 'forced_stay', 'voluntary_stay'])
        self.assertEqual([r['actor_eligible'] for r in records], [True, False, True])
        self.assertEqual([r['dt_sec'] for r in records], [1.5, 1.5, 1.])
        a2 = records[2]['reward']-records[2]['old_value']
        a1 = (records[1]['reward']+.99**1.5*records[2]['old_value']-records[1]['old_value']
              +(.99*.97)**1.5*a2)
        a0 = (records[0]['reward']+.99**1.5*records[1]['old_value']-records[0]['old_value']
              +(.99*.97)**1.5*a1)
        for r,expected in zip(records, [a0, a1, a2]):
            self.assertAlmostEqual(r['advantage'], expected)
        result = ppo_update(policy, torch.optim.Adam(policy.parameters(), lr=.0003),
                            [rollout], SETTINGS, seed=1)
        self.assertEqual(result['actor_sample_count'], 2)
        self.assertEqual(result['critic_sample_count'], 3)

    def test_missing_invalid_and_old_definitions_are_rejected(self):
        self.assertEqual(MigPPOV2Reward.from_state_dict(self.reward.state_dict()), self.reward)
        for values in [dict(hotspot_threshold=1.), dict(hotspot_threshold=.7, n_ref=0.),
                       dict(hotspot_threshold=.7, gamma_per_sec=1.1)]:
            with self.assertRaises(ValueError):
                MigPPOV2Reward(**values)
        with self.assertRaises(ValueError):
            MigPPOV2Reward.from_state_dict({'T_ref_sec': 2.})
        s = episode()
        s['requests'][0]['outcome'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'outcome'):
            self.reward.reconstruct(s, [0], decisions([0]))
        s = episode()
        s['audit']['status'] = 'incomplete'
        with self.assertRaisesRegex(ValueError, 'audited'):
            self.reward.reconstruct(s, [0], decisions([0]))


if __name__ == '__main__':
    unittest.main()
