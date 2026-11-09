"""CPU HTTP integration through the actual full-cohort sender and PPO hook."""
import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest

import torch

from rl_scheduler.full_completion import run_episode
from rl_scheduler.cohort_ppo import CohortActorCritic, FrozenController, audit_execution, build_rollout, normalizer_definition, ppo_update
from rl_scheduler.cohort_reward import CohortReward, MigPPOV2Reward
from rl_scheduler.real_bridge import ControlArbiter
from rl_scheduler.real_budget import atomic_json
from rl_scheduler.real_runner import server_request_id
from rl_scheduler.tests.test_real_runner import Producer
from rl_scheduler.tests.test_real_bridge import Clock, states
from rl_scheduler.tests.test_cohort_ppo import SETTINGS


class FullCohortHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_full_cohort_HTTP_to_reward_PPO_and_checkpoint(self):
        from aiohttp import web, ClientSession, ClientTimeout
        from aiohttp.test_utils import TestServer
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            causal = out/'causal'
            causal.mkdir()
            atomic_json(out/'service_session.json', {'causal_dir': str(causal)})
            producers = {name: Producer(causal, name) for name in
                         ('api', 'manager', *[f'local{i}' for i in range(4)], *[f'worker{i}' for i in range(4)])}
            measured = states(Clock())
            for state in measured:
                state.update(running=0, used_blocks=0, free_blocks=1000, candidate_id=None)
            async def provider():
                return [dict(s, measured_ns=time.monotonic_ns()) for s in measured]
            async def dispatch(receipt):
                raise AssertionError('forced Stay must never dispatch')
            core = ControlArbiter(provider, dispatch)
            async def control(request):
                method, body = request.match_info['method'], await request.json()
                if method == 'acquire_controller':
                    self.assertEqual(body['cooldown_sec'], 0.)
                if method == 'flush':
                    return web.json_response({'api': producers['api'].checkpoint(), 'cluster': {
                        'manager': producers['manager'].checkpoint(), 'instances': [
                            {'local': producers[f'local{i}'].checkpoint(), 'workers': [producers[f'worker{i}'].checkpoint()]}
                            for i in range(4)]}})
                if method == 'release_episode':
                    return web.json_response({'status': 'released'})
                return web.json_response(await getattr(core, method)(**body))
            async def generate(request):
                body = await request.json()
                rid, ep = body['request_id'], body['episode_id']
                sid = server_request_id('cpu_run', ep, rid)
                instance = body['initial_instance_id']
                producers['manager'].emit('real_initial_route', request_id=sid, instance_id=instance)
                producers['local'+instance].emit('request_engine_admitted', request_id=sid, instance_id=instance)
                def event(kind, **data):
                    producers['api'].emit('real_metrics_event', metric_kind=kind, run_id='cpu_run', episode_id=ep,
                        request_id=rid, server_request_id=sid, data=data)
                def frame(kind, **data):
                    return (json.dumps({'run_id': 'cpu_run', 'episode_id': ep, 'request_id': rid,
                        'server_request_id': sid, 'kind': kind, **data})+'\0').encode()
                response = web.StreamResponse()
                await response.prepare(request)
                event('accepted')
                await response.write(frame('accepted'))
                producers['local'+instance].emit('engine_request_output', request_id=sid, instance_id=instance, output_tokens=1)
                event('api_first')
                await response.write(frame('output', cumulative_tokens=1, text='a'))
                await asyncio.sleep(1.65)
                await response.write(frame('output', cumulative_tokens=4, text='abcd'))
                event('server_terminal', outcome='success', input_tokens=128, output_tokens=4)
                await response.write(frame('terminal', outcome='success'))
                return response
            app = web.Application()
            app.router.add_post('/real_control/{method}', control)
            app.router.add_post('/generate_real_metrics', generate)
            trace = [dict(request_id=f'r{i}', scheduled_arrival_sec=i*.02, prompt='CPU fixture', input_tokens=128,
                          output_tokens=4, initial_instance_index=i) for i in range(2)]
            policy = CohortActorCritic(16)
            try:
                async with TestServer(app) as server, ClientSession(timeout=ClientTimeout(total=10.)) as session:
                    result = await asyncio.wait_for(run_episode(cfg={'service': {'base_url': str(server.make_url('')).rstrip('/')},
                        'sampling': {'n': 1, 'best_of': 1, 'temperature': 0., 'ignore_eos': True}},
                        run_id='cpu_run', episode_id='ep', policy='PPO', trace=trace, run_dir=out,
                        control_session=session, decision_controller=FrozenController(policy, normalizer_definition(1.), 0),
                        controller_version='cpu-PPO-v0'), timeout=15.)
                    self.assertEqual(result['status'], 'passed', result['issues'])
                    self.assertEqual(result['client_terminal_count'], 2)
                    self.assertTrue(result['route_audit_passed'])
                    full = json.loads((out/'runs/ep/episode.json').read_text())
                    self.assertEqual(full['decision_interval_sec'], 1.5)
                    self.assertEqual(full['migration_cooldown_sec'], 0.)
                    decisions = [json.loads(line) for line in (out/'runs/ep/decisions.jsonl').read_text().splitlines()]
                    decision_times = [d['ppo']['decision_ns'] for d in decisions if d.get('ppo')]
                    self.assertGreaterEqual(len(decision_times), 2)
                    for a,b in zip(decision_times, decision_times[1:]):
                        self.assertGreaterEqual((b-a)/1e9, 1.45)
                        self.assertLess((b-a)/1e9, 2.5)
                    events = [json.loads(line) for line in (out/'runs/ep/events.jsonl').read_text().splitlines()]
                    intents = [json.loads(line) for line in (out/'runs/ep/ppo_action_intents.jsonl').read_text().splitlines()]
                    self.assertEqual(audit_execution(full, decisions, events, intents)['status'], 'passed')
                    with self.assertRaisesRegex(ValueError, 'pre-RPC'):
                        audit_execution(full, decisions, events, [])
                    # A declared route is insufficient if actual engine admission differs.
                    admission = next(e for e in events if e['data'].get('event') == 'request_engine_admitted')
                    admission['data']['instance_id'] = 'unexpected'
                    with self.assertRaisesRegex(ValueError, 'engine admission'):
                        audit_execution(full, decisions, events, intents)
                    v2 = build_rollout(full, decisions, MigPPOV2Reward(hotspot_threshold=.7))
                    self.assertEqual(v2['reward_audit']['schema'], 'mig-ppo-v2')
                    self.assertTrue(v2['records'][-1]['terminated'])
                    self.assertEqual(v2['reward_audit']['terminal_potential'], 0.)
                    self.assertTrue(all(r['gamma_t'] == .99**r['dt_sec'] for r in v2['records']))
                    rollout = build_rollout(full, decisions, CohortReward(1.))
                    optimizer = torch.optim.Adam(policy.parameters(), lr=.0003)
                    update = ppo_update(policy, optimizer, [rollout], SETTINGS, seed=1)
                    self.assertGreater(update['optimizer_steps'], 0)
                    self.assertTrue(all(r['action'] == 0 and r['old_logp'] == 0 for r in rollout['records']))
                    self.assertAlmostEqual(rollout['reward_audit']['return'], -(
                        full['primary_metrics']['client_e2e_ms']['mean']+
                        full['primary_metrics']['client_e2e_ms']['cvar95'])/1000)
                    self.assertIsNone(core.lease)
            finally:
                for producer in producers.values():
                    producer.stream.close()

    async def test_native_dispatch_has_no_fixed_route_and_audits_first_admission(self):
        from aiohttp import web, ClientSession, ClientTimeout
        from aiohttp.test_utils import TestServer
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            causal = out/'causal'
            causal.mkdir()
            atomic_json(out/'service_session.json', {'causal_dir': str(causal)})
            producers = {name: Producer(causal, name) for name in
                         ('api', 'manager', *[f'local{i}' for i in range(4)], *[f'worker{i}' for i in range(4)])}
            measured = states(Clock())
            for state in measured:
                state.update(running=0, used_blocks=0, free_blocks=1000, candidate_id=None)
            async def provider():
                return [dict(s, measured_ns=time.monotonic_ns()) for s in measured]
            async def dispatch(receipt):
                raise AssertionError('native dispatcher fixture must not migrate')
            core = ControlArbiter(provider, dispatch)
            async def control(request):
                method, body = request.match_info['method'], await request.json()
                if method == 'acquire_controller':
                    self.assertEqual(body['cooldown_sec'], 0.)
                if method == 'flush':
                    return web.json_response({'api': producers['api'].checkpoint(), 'cluster': {
                        'manager': producers['manager'].checkpoint(), 'instances': [
                            {'local': producers[f'local{i}'].checkpoint(), 'workers': [producers[f'worker{i}'].checkpoint()]}
                            for i in range(4)]}})
                if method == 'release_episode':
                    return web.json_response({'status': 'released'})
                return web.json_response(await getattr(core, method)(**body))
            counts = [0] * 4
            async def generate(request):
                body = await request.json()
                self.assertNotIn('initial_instance_id', body)
                rid, ep = body['request_id'], body['episode_id']
                sid = server_request_id('native_cpu_run', ep, rid)
                scores = [1e6 if count == 0 else 1000. / count for count in counts]
                instance_index = max(range(4), key=lambda i: (scores[i], -i))
                instance = str(instance_index)
                selection = {'dispatch_policy': 'load', 'dispatch_load_metric': 'remaining_steps',
                    'selection_algorithm': 'sort_instance_infos_then_random_choice_from_top_k',
                    'topk_random_dispatch': 1, 'top_k_candidates': [instance],
                    'selected_instance_id': instance,
                    'candidate_scores': [{'instance_id': str(i), 'metric_value': scores[i],
                        'instance_num_requests_before': counts[i]} for i in range(4)],
                    'state_updated_ns': time.monotonic_ns(), 'state_age_ms': 0.,
                    'dispatch_random_seed': 12345}
                counts[instance_index] += 1
                producers['manager'].emit('real_initial_route', request_id=sid,
                    instance_id=instance, route_source='native_llumnix_dispatch', selection=selection)
                producers['local'+instance].emit('request_engine_admitted', request_id=sid, instance_id=instance)
                def event(kind, **data):
                    producers['api'].emit('real_metrics_event', metric_kind=kind, run_id='native_cpu_run', episode_id=ep,
                        request_id=rid, server_request_id=sid, data=data)
                def frame(kind, **data):
                    return (json.dumps({'run_id': 'native_cpu_run', 'episode_id': ep, 'request_id': rid,
                        'server_request_id': sid, 'kind': kind, **data})+'\0').encode()
                response = web.StreamResponse()
                await response.prepare(request)
                event('accepted')
                await response.write(frame('accepted'))
                producers['local'+instance].emit('engine_request_output', request_id=sid, instance_id=instance, output_tokens=1)
                event('server_terminal', outcome='success', input_tokens=128, output_tokens=1)
                await response.write(frame('output', cumulative_tokens=1, text='a'))
                await response.write(frame('terminal', outcome='success'))
                return response
            app = web.Application()
            app.router.add_post('/real_control/{method}', control)
            app.router.add_post('/generate_real_metrics', generate)
            trace = [dict(request_id=f'native-{i}', scheduled_arrival_sec=i*.02, prompt='CPU fixture',
                input_tokens=128, output_tokens=1, initial_instance_index=None, routing_mode='native')
                for i in range(4)]
            try:
                async with TestServer(app) as server, ClientSession(timeout=ClientTimeout(total=10.)) as session:
                    result = await asyncio.wait_for(run_episode(cfg={'service': {
                        'base_url': str(server.make_url('')).rstrip('/')},
                        'sampling': {'n': 1, 'best_of': 1, 'temperature': 0., 'ignore_eos': True}},
                        run_id='native_cpu_run', episode_id='native_ep', policy='Stay', trace=trace,
                        run_dir=out, control_session=session, decision_interval_sec=1.5,
                        migration_cooldown_sec=0., episode_horizon_sec=10.), timeout=15.)
                    self.assertEqual(result['status'], 'passed', result['issues'])
                    self.assertTrue(result['route_audit_passed'], result['route_audit'])
                    audit = result['route_audit']
                    self.assertEqual(len(audit['accepted_request_ids']), 4)
                    self.assertTrue(audit['selection_evidence_complete'])
                    self.assertEqual(audit['manager_first_routes'], audit['engine_first_admissions'])
                    full = json.loads((out/'runs/native_ep/episode.json').read_text())
                    self.assertEqual(full['routing_mode'], 'native')
                    self.assertEqual(full['decision_interval_sec'], 1.5)
                    self.assertEqual(full['migration_cooldown_sec'], 0.)
                    self.assertEqual(audit_execution(full, [],
                        [json.loads(line) for line in (out/'runs/native_ep/events.jsonl').read_text().splitlines()],
                        [])['status'], 'passed')
            finally:
                for producer in producers.values():
                    producer.stream.close()


if __name__ == '__main__':
    unittest.main()
