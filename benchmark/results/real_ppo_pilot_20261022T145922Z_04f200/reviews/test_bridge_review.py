"""Independent CPU regression expectations. Failures expose reviewed defects.

No Ray/model import, no service start, no production edits. AST extraction runs
the exact reviewed method bodies while replacing only external RPC objects.
"""
import ast
import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from rl_scheduler.real_bridge import ControlArbiter
from rl_scheduler.real_runner import flatten_acks, checkpoint, LiveEpisodeIO
from rl_scheduler.real_metrics import RequestLedger, Event, Checkpoint, NS
from rl_scheduler.tests.test_real_bridge import Clock, states, extract_method, load_file
from rl_scheduler.tests.test_real_env import environment, snapshot


class BridgeReview(unittest.IsolatedAsyncioTestCase):
    async def test_R1_same_process_worker_local_flush_must_not_regress(self):
        # Non-SPMD TP=1: executor.driver_worker is a local Python object.
        # Llumlet flushes worker first, then itself; both share causal_events globals.
        flush = extract_method('llumnix/llumlet/llumlet.py', 'Llumlet', 'real_control_flush',
                               {'asyncio': asyncio})
        counter = [10]
        def local_flush(*args):
            counter[0] += 1
            return dict(status='passed', source='same-process', last_sequence=0,
                        through_ns=counter[0], durable=True, pending_delivery=0)
        async def workers(*args, **kwargs):
            return [local_flush()]
        owner = SimpleNamespace(backend_engine=SimpleNamespace(_run_workers_async=workers),
                                real_control_state=lambda: {'pending_delivery': 0})
        # Stub the import target: importing llumnix.__init__ would also import
        # Ray/vLLM even though no cluster or model is needed for this test.
        with patch.dict(sys.modules, {'llumnix.metrics.causal_events': SimpleNamespace(flush=local_flush)}):
            raw = await flush(owner)
        ledger = RequestLedger('r', 'e', start_ns=0, arrival_end_ns=1, score_deadline_ns=2,
                               expected_sources=['same-process'], expected_request_ids=[])
        for ack in flatten_acks(raw):
            ledger.acknowledge(checkpoint(ack))
        self.assertEqual(ledger.audit(require_cleanup=True)['status'], 'passed',
                         {'flush': raw, 'issues': sorted(ledger.issues)})

    async def test_R2_unknown_submit_must_query_before_new_decision(self):
        env, clock, bridge, io = environment()
        calls = []
        async def submit(*args):
            calls.append(('submit', args[0]))
            return {'receipt_status': 'unknown', 'terminal_status': None,
                    'rejection_reason': 'transport_uncertain_query_before_any_new_action'}
        async def query(decision_id):
            calls.append(('query', decision_id))
            return {'receipt_status': 'unknown', 'terminal_status': None}
        bridge.submit_migration, bridge.query_receipt = submit, query
        await env.reset()
        await env.step(1)
        try:
            await env.step(1)
        except RuntimeError:
            pass  # stopping on uncertainty is also valid
        # Assert before finish(), which currently is the only place that queries.
        second = next((i for i, c in enumerate(calls[1:], 1) if c[0] == 'submit'), None)
        self.assertTrue(second is None or ('query', calls[0][1]) in calls[1:second], calls)

    async def test_R3_controller_acquisition_must_fence_native_during_await(self):
        clock = Clock()
        entered, resume = asyncio.Event(), asyncio.Event()
        native_busy = [False]
        async def provider():
            measured = states(clock)
            entered.set()
            await resume.wait()  # real gather(actor.state.remote()) yields here
            return measured
        async def dispatch(receipt):
            raise AssertionError('no controlled dispatch expected')
        core = ControlArbiter(provider, dispatch, clock=clock,
                              native_idle=lambda: not native_busy[0])
        task = asyncio.create_task(core.acquire_controller('r', 'v', 'e'))
        await entered.wait()
        # Manager._migrate checks ordinary_blocked without taking core._lock.
        if not core.ordinary_blocked:
            native_busy[0] = True
        resume.set()
        try:
            lease = await task
        except RuntimeError:
            lease = None
        self.assertFalse(lease is not None and native_busy[0],
                         'ordinary migration admitted during acquire and lease nevertheless granted')

    async def test_R4_abort_ack_must_cover_migrated_owner_before_output(self):
        tree = ast.parse((ROOT/'llumnix/entrypoints/vllm/api_server.py').read_text())
        method = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
                      and n.name == '_real_metrics_abort_confirmed')
        unit = ast.Module(body=[method], type_ignores=[])
        ast.fix_missing_locations(unit)
        destination_running = {'r'}
        aborted, cleared = [], []
        async def source_abort(rid):
            aborted.append(('source', rid))  # no-op: source has already committed r
        async def rpc(method, rid, **kwargs):
            return await method(rid)
        client = SimpleNamespace(
            # Actual API routing is updated only by get_request_outputs_loop().
            _get_instance_for_abort=lambda rid: (['source'], [SimpleNamespace(abort=source_abort)]),
            _clear_client_request_states=lambda rid: cleared.append(rid))
        ns = {'llumnix_client': client, 'asyncio_wait_for_ray_remote_call_with_timeout': rpc}
        exec(compile(unit, 'api_server.py', 'exec'), ns)
        confirmed = await ns['_real_metrics_abort_confirmed']('r')
        self.assertFalse(confirmed and 'r' in destination_running,
                         {'confirmed': confirmed, 'destination_still_running': True,
                          'aborted': aborted, 'client_states_cleared': cleared})

    async def test_R5_reset_must_reject_pending_API_delivery(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            causal = run/'causal'; causal.mkdir()
            (run/'service_session.json').write_text(json.dumps({'causal_dir': str(causal)}))
            def ack(name, pending=0):
                return dict(source=name, last_sequence=0, through_ns=time.monotonic_ns(),
                            status='passed', durable=True, pending_delivery=pending)
            async def snap():
                return snapshot(Clock())
            async def call(*args):
                return {'api': ack('api', 1), 'cluster': {'manager': ack('manager'),
                        'instances': [{'local': ack(f'l{i}'), 'workers': [ack(f'w{i}')]} for i in range(4)]}}
            bridge = SimpleNamespace(snapshot=snap, call=call)
            io = LiveEpisodeIO(bridge, None, run_id='r', episode_id='e', run_dir=run,
                               trace=[], sampling={}, arrival_sec=1)
            try:
                with self.assertRaises(RuntimeError, msg='pending API delivery was silently admitted'):
                    await io.prepare()
            finally:
                await io.close()

    async def test_R6_live_plan_requires_review_gate(self):
        module = load_file('_review_cli', 'benchmark/run_real_ppo.py')
        import yaml
        run = Path(__file__).resolve().parents[1]
        cfg = yaml.safe_load((ROOT/'rl_scheduler/configs/real_pilot.yaml').read_text())
        args = module.parse_args(['--config', str(ROOT/'rl_scheduler/configs/real_pilot.yaml'),
                                  '--stage', 'telemetry', '--output-dir', str(run), '--manage-services'])
        # Preserve all current gate checks. Stub only budget I/O so this CPU
        # review cannot mutate the shared experiment clock.
        budget = SimpleNamespace(require=lambda duration: {'remaining': {'total_experiment_wall_time_sec': 28800}})
        review_path = run/'reviews/bridge_review.json'
        self.assertTrue(not review_path.exists() or json.loads(review_path.read_text())['status'] != 'passed')
        with patch.object(module, 'BudgetLedger', return_value=budget):
            with self.assertRaises(RuntimeError, msg='live intent admitted without passed independent review'):
                module.plan(args, cfg)

    async def test_R7_failed_worker_flush_must_not_disappear_from_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            causal = run/'causal'; causal.mkdir()
            (run/'service_session.json').write_text(json.dumps({'causal_dir': str(causal)}))
            def ack(name):
                return dict(source=name, last_sequence=0, through_ns=time.monotonic_ns(),
                            status='passed', durable=True, pending_delivery=0)
            async def snap():
                return snapshot(Clock())
            async def call(*args):
                return {'api': ack('api'), 'cluster': {'manager': ack('manager'),
                        'instances': [{'local': ack(f'l{i}'), 'workers': [
                            {'status': 'incomplete', 'reason': 'pending_cuda_telemetry'}
                            if i == 0 else ack(f'w{i}')]} for i in range(4)]}}
            io = LiveEpisodeIO(SimpleNamespace(snapshot=snap, call=call), None,
                               run_id='r', episode_id='e', run_dir=run,
                               trace=[], sampling={}, arrival_sec=1)
            try:
                with self.assertRaises(RuntimeError, msg='failed worker acknowledgement dropped by flatten_acks'):
                    await io.prepare()
            finally:
                await io.close()

    async def test_independent_request_lifecycle_and_late_completion(self):
        ledger = RequestLedger('r', 'e', start_ns=0, arrival_end_ns=4*NS, score_deadline_ns=8*NS,
                               expected_sources=['client', 'server'], expected_request_ids=['a', 'b'])
        seq = {'client': 0, 'server': 0}
        def add(source, kind, t, rid, received=None, **data):
            seq[source] += 1
            ledger.ingest([Event(source, seq[source], kind, int(t*NS), int((t if received is None else received)*NS),
                                 'r', 'e', rid, 'server-'+rid, data)])
        for rid in ('a', 'b'):
            add('client', 'planned', 0, rid, scheduled_ns=0)
            add('client', 'sent', 0, rid)
            add('server', 'accepted', .1, rid)
        add('server', 'engine_first', .2, 'a')
        add('server', 'api_first', .3, 'a')
        add('client', 'client_output', 1, 'a', cumulative_tokens=1)
        add('client', 'client_output', 1.3, 'a', cumulative_tokens=4, received=2)
        self.assertEqual(ledger.observe(int(1.5*NS))['requests'][0]['actual_output_tokens'], 1)
        add('server', 'server_terminal', 1.4, 'a', outcome='success')
        add('client', 'client_terminal', 2, 'a', outcome='success', http_status=200)
        # An internal completion cannot finish a client request.
        add('server', 'server_terminal', 7, 'b', outcome='success')
        for s in seq:
            ledger.acknowledge(Checkpoint(s, seq[s], 8*NS, 8*NS, 0, True))
        score = ledger.freeze_score()
        self.assertEqual(score['status'], 'passed')
        self.assertEqual(score['fixed']['completed'], 1)
        self.assertEqual(score['fixed']['client_output_tokens'], 4)
        self.assertEqual(score['fixed']['outstanding_ids'], ['b'])
        self.assertEqual(score['fixed']['outstanding_request_seconds'], 10.)
        self.assertEqual(score['fixed']['tpot_conditional_ms'], [100.])
        add('client', 'client_output', 9, 'b', cumulative_tokens=64)
        add('client', 'client_terminal', 10, 'b', outcome='success', http_status=200)
        for s in seq:
            ledger.acknowledge(Checkpoint(s, seq[s], 11*NS, 11*NS, 0, True))
        self.assertEqual(ledger.freeze_score(), score)
        self.assertEqual(ledger.audit(through_ns=11*NS, require_cleanup=True)['status'], 'passed')

    async def test_independent_rejection_and_success_through_original_coordinator(self):
        module = load_file('_independent_adapter', 'llumnix/real_control.py')
        clock = Clock(); measured = states(clock); events = []; dispatched = []
        status = SimpleNamespace(RUNNING='running', WAITING='waiting')
        migration_status = SimpleNamespace(FINISHED='finished', ABORTED_DST='aborted_dst', ABORTED_SRC='aborted_src')
        log = SimpleNamespace(info=lambda *args: None)
        coordinator_method = extract_method('llumnix/llumlet/migration_coordinator.py',
            'MigrationCoordinator', '_migrate_out_one_request',
            {'time': time, 'RequestStatus': status, 'MigrationStatus': migration_status,
             'emit_causal_event': lambda event, **fields: events.append({'event': event, **fields}), 'logger': log})
        request = SimpleNamespace(request_id='request-0', llumnix_status='running',
                                  is_migrating=False, stage_num_blocks_list=[7, 13])
        backend = SimpleNamespace(free_src_request=lambda req: events.append({'event': 'free_src'}),
                                  pop_migrating_out_request_last_stage=lambda req: events.append({'event': 'unpause'}))
        async def transfer(*args):
            return migration_status.FINISHED
        async def commit(*args):
            return SimpleNamespace(success=True)
        coordinator = SimpleNamespace(instance_id='0', enable_engine_migration_interface=False,
            migration_scheduler=SimpleNamespace(request_migration_policy='SR',
                _get_shortest_running_request=lambda *args: [request]), backend_engine=backend,
            _migrate_out_running_request=transfer, _dst_commit_dst_request=commit)
        async def original(*args, **kwargs):
            return await coordinator_method(coordinator, *args, **kwargs)
        coordinator._migrate_out_one_request = original
        owner = SimpleNamespace(migration_coordinator=coordinator)
        async def dst_state():
            return deepcopy(measured[1])
        async def reserve(*args):
            return {'status': 'prepared'}
        async def unreserve(*args):
            return {'status': 'released'}
        dst = SimpleNamespace(real_control_state=SimpleNamespace(remote=dst_state),
                             real_control_reserve=SimpleNamespace(remote=reserve),
                             real_control_unreserve=SimpleNamespace(remote=unreserve))
        async def provider():
            return deepcopy(measured)
        async def dispatch(receipt):
            dispatched.append(receipt)
            return await module.controlled_migrate(owner, dst, SimpleNamespace(instance_id='1'), 'ordinary',
                expected_candidate_id=receipt['expected_candidate_id'], decision_id=receipt['decision_id'],
                migration_id=receipt['migration_id'], source_epoch=receipt['source_epoch'],
                destination_epoch=receipt['destination_epoch'])
        core = ControlArbiter(provider, dispatch, clock=clock)
        with patch.object(module, 'instance_state', lambda owner: deepcopy(measured[0])), \
             patch.object(module.time, 'monotonic_ns', clock):
            lease = await core.acquire_controller('run', 'review', 'episode')
            snap = await core.snapshot()
            rejected = await core.submit_migration(lease['lease_id'], 'reject', '0', '1', 'stale-candidate', snap['version'])
            self.assertEqual(rejected['rejection_reason'], 'SR_candidate_changed')
            self.assertFalse(dispatched)
            accepted = await core.submit_migration(lease['lease_id'], 'success', '0', '1', 'request-0', snap['version'])
            self.assertEqual(accepted['receipt_status'], 'accepted')
            await asyncio.gather(*list(core._tasks))
            final = await core.query_receipt(lease['lease_id'], 'success')
            repeated = await core.submit_migration(lease['lease_id'], 'success', '0', '1', 'request-0', snap['version'])
        self.assertEqual(final['terminal_status'], 'succeeded')
        self.assertEqual(final['actual_request_id'], 'request-0')
        self.assertEqual(final['copied_blocks'], 20)
        self.assertEqual(repeated, final)
        self.assertEqual(len(dispatched), 1)
        self.assertIsNone(core.slot)
        correlated = [e for e in events if e.get('migration_id') == accepted['migration_id']]
        self.assertEqual([e['event'] for e in correlated], ['migration_attempt_started',
                         'migration_target_commit_started', 'migration_request_resumed', 'migration_attempt_completed'])
        (Path(__file__).parent/'migration_chain.json').write_text(json.dumps(
            {'scope': 'CPU original coordinator body; transfer/actors mocked',
             'rejection': rejected, 'acceptance': accepted, 'terminal': final, 'events': events}, indent=2)+'\n')


if __name__ == '__main__':
    unittest.main(verbosity=2)
