"""Hand-computed lifecycle invariants and CPU-only HTTP/producer checks.

Run in the existing Llumnix environment:
  .venv/bin/python -m unittest rl_scheduler.tests.test_real_metrics -v
"""
import asyncio
from dataclasses import asdict, replace
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rl_scheduler.real_metrics import (NS, Event, Checkpoint, EventJournal, EventCursor,
                                       RequestLedger, CausalEventAdapter, migration_diagnostics)
from rl_scheduler.real_client import NulJSONDecoder, StreamingRequestCollector

ROOT = Path(__file__).resolve().parents[2]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Fixture:
    def __init__(self, **kwargs):
        self.ledger = RequestLedger('run', 'ep', start_ns=0, arrival_end_ns=4*NS,
                                    score_deadline_ns=8*NS, expected_sources=['s'], **kwargs)
        self.sequence = 0

    def add(self, kind, t, rid='a', received=None, **data):
        self.sequence += 1
        e = Event('s', self.sequence, kind, round(t*NS), round((t if received is None else received)*NS),
                  'run', 'ep', rid, data.pop('server_request_id', None), data)
        self.ledger.ingest([e])
        return e

    def plan_send(self, rid='a', sent=0):
        self.add('planned', 0, rid, scheduled_ns=round(sent*NS))
        self.add('sent', sent, rid)

    def output(self, t, n, rid='a', **kwargs):
        return self.add('client_output', t, rid, cumulative_tokens=n, **kwargs)

    def terminal(self, t, rid='a', **kwargs):
        return self.add('client_terminal', t, rid, **({'outcome': 'success', 'http_status': 200} | kwargs))

    def ack(self, through=8, pending=0, durable=True):
        # Fixture manifest is the explicitly constructed offered cohort.
        if self.ledger.expected_request_ids is None:
            self.ledger.expected_request_ids = frozenset(e.request_id for e in self.ledger.events.values() if e.kind == 'planned')
        self.ledger.acknowledge(Checkpoint('s', self.sequence, round(through*NS),
                                          round(through*NS), pending, durable))


class AccountingTests(unittest.TestCase):
    def test_hand_computed_windows_and_cleanup_are_immutable(self):
        f = Fixture()
        f.plan_send('a', 0); f.output(1, 1); f.output(1.3, 4); f.terminal(2)
        f.plan_send('b', 1); f.output(2, 1, 'b'); f.terminal(4, 'b')
        f.plan_send('c', 2); f.output(3, 2, 'c')
        f.add('planned', 0, 'not_sent', scheduled_ns=3*NS)
        f.ack()
        score = f.ledger.freeze_score()
        self.assertEqual(score['status'], 'passed')
        fixed, arrival = score['fixed'], score['arrival']
        self.assertEqual((fixed['offered'], fixed['sent'], fixed['completed']), (4, 3, 2))
        self.assertEqual(fixed['completion_rate'], .5)
        self.assertEqual(fixed['slo_goodput_rps'], 2/8)
        self.assertEqual(fixed['client_output_tokens'], 7)
        self.assertEqual(fixed['client_token_throughput_tps'], 7/8)
        self.assertEqual(fixed['outstanding_request_seconds'], 11)
        self.assertEqual(fixed['output_gap_request_seconds'], 5)
        self.assertEqual(fixed['overdue_first_request_seconds'], 0)
        self.assertEqual(fixed['outstanding_ids'], ['c'])
        self.assertEqual(fixed['tpot_conditional_ms'], [100.])
        self.assertEqual(fixed['single_token_completed'], 1)
        self.assertEqual(arrival['completed'], 2)
        self.assertEqual(arrival['outstanding_request_seconds'], 7)
        f.output(9, 4, 'c'); f.terminal(10, 'c'); f.ack(12)
        self.assertEqual(f.ledger.freeze_score(), score)
        with tempfile.TemporaryDirectory() as directory:
            result = f.ledger.export(Path(directory), cleanup_through_ns=12*NS)
            self.assertEqual(result, {'score_status': 'passed', 'cleanup_status': 'passed'})
            cleanup = json.loads((Path(directory)/'cleanup.json').read_text())
            self.assertEqual(cleanup['late_completed'], 1)
            self.assertEqual(len((Path(directory)/'requests.jsonl').read_text().splitlines()), 4)

    def test_deadline_before_at_after_and_single_token(self):
        for t, expected in [(8-1e-9, 1), (8, 1), (8+1e-9, 0)]:
            with self.subTest(t=t):
                f=Fixture(); f.plan_send(); f.output(1,1); f.terminal(t); f.ack(9)
                score=f.ledger.freeze_score()['fixed']
                self.assertEqual(score['completed'], expected)
                self.assertEqual(score['single_token_completed'], expected)
                self.assertEqual(score['tpot_conditional_ms'], [])
        f=Fixture(); f.plan_send(); f.output(0,1); f.terminal(0); f.ack()
        self.assertEqual(f.ledger.freeze_score()['fixed']['client_output_tokens'],1)

    def test_late_receipt_never_backfills_historical_observation(self):
        f=Fixture(); f.plan_send(); f.output(1,2,received=6); f.terminal(2,received=6)
        self.assertEqual(f.ledger.observe(3*NS)['outstanding'], 1)
        f.ack(8)
        self.assertEqual(f.ledger.freeze_score()['fixed']['completed'], 1)
        self.assertEqual(f.ledger.observe(3*NS)['outstanding'], 1)
        self.assertEqual(f.ledger.observe(6*NS)['outstanding'], 0)

    def test_duplicates_do_not_multiply_tokens_or_terminal(self):
        f=Fixture(); f.plan_send(); e=f.output(1,3)
        f.ledger.ingest([e, replace(e, received_ns=2*NS)])
        f.output(1.5,3); t=f.terminal(2); f.ledger.ingest([t]); f.terminal(2); f.ack()
        score=f.ledger.freeze_score()
        self.assertEqual(score['status'],'passed')
        self.assertEqual(score['fixed']['client_output_tokens'],3)
        self.assertEqual(score['fixed']['completed'],1)
        self.assertEqual(f.ledger.observe(2*NS)['requests'][0]['client_last_ns'],NS)

    def test_regression_is_not_new_tokens_and_blocks_completeness(self):
        f=Fixture(); f.plan_send(); f.output(1,5); f.output(2,3); f.output(3,6); f.terminal(4); f.ack()
        score=f.ledger.freeze_score()
        self.assertEqual(score['fixed']['client_output_tokens'],6)
        self.assertEqual(score['status'],'incomplete')
        self.assertIn('token_count_regression:a',score['audit']['issues'])

    def test_all_sent_hold_until_client_terminal_even_after_engine_finished(self):
        f=Fixture(); f.plan_send(); f.add('accepted', .1, server_request_id='backend')
        f.add('server_terminal',1,outcome='success'); f.output(2,1); f.terminal(5); f.ack()
        self.assertEqual(f.ledger.freeze_score()['fixed']['outstanding_request_seconds'],5)
        self.assertEqual(f.ledger.observe(3*NS)['outstanding'],1)

    def test_first_and_gap_integrals_hand_calculated(self):
        f=Fixture(); f.plan_send(); f.output(5,1); f.terminal(8); f.ack()
        fixed=f.ledger.freeze_score()['fixed']
        self.assertEqual(fixed['overdue_first_request_seconds'],3) # [2,5]
        self.assertEqual(fixed['output_gap_request_seconds'],2)    # [6,8]
        self.assertEqual(fixed['longest_no_output_ms'],3000)

    def test_manual_499_timeout_and_unknown_are_separate(self):
        f=Fixture()
        for name in ('manual','timeout','unknown'):
            f.plan_send(name)
        f.add('cancel_requested',2,'manual',reason='manual_drain')
        f.terminal(3,'manual',outcome='cancelled',http_status=499,reason='manual_drain')
        f.add('cancel_ack',4,'manual',confirmed=True)
        f.terminal(3,'timeout',outcome='timeout',http_status=None)
        f.terminal(3,'unknown',outcome='cancelled',http_status=499)
        f.ack()
        score=f.ledger.freeze_score()
        self.assertEqual(score['fixed']['terminal_outcomes'],{'manual_cancel':1,'timeout':1,'unknown':1})
        self.assertEqual(score['status'],'incomplete')

    def test_unconfirmed_cancellation_blocks_cleanup(self):
        f=Fixture(); f.plan_send(); f.add('cancel_requested',8)
        f.terminal(9,outcome='cancelled',http_status=499,reason='manual_drain'); f.ack(10)
        self.assertEqual(f.ledger.freeze_score()['status'],'passed')
        self.assertEqual(f.ledger.observe(10*NS)['requests'][0]['outcome'],'unknown')
        self.assertIn('cancel_unconfirmed:a',f.ledger.audit(through_ns=10*NS,require_cleanup=True)['issues'])
        f.add('cancel_ack',11,confirmed=False); f.ack(12)
        self.assertEqual(f.ledger.audit(through_ns=12*NS,require_cleanup=True)['status'],'incomplete')

    def test_missing_terminal_is_censored_at_deadline_but_cleanup_incomplete(self):
        f=Fixture(); f.plan_send(); f.ack()
        self.assertEqual(f.ledger.freeze_score()['fixed']['outstanding_ids'],['a'])
        self.assertEqual(f.ledger.freeze_score()['status'],'passed')
        self.assertIn('missing_client_terminal:a',f.ledger.audit(require_cleanup=True)['issues'])

    def test_empty_engine_queue_does_not_confirm_pending_delivery(self):
        f=Fixture(); f.plan_send(); f.output(1,1); f.terminal(2); f.ack(pending=1)
        self.assertEqual(f.ledger.freeze_score()['status'],'passed')
        self.assertIn('delivery_not_drained:s',f.ledger.audit(require_cleanup=True)['issues'])

    def test_sequence_gap_late_missing_tail_and_flush_required(self):
        f=Fixture(); f.plan_send(); f.output(1,1); f.terminal(2)
        missing=f.ledger.events.pop(('s',3)); f.ack()
        self.assertEqual(f.ledger.freeze_score()['status'],'incomplete')
        self.assertEqual(f.ledger.audit()['sequence_gaps'],{'s':[3]})
        f.ledger.ingest([missing]); self.assertEqual(f.ledger.freeze_score()['status'],'passed')
        g=Fixture(); g.plan_send(); g.ack(durable=False)
        self.assertEqual(g.ledger.freeze_score()['status'],'incomplete')
        g.ledger.acknowledge(Checkpoint('s',3,8*NS,8*NS,0,True))
        self.assertEqual(g.ledger.audit()['sequence_gaps'],{'s':[3]})

    def test_new_event_behind_watermark_is_detected_and_score_stays_frozen(self):
        f=Fixture(); f.plan_send(); f.ack(); score=f.ledger.freeze_score()
        f.output(1,1)
        self.assertEqual(f.ledger.freeze_score(),score)
        self.assertIn('event_behind_watermark',','.join(f.ledger.audit()['issues']))

    def test_sampled_counts_never_become_client_tokens(self):
        f=Fixture(); f.plan_send(); f.output(1,64,sampled=True); f.terminal(2); f.ack()
        score=f.ledger.freeze_score()
        self.assertEqual(score['fixed']['client_output_tokens'],0)
        self.assertEqual(score['status'],'incomplete')
        self.assertFalse(score['fixed']['exact_token_itl_available'])

    def test_retry_conflict_identity_mapping_and_unknown_source(self):
        f=Fixture(); f.plan_send(); f.add('sent',1); f.add('accepted',1,server_request_id='x')
        f.output(2,1,server_request_id='y'); f.terminal(3); f.ack()
        issues=f.ledger.audit()['issues']
        self.assertIn('conflicting_sent:a',issues)
        self.assertIn('ambiguous_server_mapping:a',issues)
        event=Event('foreign',1,'diagnostic',0,0,'run','ep')
        f.ledger.ingest([event]); self.assertIn('unregistered_source:foreign',f.ledger.audit()['issues'])

    def test_conflicting_duplicate_sequence_and_clock_validation(self):
        f=Fixture(); f.plan_send(); e=f.output(1,2)
        f.ledger.ingest([replace(e,data={'cumulative_tokens':8})])
        self.assertEqual(f.ledger.audit()['status'],'incomplete')
        with self.assertRaises(ValueError):
            replace(e,received_ns=0)

    def test_offered_manifest_missing_or_mismatched_is_incomplete(self):
        f=Fixture(); f.plan_send()
        self.assertIn('offered_request_manifest_missing', f.ledger.audit()['issues'])
        f.ledger.expected_request_ids=frozenset(['a','missing_from_events'])
        f.ack()
        self.assertIn('offered_request_manifest_mismatch', f.ledger.audit()['issues'])

    def test_api_and_engine_first_are_not_client_first(self):
        f=Fixture(); f.plan_send(); f.add('engine_first',1); f.add('api_first',2); f.ack()
        fixed=f.ledger.freeze_score()['fixed']
        self.assertEqual(fixed['first_output_censored'],1)
        self.assertEqual(fixed['client_output_tokens'],0)
        self.assertEqual(fixed['overdue_first_request_seconds'],6)

    def test_accepted_requests_need_separate_server_terminal_for_cleanup(self):
        f=Fixture(); f.plan_send(); f.add('accepted',.1,server_request_id='backend')
        f.output(1,1); f.terminal(2); f.ack()
        self.assertIn('missing_server_terminal:a',f.ledger.audit(require_cleanup=True)['issues'])


class IOTests(unittest.TestCase):
    def test_journal_partial_cursor_receipt_and_watermark(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'events.jsonl'; journal=EventJournal(path,'run','ep')
            event=journal.record('planned','a',scheduled_ns=1)
            cp=journal.checkpoint(pending_delivery=0)
            cursor=EventCursor(); arrival=time.monotonic_ns()
            events=cursor.poll(path,received_ns=arrival)
            self.assertEqual(len(events),1); self.assertEqual(events[0].received_ns,arrival)
            self.assertEqual(cursor.poll(path),[])
            self.assertTrue(cursor.checkpoint(path.with_suffix('.jsonl.checkpoint.json')).durable)
            with self.assertRaises(ValueError):journal.record('sent','a',occurred_ns=cp.through_ns)
            journal.close()
            # A split JSONL record is not consumed until its newline is present.
            raw=(json.dumps(asdict(event))+'\n').encode()
            with path.open('ab') as stream:stream.write(raw[:15])
            self.assertEqual(cursor.poll(path),[])
            with path.open('ab') as stream:stream.write(raw[15:])
            self.assertEqual(len(cursor.poll(path)),1)
            path.write_bytes(b'')
            self.assertEqual(cursor.poll(path),[])
            self.assertTrue(cursor.issues)

    def test_decoder_fragmented_utf8_coalesced_frames_and_truncated_eof(self):
        decoder=NulJSONDecoder(); wire=json.dumps({'text':'中文'},ensure_ascii=False).encode()+b'\0'+b'{"x":2}\0'
        records=[]
        for byte in wire: records.extend(decoder.feed(bytes([byte])))
        decoder.finish(); self.assertEqual(records,[{'text':'中文'},{'x':2}])
        decoder.feed(b'{')
        with self.assertRaises(ValueError):decoder.finish()

    def test_adapter_keeps_sequences_and_diagnostic_samples(self):
        adapter=CausalEventAdapter('run','ep',{'backend':'a'})
        base={'host':'h','pid':1,'monotonic_ns':NS,'request_id':'backend'}
        raw=base|{'event_sequence':1,'event':'output_queue_dequeue','output_tokens':64}
        event=adapter.adapt(raw,received_ns=2*NS)
        self.assertEqual(event.kind,'diagnostic')
        raw=base|{'event_sequence':2,'event':'engine_request_output','output_tokens':1}
        self.assertEqual(adapter.adapt(raw,received_ns=2*NS).kind,'engine_first')
        raw['event_sequence']=3
        self.assertEqual(adapter.adapt(raw,received_ns=2*NS).kind,'diagnostic')

    def test_migration_missing_stages_are_not_inferred(self):
        def e(kind,t,**kw):return {'event':kind,'monotonic_ns':int(t*NS),'migration_id':'m','request_id':'r','target_instance_id':'dst',**kw}
        records=[e('migration_attempt_started',1),e('migration_attempt_completed',5)]
        row=migration_diagnostics(records)[0]
        self.assertEqual(row['total_ms'],4000)
        self.assertIsNone(row['source_pause_ms']); self.assertIsNone(row['target_commit_ms'])
        records += [e('migration_request_pause_started',2),e('migration_target_commit_started',3),
                    e('migration_request_resumed',4,resume_reason='destination_commit_acknowledged'),
                    e('engine_request_output',4.5,instance_id='dst',output_tokens=2)]
        row=migration_diagnostics(records)[0]
        self.assertEqual(row['source_pause_ms'],2000)
        self.assertEqual(row['target_commit_ms'],1000)
        self.assertEqual(row['post_commit_first_engine_output_ms'],500)

    def test_producer_flush_is_durable_and_thread_sequence_ordered(self):
        module=load_file('_real_causal_test','llumnix/metrics/causal_events.py')
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{'LLUMNIX_CAUSAL_TRACE_DIR':d}):
            threads=[threading.Thread(target=lambda:[module.emit('diagnostic',n=i) for i in range(40)]) for _ in range(4)]
            for thread in threads:thread.start()
            for thread in threads:thread.join()
            ack=module.flush(5)
            self.assertEqual(ack['status'],'passed'); self.assertTrue(ack['durable'])
            rows=[json.loads(line) for line in next(Path(d).glob('events*.jsonl')).read_text().splitlines()]
            self.assertEqual([r['event_sequence'] for r in rows],list(range(1,161)))
            self.assertEqual(ack['last_sequence'],160)
            self.assertTrue(all(r['stream_id']==ack['source'] for r in rows))
            module._WRITER.close()
            self.assertEqual(module.flush()['status'],'incomplete')
        with patch.dict(os.environ,{},clear=True):
            self.assertEqual(module.flush()['status'],'incomplete')


class ClientHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_http_fragmentation_and_incomplete_eof(self):
        from aiohttp import web, ClientSession, ClientTimeout
        from aiohttp.test_utils import TestServer
        async def handle(request):
            body=await request.json(); rid=body['request_id']
            response=web.StreamResponse(); await response.prepare(request)
            def frame(kind,**kw):return (json.dumps({'run_id':'run','episode_id':'ep','request_id':rid,
                                        'server_request_id':'server-'+rid,'kind':kind,**kw})+'\0').encode()
            data=frame('accepted')+frame('output',cumulative_tokens=3)+frame('output',cumulative_tokens=3)
            if rid!='broken':data+=frame('output',cumulative_tokens=5)+frame('terminal',outcome='success')
            for index in range(0,len(data),17):
                await response.write(data[index:index+17]); await asyncio.sleep(0)
            return response
        app=web.Application(); app.router.add_post('/generate',handle)
        with tempfile.TemporaryDirectory() as d:
            journal=EventJournal(Path(d)/'events.jsonl','run','ep'); client=StreamingRequestCollector(journal)
            async with TestServer(app) as server, ClientSession(timeout=ClientTimeout(total=5)) as session:
                for rid in ['ok','broken']:
                    journal.record('planned',rid,scheduled_ns=time.monotonic_ns())
                    await client.issue(session,str(server.make_url('/generate')),rid,{'prompt':'x'})
                with self.assertRaises(ValueError):await client.issue(session,str(server.make_url('/generate')),'ok',{})
            journal.checkpoint(); journal.close(); events=EventCursor().poll(Path(d)/'events.jsonl')
            outcomes={e.request_id:e.data['outcome'] for e in events if e.kind=='client_terminal'}
            self.assertEqual(outcomes,{'ok':'success','broken':'unknown'})
            counts=[e.data['cumulative_tokens'] for e in events if e.kind=='client_output' and e.request_id=='ok']
            self.assertEqual(counts,[3,3,5])
            self.assertEqual(len([e for e in events if e.kind=='client_first' and e.request_id=='ok']),1)

    async def test_client_timeout_has_terminal(self):
        from aiohttp import web, ClientSession, ClientTimeout
        from aiohttp.test_utils import TestServer
        async def handle(request):
            await asyncio.sleep(.04); return web.Response(text='')
        app=web.Application(); app.router.add_post('/',handle)
        with tempfile.TemporaryDirectory() as d:
            journal=EventJournal(Path(d)/'events','run','ep'); client=StreamingRequestCollector(journal)
            async with TestServer(app) as server, ClientSession(timeout=ClientTimeout(total=.005)) as session:
                await client.issue(session,str(server.make_url('/')),'a',{})
            journal.checkpoint(); journal.close(); events=EventCursor().poll(Path(d)/'events')
            self.assertEqual(events[-1].data['outcome'],'timeout')


class ServerRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_backend_ack_and_stream_exit_required_for_cancel(self):
        import httpx
        from fastapi import FastAPI
        module=load_file('_real_cancel_test','llumnix/entrypoints/vllm/real_metrics_stream.py')
        ready,stopped=asyncio.Event(),asyncio.Event()
        class FakeClient:
            async def generate(self,prompt,params,request_id):
                ready.set()
                async def generator():
                    await stopped.wait()
                    if False:yield None
                return SimpleNamespace(generator=generator)
        async def barrier(request_id):
            stopped.set(); return True
        events=[]; app=FastAPI()
        module.install_real_metrics_routes(app,lambda:FakeClient(),lambda **kw:kw,
            lambda event,**kw:events.append(kw),lambda timeout:{'status':'incomplete'},barrier)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            body={'run_id':'run','episode_id':'ep','request_id':'a','prompt':'x'}
            task=asyncio.create_task(client.post('/generate_real_metrics',json=body))
            await asyncio.wait_for(ready.wait(),2)
            result=await client.post('/cancel_real_metrics',json={k:body[k] for k in ('run_id','episode_id')}|{'request_ids':['a']})
            self.assertEqual(result.json()['results'][0]['status'],'cancelled_confirmed')
            response=await asyncio.wait_for(task,2)
            frames=NulJSONDecoder().feed(response.content)
            self.assertEqual(frames[-1]['outcome'],'cancelled')
            self.assertEqual([e['data']['outcome'] for e in events if e['metric_kind']=='server_terminal'],['cancelled'])

    async def test_stream_frames_ids_tokens_terminal_duplicate_and_unknown_abort(self):
        import httpx
        from fastapi import FastAPI
        module=load_file('_real_stream_test','llumnix/entrypoints/vllm/real_metrics_stream.py')
        class FakeClient:
            async def generate(self,prompt,params,request_id):
                async def generator():
                    for count,finished in [(1,False),(4,True)]:
                        output=SimpleNamespace(outputs=[SimpleNamespace(token_ids=list(range(count)),text='x'*count)],finished=finished)
                        yield SimpleNamespace(get_engine_output=lambda output=output:output)
                return SimpleNamespace(generator=generator)
        events=[]; app=FastAPI()
        module.install_real_metrics_routes(app,lambda:FakeClient(),lambda **kw:kw,
                                           lambda event,**kw:events.append(kw),lambda timeout:{'status':'incomplete'})
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            body={'run_id':'run','episode_id':'ep','request_id':'a','prompt':'private','max_tokens':4}
            response=await client.post('/generate_real_metrics',json=body)
            self.assertEqual(response.status_code,200)
            decoder=NulJSONDecoder(); frames=decoder.feed(response.content); decoder.finish()
            self.assertEqual([f['kind'] for f in frames],['accepted','output','output','terminal'])
            self.assertEqual([f['cumulative_tokens'] for f in frames if f['kind']=='output'],[1,4])
            self.assertEqual(frames[-1]['outcome'],'success')
            self.assertEqual((await client.post('/generate_real_metrics',json=body)).status_code,409)
            result=await client.post('/cancel_real_metrics',json={'run_id':'run','episode_id':'ep','request_ids':['missing']})
            self.assertEqual(result.json()['results'][0]['status'],'unknown')
            self.assertEqual((await client.get('/real_metrics_flush')).json()['status'],'incomplete')
            self.assertNotIn('private',json.dumps(events))
        self.assertEqual([e['metric_kind'] for e in events],['accepted','api_first','server_terminal'])


if __name__ == '__main__':
    unittest.main()
