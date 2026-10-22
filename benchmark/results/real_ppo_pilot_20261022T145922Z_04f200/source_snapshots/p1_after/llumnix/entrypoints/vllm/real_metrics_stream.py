"""Optional complete output-count stream for real metrics; old routes are unchanged.

No token text, prompt text, model parameters or credentials enter telemetry.
These routes are installed on the existing internal API, not a new listener.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse


def install_real_metrics_routes(app, get_client, sampling_factory, emit, flush, abort_confirmed=None):
    active: dict[tuple, dict[str, Any]] = {}
    seen: set[tuple] = set()

    def publish(state, kind, **data):
        emit('real_metrics_event', metric_kind=kind, run_id=state['run_id'],
             episode_id=state['episode_id'], request_id=state['request_id'],
             server_request_id=state['server_request_id'], data=data)

    @app.post('/generate_real_metrics')
    async def generate_real_metrics(request: Request):
        body = await request.json()
        run_id, episode_id, request_id = (body.pop(name, None) for name in
                                         ('run_id', 'episode_id', 'request_id'))
        if not all(isinstance(value, str) and value for value in (run_id, episode_id, request_id)):
            return JSONResponse({'error': 'run/episode/request IDs required'}, status_code=400)
        key = (run_id, episode_id, request_id)
        if key in seen:
            return JSONResponse({'error': 'duplicate request; retries are not dispatched'}, status_code=409)
        if 'prompt' not in body:
            return JSONResponse({'error': 'prompt required'}, status_code=400)
        prompt = body.pop('prompt')
        body.pop('stream', None)
        if body.get('n', 1) != 1 or body.get('best_of', 1) != 1:
            return JSONResponse({'error': 'metrics stream requires n=best_of=1'}, status_code=400)
        try:
            params = sampling_factory(**body)
        except (TypeError, ValueError):
            return JSONResponse({'error': 'invalid sampling parameters'}, status_code=400)
        # Namespace server IDs so two episodes cannot alias a backend request.
        import hashlib
        server_id = 'real-' + hashlib.sha256(json.dumps(key).encode()).hexdigest()
        state = {'run_id': run_id, 'episode_id': episode_id, 'request_id': request_id,
                 'server_request_id': server_id, 'done': asyncio.Event(), 'cancel': False,
                 'ready': asyncio.Event(), 'outcome': 'unknown', 'tokens': 0}
        seen.add(key)
        active[key] = state

        def frame(kind, **data):
            return (json.dumps({name: state[name] for name in
                                ('run_id', 'episode_id', 'request_id', 'server_request_id')} |
                               {'kind': kind, **data}, allow_nan=False) + '\0').encode()

        async def stream():
            state['task'] = asyncio.current_task()
            try:
                generator = await get_client().generate(prompt, params, server_id)
                publish(state, 'accepted')
                state['ready'].set()
                yield frame('accepted')
                first = True
                async for wrapped in generator.generator():
                    output = wrapped.get_engine_output()
                    count = len(output.outputs[0].token_ids) if output.outputs else 0
                    state['tokens'] = count
                    if count and first:
                        publish(state, 'api_first')
                        first = False
                    yield frame('output', cumulative_tokens=count, text=output.outputs[0].text if output.outputs else '')
                    if output.finished:
                        state['outcome'] = 'success'
                if state['outcome'] != 'success' and state['cancel']:
                    state['outcome'] = 'cancelled'
                publish(state, 'server_terminal', outcome=state['outcome'])
                yield frame('terminal', outcome=state['outcome'],
                            reason='manual_drain' if state['cancel'] else None)
            except asyncio.CancelledError:
                # A disconnected HTTP consumer is not proof of backend cancellation.
                state['outcome'] = 'unknown'
                publish(state, 'server_terminal', outcome='unknown', reason='stream_disconnected')
                raise
            except Exception as exc:
                state['outcome'] = 'service_error'
                publish(state, 'server_terminal', outcome='service_error', error_type=type(exc).__name__)
                yield frame('terminal', outcome='service_error')
            finally:
                state['ready'].set()
                state['done'].set()
        return StreamingResponse(stream(), media_type='application/x-ndjson')

    @app.post('/cancel_real_metrics')
    async def cancel_real_metrics(request: Request):
        body = await request.json()
        results = []
        # Entire batch shares a 55 s server deadline, leaving room within client 60 s.
        deadline = time.monotonic() + 55
        for request_id in dict.fromkeys(body.get('request_ids', [])):
            key = (body.get('run_id'), body.get('episode_id'), request_id)
            state = active.get(key)
            status = 'unknown'
            if state:
                if state['done'].is_set() and state['outcome'] == 'success':
                    status = 'already_terminal'
                else:
                    state['cancel'] = True
                    publish(state, 'diagnostic', stage='cancel_requested', reason='manual_drain')
                    try:
                        await asyncio.wait_for(state['ready'].wait(), max(0., deadline-time.monotonic()))
                        if abort_confirmed is None:
                            status = 'unconfirmed_no_barrier'
                        else:
                            confirmed = await asyncio.wait_for(abort_confirmed(state['server_request_id']), max(0., deadline-time.monotonic()))
                            await asyncio.wait_for(state['done'].wait(), max(0., deadline-time.monotonic()))
                            status = ('cancelled_confirmed' if confirmed and state['outcome'] == 'cancelled'
                                      else 'already_terminal' if state['outcome'] == 'success' else 'unknown')
                    except asyncio.TimeoutError:
                        status = 'unconfirmed_timeout'
                    except Exception:
                        status = 'unconfirmed_error'
                publish(state, 'diagnostic', stage='cancel_ack', confirmed=status == 'cancelled_confirmed', status=status)
            # Unknown lookup does not cancel a future request and is never confirmed.
            results.append({'request_id': request_id, 'status': status,
                            'server_request_id': state['server_request_id'] if state else None})
        return JSONResponse({'results': results})

    @app.get('/real_metrics_flush')
    async def real_metrics_flush():
        # This confirms this API process only; runner must collect all producers.
        result = await asyncio.to_thread(flush, 5.)
        return JSONResponse(result)
