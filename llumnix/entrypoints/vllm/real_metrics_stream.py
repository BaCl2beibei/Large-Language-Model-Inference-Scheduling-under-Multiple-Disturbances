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
from llumnix.metrics.causal_events import OUTPUT_UPDATE_SAMPLE_INTERVAL, output_update_sample_due


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
        initial_instance_id = body.pop('initial_instance_id', None)
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
                 'ready': asyncio.Event(), 'outcome': 'unknown', 'tokens': 0,
                 'input_tokens': None, 'streamed_tokens': 0,
                 'cancel_checked': asyncio.Event(), 'cancel_proven': False}
        seen.add(key)
        active[key] = state

        def frame(kind, **data):
            return (json.dumps({name: state[name] for name in
                                ('run_id', 'episode_id', 'request_id', 'server_request_id')} |
                               {'kind': kind, **data}, allow_nan=False) + '\0').encode()

        async def stream():
            state['task'] = asyncio.current_task()
            try:
                route_kwargs = {'real_initial_instance_id': initial_instance_id} if initial_instance_id is not None else {}
                generator = await get_client().generate(prompt, params, server_id, **route_kwargs)
                publish(state, 'accepted')
                state['ready'].set()
                yield frame('accepted')
                first = True
                async for wrapped in generator.generator():
                    output = wrapped.get_engine_output()
                    state['input_tokens'] = len(getattr(output, 'prompt_token_ids', []))
                    count = len(output.outputs[0].token_ids) if output.outputs else 0
                    state['tokens'] = count
                    if count and first:
                        publish(state, 'api_first')
                        first = False
                    # The full-completion collector consumes only cumulative token
                    # counts. Re-sending the entire growing text prefix on every
                    # decode step makes long cohorts transfer O(tokens^2) bytes
                    # and can starve output delivery and controller heartbeats.
                    if (output_update_sample_due(count, output.finished)
                            and (count != state['streamed_tokens'] or output.finished)):
                        state['streamed_tokens'] = count
                        yield frame('output', cumulative_tokens=count,
                                    finished=bool(output.finished))
                    if output.finished:
                        state['outcome'] = 'success'
                if state['outcome'] != 'success' and state['cancel']:
                    await state['cancel_checked'].wait()
                    state['outcome'] = 'cancelled' if state['cancel_proven'] else 'unknown'
                publish(state, 'server_terminal', outcome=state['outcome'],
                        input_tokens=state['input_tokens'], output_tokens=state['tokens'])
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
        request_ids = list(dict.fromkeys(body.get('request_ids', [])))
        results = {}
        pending = []
        # Cleanup can begin with a large overloaded cohort. Give the shared
        # batch enough time for requests to finish naturally before confirming
        # cancellation, while keeping a finite bound for service shutdown.
        deadline = time.monotonic() + 205
        # Fence every stream before waiting on any backend. A large overloaded
        # episode can have hundreds of outstanding requests; serially awaiting
        # each backend proof spends the whole cleanup window on the first few.
        for request_id in request_ids:
            key = (body.get('run_id'), body.get('episode_id'), request_id)
            state = active.get(key)
            if state is None:
                # Unknown lookup does not cancel a future request and is never confirmed.
                results[request_id] = {'request_id': request_id, 'status': 'unknown',
                                       'server_request_id': None}
            elif state['done'].is_set() and state['outcome'] == 'success':
                status = 'already_terminal'
                publish(state, 'diagnostic', stage='cancel_ack', confirmed=False, status=status)
                results[request_id] = {'request_id': request_id, 'status': status,
                                       'server_request_id': state['server_request_id']}
            else:
                state['cancel'] = True
                publish(state, 'diagnostic', stage='cancel_requested', reason='manual_drain')
                pending.append((request_id, state))

        semaphore = asyncio.Semaphore(8)

        async def confirm(request_id, state):
            async with semaphore:
                status = 'unknown'
                error_detail = None
                try:
                    # A request may finish while it waits behind other
                    # confirmations. Its successful terminal is already a
                    # stronger outcome than a cancellation proof.
                    if state['done'].is_set() and state['outcome'] == 'success':
                        status = 'already_terminal'
                        return
                    await asyncio.wait_for(state['ready'].wait(),
                                           max(.001, deadline-time.monotonic()))
                    if abort_confirmed is None:
                        status = 'unconfirmed_no_barrier'
                    else:
                        confirmed = await asyncio.wait_for(
                            abort_confirmed(state['server_request_id']),
                            max(.001, deadline-time.monotonic()))
                        state['cancel_proven'] = confirmed is True
                        await asyncio.wait_for(state['done'].wait(),
                                               max(.001, deadline-time.monotonic()))
                        status = ('cancelled_confirmed' if confirmed and state['outcome'] == 'cancelled'
                                  else 'already_terminal' if state['outcome'] == 'success' else 'unknown')
                except asyncio.TimeoutError:
                    status = 'unconfirmed_timeout'
                except Exception as exc:
                    status = 'unconfirmed_error'
                    error_detail = f'{type(exc).__name__}: {str(exc)[:256]}'
                finally:
                    state['cancel_checked'].set()
                    publish(state, 'diagnostic', stage='cancel_ack',
                            confirmed=status == 'cancelled_confirmed', status=status,
                            error=error_detail)
                    results[request_id] = {'request_id': request_id, 'status': status,
                                           'server_request_id': state['server_request_id'],
                                           'error': error_detail}

        await asyncio.gather(*(confirm(request_id, state) for request_id, state in pending))
        return JSONResponse({'results': [results[request_id] for request_id in request_ids]})

    @app.get('/real_metrics_flush')
    async def real_metrics_flush():
        # This confirms this API process only; runner must collect all producers.
        result = await asyncio.to_thread(flush, 5.)
        return JSONResponse(result)

    return active
