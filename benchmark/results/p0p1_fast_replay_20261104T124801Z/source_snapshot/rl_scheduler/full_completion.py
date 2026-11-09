"""Full-completion Balance-vs-Stay workload and measurement path.

This module deliberately does not call the legacy fixed-window scorer. A
complete frozen request cohort is the measurement unit; an incomplete run has
null primary latency values and remains in the artifact ledger.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import time

from .real_bridge import RealBridge, digest
from .real_budget import atomic_json
from .real_env import assert_idle
from .real_metrics import _percentile
from .real_observation import RealObservationBuilder
from .real_opportunity import balanced_action
from .real_opportunity_runner import OpportunityLiveEpisodeIO


def percentile_type7(values, p):
    values = sorted(float(x) for x in values)
    if not values:
        return None
    h = (len(values) - 1) * p
    lo, hi = math.floor(h), math.ceil(h)
    return values[lo] + (values[hi] - values[lo]) * (h - lo)


def cvar95(values):
    """Empirical upper-tail mean with exact fractional boundary weight."""
    values = sorted(float(x) for x in values)
    if not values:
        return None
    alpha = 0.05
    tail_mass = alpha * len(values)
    boundary = values[max(0, math.ceil((1.0 - alpha) * len(values)) - 1)]
    excess = sum(max(value - boundary, 0.0) for value in values)
    # This Rockafellar-Uryasev form handles a quantile tie without rounding N*.05.
    return boundary + excess / tail_mass


def summarize_values(values):
    values = [float(x) for x in values]
    if not values:
        return {key: None for key in ('mean', 'median', 'p90', 'p95', 'p99', 'max', 'cvar95')}
    return {
        'mean': sum(values) / len(values),
        'median': percentile_type7(values, .5),
        'p90': percentile_type7(values, .90),
        'p95': percentile_type7(values, .95),
        'p99': percentile_type7(values, .99),
        'max': max(values),
        'cvar95': cvar95(values),
    }


_PROMPT_WORDS = ('hello', 'world', 'request', 'queue', 'model', 'system', 'latency',
                 'compute', 'alpha', 'beta', 'service', 'small', 'fast', 'stable',
                 'token', 'worker', 'batch', 'stream', 'result', 'sample')


def exact_prompt(tokenizer, target_tokens: int, seed: int) -> tuple[str, int]:
    """Build and re-tokenize plain-text prompts at the requested model token count."""
    rng = random.Random(seed)
    pool = [word for word in _PROMPT_WORDS
            if len(tokenizer.encode(word).ids) == 1 and
            len(tokenizer.encode(' ' + word).ids) == 1]
    if not pool:
        pool = ['hello']
    for _ in range(100):
        text = ' '.join(rng.choice(pool) for _ in range(target_tokens))
        count = len(tokenizer.encode(text).ids)
        if count == target_tokens:
            return text, count
    text = ' '.join(['hello'] * target_tokens)
    count = len(tokenizer.encode(text).ids)
    if count != target_tokens:
        raise ValueError(f'tokenizer cannot materialize an exact {target_tokens}-token prompt')
    return text, count


def _uniform_times(n, horizon, rng):
    return sorted(rng.random() * horizon for _ in range(n))


def _segmented_times(n, horizon, burst, rng):
    """Inverse normalized intensity for 1× / burst× / 1× over 25/50/25%."""
    total_area = .5 + .5 * burst
    result = []
    for value in sorted(rng.random() for _ in range(n)):
        area = value * total_area
        if area < .25:
            fraction = area
        elif area < .25 + .5 * burst:
            fraction = .25 + (area - .25) / burst
        else:
            fraction = .75 + area - (.25 + .5 * burst)
        result.append(fraction * horizon)
    return result


def _routes_s1(n, seed, hot_fraction):
    rng = random.Random(seed ^ 0x51A1)
    hot = seed % 4
    indices = list(range(n))
    rng.shuffle(indices)
    hot_count = math.floor(n * hot_fraction + .5)
    routes = [None] * n
    for index in indices[:hot_count]:
        routes[index] = hot
    other = [gpu for gpu in range(4) if gpu != hot]
    for offset, index in enumerate(indices[hot_count:]):
        routes[index] = other[offset % 3]
    return routes, hot


def _profile_outputs(profile, n, seed, *, rng=None):
    rng = rng or random.Random(seed ^ 0xB1A0)
    if profile == 'decode1536':
        return [1536] * n
    if profile == 'decode512':
        return [512] * n
    if profile == 'decode64':
        return [64] * n
    if profile in ('bimodal_0.1', 'bimodal_0.2'):
        fraction = .1 if profile.endswith('_0.1') else .2
        long_count = math.floor(n * fraction + .5)
        picks = set(rng.sample(range(n), long_count))
        return [1536 if index in picks else 128 for index in range(n)]
    raise ValueError('unknown profile ' + profile)


def make_trace(*, candidate, n, seed, lambda_ref, tokenizer, profile_override=None,
               synchronous=False, episode_id_prefix='r'):
    """Materialize one immutable trace, including exact route and token quotas."""
    rng = random.Random(seed)
    profile = profile_override or candidate['profile']
    outputs = _profile_outputs(profile, n, seed)
    rho = candidate.get('rho', 1.0)
    lam = float(lambda_ref) * float(rho)
    if synchronous:
        times = [0.0] * n
    else:
        if not math.isfinite(lam) or lam <= 0:
            raise ValueError('positive measured lambda_ref is required')
        horizon = n / lam
        if candidate['family'] == 'S3':
            times = _segmented_times(n, horizon, float(candidate['burst_multiplier']), rng)
        else:
            times = _uniform_times(n, horizon, rng)

    if candidate['family'] == 'S1':
        routes, hot = _routes_s1(n, seed, float(candidate['hot_fraction']))
    elif candidate['family'] == 'S2':
        if n % 4:
            raise ValueError('S2 requires request count divisible by four')
        hot = seed % 4
        routes = [gpu for gpu in range(4) for _ in range(n // 4)]
        long_count = math.floor(n * float(candidate['long_fraction']) + .5)
        hot_long = math.floor(long_count * .8 + .5)
        route_indices = {gpu: [i for i, route in enumerate(routes) if route == gpu]
                         for gpu in range(4)}
        lengths = [128] * n
        picked = set(rng.sample(route_indices[hot], hot_long))
        remainder = long_count - hot_long
        other_gpus = [gpu for gpu in range(4) if gpu != hot]
        candidates = [index for gpu in other_gpus for index in route_indices[gpu]]
        rng.shuffle(candidates)
        picked.update(candidates[:remainder])
        for index in picked:
            lengths[index] = 1536
        outputs = lengths
    elif candidate['family'] == 'S3':
        hot = seed % 4
        routes = [None] * n
        horizon = n / lam
        for index, at in enumerate(times):
            current = hot if at < horizon / 2 else (hot + 1) % 4
            others = [gpu for gpu in range(4) if gpu != current]
            draw = rng.random()
            routes[index] = current if draw < .8 else others[min(2, int((draw - .8) / (.2 / 3)))]
    elif candidate['family'] == 'C0':
        hot = None
        routes = [index % 4 for index in range(n)]
    elif candidate['family'] == 'C1':
        routes, hot = _routes_s1(n, seed, float(candidate['hot_fraction']))
    else:
        raise ValueError('unknown candidate family ' + candidate['family'])

    trace = []
    for index, (at, route, output_tokens) in enumerate(zip(times, routes, outputs)):
        prompt, prompt_tokens = exact_prompt(tokenizer, 128, seed * 100000 + index)
        if prompt_tokens + output_tokens > 2112:
            raise ValueError('token budget exceeds model context length')
        trace.append({
            'request_id': f'{episode_id_prefix}{index:05d}',
            'scheduled_arrival_sec': round(at, 9),
            'request_type': profile,
            'prompt': prompt,
            'input_tokens': prompt_tokens,
            'output_tokens': output_tokens,
            'initial_instance_index': route,
            'hot_instance_index': hot,
        })
    return trace


class FullCompletionIO(OpportunityLiveEpisodeIO):
    """Request sender that waits for every planned request, including late sends."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prepared_requests = {}
        self._prepared_http_requests = {}
        self._sender_loop_count = 4
        self._sender_loops = []
        self._sender_threads = []
        self._sender_sessions = []

    async def _open_sender_session(self, sender_index):
        import aiohttp
        timeout = aiohttp.ClientTimeout(total=None, connect=30., sock_connect=30., sock_read=None)
        session = aiohttp.ClientSession(
            timeout=timeout, connector=aiohttp.TCPConnector(limit=512))
        self._sender_sessions[sender_index] = session
        if sender_index == 0:
            self._sender_session = session

    def _start_sender_pool(self):
        import threading
        if self._sender_loops:
            return
        self._sender_sessions = [None] * self._sender_loop_count
        for sender_index in range(self._sender_loop_count):
            loop = asyncio.new_event_loop()

            def run_loop(target_loop=loop):
                asyncio.set_event_loop(target_loop)
                target_loop.run_forever()

            thread = threading.Thread(
                target=run_loop, name=f'full-completion-sender-{sender_index}', daemon=True)
            thread.start()
            self._sender_loops.append(loop)
            self._sender_threads.append(thread)
            try:
                asyncio.run_coroutine_threadsafe(
                    self._open_sender_session(sender_index), loop).result(timeout=10.)
            except BaseException:
                loop.call_soon_threadsafe(loop.stop)
                thread.join(timeout=5.)
                loop.close()
                raise
        self._sender_loop = self._sender_loops[0]
        self._sender_thread = self._sender_threads[0]

    async def _prepare_http_requests(self, sender_index):
        session = self._sender_sessions[sender_index]
        for index, row in enumerate(self.trace):
            if index % self._sender_loop_count != sender_index:
                continue
            request_id = row['request_id']
            url, _payload, encoded = self._prepared_requests[request_id]
            self._prepared_http_requests[request_id] = session.post(
                url, data=encoded, headers={'Content-Type': 'application/json'})

    async def prepare(self):
        await super().prepare()
        url = self.bridge.base_url + '/generate_real_metrics'
        for row in self.trace:
            body = {**self.sampling, 'prompt': row['prompt'],
                    'max_tokens': row['output_tokens'], 'min_tokens': row['output_tokens']}
            gpu_index = row.get('initial_instance_index')
            if gpu_index is not None:
                body['initial_instance_id'] = self.mapping[gpu_index][0]
                self.expected_routes[row['request_id']] = body['initial_instance_id']
            payload = {**body, 'request_id': row['request_id'],
                       'run_id': self.journal.run_id, 'episode_id': self.journal.episode_id}
            encoded = json.dumps(payload, allow_nan=False, separators=(',', ':')).encode('utf-8')
            self._prepared_requests[row['request_id']] = (url, payload, encoded)
        await asyncio.to_thread(self._start_sender_pool)
        prepared = [asyncio.run_coroutine_threadsafe(
            self._prepare_http_requests(sender_index), self._sender_loops[sender_index])
            for sender_index in range(self._sender_loop_count)]
        await asyncio.gather(*(asyncio.wrap_future(future) for future in prepared))

    def ledger(self, start, arrival, _deadline):
        from .real_opportunity_runner import OpportunityRequestLedger
        return OpportunityRequestLedger(self.run_id, self.episode_id, start_ns=start,
            arrival_end_ns=arrival, score_deadline_ns=None, expected_sources=self.sources,
            expected_request_ids=[row['request_id'] for row in self.trace],
            sequence_baselines=self.sequence_baselines, full_completion=True)

    async def begin(self, start_ns, arrival_end_ns, deadline_ns, ledger):
        self.start_ns, self.arrival_end_ns, self.deadline_ns = start_ns, arrival_end_ns, None
        for row in self.trace:
            scheduled = start_ns + int(row['scheduled_arrival_sec'] * 1e9)
            self.journal.record('planned', row['request_id'], scheduled_ns=scheduled)
        launches = []
        for sender_index, loop in enumerate(self._sender_loops):
            rows = [row for index, row in enumerate(self.trace)
                    if index % self._sender_loop_count == sender_index]
            launches.append(asyncio.run_coroutine_threadsafe(
                self._launch_requests(rows, sender_index), loop))
        await asyncio.gather(*(asyncio.wrap_future(future) for future in launches))
        await self.poll(ledger)

    async def _launch_requests(self, rows, sender_index):
        import concurrent.futures
        for row in rows:
            completion = concurrent.futures.Future()
            self.tasks[row['request_id']] = completion
            task = asyncio.create_task(self._issue(row, sender_index))
            task.add_done_callback(
                lambda completed, future=completion: self._copy_task_result(completed, future))

    @staticmethod
    def _copy_task_result(task, completion):
        if completion.cancelled():
            return
        if task.cancelled():
            completion.cancel()
            return
        error = task.exception()
        if error is not None:
            completion.set_exception(error)
        else:
            completion.set_result(task.result())

    async def _issue(self, row, sender_index):
        delay = self.start_ns + int(row['scheduled_arrival_sec'] * 1e9) - time.monotonic_ns()
        if delay > 0:
            await asyncio.sleep(delay / 1e9)
        url, payload, encoded = self._prepared_requests[row['request_id']]
        prepared_request = self._prepared_http_requests.pop(row['request_id'])
        await self.client.issue(self._sender_sessions[sender_index], url, row['request_id'], payload,
                                encoded_payload=encoded, prepared_request=prepared_request)

    async def wait_all(self):
        futures = list(self.tasks.values())
        if futures:
            return await asyncio.gather(*self._wrap_futures(futures), return_exceptions=True)
        return []

    async def stop_arrivals(self):
        # This API name is required by the inherited contract. The cohort is
        # frozen, so all rows are still issued; there is no cancel/drop phase.
        await self.wait_all()

    async def close(self):
        await self.wait_all()
        for request in self._prepared_http_requests.values():
            coroutine = getattr(request, '_coro', None)
            if coroutine is not None:
                try:
                    coroutine.close()
                except RuntimeError:
                    pass
        self._prepared_http_requests.clear()
        for loop, thread, session in zip(
                self._sender_loops, self._sender_threads, self._sender_sessions):
            if session is not None and not session.closed:
                closing = asyncio.run_coroutine_threadsafe(session.close(), loop)
                await asyncio.to_thread(closing.result)
            loop.call_soon_threadsafe(loop.stop)
            await asyncio.to_thread(thread.join)
            loop.close()
        self._sender_loops.clear()
        self._sender_threads.clear()
        self._sender_sessions.clear()
        self._sender_loop = None
        self._sender_thread = None
        self._sender_session = None
        self.journal.close()


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + '\n')
        stream.flush()
        import os
        os.fsync(stream.fileno())


def summarize_full_cohort(ledger, trace, *, through_ns, send_lag_p99_ms=None, send_lag_max_ms=None):
    """Summarize only an exact, fully successful, reconciled frozen cohort."""
    rows, row_problems = ledger._rows(occurred_by_ns=through_ns)
    expected = {row['request_id'] for row in trace}
    indexed = {row['request_id']: row for row in rows}
    failures = set(row_problems)
    server_lengths = {}
    for event in ledger.events.values():
        if event.kind == 'server_terminal':
            request_id = event.request_id
            if request_id:
                if request_id in server_lengths:
                    failures.add('duplicate_server_completion:' + request_id)
                server_lengths[request_id] = {
                    'input_tokens': event.data.get('input_tokens'),
                    'output_tokens': event.data.get('output_tokens'),
                }
    if len(indexed) != len(rows):
        failures.add('duplicate_request_rows')
    if set(indexed) != expected:
        failures.add('request_id_set_mismatch')
    request_metrics = []
    for request in trace:
        row = indexed.get(request['request_id'])
        if row is None:
            continue
        if row['sent_ns'] is None:
            failures.add('not_sent:' + request['request_id'])
        if row['outcome'] != 'success':
            failures.add('not_success:' + request['request_id'] + ':' + str(row['outcome']))
        if row['client_terminal_ns'] is None:
            failures.add('missing_client_terminal:' + request['request_id'])
        if row['server_terminal_ns'] is None or row['server_outcome'] != 'success':
            failures.add('missing_server_success:' + request['request_id'])
        if row['actual_output_tokens'] != request['output_tokens']:
            failures.add('output_length_mismatch:' + request['request_id'])
        server_length = server_lengths.get(request['request_id'])
        if server_length is None:
            failures.add('missing_server_length_audit:' + request['request_id'])
        elif (server_length['input_tokens'] != request['input_tokens'] or
              server_length['output_tokens'] != request['output_tokens']):
            failures.add('server_length_mismatch:' + request['request_id'])
        if row['scheduled_ns'] is None:
            failures.add('missing_schedule:' + request['request_id'])
        if row['sent_ns'] is None or row['client_terminal_ns'] is None or row['scheduled_ns'] is None:
            continue
        terminal = row['client_terminal_ns']
        last = row['client_last_ns']
        first = row['client_first_ns']
        request_metrics.append({
            'request_id': request['request_id'],
            'scheduled_ns': row['scheduled_ns'], 'sent_ns': row['sent_ns'],
            'client_first_ns': first, 'client_last_ns': last,
            'client_terminal_ns': terminal,
            'expected_input_tokens': request['input_tokens'],
            'expected_output_tokens': request['output_tokens'],
            'actual_output_tokens': row['actual_output_tokens'],
            'initial_instance_index': request['initial_instance_index'],
            'send_lag_ms': (row['sent_ns'] - row['scheduled_ns']) / 1e6,
            'client_e2e_ms': (terminal - row['sent_ns']) / 1e6,
            'schedule_e2e_ms': (terminal - row['scheduled_ns']) / 1e6,
            'ttft_ms': row['ttft_ms'], 'tpot_ms': row['tpot_ms'],
            'decode_span_ms': ((last - first) / 1e6 if last is not None and first is not None else None),
            'terminal_tail_ms': ((terminal - last) / 1e6 if last is not None else None),
        })
    data_complete = (len(request_metrics) == len(expected) and not failures)
    lag_values = [row['send_lag_ms'] for row in request_metrics]
    lag_p99 = percentile_type7(lag_values, .99)
    lag_max = max(lag_values, default=None)
    send_lag_gate_enabled = send_lag_p99_ms is not None or send_lag_max_ms is not None
    lag_gate_passed = True
    if send_lag_p99_ms is not None:
        lag_gate_passed = lag_gate_passed and lag_p99 is not None and lag_p99 <= send_lag_p99_ms
    if send_lag_max_ms is not None:
        lag_gate_passed = lag_gate_passed and lag_max is not None and lag_max <= send_lag_max_ms
    load_valid = (data_complete and lag_gate_passed) if send_lag_gate_enabled else None
    if data_complete and send_lag_gate_enabled and not load_valid:
        failures.add('load_invalid_send_lag')

    def values(key):
        return [row[key] for row in request_metrics if row[key] is not None]

    primary = None
    if data_complete:
        primary = {
            'client_e2e_ms': summarize_values(values('client_e2e_ms')),
            'schedule_e2e_ms': summarize_values(values('schedule_e2e_ms')),
            'ttft_ms': summarize_values(values('ttft_ms')),
            'tpot_ms': summarize_values(values('tpot_ms')),
            'decode_span_ms': summarize_values(values('decode_span_ms')),
            'terminal_tail_ms': summarize_values(values('terminal_tail_ms')),
            'send_lag_ms': summarize_values(lag_values),
            'cohort_throughput_rps': (len(expected) * 1e9 /
                (max(r['client_terminal_ns'] for r in request_metrics) -
                 min(r['scheduled_ns'] for r in request_metrics))),
            'output_tokens': sum(r['actual_output_tokens'] for r in request_metrics),
        }
    audit = ledger.audit(through_ns=through_ns, require_cleanup=True)
    if audit['status'] != 'passed':
        failures.update(audit['issues'])
    complete = data_complete and audit['status'] == 'passed'
    if not complete:
        primary = None
    return {
        'status': 'passed' if complete and (not send_lag_gate_enabled or load_valid) else
                  'load_invalid' if complete and send_lag_gate_enabled else 'measurement_incomplete',
        'cohort_complete': complete, 'load_valid': load_valid,
        'send_lag_gate': {
            'enabled': send_lag_gate_enabled,
            'result': ('not_applied' if not send_lag_gate_enabled else
                       'passed' if data_complete and load_valid else
                       'failed' if data_complete else 'not_evaluated_incomplete_cohort'),
            'p99_limit_ms': send_lag_p99_ms,
            'max_limit_ms': send_lag_max_ms,
        },
        'planned_request_count': len(expected), 'measured_request_count': len(request_metrics),
        'primary_metrics': primary if complete else None,
        'diagnostic_partial': None if complete else {
            'completed_successes': sum(row.get('outcome') == 'success' for row in rows),
            'observed_rows': len(rows),
        },
        'audit': audit, 'issues': sorted(failures),
        'requests': request_metrics,
        'ecdf': [
            {'rank': i + 1, 'probability': (i + 1) / len(sorted(values('client_e2e_ms'))),
             'client_e2e_ms': value}
            for i, value in enumerate(sorted(values('client_e2e_ms')))
        ] if complete else [],
    }


def compare_pair(stay, balanced):
    if any(row.get('status') != 'passed' for row in (stay, balanced)):
        return {'status': 'incomplete', 'mean_gain': None, 'p99_gain': None,
                'p95_gain': None, 'cvar95_gain': None}
    s = stay['primary_metrics']['client_e2e_ms']
    b = balanced['primary_metrics']['client_e2e_ms']
    return {
        'status': 'passed',
        'mean_gain': 1.0 - b['mean'] / s['mean'],
        'p99_gain': 1.0 - b['p99'] / s['p99'],
        'p95_gain': 1.0 - b['p95'] / s['p95'],
        'cvar95_gain': 1.0 - b['cvar95'] / s['cvar95'],
        'mean_delta_ms': b['mean'] - s['mean'],
        'p99_delta_ms': b['p99'] - s['p99'],
    }


def _append_jsonl(path, value):
    import os
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, allow_nan=False, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


async def wait_service_ready(cfg, process_probe, diagnostic_interval_sec=15.):
    import aiohttp
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3.)) as session:
        last_note = time.monotonic() - diagnostic_interval_sec
        while True:
            if not process_probe():
                raise RuntimeError('owned API process exited before readiness')
            diagnostic = None
            try:
                async with session.get(cfg['service']['base_url'] + '/is_ready') as response:
                    body = await response.text()
                    try:
                        payload = json.loads(body)
                    except json.JSONDecodeError:
                        payload = None
                    diagnostic = {'http_status': response.status, 'body': body[:300],
                                  'payload': payload}
                    if response.status == 200 and payload is True:
                        return
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                diagnostic = {'error': type(exc).__name__ + ': ' + str(exc)}
            if time.monotonic() - last_note >= diagnostic_interval_sec:
                print('Readiness probe: ' + json.dumps(diagnostic, sort_keys=True), flush=True)
                last_note = time.monotonic()
            await asyncio.sleep(1.)


async def _controller_heartbeat(bridge, stop_event, errors, *, interval_sec=5.):
    """Renew the controller lease independently from request-log reconciliation."""
    import aiohttp
    timeout = aiohttp.ClientTimeout(total=5., connect=2., sock_connect=2.)
    last_transport_error = None
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_sec)
                return
            except asyncio.TimeoutError:
                pass
            if not bridge.lease:
                return
            lease_id = bridge.lease['lease_id']
            try:
                async with session.post(bridge.base_url + '/real_control/renew_controller_nonblocking',
                                       json={'lease_id': lease_id}) as response:
                    result = await response.json()
                    if response.status != 200:
                        raise RuntimeError(result.get(
                            'error', 'controller heartbeat rejected with HTTP ' + str(response.status)))
                    if result.get('lease_id') != lease_id:
                        raise RuntimeError('controller heartbeat returned a different lease')
                    bridge.lease = result
                    last_transport_error = None
            except RuntimeError as exc:
                errors.append(type(exc).__name__ + ': ' + str(exc))
                return
            except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError, OSError) as exc:
                last_transport_error = type(exc).__name__ + ': ' + str(exc)
                expires_ns = bridge.lease.get('expires_ns') if bridge.lease else None
                if expires_ns is not None and time.monotonic_ns() >= expires_ns:
                    errors.append('heartbeat_lease_expired_after_transport_errors: '
                                  + last_transport_error)
                    return


def cumulative_output_progress(ledger):
    """Monotone output count for progress detection, independent of rolling features."""
    return sum(state['token_high'] for state in ledger._online.values())


async def run_episode(*, cfg, run_id, episode_id, policy, trace, run_dir,
                      control_session, frozen_mapping=None, send_lag_p99_ms=None,
                      send_lag_max_ms=None, service_probe=None, stop_owned_service=None,
                      decision_controller=None, controller_version=None, balanced_threshold=9000.,
                      no_progress_timeout_sec=None, hard_stop_monotonic=None):
    base_url = cfg['service']['base_url']
    bridge = RealBridge(base_url, control_session)
    io = FullCompletionIO(bridge, control_session, run_id=run_id, episode_id=episode_id,
        run_dir=run_dir, trace=trace, sampling=cfg['sampling'],
        arrival_sec=max((row['scheduled_arrival_sec'] for row in trace), default=0.))
    result = None
    control_issues = []
    control_enabled = True
    pending_decision = None
    decisions = []
    receipts = {}
    control_active = False
    measurement_started = False
    heartbeat_stop = None
    heartbeat_task = None
    heartbeat_errors = []
    service_failure = False
    try:
        await io.prepare()
        initial = await bridge.snapshot()
        baseline = assert_idle(initial)
        mapping = initial['mapping']
        if frozen_mapping is not None and mapping != frozen_mapping:
            raise RuntimeError('stable instance mapping changed across paired episodes')
        if policy not in ('Stay', 'Balanced', 'PPO', 'PPO-STAY'):
            raise ValueError('unknown full-cohort policy')
        if (policy in ('PPO', 'PPO-STAY')) != (decision_controller is not None):
            raise ValueError('PPO requires an explicit frozen decision controller')
        await bridge.acquire_controller(run_id, controller_version or 'Balanced-9000-SR-v1', episode_id, mode='active')
        control_active = True
        heartbeat_stop = asyncio.Event()
        heartbeat_task = asyncio.create_task(_controller_heartbeat(
            bridge, heartbeat_stop, heartbeat_errors))
        # Arm the sender loop and all finite-cohort tasks before the planned
        # release. With a 256-request synchronous calibration, submitting each
        # coroutine from the control loop takes a few hundred milliseconds; that
        # preparation must not be counted as client send lag or distort lambda_ref.
        start_ns = time.monotonic_ns() + 1_000_000_000
        scheduled_span = max((row['scheduled_arrival_sec'] for row in trace), default=0.)
        arrival_end_ns = start_ns + max(1, int((scheduled_span + 1e-6) * 1e9))
        ledger = io.ledger(start_ns, arrival_end_ns, None)
        ledger.full_completion = True
        builder = RealObservationBuilder(mapping)
        measurement_started = True
        await io.begin(start_ns, arrival_end_ns, None, ledger)
        record_path = io.directory / 'decisions.jsonl'
        next_tick = time.monotonic_ns()
        last_progress_at = time.monotonic()
        last_output_count = 0
        no_progress_alerts = 0
        last_alert_at = time.monotonic()

        while True:
            now = time.monotonic_ns()
            if now >= next_tick:
                if service_probe is not None and not service_probe():
                    service_failure = True
                    control_issues.append('owned_service_exited_during_cohort')
                    break
                try:
                    await io.poll(ledger)
                except Exception as exc:
                    control_issues.append('event_poll:' + type(exc).__name__ + ':' + str(exc))
                    control_enabled = False
                if control_enabled:
                    try:
                        if heartbeat_errors:
                            raise RuntimeError('independent controller heartbeat failed: '
                                               + heartbeat_errors[0])
                        snapshot = await bridge.snapshot()
                        if snapshot.get('lease_expired') or snapshot.get('invalid_reasons'):
                            raise RuntimeError('controller lease expired or manager marked episode invalid')
                        failed_delivery = [state.get('instance_id') for state in snapshot['instances']
                                          if state.get('pending_delivery') is None]
                        if failed_delivery:
                            service_failure = True
                            control_issues.append('output_forwarding_failed:'
                                                  + ','.join(sorted(failed_delivery)))
                            if stop_owned_service is not None:
                                await asyncio.to_thread(stop_owned_service)
                            break
                        feature_start = time.perf_counter()
                        raw = builder.build(snapshot, now_ns=time.monotonic_ns(), ledger=ledger,
                            episode_start_ns=start_ns, arrival_end_ns=arrival_end_ns,
                            deadline_ns=arrival_end_ns)
                        feature_ms = (time.perf_counter() - feature_start) * 1000
                        observed = ledger.observe(time.monotonic_ns())['requests']
                        # Online token_increments is a rolling observation window.
                        # Its high-water mark is cumulative and survives window eviction.
                        token_count = cumulative_output_progress(ledger)
                        completed_count = sum(row['client_terminal_ns'] is not None for row in observed)
                        if token_count > last_output_count:
                            last_output_count = token_count
                            last_progress_at = time.monotonic()
                        task_done = all(future.done() for future in io.tasks.values())
                        if pending_decision is not None:
                            final = await bridge.query_receipt(pending_decision)
                            receipts[pending_decision] = final
                            if final.get('terminal_status') in ('succeeded', 'failed', 'rejected'):
                                pending_decision = None
                        action = 0
                        policy_record = None
                        if policy == 'Balanced' and not task_done and pending_decision is None:
                            action = balanced_action({'raw': raw}, snapshot, threshold=balanced_threshold)
                        if decision_controller is not None and not task_done:
                            # An unresolved RPC keeps its original decision ID;
                            # it must never produce a second migration action.
                            if pending_decision is not None:
                                raw = deepcopy(raw)
                                raw['action_mask'] = [True] + [False] * 12
                            policy_record = decision_controller(raw, start_ns)
                            action = policy_record['action']
                            if policy == 'PPO-STAY' and (action != 0 or not policy_record.get('measurement_only')):
                                raise ValueError('PPO-STAY requires measurement-only Stay')
                            if type(action) is not int or not 0 <= action < 13 or not raw['action_mask'][action]:
                                raise ValueError('controller returned an invalid or masked action')
                        decision_id = f'{run_id}:{episode_id}:{len(decisions)}'
                        receipt = {'receipt_status': 'stay', 'terminal_status': None,
                                   'migration_id': None}
                        src_id = dst_id = candidate_id = None
                        if action:
                            from .real_bridge import PAIRS
                            src, dst = PAIRS[action - 1]
                            src_id, dst_id = mapping[src][0], mapping[dst][0]
                            candidate_id = next(state.get('candidate_id') for state in snapshot['instances']
                                                if state['instance_id'] == src_id)
                        if policy_record is not None:
                            # The original decision identity and behavior law
                            # survive an interruption during the migration RPC.
                            _append_jsonl(io.directory / 'ppo_action_intents.jsonl', {
                                'decision_id': decision_id, 'snapshot_version': snapshot['version'],
                                'src_id': src_id, 'dst_id': dst_id, 'candidate_id': candidate_id,
                                'ppo': policy_record,
                            })
                        if action:
                            receipt = await bridge.submit_migration(decision_id, src_id, dst_id,
                                candidate_id, snapshot['version'])
                            receipts[decision_id] = receipt
                            if receipt.get('receipt_status') in ('accepted', 'unknown') and receipt.get('terminal_status') not in ('succeeded', 'failed', 'rejected'):
                                pending_decision = decision_id
                        record = {
                            'decision_id': decision_id, 'monotonic_ns': now,
                            # Preserve the exact time used by RealObservationBuilder
                            # to build this observation and its action mask. This is
                            # later used for diagnostic eligibility attribution.
                            'observation_decision_ns': raw['decision_ns'],
                            'policy': policy, 'sampled_action': action, 'executed_action': action,
                            'feature_ms': feature_ms, 'control_work_ms': (time.monotonic_ns()-now)/1e6,
                            'balanced_threshold': balanced_threshold,
                            'snapshot_version': snapshot['version'],
                            'instance_mapping': mapping, 'action_mask': raw['action_mask'],
                            'pair_features': raw['pair_features'], 'snapshot': snapshot,
                            'src_id': src_id, 'dst_id': dst_id, 'candidate_id': candidate_id,
                            'receipt': receipt, 'completed_requests': completed_count,
                            'actual_output_tokens': token_count,
                        }
                        if policy_record is not None:
                            record['ppo'] = policy_record
                        decisions.append(record)
                        _append_jsonl(record_path, record)
                    except Exception as exc:
                        control_issues.append('control:' + type(exc).__name__ + ':' + str(exc))
                        control_enabled = False
                else:
                    # An uncertain control path disables new actions, but any
                    # already-submitted action must keep being queried by ID.
                    if pending_decision is not None:
                        try:
                            final = await bridge.query_receipt(pending_decision)
                            receipts[pending_decision] = final
                            if final.get('terminal_status') in ('succeeded', 'failed', 'rejected'):
                                pending_decision = None
                        except Exception as exc:
                            control_issues.append('receipt_reconcile:' + type(exc).__name__ + ':' + str(exc))
                next_tick = time.monotonic_ns() + 1_000_000_000

                if service_failure:
                    break

            if all(future.done() for future in io.tasks.values()):
                if pending_decision is None:
                    break
            if ((no_progress_timeout_sec is not None and time.monotonic()-last_progress_at >= no_progress_timeout_sec)
                    or (hard_stop_monotonic is not None and time.monotonic() >= hard_stop_monotonic)):
                service_failure = True
                control_issues.append('fast_track_no_progress_or_hard_budget_abort')
                if stop_owned_service is not None:
                    await asyncio.to_thread(stop_owned_service)
                break
            if time.monotonic() - max(last_progress_at, last_alert_at) >= 60:
                no_progress_alerts += 1
                _append_jsonl(io.directory / 'diagnostic_alerts.jsonl', {
                    'kind': 'no_output_progress', 'elapsed_without_new_tokens_sec':
                        time.monotonic() - last_progress_at,
                    'completed_requests': sum(future.done() for future in io.tasks.values()),
                    'automatic_cancellation': False,
                })
                last_alert_at = time.monotonic()
                print(json.dumps({'episode_id': episode_id, 'alert': 'no_output_progress',
                                  'completed': sum(future.done() for future in io.tasks.values()),
                                  'planned': len(trace)}, sort_keys=True), flush=True)
            await asyncio.sleep(.1)

        if service_failure:
            # The owned API has been stopped to unblock HTTP streams after an
            # explicit output-delivery failure. Keep the whole request list.
            await io.wait_all()
            try:
                await io.poll(ledger)
            except Exception as exc:
                control_issues.append('failure_event_poll:' + type(exc).__name__ + ':' + str(exc))

        future_errors = []
        for request_id, future in io.tasks.items():
            try:
                future.result()
            except BaseException as exc:
                future_errors.append(f'{request_id}:{type(exc).__name__}:{exc}')
        if future_errors:
            control_issues.extend('sender_task:' + value for value in future_errors)
        if service_failure:
            idle = True

        # A terminal response can precede the final actor event delivery. Reconcile
        # the whole cluster and the migration coordinator without a request timeout.
        idle = service_failure
        while not idle:
            if ((hard_stop_monotonic is not None and time.monotonic() >= hard_stop_monotonic)
                    or (service_probe is not None and not service_probe())):
                service_failure = True
                control_issues.append('owned_service_stopped_during_reconciliation')
                if stop_owned_service is not None:
                    await asyncio.to_thread(stop_owned_service)
                break
            try:
                await io.poll(ledger)
                if heartbeat_errors:
                    control_issues.append('heartbeat:' + heartbeat_errors[0])
                if pending_decision is not None:
                    final = await bridge.query_receipt(pending_decision)
                    receipts[pending_decision] = final
                    if final.get('terminal_status') in ('succeeded', 'failed', 'rejected'):
                        pending_decision = None
                snapshot = await bridge.snapshot()
                if snapshot.get('invalid_reasons'):
                    control_issues.append('manager_invalid:' + json.dumps(snapshot['invalid_reasons']))
                    control_enabled = False
                if pending_decision is None and not snapshot.get('busy'):
                    try:
                        assert_idle(snapshot, baseline)
                        idle = True
                    except RuntimeError:
                        pass
            except Exception as exc:
                control_issues.append('reconcile:' + type(exc).__name__ + ':' + str(exc))
            if not idle:
                await asyncio.sleep(1.)

        if not service_failure:
            await io.flush(ledger)
            if heartbeat_stop is not None:
                heartbeat_stop.set()
            if heartbeat_task is not None:
                await asyncio.gather(heartbeat_task, return_exceptions=True)
            if heartbeat_errors:
                control_issues.extend('heartbeat:' + error for error in heartbeat_errors)
            if bridge.lease is not None:
                await bridge.call('release_episode', run_id=run_id, episode_id=episode_id)
                released = await bridge.release_controller()
                control_active = False
                if released.get('status') != 'released':
                    control_issues.append('controller_release:' + str(released))
            await io.flush(ledger)
            final_snapshot = await bridge.snapshot()
            try:
                assert_idle(final_snapshot, baseline)
            except RuntimeError as exc:
                control_issues.append('final_idle:' + str(exc))
        else:
            if heartbeat_stop is not None:
                heartbeat_stop.set()
            if heartbeat_task is not None:
                await asyncio.gather(heartbeat_task, return_exceptions=True)
            control_issues.extend('heartbeat:' + error for error in heartbeat_errors)
            control_issues.append('service_failure_cleanup_no_final_server_flush')
            control_issues.append('controller_release_unavailable_after_service_failure')
        through = min((cp.through_ns for cp in ledger.checkpoints.values()),
                      default=time.monotonic_ns())
        summary = summarize_full_cohort(ledger, trace, through_ns=through,
            send_lag_p99_ms=send_lag_p99_ms, send_lag_max_ms=send_lag_max_ms)
        route_events = [event for event in ledger.events.values()
                        if event.data.get('event') == 'real_initial_route']
        observed_routes = {event.request_id: event.data.get('instance_id') for event in route_events
                           if event.request_id}
        route_valid = observed_routes == io.expected_routes
        if not route_valid:
            control_issues.append('initial_route_audit_mismatch')
        if pending_decision is not None or any(
                receipt.get('terminal_status') not in ('succeeded', 'failed', 'rejected')
                for receipt in receipts.values()
                if receipt.get('receipt_status') in ('accepted', 'unknown')):
            control_issues.append('migration_terminal_unconfirmed')
        if any(receipt.get('terminal_status') == 'unknown' for receipt in receipts.values()):
            control_issues.append('migration_terminal_unknown')
        if control_issues and summary['status'] == 'passed':
            summary['status'] = 'control_invalid'
            summary['issues'] = sorted(set(summary['issues']) | set(control_issues))
        summary.update({
            'schema': 'full-completion-episode-v1', 'run_id': run_id,
            'episode_id': episode_id, 'policy': policy,
            'trace_sha256': hashlib.sha256(json.dumps(trace, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
            'arrival_window_sec': (arrival_end_ns - start_ns) / 1e9,
            'frozen_mapping': mapping, 'expected_initial_routes': io.expected_routes,
            'observed_initial_routes': observed_routes, 'route_audit_passed': route_valid,
            'balanced_threshold': balanced_threshold,
            'decision_count': len(decisions), 'actual_migration_attempts': sum(bool(row['receipt'].get('migration_id')) for row in decisions),
            'migration_receipts': receipts, 'control_issues': control_issues,
            'no_progress_alert_count': no_progress_alerts,
            'client_terminal_count': sum(row['client_terminal_ns'] is not None for row in summary['requests']),
            'output_token_total': sum(row['actual_output_tokens'] for row in summary['requests']),
        })
        atomic_json(io.directory / 'episode.json', summary)
        _write_jsonl(io.directory / 'requests.jsonl', summary['requests'])
        _write_jsonl(io.directory / 'ecdf.jsonl', summary['ecdf'])
        _write_jsonl(io.directory / 'events.jsonl', [deepcopy(event.__dict__) for event in ledger.events.values()])
        atomic_json(io.directory / 'event_cursor.json', {
            'offsets': io.cursor.offsets, 'identities': io.cursor.identities,
            'issues': io.cursor.issues,
        })
        result = {key: value for key, value in summary.items() if key not in ('requests', 'ecdf')}
        return result
    finally:
        if measurement_started:
            # No routine cancellation: if the caller is interrupted, keep draining
            # the frozen request set before releasing any owned service state.
            await io.wait_all()
        if heartbeat_stop is not None:
            heartbeat_stop.set()
        if heartbeat_task is not None:
            await asyncio.gather(heartbeat_task, return_exceptions=True)
        if heartbeat_errors:
            control_issues.extend('heartbeat:' + error for error in heartbeat_errors)
        if control_active and bridge.lease is not None and not service_failure:
            try:
                await bridge.call('release_episode', run_id=run_id, episode_id=episode_id)
                await bridge.release_controller()
            except Exception as exc:
                control_issues.append('final_release:' + type(exc).__name__ + ':' + str(exc))
        await io.close()


def profile_summary_row(episode):
    primary = episode.get('primary_metrics') or {}
    e2e = primary.get('client_e2e_ms') or {}
    return {
        'candidate_id': episode.get('candidate_id'), 'seed': episode.get('seed'),
        'policy': episode.get('policy'), 'status': episode.get('status'),
        'mean_e2e_ms': e2e.get('mean'), 'p95_e2e_ms': e2e.get('p95'),
        'p99_e2e_ms': e2e.get('p99'), 'cvar95_e2e_ms': e2e.get('cvar95'),
        'actual_migration_attempts': episode.get('actual_migration_attempts'),
    }


def runtime_dependencies_hash(paths):
    hasher = hashlib.sha256()
    for path in sorted(Path(path) for path in paths):
        hasher.update(str(path).encode())
        hasher.update(path.read_bytes())
    return hasher.hexdigest()
