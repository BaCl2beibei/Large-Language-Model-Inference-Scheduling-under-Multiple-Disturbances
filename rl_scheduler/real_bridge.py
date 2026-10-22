"""CPU-only controller protocol and localhost transport for the real v0 cluster.

The Manager owns this arbiter. An accepted decision occupies the slot before
any await/dispatch; uncertain RPC results retain that slot until reconciliation.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import json
import math
import time
import uuid
from urllib.parse import urlparse

PAIRS = tuple((src, dst) for src in range(4) for dst in range(4) if src != dst)
TERMINALS = frozenset({'succeeded', 'failed', 'rejected'})


class CumulativeTokenCounter:
    """Measured local production, with a transferred request's prefix as baseline."""
    def __init__(self):self.counts={};self.total=0

    def baseline(self,request_id,count):
        self.counts[request_id]=max(self.counts.get(request_id,0),int(count))

    def observe(self,request_id,count):
        previous=self.counts.get(request_id,0)
        delta=max(0,int(count)-previous)
        self.counts[request_id]=max(previous,int(count))
        self.total+=delta
        return delta


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def capacity_ok(destination, blocks):
    try:
        total, free = int(destination['total_blocks']), int(destination['free_blocks'])
        waiting = math.ceil(float(destination['waiting_blocks']))
        blocks = int(blocks)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return (total > 0 and free >= 0 and waiting >= 0 and blocks > 0
            and free - waiting - blocks >= max(16, math.ceil(total * .05)))


def eligible_pair(source, destination, now_ns, max_age_sec=2.):
    for state in (source, destination):
        if (state.get('healthy') is not True or state.get('type') != 'neutral'
                or state.get('is_migrating') is not False or state.get('has_slot') is not True):
            return False, 'unhealthy_or_busy'
        stamp = state.get('measured_ns')
        if stamp is None or not 0 <= now_ns - stamp <= max_age_sec * 1e9:
            return False, 'stale_state'
    if not source.get('candidate_id') or source.get('candidate_stage') != 'decode':
        return False, 'no_SR_candidate'
    if source.get('candidate_policy') != 'SR':
        return False, 'SR_policy_required'
    if not capacity_ok(destination, source.get('candidate_blocks')):
        return False, 'insufficient_capacity'
    return True, None


class ControlArbiter:
    """Single Manager event-loop ownership; explicit lease fencing, no auto takeover.

    state_provider returns freshly measured instance dicts. dispatch must await
    the existing controlled migrate_out result, not its legacy enqueue-only API.
    """
    def __init__(self, state_provider, dispatch, *, reconcile=None, native_idle=lambda: True,
                 clock=time.monotonic_ns, event_sink=lambda record: None,
                 lease_sec=30., cooldown_sec=5., max_age_sec=2.):
        self.state_provider, self.dispatch, self.reconcile = state_provider, dispatch, reconcile
        self.native_idle, self.clock, self.event_sink = native_idle, clock, event_sink
        self.lease_sec, self.cooldown_sec, self.max_age_sec = lease_sec, cooldown_sec, max_age_sec
        self.epoch = uuid.uuid4().hex
        self.lease = None
        self.slot = None
        self.receipts = {}
        self.fingerprints = {}
        self.last_terminal_ns = None
        self.latest = None
        self.version = 0
        self.invalid_reasons = []
        self._lock = asyncio.Lock()
        self._tasks = set()
        self.release_requested = False
        self.acquiring = False

    @property
    def ordinary_blocked(self):
        return self.acquiring or self.lease is not None or self.slot is not None

    def external_intervention(self, reason):
        if self.ordinary_blocked:
            self.invalid_reasons.append(str(reason))
            self.event_sink({'event': 'controller_intervention', 'reason': str(reason), 'at_ns': self.clock()})

    def _lease_valid(self, lease_id):
        return (self.lease is not None and self.lease['lease_id'] == lease_id
                and self.clock() <= self.lease['expires_ns'] and not self.release_requested)

    def _check_mapping(self, states):
        mapping = tuple(sorted((s['instance_id'], s['epoch']) for s in states))
        if len(mapping) != 4 or len({x[0] for x in mapping}) != 4:
            raise RuntimeError('exactly four independent instances required')
        if self.lease and mapping != tuple(tuple(x) for x in self.lease['mapping']):
            self.external_intervention('instance_index_invalidated')
            raise RuntimeError('instance mapping changed during lease')
        return mapping

    async def snapshot(self):
        async with self._lock:
            states = await self.state_provider()
            mapping = self._check_mapping(states)
            self.version += 1
            now = self.clock()
            self.latest = {'version': f'{self.epoch}:{self.version}', 'measured_ns': now,
                           'received_ns': now, 'instances': sorted(states, key=lambda s: s['instance_id']),
                           'mapping': [list(x) for x in mapping], 'busy': self.slot is not None,
                           'last_terminal_ns': self.last_terminal_ns,
                           'lease_expired': self.lease is not None and now > self.lease['expires_ns'],
                           'invalid_reasons': list(self.invalid_reasons)}
            return deepcopy(self.latest)

    async def acquire_controller(self, run_id, controller_version, episode_id, mode='active'):
        if mode not in ('active', 'shadow', 'stay'):
            raise ValueError('unsupported control mode')
        async with self._lock:
            if self.lease is not None or self.slot is not None or not self.native_idle():
                raise RuntimeError('controller or native migration still owns cluster')
            self.acquiring = True  # Manager's ordinary scheduler reads this before any await.
            interventions = len(self.invalid_reasons)
            try:
                states = await self.state_provider()
                mapping = self._check_mapping(states)
                if not self.native_idle() or len(self.invalid_reasons)!=interventions:
                    raise RuntimeError('native activity or intervention during acquisition')
                if any(s.get('healthy') is not True or s.get('is_migrating') is not False
                       or s.get('type') != 'neutral' for s in states):
                    raise RuntimeError('cluster unhealthy, non-neutral, or migrating')
            finally:
                self.acquiring = False
            now = self.clock()
            self.lease = {'lease_id': f'{self.epoch}-{uuid.uuid4().hex}', 'run_id': run_id,
                          'episode_id': episode_id, 'controller_version': controller_version,
                          'mapping': [list(x) for x in mapping], 'expires_ns': now+int(self.lease_sec*1e9),
                          'mode': mode}
            self.invalid_reasons = []
            self.release_requested = False
            return deepcopy(self.lease)

    async def renew_controller(self, lease_id):
        async with self._lock:
            if not self._lease_valid(lease_id):
                raise RuntimeError('expired or unknown lease; cleanup before takeover')
            self.lease['expires_ns'] = self.clock()+int(self.lease_sec*1e9)
            return deepcopy(self.lease)

    async def submit_migration(self, lease_id, decision_id, src_id, dst_id,
                               expected_candidate_id, snapshot_version):
        payload = {'lease_id': lease_id, 'decision_id': decision_id, 'src_id': src_id,
                   'dst_id': dst_id, 'expected_candidate_id': expected_candidate_id,
                   'snapshot_version': snapshot_version}
        fingerprint = digest(payload)
        async with self._lock:
            if decision_id in self.receipts:
                if self.fingerprints[decision_id] != fingerprint:
                    return {'decision_id': decision_id, 'receipt_status': 'conflict',
                            'rejection_reason': 'decision_id_payload_changed', 'terminal_status': None}
                return deepcopy(self.receipts[decision_id])
            receipt = {**payload, 'receipt_status': 'rejected', 'rejection_reason': None,
                       'migration_id': None, 'terminal_status': 'rejected',
                       'accepted_ns': None, 'terminal_ns': self.clock()}
            self.fingerprints[decision_id] = fingerprint
            self.receipts[decision_id] = receipt
            def reject(reason):
                receipt['rejection_reason'] = reason
                receipt['terminal_ns'] = self.clock()
                self.event_sink(deepcopy(receipt))
                return deepcopy(receipt)
            if not self._lease_valid(lease_id):
                return reject('lease_invalid')
            receipt.update({k: self.lease[k] for k in ('run_id', 'episode_id', 'controller_version')})
            if self.invalid_reasons:
                return reject('episode_invalid')
            if self.slot is not None or not self.native_idle():
                return reject('global_slot_busy')
            now = self.clock()
            if self.last_terminal_ns is not None and now-self.last_terminal_ns < self.cooldown_sec*1e9:
                return reject('cooldown')
            if not self.latest or self.latest['version'] != snapshot_version:
                return reject('snapshot_version_changed')
            if now-self.latest['measured_ns'] > self.max_age_sec*1e9:
                return reject('stale_snapshot')
            if src_id == dst_id:
                return reject('invalid_pair')
            try:
                states = await self.state_provider()
            except Exception as exc:
                return reject('state_unavailable:'+type(exc).__name__)
            try:
                self._check_mapping(states)
            except RuntimeError:
                return reject('instance_index_invalidated')
            by_id = {s['instance_id']: s for s in states}
            if src_id not in by_id or dst_id not in by_id:
                return reject('unknown_instance')
            source, destination = by_id[src_id], by_id[dst_id]
            if source.get('candidate_id') != expected_candidate_id:
                return reject('SR_candidate_changed')
            valid, reason = eligible_pair(source, destination, self.clock(), self.max_age_sec)
            if not valid:
                return reject(reason)
            # Recheck the lease after network reads; expiry is never implicitly extended.
            if not self._lease_valid(lease_id):
                return reject('lease_expired_during_validation')
            if self.lease['mode'] in ('shadow', 'stay'):
                receipt.update(receipt_status='shadow' if self.lease['mode']=='shadow' else 'stay',
                               terminal_status=None, terminal_ns=None)
                self.event_sink(deepcopy(receipt))
                return deepcopy(receipt)
            receipt.update(receipt_status='accepted', rejection_reason=None, terminal_status=None,
                           accepted_ns=self.clock(), terminal_ns=None,
                           migration_id='real-'+digest([self.epoch, self.lease['run_id'], decision_id]))
            self.slot = decision_id  # atomic, before the task can run
            receipt['source_epoch'], receipt['destination_epoch'] = source['epoch'], destination['epoch']
            task = asyncio.create_task(self._execute(decision_id))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            self.event_sink(deepcopy(receipt))
            return deepcopy(receipt)

    async def _execute(self, decision_id):
        receipt = self.receipts[decision_id]
        try:
            result = await self.dispatch(deepcopy(receipt))
        except BaseException as exc:
            # A timeout/connection loss cannot establish whether remote execution ran.
            result = {'terminal_status': 'unknown', 'error_type': type(exc).__name__}
        async with self._lock:
            self._apply_result(decision_id, result)

    def _apply_result(self, decision_id, result):
        receipt = self.receipts[decision_id]
        if result.get('migration_id') not in (None, receipt['migration_id']):
            result = {'terminal_status': 'unknown', 'error_type': 'migration_id_mismatch'}
        if result.get('terminal_status') == 'succeeded' and result.get('actual_request_id') != receipt['expected_candidate_id']:
            result = {'terminal_status': 'unknown', 'error_type': 'actual_request_mismatch'}
        receipt.update({key: deepcopy(value) for key, value in result.items()
                        if key not in ('decision_id', 'migration_id', 'lease_id', 'src_id', 'dst_id')})
        if receipt.get('terminal_status') in TERMINALS:
            receipt['terminal_received_ns'] = self.clock()
            receipt['terminal_ns'] = result.get('terminal_ns',result.get('migration_end_ns')) or self.clock()
            if receipt['terminal_ns'] > self.clock():
                receipt.update(terminal_status='unknown',terminal_ns=None,error_type='future_terminal_time')
                return
            self.last_terminal_ns = receipt['terminal_ns']
            if self.slot == decision_id:
                self.slot = None
        else:
            receipt['terminal_status'] = 'unknown'
            receipt['terminal_ns'] = None
        self.event_sink(deepcopy(receipt))

    async def query_receipt(self, lease_id, decision_id):
        async with self._lock:
            receipt = self.receipts.get(decision_id)
            if receipt is None or receipt.get('lease_id') != lease_id:
                return {'decision_id': decision_id, 'receipt_status': 'unknown', 'terminal_status': None}
            if receipt.get('terminal_status') == 'unknown' and self.reconcile:
                try:
                    result = await self.reconcile(deepcopy(receipt))
                    if result and result.get('terminal_status') in TERMINALS:
                        self._apply_result(decision_id, result)
                except Exception:
                    pass
            return deepcopy(receipt)

    async def release_controller(self, lease_id):
        async with self._lock:
            if self.lease is None:
                return {'status': 'released'}
            if self.lease['lease_id'] != lease_id:
                raise RuntimeError('lease belongs to another controller')
            self.release_requested = True
            if self.slot is not None or not self.native_idle():
                return {'status': 'draining', 'decision_id': self.slot}
            states = await self.state_provider()
            if any(s.get('is_migrating') is not False for s in states):
                return {'status': 'draining', 'decision_id': self.slot}
            self.lease = None
            self.latest = None
            return {'status': 'released'}


class RealBridge:
    """HTTP transport: single submission, query on uncertainty, never blind retry."""
    def __init__(self, base_url, session):
        parsed = urlparse(base_url)
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
            raise ValueError('control transport must be loopback HTTP')
        self.base_url, self.session = base_url.rstrip('/'), session
        self.lease = None
        self.pending = {}
        self.unresolved = set()

    async def call(self, method, **payload):
        async with self.session.post(self.base_url+'/real_control/'+method, json=payload) as response:
            result = await response.json()
            if response.status >= 500:
                raise ConnectionError('uncertain control RPC')
            if response.status != 200:
                raise RuntimeError(result.get('error', 'control RPC failed'))
            return result

    async def snapshot(self):
        result = await self.call('snapshot')
        result['received_ns'] = time.monotonic_ns()
        return result

    async def acquire_controller(self, run_id, controller_version, episode_id, mode='active'):
        self.lease = await self.call('acquire_controller',run_id=run_id,
                                    controller_version=controller_version,episode_id=episode_id,mode=mode)
        return self.lease

    async def renew_controller(self):
        self.lease = await self.call('renew_controller',lease_id=self.lease['lease_id'])
        return self.lease

    async def submit_migration(self, decision_id, src_id, dst_id, expected_candidate_id, snapshot_version):
        payload = {'lease_id': self.lease['lease_id'], 'decision_id': decision_id, 'src_id': src_id,
                   'dst_id': dst_id,'expected_candidate_id': expected_candidate_id,'snapshot_version': snapshot_version}
        if decision_id in self.pending:
            if self.pending[decision_id] != payload:
                raise ValueError('decision ID reused with different payload')
            return await self.query_receipt(decision_id)
        for previous in tuple(self.unresolved):
            await self.query_receipt(previous)
        if self.unresolved:
            raise RuntimeError('unresolved decision fence; stop sampling and clean up')
        self.pending[decision_id] = payload
        import aiohttp
        try:
            result = await self.call('submit_migration',**payload)
        except (asyncio.TimeoutError, ConnectionError, OSError, aiohttp.ClientError):
            result = {'decision_id': decision_id, 'receipt_status': 'unknown', 'terminal_status': None,
                      'rejection_reason': 'transport_uncertain_query_before_any_new_action'}
        self._remember(decision_id,result)
        return result

    def _remember(self,decision_id,result):
        if result.get('terminal_status') in TERMINALS or result.get('receipt_status') in ('stay','shadow'):
            self.unresolved.discard(decision_id)
        else:self.unresolved.add(decision_id)

    async def query_receipt(self, decision_id):
        result=await self.call('query_receipt',lease_id=self.lease['lease_id'],decision_id=decision_id)
        self._remember(decision_id,result)
        return result

    async def release_controller(self):
        if not self.lease:
            return {'status':'released'}
        result=await self.call('release_controller',lease_id=self.lease['lease_id'])
        if result['status']=='released':
            self.lease=None
        return result
