"""Duck-typed vLLM-v0 adapters; KV movement stays in MigrationCoordinator."""
from __future__ import annotations

from copy import deepcopy
import time
import uuid

from rl_scheduler.real_bridge import capacity_ok, eligible_pair


class ControlledRejection(Exception):pass


async def cancel_instance_request(owner, request_id):
    """Install a permanent per-epoch fence, then prove absence from v0 queues.

    In-flight transfers/adds are allowed to finish under the fence. We never free
    blocks being used by a transfer or infer absence from an abort RPC return.
    """
    cancelled = owner.__dict__.setdefault('_real_cancelled_requests', set())
    cancelled.add(request_id)
    coordinator = owner.migration_coordinator
    coordinator.__dict__.setdefault('_real_cancelled_requests', set()).add(request_id)
    engine = owner.backend_engine.engine
    engine.__dict__.setdefault('_real_cancelled_requests', set()).add(request_id)
    if len(engine.scheduler)!=1:
        raise RuntimeError('cancellation requires one v0 scheduler')
    scheduler = engine.scheduler[0]
    if (getattr(owner, '_real_generating', set())
            or getattr(owner, '_real_inbound_tasks', set())
            or coordinator.migrating_out_request_id_set
            or getattr(owner, '_real_control_inflight', None)):
        return {'status':'unknown', 'reason':'request_operations_inflight'}
    # Reuse the engine's step/output-processing barrier before freeing storage.
    # Its boundary callback also removes waiting/swapped requests under the fence.
    resident = (request_id in engine.request_ids
                or request_id in scheduler.migrating_out_request_last_stage
                or any(any(r.request_id==request_id for r in queue)
                       for queue in (scheduler.running,scheduler.waiting,scheduler.swapped)))
    if resident:
        await owner.backend_engine.remove_running_request(request_id)
    if (getattr(owner, '_real_generating', set()) or getattr(owner, '_real_inbound_tasks', set())
            or coordinator.migrating_out_request_id_set or getattr(owner, '_real_control_inflight', None)):
        return {'status':'unknown', 'reason':'request_operations_inflight'}
    await owner.backend_engine.abort_request({request_id})
    # No transfer is active now; paused and preallocated storage may be reclaimed.
    paused = scheduler.migrating_out_request_last_stage.get(request_id)
    if paused is not None:
        owner.backend_engine.free_src_request(paused)
        scheduler.migrating_out_request_last_stage.pop(request_id, None)
    coordinator.free_pre_alloc_cache(request_id)
    coordinator.__dict__.setdefault('_real_capacity_guards', {}).pop(request_id, None)
    await owner.backend_engine.abort_request({request_id})
    queues = (scheduler.running, scheduler.waiting, scheduler.swapped)
    queues_empty = all(all(r.request_id!=request_id for r in queue) for queue in queues)
    # v0's inherited abort does not update Llumnix's diagnostic request_ids set.
    if queues_empty and request_id not in scheduler.migrating_out_request_last_stage:
        engine.clear_request(request_id)
    absent = (queues_empty
              and request_id not in scheduler.pre_alloc_cache_dict
              and request_id not in scheduler.migrating_out_request_last_stage
              and request_id not in engine.request_ids
              and request_id not in coordinator.migrating_in_request_id_set
              and request_id not in coordinator.migrating_out_request_id_set
              and request_id not in coordinator.pending_migrate_in_request_time)
    pending = engine.output_forwarder.thread_forwarder.real_pending_delivery()
    return {'status':'absent' if absent and pending==0 else 'unknown',
            'instance_id':owner.instance_id, 'epoch':instance_state(owner)['epoch'],
            'pending_delivery':pending, 'fenced':True}


def instance_state(owner):
    if not hasattr(owner, '_real_epoch'):
        owner._real_epoch = uuid.uuid4().hex
    backend = owner.backend_engine
    engine = getattr(backend, 'engine', None)
    schedulers = getattr(engine, 'scheduler', [])
    if len(schedulers) != 1:
        raise RuntimeError('real control requires one vLLM v0 scheduler per instance')
    scheduler = schedulers[0]
    info = scheduler._get_instance_info([])  # synchronous live queue/block inspection
    coordinator = owner.migration_coordinator
    candidates = coordinator.migration_scheduler._get_shortest_running_request(0, float('inf'))
    candidate = candidates[0] if candidates else None
    blocks = (len(scheduler.block_manager.get_block_table(candidate.get_seqs()[0]))
              if candidate is not None else None)
    status = getattr(owner.unit_status, 'value', owner.unit_status)
    instance_type = getattr(owner.instance_args.instance_type, 'value', owner.instance_args.instance_type)
    return {'instance_id': owner.instance_id, 'epoch': owner._real_epoch,
            'measured_ns': time.monotonic_ns(), 'healthy': str(status).lower()=='healthy',
            'type': str(instance_type).lower(), 'is_migrating': coordinator.is_migrating(),
            'has_slot': coordinator.has_migration_slot(), 'total_blocks': info.num_total_gpu_blocks,
            'free_blocks': info.num_free_gpu_blocks, 'used_blocks': info.num_used_gpu_blocks,
            'waiting_blocks': info.num_blocks_all_waiting_requests,
            'running': info.num_running_requests, 'waiting': info.num_waiting_requests,
            'head_wait_sec': info.waiting_time_first_waiting_request,
            'candidate_id': str(candidate.request_id) if candidate is not None else None,
            'candidate_blocks': blocks, 'candidate_length': candidate.request_len if candidate is not None else None,
            'candidate_stage': str(getattr(candidate.inference_type, 'value', '')).lower() if candidate is not None else None,
            'candidate_policy': coordinator.migration_scheduler.request_migration_policy,
            'output_tokens_total': getattr(engine, '_real_output_tokens_total', None),
            'last_step_ms': getattr(engine.instance_info, 'last_inference_latency', None),
            'pending_delivery': engine.output_forwarder.thread_forwarder.real_pending_delivery(),
            'preallocated_requests': len(scheduler.pre_alloc_cache_dict),
            'paused_requests': len(scheduler.migrating_out_request_last_stage)}


async def controlled_migrate(owner, dst_actor, dst_context, migration_type, *,
                             expected_candidate_id, decision_id, migration_id,
                             source_epoch, destination_epoch):
    results = owner.__dict__.setdefault('_real_decision_results', {})
    if decision_id in results:
        return deepcopy(results[decision_id])
    receipt = {'migration_id': migration_id, 'decision_id': decision_id,
               'terminal_status': None, 'actual_request_id': None, 'copied_blocks': None}
    results[decision_id] = receipt
    def reject(reason):
        receipt.update(terminal_status='rejected', rejection_reason=reason, terminal_ns=time.monotonic_ns())
        return deepcopy(receipt)
    coordinator = owner.migration_coordinator
    if expected_candidate_id in getattr(owner, '_real_cancelled_requests', set()):
        return reject('request_cancelled')
    if coordinator.enable_engine_migration_interface:
        return reject('engine_migration_interface_unsupported')
    if coordinator.migration_scheduler.request_migration_policy != 'SR':
        return reject('SR_policy_required')
    if getattr(owner, '_real_control_inflight', None) is not None:
        return reject('source_busy')
    owner._real_control_inflight = decision_id
    guard_installed = False
    known_terminal = False
    try:
        source = instance_state(owner)
        if source['epoch'] != source_epoch or source['candidate_id'] != expected_candidate_id:
            return reject('source_epoch_or_SR_candidate_changed')
        destination = await dst_actor.real_control_state.remote()
        if destination['epoch'] != destination_epoch:
            return reject('destination_epoch_changed')
        valid, reason = eligible_pair(source, destination, time.monotonic_ns())
        if not valid:
            return reject(reason)
        guard = await dst_actor.real_control_reserve.remote(expected_candidate_id, destination_epoch)
        if guard.get('status') != 'prepared':
            return reject('destination_not_prepared')
        guard_installed = True
        # Destination RPC yielded: inspect SR again immediately before claiming it.
        source = instance_state(owner)
        requests = coordinator.migration_scheduler._get_shortest_running_request(0, float('inf'))
        if expected_candidate_id in getattr(owner, '_real_cancelled_requests', set()):
            raise ControlledRejection('request_cancelled_at_execution')
        if source['epoch'] != source_epoch or not requests or str(requests[0].request_id) != expected_candidate_id:
            raise ControlledRejection('SR_candidate_changed_at_execution')
        request = requests[0]
        if source['is_migrating'] or not source['has_slot']:
            raise ControlledRejection('source_busy_at_execution')
        request.is_migrating = True
        coordinator.__dict__.setdefault('_real_uncertain_requests', set()).discard(expected_candidate_id)
        coordinator.__dict__.setdefault('_real_request_guards', {})[expected_candidate_id] = destination_epoch
        receipt['migration_start_ns'] = time.monotonic_ns()
        migrated = await coordinator._migrate_out_one_request(
            dst_actor, dst_context.instance_id, request, migration_type, migration_id=migration_id)
        if expected_candidate_id in coordinator._real_uncertain_requests:
            receipt.update(terminal_status='unknown', error_type='underlying_transfer_RPC_uncertain')
        else:
            receipt.update(terminal_status='succeeded' if expected_candidate_id in [str(x) for x in migrated] else 'failed',
                           actual_request_id=expected_candidate_id,
                           copied_blocks=sum(request.stage_num_blocks_list),
                           migration_end_ns=time.monotonic_ns())
            receipt['terminal_ns']=receipt['migration_end_ns']
            known_terminal = True
    except ControlledRejection as exc:
        reject(str(exc))
    except BaseException as exc:
        receipt.update(terminal_status='unknown', error_type=type(exc).__name__)
    finally:
        if receipt.get('terminal_status') == 'rejected':
            known_terminal = True
        if guard_installed and known_terminal:
            try:
                await dst_actor.real_control_unreserve.remote(expected_candidate_id)
            except Exception:
                receipt.update(terminal_status='unknown', error_type='destination_guard_release_uncertain')
                known_terminal = False
        if known_terminal:
            owner._real_control_inflight = None
            coordinator.__dict__.setdefault('_real_request_guards', {}).pop(expected_candidate_id,None)
        # Unknown leaves both the source fence and Manager slot occupied.
    return deepcopy(receipt)


class ManagerControlMixin:
    def _real_controller(self):
        from rl_scheduler.real_bridge import ControlArbiter
        from llumnix.metrics.causal_events import emit
        import asyncio
        if not hasattr(self, '_real_arbiter'):
            if self.enable_pd_disagg:
                raise RuntimeError('real control supports non-PD only')
            async def states():
                return list(await asyncio.gather(*(actor.real_control_state.remote()
                                                  for actor in list(self.instances.values()))))
            async def dispatch(receipt):
                from llumnix.utils import MigrationType
                return await self.instances[receipt['src_id']].migrate_out.remote(
                    self.instances[receipt['dst_id']], self.instance_context[receipt['dst_id']],
                    MigrationType.NEUTRAL_LOAD_BALANCE,
                    expected_candidate_id=receipt['expected_candidate_id'],
                    decision_id=receipt['decision_id'], migration_id=receipt['migration_id'],
                    source_epoch=receipt['source_epoch'], destination_epoch=receipt['destination_epoch'])
            async def reconcile(receipt):
                actor = self.instances.get(receipt['src_id'])
                return await actor.real_control_result.remote(receipt['decision_id']) if actor else None
            self._real_arbiter = ControlArbiter(states, dispatch, reconcile=reconcile,
                native_idle=lambda: not getattr(self, '_real_native_tasks', set()) and not getattr(self, '_real_native_fault', False),
                event_sink=lambda record: emit('real_control_receipt', record=record))
        return self._real_arbiter

    async def real_control(self, method, payload):
        allowed = {'snapshot','acquire_controller','renew_controller','submit_migration',
                   'query_receipt','release_controller'}
        if method not in allowed:
            raise ValueError('unsupported control method')
        return await getattr(self._real_controller(), method)(**payload)

    async def real_control_flush(self):
        import asyncio
        from llumnix.metrics.causal_events import flush
        instances = await asyncio.gather(*(actor.real_control_flush.remote()
                                           for actor in list(self.instances.values())))
        manager = await asyncio.to_thread(flush,5.)
        manager['pending_delivery'] = (len(getattr(self,'_real_native_tasks',set()))
                                      + sum(len(tasks) for tasks in getattr(self,'_real_dispatch_tasks',{}).values()))
        return {'manager': manager, 'instances': instances}

    async def real_cancel_request(self, request_id):
        import asyncio
        self.__dict__.setdefault('_real_cancelled_requests', set()).add(request_id)
        actors = dict(self.instances)
        if len(actors)!=4 or self.enable_pd_disagg:
            return {'status':'unknown', 'reason':'unsupported_or_changed_inventory'}
        acknowledgements = await asyncio.gather(*(actor.real_cancel_request.remote(request_id)
                                                   for actor in actors.values()))
        valid_roles = all(ack.get('instance_id')==instance_id and ack.get('epoch')
                          and ack.get('status')=='absent' and ack.get('fenced') is True
                          and ack.get('pending_delivery')==0
                          for instance_id,ack in zip(actors,acknowledgements))
        if not valid_roles:
            return {'status':'unknown', 'instances':acknowledgements}
        current = await asyncio.gather(*(actor.real_control_state.remote() for actor in actors.values()))
        controller = getattr(self, '_real_arbiter', None)
        stable = (actors==self.instances and not getattr(self,'_real_native_fault',False)
                  and not getattr(self,'_real_native_tasks',set())
                  and not self.__dict__.get('_real_dispatch_tasks',{}).get(request_id)
                  and not (controller and (controller.slot or controller.invalid_reasons)))
        known = stable and all(state.get('instance_id')==ack['instance_id'] and state.get('epoch')==ack['epoch']
                               and state.get('healthy') is True and state.get('type')=='neutral'
                               and state.get('is_migrating') is False
                               for state,ack in zip(current,acknowledgements))
        return {'status':'absent' if known else 'unknown', 'instances':acknowledgements}
