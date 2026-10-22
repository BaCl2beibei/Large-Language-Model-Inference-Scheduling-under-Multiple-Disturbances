# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import atexit
import queue
import threading
import os
from typing import Dict, List, Union
import math

import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import PlacementGroup
import ray.exceptions
import ray.actor

from vllm.utils import is_pin_memory_available
from vllm.worker.worker import Worker
from vllm.config import CacheConfig,  ModelConfig, ParallelConfig
from vllm.worker.cache_engine import CacheEngine
from vllm.utils import GiB_bytes
from vllm.sequence import SequenceGroupMetadata, SequenceGroupMetadataDelta, ExecuteModelRequest
from vllm import envs as vllm_envs

from llumnix.logging.logger import init_logger
from llumnix.backends.vllm.utils import _sample_with_torch
from llumnix.backends.vllm.migration_backend import MigrationBackendBase, get_migration_backend
from llumnix.internal_config import MigrationConfig
from llumnix.utils import convert_bytes, log_worker_exception
from llumnix.ray_utils import log_actor_ray_info
from llumnix.metrics.causal_events import emit as emit_causal_event
from llumnix.metrics.causal_events import enabled as causal_telemetry_enabled
from llumnix.metrics.causal_events import metadata_signature

logger = init_logger(__name__)


class MigrationWorker(Worker):
    def real_control_flush(self):
        from llumnix.metrics.causal_events import flush
        deadline = time.monotonic() + 5.
        pending = getattr(self, '_causal_cuda_queue', None)
        if pending is not None:
            with pending.all_tasks_done:
                while pending.unfinished_tasks and time.monotonic() < deadline:
                    pending.all_tasks_done.wait(max(0., deadline-time.monotonic()))
                if pending.unfinished_tasks:
                    return {'status': 'incomplete', 'reason': 'pending_cuda_telemetry'}
        ack = flush(max(0., deadline-time.monotonic()))
        ack['pending_delivery'] = 0
        return ack

    def __init__(self, *args, **kwargs) -> None:
        # replace sampler
        # pylint: disable=import-outside-toplevel
        import vllm.model_executor.layers.sampler
        vllm.model_executor.layers.sampler._sample_with_torch = _sample_with_torch
        log_actor_ray_info(self.__class__.__name__)
        self.migrating_out_seq_group_metadata: Dict[str, Union[SequenceGroupMetadata, SequenceGroupMetadataDelta]] = {}
        self.migrating_in_seq_group_metadata: Dict[str, Union[SequenceGroupMetadata, SequenceGroupMetadataDelta]] = {}

        self.instance_id = None
        self.rank = None
        self._causal_cuda_queue = None
        self._causal_cuda_thread = None
        self._causal_cuda_init_lock = threading.Lock()

        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}(iid={self.instance_id[:5]}, rank={self.rank})"

    def load_model(self):
        torch.cuda.set_device(self.device)
        return super().load_model()

    def get_global_rank(self):
        return self.global_rank

    def reserve_memory_for_migration(self,
                                     migration_config: MigrationConfig,
                                     model_config: ModelConfig,
                                     cache_config: CacheConfig,
                                     parallel_config: ParallelConfig) -> int:
        migrate_cache_blocks_size = migration_config.migration_buffer_blocks
        migration_num_layers = migration_config.migration_num_layers
        dummy_cache_size = migration_num_layers * migrate_cache_blocks_size * CacheEngine.get_cache_block_size(
            cache_config, model_config, parallel_config) // model_config.get_num_layers(parallel_config)

        # For nccl migration backend, reserve gpu memory for dummy cache in migration backend. For other backends,
        # CPU memory is used for the dummy cache, which is almost unlimited, so no special action is needed.
        if migration_config.migration_backend == "nccl" and parallel_config.world_size == 1:
            device = torch.device(f"cuda:{self.local_rank}")
            _, total_memory = torch.cuda.mem_get_info(device)
            migration_memory_ratio = math.ceil(dummy_cache_size / total_memory * 10000) / 10000
            cache_config.gpu_memory_utilization -= migration_memory_ratio

            if cache_config.gpu_memory_utilization <= 0:
                raise ValueError("Nccl migration backend take {:.4f} gpu memory, which is greater than gpu_memory_utilization {:.4f}. "
                                 "try to increase gpu-memory-utilization or reduce migration-buffer-blocks."
                                 .format(migration_memory_ratio, cache_config.gpu_memory_utilization))

            logger.info("Nccl migration backend take {:.4f} gpu memory, left gpu_memory_utilization {:.4f} for kv cache."
                        .format(migration_memory_ratio, cache_config.gpu_memory_utilization))

        return dummy_cache_size

    def init_migration(self,
                       instance_id: str,
                       migration_config: MigrationConfig,
                       src_worker_handle_list: List[ray.actor.ActorHandle],
                       placement_group: PlacementGroup) -> None:
        # for proxy actor
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=placement_group,
            placement_group_bundle_index=0,
            placement_group_capture_child_tasks=True,
        )

        pin_memory = is_pin_memory_available()
        if not pin_memory:
            # Pinning memory in WSL is not supported.
            # https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications
            logger.warning("Using 'pin_memory=False' as WSL is detected. "
                           "This may slow down the performance.")

        self.instance_id = instance_id
        self.global_world_size = 0
        self.global_rank = -1
        self.use_ray_spmd_worker = vllm_envs.VLLM_USE_RAY_SPMD_WORKER
        self.migration_backend: MigrationBackendBase = get_migration_backend(instance_id=instance_id,
                                                                             migration_config=migration_config,
                                                                             cache_engine=self.cache_engine,
                                                                             worker_handle_list=src_worker_handle_list,
                                                                             scheduling_strategy=scheduling_strategy,
                                                                             is_driver_worker=self.is_driver_worker,
                                                                             gpu_cache=self.gpu_cache,
                                                                             worker_rank=self.rank,
                                                                             local_rank=self.local_rank,
                                                                             use_ray_spmd_worker=self.use_ray_spmd_worker,
                                                                             worker_stage_seq_group_metadata_callback=self._stage_seq_group_metadata)

    def recv_cache(self,
                   request_id: str,
                   src_worker_handle_list: List[ray.actor.ActorHandle],
                   src_blocks: List[int],
                   dst_blocks: List[int],
                   is_last_stage: bool = False) -> bool:
        src_worker_handle = src_worker_handle_list[self.rank]
        try:
            start_time = time.time()
            self.migration_backend.recv_cache(request_id, src_worker_handle, src_blocks, dst_blocks, is_last_stage)
            end_time = time.time()
            total_kv_cache_size = len(src_blocks) * CacheEngine.get_cache_block_size(
                self.cache_config, self.model_config, self.parallel_config)
            speed = total_kv_cache_size / GiB_bytes / (end_time - start_time)
            logger.info("Recv kv cache done, num_blocks: {}, total_kv_cache_size: {}, time: {:.2f}s, speed: {:.5f}GB/s."
                        .format(len(src_blocks), convert_bytes(total_kv_cache_size), end_time - start_time, speed))
            return True
        # pylint: disable=broad-except
        except Exception as e:
            log_worker_exception(e, self.instance_id, self.rank, "recv_cache", request_id)
            return False

    def do_recv(self, *args, **kwargs):
        return self.migration_backend.do_recv(*args, **kwargs)

    def do_send(self, *args, request_id: str = None, send_worker_metadata: bool = False, **kwargs):
        if not send_worker_metadata:
            return self.migration_backend.do_send(*args, **kwargs)
        return self.migration_backend.do_send(*args, **kwargs), self._get_seq_group_metadata(request_id)

    def _get_seq_group_metadata(self, request_id: str) -> Union[SequenceGroupMetadata, SequenceGroupMetadataDelta]:
        # Only send sequence group metadata in last stage (blocking migration), so the request id must exist.
        assert request_id in self._seq_group_metadata_cache, \
            f"the request id {request_id} of running request that migrating out should exist in sequence group metadata cache"
        src_seq_group_metadata = self._seq_group_metadata_cache.pop(request_id)
        self._add_migrating_out_seq_group_metadata(request_id, src_seq_group_metadata)
        return src_seq_group_metadata

    def _stage_seq_group_metadata(self, request_id: str,
            src_seq_group_metadata: Union[SequenceGroupMetadata, SequenceGroupMetadataDelta]) -> None:
        self.migrating_in_seq_group_metadata[request_id] = src_seq_group_metadata

    def commit_seq_group_metadata(self, request_id: str) -> None:
        assert request_id in self.migrating_in_seq_group_metadata, \
            f"the request id {request_id} of running request that migrating in should exist in migrating in sequence group metadata"
        self._seq_group_metadata_cache[request_id] = self.migrating_in_seq_group_metadata.pop(request_id)

    def _add_migrating_out_seq_group_metadata(self, request_id: str,
            seq_group_metadata: Union[SequenceGroupMetadata, SequenceGroupMetadataDelta]) -> None:
        self.migrating_out_seq_group_metadata[request_id] = seq_group_metadata

    def pop_migrating_out_seq_group_metadata(self, request_id: str) -> bool:
        seq_group_metadata = self.migrating_out_seq_group_metadata.pop(request_id, None)
        return seq_group_metadata is not None

    def free_migrating_in_seq_group_metadata(self) -> None:
        self.migrating_in_seq_group_metadata.clear()

    def restore_migrating_out_seq_group_metadata(self, request_id: str) -> None:
        seq_group_metadata = self.migrating_out_seq_group_metadata.pop(request_id, None)
        if seq_group_metadata is not None:
            self._seq_group_metadata_cache[request_id] = seq_group_metadata
        return seq_group_metadata is not None

    def _execute_model_spmd(self, execute_model_req: ExecuteModelRequest, *args, **kwargs):
        if execute_model_req is not None:
            self._update_cached_seq_group_metadata(execute_model_req.seq_group_metadata_list)
        return self._profile_causal_cuda_execution(
            execute_model_req, super()._execute_model_spmd, *args, **kwargs
        )

    def execute_model(self, execute_model_req: ExecuteModelRequest = None):
        # vLLM's RayGPUExecutor uses this path when Ray SPMD workers are disabled.
        return self._profile_causal_cuda_execution(
            execute_model_req, super().execute_model
        )

    def _profile_causal_cuda_execution(self, execute_model_req, execute_model, *args, **kwargs):
        if not causal_telemetry_enabled() or execute_model_req is None:
            return execute_model(execute_model_req, *args, **kwargs)
        metadata_list = getattr(execute_model_req, "seq_group_metadata_list", ()) or ()
        if not metadata_list:
            return execute_model(execute_model_req, *args, **kwargs)

        self._ensure_causal_cuda_writer()
        signature = metadata_signature(metadata_list)
        request_ids = [str(getattr(item, "request_id", "")) for item in metadata_list]
        context_tokens = 0
        prefill_requests = 0
        for metadata in metadata_list:
            prefill_requests += int(bool(getattr(metadata, "is_prompt", False)))
            for sequence in (getattr(metadata, "seq_data", {}) or {}).values():
                try:
                    context_tokens += int(sequence.get_len())
                except (AttributeError, TypeError, ValueError):
                    continue

        gpu_index = int(torch.cuda.current_device())
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        start_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        succeeded = False
        try:
            output = execute_model(execute_model_req, *args, **kwargs)
            succeeded = True
            return output
        finally:
            try:
                end_event.record()
                end_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                self._causal_cuda_queue.put_nowait({
                    "start_event": start_event,
                    "end_event": end_event,
                    "gpu_index": gpu_index,
                    "request_ids": request_ids,
                    "metadata_signature": signature,
                    "context_tokens": context_tokens,
                    "num_prefill_requests": prefill_requests,
                    "num_decode_requests": len(metadata_list) - prefill_requests,
                    "cpu_call_start_ns": start_ns,
                    "cpu_call_end_ns": end_ns,
                    "succeeded": succeeded,
                })
            except Exception as exc:  # keep a telemetry failure from masking the model call
                emit_causal_event(
                    "gpu_execution_event_record_failed",
                    instance_id=str(self.instance_id or ""),
                    worker_rank=self.rank,
                    gpu_index=gpu_index,
                    request_ids=request_ids,
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:300],
                )

    def _ensure_causal_cuda_writer(self) -> None:
        if self._causal_cuda_thread is not None:
            return
        with self._causal_cuda_init_lock:
            if self._causal_cuda_thread is not None:
                return
            self._causal_cuda_queue = queue.Queue()
            emit_causal_event(
                "gpu_event_capture_initialized",
                instance_id=str(self.instance_id or ""),
                worker_rank=self.rank,
            )
            self._causal_cuda_thread = threading.Thread(
                target=self._drain_causal_cuda_events,
                name=f"llumnix-cuda-events-{os.getpid()}",
                daemon=True,
            )
            self._causal_cuda_thread.start()
            atexit.register(self._shutdown_causal_cuda_writer)

    def _shutdown_causal_cuda_writer(self) -> None:
        if self._causal_cuda_queue is None or self._causal_cuda_thread is None:
            return
        self._causal_cuda_queue.join()
        self._causal_cuda_queue.put_nowait(None)
        if threading.current_thread() is not self._causal_cuda_thread:
            self._causal_cuda_thread.join()

    def _drain_causal_cuda_events(self) -> None:
        while True:
            item = self._causal_cuda_queue.get()
            try:
                if item is None:
                    return
                try:
                    torch.cuda.set_device(item["gpu_index"])
                    while not item["end_event"].query():
                        time.sleep(0.002)
                    elapsed_ms = float(item["start_event"].elapsed_time(item["end_event"]))
                    emit_causal_event(
                        "gpu_execution_completed",
                        instance_id=str(self.instance_id or ""),
                        worker_rank=self.rank,
                        gpu_index=item["gpu_index"],
                        request_ids=item["request_ids"],
                        metadata_signature=item["metadata_signature"],
                        context_tokens=item["context_tokens"],
                        num_prefill_requests=item["num_prefill_requests"],
                        num_decode_requests=item["num_decode_requests"],
                        cpu_call_start_ns=item["cpu_call_start_ns"],
                        cpu_call_end_ns=item["cpu_call_end_ns"],
                        gpu_elapsed_ms=elapsed_ms,
                        succeeded=item["succeeded"],
                        timing_method="torch_cuda_events_async_no_synchronize",
                    )
                except Exception as exc:  # pragma: no cover - runtime driver variation
                    emit_causal_event(
                        "gpu_execution_timing_failed",
                        instance_id=str(self.instance_id or ""),
                        worker_rank=self.rank,
                        gpu_index=item["gpu_index"],
                        request_ids=item["request_ids"],
                        error_type=type(exc).__name__,
                        error_message=str(exc)[:300],
                    )
            finally:
                self._causal_cuda_queue.task_done()

    def _update_cached_seq_group_metadata(
            self,
            seq_group_metadata_list: List[Union[SequenceGroupMetadata, SequenceGroupMetadataDelta]]) -> None:
        # Update seq_data of cached seq_grou_metadata in worker (the src seq_id is different from the dst seq_id).
        for metadata_or_delta in seq_group_metadata_list:
            request_id = metadata_or_delta.request_id
            if request_id in self._seq_group_metadata_cache:
                seq_group_metadata = self._seq_group_metadata_cache[request_id]
                if isinstance(metadata_or_delta, SequenceGroupMetadataDelta):
                    for new_seq_id, old_seq_id in zip(metadata_or_delta.seq_data_delta.keys(), \
                                                      seq_group_metadata.seq_data.keys()):
                        if new_seq_id != old_seq_id:
                            seq_group_metadata.seq_data[new_seq_id] = seq_group_metadata.seq_data.pop(old_seq_id)

    def rebuild_migration_backend(self, instance_rank: Dict[str, int], group_name: str) -> bool:
        self.migration_backend.destory_backend()

        ret = True
        if group_name is not None:
            num_instance = len(instance_rank)
            self.global_world_size = num_instance * self.parallel_config.world_size
            self.global_rank = self.rank + instance_rank[self.instance_id] * self.parallel_config.world_size
            ret = self.migration_backend.init_backend(group_name, self.global_world_size, self.global_rank)

        return ret

    def warmup(self) -> bool:
        return self.migration_backend.warmup()
