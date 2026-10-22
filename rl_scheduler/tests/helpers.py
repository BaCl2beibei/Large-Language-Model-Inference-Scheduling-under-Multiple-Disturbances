from types import SimpleNamespace


class FakeInstanceInfo:
    def __init__(self, instance_id, *, now=1000.0, blocks=8, total=256, free=200,
                 running=1, waiting=0, candidate=True, stage="decode", healthy=True):
        self.instance_id = instance_id
        self.instance_type = "neutral"
        self.unit_status = "healthy" if healthy else "stopped"
        self.timestamp = now
        self.num_running_requests = running
        self.num_waiting_requests = waiting
        self.waiting_time_first_waiting_request = 0.25
        self.num_total_gpu_blocks = total
        self.num_free_gpu_blocks = free
        self.num_used_gpu_blocks = total - free
        self.num_blocks_all_waiting_requests = waiting * 4
        self.gpu_cache_usage = (total - free) / total
        self.num_killed_requests = 0
        self.decode_batch_size = running
        self.last_inference_latency = 8.0
        self.num_batched_tokens = 256
        self.remaining_steps = SimpleNamespace(remaining_steps=24.0)
        self.migration_candidate_request_id = f"candidate-{instance_id}" if candidate else None
        self.migration_candidate_kv_blocks = blocks if candidate else None
        self.migration_candidate_stage = stage if candidate else None
        self.migration_candidate_valid = candidate
        self.migration_candidate_timestamp = now
        self.has_migration_slot = True
        self.is_migrating = False
        self.can_accept_migration = True

    def is_unit_healthy(self):
        return self.unit_status == "healthy"


def fake_infos(now=1000.0):
    return {
        f"instance_{idx}": FakeInstanceInfo(f"instance_{idx}", now=now, blocks=8 + idx,
                                             total=256, free=220 - idx * 10,
                                             running=idx + 1, waiting=idx)
        for idx in range(4)
    }

