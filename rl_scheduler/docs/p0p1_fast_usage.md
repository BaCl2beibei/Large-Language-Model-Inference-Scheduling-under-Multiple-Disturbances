# P0/P1 bounded screen entry

Use the repository `.venv/bin/python`. No command invokes PPO training.

```bash
.venv/bin/python benchmark/run_p0p1_fast.py plan --run-dir benchmark/results/p0p1_fast_<UTC timestamp>
.venv/bin/python benchmark/run_p0p1_fast.py run --run-dir benchmark/results/p0p1_fast_<UTC timestamp>
.venv/bin/python benchmark/run_p0p1_fast.py analyze --run-dir benchmark/results/p0p1_fast_<UTC timestamp>
```

`plan` materializes both candidates' exact-token traces and hashes without starting GPU services. It checks existing ledgers for the proposed seeds. `run` refuses a previously started directory and verifies source/config hashes. A private supervisor service owns Ray and the four replicas. Only those owned processes are cleaned up. Initial readiness has a total eight-minute limit. After warmup, the threshold is frozen to the smallest actual KV capacity divided by eight. It is passed directly to `run_episode`; old callers retain the 9000 default.

PPO-STAY uses the current CPU actor/critic with random weights frozen by seed 141100 and one Torch thread. Its argmax suggestion and value are diagnostic. The executed action is always zero; records have `measurement_only=true` and no behavior log-probability. `build_rollout` rejects measurement-only records.

Each round drains the entire frozen cohort and checks lifecycle, actual first engine admission, and coordinator migration events. The first audit failure stops the sequence. P0 failure also stops the sequence. Q15-A and the sole fallback Q15-B use only the preregistered branch order. The launcher retains at least eight minutes and the longest measured round plus thirty seconds before starting another round, and starts no rounds after 45 minutes. A 60-minute watchdog or 120 seconds without new output may stop the owned service; such evidence is incomplete.

`analyze` reads only this directory's append-only episode inventory and raw files. It reports decision denominators before the last client completion, distinct successful request coverage, overlapping no-opportunity reasons, CPU conversion/forward and control interval timings, P0 deviations, and every per-seed P1 gate. The Balanced filter count also includes ordering and post-move balance checks, so it must not be described as a pure threshold effect. Missing or failed pairs cannot pass. A recommendation requires both seeds and the fixed mean/cost improvement gates.

The authoritative protocol is [p0p1_fast_track_prompt.md](p0p1_fast_track_prompt.md). This entry does not implement a general experiment platform or the full research protocol.
