# Round 3 decision: matched output lengths could not be enforced

## Controlled QPS 15 probe

The seed 505 QPS 15 arrival trace is byte-for-byte identical to the earlier seed 505 trace (`09a0dc7c657eab19985e65847a773d6fdc9a85416a2eb8b189432c638e284cc6`). A per-request output-length manifest was sampled from successful calibration responses only and held fixed across the proposed OFF/ON pair. No held-out response lengths were used to choose those targets.

The OFF run passed the four-worker startup gate and sent all 942 measurement requests. The client set `min_tokens=max_tokens` to each manifest target for every measurement request. The server completed 775 requests with HTTP 200, returned 161 HTTP 499 responses, recorded six requests without a response status at the drain boundary, and recorded no client timeouts. Of the 775 successful responses, 628 matched their manifest target and 147 did not (19.0%). Mean absolute target error was 15.16 tokens; mean target and observed output lengths were 93.06 and 77.90 tokens. The [per-request failures](qps_15/off/output_length_enforcement_audit.json), [summary](qps_15/off/output_length_enforcement_audit_summary.json), raw requests, and server logs are preserved.

vLLM 0.6.3 accepted `min_tokens`, `max_tokens`, and `ignore_eos` in `SamplingParams`, and the request body override was applied to all 942 measurement requests. Server logs confirm that max-token targets arrived, but actual `num_output_tokens_cf` values often ended earlier. This Llumnix request path therefore did not provide a reliable way to reproduce the same realized output lengths.

## Decision and scope

The OFF/ON pair stopped before the ON run, and the remaining four QPS levels were not collected. This is not a five-QPS system validation and seed 505 is not claimed as an independent final admission holdout. Continuing with a five-QPS comparison would violate the matched-length condition, so Round 3 ends here. No fourth improvement round or further real runs were started.

The candidate explicit `realized_output_tokens` trace input remains in the simulator with bounds checks and source auditing. Directed tests cover its override behavior and trace loading. The system-level admission criteria remain unmet; do not start formal 2,000-episode PPO training or post-training evaluation.

The previous complete five-QPS seed 404 system validation already missed the admission thresholds: V3 closed-loop MAPE was 38.13% for throughput, 72.83% mean TTFT, 56.00% P95 TTFT, 21.48% mean TPOT, and 25.60% P95 TPOT. Queue mean/P95 MAE was 7.199/31.543 requests per instance; only the KV error limits passed. Those results are documented in `../ppo_system_validation_round2_causal_repairs_20261022/system_validation.md` and remain diagnostic because seed 404 had already been used for model comparison. The incomplete seed 505 probe does not replace that validation.

## Code verification

`python -m compileall -q rl_scheduler` exited 0. `python -m pytest -q rl_scheduler` exited 0 with 37 passed and one existing `asyncio_fixture_loop_scope` configuration warning. The two new directed checks cover explicit realized-length overrides, budget rejection, and held-out trace loading.

## Run accounting and cleanup

Across the prior seed 505 Migration OFF/ON pilot, the seed-controlled OFF/ON pair, and this probe, five additional real runs were attempted against the 12-run cap. The failed output-length probe used the same trace as the earlier seed 505 investigation. The run state and mismatch audit preserve the failure details. GPU utilization returned to idle after the server process exited.
