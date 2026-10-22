# Mig-PPO data audit and refined calibration

## Data coverage

Calibration: 15 runs, 49058 service rows, 37815 unique valid engine samples after deduplicating 10961 repeated snapshots.
Validation seed 404: 5 runs, 16462 service rows, 12855 unique valid engine samples after deduplicating 3535 repeated snapshots.

Service latency is `executor_iteration_wall_ms`, a host wall-clock interval from Ray executor dispatch through the awaited worker result. It is neither isolated GPU time nor request TTFT. `profiling_data` supplies batch size, aggregate sequence length, stage label, and latency. `bs` is actual scheduler `num_batched_tokens`; it lives in the scheduler snapshot and is aligned through the preceding step when available.

Same-step outer label mismatches: calibration 4487, validation 1491. Predecessor-step joins: calibration 32803, validation 11308; full feature matches 32504/32803 and 11185/11308. Unmatched running/queue/KV features are not filled with fabricated values; P2+ Prefill candidates use only rows with a validated predecessor join.

## Migration label semantics

Calibration terminal statuses: `{'FINISHED': 58, 'ABORTED_DST': 6, 'ABORTED_SRC': 10}`; validation statuses: `{'FINISHED': 26, 'ABORTED_DST': 5}`.
`cost_ms` is the wall-clock duration of `_migrate_out_one_request` from method entry through terminal status handling. The CSV has no phase-level durations and no direct request service-pause label. The simulator may model an in-flight request pause as an explicit approximation; the data do not identify that pause duration.

## Service model selection

Coefficients, regularization and candidate selection use only calibration seeds 101/202/303. Selection uses leave-one-run-out CV over QPS-seed run groups. Seed 404 is report-only independent validation and does not affect the promoted candidate.

| Stage | Candidate | grouped CV MAE | grouped CV RMSE | CV R² | seed404 MAE | seed404 RMSE | seed404 R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| prefill | P0 | 41.505 | 93.342 | 0.0375 | 42.698 | 217.445 | -0.0007 |
| prefill | P1 | 41.440 | 93.306 | 0.0382 | 42.691 | 217.464 | -0.0009 |
| prefill | P2 | 41.440 | 93.310 | 0.0382 | 42.691 | 217.464 | -0.0009 |
| prefill | P3 | 41.463 | 93.264 | 0.0391 | 42.461 | 217.346 | 0.0002 |
| prefill | P4 | 41.566 | 93.422 | 0.0359 | 42.709 | 217.473 | -0.0010 |
| decode | D0 | 27.538 | 102.300 | 0.0199 | 24.645 | 57.956 | 0.0358 |
| decode | D1 | 27.538 | 102.300 | 0.0199 | 24.645 | 57.956 | 0.0358 |
| decode | D2 | 27.538 | 102.300 | 0.0199 | 24.645 | 57.956 | 0.0358 |
| decode | D3 | 26.498 | 101.827 | 0.0289 | 23.973 | 57.667 | 0.0454 |
| decode | D4 | 27.429 | 102.123 | 0.0233 | 24.655 | 57.828 | 0.0400 |

Promoted prefill candidate: **P1** (CV winner P1).

Promoted decode candidate: **D3** (CV winner D3).

## Migration model selection

| Candidate | grouped CV MAE | grouped CV RMSE | seed404 MAE | seed404 RMSE |
|---|---:|---:|---:|---:|
| M0_global_ols | 389.687 | 712.989 | 384.212 | 554.191 |
| M1_global_huber | 344.146 | 707.202 | 338.125 | 560.118 |
| M2_huber_pair_shrunk | 342.289 | 698.536 | 334.977 | 566.915 |

Promoted migration model: **M2_huber_pair_shrunk**.

## Raw file schemas and linkage

Full field-by-field column names, numeric distributions, missingness, IQR outlier fractions, and categorical counts are in `data_audit.json`. The request CSV keys are `run_id/phase/request_id/server_request_id`; trace IDs join by the terminal `rNNNNN`; per-instance scheduler samples join on `(instance_id, step_id)` and service state is taken from `step_id - 1`; migration events join through `run_id/phase/request_ids`; GPU metrics join through `run_id/phase/gpu_index/wall_time`.

Timestamp fields are Unix seconds for `timestamp`, `observed_wall_time`, `wall_time`, and `actual_send_wall_time`; `scheduled_arrival_sec`, `actual_send_sec`, `completion_sec`, `relative_sec`, and trace arrivals are phase-relative seconds. `cost_ms`, request latency fields, and profiling tuple latency are milliseconds; input/output/sequence lengths are tokens; `blocks` is KV blocks; GPU memory is MiB, utilization is percent, temperature is Celsius, and power is watts.

The raw `summary.csv` is a 163-column aggregate with run rows and aggregate rows (`row_type` distinguishes them); only per-run rows should be compared with individual traces. `service_instance.csv` is the scheduler polling schema (`profiling_data`, `bs`, `seq_lens`, queue/KV/admission fields) and includes repeated polling snapshots.

## Limitations

Request-level responses and GPU metrics are kept for system validation, not used as labels in the service-time fits. The fixed ON replay and autonomous closed-loop simulation are separate outputs. Initial instance routing, omitted request-to-instance assignments, aborted events without request IDs, model output distributions, and the absence of an explicit migration pause label remain assumptions and are listed in the system validation report.
