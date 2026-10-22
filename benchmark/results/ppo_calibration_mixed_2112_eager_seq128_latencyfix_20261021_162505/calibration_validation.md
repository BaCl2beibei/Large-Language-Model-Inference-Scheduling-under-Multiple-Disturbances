# Mig-PPO Calibration and Holdout Validation

Calibration fit uses only QPS 9/12/15/18/21 at seeds 101/202/303. Seed 404 is excluded from fitting and used only for holdout validation.
Fitter: nonnegative least squares via active-set enumeration. The fitted YAML overlays on the checked-in `train.yaml` environment without changing its QPS, capacity, timing, or workload settings.

## Fitted service model

Prefill: 0.035418 ms/token × prompt tokens + 73.5321 ms.
Decode iteration: 0.738589 ms/request × batch + 0 ms/context-token × total context tokens + 56.0952 ms.
Rows: calibration 7311 prefill + 41370 decode; holdout 2528 prefill + 13836 decode.

| Service | Split | n | MAE (ms) | RMSE (ms) | R² | Mean prediction error (ms) |
|---|---:|---:|---:|---:|---:|---:|
| prefill | calibration | 7311 | 55.248 | 122.057 | 0.0303 | 0.000 |
| prefill | holdout_seed404 | 2528 | 55.098 | 249.818 | -0.0031 | -0.736 |
| decode | calibration | 41370 | 42.361 | 149.423 | 0.0319 | -0.000 |
| decode | holdout_seed404 | 13836 | 39.134 | 135.160 | 0.0325 | 5.001 |
| all | calibration | 48681 | 44.296 | 145.642 | 0.0354 | -0.000 |
| all | holdout_seed404 | 16364 | 41.600 | 158.390 | 0.0237 | 4.115 |

Decode identifiability: calibration batch size and total context-token count have correlation 0.9893. The context-token coefficient reached the nonnegative lower bound of zero; this is not evidence that context length has no cost, because the two predictors are too correlated to separate reliably.

## Fitted migration model

Global `(alpha + beta × blocks)`: (595.4620380026928, 3.9159924139507627, 685.7839515672595, 74); directed-pair models: 6.

| Split | n | MAE (ms) | RMSE (ms) | R² | Mean prediction error (ms) |
|---|---:|---:|---:|---:|---:|
| calibration | 74 | 347.810 | 631.908 | 0.1530 | 42.399 |
| holdout_seed404 | 31 | 385.359 | 537.601 | 0.0787 | 72.881 |

Holdout terminal-event statuses: {'MigrationStatus.FINISHED': 26, 'MigrationStatus.ABORTED_DST': 5}; prediction sources: {'global': 12, 'pair': 19}; copied-block range: [0.0, 66.0].

## Interpretation

Holdout mean bias is low for prefill and decode after the constrained refit, but per-row R² is weak and migration duration error remains large. Treat these curves as first-order simulator parameters; they do not validate counterfactual PPO policy outcomes or support real-cluster performance claims.

Collection provenance note:
{'reused_completed_runs': 17, 'supplemental_runs': ['qps_21_seed202', 'qps_21_seed303', 'qps_21_seed404'], 'note': 'The 17 preserved runs were collected before the scheduler migration-out slot reservation guard was added. The three supplemental runs were collected after that guard was added. Workload, model, serving configuration, and latency instrumentation were kept the same; the scheduler code version differs for these supplemental runs.'}
