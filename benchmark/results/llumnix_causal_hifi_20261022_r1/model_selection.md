# Model selection record

## Split and training data

The selected artifacts in `models_seed606_909/` were fitted only from whole
calibration run groups with seeds 606 and 909. The independent seed 1213 was
not supplied to the fit. Service-model candidate selection held out complete
QPS groups; output-cadence selection compared QPS-interpolated quantile
profiles with one global quantile profile.

The service audit joined 45,718 engine-step measurements from 45,735 scheduled
batch events and 45,730 GPU timing events across 21 calibration runs. It keeps
engine-step wall time as the prediction target and GPU execution as a separate
explanatory observation.

## Selected components

| Component | Selected representation | Selection or evidence |
|:---|:---|:---|
| Batch execution | Ridge model over batch type, prefill/decode counts and tokens, context length, running/waiting counts, KV use, and scheduler flags | Whole-QPS grouped CV macro MAE 38.23 ms; type median 59.40 ms; nonlinear ridge 38.74 ms. |
| Output cadence | QPS-interpolated empirical quantile profiles of API-visible token intervals | Whole-QPS CV quantile-profile MAE 79.51 ms versus 141.61 ms for one global profile; 1,074,331 positive intervals. |
| Realized output length | Discrete Kaplan–Meier profiles with explicit unidentified tail mass; budget and realized length remain distinct | 87 training profiles; terminal and censored requests are retained. Seed 1213 is excluded. |
| Migration | Separate total, pause, token-gap, and recovery components | Only two matched successful calibration migrations. The fitted recovery term is not reliable enough for acceptance. |

The simulator uses a 50 ms event loop with instance queues, continuous batch
admission, model-based prefill/decode batch times, sampled output lengths, KV
block growth/release, preemption, and migration state. Migration ON invokes the
runtime Balanced/RemainingStepsLoad rule parsed from the run configuration.

## Known selection gap

The cadence model describes intervals between API-visible updates, but the
simulator still emits the first token and completes a request at the modeled
engine step. It has no separate engine-output-to-API delivery queue. The seed
1213 diagnostic overlay found multi-second and highly variable first-output
delivery delay at high OFF loads. This overlay is used only for error
attribution; it did not change the selected model. A delivery-queue model must
be fitted from calibration runs and evaluated on a fresh independent seed.

The migration model is also provisional: two training events cannot identify a
stable recovery distribution. No component selection from this round authorizes
formal PPO training.
