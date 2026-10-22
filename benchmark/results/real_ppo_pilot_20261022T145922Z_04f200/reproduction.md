# P1 CPU reproduction

From the project root, run:

```sh
.venv/bin/python benchmark/results/real_ppo_pilot_20261022T145922Z_04f200/p1_evidence/verify_p1.py
```

This runs 28 CPU unit/HTTP/ASGI tests, compilation, whitespace checks and a read-only GPU process inventory; exports the hand-computed cohort. It does not start Ray, an inference server, PPO, or a GPU experiment. Generated timestamps/log hashes change on rerun; revalidate gate bindings before consumption. Do not regenerate a passed gate without inspecting every result and the source/config/evidence hashes.

Interfaces, exact formulas, clock domains, required producer/offered manifests, cleanup limitations and unverified live paths are documented in implementation.md. Historic P0 evidence remains under its original gate; p1_evidence/p0_review.json provides this step's independent verification and canonical source-hash supplement.

## Step 03 CPU reproduction

```sh
.venv/bin/python benchmark/results/real_ppo_pilot_20261022T145922Z_04f200/p2_evidence/verify_p2.py
```

This reruns 63 CPU tests, compilation, whitespace checks, the minimal trace-contract reproduction and read-only GPU/supervisor inventories. It does not start a service, GPU experiment, simulation fitting or PPO. Saved smoke commands are review artifacts under commands/05_smoke.json; do not execute them in step 03. CPU --dry-run verification used all four prepared argv arrays with --dry-run appended; see p2_evidence/cli_validation_commands.json. implementation.md and smoke_spec.yaml describe contracts, boundaries and unverified live paths.

Re-running verification changes logs and hashes. Recheck source/config/evidence and actual exit codes before issuing a fresh gate. Do not silently hand-edit a gate status.
