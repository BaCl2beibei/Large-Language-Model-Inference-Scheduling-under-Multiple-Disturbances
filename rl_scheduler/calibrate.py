"""Fit Mig-PPO service-time and migration models from Llumnix experiment CSVs."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

from rl_scheduler.cost_estimator import MigPPOMigrationCostEstimator, nonnegative_lstsq


_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_profiling_data(value: str):
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (tuple, list)) and len(parsed) >= 4:
            kind = str(parsed[0]).strip(" '\"").lower()
            return kind, float(parsed[1]), float(parsed[2]), float(parsed[3])
    except (ValueError, SyntaxError, TypeError):
        pass
    text = str(value).strip()
    kind_match = re.search(r"prefill|decode", text, re.IGNORECASE)
    numbers = [float(x) for x in _NUMBER_RE.findall(text)]
    if not kind_match or len(numbers) < 3:
        return None
    # Most CSV rows use ``(kind, num_seqs, total_seq_len, latency_ms)``; the
    # leading string does not produce a number, so three numeric values remain.
    if len(numbers) >= 4:
        _, batch, seq_len, latency = numbers[-4:]
    else:
        batch, seq_len, latency = numbers[-3:]
    return kind_match.group(0).lower(), batch, seq_len, latency


def fit_service_profiles(path: str | Path) -> dict[str, object]:
    prefill_x, prefill_y = [], []
    decode_x, decode_y = [], []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = parse_profiling_data(row.get("profiling_data", ""))
            if not parsed:
                continue
            kind, batch, seq_len, latency = parsed
            if not all(math.isfinite(x) for x in (batch, seq_len, latency)) or latency <= 0:
                continue
            if kind == "prefill":
                prefill_x.append(seq_len)
                prefill_y.append(latency)
            elif kind == "decode":
                decode_x.append((batch, seq_len))
                decode_y.append(latency)
    if len(prefill_x) < 2 or len(decode_x) < 3:
        raise ValueError("profiling CSV needs at least 2 prefill and 3 decode samples")
    prefill_design = np.column_stack((np.asarray(prefill_x), np.ones(len(prefill_x))))
    prefill_coef = nonnegative_lstsq(prefill_design, np.asarray(prefill_y))
    decode_features = np.asarray(decode_x)
    decode_design = np.column_stack((decode_features[:, 0], decode_features[:, 1], np.ones(len(decode_x))))
    decode_coef = nonnegative_lstsq(decode_design, np.asarray(decode_y))
    return {
        "prefill_ms_per_token": float(prefill_coef[0]),
        "prefill_intercept_ms": float(prefill_coef[1]),
        "decode_ms_per_request": float(decode_coef[0]),
        "decode_ms_per_context_token": float(decode_coef[1]),
        "decode_intercept_ms": float(decode_coef[2]),
        "prefill_samples": len(prefill_x), "decode_samples": len(decode_x),
    }


def fit_migration_profile(path: str | Path) -> dict[str, object]:
    estimator = MigPPOMigrationCostEstimator().fit_csv(path)
    if estimator.global_model is None:
        raise ValueError("migration CSV needs at least two terminal cost samples")
    return estimator.state_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Mig-PPO simulation calibration from Llumnix logs")
    parser.add_argument("--instance-csv", required=True, help="Llumnix *_instance.csv with profiling_data")
    parser.add_argument("--migration-csv", required=True, help="Migration terminal-event CSV with blocks/cost_ms")
    parser.add_argument("--output", default="mig_ppo_calibration.yaml")
    args = parser.parse_args()
    service = fit_service_profiles(args.instance_csv)
    migration = fit_migration_profile(args.migration_csv)
    payload = {
        "environment": {
            **{key: value for key, value in service.items() if key.endswith("_ms") or key.endswith("_token") or key.endswith("_request") or key.endswith("_tokens")},
            "migration_cost_estimator": migration,
            "calibration_status": "service_and_migration_calibrated",
        },
        "calibration_metadata": {
            "instance_csv": str(Path(args.instance_csv)),
            "migration_csv": str(Path(args.migration_csv)),
            "service_fit_samples": {"prefill": service["prefill_samples"], "decode": service["decode_samples"]},
        },
    }
    Path(args.output).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(json.dumps({"output": args.output, "calibration_status": "service_and_migration_calibrated",
                      "samples": payload["calibration_metadata"]["service_fit_samples"]}, indent=2))


if __name__ == "__main__":
    main()

