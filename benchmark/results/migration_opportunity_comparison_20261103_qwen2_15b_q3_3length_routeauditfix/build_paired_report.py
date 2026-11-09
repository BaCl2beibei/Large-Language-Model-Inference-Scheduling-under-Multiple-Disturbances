#!/usr/bin/env python3
"""Build a trace- and GPU-audited report from completed real episodes."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = Path(__file__).resolve().parent
LEDGER = RUN_DIR / "completed_runs.jsonl"
POLICIES = ("Stay", "Balanced-L", "Balanced-M", "Balanced-H")
SCENARIO_ORDER = (
    "i512_o512_q3_s192_b4096",
    "i512_o1024_q3_s192_b4096",
    "i1024_o1024_q3_s192_b4096",
)
RESOURCE_FIELDS = (
    "kv_pct", "gpu_pct", "memory_mib", "memory_pct", "running", "waiting", "output_tps",
)


def load(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(numerator, denominator):
    return 100 * numerator / denominator if numerator is not None and denominator not in (None, 0) else None


def display(value, digits=2):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def write_csv(path: Path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def record_key(record):
    return record["case"]["scenario"], record["seed"], record["policy"]


def main():
    records = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
    by_key = {}
    for record in records:
        key = record_key(record)
        if record["policy"] == "Stay" and record["phase"] not in ("scan", "paired"):
            continue
        if record["policy"] != "Stay" and record["phase"] != "paired":
            continue
        by_key[key] = record

    available_scenarios = []
    missing = []
    for scenario in SCENARIO_ORDER:
        seed_set = {seed for (name, seed, _policy) in by_key if name == scenario}
        for seed in sorted(seed_set):
            expected = {(scenario, seed, policy) for policy in POLICIES}
            absent = sorted(policy for name, s, policy in expected if (name, s, policy) not in by_key)
            if absent:
                missing.append({"scenario": scenario, "seed": seed, "missing_policies": absent})
            else:
                available_scenarios.append((scenario, seed))

    summary_rows = []
    gpu_rows = []
    balance_rows = []
    migration_rows = []
    audit_groups = []
    report_sections = []

    for scenario, seed in available_scenarios:
        group = {}
        gpu_maps = {}
        checks = {}
        for policy in POLICIES:
            record = by_key[(scenario, seed, policy)]
            summary_path = RUN_DIR / record["summary"]
            summary = load(summary_path)
            gpu_map_path = summary_path.parent / "gpu_mapping.json"
            gpu_map = load(gpu_map_path)
            group[policy] = (record, summary, summary_path, gpu_map_path)
            gpu_maps[policy] = gpu_map
            checks[policy] = {
                "summary_status": summary.get("status"),
                "route_audit": summary.get("routes", {}).get("status"),
                "submission_audit": summary.get("submission", {}).get("status"),
                "trace_hash": summary.get("trace_hash"),
                "summary_sha256": sha256(summary_path),
                "resource_stream_sha256": sha256(summary_path.parent / "resources.jsonl"),
                "summary_path": str(summary_path.relative_to(RUN_DIR)),
                "window_sec": summary.get("window", {}).get("end", 0) - summary.get("window", {}).get("start", 0),
                "physical_gpu_uuids": sorted(gpu_map.values()),
            }

        stay_record, stay, _stay_path, _stay_map_path = group["Stay"]
        trace_hash = stay["trace_hash"]
        stay_gpus = set(gpu_maps["Stay"].values())
        for policy in POLICIES:
            if group[policy][1]["trace_hash"] != trace_hash:
                raise RuntimeError(f"trace mismatch: {scenario} seed={seed} {policy}")
            if set(gpu_maps[policy].values()) != stay_gpus or len(stay_gpus) != 4:
                raise RuntimeError(f"physical GPU mapping mismatch: {scenario} seed={seed} {policy}")
            summary = group[policy][1]
            check = checks[policy]
            if summary.get("status") != "passed" or check["route_audit"] != "passed" or check["submission_audit"] != "passed":
                raise RuntimeError(f"episode or request audit failed: {scenario} seed={seed} {policy}")
            if abs(check["window_sec"] - 600.0) > 2.5:
                raise RuntimeError(f"unexpected measurement window: {scenario} seed={seed} {policy}")
        audit_groups.append({
            "scenario": scenario,
            "seed": seed,
            "same_trace_hash": trace_hash,
            "same_physical_gpu_set": True,
            "all_episodes_passed": True,
            "all_route_audits_passed": True,
            "all_submission_audits_passed": True,
            "episodes": checks,
        })

        # Convert to stable physical GPU UUIDs. Instance IDs are per-service identities.
        for policy in POLICIES:
            record, summary, summary_path, _ = group[policy]
            gpu_map = gpu_maps[policy]
            metric = summary["metrics"]
            resource = summary["resources"]
            mig = summary["migration_statistics"]
            opportunity_rows = summary.get("opportunity_rejections", [])
            decisions = len(opportunity_rows)
            legal = sum(row.get("legal_pairs", 0) > 0 for row in opportunity_rows)
            legal_actions = sum(row.get("legal_pairs", 0) > 0 and row.get("action", 0) > 0 for row in opportunity_rows)
            submitted = mig.get("submitted", 0)
            started = mig.get("started", 0)
            succeeded = mig.get("succeeded", 0)
            success_rate_started = succeeded / started if started else None

            row = {
                "scenario": scenario, "seed": seed, "policy": policy,
                "threshold_F": record.get("threshold"), "trace_hash": summary["trace_hash"],
                "offered": metric.get("offered"), "sent": metric.get("sent"),
                "accepted": metric.get("accepted"), "completed": metric.get("completed"),
                "completion_rate": metric.get("completion_rate"),
                "slo_completed": metric.get("slo_completed"), "slo_goodput_rps": metric.get("slo_goodput_rps"),
                "ttft_p95_ms": metric.get("ttft_p95_ms"), "tpot_p95_ms": metric.get("tpot_p95_ms"),
                "completed_throughput_rps": metric.get("completed_throughput_rps"),
                "client_token_throughput_tps": metric.get("client_token_throughput_tps"),
                "e2e_mean_ms": metric.get("e2e_mean_ms"), "e2e_p95_ms": metric.get("e2e_p95_ms"),
                "nonmanual_errors": metric.get("nonmanual_errors"),
                "nonmanual_error_rate": metric.get("nonmanual_error_rate"),
                "deadline_unfinished_accepted": metric.get("terminal_censored"),
                "outstanding_ids_count": len(metric.get("outstanding_ids", [])),
                "waiting_mean_4gpu": resource["waiting"]["cluster_mean"],
                "waiting_p95_worst_gpu": max((v["p95"] for v in resource["waiting"]["instances"].values() if v["p95"] is not None), default=None),
                "decision_count": decisions, "selected_migration_actions": mig.get("selected"),
                "action_share_all_decisions_pct": pct(mig.get("selected", 0), decisions),
                "legal_opportunity_decisions": legal,
                "selected_with_legal_opportunity": legal_actions,
                "action_share_legal_opportunities_pct": pct(legal_actions, legal),
                "submitted": submitted, "started": started, "succeeded": succeeded,
                "successes_per_minute": mig.get("successes_per_minute"),
                "migrated_request_coverage_pct": pct(mig.get("request_coverage"), 1.0),
                "success_rate_started_pct": pct(success_rate_started, 1.0),
                "rejected": mig.get("rejected"), "failed": mig.get("failed"), "aborted": mig.get("aborted"),
                "migration_duration_mean_sec": mig.get("duration_mean_sec"),
                "migration_duration_p95_sec": mig.get("duration_p95_sec"),
                "window_sec": checks[policy]["window_sec"],
            }
            summary_rows.append(row)

            for metric_name in RESOURCE_FIELDS:
                stats = resource[metric_name]
                per_gpu = []
                for instance_id, stat in stats["instances"].items():
                    gpu_uuid = gpu_map[instance_id]
                    per_gpu.append((gpu_uuid, stat))
                    gpu_rows.append({
                        "scenario": scenario, "seed": seed, "policy": policy,
                        "threshold_F": record.get("threshold"), "gpu_uuid": gpu_uuid,
                        "metric": metric_name, "time_weighted_mean": stat.get("mean"),
                        "time_weighted_p95": stat.get("p95"), "minimum": stat.get("min"),
                        "maximum": stat.get("max"), "coverage": stat.get("coverage"),
                    })
                means = [stat["mean"] for _, stat in per_gpu if stat["mean"] is not None]
                p95s = [stat["p95"] for _, stat in per_gpu if stat["p95"] is not None]
                peaks = [stat["max"] for _, stat in per_gpu if stat["max"] is not None]
                row[f"{metric_name}_4gpu_mean"] = stats.get("cluster_mean")
                row[f"{metric_name}_per_gpu_p95_mean"] = statistics.mean(p95s) if p95s else None
                row[f"{metric_name}_worst_gpu_p95"] = max(p95s) if p95s else None
                row[f"{metric_name}_peak_any_gpu"] = max(peaks) if peaks else None
            row["output_tps_cluster_total"] = sum(
                stat["mean"] for stat in resource["output_tps"]["instances"].values() if stat["mean"] is not None
            )

        # Relative performance gains use the matching Stay run and fixed window.
        stay_metrics = stay["metrics"]
        for policy in POLICIES:
            if policy == "Stay":
                continue
            record, summary, _path, _map_path = group[policy]
            metrics = summary["metrics"]
            matching = next(r for r in summary_rows if r["scenario"] == scenario and r["seed"] == seed and r["policy"] == policy)
            matching.update({
                "slo_goodput_improvement_pct": pct(metrics.get("slo_goodput_rps", 0) - stay_metrics.get("slo_goodput_rps", 0), stay_metrics.get("slo_goodput_rps")),
                "ttft_p95_reduction_pct": pct(stay_metrics.get("ttft_p95_ms", 0) - metrics.get("ttft_p95_ms", 0), stay_metrics.get("ttft_p95_ms")),
                "tpot_p95_reduction_pct": pct(stay_metrics.get("tpot_p95_ms", 0) - metrics.get("tpot_p95_ms", 0), stay_metrics.get("tpot_p95_ms")),
                "completion_throughput_improvement_pct": pct(metrics.get("completed_throughput_rps", 0) - stay_metrics.get("completed_throughput_rps", 0), stay_metrics.get("completed_throughput_rps")),
                "output_token_throughput_improvement_pct": pct(metrics.get("client_token_throughput_tps", 0) - stay_metrics.get("client_token_throughput_tps", 0), stay_metrics.get("client_token_throughput_tps")),
                "e2e_mean_reduction_pct": pct(stay_metrics.get("e2e_mean_ms", 0) - metrics.get("e2e_mean_ms", 0), stay_metrics.get("e2e_mean_ms")),
                "e2e_p95_reduction_pct": pct(stay_metrics.get("e2e_p95_ms", 0) - metrics.get("e2e_p95_ms", 0), stay_metrics.get("e2e_p95_ms")),
                "waiting_mean_delta": summary["resources"]["waiting"]["cluster_mean"] - stay["resources"]["waiting"]["cluster_mean"],
                "waiting_worst_gpu_p95_delta": matching["waiting_p95_worst_gpu"] - max(v["p95"] for v in stay["resources"]["waiting"]["instances"].values() if v["p95"] is not None),
            })
            for metric_name in ("kv_pct", "gpu_pct", "running", "waiting"):
                stay_std = stay["resources"][metric_name]["dispersion"]["mean"]
                balanced_std = summary["resources"][metric_name]["dispersion"]["mean"]
                difference = stay_std - balanced_std if stay_std is not None and balanced_std is not None else None
                balance_rows.append({
                    "scenario": scenario, "seed": seed, "policy": policy, "metric": metric_name,
                    "stay_time_avg_across_gpu_std": stay_std,
                    "balanced_time_avg_across_gpu_std": balanced_std,
                    "absolute_improvement_stay_minus_balanced": difference,
                    "improvement_pct": pct(difference, stay_std),
                    "percent_status": "N/A (Stay dispersion zero or missing)" if stay_std in (None, 0) else "computed",
                })

    # The summary CSV includes one row per policy. Balance and audit are separate for clarity.
    write_csv(RUN_DIR / "one_seed_strategy_summary.csv", summary_rows)
    write_csv(RUN_DIR / "one_seed_gpu_resources.csv", gpu_rows)
    write_csv(RUN_DIR / "one_seed_balance_improvements.csv", balance_rows)
    audit = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "complete_scenario_seed_groups": [{"scenario": s, "seed": seed} for s, seed in available_scenarios],
        "missing_policy_groups": missing,
        "checks": audit_groups,
        "source_ledger_sha256": sha256(LEDGER),
        "protocol_path": "configs/three_scenario_one_seed_protocol.json",
        "threshold_selection_sha256": sha256(RUN_DIR / "threshold_selection.json"),
    }
    (RUN_DIR / "one_seed_pair_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    lines = [
        "# One-seed paired comparison: QPS 3",
        "",
        "Generated from the append-only run ledger and episode summaries. The resource CSV maps episode-local instance IDs to persistent GPU UUIDs.",
        "All values use the fixed 600 s scoring window (120 s arrival + 480 s drain). Means and quantiles for sampled resources use time-weighted summaries.",
        "Model/config: Qwen2-1.5B, max_model_len=2112, gpu_memory_utilization=0.65, max_num_seqs=192, max_num_batched_tokens=4096; QPS=3; seed=91000.",
        "Balanced thresholds are frozen per scenario from the Stay run's finite, non-idle arrival-phase F samples at empirical 25th/50th/75th percentiles. The exact Stay evidence paths and hashes are in `threshold_selection.json`.",
        "",
        f"Complete scenario/seed groups: {len(available_scenarios)}; missing groups: {len(missing)}.",
        "",
        "## Paired performance and outcomes",
        "",
        "SLO goodput is completed requests meeting the configured TTFT/TPOT bounds divided by 600 s. Goodput/throughput improvements are versus the same-seed Stay trace; latency reductions use (Stay − Balanced) / Stay.",
        "",
        "| Scenario | Seed | Policy | F threshold | SLO goodput req/s | Goodput Δ% | P95 TTFT ms | TTFT ↓% | P95 TPOT ms | TPOT ↓% | Completed/accepted | Completion rate | Errors | Deadline unfinished | E2E mean ms | E2E P95 ms | Client output tok/s |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append("| " + " | ".join([
            r["scenario"], str(r["seed"]), r["policy"], display(r["threshold_F"]),
            display(r["slo_goodput_rps"], 4), display(r.get("slo_goodput_improvement_pct")),
            display(r["ttft_p95_ms"]), display(r.get("ttft_p95_reduction_pct")),
            display(r["tpot_p95_ms"]), display(r.get("tpot_p95_reduction_pct")),
            f"{r['completed']}/{r['accepted']}", display(r["completion_rate"], 3),
            display(r["nonmanual_errors"]), display(r["deadline_unfinished_accepted"]),
            display(r["e2e_mean_ms"]), display(r["e2e_p95_ms"]), display(r["client_token_throughput_tps"]),
        ]) + " |")

    lines += ["", "## Secondary performance and queue changes vs Stay", "",
              "Throughput changes use the client completion ledger; resource throughput by GPU is reported in the resource table below. Waiting deltas are Balanced − Stay; negative values mean less waiting.",
              "", "| Scenario | Seed | Policy | Completion throughput req/s | Throughput Δ% | Client output tok/s | Output Δ% | E2E mean reduction % | E2E P95 reduction % | Waiting mean Δ | Worst-GPU Waiting P95 Δ |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary_rows:
        lines.append("| " + " | ".join([r["scenario"], str(r["seed"]), r["policy"],
            display(r["completed_throughput_rps"], 4), display(r.get("completion_throughput_improvement_pct")),
            display(r["client_token_throughput_tps"]), display(r.get("output_token_throughput_improvement_pct")),
            display(r.get("e2e_mean_reduction_pct")), display(r.get("e2e_p95_reduction_pct")),
            display(r.get("waiting_mean_delta")), display(r.get("waiting_worst_gpu_p95_delta"))]) + " |")

    lines += ["", "## Balance dispersion improvement", "",
              "σ is the time-weighted average of each sampled four-GPU population standard deviation. Improvement = (σStay − σBalanced) / σStay. A zero or missing Stay denominator is N/A; the absolute difference remains in the CSV.",
              "", "| Scenario | Seed | Policy | Metric | σ Stay | σ Balanced | Absolute Δ (Stay − Balanced) | Improvement % |", "|---|---:|---|---|---:|---:|---:|---:|"]
    for r in balance_rows:
        lines.append("| " + " | ".join([r["scenario"], str(r["seed"]), r["policy"], r["metric"],
            display(r["stay_time_avg_across_gpu_std"]), display(r["balanced_time_avg_across_gpu_std"]),
            display(r["absolute_improvement_stay_minus_balanced"]), display(r["improvement_pct"])]) + " |")

    lines += ["", "## Migration behavior", "",
              "Action share is selected migration actions / all decisions. Legal-opportunity action share is chosen migration actions when at least one legal pair existed / decisions with at least one legal pair. Requested migration success rate follows the requested denominator: succeeded / started (accepted and executed). Rejected requests are separate.",
              "", "| Scenario | Seed | Policy | Decisions | Selected | Action % all | Legal decisions | Selected with legal opportunity | Action % legal | Submitted | Started | Succeeded | Success % started | Success/min | Migrated request coverage % | Rejected | Failed | Mean cost s | P95 cost s |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary_rows:
        lines.append("| " + " | ".join([r["scenario"], str(r["seed"]), r["policy"], display(r["decision_count"]),
            display(r["selected_migration_actions"]), display(r["action_share_all_decisions_pct"]),
            display(r["legal_opportunity_decisions"]), display(r["selected_with_legal_opportunity"]),
            display(r["action_share_legal_opportunities_pct"]), display(r["submitted"]), display(r["started"]),
            display(r["succeeded"]), display(r["success_rate_started_pct"]), display(r["successes_per_minute"]),
            display(r["migrated_request_coverage_pct"]), display(r["rejected"]), display(r["failed"]),
            display(r["migration_duration_mean_sec"]), display(r["migration_duration_p95_sec"])]) + " |")

    lines += ["", "## Four-GPU resource summary", "",
              "For each metric, four-GPU mean is the arithmetic mean of the four per-GPU time-weighted means. P95 columns report the mean and worst of the four per-GPU time-weighted P95s. Output token throughput is summed over the four GPUs.",
              "", "| Scenario | Seed | Policy | KV mean % | KV mean GPU P95 % | KV worst GPU P95 % | KV peak % | GPU util mean % | GPU mean GPU P95 % | GPU worst GPU P95 % | GPU peak % | Memory mean MiB | Memory peak MiB | Memory util mean % | Running mean | Running worst GPU P95 | Waiting mean | Waiting worst GPU P95 | Output tok/s total |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary_rows:
        lines.append("| " + " | ".join([r["scenario"], str(r["seed"]), r["policy"],
            display(r["kv_pct_4gpu_mean"]), display(r["kv_pct_per_gpu_p95_mean"]), display(r["kv_pct_worst_gpu_p95"]), display(r["kv_pct_peak_any_gpu"]),
            display(r["gpu_pct_4gpu_mean"]), display(r["gpu_pct_per_gpu_p95_mean"]), display(r["gpu_pct_worst_gpu_p95"]), display(r["gpu_pct_peak_any_gpu"]),
            display(r["memory_mib_4gpu_mean"]), display(r["memory_mib_peak_any_gpu"]), display(r["memory_pct_4gpu_mean"]),
            display(r["running_4gpu_mean"]), display(r["running_worst_gpu_p95"]), display(r["waiting_4gpu_mean"]),
            display(r["waiting_worst_gpu_p95"]), display(r["output_tps_cluster_total"])]) + " |")

    lines += ["", "Per-GPU time-weighted means/P95/peaks, including GPU UUIDs and coverage, are in `one_seed_gpu_resources.csv`. Full episode audit paths and SHA-256 digests are in `one_seed_pair_audit.json`.",
              "", "## Interpretation", ""]
    for scenario, seed in available_scenarios:
        rows = {r["policy"]: r for r in summary_rows if r["scenario"] == scenario and r["seed"] == seed}
        baseline = rows["Stay"]
        statements = []
        for policy in POLICIES[1:]:
            r = rows[policy]
            gain = r.get("slo_goodput_improvement_pct")
            kv = next(x for x in balance_rows if x["scenario"] == scenario and x["seed"] == seed and x["policy"] == policy and x["metric"] == "kv_pct")
            statements.append(f"{policy}: SLO goodput {display(gain)}% vs Stay; KV dispersion improvement {display(kv['improvement_pct'])}%; migrated requests {display(r['migrated_request_coverage_pct'])}%.")
        lines.append(f"- **{scenario}, seed {seed}:** Stay SLO goodput {display(baseline['slo_goodput_rps'], 4)} req/s. " + " ".join(statements))

    report = RUN_DIR / "one_seed_comparison_report.md"
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "report": str(report), "complete_groups": len(available_scenarios),
        "missing_groups": missing, "summary_rows": len(summary_rows),
        "gpu_detail_rows": len(gpu_rows), "balance_rows": len(balance_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
