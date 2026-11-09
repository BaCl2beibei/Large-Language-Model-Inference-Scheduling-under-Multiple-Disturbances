"""Prepare and summarize the bounded HOT1-H / NATIVE QPS follow-up."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "benchmark/results/mig_ppo_v2_multienv_20261105T130307Z"
NATIVE_QPS = (1.25, 1.5, 2.0)
NATIVE_ARRIVAL_SEED = 26110661
NATIVE_PROMPT_SEED = 26110662
HOT1_REPEAT_ARRIVAL_SEED = 26110671
HOT1_REPEAT_PROMPT_SEED = 26110672

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark import mig_ppo_v2_multienv as base


def _write_json(path: Path, value: object) -> None:
    base._atomic_json(path, value)


def prepare(run_dir: Path) -> dict:
    import yaml
    from tokenizers import Tokenizer

    run_dir = Path(run_dir).resolve()
    if run_dir.exists():
        raise FileExistsError(run_dir)
    if not PRIOR.is_dir():
        raise FileNotFoundError(PRIOR)

    # The follow-up uses the same already-audited serving configuration, with
    # only its Ray scratch path and config-file pointer scoped to this run.
    cfg = yaml.safe_load((PRIOR / "runtime_config.yaml").read_text())
    llumnix = yaml.safe_load((PRIOR / "real_llumnix.yaml").read_text())
    suffix = hashlib.sha256(str(run_dir).encode()).hexdigest()[:10]
    cfg["service"]["ray_temp_dir"] = f"/tmp/migppo_v2_followup_{suffix}"
    argv = cfg["service"]["server_argv"]
    argv[argv.index("--config-file") + 1] = str(run_dir / "real_llumnix.yaml")
    cfg_path = run_dir / "runtime_config.yaml"
    llumnix_path = run_dir / "real_llumnix.yaml"

    dirs = ("base_traces", "traces/selection", "runs", "comparisons",
            "causal_events", "services", "runtime")
    for name in dirs:
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    llumnix_path.write_text(yaml.safe_dump(llumnix, sort_keys=False))

    tokenizer = Tokenizer.from_file(str(Path(cfg["service"]["model_path"]) / "tokenizer.json"))
    matrix = []

    # Match HOT2-H/HOT3-H/HOT4-H's selected QPS=1 backup exactly in N, QPS,
    # arrival seed, prompt seed, length and hotspot fraction. Only hotspot
    # instance index changes.
    hot_id = "HOT1-H_backup_n48_q1p0_s26110523_followup"
    hot_path = run_dir / "traces/selection/HOT1-H" / f"{hot_id}.json"
    hot_meta = base._make_candidate_trace(tokenizer=tokenizer,
        candidate_id=hot_id, n=48, qps=1.0, arrival_seed=26110523,
        prompt_seed=26110521, environment_id="HOT1-H", output_dir=hot_path,
        hot_fraction=.90, hot_index=0)
    hot_meta.update(branch="matched_qps1_hot234", status="preregistered")
    matrix.append({"environment_id": "HOT1-H", "role": "training_candidate",
        "routing_mode": "fixed", "hotspot_instance_index": 0,
        "target_hotspot_fraction": .90, "intensity_id": "H",
        "status": "not_run", "candidates": [hot_meta],
        "selected_candidate_id": None, "selection_passed": False,
        "fit_status": "not_requested", "independent_validation_status": "not_requested"})

    native_candidates = []
    for qps in NATIVE_QPS:
        label = f"{qps:g}".replace(".", "p")
        candidate_id = f"NATIVE_n48_q{label}_s{NATIVE_ARRIVAL_SEED}"
        trace_path = run_dir / "traces/selection/NATIVE" / f"{candidate_id}.json"
        metadata = base._make_candidate_trace(tokenizer=tokenizer,
            candidate_id=candidate_id, n=48, qps=qps,
            arrival_seed=NATIVE_ARRIVAL_SEED, prompt_seed=NATIVE_PROMPT_SEED,
            environment_id="NATIVE", output_dir=trace_path)
        metadata.update(branch=f"native_qps_{label}", status="preregistered")
        native_candidates.append(metadata)
    matrix.append({"environment_id": "NATIVE", "role": "test_only",
        "routing_mode": "native", "hotspot_instance_index": None,
        "target_hotspot_fraction": None, "status": "not_run",
        "candidate_execution_order": "qps_sweep", "candidates": native_candidates,
        "selected_candidate_id": None, "selection_passed": False,
        "fit_status": "not_requested", "independent_validation_status": "not_requested"})

    run_id = run_dir.name
    _write_json(run_dir / "environment_matrix.json", matrix)
    _write_json(run_dir / "data_split.json", {
        "schema": "mig-ppo-v2-followup-data-split-v1",
        "prior_run": str(PRIOR.relative_to(ROOT)),
        "HOT1-H": "same base workload seed and QPS=1.0 candidate as H2-H/H3-H/H4-H; hotspot index only changed",
        "NATIVE": {"qps_tiers": list(NATIVE_QPS), "arrival_seed": NATIVE_ARRIVAL_SEED,
                    "prompt_seed": NATIVE_PROMPT_SEED,
                    "paired_within_each_tier": True,
                    "stop_after_first_positive_audited_tier": True},
        "independent_real_validation": "not requested; none will be collected"})
    (run_dir / "protocol.md").write_text(
        f"# HOT1-H / NATIVE follow-up\n\nRun `{run_id}`. The serving stack, Qwen2-1.5B model, four RTX 3060 workers, "
        "request sizes (128 input / 1536 output tokens), 200 s episode horizon, 1.5 s decision interval, "
        "0 s migration cooldown, SR migration and Balanced threshold rule match the prior run.\n\n"
        "HOT1-H uses N=48, QPS=1.0 and the same base trace seed/prompt seed as the selected HOT2-H/HOT3-H/HOT4-H "
        "QPS1 backup candidates; only the 90% hotspot route changes to logical instance 1.\n\n"
        f"NATIVE tries N=48 at QPS {', '.join(map(str, NATIVE_QPS))} in ascending order. Each tier is paired on its own "
        "trace. Stop this sweep after the first complete positive cost gain with at least one audited successful "
        "Balanced migration. The adoption rule is the frozen main cost C=mean(E2E)+CVaR95(E2E). No independent "
        "real validation, model fitting, PPO training or PPO update is in scope.\n")

    now = datetime.now(timezone.utc).isoformat()
    gpu = base._command("nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.used,utilization.gpu",
                        "--format=csv,noheader")
    active_gpu = base._command("nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader")
    supervisor = subprocess.run(["supervisorctl", "status"], capture_output=True, text=True, timeout=30)
    active_managed = [line for line in supervisor.stdout.splitlines()
                      if any(name in line for name in ("full_completion_", "p23_")) and " RUNNING " in line]
    if gpu["returncode"] or len([line for line in gpu["stdout"].splitlines() if line.strip()]) != 4:
        raise RuntimeError("follow-up requires four visible GPUs")
    if active_gpu["returncode"] or active_gpu["stdout"].strip():
        raise RuntimeError("GPU compute process is active; preserving it and not starting the follow-up")
    if active_managed:
        raise RuntimeError("managed serving experiment is active; preserving it: " + "\n".join(active_managed))

    old_audit = base._json(PRIOR / "environment_audit.json")
    expected_engine = old_audit["expected_engine"]
    expected_dispatch = old_audit["expected_dispatch"]
    model_identity = old_audit["model_identity"]
    audit = {"phase": "preflight", "captured_utc": now, "gpu_inventory": gpu,
        "gpu_processes_before": active_gpu,
        "supervisor_before": {"returncode": supervisor.returncode,
                              "stdout": supervisor.stdout, "stderr": supervisor.stderr},
        "model_identity": model_identity, "expected_engine": expected_engine,
        "expected_dispatch": expected_dispatch,
        "followup_scope": ["HOT1-H", "NATIVE"], "prior_run": str(PRIOR.relative_to(ROOT))}
    _write_json(run_dir / "environment_audit.json", audit)

    code_paths = base._code_paths() + ["benchmark/mig_ppo_v2_followup.py"]
    plan = {"schema": "mig-ppo-v2-followup-plan-v1", "run_id": run_id,
        "created_utc": now, "run_dir": str(run_dir), "prior_run": str(PRIOR.relative_to(ROOT)),
        "environment_ids": ["HOT1-H", "NATIVE"],
        "protocol": {"single_round_deadline_sec": base.EPISODE_HORIZON_SEC,
            "decision_interval_sec": base.DECISION_INTERVAL_SEC,
            "migration_cooldown_sec": base.MIGRATION_COOLDOWN_SEC,
            "request_count": 48, "prompt_tokens": 128, "output_tokens": 1536,
            "selection_rule": "complete audited same-trace pair; C=mean(E2E)+exact-fractional-tail-CVaR95(E2E); gain_C>0 and >=1 audited successful Balanced KV migration",
            "native_qps_tiers": list(NATIVE_QPS), "stop_on_first_positive_native_tier": True,
            "independent_real_validation": False, "ppo_training_or_update": False},
        "runtime_config_sha256": base._sha_file(cfg_path),
        "real_llumnix_sha256": base._sha_file(llumnix_path),
        "model_identity": model_identity,
        "code_hashes": {path: base._sha_file(ROOT / path) for path in code_paths},
        "taskbook_sha256": base._sha_file(ROOT / "rl_scheduler/docs/mig_ppo_v2_multienv_test_fit_prompt.md"),
        "environment_matrix_preregistered_sha256": base._canonical_hash(base._frozen_matrix_projection(matrix)),
        "frozen_before_gpu_collection": True}
    plan["freeze_sha256"] = base._canonical_hash(plan)
    _write_json(run_dir / "plan.json", plan)
    _write_json(run_dir / "budget.json", {"schema": "mig-ppo-v2-followup-budget-v1",
        "gpu_budget_limit_sec": 7200., "prior_gpu_reservation_sec": 0.,
        "gpu_reserved_origin_wall": None, "gpu_active_sec": 0., "gpu_hours": 0.,
        "phases": []})
    _write_json(run_dir / "selection_summary.json.pending", {
        "status": "not_started", "environment_count": 2,
        "environment_ids": ["HOT1-H", "NATIVE"]})
    (run_dir / "attempts.jsonl").touch()
    (run_dir / "results.csv").write_text(
        "environment_id,candidate_id,qps,status,stay_C_ms,balanced_C_ms,gain_C,successful_migrations\n")
    return {"run_dir": str(run_dir), "run_id": run_id,
            "environments": [row["environment_id"] for row in matrix],
            "native_qps_tiers": list(NATIVE_QPS), "plan_freeze_sha256": plan["freeze_sha256"]}


def prepare_hot1_repeat(run_dir: Path, qps: float = 1.0,
                        arrival_seed: int = HOT1_REPEAT_ARRIVAL_SEED,
                        prompt_seed: int = HOT1_REPEAT_PROMPT_SEED) -> dict:
    """Prepare one fresh HOT1-H QPS1 pair, with no validation stage."""
    run_dir = Path(run_dir).resolve()
    prepare(run_dir)
    import yaml
    from tokenizers import Tokenizer

    matrix_path = run_dir / "environment_matrix.json"
    matrix = base._json(matrix_path)
    hot = next(row for row in matrix if row["environment_id"] == "HOT1-H")
    cfg = yaml.safe_load((run_dir / "runtime_config.yaml").read_text())
    tokenizer = Tokenizer.from_file(str(Path(cfg["service"]["model_path"]) / "tokenizer.json"))
    qps_label = f"{qps:g}".replace(".", "p")
    candidate_id = f"HOT1-H_repeat_n48_q{qps_label}_s{arrival_seed}"
    trace_path = run_dir / "traces/selection/HOT1-H" / f"{candidate_id}.json"
    candidate = base._make_candidate_trace(tokenizer=tokenizer,
        candidate_id=candidate_id, n=48, qps=qps,
        arrival_seed=arrival_seed,
        prompt_seed=prompt_seed,
        environment_id="HOT1-H", output_dir=trace_path,
        hot_fraction=.90, hot_index=0)
    candidate.update(branch=f"fresh_seed_qps{qps_label}_repeat", status="preregistered")
    hot.update({"status": "not_run", "candidates": [candidate],
        "selected_candidate_id": None, "selection_passed": False,
        "fit_status": "not_requested", "independent_validation_status": "not_requested"})
    matrix = [hot]
    base._atomic_json(matrix_path, matrix)

    # Remove the unused Native candidates materialized by the shared preflight helper.
    for path in (run_dir / "traces/selection/NATIVE").glob("*.json"):
        path.unlink()
    (run_dir / "traces/selection/NATIVE").rmdir()
    for path in (run_dir / "traces/selection/HOT1-H").glob("*.json"):
        if path != trace_path:
            path.unlink()
    for path in (run_dir / "base_traces").glob("*.json"):
        if path.name != f"{candidate['base_trace_hash']}.json":
            path.unlink()

    run_id = run_dir.name
    plan_path = run_dir / "plan.json"
    plan = base._json(plan_path)
    plan["environment_ids"] = ["HOT1-H"]
    plan["protocol"] = {"schema": "mig-ppo-v2-hot1-repeat-protocol-v1",
        "run_id": run_id, "single_round_deadline_sec": base.EPISODE_HORIZON_SEC,
        "decision_interval_sec": base.DECISION_INTERVAL_SEC,
        "migration_cooldown_sec": base.MIGRATION_COOLDOWN_SEC,
        "n": 48, "qps": qps, "prompt_tokens": 128, "output_tokens": 1536,
        "routing_mode": "fixed", "hotspot_instance_index": 0,
        "actual_hotspot_fraction": candidate["actual_hotspot_fraction"],
        "selection_rule": "complete audited same-trace pair; C=mean(E2E)+exact-fractional-tail-CVaR95(E2E); gain_C>0 and >=1 audited successful Balanced KV migration",
        "independent_real_validation": False,
        "paired_against_prior_trace": False}
    plan["environment_matrix_preregistered_sha256"] = base._canonical_hash(
        base._frozen_matrix_projection(matrix))
    plan["repeat_candidate"] = {"prior_negative_pair": "HOT1-H_backup_n48_q1p0_s26110523_followup",
        "new_arrival_seed": arrival_seed,
        "new_prompt_seed": prompt_seed,
        "same_serving_and_scenario_settings": True,
        "fresh_trace": True}
    plan.pop("freeze_sha256", None)
    plan["freeze_sha256"] = base._canonical_hash(plan)
    base._atomic_json(plan_path, plan)
    base._atomic_json(run_dir / "data_split.json", {
        "schema": "mig-ppo-v2-hot1-repeat-split-v1",
        "prior_pair": "HOT1-H_backup_n48_q1p0_s26110523_followup",
        "repeat_candidate_id": candidate_id,
        "arrival_seed": arrival_seed,
        "prompt_seed": prompt_seed,
        "qps": qps,
        "same_trace_for_Stay_Balanced": True,
        "independent_real_validation": "not requested; none will be collected"})
    (run_dir / "protocol.md").write_text(
        f"# HOT1-H 新 trace 复测\n\nRun `{run_id}`。沿用相同环境设置：Qwen2-1.5B 四卡、48 请求、{qps:g} req/s、128 输入 token、1536 输出 token、1.5 s 决策、0 s 冷却、200 s 单轮窗口。热点为逻辑实例 1，{candidate['hotspot_request_count']}/48 请求（{candidate['actual_hotspot_fraction']:.3%}）。\n\n"
        f"本次使用新 arrival seed `{arrival_seed}` 和 prompt seed `{prompt_seed}`；Stay/Balanced 共用同一新 trace。若完整配对成本收益为正且有成功迁移审计，即判通过。独立真实验证、拟合及 PPO 更新均不在范围内。\n")
    base._atomic_json(run_dir / "selection_summary.json.pending", {
        "status": "not_started", "environment_count": 1,
        "environment_ids": ["HOT1-H"]})
    checked = base.validate_plan(run_dir)
    return {"run_dir": str(run_dir), "run_id": run_id,
        "candidate_id": candidate_id, "n": 48, "qps": qps,
        "actual_hotspot_fraction": candidate["actual_hotspot_fraction"],
        "plan_freeze_sha256": checked["freeze_sha256"]}


def report(run_dir: Path) -> dict:
    import csv

    run_dir = Path(run_dir).resolve()
    selection = base._json(run_dir / "selection_summary.json")
    rows = []
    for env in selection["environments"]:
        candidates = env.get("candidate_results", [])
        for candidate in candidates:
            rows.append({"environment_id": env["environment_id"],
                "candidate_id": candidate["candidate_id"],
                "qps": candidate.get("candidate_metadata", {}).get("target_qps"),
                "status": candidate.get("status"),
                "stay_C_ms": candidate.get("stay_C_ms"),
                "balanced_C_ms": candidate.get("balanced_C_ms"),
                "gain_C": candidate.get("gain_C"),
                "successful_migrations": candidate.get("balanced_successful_migrations", 0),
                "selection_passed": candidate.get("selection_passed", False)})
    fields = list(rows[0]) if rows else ["environment_id", "candidate_id", "qps", "status",
        "stay_C_ms", "balanced_C_ms", "gain_C", "successful_migrations", "selection_passed"]
    with (run_dir / "followup_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    split = base._json(run_dir / "data_split.json") if (run_dir / "data_split.json").is_file() else {}
    hot_note = split.get("HOT1-H") or split.get("same_trace_note") or "HOT1-H 使用固定热点路由至逻辑实例 1。"
    if isinstance(hot_note, dict):
        hot_note = json.dumps(hot_note, ensure_ascii=False)
    lines = ["# HOT1-H / NATIVE 实验补测报告", "",
        f"Run `{run_dir.name}`；结果状态：`{selection.get('status')}`。",
        "采用规则：同一 trace 的 Stay/Balanced 完整配对，Balanced 的 C=mean(E2E)+CVaR95(E2E) 更低，且至少一笔成功迁移通过审计。",
        "未采集独立真实验证数据。" + str(hot_note), "",
        "| 场景 | QPS | 状态 | Stay C (ms) | Balanced C (ms) | gain_C | 成功迁移数 | 通过 |",
        "|---|---:|---|---:|---:|---:|---:|---|"]
    for row in rows:
        fmt = lambda value: "—" if value is None else f"{float(value):.3f}"
        lines.append(f"| {row['environment_id']} | {fmt(row['qps'])} | {row['status']} | {fmt(row['stay_C_ms'])} | {fmt(row['balanced_C_ms'])} | {fmt(row['gain_C'])} | {row['successful_migrations']} | {row['selection_passed']} |")
    lines.extend(["", "逐请求路由、迁移回执、执行审计、trace hash 与每轮 200 秒窗口详见 `comparisons/` 和 `runs/`。", ""])
    (run_dir / "followup_report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"status": selection.get("status"), "candidate_count": len(rows),
            "passed": [row["candidate_id"] for row in rows if row["selection_passed"]],
            "report": str(run_dir / "followup_report.md")}


def recover_hot1_repeat(run_dir: Path) -> dict:
    """Recover a completed singleton HOT1-H pair after the collector indexing bug."""
    run_dir = Path(run_dir).resolve()
    plan_path = run_dir / "plan.json"
    plan = base._json(plan_path)
    frozen = dict(plan)
    expected_freeze = frozen.pop("freeze_sha256")
    if base._canonical_hash(frozen) != expected_freeze:
        raise RuntimeError("HOT1-H repeat plan freeze hash mismatch")

    matrix_path = run_dir / "environment_matrix.json"
    matrix = base._json(matrix_path)
    if len(matrix) != 1 or matrix[0].get("environment_id") != "HOT1-H":
        raise RuntimeError("expected a singleton HOT1-H repeat matrix")
    hot = matrix[0]
    candidate_id = hot["candidates"][0]["candidate_id"]
    comparison_path = run_dir / "comparisons" / f"{candidate_id}.json"
    comparison = base._json(comparison_path)
    if comparison.get("status") != "complete" or not comparison.get("same_trace_for_both_methods"):
        raise RuntimeError("persisted HOT1-H same-trace pair is incomplete")
    for policy in ("Stay", "Balanced"):
        episode = comparison.get("episodes", {}).get(policy, {})
        if (episode.get("status") != "passed" or episode.get("cohort_complete") is not True
                or episode.get("execution_audit_status") != "passed"
                or episode.get("route_audit", {}).get("status") != "passed"):
            raise RuntimeError(f"persisted HOT1-H {policy} audits are incomplete")
    if comparison.get("balanced_successful_migrations", 0) < 1:
        raise RuntimeError("HOT1-H pair lacks an audited successful Balanced migration")

    prior_summary_path = run_dir / "selection_summary.json"
    prior_summary = base._json(prior_summary_path)
    if prior_summary.get("status") == "complete":
        return {"status": "already_recovered", "candidate_id": candidate_id,
                "gain_C": comparison.get("gain_C")}
    if prior_summary.get("status") != "partial":
        raise RuntimeError("expected a partial summary from the recorded collector error")
    history_path = run_dir / "selection_summary_pre_recovery.json"
    if not history_path.exists():
        prior_summary_path.replace(history_path)
    hot.update({"candidate_results": [comparison], "selected_candidate_id": None,
        "selection_passed": bool(comparison.get("selection_passed")),
        "status": "selected" if comparison.get("selection_passed") else "skipped_no_positive_gain",
        "fit_status": "not_requested", "independent_validation_status": "not_requested"})
    if comparison.get("selection_passed"):
        hot["selected_candidate_id"] = candidate_id
        hot["selection"] = comparison
    hot.pop("global_failure", None)
    base._atomic_json(matrix_path, matrix)

    amendment = {"schema": "mig-ppo-v2-hot1-repeat-recovery-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cause": "collector default backup branch indexed candidate 3 after a complete singleton HOT1-H pair",
        "collector_error": prior_summary.get("global_failure"),
        "preserved_completed_comparison": str(comparison_path.relative_to(run_dir)),
        "hot1_pair_replayed": False,
        "pair_evidence": {"Stay_complete": True, "Balanced_complete": True,
            "route_audits_passed": True, "execution_audits_passed": True,
            "balanced_successful_migrations": comparison["balanced_successful_migrations"]},
        "recovery_action": "rehydrate selection summary from immutable comparison artifact; no new data collected",
        "collector_fix": "guard backup index against candidate list length; single-candidate row closes after its first complete pair"}
    amendment_path = run_dir / "execution_amendment.json"
    base._atomic_json(amendment_path, amendment)
    code_paths = base._code_paths() + ["benchmark/mig_ppo_v2_followup.py"]
    plan["code_hashes"] = {path: base._sha_file(ROOT / path) for path in code_paths}
    plan["execution_amendment_sha256"] = base._sha_file(amendment_path)
    plan["recovery_note"] = "Persisted HOT1-H pair recovered without replay after collector IndexError."
    plan.pop("freeze_sha256", None)
    plan["freeze_sha256"] = base._canonical_hash(plan)
    base._atomic_json(plan_path, plan)

    summary = {"schema": "mig-ppo-v2-selection-summary-v1", "status": "complete",
        "global_failure": None, "environments": matrix,
        "evaluation_complete_before_unified_fit": True,
        "fit_started_during_selection": False,
        "selection_frozen_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "Constrained-Balanced",
        "primary_cost": "mean(E2E)+exact-fractional-tail-CVaR95(E2E)",
        "execution_amendment_sha256": plan["execution_amendment_sha256"],
        "prior_partial_summary_sha256": base._sha_file(history_path),
        "summary_history": [{"path": history_path.name,
            "sha256": base._sha_file(history_path), "status": "partial"}],
        "collector_recovery": "pair evidence preserved from comparisons artifact; no replay"}
    summary["selection_sha256"] = base._canonical_hash(summary)
    base._atomic_json(prior_summary_path, summary)
    (run_dir / "selection_summary.json.pending").unlink(missing_ok=True)
    checked = base.validate_plan(run_dir)
    return {"status": "complete", "plan_validated": True,
        "candidate_id": candidate_id, "gain_C": comparison.get("gain_C"),
        "selection_passed": comparison.get("selection_passed"),
        "successful_migrations": comparison.get("balanced_successful_migrations"),
        "freeze_sha256": checked["freeze_sha256"]}


def resume_native_after_singleton_bug(run_dir: Path) -> dict:
    """Register HOT1-H's persisted pair and resume only the untouched NATIVE sweep."""
    run_dir = Path(run_dir).resolve()
    plan_path = run_dir / "plan.json"
    plan = base._json(plan_path)
    prior_freeze = dict(plan)
    prior_digest = prior_freeze.pop("freeze_sha256")
    if base._canonical_hash(prior_freeze) != prior_digest:
        raise RuntimeError("follow-up plan was modified before the recovery amendment")
    selection_path = run_dir / "selection_summary.json"
    if not selection_path.is_file() or base._json(selection_path).get("status") != "partial":
        raise RuntimeError("expected the partial collection summary from the recorded singleton-candidate error")
    matrix_path = run_dir / "environment_matrix.json"
    matrix = base._json(matrix_path)
    hot = next(row for row in matrix if row["environment_id"] == "HOT1-H")
    native = next(row for row in matrix if row["environment_id"] == "NATIVE")
    candidate_id = hot["candidates"][0]["candidate_id"]
    comparison_path = run_dir / "comparisons" / f"{candidate_id}.json"
    comparison = base._json(comparison_path)
    if comparison.get("status") != "complete":
        raise RuntimeError("HOT1-H pair is not complete; do not register it as consumed")
    for policy in ("Stay", "Balanced"):
        episode = comparison["episodes"].get(policy, {})
        if (episode.get("status") != "passed" or not episode.get("cohort_complete")
                or episode.get("execution_audit_status") != "passed"
                or episode.get("route_audit", {}).get("status") != "passed"):
            raise RuntimeError(f"HOT1-H {policy} evidence did not pass; refusing to resume")
    if comparison.get("selection_passed"):
        raise RuntimeError("HOT1-H unexpectedly passed; preserve it as selected and inspect before resuming")
    if native.get("status") != "not_run" or native.get("candidate_results"):
        raise RuntimeError("NATIVE already has a started/completed candidate; refusing to replay it")

    hot.update({"candidate_results": [comparison], "selected_candidate_id": None,
        "selection_passed": False, "status": "skipped_no_positive_gain",
        "fit_status": "not_in_fit_set", "independent_validation_status": "not_requested"})
    hot.pop("global_failure", None)
    base._atomic_json(matrix_path, matrix)
    amendment = {"schema": "mig-ppo-v2-followup-recovery-amendment-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cause": "collector indexed a third candidate after a complete HOT1-H singleton pair; the row intentionally had one preregistered candidate",
        "preserved_completed_comparison": str(comparison_path.relative_to(run_dir)),
        "hot1_pair_replayed": False,
        "hot1_status": hot["status"],
        "native_candidates_started": 0,
        "action": "rehydrate the audited HOT1-H result from its immutable comparison file and continue only with the untouched NATIVE QPS sweep",
        "collector_fix": "single-candidate rows now end cleanly without accessing an absent backup"}
    amendment_path = run_dir / "execution_amendment.json"
    base._atomic_json(amendment_path, amendment)

    code_paths = base._code_paths() + ["benchmark/mig_ppo_v2_followup.py"]
    plan["code_hashes"] = {path: base._sha_file(ROOT / path) for path in code_paths}
    plan["execution_amendment_sha256"] = base._sha_file(amendment_path)
    plan["recovery_note"] = "HOT1-H evidence was completed before the collector error; only NATIVE remains to run."
    plan.pop("freeze_sha256", None)
    plan["freeze_sha256"] = base._canonical_hash(plan)
    base._atomic_json(plan_path, plan)
    checked = base.validate_plan(run_dir)
    return {"plan_validated": True, "run_id": checked["run_id"],
        "hot1_candidate_id": candidate_id, "hot1_gain_C": comparison["gain_C"],
        "hot1_selection_passed": comparison["selection_passed"],
        "native_candidates_to_run": [row["candidate_id"] for row in native["candidates"]],
        "native_started": 0, "amendment_sha256": plan["execution_amendment_sha256"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "prepare-hot1-repeat", "recover-hot1-repeat", "report", "resume-native"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--qps", type=float, default=1.0)
    parser.add_argument("--arrival-seed", type=int, default=HOT1_REPEAT_ARRIVAL_SEED)
    parser.add_argument("--prompt-seed", type=int, default=HOT1_REPEAT_PROMPT_SEED)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.run_dir)
    elif args.command == "prepare-hot1-repeat":
        result = prepare_hot1_repeat(args.run_dir, qps=args.qps,
            arrival_seed=args.arrival_seed, prompt_seed=args.prompt_seed)
    elif args.command == "recover-hot1-repeat":
        result = recover_hot1_repeat(args.run_dir)
    elif args.command == "report":
        result = report(args.run_dir)
    else:
        result = resume_native_after_singleton_bug(args.run_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
