#!/usr/bin/env python3
"""Summarize Llumnix instance-info and GPU samples from an experiment result."""

from __future__ import annotations

import argparse
import ast
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def measurement_bounds(root: Path, mode: str, measure_sec: float) -> tuple[float, float]:
    requests = pd.read_csv(root / "requests.csv")
    requests = requests[(requests["mode"] == mode) & (requests["phase"] == "measurement")]
    requests = requests.dropna(subset=["actual_send_wall_time", "actual_send_sec"])
    if requests.empty:
        raise ValueError(f"No measurement request timestamps found for mode={mode}")
    # The sender stores wall and monotonic clocks relative to the same phase start.
    start_epoch = (requests["actual_send_wall_time"] - requests["actual_send_sec"]).median()
    return float(start_epoch), float(start_epoch + measure_sec)


def available_measurement_modes(root: Path) -> list[str]:
    request_path = root / "requests.csv"
    if not request_path.exists():
        return []
    requests = pd.read_csv(request_path, usecols=["mode", "phase"])
    present = set(requests.loc[requests["phase"].eq("measurement"), "mode"].astype(str))
    return [mode for mode in ("off", "on") if mode in present]


def load_instance_samples(root: Path, mode: str, measure_sec: float) -> pd.DataFrame:
    start_epoch, end_epoch = measurement_bounds(root, mode, measure_sec)
    source = root / "server_logs" / f"{mode}_instance_instance.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing Llumnix instance-info log: {source}")
    info = pd.read_csv(source)
    info["timestamp"] = pd.to_numeric(info["timestamp"], errors="coerce")
    info = info[info["timestamp"].between(start_epoch, end_epoch, inclusive="both")].copy()
    if info.empty:
        raise ValueError(f"No instance-info samples in the {measure_sec:g}s measurement window for {mode}")

    info["mode"] = mode
    info["relative_sec"] = info["timestamp"] - start_epoch
    info["gpu_cache_usage"] = pd.to_numeric(info["gpu_cache_usage"], errors="coerce")
    info["gpu_cache_usage_pct"] = info["gpu_cache_usage"] * 100.0
    if "gpu_ids" not in info:
        info["gpu_ids"] = ""
    info["num_available_gpu_blocks"] = pd.to_numeric(info["num_available_gpu_blocks"], errors="coerce")

    # Older manager logs only wrote available blocks and cache fraction. Recover
    # the missing block counts from available = total - used and usage = used / total.
    if "num_total_gpu_blocks" in info and info["num_total_gpu_blocks"].notna().any():
        total = pd.to_numeric(info["num_total_gpu_blocks"], errors="coerce")
    else:
        total = (info["num_available_gpu_blocks"] / (1.0 - info["gpu_cache_usage"])).round()
    if "num_used_gpu_blocks" in info and info["num_used_gpu_blocks"].notna().any():
        used = pd.to_numeric(info["num_used_gpu_blocks"], errors="coerce")
    else:
        used = (total * info["gpu_cache_usage"]).round()
    if "num_free_gpu_blocks" in info and info["num_free_gpu_blocks"].notna().any():
        free = pd.to_numeric(info["num_free_gpu_blocks"], errors="coerce")
    else:
        free = info["num_available_gpu_blocks"]

    info["num_total_gpu_blocks"] = total
    info["num_used_gpu_blocks"] = used
    info["num_free_gpu_blocks"] = free
    info["block_counts_source"] = "logged"
    if not {"num_total_gpu_blocks", "num_used_gpu_blocks", "num_free_gpu_blocks"}.issubset(
        set(pd.read_csv(source, nrows=0).columns)
    ):
        info["block_counts_source"] = "derived_from_cache_usage_and_available_blocks"

    for field in ("num_running_requests", "num_waiting_requests", "num_killed_requests"):
        info[field] = pd.to_numeric(info[field], errors="coerce").fillna(0)

    first_seen = info.groupby("instance_id")["timestamp"].min().sort_values()
    labels = {instance_id: f"Instance {i + 1}" for i, instance_id in enumerate(first_seen.index)}
    info["instance_label"] = info["instance_id"].map(labels)
    info["sample_interval_ms"] = info.groupby("instance_id")["timestamp"].diff() * 1000.0
    return info


def parse_gpu_ids(value) -> list[int]:
    if isinstance(value, (list, tuple)):
        items = value
    elif value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            items = parsed if isinstance(parsed, (list, tuple)) else [parsed]
        except (SyntaxError, ValueError):
            items = [part for part in text.replace("[", "").replace("]", "").split(",") if part.strip()]
    ids = []
    for item in items:
        try:
            ids.append(int(float(str(item).strip().strip("'\""))))
        except (TypeError, ValueError):
            continue
    return ids


def build_instance_gpu_metrics(root: Path, samples: pd.DataFrame, measure_sec: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    gpu_path = root / "gpu_metrics.csv"
    columns = ["mode", "scope", "instance_id", "instance_label", "gpu_index", "gpu_sample_count",
               "mean_gpu_utilization_pct", "p50_gpu_utilization_pct", "p95_gpu_utilization_pct",
               "peak_memory_used_mib", "memory_total_mib"]
    if not gpu_path.exists():
        return pd.DataFrame(), pd.DataFrame(columns=columns)
    gpu = pd.read_csv(gpu_path)
    gpu["wall_time"] = pd.to_numeric(gpu["wall_time"], errors="coerce")
    gpu["window_phase"] = "outside_measurement"
    for mode in available_measurement_modes(root):
        start_epoch, end_epoch = measurement_bounds(root, mode, measure_sec)
        mask = gpu["mode"].astype(str).eq(mode) & gpu["wall_time"].between(start_epoch, end_epoch, inclusive="both")
        gpu.loc[mask, "window_phase"] = "measurement"
    gpu = gpu[gpu["window_phase"].eq("measurement")].copy()
    gpu["gpu_index"] = pd.to_numeric(gpu["gpu_index"], errors="coerce")
    mapping_rows = []
    for (mode, instance_id), group in samples.groupby(["mode", "instance_id"]):
        ids = sorted({gpu_id for value in group["gpu_ids"] for gpu_id in parse_gpu_ids(value)})
        if len(ids) == 1:
            mapping_rows.append({"mode": mode, "instance_id": instance_id,
                                 "instance_label": group["instance_label"].iloc[0],
                                 "gpu_index": ids[0], "mapping_source": "Ray actor accelerator ID"})
    mapping = pd.DataFrame(mapping_rows)
    if mapping.empty:
        return pd.DataFrame(), pd.DataFrame(columns=columns)

    gpu["gpu_index"] = gpu["gpu_index"].astype("Int64")
    mapping["gpu_index"] = mapping["gpu_index"].astype("Int64")
    joined = gpu.merge(mapping, on=["mode", "gpu_index"], how="inner")
    joined["gpu_utilization_pct"] = pd.to_numeric(joined["gpu_utilization_pct"], errors="coerce")
    joined["memory_used_mib"] = pd.to_numeric(joined["memory_used_mib"], errors="coerce")
    joined["memory_total_mib"] = pd.to_numeric(joined["memory_total_mib"], errors="coerce")
    rows = []
    for (mode, instance_id, instance_label, gpu_index), group in joined.groupby(
        ["mode", "instance_id", "instance_label", "gpu_index"], sort=True
    ):
        util = group["gpu_utilization_pct"].dropna()
        mem_total = group["memory_total_mib"].dropna()
        rows.append({
            "mode": mode, "scope": "instance_gpu", "instance_id": instance_id,
            "instance_label": instance_label, "gpu_index": int(gpu_index),
            "gpu_sample_count": len(group), "mean_gpu_utilization_pct": util.mean(),
            "p50_gpu_utilization_pct": util.quantile(0.50),
            "p95_gpu_utilization_pct": util.quantile(0.95),
            "peak_memory_used_mib": group["memory_used_mib"].max(),
            "memory_total_mib": mem_total.max() if len(mem_total) else np.nan,
        })
    return joined, pd.DataFrame(rows, columns=columns)


def summarize_migrations(root: Path, measure_sec: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_path = root / "migration_events.csv"
    columns = ["mode", "measurement_logged_events", "measurement_successes", "measurement_non_successes",
               "success_cost_samples", "mean_success_cost_ms", "p50_success_cost_ms", "p95_success_cost_ms",
               "mean_all_logged_cost_ms", "phase_available"]
    events = pd.read_csv(event_path) if event_path.exists() and event_path.stat().st_size else pd.DataFrame()
    if events.empty or "observed_wall_time" not in events:
        rows = [{"mode": mode, "measurement_logged_events": 0, "measurement_successes": 0,
                 "measurement_non_successes": 0, "success_cost_samples": 0,
                 "mean_success_cost_ms": np.nan, "p50_success_cost_ms": np.nan,
                 "p95_success_cost_ms": np.nan, "mean_all_logged_cost_ms": np.nan,
                 "phase_available": False} for mode in ("off", "on")]
        return pd.DataFrame(), pd.DataFrame(rows, columns=columns)

    events["observed_wall_time"] = pd.to_numeric(events["observed_wall_time"], errors="coerce")
    if "logged_phase" not in events:
        events["logged_phase"] = events["phase"] if "phase" in events else "unknown"
    events["window_phase"] = "outside_measurement"
    for mode in available_measurement_modes(root):
        start_epoch, end_epoch = measurement_bounds(root, mode, measure_sec)
        mask = events["mode"].astype(str).eq(mode) & events["observed_wall_time"].between(start_epoch, end_epoch, inclusive="both")
        events.loc[mask, "window_phase"] = "measurement"
    events["phase"] = events["window_phase"]
    events.to_csv(event_path, index=False)
    measurement = events[events["window_phase"].eq("measurement")].copy()
    if "cost_ms" in measurement:
        measurement["cost_ms"] = pd.to_numeric(measurement["cost_ms"], errors="coerce")
    rows = []
    for mode in ("off", "on"):
        group = measurement[measurement["mode"].astype(str).eq(mode)]
        success = group["success"].astype(str).str.lower().eq("true") if "success" in group else pd.Series(False, index=group.index)
        successful_costs = group.loc[success, "cost_ms"].dropna() if "cost_ms" in group else pd.Series(dtype=float)
        all_costs = group["cost_ms"].dropna() if "cost_ms" in group else pd.Series(dtype=float)
        rows.append({
            "mode": mode, "measurement_logged_events": len(group),
            "measurement_successes": int(success.sum()),
            "measurement_non_successes": int((~success).sum()),
            "success_cost_samples": len(successful_costs),
            "mean_success_cost_ms": successful_costs.mean() if len(successful_costs) else np.nan,
            "p50_success_cost_ms": successful_costs.quantile(0.50) if len(successful_costs) else np.nan,
            "p95_success_cost_ms": successful_costs.quantile(0.95) if len(successful_costs) else np.nan,
            "mean_all_logged_cost_ms": all_costs.mean() if len(all_costs) else np.nan,
            "phase_available": True,
        })
    return measurement, pd.DataFrame(rows, columns=columns)


def summarize_instances(samples: pd.DataFrame, mode: str) -> list[dict]:
    rows: list[dict] = []
    groups = [("all_instances", samples)] + [
        (label, group) for label, group in samples.groupby("instance_label", sort=True)
    ]
    for entity, group in groups:
        instance_gpu_ids = sorted({gpu_id for value in group["gpu_ids"] for gpu_id in parse_gpu_ids(value)})
        rows.append({
            "mode": mode,
            "scope": "all_instances" if entity == "all_instances" else "instance",
            "entity": entity,
            "gpu_index": instance_gpu_ids[0] if len(instance_gpu_ids) == 1 else np.nan,
            "sample_count": len(group),
            "mean_sample_interval_ms": group["sample_interval_ms"].mean(),
            "p95_sample_interval_ms": group["sample_interval_ms"].quantile(0.95),
            "mean_gpu_cache_usage_pct": group["gpu_cache_usage_pct"].mean(),
            "p95_gpu_cache_usage_pct": group["gpu_cache_usage_pct"].quantile(0.95),
            "max_gpu_cache_usage_pct": group["gpu_cache_usage_pct"].max(),
            "mean_total_gpu_blocks": group["num_total_gpu_blocks"].mean(),
            "mean_used_gpu_blocks": group["num_used_gpu_blocks"].mean(),
            "mean_free_gpu_blocks": group["num_free_gpu_blocks"].mean(),
            "mean_running_requests": group["num_running_requests"].mean(),
            "p95_running_requests": group["num_running_requests"].quantile(0.95),
            "mean_waiting_requests": group["num_waiting_requests"].mean(),
            "p95_waiting_requests": group["num_waiting_requests"].quantile(0.95),
            "max_waiting_requests": group["num_waiting_requests"].max(),
            # num_killed_requests is a point-in-time gauge in this vLLM backend,
            # not an ever-increasing preemption-event counter.
            "max_num_killed_requests_gauge": group["num_killed_requests"].max(),
            "samples_with_nonzero_killed_requests_gauge": int((group["num_killed_requests"] > 0).sum()),
            "block_counts_source": ";".join(sorted(group["block_counts_source"].unique())),
        })
    return rows


def summarize_gpus(root: Path, mode: str, measure_sec: float) -> list[dict]:
    path = root / "gpu_metrics.csv"
    if not path.exists():
        return []
    gpu = pd.read_csv(path)
    gpu["wall_time"] = pd.to_numeric(gpu["wall_time"], errors="coerce")
    start_epoch, end_epoch = measurement_bounds(root, mode, measure_sec)
    gpu = gpu[(gpu["mode"] == mode) & gpu["wall_time"].between(start_epoch, end_epoch, inclusive="both")].copy()
    if gpu.empty:
        return []
    gpu["gpu_utilization_pct"] = pd.to_numeric(gpu["gpu_utilization_pct"], errors="coerce")
    gpu["memory_used_mib"] = pd.to_numeric(gpu["memory_used_mib"], errors="coerce")
    gpu["memory_total_mib"] = pd.to_numeric(gpu["memory_total_mib"], errors="coerce")
    rows = []
    for gpu_index, group in gpu.groupby("gpu_index", sort=True):
        rows.append({
            "mode": mode,
            "scope": "gpu",
            "entity": f"GPU {int(gpu_index)}",
            "gpu_sample_count": len(group),
            "mean_gpu_utilization_pct": group["gpu_utilization_pct"].mean(),
            "p95_gpu_utilization_pct": group["gpu_utilization_pct"].quantile(0.95),
            "peak_memory_used_mib": group["memory_used_mib"].max(),
            "memory_total_mib": group["memory_total_mib"].max(),
        })
    return rows


def refresh_summary_csv(root: Path, gpu_summary: pd.DataFrame,
                        migration_summary: pd.DataFrame, measure_sec: float) -> None:
    path = root / "summary.csv"
    summary = pd.read_csv(path)
    requests = pd.read_csv(root / "requests.csv")
    if "measurement_output_tokens_per_sec" not in summary:
        summary["measurement_output_tokens_per_sec"] = np.nan
    if "migration_logged_events" not in summary:
        summary["migration_logged_events"] = np.nan

    for index, row in summary[summary["row_type"].eq("replicate")].iterrows():
        mode = str(row["mode"])
        req = requests[(requests["mode"].astype(str).eq(mode)) & requests["phase"].astype(str).eq("measurement")].copy()
        completion = pd.to_numeric(req["completion_sec"], errors="coerce")
        success = req["success"].astype(str).str.lower().eq("true")
        in_window = req[success & completion.le(measure_sec)]
        output_tokens = pd.to_numeric(in_window["server_output_tokens"], errors="coerce").dropna().sum()
        summary.loc[index, "measurement_output_tokens_per_sec"] = output_tokens / measure_sec

        gpu_rows = gpu_summary[gpu_summary["mode"].astype(str).eq(mode)]
        util_by_id = {}
        for _, gpu_row in gpu_rows.iterrows():
            gpu_id = int(str(gpu_row["entity"]).split()[-1])
            util_by_id[gpu_id] = float(gpu_row["mean_gpu_utilization_pct"])
            summary.loc[index, f"gpu{gpu_id}_mean_util_pct"] = gpu_row["mean_gpu_utilization_pct"]
            summary.loc[index, f"gpu{gpu_id}_peak_memory_mib"] = gpu_row["peak_memory_used_mib"]
        if len(util_by_id) == 4:
            values = list(util_by_id.values())
            mean_util = float(np.mean(values))
            summary.loc[index, "gpu_util_range_pct"] = max(values) - min(values)
            summary.loc[index, "gpu_util_cv"] = float(np.std(values, ddof=0) / mean_util) if mean_util else np.nan

        mig = migration_summary[migration_summary["mode"].astype(str).eq(mode)]
        if not mig.empty:
            migration_row = mig.iloc[0]
            summary.loc[index, "migration_logged_events"] = migration_row["measurement_logged_events"]
            # These are terminal migration log records, not all internal attempts.
            summary.loc[index, "migration_attempts"] = np.nan
            summary.loc[index, "migration_successes"] = migration_row["measurement_successes"]
            summary.loc[index, "migration_failures"] = migration_row["measurement_non_successes"]
            summary.loc[index, "migration_mean_cost_ms"] = migration_row["mean_success_cost_ms"]
            summary.loc[index, "migration_cost_samples"] = migration_row["success_cost_samples"]

    replicate = summary[summary["row_type"].eq("replicate")]
    scalar_fields = [
        "measurement_output_tokens_per_sec", "migration_logged_events", "migration_attempts",
        "migration_successes", "migration_failures", "migration_mean_cost_ms", "migration_cost_samples",
        "gpu0_mean_util_pct", "gpu0_peak_memory_mib", "gpu1_mean_util_pct", "gpu1_peak_memory_mib",
        "gpu2_mean_util_pct", "gpu2_peak_memory_mib", "gpu3_mean_util_pct", "gpu3_peak_memory_mib",
        "gpu_util_range_pct", "gpu_util_cv",
    ]
    for (mode, qps_label), group in replicate.groupby(["mode", "qps_label"], dropna=False):
        aggregate_mask = summary["row_type"].eq("aggregate") & summary["mode"].eq(mode) & summary["qps_label"].eq(qps_label)
        for field in scalar_fields:
            if field not in group:
                continue
            values = pd.to_numeric(group[field], errors="coerce").dropna()
            mean = float(values.mean()) if len(values) else np.nan
            std = float(values.std(ddof=1)) if len(values) > 1 else (0.0 if len(values) == 1 else np.nan)
            summary.loc[aggregate_mask, f"{field}_mean"] = mean
            summary.loc[aggregate_mask, f"{field}_std"] = std
    summary.to_csv(path, index=False)


def refresh_throughput_plot(root: Path, measure_sec: float) -> None:
    summary = pd.read_csv(root / "summary.csv")
    rows = summary[summary["row_type"].eq("replicate")].sort_values("mode")
    modes = [mode for mode in ("off", "on") if mode in set(rows["mode"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    colors = {"off": "#2474B5", "on": "#E87500"}
    fields = [
        ("in_window_completion_qps", "Completed requests within 100 s", "requests/s"),
        ("measurement_output_tokens_per_sec", "Output tokens completed within 100 s", "tokens/s"),
    ]
    for ax, (field, title, ylabel) in zip(axes, fields):
        values = [pd.to_numeric(rows.loc[rows["mode"].eq(mode), field], errors="coerce").mean() for mode in modes]
        ax.bar([mode.upper() for mode in modes], values, color=[colors[mode] for mode in modes])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"Steady measurement-window throughput ({measure_sec:g} s; drain excluded)")
    fig.tight_layout()
    fig.savefig(root / "plots" / "throughput.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_plots(root: Path, samples: pd.DataFrame, gpu_summary: pd.DataFrame) -> list[str]:
    plots = root / "plots"
    plots.mkdir(exist_ok=True)
    modes = [mode for mode in ("off", "on") if mode in set(samples["mode"])]

    # Cache usage: per-mode average/P95, plus time series for each instance.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 2]})
    agg = samples.groupby("mode")["gpu_cache_usage_pct"].agg(
        mean="mean", p95=lambda x: x.quantile(0.95)
    ).reindex(modes)
    x = np.arange(len(modes))
    width = 0.34
    axes[0].bar(x - width / 2, agg["mean"], width, label="Mean")
    axes[0].bar(x + width / 2, agg["p95"], width, label="P95")
    axes[0].set_xticks(x, [mode.upper() for mode in modes])
    axes[0].set_ylabel("KV cache usage (%)")
    axes[0].set_title("KV cache utilization")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    for mode in modes:
        subset = samples[samples["mode"] == mode]
        for label, group in subset.groupby("instance_label", sort=True):
            # Scatter preserves the timestamped observations without inventing
            # a regular sampling cadence or smoothing the recorded values.
            axes[1].plot(group["relative_sec"], group["gpu_cache_usage_pct"],
                         linewidth=0.8, alpha=0.7, label=f"{mode.upper()} {label}")
    axes[1].set_xlabel("Measurement time (s)")
    axes[1].set_ylabel("KV cache usage (%)")
    axes[1].set_title("Per-instance samples")
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "instance_kv_cache.png", dpi=160)
    plt.close(fig)

    instance_summary_path = root / "instance_metrics_summary.csv"
    instance_summary = pd.DataFrame()
    if instance_summary_path.exists():
        instance_summary = pd.read_csv(instance_summary_path)
        instance_summary = instance_summary[instance_summary["scope"].eq("instance")]
        if not instance_summary.empty:
            fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 4.8), squeeze=False)
            for ax, mode in zip(axes[0], modes):
                group = instance_summary[instance_summary["mode"].eq(mode)].sort_values("entity")
                x = np.arange(len(group))
                width = 0.34
                ax.bar(x - width / 2, group["mean_gpu_cache_usage_pct"], width, label="Mean")
                ax.bar(x + width / 2, group["max_gpu_cache_usage_pct"], width, label="Maximum")
                ax.set_xticks(x, group["entity"])
                ax.set_title(mode.upper())
                ax.set_ylabel("KV cache usage (%)")
                ax.grid(axis="y", alpha=0.25)
                ax.legend()
            fig.suptitle("Per-instance KV cache mean and maximum")
            fig.tight_layout()
            fig.savefig(plots / "instance_kv_cache_per_instance.png", dpi=160)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(10, 5))
            groups = instance_summary[instance_summary["mode"].isin(modes)].copy()
            labels = [f"{mode.upper()}\n{entity}" for mode, entity in zip(groups["mode"], groups["entity"])]
            x = np.arange(len(groups))
            width = 0.38
            ax.bar(x - width / 2, groups["mean_waiting_requests"], width, label="Mean")
            ax.bar(x + width / 2, groups["max_waiting_requests"], width, label="Maximum")
            ax.set_xticks(x, labels, rotation=20)
            ax.set_ylabel("Waiting requests / instance")
            ax.set_title("Per-instance waiting queue mean and maximum")
            ax.grid(axis="y", alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(plots / "instance_waiting_queue.png", dpi=160)
            plt.close(fig)

    # Request pressure and preemption gauge per instance.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    instances = sorted(samples["instance_label"].dropna().unique())
    width = 0.8 / max(1, len(modes))
    xx = np.arange(len(instances))
    for i, mode in enumerate(modes):
        group = samples[samples["mode"] == mode].groupby("instance_label")
        stats = group[["num_running_requests", "num_waiting_requests", "num_killed_requests"]].mean()
        offset = (i - (len(modes) - 1) / 2) * width
        axes[0].bar(xx + offset, [stats.loc[k, "num_running_requests"] for k in instances],
                    width, label=mode.upper())
        axes[1].bar(xx + offset, [stats.loc[k, "num_waiting_requests"] for k in instances],
                    width, label=mode.upper())
        maxima = group["num_killed_requests"].max()
        axes[2].bar(xx + offset, [maxima.get(k, 0) for k in instances], width, label=mode.upper())
    for ax, title, ylabel in [
        (axes[0], "Mean running requests", "Requests / instance"),
        (axes[1], "Mean waiting requests", "Requests / instance"),
        (axes[2], "Max num_killed_requests gauge", "Gauge value"),
    ]:
        ax.set_xticks(xx, instances, rotation=25)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend()
    for ax, field, label in (
        (axes[1], "num_waiting_requests", "0 waiting requests in all samples"),
        (axes[2], "num_killed_requests", "0 observed in all samples"),
    ):
        if samples[field].max() == 0:
            ax.set_ylim(0, 1)
            ax.text(0.5, 0.72, label, transform=ax.transAxes,
                    ha="center", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots / "instance_request_pressure.png", dpi=160)
    plt.close(fig)

    if not gpu_summary.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        gpu_ids = sorted(gpu_summary["entity"].unique(), key=lambda s: int(s.split()[-1]))
        width = 0.36
        xx = np.arange(len(gpu_ids))
        for i, mode in enumerate(modes):
            rows = gpu_summary[gpu_summary["mode"] == mode].set_index("entity").reindex(gpu_ids)
            offset = (i - (len(modes) - 1) / 2) * width
            axes[0].bar(xx + offset, rows["mean_gpu_utilization_pct"], width, label=mode.upper())
            axes[1].bar(xx + offset, rows["peak_memory_used_mib"], width, label=mode.upper())
        axes[0].set_ylabel("Mean GPU utilization (%)")
        axes[0].set_title("GPU utilization during measurement")
        axes[1].set_ylabel("Peak memory used (MiB)")
        axes[1].set_title("Peak GPU memory during measurement")
        for ax in axes:
            ax.set_xticks(xx, gpu_ids)
            ax.grid(axis="y", alpha=0.25)
            ax.legend()
        fig.tight_layout()
        fig.savefig(plots / "gpu_utilization_memory_summary.png", dpi=160)
        plt.close(fig)
    
    instance_gpu_summary_path = root / "instance_gpu_summary.csv"
    instance_gpu_summary = pd.read_csv(instance_gpu_summary_path) if instance_gpu_summary_path.exists() else pd.DataFrame()
    if not instance_gpu_summary.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        entities = sorted(instance_gpu_summary["instance_label"].unique())
        xx = np.arange(len(entities))
        width = 0.36
        for i, mode in enumerate(modes):
            part = instance_gpu_summary[instance_gpu_summary["mode"] == mode].set_index("instance_label").reindex(entities)
            offset = (i - (len(modes) - 1) / 2) * width
            ax.bar(xx + offset, part["mean_gpu_utilization_pct"], width, label=mode.upper())
        ax.set_xticks(xx, [f"{label}\nGPU {int(instance_gpu_summary[instance_gpu_summary['instance_label'].eq(label)]['gpu_index'].iloc[0])}" for label in entities])
        ax.set_ylabel("Mean GPU utilization (%)")
        ax.set_title("Per-instance GPU utilization (Ray GPU mapping)")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / "instance_gpu_utilization.png", dpi=160)
        plt.close(fig)

    migration_plot_path = root / "migration_metrics_summary.csv"
    migration_summary = pd.read_csv(migration_plot_path) if migration_plot_path.exists() else pd.DataFrame()
    if not migration_summary.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
        xx = np.arange(len(modes))
        counts = migration_summary.set_index("mode").reindex(modes)
        axes[0].bar(xx, counts["measurement_logged_events"].fillna(0), color=["#2474B5", "#E87500"][:len(modes)])
        axes[0].set_xticks(xx, [mode.upper() for mode in modes])
        axes[0].set_ylabel("Logged migration events")
        axes[0].set_title("Measurement-window migration events")
        axes[0].grid(axis="y", alpha=0.25)
        axes[1].bar(xx - 0.18, counts["mean_success_cost_ms"], 0.36, label="Mean")
        axes[1].bar(xx + 0.18, counts["p95_success_cost_ms"], 0.36, label="P95")
        axes[1].set_xticks(xx, [mode.upper() for mode in modes])
        axes[1].set_ylabel("Logged migration duration (ms)")
        axes[1].set_title("Successful migration cost")
        axes[1].grid(axis="y", alpha=0.25)
        axes[1].legend()
        if counts["measurement_logged_events"].fillna(0).max() == 0:
            axes[1].text(0.5, 0.75, "No measurement-phase migration logged",
                         transform=axes[1].transAxes, ha="center", va="center")
        fig.tight_layout()
        fig.savefig(plots / "migration_metrics.png", dpi=160)
        plt.close(fig)

    names = ["instance_kv_cache.png", "instance_request_pressure.png"]
    if not instance_summary.empty:
        names.extend(["instance_kv_cache_per_instance.png", "instance_waiting_queue.png"])
    if not gpu_summary.empty:
        names.append("gpu_utilization_memory_summary.png")
    if not instance_gpu_summary.empty:
        names.append("instance_gpu_utilization.png")
    if not migration_summary.empty:
        names.append("migration_metrics.png")
    return names


def append_report(root: Path, samples: pd.DataFrame, summary: pd.DataFrame,
                  gpu_summary: pd.DataFrame, instance_gpu_summary: pd.DataFrame,
                  migration_summary: pd.DataFrame, plot_names: list[str], measure_sec: float) -> None:
    report_path = root / "report.md"
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else "# Experiment report\n"
    marker = "## 实例级 KV Cache 与排队指标"
    if marker in report:
        report = report.split(marker)[0].rstrip() + "\n\n"

    rows = summary[summary["scope"] == "all_instances"]
    instance_rows = summary[summary["scope"] == "instance"]
    lines = [
        marker,
        "",
        f"以下指标从 Llumnix 的 `*_instance_instance.csv` 中按请求测量窗截取（每种模式 {measure_sec:g} 秒）；样本数和采样间隔见 `instance_metrics_summary.csv`。",
        "",
        "| 模式 | 实例样本 | KV Cache 均值 | KV Cache P95 | 平均 total blocks | 平均 used blocks | 平均 free blocks | 平均运行请求/实例 | 平均等待请求/实例 | 等待请求最大值 | `num_killed_requests` 最大值 | 非零样本数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("off", "on"):
        if mode not in rows["mode"].values:
            continue
        row = rows[rows["mode"] == mode].iloc[0]
        lines.append(
            f"| {mode.upper()} | {int(row['sample_count'])} | {row['mean_gpu_cache_usage_pct']:.3f}% "
            f"| {row['p95_gpu_cache_usage_pct']:.3f}% | {row['mean_total_gpu_blocks']:.0f} "
            f"| {row['mean_used_gpu_blocks']:.1f} | {row['mean_free_gpu_blocks']:.1f} "
            f"| {row['mean_running_requests']:.2f} | {row['mean_waiting_requests']:.3f} "
            f"| {int(row['max_waiting_requests'])} "
            f"| {int(row['max_num_killed_requests_gauge'])} | {int(row['samples_with_nonzero_killed_requests_gauge'])} |"
        )

    if not instance_rows.empty:
        lines += [
            "",
            "各实例 KV Cache 峰值和排队情况：",
            "",
            "| 模式 | 实例 | GPU | KV 均值 | KV P95 | KV 最大值 | 平均运行请求 | 平均等待请求 | 最大等待请求 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for _, row in instance_rows.sort_values(["mode", "entity"]).iterrows():
            lines.append(
                f"| {row['mode'].upper()} | {row['entity']} | {int(row['gpu_index']) if pd.notna(row['gpu_index']) else 'N/A'} "
                f"| {row['mean_gpu_cache_usage_pct']:.3f}% "
                f"| {row['p95_gpu_cache_usage_pct']:.3f}% | {row['max_gpu_cache_usage_pct']:.3f}% "
                f"| {row['mean_running_requests']:.2f} | {row['mean_waiting_requests']:.2f} "
                f"| {int(row['max_waiting_requests'])} |"
            )

    if not gpu_summary.empty:
        all_gpu = gpu_summary.groupby("mode").agg(
            mean_util=("mean_gpu_utilization_pct", "mean"),
            peak_mem=("peak_memory_used_mib", "max"),
        )
        lines += [
            "",
            "GPU 计算活动均值来自精确墙钟测量窗内的 `gpu_metrics.csv` 样本；峰值显存取四卡各自峰值中的最大值：",
            "",
            "| 模式 | 四卡平均 GPU 利用率 | 四卡最高显存占用 |",
            "|---|---:|---:|",
        ]
        for mode in ("off", "on"):
            if mode in all_gpu.index:
                lines.append(f"| {mode.upper()} | {all_gpu.loc[mode, 'mean_util']:.2f}% | {all_gpu.loc[mode, 'peak_mem']:.0f} MiB |")

    if not instance_gpu_summary.empty:
        lines += [
            "",
            "实例与 GPU 的对应关系来自每个 Llumnix actor 的 Ray accelerator ID，并与 `nvidia-smi` 的 GPU index 对齐：",
            "",
            "| 模式 | 实例 | GPU | 平均 GPU 利用率 | P95 GPU 利用率 | 峰值显存 (MiB) |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for _, row in instance_gpu_summary.iterrows():
            lines.append(
                f"| {row['mode'].upper()} | {row['instance_label']} | {int(row['gpu_index'])} "
                f"| {row['mean_gpu_utilization_pct']:.2f}% | {row['p95_gpu_utilization_pct']:.2f}% "
                f"| {row['peak_memory_used_mib']:.0f} |"
            )

    if not migration_summary.empty:
        lines += [
            "",
            "迁移指标只计入正式测量阶段有完成日志的迁移操作：",
            "",
            "| 模式 | 记录迁移数 | 成功 | 非成功 | 成功迁移平均耗时 (ms) | P95 耗时 (ms) | 耗时样本数 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for _, row in migration_summary.iterrows():
            mean_cost = "N/A" if pd.isna(row["mean_success_cost_ms"]) else f"{row['mean_success_cost_ms']:.2f}"
            p95_cost = "N/A" if pd.isna(row["p95_success_cost_ms"]) else f"{row['p95_success_cost_ms']:.2f}"
            lines.append(
                f"| {row['mode'].upper()} | {int(row['measurement_logged_events'])} "
                f"| {int(row['measurement_successes'])} | {int(row['measurement_non_successes'])} "
                f"| {mean_cost} | {p95_cost} | {int(row['success_cost_samples'])} |"
            )

    lines += [
        "",
        "说明：逐实例样本保留了 `gpu_cache_usage`、total/used/free GPU blocks、运行/等待请求数及 `num_killed_requests`。块数是否来自原始字段或由缓存比例推导，见逐样本 CSV 的 `block_counts_source`。",
        "",
        "`num_killed_requests` 在当前 vLLM scheduler 中是采样时的抢占相关请求 gauge，不是累计抢占事件计数；因此这里只报告采样最大值和非零样本数。等待队列、KV 利用率均按实例快照统计。",
        "",
        "样本时间是管理器轮询得到的实例快照，不是固定频率重采样；图中保留原始点。`Instance 1–4` 是各模式内按首次出现时间命名，不代表 GPU 编号。GPU 利用率和显存图来自独立的 `nvidia-smi` 四卡采样。GPU 和迁移原始 CSV 的 `phase` 已按精确测量窗校正，采集器原始阶段标签保存在 `logged_phase`。",
        "迁移耗时取迁移完成日志中的端到端操作耗时，不等于被迁移请求的精确停顿时间；事件数也只覆盖可解析的迁移完成日志。",
        "",
        "新增图表：",
    ]
    lines.extend(f"- `plots/{name}`" for name in plot_names)
    if not instance_gpu_summary.empty:
        lines.append("- `instance_gpu_metrics.csv` / `instance_gpu_summary.csv`：Ray 实例 GPU 映射及对应利用率样本/汇总。")
    lines.append("- `migration_metrics_summary.csv` / `migration_measurement_events.csv`：正式测量窗迁移事件统计及原始筛选记录。")
    report_path.write_text(report.rstrip() + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")

    config_path = root / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        generated = config.setdefault("generated_artifacts", {})
        generated["plots"] = sorted((set(generated.get("plots", []) + plot_names)) - {"migration.png"})
        generated["analysis_csv"] = [
            "instance_metrics.csv", "instance_metrics_summary.csv", "gpu_metrics_summary.csv",
            "instance_gpu_metrics.csv", "instance_gpu_summary.csv", "migration_measurement_events.csv",
            "migration_metrics_summary.csv",
        ]
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path, help="Experiment result directory")
    args = parser.parse_args()
    root = args.result_dir.resolve()
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    measure_sec = float(config["measure_sec"])

    modes = available_measurement_modes(root)
    if not modes:
        raise ValueError(f"No formal measurement request records found in {root / 'requests.csv'}")
    sample_parts = [load_instance_samples(root, mode, measure_sec) for mode in modes]
    samples = pd.concat(sample_parts, ignore_index=True)
    samples.to_csv(root / "instance_metrics.csv", index=False)

    summary_rows = []
    gpu_rows = []
    gpu_path = root / "gpu_metrics.csv"
    if gpu_path.exists():
        gpu_raw = pd.read_csv(gpu_path)
        gpu_raw["wall_time"] = pd.to_numeric(gpu_raw["wall_time"], errors="coerce")
    if "logged_phase" not in gpu_raw:
        gpu_raw["logged_phase"] = gpu_raw["phase"]
        gpu_raw["window_phase"] = "outside_measurement"
        for mode in modes:
            start_epoch, end_epoch = measurement_bounds(root, mode, measure_sec)
            mask = gpu_raw["mode"].astype(str).eq(mode) & gpu_raw["wall_time"].between(start_epoch, end_epoch, inclusive="both")
            gpu_raw.loc[mask, "window_phase"] = "measurement"
        gpu_raw["phase"] = gpu_raw["window_phase"]
        gpu_raw.to_csv(gpu_path, index=False)
    for mode, group in samples.groupby("mode", sort=False):
        summary_rows.extend(summarize_instances(group, mode))
        gpu_rows.extend(summarize_gpus(root, mode, measure_sec))
    summary = pd.DataFrame(summary_rows)
    gpu_summary = pd.DataFrame(gpu_rows)
    summary.to_csv(root / "instance_metrics_summary.csv", index=False)
    gpu_summary.to_csv(root / "gpu_metrics_summary.csv", index=False)

    instance_gpu_samples, instance_gpu_summary = build_instance_gpu_metrics(root, samples, measure_sec)
    instance_gpu_samples.to_csv(root / "instance_gpu_metrics.csv", index=False)
    instance_gpu_summary.to_csv(root / "instance_gpu_summary.csv", index=False)
    migration_events, migration_summary = summarize_migrations(root, measure_sec)
    migration_events.to_csv(root / "migration_measurement_events.csv", index=False)
    migration_summary.to_csv(root / "migration_metrics_summary.csv", index=False)

    refresh_summary_csv(root, gpu_summary, migration_summary, measure_sec)
    plot_names = save_plots(root, samples, gpu_summary)
    throughput_plot = root / "plots" / "throughput.png"
    if throughput_plot.parent.exists():
        refresh_throughput_plot(root, measure_sec)
    gpu_plot = root / "plots" / "gpu_utilization_memory_summary.png"
    if gpu_plot.exists():
        shutil.copyfile(gpu_plot, root / "plots" / "gpu_resources.png")
    stale_migration_plot = root / "plots" / "migration.png"
    if stale_migration_plot.exists():
        stale_migration_plot.unlink()
    append_report(root, samples, summary, gpu_summary, instance_gpu_summary,
                  migration_summary, plot_names, measure_sec)
    print(f"Wrote {len(samples)} instance samples and {len(summary)} instance summaries to {root}")
    for _, row in summary[summary["scope"] == "all_instances"].iterrows():
        print(f"{row['mode'].upper()}: mean KV={row['mean_gpu_cache_usage_pct']:.3f}% "
              f"P95 KV={row['p95_gpu_cache_usage_pct']:.3f}% "
              f"mean waiting={row['mean_waiting_requests']:.3f} "
              f"preemption-gauge max={row['max_num_killed_requests_gauge']:.0f}")
    print("Plots: " + ", ".join(plot_names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
