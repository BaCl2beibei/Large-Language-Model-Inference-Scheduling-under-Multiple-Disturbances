"""Calibratable migration-cost model for Mig-PPO observations."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np


def nonnegative_lstsq(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Solve a small nonnegative least-squares problem without extra deps.

    Calibration design matrices have at most three columns. Enumerating active
    coefficient sets gives the exact NNLS solution at that size. Clipping an
    unconstrained fit coefficient-by-coefficient is not equivalent and can
    produce a biased model when input features are correlated.
    """
    columns = design.shape[1]
    best = np.zeros(columns, dtype=np.float64)
    best_sse = float(np.dot(target, target))
    for mask in range(1, 1 << columns):
        active = [index for index in range(columns) if mask & (1 << index)]
        candidate = np.linalg.lstsq(design[:, active], target, rcond=None)[0]
        if np.any(candidate < -1e-10):
            continue
        candidate = np.maximum(candidate, 0.0)
        residual = target - design[:, active] @ candidate
        sse = float(np.dot(residual, residual))
        if sse < best_sse:
            best_sse = sse
            best[:] = 0.0
            best[active] = candidate
    return best


@dataclass(frozen=True)
class MigrationCostPrediction:
    duration_ms: float
    uncertainty_ms: float
    source: str


class MigPPOMigrationCostEstimator:
    """Fit ``duration_ms = alpha + beta * copied_blocks`` to terminal events.

    Per directed pair coefficients are used only after ``min_pair_samples``
    measured migrations. Otherwise the fitted global model is used. With no
    usable measurements, explicit conservative defaults are returned and the
    source is marked ``default_uncalibrated``.
    """

    def __init__(
        self,
        default_alpha_ms: float = 50.0,
        default_beta_ms_per_block: float = 0.1,
        min_pair_samples: int = 8,
    ) -> None:
        self.default_alpha_ms = max(0.0, float(default_alpha_ms))
        self.default_beta_ms_per_block = max(0.0, float(default_beta_ms_per_block))
        self.min_pair_samples = max(2, int(min_pair_samples))
        self.global_model: Optional[tuple[float, float, float, int]] = None
        self.pair_models: dict[tuple[str, str], tuple[float, float, float, int]] = {}
        self.pair_corrections: dict[tuple[str, str], tuple[float, float, int]] = {}

    @staticmethod
    def _fit(rows: list[tuple[float, float]]) -> Optional[tuple[float, float, float, int]]:
        if len(rows) < 2:
            return None
        blocks = np.asarray([max(0.0, row[0]) for row in rows], dtype=np.float64)
        durations = np.asarray([max(0.0, row[1]) for row in rows], dtype=np.float64)
        design = np.column_stack((np.ones_like(blocks), blocks))
        alpha, beta = nonnegative_lstsq(design, durations)
        alpha = float(alpha)
        beta = float(beta)
        residual = durations - (alpha + beta * blocks)
        residual_rmse = float(np.sqrt(np.mean(np.square(residual)))) if len(rows) else 0.0
        # Include finite-sample model uncertainty so a tiny, perfectly fitted
        # calibration set is not reported as a zero-uncertainty estimate.
        sampling_uncertainty = float(np.mean(durations)) / math.sqrt(len(rows))
        uncertainty = math.hypot(residual_rmse, sampling_uncertainty)
        return alpha, beta, uncertainty, len(rows)

    def fit(self, events: Iterable[Mapping[str, object]]) -> "MigPPOMigrationCostEstimator":
        global_rows: list[tuple[float, float]] = []
        pair_rows: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
        for event in events:
            try:
                blocks = float(event.get("blocks", event.get("kv_blocks", 0)))
                duration = float(event.get("cost_ms", event.get("duration_ms", 0)))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(blocks) or not math.isfinite(duration) or blocks < 0 or duration < 0:
                continue
            status = str(event.get("status", "success")).lower()
            if status in {"aborted", "failed", "aborted_src", "aborted_dst"}:
                # Include completed transfer attempts: they still measure link
                # and setup cost. Rows without a terminal duration are skipped.
                pass
            row = (blocks, duration)
            global_rows.append(row)
            src = event.get("source_instance", event.get("src_instance"))
            dst = event.get("target_instance", event.get("dst_instance"))
            if src is not None and dst is not None:
                pair_rows[(str(src), str(dst))].append(row)

        self.global_model = self._fit(global_rows)
        self.pair_models = {
            pair: model
            for pair, rows in pair_rows.items()
            if len(rows) >= self.min_pair_samples
            for model in [self._fit(rows)]
            if model is not None
        }
        return self

    def fit_csv(self, path: str | Path) -> "MigPPOMigrationCostEstimator":
        with Path(path).open(newline="", encoding="utf-8") as f:
            return self.fit(csv.DictReader(f))

    def predict(self, source: str, target: str, blocks: float) -> MigrationCostPrediction:
        model = self.pair_models.get((str(source), str(target)))
        source_name = "pair"
        if model is None:
            model = self.global_model
            source_name = "global"
        if model is None:
            alpha = self.default_alpha_ms
            beta = self.default_beta_ms_per_block
            uncertainty = max(alpha, beta * max(0.0, float(blocks)))
            source_name = "default_uncalibrated"
        else:
            alpha, beta, uncertainty, _ = model
            correction = self.pair_corrections.get((str(source), str(target)))
            if correction is not None:
                offset, pair_sigma, _samples = correction
                alpha += offset
                uncertainty = math.hypot(uncertainty, pair_sigma)
                source_name = "pair_shrunk"
        duration = alpha + beta * max(0.0, float(blocks))
        if not math.isfinite(duration):
            duration = self.default_alpha_ms
            uncertainty = max(uncertainty, duration)
            source_name = "default_invalid_prediction"
        return MigrationCostPrediction(
            duration_ms=max(0.0, duration),
            uncertainty_ms=max(0.0, float(uncertainty)),
            source=source_name,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "default_alpha_ms": self.default_alpha_ms,
            "default_beta_ms_per_block": self.default_beta_ms_per_block,
            "min_pair_samples": self.min_pair_samples,
            "global_model": self.global_model,
            "pair_models": {"|".join(key): value for key, value in self.pair_models.items()},
            "pair_corrections": {"|".join(key): {"intercept_correction_ms": value[0],
                                                    "uncertainty_ms": value[1], "samples": value[2]}
                                 for key, value in self.pair_corrections.items()},
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "MigPPOMigrationCostEstimator":
        estimator = cls(
            default_alpha_ms=float(state.get("default_alpha_ms", 50.0)),
            default_beta_ms_per_block=float(state.get("default_beta_ms_per_block", 0.1)),
            min_pair_samples=int(state.get("min_pair_samples", 8)),
        )
        global_model = state.get("global_model")
        if global_model is not None:
            estimator.global_model = tuple(float(x) if i < 3 else int(x)
                                           for i, x in enumerate(global_model))  # type: ignore[arg-type]
        refined_global = state.get("global")
        if isinstance(refined_global, Mapping):
            estimator.global_model = (
                float(refined_global.get("alpha_ms", 50.0)),
                float(refined_global.get("beta_ms_per_block", 0.1)),
                float(refined_global.get("residual_sigma_ms", 0.0)),
                int(refined_global.get("samples", 0)),
            )
        pair_models = state.get("pair_models", {})
        if isinstance(pair_models, Mapping):
            for key, value in pair_models.items():
                parts = str(key).split("|", 1)
                if len(parts) == 2:
                    estimator.pair_models[(parts[0], parts[1])] = tuple(
                        float(x) if i < 3 else int(x) for i, x in enumerate(value)  # type: ignore[arg-type]
                    )
        pair_corrections = state.get("pair_corrections", {})
        if isinstance(pair_corrections, Mapping):
            for key, value in pair_corrections.items():
                parts = str(key).split("|", 1)
                if len(parts) == 2 and isinstance(value, Mapping):
                    estimator.pair_corrections[(parts[0], parts[1])] = (
                        float(value.get("intercept_correction_ms", 0.0)),
                        float(value.get("uncertainty_ms", 0.0)),
                        int(value.get("samples", 0)),
                    )
        return estimator

