"""Small, serializable service-time models used by the Mig-PPO simulator.

The model deliberately consumes batch/service features only. Request queue delay
is computed by the simulator's event loop and is not added to these predictions.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from typing import Mapping


def _feature_value(name: str, features: Mapping[str, float]) -> float:
    try:
        value = float(features[name])
    except (KeyError, TypeError, ValueError, OverflowError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def evaluate_term(term: Mapping[str, object], features: Mapping[str, float]) -> float:
    """Evaluate one predeclared feature term from a calibration YAML."""
    op = str(term.get("op", "linear"))
    if op == "linear":
        return _feature_value(str(term["feature"]), features)
    if op == "log1p":
        return math.log1p(max(0.0, _feature_value(str(term["feature"]), features)))
    if op == "square_log1p":
        value = math.log1p(max(0.0, _feature_value(str(term["feature"]), features)))
        return value * value
    if op == "interaction":
        left = _feature_value(str(term["left"]), features)
        right = _feature_value(str(term["right"]), features)
        return left * right
    if op == "hinge_log1p":
        value = math.log1p(max(0.0, _feature_value(str(term["feature"]), features)))
        return max(0.0, value - float(term["knot"]))
    raise ValueError(f"unsupported service-model feature operation: {op}")


def predict_service_model(model: Mapping[str, object], features: Mapping[str, float]) -> float:
    """Evaluate a centered/scaled linear model with optional nonlinear terms."""
    terms = model.get("terms", [])
    centers = model.get("centers", [])
    scales = model.get("scales", [])
    coefficients = model.get("coefficients", [])
    if not (len(terms) == len(centers) == len(scales) == len(coefficients)):
        raise ValueError("service model terms/centers/scales/coefficients have different lengths")
    prediction = float(model.get("intercept_ms", 0.0))
    for term, center, scale, coefficient in zip(terms, centers, scales, coefficients):
        scale = float(scale)
        if scale <= 0 or not math.isfinite(scale):
            scale = 1.0
        raw = evaluate_term(term, features)
        prediction += float(coefficient) * ((raw - float(center)) / scale)
    floor = float(model.get("min_prediction_ms", 0.0))
    if not math.isfinite(prediction):
        return max(0.0, floor)
    return max(floor, prediction)


def sample_qps_quantile_model(model: Mapping[str, object], qps: float, rng: object) -> float:
    """Sample one value from an empirical quantile profile interpolated by QPS."""
    profiles = model.get("profiles", {})
    if not isinstance(profiles, Mapping) or not profiles:
        return 0.0
    levels = model.get("quantile_levels", [])
    try:
        levels = [float(value) for value in levels]
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not levels:
        return 0.0
    usable: list[tuple[float, list[float]]] = []
    for key, profile in profiles.items():
        if not isinstance(profile, Mapping):
            continue
        try:
            x = float(key)
            values = [float(value) for value in profile.get("interval_quantiles_ms", [])]
        except (TypeError, ValueError, OverflowError):
            continue
        if len(values) == len(levels) and all(math.isfinite(value) and value > 0 for value in values):
            usable.append((x, values))
    usable.sort(key=lambda row: row[0])
    if not usable:
        return 0.0
    if qps <= usable[0][0]:
        quantiles = usable[0][1]
    elif qps >= usable[-1][0]:
        quantiles = usable[-1][1]
    else:
        quantiles = usable[-1][1]
        for (left_qps, left), (right_qps, right) in zip(usable, usable[1:]):
            if left_qps <= qps <= right_qps:
                weight = (float(qps) - left_qps) / (right_qps - left_qps)
                quantiles = [a + (b - a) * weight for a, b in zip(left, right)]
                break
    uniform = float(rng.random())
    index = bisect_right(levels, uniform)
    if index <= 0:
        return max(0.0, quantiles[0])
    if index >= len(levels):
        return max(0.0, quantiles[-1])
    left_level, right_level = levels[index - 1], levels[index]
    if right_level <= left_level:
        return max(0.0, quantiles[index])
    fraction = (uniform - left_level) / (right_level - left_level)
    return max(0.0, quantiles[index - 1] + (quantiles[index] - quantiles[index - 1]) * fraction)

