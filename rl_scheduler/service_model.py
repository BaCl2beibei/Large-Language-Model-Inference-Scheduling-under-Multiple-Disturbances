"""Small, serializable service-time models used by the Mig-PPO simulator.

The model deliberately consumes batch/service features only. Request queue delay
is computed by the simulator's event loop and is not added to these predictions.
"""

from __future__ import annotations

import math
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

