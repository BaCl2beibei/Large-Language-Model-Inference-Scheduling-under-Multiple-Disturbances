import pytest

from rl_scheduler.service_model import predict_service_model


def test_centered_interaction_model_predicts_and_floors_latency():
    model = {"terms": [{"op": "linear", "feature": "batch_size"},
                       {"op": "interaction", "left": "batch_size", "right": "avg_context_tokens"}],
             "centers": [2.0, 80.0], "scales": [1.0, 10.0],
             "coefficients": [3.0, -4.0], "intercept_ms": 15.0, "min_prediction_ms": 1.0}
    assert predict_service_model(model, {"batch_size": 3, "avg_context_tokens": 30}) == pytest.approx(14.0)
    assert predict_service_model(model, {"batch_size": 1, "avg_context_tokens": 10}) == pytest.approx(40.0)


def test_service_model_rejects_inconsistent_serialized_coefficients():
    with pytest.raises(ValueError, match="different lengths"):
        predict_service_model({"terms": [{"op": "linear", "feature": "batch_size"}],
                               "centers": [], "scales": [], "coefficients": []}, {"batch_size": 1})
