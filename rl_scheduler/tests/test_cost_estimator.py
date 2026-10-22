from rl_scheduler.cost_estimator import MigPPOMigrationCostEstimator
import pytest


def test_global_and_pair_cost_models_fit_and_round_trip():
    events = [
        {"source_instance": "a", "target_instance": "b", "blocks": blocks,
         "cost_ms": 10 + 2 * blocks, "status": "success"}
        for blocks in (1, 2, 3, 4, 5, 6, 7, 8)
    ]
    estimator = MigPPOMigrationCostEstimator(min_pair_samples=4).fit(events)
    prediction = estimator.predict("a", "b", 10)
    assert prediction.source == "pair"
    assert abs(prediction.duration_ms - 30.0) < 1e-6
    restored = MigPPOMigrationCostEstimator.from_state_dict(estimator.state_dict())
    assert abs(restored.predict("a", "b", 10).duration_ms - 30.0) < 1e-6


def test_uncalibrated_model_marks_estimate_uncertainty():
    prediction = MigPPOMigrationCostEstimator().predict("a", "b", 10)
    assert prediction.source == "default_uncalibrated"
    assert prediction.uncertainty_ms > 0


def test_refined_robust_yaml_schema_loads_and_applies_pair_shrinkage():
    state = {"global": {"alpha_ms": 100.0, "beta_ms_per_block": 2.0,
                        "residual_sigma_ms": 10.0, "samples": 20},
             "pair_corrections": {"a|b": {"intercept_correction_ms": 5.0,
                                           "uncertainty_ms": 3.0, "samples": 9}}}
    estimator = MigPPOMigrationCostEstimator.from_state_dict(state)
    prediction = estimator.predict("a", "b", 10)
    assert prediction.duration_ms == 125.0
    assert prediction.source == "pair_shrunk"
    assert prediction.uncertainty_ms == pytest.approx((10.0**2 + 3.0**2) ** 0.5)

