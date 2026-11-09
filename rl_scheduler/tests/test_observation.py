import numpy as np

from rl_scheduler.cost_estimator import MigPPOMigrationCostEstimator
from rl_scheduler.observation import RollingArrivalRate, build_mig_ppo_observation
from rl_scheduler.tests.helpers import fake_infos


def test_observation_shapes_candidate_blocks_and_finite_values():
    infos = fake_infos()
    infos["instance_0"].remaining_steps.remaining_steps = float("inf")
    obs = build_mig_ppo_observation(
        infos, now=1000.0, arrival_rate_rps=1.25, migrations_in_progress=0,
    )
    assert obs.instance_features.shape == (4, 14)
    assert obs.instance_missing.shape == (4, 14)
    assert obs.global_features.shape == (7,)
    assert obs.pair_features.shape == (12, 5)
    assert obs.pair_cost_uncertainty_ms.shape == (12, 1)
    assert obs.pair_cost_source.shape == (12, 3)
    assert np.all(obs.pair_cost_source[:, 2] == 1.0)
    assert obs.candidate_blocks == (8, 9, 10, 11)
    assert obs.candidate_request_ids[0] == "candidate-instance_0"
    assert np.isfinite(obs.instance_features).all()
    assert obs.instance_missing[0, 9] == 1
    assert np.isfinite(obs.pair_features).all()
    assert np.isfinite(obs.pair_cost_uncertainty_ms).all()
    assert obs.action_mask.action_mask.shape == (13,)


def test_rolling_arrival_rate_uses_only_events_in_past_window():
    rate = RollingArrivalRate(window_sec=10.0)
    rate.start(0.0)
    rate.record(2.0)
    rate.record(8.0)
    assert rate.rate(10.0) == 0.2
    assert rate.rate(18.0) == 0.1
    assert rate.rate(30.0) == 0.0


def test_observation_marks_global_cost_model_fallback():
    estimator = MigPPOMigrationCostEstimator(min_pair_samples=8).fit([
        {"blocks": 2, "cost_ms": 14.0}, {"blocks": 6, "cost_ms": 26.0},
    ])
    obs = build_mig_ppo_observation(fake_infos(), now=1000.0, cost_estimator=estimator)
    assert np.all(obs.pair_cost_source[:, 1] == 1.0)
    assert np.all(obs.pair_cost_uncertainty_ms > 0.0)
    assert np.isfinite(obs.pair_cost_uncertainty_ms).all()


def test_stale_instance_state_is_marked_missing_and_candidate_is_invalid():
    infos = fake_infos(now=900.0)
    obs = build_mig_ppo_observation(infos, now=1000.0, arrival_rate_rps=float("nan"))
    assert obs.instance_features[0].tolist() == [0.0] * 14
    assert obs.instance_missing[0].tolist() == [1.0] * 14
    assert obs.candidate_request_ids == (None, None, None, None)
    assert obs.global_missing[1] == 1
    assert not obs.pair_mask.any()

