from rl_scheduler.reward import MigPPORewardCalculator, MigPPORewardMetrics


def test_reward_uses_actual_costs_and_returns_weighted_components():
    metrics = MigPPORewardMetrics(
        window_sec=1.0, instance_count=4, completed_requests=2,
        completed_output_tokens=100, accumulated_waiting_request_seconds=4.0,
        observed_ttft_ms=[100.0, 3000.0], observed_tpot_ms=[50.0, 200.0],
        ttft_slo_violations=1, tpot_slo_violations=1,
        kv_usage_samples=[0.8], migration_cost_ms=250.0,
        migration_operations=1, migration_succeeded=1,
    )
    breakdown = MigPPORewardCalculator().compute(metrics)
    assert breakdown.reward == sum(breakdown.weighted.values())
    assert breakdown.raw["actual_migration_cost_ms"] == 250.0
    assert breakdown.migration_succeeded == 1
    assert breakdown.normalized["migration_cost"] == 0.25
    assert breakdown.weighted["migration_count"] < 0


def test_unfinished_and_timed_out_requests_are_visible_and_penalized():
    metrics = MigPPORewardMetrics(
        window_sec=1.0, instance_count=4, timeout_requests=1, unfinished_requests=2,
    )
    breakdown = MigPPORewardCalculator().compute(metrics)
    assert breakdown.raw["timeout_requests"] == 1
    assert breakdown.raw["unfinished_requests"] == 2
    assert breakdown.normalized["slo_violations"] > 0

