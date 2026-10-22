import csv

from rl_scheduler.calibrate import fit_migration_profile, fit_service_profiles


def test_fit_llumnix_service_and_migration_logs(tmp_path):
    instance_csv = tmp_path / "instance.csv"
    with instance_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["profiling_data"])
        writer.writeheader()
        for row in [
            "('prefill', 1, 100, 12.0)", "('prefill', 1, 200, 22.0)",
            "('decode', 1, 100, 5.0)", "('decode', 2, 100, 8.0)",
            "('decode', 1, 200, 7.0)",
        ]:
            writer.writerow({"profiling_data": row})
    service = fit_service_profiles(instance_csv)
    assert abs(service["prefill_ms_per_token"] - 0.1) < 1e-6
    assert abs(service["prefill_intercept_ms"] - 2.0) < 1e-6
    assert abs(service["decode_ms_per_request"] - 3.0) < 1e-6
    assert abs(service["decode_ms_per_context_token"] - 0.02) < 1e-6

    migration_csv = tmp_path / "migration.csv"
    with migration_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_instance", "target_instance", "blocks", "cost_ms", "status"])
        writer.writeheader()
        for blocks in range(1, 9):
            writer.writerow({"source_instance": "a", "target_instance": "b", "blocks": blocks,
                             "cost_ms": 20 + 1.5 * blocks, "status": "success"})
    model = fit_migration_profile(migration_csv)
    assert model["global_model"] is not None
    assert "a|b" in model["pair_models"]

