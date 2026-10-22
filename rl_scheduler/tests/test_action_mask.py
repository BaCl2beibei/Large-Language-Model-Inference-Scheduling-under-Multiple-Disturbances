from rl_scheduler.action_mask import MigPPOActionMaskConfig, build_mig_ppo_action_mask
from rl_scheduler.tests.helpers import fake_infos


def test_directed_pairs_exclude_self_and_keep_stay():
    mask = build_mig_ppo_action_mask(fake_infos(), now=1000.0,
                                    config=MigPPOActionMaskConfig(safety_margin_blocks=8))
    assert len(mask.pairs) == 12
    assert mask.stay_valid
    assert all(src != dst for src, dst in mask.pairs)
    assert mask.action_mask[0]
    assert mask.pair_mask.any()


def test_filters_stale_unhealthy_busy_incompatible_and_capacity_shortage():
    infos = fake_infos()
    infos["instance_0"].timestamp = 900.0
    infos["instance_1"].unit_status = "stopped"
    infos["instance_2"].is_migrating = True
    infos["instance_3"].num_free_gpu_blocks = 20
    infos["instance_3"].num_total_gpu_blocks = 256
    mask = build_mig_ppo_action_mask(infos, now=1000.0,
                                    config=MigPPOActionMaskConfig(max_state_age_sec=2.0,
                                                                 safety_margin_blocks=16))
    assert not mask.pair_mask.any()


def test_cooldown_and_inflight_force_stay():
    infos = fake_infos()
    cfg = MigPPOActionMaskConfig(safety_margin_blocks=8, migration_cooldown_sec=5)
    cooldown = build_mig_ppo_action_mask(infos, now=1000.0, config=cfg, last_migration_time=999.0)
    inflight = build_mig_ppo_action_mask(infos, now=1000.0, config=cfg, migration_in_progress=True)
    assert cooldown.action_mask.tolist() == [True] + [False] * 12
    assert inflight.action_mask.tolist() == [True] + [False] * 12


def test_non_decode_sr_candidate_is_not_migratable():
    infos = fake_infos()
    infos["instance_0"].migration_candidate_stage = "prefill"
    mask = build_mig_ppo_action_mask(infos, now=1000.0,
                                     config=MigPPOActionMaskConfig(safety_margin_blocks=8))
    assert not mask.pair_mask[:3].any()
    assert mask.action_mask[0]

