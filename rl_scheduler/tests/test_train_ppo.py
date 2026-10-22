from rl_scheduler.train_ppo import _resolve_episode_target


def test_explicit_timesteps_override_configured_episode_target():
    assert _resolve_episode_target(episodes=None, total_timesteps=8,
                                   training={"total_episodes": 2000}) is None


def test_explicit_episode_target_takes_precedence():
    assert _resolve_episode_target(episodes=2, total_timesteps=8,
                                   training={"total_episodes": 2000}) == 2


def test_configured_episode_target_is_used_without_cli_limits():
    assert _resolve_episode_target(episodes=None, total_timesteps=None,
                                   training={"total_episodes": 2000}) == 2000
