import pytest

torch = pytest.importorskip("torch", reason="PyTorch is an optional Mig-PPO training dependency")

from rl_scheduler.policy import MigPPOActorCritic


def test_joint_hierarchical_logprob_entropy_and_mask():
    # Four instances produce 12 directed pairs. Only the first pair is legal.
    model = MigPPOActorCritic(28, 14, 14, num_instances=4, hidden_size=16)
    instances = torch.zeros((4, 28))
    global_features = torch.zeros(14)
    pairs = torch.zeros((12, 14))
    mask = torch.zeros(12, dtype=torch.bool)
    mask[0] = True
    sampled = model.act(instances, global_features, pairs, mask, deterministic=True)
    assert sampled.action.item() in (0, 1)
    assert torch.isfinite(sampled.log_probability).all()
    assert torch.isfinite(sampled.entropy).all()
    logp, entropy, value, _ = model.evaluate_actions(
        instances.unsqueeze(0), global_features.unsqueeze(0), pairs.unsqueeze(0),
        mask.unsqueeze(0), torch.tensor([0]),
    )
    assert torch.isfinite(logp).all()
    assert torch.isfinite(entropy).all()
    assert value.shape == (1,)
    with pytest.raises(ValueError):
        model.evaluate_actions(instances.unsqueeze(0), global_features.unsqueeze(0),
                               pairs.unsqueeze(0), mask.unsqueeze(0), torch.tensor([2]))

