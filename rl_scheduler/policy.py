"""Shared-encoder actor-critic with the exact hierarchical Mig-PPO action law."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import torch
    from torch import nn
    from torch.distributions import Bernoulli, Categorical
except ImportError as exc:  # Keep observation/env tools usable without PyTorch.
    torch = None
    nn = object  # type: ignore[assignment,misc]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


if torch is not None:
    @dataclass
    class MigPPOActionOutput:
        action: torch.Tensor
        log_probability: torch.Tensor
        entropy: torch.Tensor
        value: torch.Tensor


    class MigPPOActorCritic(nn.Module):
        """Two-head hierarchical policy: Stay gate plus conditional pair policy."""

        def __init__(self, instance_feature_dim: int, global_feature_dim: int,
                     pair_feature_dim: int, num_instances: int, hidden_size: int = 64) -> None:
            super().__init__()
            self.num_instances = int(num_instances)
            if self.num_instances < 1:
                raise ValueError("num_instances must be positive")
            src_indices, dst_indices = [], []
            for src in range(self.num_instances):
                for dst in range(self.num_instances):
                    if src != dst:
                        src_indices.append(src)
                        dst_indices.append(dst)
            self.register_buffer("pair_src_indices", torch.tensor(src_indices, dtype=torch.long))
            self.register_buffer("pair_dst_indices", torch.tensor(dst_indices, dtype=torch.long))
            self.instance_encoder = nn.Sequential(
                nn.Linear(instance_feature_dim, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            )
            self.global_encoder = nn.Sequential(
                nn.Linear(global_feature_dim, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            )
            pooled_size = hidden_size * 3
            self.gate_head = nn.Sequential(
                nn.Linear(pooled_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )
            self.pair_head = nn.Sequential(
                nn.Linear(hidden_size * 3 + pair_feature_dim, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1),
            )
            self.critic_head = nn.Sequential(
                nn.Linear(pooled_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1),
            )

        def forward(self, instance_features, global_features, pair_features, pair_mask):
            if instance_features.dim() == 2:
                instance_features = instance_features.unsqueeze(0)
                global_features = global_features.unsqueeze(0)
                pair_features = pair_features.unsqueeze(0)
                pair_mask = pair_mask.unsqueeze(0)
            if instance_features.shape[1] != self.num_instances:
                raise ValueError(f"expected {self.num_instances} instances, got {instance_features.shape[1]}")
            if pair_features.shape[1] != len(self.pair_src_indices):
                raise ValueError("pair feature count does not match directed instance pairs")
            instance_embeddings = self.instance_encoder(instance_features)
            mean_pool = instance_embeddings.mean(dim=1)
            max_pool = instance_embeddings.max(dim=1).values
            global_embedding = self.global_encoder(global_features)
            pooled = torch.cat((mean_pool, max_pool, global_embedding), dim=-1)
            gate_logits = self.gate_head(pooled).squeeze(-1)
            values = self.critic_head(pooled).squeeze(-1)
            src_embeddings = instance_embeddings.index_select(1, self.pair_src_indices)
            dst_embeddings = instance_embeddings.index_select(1, self.pair_dst_indices)
            context = global_embedding.unsqueeze(1).expand(-1, pair_features.shape[1], -1)
            pair_input = torch.cat((src_embeddings, dst_embeddings, context, pair_features), dim=-1)
            pair_logits = self.pair_head(pair_input).squeeze(-1)
            pair_logits = pair_logits.masked_fill(~pair_mask.bool(), -1e9)
            return gate_logits, pair_logits, values

        @staticmethod
        def _joint_distribution(gate_logits, pair_logits, pair_mask):
            has_pair = pair_mask.bool().any(dim=-1)
            gate_probability = torch.sigmoid(gate_logits)
            gate_probability = torch.where(has_pair, gate_probability, torch.zeros_like(gate_probability))
            pair_log_probabilities = torch.log_softmax(pair_logits, dim=-1)
            pair_probabilities = torch.softmax(pair_logits, dim=-1)
            eps = torch.finfo(gate_probability.dtype).eps
            p = gate_probability.clamp(eps, 1.0 - eps)
            stay_log_probability = torch.log1p(-p)
            migration_log_probability = torch.log(p).unsqueeze(-1) + pair_log_probabilities
            joint_log_probabilities = torch.cat((stay_log_probability.unsqueeze(-1), migration_log_probability), dim=-1)
            joint_entropy = -(1.0 - gate_probability) * torch.log1p(-p) \
                - gate_probability * torch.log(p) \
                - gate_probability * (pair_probabilities * pair_log_probabilities).sum(dim=-1)
            joint_entropy = torch.where(has_pair, joint_entropy, torch.zeros_like(joint_entropy))
            return gate_probability, pair_probabilities, joint_log_probabilities, joint_entropy

        def act(self, instance_features, global_features, pair_features, pair_mask,
                deterministic: bool = False) -> MigPPOActionOutput:
            gate_logits, pair_logits, values = self.forward(
                instance_features, global_features, pair_features, pair_mask)
            batched_mask = pair_mask.unsqueeze(0) if pair_mask.dim() == 1 else pair_mask
            gate_p, pair_p, joint_logp, entropy = self._joint_distribution(gate_logits, pair_logits, batched_mask)
            if deterministic:
                actions = joint_logp.argmax(dim=-1)
            else:
                migrate = torch.bernoulli(gate_p).to(torch.long)
                pair_dist = Categorical(probs=pair_p)
                pair_choice = pair_dist.sample() + 1
                actions = torch.where(migrate.bool(), pair_choice, torch.zeros_like(pair_choice))
            log_probability = joint_logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            return MigPPOActionOutput(actions, log_probability, entropy, values)

        def evaluate_actions(self, instance_features, global_features, pair_features,
                             pair_mask, actions):
            gate_logits, pair_logits, values = self.forward(
                instance_features, global_features, pair_features, pair_mask)
            batched_mask = pair_mask.unsqueeze(0) if pair_mask.dim() == 1 else pair_mask
            gate_p, _, joint_logp, entropy = self._joint_distribution(gate_logits, pair_logits, batched_mask)
            actions = actions.long().view(-1)
            invalid_migration = (actions > 0) & ~batched_mask.gather(1, (actions - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
            if torch.any(invalid_migration):
                raise ValueError("PPO rollout contains a masked migration action")
            selected = joint_logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            return selected, entropy, values, gate_p
else:
    class MigPPOActorCritic:  # pragma: no cover - depends on an optional package
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("Mig-PPO training requires PyTorch; install the 'rl' optional dependencies") from _TORCH_IMPORT_ERROR

