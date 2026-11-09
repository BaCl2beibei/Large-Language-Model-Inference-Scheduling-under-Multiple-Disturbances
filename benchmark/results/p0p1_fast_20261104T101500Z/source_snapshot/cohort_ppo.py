"""CPU policy and clipped PPO for causal, audited complete-cohort rollouts."""
import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .real_bridge import PAIRS, digest
from .real_observation import INSTANCE_FEATURES, PAIR_FEATURES, SCALES
from .cohort_reward import complete_returns
from .real_metrics import Event
from .real_opportunity_runner import reconcile_migrations

SAFE_GLOBALS = ('outstanding', 'client_tokens_per_sec_5s', 'busy',
                'cooldown_remaining_sec', 'healthy_instances')
OBS_SCHEMA = 'causal-full-cohort-v1'


def normalizer_definition(T_ref_sec):
    return {'schema': OBS_SCHEMA, 'instance_features': list(INSTANCE_FEATURES),
            'pair_features': list(PAIR_FEATURES), 'global_features': [*SAFE_GLOBALS, 'elapsed_sec'],
            'instance_scales': [SCALES[n] for n in INSTANCE_FEATURES],
            'pair_scales': [SCALES[n] for n in PAIR_FEATURES],
            'global_scales': [*[SCALES[n] for n in SAFE_GLOBALS], T_ref_sec],
            'missing_bits_appended': True, 'fitted_during_training': False,
            'excluded': ['arrival_phase', 'drain_phase', 'remaining_arrival_sec', 'remaining_total_sec',
                         'scenario_id', 'winner_label', 'future_arrivals', 'output_token_budget',
                         'future_request_length', 'future_hotspot', 'tail_identity']}


def causal_observation(raw, episode_start_ns, definition):
    if definition['schema'] != OBS_SCHEMA:
        raise ValueError('unknown normalization schema')
    def group(name):
        names = raw[name+'_feature_names']
        expected = definition[name+'_features']
        if name != 'global' and names != expected:
            raise ValueError('feature order changed')
        def row(values, missing):
            if len(values) != len(expected) or len(missing) != len(expected):
                raise ValueError('feature dimension mismatch')
            if any(not math.isfinite(float(v)) for v in values):
                raise ValueError('nonfinite observation')
            return [0. if absent else float(v)/s for v, s, absent in
                    zip(values, definition[name+'_scales'], missing)] + [float(x) for x in missing]
        if name == 'global':
            indices = [names.index(n) for n in SAFE_GLOBALS]
            values = [raw['global_features'][i] for i in indices]
            missing = [raw['global_missing'][i] for i in indices]
            values.append(max(0., (raw['decision_ns']-episode_start_ns)/1e9))
            missing.append(False)
            return row(values, missing)
        return [row(v, m) for v, m in zip(raw[name+'_features'], raw[name+'_missing'])]
    mask = list(raw['action_mask'])
    if len(mask) != 13 or mask[0] is not True:
        raise ValueError('Stay plus 12 directed SR candidate/destination actions required')
    return {'schema': OBS_SCHEMA, 'normalizer_hash': digest(definition),
            'instance_features': group('instance'), 'global_features': group('global'),
            'pair_features': group('pair'), 'action_mask': mask}


def tensors(observations):
    return tuple(torch.tensor([o[key] for o in observations], dtype=dtype) for key, dtype in
                 [('instance_features', torch.float32), ('global_features', torch.float32),
                  ('pair_features', torch.float32), ('action_mask', torch.bool)])


class CohortActorCritic(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.instance = nn.Sequential(nn.Linear(24, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.global_net = nn.Sequential(nn.Linear(12, hidden_size), nn.Tanh())
        self.stay = nn.Linear(hidden_size*3, 1)
        self.pair = nn.Sequential(nn.Linear(hidden_size*3+6, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))
        self.critic = nn.Sequential(nn.Linear(hidden_size*3, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))
        self.register_buffer('src', torch.tensor([p[0] for p in PAIRS]))
        self.register_buffer('dst', torch.tensor([p[1] for p in PAIRS]))
        # Most early time steps should preserve Stay while still exploring legal moves.
        nn.init.constant_(self.stay.bias, math.log(12.))

    def distribution(self, instances, globals_, pairs, mask):
        if mask.shape[-1] != 13 or not mask[:, 0].all():
            raise ValueError('Stay must always be legal')
        encoded = self.instance(instances)
        global_emb = self.global_net(globals_)
        pooled = torch.cat([encoded.mean(1), encoded.max(1).values, global_emb], -1)
        pair_input = torch.cat([encoded[:, self.src], encoded[:, self.dst],
                                global_emb[:, None].expand(-1, 12, -1), pairs], -1)
        logits = torch.cat([self.stay(pooled), self.pair(pair_input).squeeze(-1)], -1)
        logits = logits.masked_fill(~mask, -torch.inf)
        return Categorical(logits=logits), self.critic(pooled).squeeze(-1)

    def evaluate(self, inputs, actions):
        if actions.dtype != torch.long or (actions < 0).any() or (actions >= 13).any():
            raise ValueError('invalid rollout action')
        if not inputs[-1].gather(1, actions[:, None]).all():
            raise ValueError('rollout action was masked at sampling time')
        dist, value = self.distribution(*inputs)
        return dist.log_prob(actions), dist.entropy(), value


class FrozenController:
    def __init__(self, policy, normalizer, policy_version, deterministic=False):
        self.policy, self.normalizer, self.policy_version = policy, normalizer, policy_version
        self.deterministic = deterministic
        self.policy.eval()

    def __call__(self, raw, episode_start_ns):
        observation = causal_observation(raw, episode_start_ns, self.normalizer)
        with torch.no_grad():
            dist, value = self.policy.distribution(*tensors([observation]))
            action = dist.probs.argmax(-1) if self.deterministic else dist.sample()
            logp = dist.log_prob(action)
        return {'action': int(action.item()), 'old_logp': float(logp.item()), 'old_value': float(value.item()),
                'decision_ns': raw['decision_ns'], 'observation': observation,
                'policy_version': self.policy_version}


def audit_execution(summary, decisions, event_rows, intent_rows):
    """Bind intended routing to first engine admission and actual coordinator work."""
    if summary['status'] != 'passed' or not summary['cohort_complete']:
        raise ValueError('cannot accept incomplete environment evidence')
    expected = {d['decision_id']: d['ppo'] for d in decisions if 'ppo' in d}
    intents = {d['decision_id']: d['ppo'] for d in intent_rows}
    if len(intents) != len(intent_rows) or intents != expected:
        raise ValueError('durable pre-RPC behavior intent does not match final decisions')
    events = [Event(**row) for row in event_rows]
    admissions = {}
    for event in sorted(events, key=lambda e: e.occurred_ns):
        if event.request_id and event.data.get('event') == 'request_engine_admitted':
            admissions.setdefault(event.request_id, event.data['instance_id'])
    if admissions != summary['expected_initial_routes']:
        raise ValueError('initial engine admission differs from frozen routes')
    migrated = reconcile_migrations([dict(d, monotonic_time=d['monotonic_ns']) for d in decisions],
                                    summary['migration_receipts'], events)
    for receipt in summary['migration_receipts'].values():
        if receipt.get('terminal_status') == 'succeeded':
            if (receipt.get('actual_request_id') != receipt.get('expected_candidate_id')
                    or not receipt.get('copied_blocks', 0) > 0):
                raise ValueError('migration success lacks the exact request ID and copied KV blocks')
    return {'status': 'passed', 'engine_initial_routes': admissions,
            'durable_behavior_intents': len(intents),
            'reconciled_migrations': migrated,
            'coordinator_attempts': sum(m['attempt_ns'] is not None for m in migrated)}


def build_rollout(summary, decisions, reward_definition):
    if any(d.get('ppo', {}).get('measurement_only') for d in decisions):
        raise ValueError('measurement-only decisions cannot become PPO rollout')
    if summary.get('status') != 'passed' or not summary.get('cohort_complete'):
        raise ValueError('incomplete episode cannot become a PPO rollout')
    end = max(r['client_terminal_ns'] for r in summary['requests'])
    records = [d['ppo'] for d in decisions if 'ppo' in d and d['ppo']['decision_ns'] < end]
    if not records or len({r['policy_version'] for r in records}) != 1:
        raise ValueError('one frozen behavior policy version per complete episode required')
    for row in records:
        if not math.isfinite(row['old_logp']) or not math.isfinite(row['old_value']):
            raise ValueError('nonfinite behavior policy output')
        if not row['observation']['action_mask'][row['action']]:
            raise ValueError('invalid saved behavior action')
    rewards = reward_definition.reconstruct(summary, [r['decision_ns'] for r in records])
    targets = complete_returns([w['reward'] for w in rewards['windows']])
    return {'schema': 'complete-cohort-ppo-rollout-v1', 'episode_id': summary['episode_id'],
            'policy_version': records[0]['policy_version'], 'complete': True,
            'records': [dict(r, reward=w['reward'], return_target=t,
                             advantage=t-r['old_value'], terminated=w['terminated'], truncated=False)
                        for r, w, t in zip(records, rewards['windows'], targets)],
            'reward_audit': rewards}


def ppo_update(policy, optimizer, rollouts, settings, *, seed):
    if not rollouts or any(not r.get('complete') for r in rollouts):
        raise ValueError('PPO updates require complete audited episodes')
    if len({r['policy_version'] for r in rollouts}) != 1:
        raise ValueError('mixed behavior policy versions')
    records = [r for ep in rollouts for r in ep['records']]
    inputs = tensors([r['observation'] for r in records])
    actions = torch.tensor([r['action'] for r in records], dtype=torch.long)
    old_logp = torch.tensor([r['old_logp'] for r in records])
    returns = torch.tensor([r['return_target'] for r in records])
    advantages = torch.tensor([r['advantage'] for r in records])
    policy.eval()
    with torch.no_grad():
        current_logp, _, _ = policy.evaluate(inputs, actions)
    if not torch.allclose(current_logp, old_logp, atol=2e-6, rtol=1e-6):
        raise ValueError('behavior log-probability cannot be reproduced before PPO update')
    advantages = (advantages-advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    rng = np.random.default_rng(seed)
    updates, epochs, stop = [], 0, False
    policy.train()
    for epoch in range(settings['update_epochs']):
        permutation = rng.permutation(len(records))
        for start in range(0, len(records), settings['minibatch_size']):
            idx = torch.tensor(permutation[start:start+settings['minibatch_size']], dtype=torch.long)
            logp, entropy, values = policy.evaluate(tuple(x[idx] for x in inputs), actions[idx])
            logratio = logp-old_logp[idx]
            ratio = logratio.exp()
            kl = ((ratio-1)-logratio).mean()
            if not torch.isfinite(kl):
                raise ValueError('nonfinite PPO KL')
            if kl.item() > settings['target_kl']:
                stop = True
                break
            unclipped = ratio * advantages[idx]
            clipped = ratio.clamp(1-settings['clip_epsilon'], 1+settings['clip_epsilon']) * advantages[idx]
            actor_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = .5*(values-returns[idx]).square().mean()
            loss = actor_loss + settings['value_coefficient']*value_loss - settings['entropy_coefficient']*entropy.mean()
            if not torch.isfinite(loss):
                raise ValueError('nonfinite PPO loss')
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), settings['max_grad_norm'], error_if_nonfinite=True)
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in policy.parameters()):
                raise ValueError('PPO produced nonfinite policy parameters; keep the previous checkpoint')
            updates.append({'loss': loss.item(), 'actor_loss': actor_loss.item(), 'value_loss': value_loss.item(),
                            'entropy': entropy.mean().item(), 'approx_kl': kl.item(), 'grad_norm': norm.item(),
                            'clip_fraction': ((ratio-1).abs()>settings['clip_epsilon']).float().mean().item()})
        epochs = epoch+1
        if stop:
            break
    if not updates:
        raise ValueError('PPO performed no optimizer steps')
    return {'optimizer_steps': len(updates), 'epochs': epochs, 'target_kl_stopped': stop,
            'transition_count': len(records), 'complete_episode_count': len(rollouts),
            'metrics': updates, 'behavior_logp_reproduced': True}
