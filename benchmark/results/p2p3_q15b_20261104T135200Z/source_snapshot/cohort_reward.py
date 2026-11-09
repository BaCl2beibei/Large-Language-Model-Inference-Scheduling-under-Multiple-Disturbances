"""Exact undiscounted full-cohort latency reward; no censored-window scorer."""
from dataclasses import dataclass
import math

from .full_completion import cvar95


@dataclass(frozen=True)
class CohortReward:
    T_ref_sec: float
    beta: float = 1.0
    gamma: float = 1.0

    def __post_init__(self):
        if not math.isfinite(self.T_ref_sec) or self.T_ref_sec <= 0:
            raise ValueError('a measured frozen positive T_ref is required')
        if not math.isfinite(self.beta) or self.beta < 0 or self.gamma != 1.0:
            raise ValueError('finite nonnegative beta and gamma=1 required')

    def reconstruct(self, summary, decision_times_ns):
        """Integrate actual sent->terminal intervals, then add the exact tail once.

        All rewards are finalized only after the whole cohort passes its audit.
        Rollout storage boundaries and transport failures are never terminals.
        """
        if (summary.get('status') != 'passed' or not summary.get('cohort_complete')
                or summary.get('audit', {}).get('status') != 'passed'
                or not summary.get('route_audit_passed')):
            raise ValueError('complete, audited, routed cohort required before PPO update')
        requests = summary['requests']
        n = summary['planned_request_count']
        if n <= 0 or len(requests) != n or len({r['request_id'] for r in requests}) != n:
            raise ValueError('exact complete request identity set required')
        times = list(decision_times_ns)
        if not times or any(type(t) is not int for t in times) or any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError('strictly increasing actual decision timestamps required')
        latencies = []
        for row in requests:
            sent, terminal = row['sent_ns'], row['client_terminal_ns']
            if type(sent) is not int or type(terminal) is not int or terminal <= sent:
                raise ValueError('normal positive client request lifetime required')
            if row['actual_output_tokens'] != row['expected_output_tokens']:
                raise ValueError('actual output lengths must match the frozen trace')
            latencies.append((terminal - sent) / 1e9)
        end = max(r['client_terminal_ns'] for r in requests)
        if times[-1] >= end:
            raise ValueError('post-terminal observations cannot become training actions')
        # If collection started slightly after the first HTTP call, its already
        # incurred (action-independent) area belongs to the first reward once.
        edges = [min(times[0], min(r['sent_ns'] for r in requests)), *times[1:], end]
        tail = cvar95(latencies)
        windows = []
        for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
            area = math.fsum(max(0, min(hi, r['client_terminal_ns']) - max(lo, r['sent_ns'])) / 1e9
                             for r in requests)
            dense = -area / (n * self.T_ref_sec)
            terminal = -self.beta * tail / self.T_ref_sec if i == len(times) - 1 else 0.0
            windows.append({'start_ns': lo, 'end_ns': hi, 'dt_sec': (hi-lo)/1e9,
                            'outstanding_request_seconds': area, 'dense_reward': dense,
                            'tail_reward': terminal, 'reward': dense + terminal,
                            'terminated': i == len(times)-1, 'truncated': False})
        objective = (math.fsum(latencies)/n + self.beta*tail) / self.T_ref_sec
        total = math.fsum(w['reward'] for w in windows)
        if not math.isclose(total, -objective, rel_tol=1e-11, abs_tol=1e-11):
            raise ValueError('reward does not equal the frozen full-cohort objective')
        return {'windows': windows, 'return': total, 'objective': objective,
                'mean_e2e_sec': math.fsum(latencies)/n, 'cvar95_e2e_sec': tail,
                'T_ref_sec': self.T_ref_sec, 'beta': self.beta, 'gamma': self.gamma}


def complete_returns(rewards):
    """gamma=lambda=1 GAE targets, with V(terminal)=0 and no buffer bootstrap."""
    future, result = 0.0, []
    for reward in reversed(rewards):
        if not math.isfinite(reward):
            raise ValueError('nonfinite reward')
        future += reward
        result.append(future)
    return list(reversed(result))
