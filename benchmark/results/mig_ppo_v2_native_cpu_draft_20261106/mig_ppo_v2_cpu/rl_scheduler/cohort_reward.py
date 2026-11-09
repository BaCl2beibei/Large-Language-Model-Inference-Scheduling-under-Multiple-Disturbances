"""Mig-PPO V2 reward and the historical full-cohort objective for old audits."""
from dataclasses import asdict, dataclass
from bisect import bisect_right
import math

from .cohort_statistics import cvar95
from .real_observation import used_fraction


@dataclass(frozen=True)
class MigPPOV2Reward:
    # Required: set from the measured KV/congestion relation, never assume 80%.
    hotspot_threshold: float
    n_ref: float = 64.
    q_ref: float = 64.
    q_max_ref: float = 64.
    failure_ref: float = 64.
    ttft_ref_sec: float = 2.
    token_gap_ref_sec: float = .15
    kappa: float = .5
    alpha: float = .40
    beta: float = .30
    mu: float = .20
    nu: float = .10
    xi: float = 1.
    eta: float = .10
    w_Q: float = .45
    w_H: float = .35
    w_B: float = .20
    gamma_per_sec: float = .99
    lambda_per_sec: float = .97
    invalid_migration_cost: float = .01
    schema = 'mig-ppo-v2'

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('finite nonnegative reward parameter required: ' + name)
        if not 0 <= self.hotspot_threshold < 1:
            raise ValueError('calibrated hotspot_threshold must be in [0, 1)')
        for name in ('n_ref', 'q_ref', 'q_max_ref', 'failure_ref', 'ttft_ref_sec', 'token_gap_ref_sec'):
            if getattr(self, name) <= 0:
                raise ValueError('fixed positive reference required: ' + name)
        if not 0 < self.gamma_per_sec <= 1 or not 0 < self.lambda_per_sec <= 1:
            raise ValueError('time discount bases must be in (0, 1]')

    def state_dict(self):
        return {'schema': self.schema, **asdict(self)}

    @classmethod
    def from_state_dict(cls, state):
        values = dict(state)
        if values.pop('schema', None) != cls.schema:
            raise ValueError('Mig-PPO V2 reward schema required; prepare a new run directory')
        return cls(**values)

    def pressures(self, snapshot):
        states = snapshot['instances']
        if len(states) != 4 or len({s['instance_id'] for s in states}) != 4:
            raise ValueError('four distinct measured instances required for reward')
        q = [s.get('waiting') for s in states]
        u = [used_fraction(s) for s in states]
        if any(v is None or not math.isfinite(v) or v < 0 for v in q+u) or any(v > 1 for v in u):
            raise ValueError('missing or invalid queue/KV reward measurements')
        Q = .5*min(1., math.fsum(q)/4/self.q_ref) + .5*min(1., max(q)/self.q_max_ref)
        H = min(1., max(0., (max(u)-self.hotspot_threshold)/(1-self.hotspot_threshold)))
        mean = math.fsum(u)/4
        B = 2*math.sqrt(math.fsum((v-mean)**2 for v in u)/4)
        return {'Q': Q, 'H': H, 'B': B, 'B_star': H*B}

    def reconstruct(self, summary, decision_times_ns, decisions):
        if (summary.get('status') != 'passed' or not summary.get('cohort_complete')
                or summary.get('audit', {}).get('status') != 'passed'
                or not summary.get('route_audit_passed')):
            raise ValueError('complete audited routed episode required for V2 reward')
        requests = summary['requests']
        n = summary['planned_request_count']
        if n <= 0 or len(requests) != n or len({r['request_id'] for r in requests}) != n:
            raise ValueError('exact complete request identity set required')
        times, decisions = list(decision_times_ns), list(decisions)
        if (not times or len(times) != len(decisions) or any(type(t) is not int for t in times)
                or any(a >= b for a,b in zip(times, times[1:]))):
            raise ValueError('strictly increasing actual decision timestamps required')
        if any(type(r.get('client_terminal_ns')) is not int for r in requests):
            raise ValueError('every request must have an audited terminal, including failures')
        end = max(r['client_terminal_ns'] for r in requests)
        if times[-1] >= end:
            raise ValueError('post-terminal observations cannot become training actions')
        edges = [*times, end]
        area, failures, migration_failures = [0.]*len(times), [0]*len(times), [0]*len(times)

        def window_at(at):
            return min(len(times)-1, bisect_right(edges, at)-1)

        def primitive(age, ref):
            # Integral of min(age/ref, 2), in seconds.
            age = max(0., age)
            return age*age/(2*ref) if age <= 2*ref else 2*age-2*ref

        def add_segment(start, stop, ref, weight):
            lo, hi = max(start, times[0]), min(stop, end)
            if hi <= lo:
                return
            i = window_at(lo)
            while lo < hi:
                right = min(hi, edges[i+1])
                area[i] += weight*(primitive((right-start)/1e9, ref)-primitive((lo-start)/1e9, ref))/self.n_ref
                lo, i = right, i+1

        for r in requests:
            sent, terminal = r.get('sent_ns'), r['client_terminal_ns']
            if type(sent) is not int or terminal <= sent:
                raise ValueError('positive measured client lifetime required')
            outcome = r.get('outcome')
            if outcome not in ('success', 'timeout', 'service_error', 'failed', 'aborted', 'cancelled', 'manual_cancel', 'error'):
                raise ValueError('explicit audited request outcome required')
            if outcome == 'success' and r['actual_output_tokens'] != r['expected_output_tokens']:
                raise ValueError('successful output length differs from frozen trace')
            if outcome != 'success' and terminal >= times[0]:
                failures[window_at(terminal)] += 1
            first = r.get('client_first_ns')
            if first is not None and (type(first) is not int or not sent <= first <= terminal):
                raise ValueError('invalid measured first-token time')
            add_segment(sent, terminal if first is None else first, self.ttft_ref_sec, 1.)
            if 'token_increments' not in r:
                raise ValueError('actual client token receipt timestamps required; aggregate TPOT is insufficient')
            tokens, previous, count = [], sent, 0
            for at, increment in r['token_increments']:
                if type(at) is not int or type(increment) is not int or increment < 0 or not previous <= at <= terminal:
                    raise ValueError('invalid client token receipt timeline')
                previous = at
                if increment > 0:
                    tokens.append(at)
                    count += increment
            if count != r['actual_output_tokens'] or (tokens[0] if tokens else None) != first:
                raise ValueError('token receipt timeline does not reconcile with request')
            for a,b in zip(tokens, [*tokens[1:], terminal]):
                add_segment(a, b, self.token_gap_ref_sec, self.kappa)

        pressures = [self.pressures(d['snapshot']) for d in decisions]
        for at,d in zip(times, decisions):
            if d.get('observation_decision_ns', at) != at or d['snapshot']['received_ns'] > at:
                raise ValueError('reward snapshot differs from the saved decision time')
        # Count each rejected/failed migration exactly once, at its terminal receipt.
        receipts = summary.get('migration_receipts', {})
        seen = set()
        for at,d in zip(times, decisions):
            action = d.get('sampled_action', (d.get('ppo') or {}).get('action', 0))
            if not action:
                continue
            receipt = receipts.get(d.get('decision_id'), d.get('receipt', {}))
            failed = receipt.get('terminal_status') in ('failed', 'rejected') or bool(d.get('rejection'))
            identity = receipt.get('migration_id') or d.get('decision_id') or at
            if failed and identity not in seen:
                seen.add(identity)
                terminal_at = receipt.get('terminal_ns') or at
                if times[0] <= terminal_at:
                    # A migration can settle after the last client terminal;
                    # retain its cost in the final transition rather than lose it.
                    migration_failures[window_at(terminal_at)] += 1
        windows = []
        specs = [('queue', 'Q', self.beta, self.w_Q), ('load', 'H', self.mu, self.w_H),
                 ('balance', 'B_star', self.nu, self.w_B)]
        for i,(lo,hi) in enumerate(zip(edges, edges[1:])):
            dt = (hi-lo)/1e9
            gamma, lam = self.gamma_per_sec**dt, self.lambda_per_sec**dt
            terminal = i == len(times)-1
            p, next_p = pressures[i], pressures[i+1] if not terminal else dict(Q=0., H=0., B_star=0.)
            parts = {'time': -self.alpha*area[i]-self.xi*failures[i]/self.failure_ref}
            for name,key,weight,shaping_weight in specs:
                parts[name] = -dt*weight*p[key]+self.eta*shaping_weight*(p[key]-gamma*next_p[key])
            cost = -self.invalid_migration_cost*migration_failures[i]
            windows.append(dict(start_ns=lo, end_ns=hi, dt_sec=dt, gamma_t=gamma, lambda_t=lam,
                D_integral=area[i], D_mean=area[i]/dt, pressures=p, next_pressures=next_p,
                failure_count=failures[i], normalized_failures=failures[i]/self.failure_ref,
                invalid_migration_count=migration_failures[i], migration_cost=cost,
                components=parts, reward=math.fsum(parts.values())+cost, terminated=terminal, truncated=False))
        return {'schema': self.schema, 'definition': self.state_dict(), 'windows': windows,
                'return': math.fsum(w['reward'] for w in windows),
                'state_integration': 'decision-snapshot zero-order hold',
                'time_integration': 'exact capped ramps between client token receipt timestamps',
                'terminal_potential': 0.}


def time_discounted_gae(windows, values):
    """Real-time GAE for a complete episode; the true terminal has V=0."""
    if not windows or len(windows) != len(values) or not windows[-1]['terminated']:
        raise ValueError('complete aligned episode required for GAE')
    if any(w.get('truncated') or w['terminated'] for w in windows[:-1]) or windows[-1].get('truncated'):
        raise ValueError('storage boundaries are not episode terminals')
    advantage, next_value, advantages = 0., 0., []
    for w,value in reversed(list(zip(windows, values))):
        if not all(math.isfinite(v) for v in (w['reward'], value, w['gamma_t'], w['lambda_t'])):
            raise ValueError('nonfinite GAE input')
        delta = w['reward'] + w['gamma_t']*next_value-value
        advantage = delta + w['gamma_t']*w['lambda_t']*advantage
        advantages.append(advantage)
        next_value = value
    advantages.reverse()
    return advantages, [a+v for a,v in zip(advantages, values)]


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
