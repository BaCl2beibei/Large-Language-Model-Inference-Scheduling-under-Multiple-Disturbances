"""CPU-only protocol/statistics for the real migration opportunity scan.

Missing samples remain missing. All utilization summaries use bounded forward
holds and population dispersion; no simulation or historical results are used.
"""
from __future__ import annotations

import math
import random
import statistics
from collections import Counter

from .real_bridge import PAIRS, eligible_pair
from .real_opportunity import balanced_action
from .real_observation import GLOBAL_FEATURES


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def weighted_summary(points, start, end, max_gap=2.5):
    if end <= start or max_gap <= 0:
        raise ValueError('positive measurement window and gap required')
    points = sorted(points)
    if len({t for t, _ in points}) != len(points):
        raise ValueError('duplicate sample timestamp')
    values = []
    for i, (at, value) in enumerate(points):
        until = min(end, at + max_gap, points[i+1][0] if i+1 < len(points) else end)
        weight = until - max(start, at)
        if finite(value) and weight > 0:
            values.append((value, weight))
    covered = sum(w for _, w in values)
    if not covered:
        return dict(mean=None, p95=None, min=None, max=None, coverage=0.)
    ordered = sorted(values)
    cumulative = 0.
    p95 = None
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= .95 * covered:
            p95 = value
            break
    return dict(mean=sum(v*w for v, w in values)/covered, p95=p95,
                min=min(v for v, _ in values), max=max(v for v, _ in values),
                coverage=covered/(end-start))


def improvement(stay, balanced, lower=False):
    if not finite(stay) or not finite(balanced):
        return dict(percent=None, absolute=None)
    delta = stay-balanced if lower else balanced-stay
    return dict(percent=100*delta/stay if stay else None, absolute=delta)


def remaining_steps(state):
    values = [state.get(k) for k in ('free_blocks', 'waiting_blocks', 'running', 'waiting')]
    if any(not finite(v) or v < 0 for v in values):
        return None
    free, blocked, running, waiting = values
    return (free-blocked)/(running+waiting) if running+waiting else None


def select_thresholds(values):
    values = sorted(v for v in values if finite(v))
    if len(set(values)) < 3:
        raise ValueError('insufficient distinct finite measured F values; refine Stay scan')
    # Empirical nearest-rank quantiles; idle F=+infinity is excluded, never zeroed.
    thresholds = [values[math.ceil(q*len(values))-1] for q in (.25, .5, .75)]
    if len(set(thresholds)) != 3:
        raise ValueError('degenerate F quantiles; refine Stay scan instead of inventing thresholds')
    return dict(zip(('Balanced-L', 'Balanced-M', 'Balanced-H'), thresholds))


def coarse_matrix():
    rows = []
    # Workload pressure first; scheduler-limit controls are a later local refinement.
    for inp, out in ((512, 512), (512, 1024), (1024, 1024)):
        for qps in (3, 6, 9):
            rows.append(dict(scenario=f'i{inp}_o{out}_q{qps}_s192_b4096',
                input_tokens=inp, output_tokens=out, qps=qps,
                max_num_seqs=192, max_num_batched_tokens=4096,
                gpu_memory_utilization=.65, max_model_len=2112))
    return rows


def refinement(candidate):
    return [{**candidate, 'scenario': candidate['scenario']+f'_control_s{s}_b{b}',
             'max_num_seqs': s, 'max_num_batched_tokens': b, 'control_of':candidate['scenario']}
            for s, b in ((128, 4096), (256, 4096), (192, 2112))]


def trace_for(case, seed, tokenizer, arrival_sec=120.):
    from .full_completion import exact_prompt
    if case['input_tokens']+case['output_tokens'] > case['max_model_len']:
        raise ValueError('input + output exceed model context')
    if case['qps'] <= 0 or arrival_sec <= 0:
        raise ValueError('positive Poisson rate and arrival window required')
    prompt, count = exact_prompt(tokenizer, case['input_tokens'], seed)
    rng = random.Random(seed)
    at = rng.expovariate(case['qps'])
    rows = []
    while at < arrival_sec:
        index = len(rows)
        rows.append(dict(request_id=f'r{index:06d}', scheduled_arrival_sec=at,
            prompt=prompt, input_tokens=count, output_tokens=case['output_tokens'],
            request_type=f'in{count}_out{case["output_tokens"]}', route_seed=seed*1000000+index))
        at += rng.expovariate(case['qps'])
    return rows


def paired_schedule(cases, seeds, thresholds):
    if len(set(seeds)) < 3:
        raise ValueError('at least three independent seeds required')
    rows = []
    names = ['Stay', 'Balanced-L', 'Balanced-M', 'Balanced-H']
    for case in cases:
        for i, seed in enumerate(seeds):
            order = names[i % 4:] + names[:i % 4]
            if i % 2:
                order = list(reversed(order))
            for policy in order:
                rows.append(dict(case=case, seed=seed, policy=policy,
                    threshold=None if policy == 'Stay' else thresholds[case['scenario']][policy]))
    return rows


def opportunity(snapshot, raw, threshold, now_ns):
    states = {s['instance_id']: s for s in snapshot['instances']}
    loads = {k: remaining_steps(v) for k, v in states.items()}
    # Idle destinations have infinite headroom, but never enter the empirical F pool.
    for key, state in states.items():
        if state.get('running') == state.get('waiting') == 0 and finite(state.get('free_blocks')):
            loads[key] = math.inf
    reasons = Counter()
    for src, dst in PAIRS:
        a, b = (states[snapshot['mapping'][i][0]] for i in (src, dst))
        valid, reason = eligible_pair(a, b, now_ns)
        if not valid:
            reasons[reason] += 1
    legal = sum(bool(v) for v in raw['action_mask'][1:])
    action = balanced_action({'raw': raw}, snapshot, threshold=threshold) if threshold is not None else 0
    srcs = sum(v is not None and v <= threshold for v in loads.values()) if threshold is not None else None
    dsts = sum(v is not None and v >= threshold for v in loads.values()) if threshold is not None else None
    threshold_pairs = sum(raw['action_mask'][i+1] and loads.get(snapshot['mapping'][s][0]) is not None
        and loads.get(snapshot['mapping'][d][0]) is not None
        and loads[snapshot['mapping'][s][0]] <= threshold <= loads[snapshot['mapping'][d][0]]
        for i, (s, d) in enumerate(PAIRS)) if threshold is not None else None
    return dict(legal_pairs=legal, potential_sources=srcs, potential_destinations=dsts,
        threshold_pairs=threshold_pairs, threshold_blocked=bool(legal and threshold_pairs == 0),
        balanced_predicate_or_sorting_blocked=bool(threshold_pairs and not action),
        busy=bool(snapshot['busy']), cooldown=raw['global_features'][
            raw.get('global_feature_names', GLOBAL_FEATURES).index('cooldown_remaining_sec')] > 0,
        pair_rejections=dict(reasons), action=action)


def migration_statistics(decisions, migrations, accepted, duration_sec):
    if duration_sec <= 0:
        raise ValueError('positive duration required')
    success = [m for m in migrations if m['terminal_status'] == 'succeeded']
    opportunities = sum(d['opportunity']['legal_pairs'] > 0 for d in decisions)
    actions = sum(d['sampled_action'] > 0 for d in decisions)
    conditional_actions = sum(d['sampled_action'] > 0 and d['opportunity']['legal_pairs'] > 0 for d in decisions)
    unique = {m['actual_request_id'] for m in success}
    if None in unique or len(unique) > accepted:
        raise ValueError('unreconciled successful request identities')
    times = sorted(m['terminal_ns']/1e9 for m in success)
    costs = [(m['terminal_ns']-m['attempt_ns'])/1e9 for m in success]
    return dict(selected=actions, submitted=sum(d['sampled_action'] > 0 and d.get('rejection_reason')!='saved_action_mask_invalid' for d in decisions), started=sum(m['attempt_ns'] is not None for m in migrations),
        succeeded=len(success), rejected=sum(m['terminal_status']=='rejected' for m in migrations),
        failed=sum(m['terminal_status']=='failed' for m in migrations),
        aborted=sum(m['terminal_status']=='aborted' for m in migrations),
        successes_per_minute=len(success)*60/duration_sec,
        request_coverage=len(unique)/accepted if accepted else None,
        opportunity_ratio=opportunities/len(decisions) if decisions else None,
        action_given_opportunity=conditional_actions/opportunities if opportunities else None,
        success_rate=len(success)/len(migrations) if migrations else None,
        migrations_per_accepted_request=len(success)/accepted if accepted else None,
        success_intervals_sec=[b-a for a,b in zip(times,times[1:])],
        duration_mean_sec=statistics.mean(costs) if costs else None,
        duration_p95_sec=sorted(costs)[math.ceil(.95*len(costs))-1] if costs else None,
        per_instance_out=dict(Counter(m['src_id'] for m in success)),
        per_instance_in=dict(Counter(m['dst_id'] for m in success)))


RESOURCE_FIELDS = ('kv_pct', 'gpu_pct', 'memory_mib', 'memory_pct', 'running', 'waiting',
                   'head_wait_sec', 'free_blocks', 'used_blocks', 'prefill_tps', 'decode_tps', 'output_tps')


def resource_summary(samples, start, end):
    result = {}
    ids = sorted({r['instance_id'] for sample in samples for r in sample['instances']})
    if len(ids) != 4:
        raise ValueError('four instance identities required')
    for metric in RESOURCE_FIELDS:
        per_instance = {key: weighted_summary([((s.get('gpu_at') or s['at']) if metric in ('gpu_pct','memory_mib','memory_pct') else s['at'], next(
            (r.get(metric) for r in s['instances'] if r['instance_id']==key), None)) for s in samples], start, end)
            for key in ids}
        std, spread = [], []
        for sample in samples:
            values = {r['instance_id']: r.get(metric) for r in sample['instances']}
            complete = set(values)==set(ids) and all(finite(v) for v in values.values())
            at = (sample.get('gpu_at') or sample['at']) if metric in ('gpu_pct','memory_mib','memory_pct') else sample['at']
            std.append((at, statistics.pstdev(values.values()) if complete else None))
            spread.append((at, max(values.values())-min(values.values()) if complete else None))
        means = [r['mean'] for r in per_instance.values()]
        result[metric] = dict(instances=per_instance,
            cluster_mean=statistics.mean(means) if all(finite(v) for v in means) else None,
            dispersion=weighted_summary(std, start, end), spread=weighted_summary(spread, start, end))
    return result
