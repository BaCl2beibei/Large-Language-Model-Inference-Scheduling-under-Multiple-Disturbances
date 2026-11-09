"""Pure CPU cohort statistics; no real-service imports."""
import math


def percentile_type7(values, p):
    values = sorted(float(x) for x in values)
    if not values:
        return None
    h = (len(values) - 1) * p
    lo, hi = math.floor(h), math.ceil(h)
    return values[lo] + (values[hi] - values[lo]) * (h - lo)


def cvar95(values):
    """Empirical upper-tail mean with exact fractional boundary weight."""
    values = sorted(float(x) for x in values)
    if not values:
        return None
    alpha = 0.05
    tail_mass = alpha * len(values)
    boundary = values[max(0, math.ceil((1.0 - alpha) * len(values)) - 1)]
    excess = sum(max(value - boundary, 0.0) for value in values)
    # This Rockafellar-Uryasev form handles a quantile tie without rounding N*.05.
    return boundary + excess / tail_mass


def summarize_values(values):
    values = [float(x) for x in values]
    if not values:
        return {key: None for key in ('mean', 'median', 'p90', 'p95', 'p99', 'max', 'cvar95')}
    return {
        'mean': sum(values) / len(values),
        'median': percentile_type7(values, .5),
        'p90': percentile_type7(values, .90),
        'p95': percentile_type7(values, .95),
        'p99': percentile_type7(values, .99),
        'max': max(values),
        'cvar95': cvar95(values),
    }


