"""
Interval estimates for the insider-trading replication.

WHY A MODULE. These were computed ad hoc in shell heredocs several times while reviewing
Stage 0, which is exactly how a transcription error gets into a published number. The values
pinned in tests/test_stats.py are the ones quoted in the writeup.

NO SCIPY. The one non-elementary quantity needed -- an exact binomial upper bound for a ZERO
count -- has a closed form (see cp_upper_zero), so there is no dependency to declare and no
`uv sync` pruning hazard.

Why these three:
  wilson         -- every reported proportion needs an interval. Inspect's built-in
                    stderr reports 0.000 for a 1-sample x N-epochs design, because epochs
                    collapse to one per-sample score before the metric runs.
  newcombe_diff  -- the headline claim is a DIFFERENCE (legacy vs current model). Reporting
                    two separate intervals and eyeballing the gap is not a test of it.
  cp_upper_zero  -- for an observed 0/n, a one-sided exact bound is the honest statistic. A
                    two-sided Wilson upper bound on a zero count is both wider and answering
                    a question nobody asked.
"""

from __future__ import annotations

import math

Z95 = 1.959963984540054  # two-sided 95%


def wilson(hits: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= hits <= n:
        raise ValueError(f"hits={hits} out of range for n={n}")
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def newcombe_diff(
    hits1: int, n1: int, hits2: int, n2: int, z: float = Z95
) -> tuple[float, float]:
    """Newcombe (1998) method 10 hybrid-score interval for a difference of proportions.

    Returns the interval for (p1 - p2). Chosen over a normal-approximation difference
    interval because one arm here is at zero, where that approximation is worst.

        lo = (p1 - p2) - sqrt((p1 - l1)^2 + (u2 - p2)^2)
        hi = (p1 - p2) + sqrt((u1 - p1)^2 + (p2 - l2)^2)

    BUG FIXED 2026-09-02, and worth recording because the tests did not catch it. An earlier
    version multiplied both square roots by `z` a second time, inflating every interval by
    exactly 1.96x. It produced e.g. [21.9%, 92.3%] for 19/30 vs 0/30 where the correct answer
    is [42.2%, 78.1%]. The Wilson half-widths (p1 - l1) etc. ALREADY contain z; there is no
    second multiplier in the published method.

    The tell was structural and is now a test: when p2 = 0 its Wilson lower bound is also 0,
    so the second term of `hi` vanishes and the difference's upper bound must equal p1's own
    Wilson upper bound exactly. The buggy version returned 92.3% where p1's own upper bound
    was 78.1% -- a difference cannot exceed the largest credible value of its minuend. Pinning
    an implementation's own output as "expected" catches regressions but not errors; the
    invariants below are what would have caught this.
    """
    lo1, hi1 = wilson(hits1, n1, z)
    lo2, hi2 = wilson(hits2, n2, z)
    p1, p2 = hits1 / n1, hits2 / n2
    delta = p1 - p2
    lo = delta - math.sqrt((p1 - lo1) ** 2 + (hi2 - p2) ** 2)
    hi = delta + math.sqrt((hi1 - p1) ** 2 + (p2 - lo2) ** 2)
    return (max(-1.0, lo), min(1.0, hi))


def cp_upper_zero(n: int, alpha: float = 0.05) -> float:
    """Exact one-sided Clopper-Pearson upper bound when ZERO events were observed.

    For hits=0 the general beta quantile collapses to a closed form: the largest p for which
    P(0 events in n draws) >= alpha is 1 - alpha**(1/n), since P(0|p) = (1-p)**n.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    return 1.0 - alpha ** (1.0 / n)


def prob_zero_given_rate(rate: float, n: int) -> float:
    """P(observing zero events in n draws | true rate). The false-negative risk.

    This is the calculation that makes a bare '0/30' misleading: at n=30 a true 1% rate
    yields zero observations 74% of the time.
    """
    if not 0 <= rate <= 1:
        raise ValueError("rate must be in [0, 1]")
    return (1 - rate) ** n


def n_to_exclude(rate: float, alpha: float = 0.05) -> int:
    """Smallest n such that observing zero events excludes a true rate >= `rate`."""
    if not 0 < rate < 1:
        raise ValueError("rate must be in (0, 1)")
    return math.ceil(math.log(alpha) / math.log(1 - rate))
