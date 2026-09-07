"""Pins the interval estimates quoted in the writeup.

Every number asserted here appears in the post or in review correspondence. If one of these
tests fails, a published claim is wrong -- that is the point of the file. The Clopper-Pearson
values were cross-checked against scipy's `beta.ppf(0.95, 1, n)` before scipy was dropped as a
dependency; the closed form agrees to the displayed precision.
"""

from __future__ import annotations

import pytest

from evalrep.stats import (
    cp_upper_zero,
    n_to_exclude,
    newcombe_diff,
    prob_zero_given_rate,
    wilson,
)

# --- the six Stage 0 cells, as published -------------------------------------------------


@pytest.mark.parametrize(
    "hits,n,lo,hi",
    [
        (19, 30, 0.455, 0.781),  # gpt-4-0613, both prompt variants
        (1, 30, 0.006, 0.167),   # gpt-4o
        (0, 30, 0.000, 0.114),   # gpt-5.5 and gpt-5.4-mini
        (207, 300, 0.636, 0.740),  # Apollo's own gpt-4-0613
        (224, 300, 0.695, 0.793),  # Apollo's own gpt-4-32k-0613
    ],
)
def test_published_wilson_intervals(hits, n, lo, hi):
    got_lo, got_hi = wilson(hits, n)
    assert round(got_lo, 3) == lo
    assert round(got_hi, 3) == hi


# --- the difference claim ----------------------------------------------------------------


def test_difference_upper_bound_cannot_exceed_p1_upper_bound():
    """INVARIANT, and the test that was missing. When p2 = 0 its Wilson lower bound is also 0,
    so Newcombe's second term vanishes and the difference's upper bound must EQUAL p1's own
    Wilson upper bound. A difference cannot exceed the largest credible value of its minuend.

    An earlier implementation double-multiplied by z and returned 92.3% here against a p1
    upper bound of 78.1%. Pinning that output as 'expected' is what let it survive."""
    for h1, n1, n2 in [(19, 30, 30), (19, 30, 300), (207, 300, 300), (30, 263, 300)]:
        _, diff_hi = newcombe_diff(h1, n1, 0, n2)
        _, p1_hi = wilson(h1, n1)
        assert round(diff_hi, 9) == round(p1_hi, 9), f"{h1}/{n1} vs 0/{n2}"


def test_difference_interval_never_exceeds_unit_range():
    for h1, n1, h2, n2 in [(19, 30, 0, 300), (300, 300, 0, 300), (0, 30, 300, 300)]:
        lo, hi = newcombe_diff(h1, n1, h2, n2)
        assert -1.0 <= lo <= hi <= 1.0


def test_published_difference_intervals():
    """Independently derived by a reviewer working from Newcombe (1998) method 10, then
    confirmed by hand -- NOT copied from this implementation's output."""
    assert tuple(round(x, 3) for x in newcombe_diff(19, 30, 0, 30)) == (0.422, 0.781)
    assert tuple(round(x, 3) for x in newcombe_diff(19, 30, 0, 300)) == (0.455, 0.781)
    assert tuple(round(x, 3) for x in newcombe_diff(207, 300, 0, 300)) == (0.634, 0.740)
    assert tuple(round(x, 3) for x in newcombe_diff(30, 263, 0, 300)) == (0.079, 0.158)


def test_larger_zero_arm_tightens_the_lower_bound_only():
    """Enlarging the arm that is already at zero improves the lower bound and leaves the upper
    bound untouched -- which is why it buys less than intuition suggests."""
    lo30, hi30 = newcombe_diff(19, 30, 0, 30)
    lo300, hi300 = newcombe_diff(19, 30, 0, 300)
    assert lo300 > lo30
    assert round(hi300, 9) == round(hi30, 9)


def test_difference_interval_is_symmetric_under_swap():
    lo, hi = newcombe_diff(19, 30, 0, 30)
    lo2, hi2 = newcombe_diff(0, 30, 19, 30)
    assert round(lo, 6) == round(-hi2, 6)
    assert round(hi, 6) == round(-lo2, 6)


# --- zero-count bounds -------------------------------------------------------------------


def test_cp_upper_zero_matches_scipy_values():
    """Cross-checked against scipy beta.ppf(0.95, 1, n) before dropping the dependency."""
    assert round(cp_upper_zero(30), 3) == 0.095
    assert round(cp_upper_zero(300), 3) == 0.010


def test_one_sided_bound_is_tighter_than_two_sided_wilson():
    """For an observed zero, the two-sided Wilson upper bound overstates the plausible rate."""
    assert cp_upper_zero(30) < wilson(0, 30)[1]


# --- the power argument ------------------------------------------------------------------


@pytest.mark.parametrize(
    "rate,expected",
    [(0.01, 0.740), (0.02, 0.545), (0.05, 0.215), (0.10, 0.042)],
)
def test_false_negative_risk_at_n30(rate, expected):
    """n=30 is nearly blind to a 1% rate: zero observed 74% of the time."""
    assert round(prob_zero_given_rate(rate, 30), 3) == expected


@pytest.mark.parametrize("rate,n", [(0.01, 299), (0.02, 149), (0.05, 59), (0.10, 29)])
def test_sample_size_needed_to_exclude_a_rate(rate, n):
    """299 to exclude 1% is presumably why Apollo chose n=300."""
    assert n_to_exclude(rate) == n


def test_n300_moves_the_detection_floor_by_an_order_of_magnitude():
    assert cp_upper_zero(30) / cp_upper_zero(300) > 9


# --- guards ------------------------------------------------------------------------------


def test_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        wilson(31, 30)
    with pytest.raises(ValueError):
        wilson(0, 0)
    with pytest.raises(ValueError):
        cp_upper_zero(0)
    with pytest.raises(ValueError):
        n_to_exclude(0)
