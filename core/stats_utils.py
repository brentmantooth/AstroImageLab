"""Shared statistical-comparison utilities used by both analysis/ and report/."""
from __future__ import annotations


def mannwhitney_effect(va, vb) -> tuple[float | None, float | None]:
    """Mann-Whitney U p-value and Cliff's delta for two independent samples.

    Returns (p_value, delta), or (None, None) if either sample has fewer than
    3 values. delta > 0 means va's values tend to be higher than vb's; |delta|
    is in [0, 1]. Uses the exact identity delta = 2*U/(n1*n2) - 1 (U = the
    Mann-Whitney U statistic for va) rather than an O(n1*n2) pairwise sign
    matrix, so this scales to large samples (tens of thousands of pixel
    values) as well as small ones (dozens of per-star measurements).
    """
    from scipy.stats import mannwhitneyu
    n1, n2 = len(va), len(vb)
    if n1 < 3 or n2 < 3:
        return None, None
    u_stat, p = mannwhitneyu(va, vb, alternative="two-sided")
    delta = 2.0 * float(u_stat) / (n1 * n2) - 1.0
    return float(p), delta
