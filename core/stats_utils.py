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


def combined_se_z_test(val_a: float | None, se_a: float | None,
                        val_b: float | None, se_b: float | None
                        ) -> tuple[float | None, float | None]:
    """Two-sided z-test comparing two independent estimates with known SEs.

    z = (val_a - val_b) / sqrt(se_a**2 + se_b**2), p from the standard normal
    (not Student-t: there is no shared degrees-of-freedom between two
    independently-fit estimates). Returns (None, None) if either value/SE is
    None or both SEs are zero. This is an approximate (Wald-style) comparison,
    appropriate for combining two independent point estimates -- not a
    substitute for a paired test when the two samples are pixel-paired (see
    core.stats_utils.mannwhitney_effect for that case).
    """
    from scipy.stats import norm
    if val_a is None or se_a is None or val_b is None or se_b is None:
        return None, None
    denom = (se_a ** 2 + se_b ** 2) ** 0.5
    if denom <= 0:
        return None, None
    z = (val_a - val_b) / denom
    p = 2.0 * float(norm.sf(abs(z)))
    return float(z), p
