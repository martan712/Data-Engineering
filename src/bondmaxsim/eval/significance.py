"""Paired permutation test for per-query IR metric differences.

Single responsibility: given two aligned per-query metric vectors (candidate
and baseline over the same queries), decide whether their mean difference is
distinguishable from zero.  Used by stage5 e02 to check whether the small
nDCG@10 inversions of approximate arms over the exact scan are systematic or
noise: exactness is defined with respect to the MaxSim score, not relevance,
so an approximate arm CAN land above the exact scan by luck of the swapped-in
documents.

The test is the standard paired (sign-flip) permutation test: under the null
hypothesis the sign of each per-query difference is arbitrary, so the observed
mean difference is compared against the distribution of mean differences over
random sign assignments (two-sided).
"""

from __future__ import annotations

import numpy as np


def paired_permutation_test(
    candidate: dict[str, float],
    baseline: dict[str, float],
    n_permutations: int = 20_000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Two-sided paired sign-flip permutation test on per-query differences.

    Parameters
    ----------
    candidate, baseline : {query_id: metric_value} over the same query set
    n_permutations      : number of random sign assignments
    seed                : RNG seed (fixed for reproducible p-values)

    Returns
    -------
    {"mean_diff": mean(candidate - baseline),
     "p_value": two-sided permutation p-value,
     "n_queries": total queries,
     "n_changed": queries where the metric differs,
     "n_helped": queries where candidate > baseline,
     "n_hurt": queries where candidate < baseline}
    """
    if set(candidate) != set(baseline):
        raise ValueError("candidate and baseline cover different query sets")
    qids = sorted(candidate)
    d = np.array([candidate[q] - baseline[q] for q in qids])
    mean = float(d.mean())
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(n_permutations, len(d)))
    null = (flips * d).mean(axis=1)
    return {
        "mean_diff": mean,
        "p_value": float((np.abs(null) >= abs(mean)).mean()),
        "n_queries": len(d),
        "n_changed": int((d != 0).sum()),
        "n_helped": int((d > 0).sum()),
        "n_hurt": int((d < 0).sum()),
    }
