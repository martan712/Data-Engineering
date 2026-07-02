"""Wide-block kernel threshold-seed resolution (RunConfig.threshold_policy -> tau_seed).

Single responsibility: translate the mechanism testbed's threshold_policy
config value into the wide-block kernel's tau_seed scalar, per Stage 1 §4.4.
"""

from __future__ import annotations

import numpy as np

from bondmaxsim.oracle.exact_maxsim import exact_maxsim_scores
from bondmaxsim.testbed.config import RunConfig
from bondmaxsim.testbed.packing_cache import PackingCache
from bondmaxsim.threshold.policies import oracle_threshold, seed_threshold

# Cheap "first checkpoint" proxy for the seed threshold policy: partial
# MaxSim scores computed from only the first few dimensions of the
# per-query scan order (Stage 1 §4.4 "seeded threshold").
_SEED_CHECKPOINT_DIMS = 32
_SEED_FRACTION = 0.05

# Safety margin subtracted from any tau_seed derived from a NumPy exact-
# score computation (oracle/seed policies) before handing it to the
# kernel, to absorb cross-implementation fp32 summation-order noise at
# near-tie thresholds (see resolve_tau_seed docstring).
_TAU_SEED_EPS = 1e-3


def resolve_tau_seed(
    config: RunConfig,
    query: np.ndarray,
    order: np.ndarray,
    dimension_order: str,
    packing: PackingCache,
) -> float:
    """Resolve the wide-block kernel's tau_seed from RunConfig.threshold_policy.

    Stage 1 §4.4: "self_bound"/"exact_safe_topk" -> -inf (self-bounded;
    relies entirely on staged finalization across groups).  "oracle" ->
    the true k-th best exact score (upper bound on pruning potential;
    rotation-invariant, computed on the original un-rotated corpus).
    "seed" -> bondmaxsim.threshold.policies.seed_threshold using a cheap
    partial-score proxy: exact MaxSim restricted to the first
    _SEED_CHECKPOINT_DIMS dimensions of THIS query's scan order (in the
    same effective space -- rotated for order="ada" -- the kernel itself
    will see first), consistent with the policy's "first checkpoint" intent.

    Both "oracle" and "seed" derive tau from a NumPy exact-score
    computation whose fp32 summation order differs from the kernel's own
    incremental per-dimension-block accumulation (Stage 1 §4.2: fp
    exactness is empirical, not proven).  The theorem only requires tau
    to be a valid *lower* bound on the true k-th best score; a tau set
    to EXACTLY that score is a hairline tie that a cross-implementation
    fp32 mismatch can occasionally break (a document whose kernel-
    internal score computes a few ULPs below the NumPy tau gets
    spuriously pruned).  Subtracting a small safety margin keeps tau a
    strict lower bound in practice without materially loosening pruning.
    """
    policy = config.threshold_policy
    if policy in ("self_bound", "exact_safe_topk"):
        return float("-inf")

    exact_scores = exact_maxsim_scores(query, packing.flat_tokens, packing.doc_starts)

    if policy == "oracle":
        return oracle_threshold(exact_scores, config.k) - _TAU_SEED_EPS

    if policy == "seed":
        if dimension_order == "ada":
            flat_eff_tm = packing.get_flat_tokens_rot()
            R = packing.get_ada_rotation()
            q_eff = (query @ R).astype(np.float32)
        else:
            flat_eff_tm = packing.flat_tokens
            q_eff = query
        n_dims = min(packing.D, _SEED_CHECKPOINT_DIMS)
        dims = order[:n_dims]
        partial_scores = exact_maxsim_scores(
            q_eff[:, dims], flat_eff_tm[:, dims], packing.doc_starts
        )
        return seed_threshold(
            partial_scores, exact_scores, config.k, seed_fraction=_SEED_FRACTION
        ) - _TAU_SEED_EPS

    raise ValueError(
        f"Unknown threshold_policy for wide_block kernel: {policy!r}. "
        "Expected one of 'self_bound', 'exact_safe_topk', 'oracle', 'seed'."
    )
