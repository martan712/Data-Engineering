# Agreement contract

Status: Accepted
Version: 1

## Separate operations

The implementation exposes four distinct concepts:

- `strict_top_k_set_equal`: exact equality of two unique top-k ID sets. Ranked
  order is intentionally ignored.
- `validate_boundary_tie_equivalence`: validates a returned top-k against an
  oracle using independently exact-scored substitutions at the oracle k-th
  boundary.
- `recall_vs_oracle_set`: ordinary set recall against one oracle tie-breaking
  choice. A value of `1.0` is not called strict ranked equality or a proof of
  tie equivalence.
- `validate_ranked_order`: optional validation that scores are finite,
  non-increasing, and consistently tie-broken for consumers that require an
  ordered ranking.

`validate_boundary_tie_equivalence` receives returned IDs, oracle top-k IDs,
oracle top-k scores, total document count, and either an exact score mapping
covering every returned ID or an independent exact scorer. Scores emitted only
by the implementation under test are not acceptable evidence.

## Structural rules

Both strict and tie-aware gates require exactly `k` IDs, integer-representable
IDs in `[0, num_documents)`, and no duplicates. Oracle scores and every exact
replacement score must be finite. The oracle arrays must also have cardinality
`k`, unique valid IDs, and finite non-increasing scores. A malformed input is a
validation failure, never partial credit.

Let `tau` be the minimum oracle top-k score. A tie-equivalent result must:

1. contain every oracle document whose exact score is above `tau` by more than
   the tie tolerance;
2. give every returned ID outside the oracle set an independently computed
   exact score equal to `tau` within that tolerance; and
3. satisfy the structural rules above.

The tolerance is symmetric and fp32-specific. Two finite scores `a` and `b`
are boundary-equal when:

```text
abs(float32(a) - float32(b)) <= 8 * max(spacing(abs(float32(a))),
                                        spacing(abs(float32(b))),
                                        smallest_positive_subnormal_float32)
```

Eight ULPs cover small accumulation/reduction-order differences while scaling
with the score magnitude. It is applied only to independently exact scores at
the boundary; it never permits a missing document that is unambiguously above
the boundary or turns ordinary recall into exactness.

## Result fields

Agreement results serialize as:

```text
strict_top_k_set_equal: boolean
boundary_tie_equivalent: boolean
recall_vs_oracle_set: number in [0, 1]
ranked_order_valid: boolean or null
k: positive integer
fp32_tie_max_ulps: 8
failure_codes: array of stable strings
```

Hard exact-safe gates pass when `strict_top_k_set_equal` is true or
`boundary_tie_equivalent` is true. Reports always retain both fields. Stable
failure codes distinguish malformed output, above-boundary miss,
below-boundary replacement, non-finite score, and ranked-order failure.
