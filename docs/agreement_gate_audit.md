# Historical agreement-gate audit

Status: code paths repaired; historical result artifacts not rerun.

The former score-aware `exact_agreement()` accepted a replacement document
without independently scoring it. Consequently, existing exactness claims
from the following experiment families are historical and unverified until
their artifacts are regenerated with `strict_top_k_set_equal` and
`boundary_tie_equivalent` fields:

- Stage 2: `s02_exact_agreement` and `s03_two_mode_smoke`, including every
  `stage2_testbed_exact_agreement_*` and `stage2_testbed_two_mode_smoke_*` JSON.
- Stage 3: mechanism experiments e02 through e09, including all corresponding
  `stage3_mechanism_e0[2-9]_*` JSON files.
- Stage 4: e01 fixed-candidate arms and e02 seeded-threshold recovery. Their
  `stage4_integration_e01_*` and `stage4_integration_e02_*` JSON files used the
  old score-aware path. The e03 partitioned frontier used ordinary recall for
  approximate arms and must not be relabeled as exactness evidence.
- Stage 5: e01 IR evaluation exact-safe dense/BOND gates and all
  `stage5_corect_e01_*` JSON files.

This audit does not invalidate the mathematical exact-safe bound. It marks the
old software evidence as insufficient. The final benchmark stage must rerun
the hard gate over every declared query and retain, per run, strict set
equality, independently verified boundary-tie equivalence, ordinary set
recall, stable failure codes, input hashes, code revision, and environment.

The repaired testbed computes an independent exact score for every document,
uses those scores for every returned substitution, and serializes the strict
and tie-equivalent outcomes separately. Approximate arms may still report
`recall_vs_exact_at_10`; that metric remains ordinary set recall and is not an
exactness claim.
