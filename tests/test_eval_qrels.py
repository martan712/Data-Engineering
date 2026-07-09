"""Stage 5 eval gates: qrels metrics, TSV round-trip, CoRECT agreement.

The qrels metrics are checked against hand-computed values on a tiny fixture
(so a ranx upgrade changing semantics is caught), the qrels TSV round-trips
through the data layer, and the CoRECT adapter must agree with ranx on the
same run (the plan's "CoRECT wrapper smoke test" at the synthetic level; the
Stage 5 driver repeats it on the real dense_fused run before scaling).

Requires the [retrieval] extra (ranx, pytrec_eval); skips cleanly if absent.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("ranx")

from bondmaxsim.data.beir_ids import load_qrels_tsv, save_qrels_tsv  # noqa: E402
from bondmaxsim.eval.qrels import (  # noqa: E402
    compute_quality_metrics,
    load_qrels,
    mrr_at_10,
    ndcg_at_10,
    recall_at_100,
)

# ---------------------------------------------------------------------------
# Fixture: 2 queries, hand-checkable
#
# q1: qrels {d1:2, d3:1}; run ranks d2(miss), d1(rel=2), d3(rel=1)
#     DCG@10  = 2/log2(3) + 1/log2(4) = 1.26186 + 0.5 = 1.76186
#     IDCG@10 = 2/log2(2) + 1/log2(3) = 2 + 0.63093 = 2.63093
#     nDCG@10 = 0.66968...   recall@100 = 1.0   RR@10 = 1/2
# q2: qrels {d5:1}; run ranks d6, d7 (d5 never retrieved)
#     nDCG@10 = 0            recall@100 = 0.0   RR@10 = 0
# ---------------------------------------------------------------------------

QRELS = {"q1": {"d1": 2, "d3": 1}, "q2": {"d5": 1}}
RUN = {
    "q1": {"d2": 0.9, "d1": 0.8, "d3": 0.7},
    "q2": {"d6": 0.6, "d7": 0.5},
}

_Q1_NDCG = (2 / math.log2(3) + 1 / math.log2(4)) / (2 + 1 / math.log2(3))


def test_ndcg_at_10_hand_computed():
    expected = (_Q1_NDCG + 0.0) / 2
    assert ndcg_at_10(RUN, QRELS) == pytest.approx(expected, abs=1e-9)


def test_recall_at_100_hand_computed():
    assert recall_at_100(RUN, QRELS) == pytest.approx(0.5, abs=1e-9)


def test_mrr_at_10_hand_computed():
    assert mrr_at_10(RUN, QRELS) == pytest.approx(0.25, abs=1e-9)


def test_compute_quality_metrics_matches_singles():
    out = compute_quality_metrics(RUN, QRELS)
    assert out["nDCG_at_10"] == pytest.approx(ndcg_at_10(RUN, QRELS))
    assert out["recall_at_100"] == pytest.approx(recall_at_100(RUN, QRELS))
    assert out["MRR_at_10"] == pytest.approx(mrr_at_10(RUN, QRELS))


def test_unjudged_queries_dropped():
    run = dict(RUN)
    run["q_unjudged"] = {"d1": 1.0}
    assert ndcg_at_10(run, QRELS) == pytest.approx(ndcg_at_10(RUN, QRELS))


def test_no_overlap_raises():
    with pytest.raises(ValueError, match="No overlap"):
        ndcg_at_10({"qX": {"d1": 1.0}}, QRELS)


def test_qrels_tsv_round_trip(tmp_path):
    path = tmp_path / "qrels.tsv"
    save_qrels_tsv(QRELS, path)
    assert load_qrels_tsv(path) == QRELS
    assert load_qrels(path) == QRELS  # eval-layer re-export


def test_corect_agreement_with_ranx():
    pytest.importorskip("pytrec_eval")
    from bondmaxsim.eval.corect import corect_smoke_test

    assert corect_smoke_test()  # synthetic fixture
    assert corect_smoke_test(RUN, QRELS)  # this file's fixture
