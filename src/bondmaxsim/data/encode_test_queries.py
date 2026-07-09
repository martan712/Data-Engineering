"""Encode test-split queries for datasets whose ported queries lack qrels.

Single responsibility: one-shot generation of data/embeddings/<ds>_test_queries.npz
for datasets where the archive cache encoded the FIRST 200 rows of the BEIR
queries split and those rows carry no test qrels (scifact, nfcorpus — their
splits lead with train queries).  The main <ds>.npz is never touched: Stage 3/4
timing reproducibility rests on it.

Encodes with the same model and settings as the archive cache
(lightonai/GTE-ModernColBERT-v1 via PyLate, is_query=True) and verifies
unit-norm before writing.  Requires the [retrieval] extra.

Ported artifact: encoding loop from
  archive/preliminaries/02_bond_variance/bond_variance_analysis.py (encode_or_load).
Stage 1 reference: docs/project_b_analysis_and_research_plan.md Stage 5 section
  (qrels metrics require queries with test judgments).
"""

from __future__ import annotations

import numpy as np

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.beir_ids import load_qrels_tsv, qrels_path
from bondmaxsim.oracle.normalization import check_unit_norm

OUT_DIR = REPO_ROOT / "data" / "embeddings"

MODEL_NAME = "lightonai/GTE-ModernColBERT-v1"

# Datasets whose ported 200 queries have zero overlap with test qrels
# (established by bondmaxsim.data.beir_ids.check_dataset, 2026-07-09).
DATASETS: tuple[str, ...] = ("scifact", "nfcorpus")

BATCH_SIZE = 32


def encode_test_queries(dataset: str, model=None) -> dict[str, int]:
    """Encode all test-qrels queries of a dataset to <ds>_test_queries.npz.

    Parameters
    ----------
    dataset : BEIR dataset name (must have data/qrels/<ds>.tsv already —
              run bondmaxsim.data.beir_ids first)
    model   : optional pre-loaded pylate ColBERT model (reused across datasets)

    Returns
    -------
    Summary dict: num_queries, total_query_tokens.
    """
    from datasets import load_dataset as hf_load_dataset
    from pylate import models

    qrels = load_qrels_tsv(qrels_path(dataset))
    test_qids = sorted(qrels.keys())

    queries_ds = hf_load_dataset(f"BeIR/{dataset}", "queries", split="queries")
    text_by_id = {str(row["_id"]): row["text"] for row in queries_ds}
    missing = [q for q in test_qids if q not in text_by_id]
    if missing:
        raise ValueError(
            f"{dataset}: {len(missing)} test-qrels queries missing from the "
            f"queries split (e.g. {missing[:3]})"
        )
    texts = [text_by_id[q] for q in test_qids]

    if model is None:
        model = models.ColBERT(MODEL_NAME)
    embs = model.encode(
        texts, is_query=True, batch_size=BATCH_SIZE, show_progress_bar=True
    )

    chunks: list[np.ndarray] = []
    starts: list[int] = []
    offset = 0
    for i, emb in enumerate(embs):
        mat = np.asarray(emb, dtype=np.float32)
        if mat.ndim != 2 or mat.shape[1] != 128:
            raise ValueError(
                f"{dataset} query[{i}]: expected shape [*, 128], got {mat.shape}"
            )
        starts.append(offset)
        chunks.append(mat)
        offset += mat.shape[0]
    query_values = np.concatenate(chunks, axis=0)
    query_starts = np.array(starts, dtype=np.int64)

    stats = check_unit_norm(query_values)
    if stats["n_violating"] > 0:
        raise AssertionError(
            f"{dataset} test-query tokens failed unit-norm check: {stats}"
        )

    out_path = OUT_DIR / f"{dataset}_test_queries.npz"
    np.savez(
        out_path,
        query_values=query_values,
        query_starts=query_starts,
        query_ids=np.array(test_qids),
    )
    print(f"  wrote {out_path.name}: {len(test_qids)} queries, {offset} tokens")
    return {"num_queries": len(test_qids), "total_query_tokens": offset}


def main() -> None:
    """Encode test queries for all datasets that need them."""
    from pylate import models

    print("=== BOND-MaxSim Stage 5: encode test-split queries ===\n")
    model = models.ColBERT(MODEL_NAME)
    for ds in DATASETS:
        print(f"[{ds}]")
        encode_test_queries(ds, model=model)
        print()


if __name__ == "__main__":
    main()
