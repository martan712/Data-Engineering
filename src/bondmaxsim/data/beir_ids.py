"""BEIR doc/query ID sidecars and test qrels for the ported embeddings.

Single responsibility: reconstruct the BEIR string IDs for the rows of the
data/embeddings/<ds>.npz archives (which store embeddings only) and fetch the
test-split qrels, so Stage 5 can map kernel doc indices back to BEIR IDs and
evaluate against relevance judgments.

Provenance contract (why row order == ID order): the archive caches were
encoded by archive/preliminaries/02_bond_variance/bond_variance_analysis.py in
HuggingFace `BeIR/<ds>` corpus-split row order (max_docs=None, text-only) and
queries-split row order (first 200 rows), and port_embeddings.py preserved
that order.  verify_row_order() re-encodes a few docs to check this
empirically before any Stage 5 claim rests on it.

Ported artifact: qrels/ID plumbing from
  archive/colbert_scripts/02b_corect_bruteforce.py (mteb/<ds> qrels loading).
Stage 1 reference: docs/project_b_analysis_and_research_plan.md Stage 5 section
  (qrels metrics need BEIR IDs; CoRECT smoke test before scaling).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bondmaxsim.config import REPO_ROOT

# ---------------------------------------------------------------------------
# Paths and dataset registry
# ---------------------------------------------------------------------------

IDS_DIR: Path = REPO_ROOT / "data" / "beir_ids"
"""Sidecar directory: <ds>_ids.json with row-order corpus/query BEIR IDs."""

QRELS_DIR: Path = REPO_ROOT / "data" / "qrels"
"""Test-split qrels TSVs (query-id \\t corpus-id \\t score, with header)."""

DATASETS: tuple[str, ...] = ("scifact", "nfcorpus", "arguana", "scidocs")

_EMBED_DIR: Path = REPO_ROOT / "data" / "embeddings"


# ---------------------------------------------------------------------------
# Fetching from HuggingFace
# ---------------------------------------------------------------------------


def fetch_ids(dataset: str) -> tuple[list[str], list[str]]:
    """Fetch row-order corpus and query BEIR IDs from HuggingFace BeIR/<ds>.

    Returns
    -------
    corpus_ids : list[str] — _id of every corpus row, in split row order
    query_ids  : list[str] — _id of every queries row, in split row order
    """
    from datasets import load_dataset  # deferred: [retrieval]-adjacent dep

    corpus_ds = load_dataset(f"BeIR/{dataset}", "corpus", split="corpus")
    queries_ds = load_dataset(f"BeIR/{dataset}", "queries", split="queries")
    corpus_ids = [str(x) for x in corpus_ds["_id"]]
    query_ids = [str(x) for x in queries_ds["_id"]]
    return corpus_ids, query_ids


def fetch_qrels(dataset: str) -> dict[str, dict[str, int]]:
    """Fetch test-split qrels for a BEIR dataset.

    Tries `BeIR/<ds>-qrels` (split 'test') first, then falls back to
    `mteb/<ds>` ('default', split 'test') — the source the archived
    02b_corect_bruteforce.py used.  Both mirror the original BEIR IDs.

    Returns
    -------
    {query_id: {doc_id: relevance_int}}
    """
    from datasets import load_dataset

    try:
        ds = load_dataset(f"BeIR/{dataset}-qrels", split="test")
    except Exception:
        ds = load_dataset(f"mteb/{dataset}", "default", split="test")

    qrels: dict[str, dict[str, int]] = {}
    for row in ds:
        qid = str(row["query-id"])
        did = str(row["corpus-id"])
        qrels.setdefault(qid, {})[did] = int(row["score"])
    return qrels


# ---------------------------------------------------------------------------
# Sidecar IO
# ---------------------------------------------------------------------------


def save_sidecar(
    dataset: str, corpus_ids: list[str], query_ids: list[str]
) -> Path:
    """Write data/beir_ids/<ds>_ids.json.  Overwrites (idempotent)."""
    IDS_DIR.mkdir(parents=True, exist_ok=True)
    path = IDS_DIR / f"{dataset}_ids.json"
    payload = {
        "dataset": dataset,
        "hf_source": f"BeIR/{dataset}",
        "corpus_ids": corpus_ids,
        "query_ids": query_ids,
    }
    path.write_text(json.dumps(payload))
    return path


def load_ids(dataset: str) -> dict:
    """Load the ID sidecar for a dataset.

    Returns the saved dict with keys 'corpus_ids' and 'query_ids' (row order).

    Raises
    ------
    FileNotFoundError with regeneration instructions if missing.
    """
    path = IDS_DIR / f"{dataset}_ids.json"
    if not path.exists():
        raise FileNotFoundError(
            f"ID sidecar not found: {path}\n"
            "Generate it by running:\n"
            "    uv run python -m bondmaxsim.data.beir_ids"
        )
    return json.loads(path.read_text())


def save_qrels_tsv(qrels: dict[str, dict[str, int]], path: Path) -> None:
    """Write qrels as a BEIR-format TSV (header: query-id, corpus-id, score)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["query-id\tcorpus-id\tscore"]
    for qid in sorted(qrels):
        for did, rel in sorted(qrels[qid].items()):
            lines.append(f"{qid}\t{did}\t{rel}")
    path.write_text("\n".join(lines) + "\n")


def qrels_path(dataset: str) -> Path:
    """Canonical path of the test-qrels TSV for a dataset."""
    return QRELS_DIR / f"{dataset}.tsv"


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def check_dataset(dataset: str) -> dict:
    """Cross-check sidecar + qrels against the ported .npz row counts.

    Returns a summary dict:
      num_docs_npz / num_docs_ids : must match (row-order precondition)
      num_queries_npz             : queries actually encoded (first N rows)
      num_qrels_queries           : queries in the test qrels
      num_evaluable               : encoded queries that appear in test qrels
    """
    ids = load_ids(dataset)

    npz = np.load(_EMBED_DIR / f"{dataset}.npz")
    num_docs_npz = int(len(npz["doc_starts"]))
    num_queries_npz = int(len(npz["query_starts"]))

    qrels = load_qrels_tsv(qrels_path(dataset))
    encoded_qids = ids["query_ids"][:num_queries_npz]
    evaluable = [q for q in encoded_qids if q in qrels]

    return {
        "dataset": dataset,
        "num_docs_npz": num_docs_npz,
        "num_docs_ids": len(ids["corpus_ids"]),
        "docs_match": num_docs_npz == len(ids["corpus_ids"]),
        "num_queries_npz": num_queries_npz,
        "num_qrels_queries": len(qrels),
        "num_evaluable": len(evaluable),
    }


def load_qrels_tsv(path: Path) -> dict[str, dict[str, int]]:
    """Load a BEIR-format qrels TSV written by save_qrels_tsv.

    Returns
    -------
    {query_id: {doc_id: relevance_int}}
    """
    qrels: dict[str, dict[str, int]] = {}
    with open(path) as f:
        header = f.readline()  # skip "query-id\tcorpus-id\tscore"
        assert header.startswith("query-id"), f"unexpected qrels header: {header!r}"
        for line in f:
            qid, did, rel = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, {})[did] = int(rel)
    return qrels


def verify_row_order(dataset: str, n_docs: int = 3, seed: int = 42) -> dict:
    """Empirically verify the row-order provenance contract.

    Re-encodes n_docs randomly chosen corpus texts (text-only, as the archive
    did) with GTE-ModernColBERT-v1 and compares against the stored embeddings.
    Requires the [retrieval] extra (pylate/torch) and network/model cache.

    Returns {"dataset", "checked": [(row, mean_cos), ...], "ok": bool} where
    mean_cos is the mean per-token cosine between stored and re-encoded tokens
    (tolerance 0.999 — original encode may have run on different hardware).
    """
    from datasets import load_dataset as hf_load_dataset
    from pylate import models

    corpus_ds = hf_load_dataset(f"BeIR/{dataset}", "corpus", split="corpus")
    npz = np.load(_EMBED_DIR / f"{dataset}.npz")
    doc_values, doc_starts = npz["doc_values"], npz["doc_starts"]
    num_docs = len(doc_starts)
    total_tokens = len(doc_values)

    rng = np.random.default_rng(seed)
    rows = sorted(rng.choice(num_docs, size=n_docs, replace=False).tolist())

    model = models.ColBERT("lightonai/GTE-ModernColBERT-v1")
    texts = [corpus_ds[r]["text"] for r in rows]
    fresh = model.encode(texts, is_query=False, batch_size=8)

    checked: list[tuple[int, float]] = []
    ok = True
    for row, emb in zip(rows, fresh):
        start = int(doc_starts[row])
        end = int(doc_starts[row + 1]) if row + 1 < num_docs else total_tokens
        stored = doc_values[start:end]
        emb = np.asarray(emb, dtype=np.float32)
        if stored.shape != emb.shape:
            checked.append((row, float("nan")))
            ok = False
            continue
        cos = float(np.mean(np.sum(stored * emb, axis=1)))
        checked.append((row, cos))
        ok = ok and cos > 0.999
    return {"dataset": dataset, "checked": checked, "ok": ok}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Fetch + save ID sidecars and test qrels for all datasets, then verify."""
    print("=== BOND-MaxSim Stage 5: BEIR ID sidecars + test qrels ===\n")

    for ds in DATASETS:
        print(f"[{ds}]")
        corpus_ids, query_ids = fetch_ids(ds)
        sidecar = save_sidecar(ds, corpus_ids, query_ids)
        print(f"  ids     -> {sidecar.name}  ({len(corpus_ids)} docs, {len(query_ids)} queries)")
        qrels = fetch_qrels(ds)
        save_qrels_tsv(qrels, qrels_path(ds))
        print(f"  qrels   -> {qrels_path(ds).name}  ({len(qrels)} test queries)")
        summary = check_dataset(ds)
        print(
            f"  check   -> docs_match={summary['docs_match']} "
            f"(npz {summary['num_docs_npz']} vs ids {summary['num_docs_ids']}), "
            f"evaluable queries {summary['num_evaluable']}/{summary['num_queries_npz']}"
        )
        print()


if __name__ == "__main__":
    main()
