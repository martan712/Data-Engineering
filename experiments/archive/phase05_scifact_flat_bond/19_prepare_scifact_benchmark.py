"""Prepare a small SciFact/BEIR benchmark subset.

The script downloads the public SciFact BEIR zip if it is not already cached,
then creates a deterministic small subset with real queries and qrels.
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
import zipfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils_colbert import save_json


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "artifacts" / "raw" / "beir"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "scifact_benchmark"
SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


def download_if_needed(zip_path: Path) -> None:
    if zip_path.exists():
        print(f"Using cached download: {zip_path}")
        return
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SciFact from {SCIFACT_URL}")
    urllib.request.urlretrieve(SCIFACT_URL, zip_path)


def extract_if_needed(zip_path: Path, extract_dir: Path) -> Path:
    dataset_dir = extract_dir / "scifact"
    if dataset_dir.exists():
        print(f"Using extracted dataset: {dataset_dir}")
        return dataset_dir
    print(f"Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    return dataset_dir


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_qrels(dataset_dir: Path, split: str) -> dict[str, dict[str, int]]:
    preferred = dataset_dir / "qrels" / f"{split}.tsv"
    if not preferred.exists():
        available = sorted((dataset_dir / "qrels").glob("*.tsv"))
        if not available:
            raise FileNotFoundError(f"No qrels TSV files under {dataset_dir / 'qrels'}")
        preferred = available[0]

    qrels: dict[str, dict[str, int]] = {}
    with preferred.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            query_id = row.get("query-id") or row.get("query_id") or row.get("query")
            corpus_id = row.get("corpus-id") or row.get("corpus_id") or row.get("docid")
            score_text = row.get("score") or row.get("relevance") or "1"
            if query_id is None or corpus_id is None:
                raise ValueError(f"Unexpected qrels columns in {preferred}: {reader.fieldnames}")
            score = int(float(score_text))
            if score <= 0:
                continue
            qrels.setdefault(str(query_id), {})[str(corpus_id)] = score
    print(f"Loaded qrels: {preferred}")
    return qrels


def document_text(record: dict) -> str:
    title = (record.get("title") or "").strip()
    text = (record.get("text") or "").strip()
    if title and text:
        return f"{title}. {text}"
    return title or text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-documents", type=int, default=1000)
    parser.add_argument("--max-queries", type=int, default=50)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    zip_path = RAW_DIR / "scifact.zip"
    download_if_needed(zip_path)
    dataset_dir = extract_if_needed(zip_path, RAW_DIR)

    corpus_records = load_jsonl(dataset_dir / "corpus.jsonl")
    query_records = load_jsonl(dataset_dir / "queries.jsonl")
    qrels_all = load_qrels(dataset_dir, args.split)

    corpus_by_id = {str(record["_id"]): record for record in corpus_records}
    queries_with_qrels = [
        record for record in query_records if str(record["_id"]) in qrels_all
    ]
    selected_queries = queries_with_qrels[: args.max_queries]
    selected_query_ids = [str(record["_id"]) for record in selected_queries]

    relevant_doc_ids: list[str] = []
    seen_relevant = set()
    for query_id in selected_query_ids:
        for doc_id in qrels_all[query_id]:
            if doc_id in corpus_by_id and doc_id not in seen_relevant:
                relevant_doc_ids.append(doc_id)
                seen_relevant.add(doc_id)

    selected_doc_ids = list(relevant_doc_ids)
    for record in corpus_records:
        doc_id = str(record["_id"])
        if doc_id in seen_relevant:
            continue
        selected_doc_ids.append(doc_id)
        if len(selected_doc_ids) >= args.max_documents:
            break

    selected_doc_id_set = set(selected_doc_ids)
    qrels = {
        query_id: [
            doc_id
            for doc_id in qrels_all[query_id]
            if doc_id in selected_doc_id_set
        ]
        for query_id in selected_query_ids
    }
    qrels = {query_id: doc_ids for query_id, doc_ids in qrels.items() if doc_ids}
    selected_queries = [
        record for record in selected_queries if str(record["_id"]) in qrels
    ]

    documents = [
        {
            "id": doc_id,
            "title": (corpus_by_id[doc_id].get("title") or "").strip(),
            "text": document_text(corpus_by_id[doc_id]),
        }
        for doc_id in selected_doc_ids
    ]
    queries = [
        {
            "id": str(record["_id"]),
            "text": (record.get("text") or "").strip(),
        }
        for record in selected_queries
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "documents.jsonl").open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(json.dumps(document, ensure_ascii=True) + "\n")
    with (OUTPUT_DIR / "queries.jsonl").open("w", encoding="utf-8") as file:
        for query in queries:
            file.write(json.dumps(query, ensure_ascii=True) + "\n")
    save_json(OUTPUT_DIR / "qrels.json", qrels)
    save_json(
        OUTPUT_DIR / "metadata.json",
        {
            "dataset": "BEIR SciFact",
            "source_url": SCIFACT_URL,
            "split": args.split,
            "documents": len(documents),
            "queries": len(queries),
            "qrels_labels": sum(len(doc_ids) for doc_ids in qrels.values()),
            "max_documents": args.max_documents,
            "max_queries": args.max_queries,
            "selection": (
                "Selected first queries with qrels, included all relevant docs "
                "for them, then filled remaining documents with deterministic "
                "corpus-order distractors."
            ),
        },
    )

    print("Prepared SciFact benchmark subset")
    print(f"documents: {len(documents)}")
    print(f"queries: {len(queries)}")
    print(f"qrels labels: {sum(len(doc_ids) for doc_ids in qrels.values())}")
    print(f"output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
