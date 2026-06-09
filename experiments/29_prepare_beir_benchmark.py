"""Prepare a BEIR benchmark subset (dataset- and size-agnostic).

Generalizes experiment 19 so any public BEIR dataset can be turned into a
corpus/queries/qrels triple at an arbitrary size. Used for the CoRECT/BEIR
scale-up: e.g. the full SciFact corpus keeps the same queries but adds all
remaining documents as distractors, isolating the "corpus complexity" axis.

Examples
--------
Full SciFact corpus, first 50 test queries with qrels:

    python 29_prepare_beir_benchmark.py --dataset scifact --max-documents 0 --max-queries 50 --name scifact_full

A 3000-document SciFact subset:

    python 29_prepare_beir_benchmark.py --dataset scifact --max-documents 3000 --max-queries 50 --name scifact_3000
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
import zipfile
from pathlib import Path

from utils_colbert import save_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "artifacts" / "raw" / "beir"
BEIR_URL_TEMPLATE = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
)


def download_if_needed(zip_path: Path, url: str) -> None:
    if zip_path.exists():
        print(f"Using cached download: {zip_path}")
        return
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from {url}")
    urllib.request.urlretrieve(url, zip_path)


def extract_if_needed(zip_path: Path, extract_dir: Path, dataset: str) -> Path:
    dataset_dir = extract_dir / dataset
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="scifact", help="BEIR dataset name")
    parser.add_argument("--url", default=None, help="Override the dataset zip URL")
    parser.add_argument(
        "--max-documents",
        type=int,
        default=0,
        help="Maximum documents to keep (0 = full corpus).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=50,
        help="Maximum queries with qrels to keep (0 = all).",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--name",
        default=None,
        help="Output folder stem; defaults to '<dataset>_full' or '<dataset>_<max_documents>'.",
    )
    args = parser.parse_args()

    url = args.url or BEIR_URL_TEMPLATE.format(dataset=args.dataset)
    zip_path = RAW_DIR / f"{args.dataset}.zip"
    download_if_needed(zip_path, url)
    dataset_dir = extract_if_needed(zip_path, RAW_DIR, args.dataset)

    corpus_records = load_jsonl(dataset_dir / "corpus.jsonl")
    query_records = load_jsonl(dataset_dir / "queries.jsonl")
    qrels_all = load_qrels(dataset_dir, args.split)

    corpus_by_id = {str(record["_id"]): record for record in corpus_records}
    queries_with_qrels = [record for record in query_records if str(record["_id"]) in qrels_all]
    if args.max_queries > 0:
        queries_with_qrels = queries_with_qrels[: args.max_queries]
    selected_query_ids = [str(record["_id"]) for record in queries_with_qrels]

    relevant_doc_ids: list[str] = []
    seen_relevant = set()
    for query_id in selected_query_ids:
        for doc_id in qrels_all[query_id]:
            if doc_id in corpus_by_id and doc_id not in seen_relevant:
                relevant_doc_ids.append(doc_id)
                seen_relevant.add(doc_id)

    # max_documents == 0 means keep the entire corpus.
    document_budget = args.max_documents if args.max_documents > 0 else len(corpus_records)
    selected_doc_ids = list(relevant_doc_ids)
    for record in corpus_records:
        if len(selected_doc_ids) >= document_budget:
            break
        doc_id = str(record["_id"])
        if doc_id in seen_relevant:
            continue
        selected_doc_ids.append(doc_id)

    selected_doc_id_set = set(selected_doc_ids)
    qrels = {
        query_id: [doc_id for doc_id in qrels_all[query_id] if doc_id in selected_doc_id_set]
        for query_id in selected_query_ids
    }
    qrels = {query_id: doc_ids for query_id, doc_ids in qrels.items() if doc_ids}
    selected_queries = [record for record in queries_with_qrels if str(record["_id"]) in qrels]

    documents = [
        {
            "id": doc_id,
            "title": (corpus_by_id[doc_id].get("title") or "").strip(),
            "text": document_text(corpus_by_id[doc_id]),
        }
        for doc_id in selected_doc_ids
    ]
    queries = [
        {"id": str(record["_id"]), "text": (record.get("text") or "").strip()}
        for record in selected_queries
    ]

    stem = args.name or (
        f"{args.dataset}_full" if args.max_documents == 0 else f"{args.dataset}_{args.max_documents}"
    )
    output_dir = PROJECT_ROOT / "artifacts" / f"{stem}_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "documents.jsonl").open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(json.dumps(document, ensure_ascii=True) + "\n")
    with (output_dir / "queries.jsonl").open("w", encoding="utf-8") as file:
        for query in queries:
            file.write(json.dumps(query, ensure_ascii=True) + "\n")
    save_json(output_dir / "qrels.json", qrels)
    save_json(
        output_dir / "metadata.json",
        {
            "dataset": f"BEIR {args.dataset}",
            "source_url": url,
            "split": args.split,
            "documents": len(documents),
            "queries": len(queries),
            "qrels_labels": sum(len(doc_ids) for doc_ids in qrels.values()),
            "relevant_documents": len(relevant_doc_ids),
            "max_documents": args.max_documents,
            "max_queries": args.max_queries,
            "full_corpus": args.max_documents == 0,
            "selection": (
                "Selected first queries with qrels, included all their relevant "
                "docs, then filled remaining documents with corpus-order "
                "distractors up to the document budget."
            ),
        },
    )

    print(f"Prepared BEIR {args.dataset} benchmark: {stem}")
    print(f"documents: {len(documents)}  (relevant: {len(relevant_doc_ids)})")
    print(f"queries: {len(queries)}")
    print(f"qrels labels: {sum(len(doc_ids) for doc_ids in qrels.values())}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
