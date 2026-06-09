"""Create a deterministic local corpus for ColBERT/PDX experiments.

The corpus is intentionally synthetic but human-readable. It mixes project
topics with unrelated distractors and includes small manual relevance labels.
No external downloads or random generation are used.
"""

from __future__ import annotations

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils_colbert import save_json


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "local_corpus"
DOCUMENTS_PATH = OUTPUT_DIR / "documents.jsonl"
QUERIES_PATH = OUTPUT_DIR / "queries.jsonl"
QRELS_PATH = OUTPUT_DIR / "qrels.json"


TOPICS = {
    "colbert": {
        "name": "ColBERT and late interaction",
        "sentences": [
            "ColBERT stores a passage as many token vectors and scores it with late interaction MaxSim.",
            "A query token in ColBERT searches for its strongest matching document token before scores are summed.",
            "Late interaction keeps token-level evidence instead of compressing a document into one dense vector.",
            "ColBERT retrieval often uses approximate search to collect token matches before document scoring.",
            "MaxSim scoring rewards documents that contain strong matches for several query token embeddings.",
            "Token vectors make ColBERT more expressive than a single pooled embedding for each passage.",
            "A ColBERT index must preserve the mapping from token vectors back to their source document.",
            "The late interaction score is a sum over query-token maxima against document-token vectors.",
            "ColBERT can rerank candidates by computing exact token-level similarities for each document.",
            "Multi-vector retrieval increases index size because every document contributes many token embeddings.",
            "Query embeddings in ColBERT are also token-level vectors used independently during matching.",
            "A flat token-vector index is a simple first representation for ColBERT-style document embeddings.",
            "ColBERT-style search needs aggregation after token retrieval to produce document rankings.",
            "Exact ColBERT scoring compares every query token against every document token in a candidate passage.",
            "Late interaction separates candidate generation from final document-level MaxSim scoring.",
        ],
    },
    "pylate": {
        "name": "PyLate and embedding export",
        "sentences": [
            "PyLate provides a convenient ColBERT implementation for quick retrieval baselines.",
            "The PyLate model can encode documents into variable-length token-vector matrices.",
            "Exporting PyLate embeddings as packed arrays keeps values and offsets reproducible.",
            "A packed embedding file stores all token vectors in one matrix and reconstructs items with offsets.",
            "Using the same PyLate model across experiments keeps the ColBERT baseline consistent.",
            "PyLate document embeddings are not single vectors, so downstream indexes need token metadata.",
            "The export script records document identifiers so token hits can be mapped back to passages.",
            "Query embeddings from PyLate are shorter token-vector matrices than document embeddings.",
            "A local PyLate export lets the project test retrieval code without re-encoding every run.",
            "The GTE ModernColBERT model is used as the PyLate-compatible baseline in this project.",
            "PyLate helps create a reproducible bridge between text passages and ColBERT token vectors.",
            "The embedding export records shapes to verify variable-length multi-vector structure.",
            "CPU PyLate runs are slower than GPU runs but enough for small local smoke experiments.",
            "Document and query text should be saved beside embeddings for debugging ranking results.",
            "PyLate encoding is the only neural model step in the current local pipeline.",
        ],
    },
    "pdx": {
        "name": "PDX and vector layout",
        "sentences": [
            "PDX stores vectors in a dimension-oriented layout to improve locality during similarity search.",
            "A PDX block groups vectors so dimensions can be scanned in a cache-friendly order.",
            "Dimension-oriented layout can help search algorithms prune or skip unnecessary vector work.",
            "PDXearch exposes Python wrappers around compiled C++ vector search indexes.",
            "The current Python PDX API supports squared L2 distance for the tested BOND index.",
            "PDX can materialize a NumPy matrix into a byte layout consumed by the compiled extension.",
            "Flat PDX indexing treats every row of the input matrix as an independent vector.",
            "For ColBERT experiments, each document token vector can become one PDX row.",
            "The PDX token index needs side metadata because PDX itself returns vector ids, not document scores.",
            "PDX examples include random vector search and larger HDF5 benchmark scripts.",
            "The PDX sigmod branch contains the BOND searcher and Python index factory wrappers.",
            "Building PDX is easier in native Linux storage than through a mounted Windows checkout.",
            "Squared L2 on normalized vectors can stand in for cosine similarity in the PDX API.",
            "PDX is a layout and search baseline, not a complete ColBERT scoring system by itself.",
            "A PDX token-vector index can generate document candidates before exact MaxSim reranking.",
        ],
    },
    "bond": {
        "name": "BOND branch and bound",
        "sentences": [
            "BOND accelerates nearest-neighbor search by scanning vector dimensions and pruning candidates.",
            "The branch-and-bound idea maintains distance bounds while dimensions are processed incrementally.",
            "A candidate can be eliminated when its partial distance can no longer enter the top-k results.",
            "BOND was designed for vertically decomposed data where dimensions are stored separately.",
            "Dimension ordering can affect how quickly BOND identifies candidates that should be pruned.",
            "PDX-BOND combines the PDX layout with BOND-style exact search over vector dimensions.",
            "For exact L2 search, BOND should return the same nearest neighbors as brute-force scanning.",
            "The PDX-BOND smoke test compares compiled search results against NumPy squared L2 distances.",
            "Branch-and-bound search is most useful when bounds become tight before all dimensions are scanned.",
            "BOND-style pruning is related to avoiding full distance computations for unpromising vectors.",
            "A flat ColBERT token index can use BOND to retrieve nearest token vectors for each query token.",
            "BOND does not directly compute the document-level MaxSim aggregation used by ColBERT.",
            "The candidate-generation role of BOND is to find promising token matches quickly.",
            "Exact PDX-BOND over normalized vectors can produce cosine-equivalent token neighbors.",
            "Small correctness tests should precede any attempt to benchmark BOND at scale.",
        ],
    },
    "adsampling": {
        "name": "ADSampling",
        "sentences": [
            "ADSampling speeds up distance comparisons by sampling vector dimensions adaptively.",
            "Adaptive dimension sampling can stop comparing a vector once the result is unlikely to change.",
            "The ADSampling baseline is useful related work for pruning distance computations.",
            "In PDX examples, ADSampling appears together with IVF search and random vector data.",
            "Dimension sampling and branch-and-bound both reduce unnecessary vector arithmetic.",
            "ADSampling can be compared with PDX-BOND as another way to avoid full comparisons.",
            "Approximate distance estimates need careful validation against exact nearest-neighbor results.",
            "Sampling dimensions adaptively may work differently for normalized token vectors than raw vectors.",
            "The project should record whether ADSampling changes rankings or only speeds comparisons.",
            "ADSampling is not yet connected to the ColBERT token-vector experiments.",
            "A future baseline could compare token retrieval using ADSampling and PDX-BOND.",
            "Adaptive sampling is most attractive when high-dimensional vectors are expensive to compare.",
            "ADSampling chooses dimensions based on evidence accumulated during the comparison.",
            "The current iteration only documents ADSampling as related vector-search work.",
            "Distance approximation methods require recall measurements against an exact reference.",
        ],
    },
    "corect": {
        "name": "CoRECT and retrieval evaluation",
        "sentences": [
            "CoRECT can provide an evaluation framework for retrieval and embedding experiments.",
            "Retrieval evaluation needs query relevance labels, rankings, and metrics such as recall and MRR.",
            "A small qrels file lets local experiments compare rankings against expected relevant passages.",
            "CoRECT is a later-stage tool for organizing dense retrieval evaluations more systematically.",
            "Manual relevance labels are enough for early local smoke tests before using larger benchmarks.",
            "Evaluation should separate ranking agreement with exact MaxSim from relevance against qrels.",
            "A candidate generator can have high recall even if its approximate ranking is imperfect.",
            "MRR@10 measures whether a relevant document appears early in the result list.",
            "Recall@k measures whether known relevant documents appear in the first k retrieved documents.",
            "CoRECT-style experiments should keep data, queries, qrels, and metrics reproducible.",
            "The current local corpus uses handcrafted qrels instead of downloaded benchmark labels.",
            "Evaluation logs should preserve commands, timings, and any failed setup attempts.",
        ],
    },
    "dense": {
        "name": "Dense retrieval",
        "sentences": [
            "Dense retrieval maps text into vectors so nearest-neighbor search can find related passages.",
            "A single-vector retriever compresses a passage into one embedding before indexing.",
            "Multi-vector retrievers keep several embeddings per passage to preserve token-level evidence.",
            "ANN indexes often trade exact ranking for faster candidate generation on large corpora.",
            "Reranking can recover quality by applying a more expensive scoring function to candidates.",
            "Embedding dimensions, distance metrics, and normalization choices affect retrieval behavior.",
            "Candidate generation is useful when full exact scoring against every document is too slow.",
            "A dense retrieval pipeline often has indexing, search, aggregation, and evaluation stages.",
            "Vector search systems need reproducible baselines before optimization work begins.",
            "Local dense retrieval tests should use small corpora before moving to public benchmarks.",
            "Exact scoring provides a reference for measuring candidate recall and ranking loss.",
            "Embedding experiments should report both retrieval metrics and runtime measurements.",
        ],
    },
    "cooking": {
        "name": "Cooking distractors",
        "sentences": [
            "A soup recipe can start with onions, carrots, celery, and a slow simmered broth.",
            "Baking bread requires flour, water, yeast, salt, and enough time for fermentation.",
            "A sharp knife and stable cutting board make vegetable preparation easier and safer.",
            "Pasta sauce can be improved by reducing tomatoes with garlic and olive oil.",
            "Roasting vegetables caramelizes their edges and concentrates their flavor.",
            "A kitchen scale helps measure ingredients more consistently than cups.",
            "Stir frying works best when ingredients are cut evenly before the pan is hot.",
            "Marinating chicken with herbs and lemon can add flavor before grilling.",
            "A stew becomes richer when it cooks gently for several hours.",
            "Fresh herbs should often be added near the end of cooking to keep their aroma.",
            "Rice needs the right water ratio and a covered pot for even steaming.",
            "Chocolate cake can dry out if it is baked too long.",
            "Salad dressing balances acidity, oil, salt, and a small amount of sweetness.",
            "A cast iron pan holds heat well for searing meat and vegetables.",
            "Meal planning can reduce food waste during a busy week.",
        ],
    },
    "football": {
        "name": "Football distractors",
        "sentences": [
            "The football team pressed high up the pitch to force quick turnovers.",
            "A goalkeeper organized the defensive line before the free kick.",
            "The midfielder switched play with a long diagonal pass to the wing.",
            "A counterattack can be dangerous when the opponent commits many players forward.",
            "The coach adjusted the formation after halftime to protect the lead.",
            "Set pieces often decide close football matches.",
            "A striker needs good timing to stay onside during a through ball.",
            "The fullback overlapped on the outside to create width in attack.",
            "A compact defensive block can make central passes difficult.",
            "The referee added stoppage time after several substitutions.",
            "Fitness and recovery affect a team's pressing intensity late in the match.",
            "A clean first touch helps a player escape pressure in midfield.",
            "Penalty shootouts test technique and composure after a tied match.",
            "The away supporters celebrated after a late winning goal.",
            "Video analysis helps teams prepare for an opponent's tactical patterns.",
        ],
    },
    "travel": {
        "name": "Travel distractors",
        "sentences": [
            "A train journey through the mountains can offer wide views of valleys and lakes.",
            "Packing light makes it easier to move through airports and city streets.",
            "Travelers should check visa rules before booking an international trip.",
            "A walking tour can reveal small details of a historic neighborhood.",
            "Off-season travel often means lower prices and quieter museums.",
            "A good itinerary leaves time for rest between major sights.",
            "Local transit cards can simplify bus and metro travel in large cities.",
            "A seaside town may be calm in the morning and busy by sunset.",
            "Travel insurance can help when flights are delayed or luggage is lost.",
            "A map of nearby restaurants is useful after a late arrival.",
            "Booking accommodation near public transport can reduce daily travel time.",
            "Museums sometimes require timed tickets during peak tourist seasons.",
            "Hiking routes should be checked for weather and trail conditions.",
            "A reusable water bottle is useful on long sightseeing days.",
            "Language phrasebooks can help with greetings and simple questions.",
        ],
    },
    "weather": {
        "name": "Weather distractors",
        "sentences": [
            "The weather forecast predicted rain in the morning and clear skies by evening.",
            "A cold front can bring lower temperatures and stronger wind.",
            "Humidity makes summer heat feel more intense than the thermometer suggests.",
            "Meteorologists track pressure systems to understand changing weather patterns.",
            "A thunderstorm developed quickly after warm air rose over the hills.",
            "Climate observations require consistent measurements over many years.",
            "Fog can reduce visibility for drivers during early morning commutes.",
            "Snowfall totals vary when the freezing line moves during a storm.",
            "Weather radar helps detect bands of heavy precipitation.",
            "A heat advisory warns people to avoid strenuous activity in the afternoon.",
            "Coastal breezes can keep temperatures lower near the water.",
            "Drought conditions worsen when rainfall stays below average for months.",
            "A barometer can show pressure changes before a storm arrives.",
            "Seasonal forecasts describe broad tendencies rather than exact daily weather.",
            "Wind chill estimates how cold air feels on exposed skin.",
        ],
    },
    "history": {
        "name": "History distractors",
        "sentences": [
            "Medieval trade routes connected towns through markets, rivers, and coastal ports.",
            "Historians compare written records with archaeology to understand past societies.",
            "The printing press helped ideas circulate more quickly across Europe.",
            "Ancient city walls show how communities prepared for military threats.",
            "A museum archive can preserve letters, maps, and official documents.",
            "Industrialization changed labor, transportation, and urban life.",
            "Oral histories capture personal memories that may not appear in official records.",
            "A treaty can reshape borders and political alliances after a conflict.",
            "Historical timelines help organize events by cause and consequence.",
            "Maritime exploration depended on navigation tools and shipbuilding knowledge.",
            "Census records can reveal demographic change across generations.",
            "Public monuments often reflect the values of the period in which they were built.",
            "Agricultural innovations changed food production and settlement patterns.",
            "Primary sources require context before historians draw conclusions from them.",
            "The study of history often combines social, economic, and political evidence.",
        ],
    },
}


QUERIES = [
    {
        "id": "q01",
        "text": "How does ColBERT score documents with late interaction token vectors?",
        "relevant_topic": "colbert",
    },
    {
        "id": "q02",
        "text": "What does PDX change about vector layout for search?",
        "relevant_topic": "pdx",
    },
    {
        "id": "q03",
        "text": "How does BOND use branch and bound for exact nearest neighbors?",
        "relevant_topic": "bond",
    },
    {
        "id": "q04",
        "text": "Why can ADSampling speed up distance comparisons?",
        "relevant_topic": "adsampling",
    },
    {
        "id": "q05",
        "text": "How can CoRECT evaluate retrieval embedding experiments?",
        "relevant_topic": "corect",
    },
    {
        "id": "q06",
        "text": "How are PyLate ColBERT embeddings exported for indexing experiments?",
        "relevant_topic": "pylate",
    },
    {
        "id": "q07",
        "text": "What is dense retrieval with embeddings and reranking?",
        "relevant_topic": "dense",
    },
    {
        "id": "q08",
        "text": "Which passages are about cooking recipes and kitchen preparation?",
        "relevant_topic": "cooking",
    },
    {
        "id": "q09",
        "text": "Which documents discuss football matches and team tactics?",
        "relevant_topic": "football",
    },
    {
        "id": "q10",
        "text": "Which passages describe weather forecasts and climate observations?",
        "relevant_topic": "weather",
    },
]


def make_documents() -> list[dict[str, str]]:
    documents = []
    for topic_id, topic in TOPICS.items():
        for index, sentence in enumerate(topic["sentences"], start=1):
            doc_id = f"{topic_id}-{index:03d}"
            text = f"{sentence} Topic: {topic['name']}."
            documents.append({"id": doc_id, "topic": topic_id, "text": text})
    return documents


def make_qrels() -> dict[str, list[str]]:
    qrels = {}
    for query in QUERIES:
        topic_id = query["relevant_topic"]
        qrels[query["id"]] = [f"{topic_id}-{index:03d}" for index in range(1, 6)]
    return qrels


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    documents = make_documents()
    queries = [
        {"id": query["id"], "text": query["text"], "relevant_topic": query["relevant_topic"]}
        for query in QUERIES
    ]
    qrels = make_qrels()

    write_jsonl(DOCUMENTS_PATH, documents)
    write_jsonl(QUERIES_PATH, queries)
    save_json(QRELS_PATH, qrels)

    print("Created local ColBERT corpus")
    print(f"documents: {len(documents)}")
    print(f"queries: {len(queries)}")
    print(f"qrels: {len(qrels)} queries, {sum(len(ids) for ids in qrels.values())} labels")
    print(f"documents path: {DOCUMENTS_PATH}")
    print(f"queries path: {QUERIES_PATH}")
    print(f"qrels path: {QRELS_PATH}")


if __name__ == "__main__":
    main()
