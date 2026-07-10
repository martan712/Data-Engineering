"""BM25 lexical baseline over the BEIR corpus text.

Single responsibility: index the raw document text of a BEIR dataset with BM25
and retrieve ranked BEIR doc ids per query text.  Stage 5 e03 uses this as the
standard lexical quality reference next to the MaxSim arms: it shares nothing
with the embedding stack, so it anchors the absolute qrels numbers.

Two deliberate consistency choices with the neural arms:
- Documents are indexed TEXT-ONLY (no title concatenation), matching how the
  archive cache encoded the embeddings (see the e01 embedding_note); published
  BEIR BM25 numbers typically index title+text and are not directly
  comparable.
- Queries and qrels are the same evaluable test set the other arms use.

Implementation: bm25s (Lucene scoring, k1=1.5, b=0.75) with English stopword
removal and Snowball stemming via PyStemmer.  Requires the [retrieval] extra.
"""

from __future__ import annotations


class BM25Baseline:
    """BM25 retrieval over a list of (doc_id, text) pairs."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.retriever = None
        self.doc_ids: list[str] | None = None
        self._stemmer = None

    def _tokenize(self, texts: list[str]):
        import bm25s

        return bm25s.tokenize(texts, stopwords="en", stemmer=self._stemmer,
                              show_progress=False)

    def build(self, doc_ids: list[str], texts: list[str]) -> dict:
        """Index the corpus; returns build stats.

        Parameters
        ----------
        doc_ids : BEIR corpus ids, aligned with texts
        texts   : raw document text (text-only, no title concatenation)
        """
        import Stemmer
        import bm25s

        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must be aligned")
        self._stemmer = Stemmer.Stemmer("english")
        self.doc_ids = list(doc_ids)
        self.retriever = bm25s.BM25(k1=self.k1, b=self.b)
        self.retriever.index(self._tokenize(texts), show_progress=False)
        return {"n_docs": len(doc_ids), "k1": self.k1, "b": self.b,
                "scoring": "lucene", "stopwords": "en",
                "stemmer": "snowball-english"}

    def search(self, query_texts: list[str], k: int
               ) -> list[tuple[list[str], list[float]]]:
        """Retrieve the top-k BEIR doc ids per query, scores descending.

        Returns
        -------
        one (doc_ids, scores) pair per query, as the eval layer expects
        """
        if self.retriever is None:
            raise RuntimeError("call build() first")
        idxs, scores = self.retriever.retrieve(
            self._tokenize(query_texts), k=k, show_progress=False)
        return [([self.doc_ids[int(d)] for d in idxs[q]],
                 [float(s) for s in scores[q]])
                for q in range(len(query_texts))]
