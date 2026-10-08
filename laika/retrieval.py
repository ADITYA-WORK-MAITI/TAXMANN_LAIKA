"""Hybrid retrieval: dense vectors (FAISS) and keywords (BM25), merged with RAG Fusion.

For each question the model writes a few rephrasings. Every rephrasing is run
against both indexes. All the ranked lists are then merged with Reciprocal Rank
Fusion, so a chunk that ranks well across many lists comes out on top.
"""

import re
from collections.abc import Callable, Sequence

import faiss
import numpy as np
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from laika.config import settings

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = 60) -> list[int]:
    """Merge several ranked lists of ids into one list, best first.

    Each id gets a score of 1 / (k + rank) from every list it appears in.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.__getitem__, reverse=True)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a), np.asarray(b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator else 0.0


class HybridIndex:
    """Searchable index over the text chunks of the uploaded documents."""

    def __init__(self, chunks: list[str], embeddings: Embeddings):
        if not chunks:
            raise ValueError("Cannot build an index without any text.")
        self.chunks = chunks
        self.embeddings = embeddings
        vectors = self._normalize(embeddings.embed_documents(chunks))
        self.vector_index = faiss.IndexFlatIP(vectors.shape[1])  # inner product of unit vectors = cosine
        self.vector_index.add(vectors)
        self.bm25 = BM25Okapi([tokenize(chunk) for chunk in chunks])

    @staticmethod
    def _normalize(vectors: Sequence[Sequence[float]]) -> np.ndarray:
        array = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(array)
        return array

    def vector_ranking(self, query: str, top_k: int) -> list[int]:
        query_vector = self._normalize([self.embeddings.embed_query(query)])
        _, ids = self.vector_index.search(query_vector, min(top_k, len(self.chunks)))
        return [int(i) for i in ids[0] if i >= 0]

    def keyword_ranking(self, query: str, top_k: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = np.argsort(scores)[::-1][:top_k]
        return [int(i) for i in ranked if scores[i] > 0]

    def search(self, queries: Sequence[str], top_k: int = settings.top_k) -> list[str]:
        """Return the `top_k` best chunks for a group of related queries."""
        rankings = []
        for query in queries:
            rankings.append(self.vector_ranking(query, top_k))
            rankings.append(self.keyword_ranking(query, top_k))
        return [self.chunks[i] for i in reciprocal_rank_fusion(rankings)[:top_k]]


def parse_query_variants(reply: str, limit: int) -> list[str]:
    """Turn a model reply with one query per line into a clean list."""
    lines = (re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip() for line in reply.splitlines())
    return [line for line in lines if line][:limit]


def generate_query_variants(question: str, ask: Callable[[str], str]) -> list[str]:
    """Return the question plus a few rephrasings written by the model."""
    n = settings.query_variants
    prompt = (
        f"Write {n} different search queries that would help answer the question below. "
        f"Return one query per line with no numbering or extra text.\n\nQuestion: {question}"
    )
    try:
        variants = parse_query_variants(ask(prompt), n)
    except Exception:
        variants = []
    return [question, *variants]
