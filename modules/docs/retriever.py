"""
Section retriever: given a query string and a list of sections, return
the top-k most relevant sections using BM25 (rank_bm25).
Falls back to TF-IDF-style keyword overlap if rank_bm25 is not installed.
"""

from __future__ import annotations

import re
from typing import Optional


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z]{2,}\b", text.lower())


def _stopwords() -> set[str]:
    return {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "not", "no", "this", "that",
        "these", "those", "it", "its", "by", "from", "as", "if", "then",
        "than", "so", "we", "i", "you", "he", "she", "they", "their", "our",
    }


def _clean_tokens(tokens: list[str]) -> list[str]:
    stops = _stopwords()
    return [t for t in tokens if t not in stops and len(t) > 2]


def retrieve_sections(
    query: str,
    sections: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """
    Rank sections by relevance to query and return top_k with scores.

    Each returned dict is the original section dict extended with:
      - "score": float relevance score
      - "rank": int (1-based)
    """
    if not sections:
        return []

    query_tokens = _clean_tokens(_tokenize(query))
    if not query_tokens:
        return [dict(s, score=0.0, rank=i + 1) for i, s in enumerate(sections[:top_k])]

    corpus = [
        _clean_tokens(_tokenize((s.get("heading", "") + " " + s["content"])))
        for s in sections
    ]

    try:
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_tokens).tolist()
    except ImportError:
        # Simple overlap fallback
        query_set = set(query_tokens)
        scores = [
            len(query_set & set(doc_tokens)) / max(len(query_set), 1)
            for doc_tokens in corpus
        ]

    ranked = sorted(
        enumerate(scores), key=lambda x: x[1], reverse=True
    )[:top_k]

    results = []
    for rank, (idx, score) in enumerate(ranked, start=1):
        entry = dict(sections[idx])
        entry["score"] = round(float(score), 4)
        entry["rank"] = rank
        results.append(entry)

    return results


def build_context(retrieved: list[dict], max_chars: int = 3000) -> str:
    """
    Concatenate retrieved sections into a single context string for the LLM.
    Respects max_chars budget.
    """
    parts: list[str] = []
    total = 0
    for sec in retrieved:
        heading = sec.get("heading", "")
        content = sec.get("content", "")
        block = f"### {heading}\n{content}" if heading else content
        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                parts.append(block[:remaining])
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)
