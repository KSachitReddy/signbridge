"""
Document manager: orchestrates upload → parse → index → query lifecycle.
All document files are stored under database/docs/.
"""

from __future__ import annotations

import os
import shutil
from typing import Optional

from modules.database import (
    add_document,
    add_doc_sections,
    get_doc_sections,
    get_all_documents,
    delete_document,
    add_doc_query,
    get_doc_queries,
)
from modules.docs.parser import parse_document
from modules.docs.retriever import retrieve_sections, build_context
from modules.docs.llm import answer

DOCS_STORE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "database", "docs")
)


def _ensure_store():
    os.makedirs(DOCS_STORE_DIR, exist_ok=True)


def upload_and_index(
    source_path: str,
    filename: Optional[str] = None,
    max_chars: int = 2000,
) -> dict:
    """
    Copy file to the docs store, parse it into sections, persist to DB.
    Returns {"doc_id": int, "filename": str, "section_count": int, "page_count": int}.
    """
    _ensure_store()
    if filename is None:
        filename = os.path.basename(source_path)
    dest_path = os.path.join(DOCS_STORE_DIR, filename)

    # Avoid overwriting: suffix with _N if name collides
    base, ext = os.path.splitext(filename)
    counter = 1
    while os.path.exists(dest_path):
        dest_path = os.path.join(DOCS_STORE_DIR, f"{base}_{counter}{ext}")
        counter += 1
    shutil.copy2(source_path, dest_path)

    sections, page_count = parse_document(dest_path, max_chars=max_chars)
    file_size = os.path.getsize(dest_path)
    stored_name = os.path.basename(dest_path)

    doc_id = add_document(
        filename=stored_name,
        filepath=dest_path,
        file_size=file_size,
        page_count=page_count,
    )
    add_doc_sections(doc_id, sections)

    return {
        "doc_id": doc_id,
        "filename": stored_name,
        "section_count": len(sections),
        "page_count": page_count,
    }


def list_documents() -> list[dict]:
    """Return all indexed documents from the DB."""
    return get_all_documents()


def remove_document(doc_id: int) -> None:
    """Delete document record from DB and its file from disk."""
    docs = get_all_documents()
    filepath = next((d["filepath"] for d in docs if d["id"] == doc_id), None)
    delete_document(doc_id)
    if filepath and os.path.isfile(filepath):
        try:
            os.remove(filepath)
        except OSError:
            pass


def query_document(
    doc_id: int,
    question: str,
    top_k: int = 3,
    model_path: Optional[str] = None,
    use_ollama_fallback: bool = True,
    ollama_model: str = "llama3.2",
) -> dict:
    """
    Retrieve top_k sections from doc_id, build context, run LLM, persist query.
    Returns {"answer": str, "sources": list[dict]}.
    """
    sections = get_doc_sections(doc_id)
    if not sections:
        return {"answer": "No content found for this document.", "sources": []}

    retrieved = retrieve_sections(question, sections, top_k=top_k)
    context = build_context(retrieved)

    response = answer(
        question=question,
        context=context,
        model_path=model_path,
        use_ollama_fallback=use_ollama_fallback,
        ollama_model=ollama_model,
    )

    sources = [
        {"heading": r.get("heading", ""), "score": r.get("score", 0.0), "rank": r.get("rank", 0)}
        for r in retrieved
    ]
    add_doc_query(doc_id, question, response, sources)

    return {"answer": response, "sources": sources}


def get_query_history(doc_id: Optional[int] = None) -> list[dict]:
    """Return past Q&A records, optionally filtered by doc_id."""
    return get_doc_queries(doc_id)
