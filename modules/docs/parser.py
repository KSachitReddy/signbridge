"""
Document parser: converts PDF/text files into a list of section dicts.
Uses pymupdf4llm for PDF extraction (produces Markdown), then splits on
Markdown headings to create retrievable section chunks.
"""

from __future__ import annotations

import os
import re
from typing import Optional


def _split_markdown_sections(md_text: str) -> list[dict]:
    """Split Markdown text on headings (# / ## / ###) into section dicts."""
    heading_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    sections: list[dict] = []
    matches = list(heading_pattern.finditer(md_text))

    if not matches:
        # No headings — treat entire document as one section
        content = md_text.strip()
        if content:
            sections.append({"heading": "", "content": content})
        return sections

    # Text before the first heading
    preamble = md_text[: matches[0].start()].strip()
    if preamble:
        sections.append({"heading": "", "content": preamble})

    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        content = md_text[start:end].strip()
        if content:
            sections.append({"heading": heading, "content": content})

    return sections


def _chunk_large_sections(sections: list[dict], max_chars: int = 2000) -> list[dict]:
    """Split sections whose content exceeds max_chars into smaller pieces."""
    out: list[dict] = []
    for sec in sections:
        content = sec["content"]
        if len(content) <= max_chars:
            out.append(sec)
            continue
        # Split on paragraph boundaries
        paragraphs = re.split(r"\n{2,}", content)
        chunk = ""
        part = 0
        for para in paragraphs:
            if len(chunk) + len(para) + 2 > max_chars and chunk:
                out.append({"heading": sec["heading"], "content": chunk.strip()})
                chunk = para
                part += 1
            else:
                chunk = (chunk + "\n\n" + para).strip() if chunk else para
        if chunk:
            out.append({"heading": sec["heading"], "content": chunk.strip()})
    return out


def parse_pdf(filepath: str, max_chars: int = 2000) -> tuple[list[dict], int]:
    """
    Extract a PDF with pymupdf4llm and return (sections, page_count).
    Falls back to plain-text extraction if pymupdf4llm is unavailable.
    """
    try:
        import pymupdf4llm
        import fitz  # PyMuPDF (bundled with pymupdf4llm)
        doc = fitz.open(filepath)
        page_count = len(doc)
        doc.close()
        md_text = pymupdf4llm.to_markdown(filepath)
    except ImportError:
        # Graceful fallback using basic PyMuPDF
        try:
            import fitz
            doc = fitz.open(filepath)
            page_count = len(doc)
            md_text = "\n\n".join(page.get_text() for page in doc)
            doc.close()
        except ImportError:
            raise RuntimeError(
                "Neither pymupdf4llm nor PyMuPDF (fitz) is installed. "
                "Run: pip install pymupdf4llm"
            )

    sections = _split_markdown_sections(md_text)
    sections = _chunk_large_sections(sections, max_chars=max_chars)
    return sections, page_count


def parse_text_file(filepath: str, max_chars: int = 2000) -> tuple[list[dict], int]:
    """Parse a plain-text or Markdown file into sections."""
    with open(filepath, encoding="utf-8", errors="replace") as f:
        text = f.read()
    sections = _split_markdown_sections(text)
    sections = _chunk_large_sections(sections, max_chars=max_chars)
    return sections, 1


def parse_document(
    filepath: str, max_chars: int = 2000
) -> tuple[list[dict], int]:
    """
    Dispatch to the appropriate parser based on file extension.
    Returns (sections, page_count).
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return parse_pdf(filepath, max_chars=max_chars)
    elif ext in (".md", ".txt"):
        return parse_text_file(filepath, max_chars=max_chars)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .md, .txt")
