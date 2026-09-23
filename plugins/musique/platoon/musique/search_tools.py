"""Deterministic, task-local MuSiQue passage search."""

from __future__ import annotations

import re
from typing import Any


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _passage_text(paragraph: dict[str, Any]) -> str:
    for key in ("paragraph_text", "text", "paragraph", "contents"):
        value = _text(paragraph.get(key))
        if value:
            return value
    return ""


def build_passages(task_misc: dict[str, Any]) -> list[dict[str, str]]:
    """Project benchmark passages into the fields that retrieval may expose."""
    existing = task_misc.get("retrieval_passages")
    if isinstance(existing, list):
        projected: list[dict[str, str]] = []
        for index, item in enumerate(existing, start=1):
            if not isinstance(item, dict):
                continue
            text = _text(item.get("text"))
            if text:
                projected.append(
                    {
                        "passage_id": _text(item.get("passage_id")) or f"p{index}",
                        "title": _text(item.get("title")) or f"Passage {index}",
                        "text": text,
                    }
                )
        if projected:
            return projected

    paragraphs = task_misc.get("paragraphs")
    projected = []
    if isinstance(paragraphs, list):
        for index, paragraph in enumerate(paragraphs, start=1):
            if not isinstance(paragraph, dict):
                continue
            text = _passage_text(paragraph)
            if text:
                projected.append(
                    {
                        "passage_id": _text(paragraph.get("id")) or f"p{index}",
                        "title": _text(paragraph.get("title")) or f"Passage {index}",
                        "text": text,
                    }
                )
    return projected


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(value)}


def search(
    query: str,
    max_results: int = 5,
    *,
    task_misc: dict[str, Any] | None = None,
) -> str:
    """Search the current task's passages and return readable plain text."""
    query = _text(query)
    if not query:
        return "Search query is empty."
    if task_misc is None:
        return f"No MuSiQue search corpus is configured for query: {query}"
    try:
        limit = max(1, min(int(max_results), 20))
    except (TypeError, ValueError):
        limit = 5
    query_tokens = _tokens(query)
    passages = build_passages(task_misc)
    ranked: list[tuple[int, int, dict[str, str]]] = []
    for index, passage in enumerate(passages):
        score = len(query_tokens & _tokens(passage["text"]))
        score += 2 * len(query_tokens & _tokens(passage["title"]))
        if score:
            ranked.append((score, index, passage))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    if not ranked:
        return f'No results for query: "{query}"'
    lines = [f'Query: "{query}"']
    for result_index, (score, _, passage) in enumerate(ranked[:limit], start=1):
        lines.extend(
            [
                f"{result_index}. [{passage['passage_id']}] {passage['title']} (lexical_score={score})",
                passage["text"],
            ]
        )
    return "\n".join(lines)


def render_passages(task_misc: dict[str, Any]) -> str:
    """Render only retrieval-safe passage fields for a child task."""
    return "\n\n".join(
        f"[{item['passage_id']}] {item['title']}\n{item['text']}"
        for item in build_passages(task_misc)
    )
