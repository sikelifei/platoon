"""Answer normalization and scoring for MuSiQue inference."""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from typing import Any


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _token_f1(prediction: str, gold: str) -> float:
    predicted_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not predicted_tokens or not gold_tokens:
        return float(predicted_tokens == gold_tokens)
    overlap = Counter(predicted_tokens) & Counter(gold_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(predicted_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_candidates(example: dict[str, Any]) -> list[str]:
    candidates: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
        elif isinstance(value, (list, tuple)):
            for item in value:
                add(item)
        elif isinstance(value, dict):
            for key in ("text", "answer", "value"):
                if key in value:
                    add(value[key])

    for key in ("answer", "answer_aliases", "answer_alias", "answers", "target"):
        add(example.get(key))

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = normalize_answer(candidate)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(candidate)
    return unique


def extract_prediction(message: str) -> str:
    """Extract an answer from a finish message without requiring one prompt format."""
    text = str(message or "").strip()
    if not text:
        return ""

    fenced = re.search(r"\x60\x60\x60(?:json)?\s*(.*?)\s*\x60\x60\x60", text, flags=re.IGNORECASE | re.DOTALL)
    candidate = fenced.group(1).strip() if fenced else text
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        for key in ("answer", "final_answer", "prediction"):
            value = parsed.get(key)
            if isinstance(value, str):
                return value.strip()

    matches = list(
        re.finditer(
            r"(?:final\s+answer|answer|prediction)\s*[:：]\s*(.+)",
            candidate,
            flags=re.IGNORECASE,
        )
    )
    if matches:
        return matches[-1].group(1).strip().strip(" \\t").strip()
    return candidate


def score_answer(message: str, example: dict[str, Any]) -> dict[str, float | str]:
    prediction = extract_prediction(message)
    golds = answer_candidates(example)
    if not golds:
        return {
            "answer_em": 0.0,
            "answer_f1": 0.0,
            "prediction": prediction,
            "reason": "No gold answer is available for this split.",
        }

    scores = [
        (_token_f1(prediction, gold), normalize_answer(prediction) == normalize_answer(gold))
        for gold in golds
    ]
    best_f1, best_em = max(scores, key=lambda item: (item[0], item[1]))
    return {
        "answer_em": float(best_em),
        "answer_f1": float(best_f1),
        "prediction": prediction,
        "gold_answer": golds[0],
        "reason": "MuSiQue answer F1 against the best available gold alias.",
    }
