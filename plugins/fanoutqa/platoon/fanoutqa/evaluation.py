"""Thin wrappers around FanOutQA official metrics and evaluator."""

from __future__ import annotations

import importlib
from typing import Any

import fanoutqa
from fanoutqa.models import AnswerType, DevQuestion

from .tasks import QUESTION_CACHE


def _official_module(name: str):
    return importlib.import_module(name)


def score_answer(reference: AnswerType, candidate: str):
    """Return official fanoutqa.eval.string.answer_in_text output unchanged."""
    return _official_module("fanoutqa.eval.string").answer_in_text(reference, candidate)


def score_episode(source_id: str, candidate: str) -> tuple[float, dict[str, Any]]:
    question = QUESTION_CACHE.get(source_id)
    if question is None:
        QUESTION_CACHE.update({question.id: question for question in fanoutqa.load_dev()})
        question = QUESTION_CACHE.get(source_id)
    if question is None:
        raise ValueError(f"FanOutQA dev reference not found for source_id={source_id!r}")
    result = score_answer(question.answer, candidate)
    score = float(result.score)
    # Do not serialize result.missing: it contains gold strings.
    return score, {"acc.loose": score, "acc.strict": float(result.found)}


def evaluate_answers(
    answers: list[dict[str, str]],
    questions: list[DevQuestion] | None = None,
    **kwargs: Any,
):
    """Run official FanOutQA full evaluation using original question IDs."""
    if questions is None:
        questions = fanoutqa.load_dev()
    return _official_module("fanoutqa.eval").evaluate(questions, answers, **kwargs)
