"""Official FanOutQA dataset loading with policy-safe Platoon tasks."""

from __future__ import annotations

from typing import Literal

from platoon.envs.base import Task

import fanoutqa
from fanoutqa.models import DevQuestion

DatasetSplit = Literal["dev", "test"]
QUESTION_CACHE: dict[str, DevQuestion] = {}


def load_tasks(split: DatasetSplit = "dev", *, max_steps: int = 30, num_tasks: int | None = None) -> list[Task]:
    """Load an official split and expose only id, question, and categories."""
    if split not in {"dev", "test"}:
        raise ValueError("split must be 'dev' or 'test'")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if num_tasks is not None and num_tasks < 0:
        raise ValueError("num_tasks must be non-negative or None")

    questions = fanoutqa.load_dev() if split == "dev" else fanoutqa.load_test()
    if split == "dev":
        QUESTION_CACHE.update({question.id: question for question in questions})
    if num_tasks is not None:
        questions = questions[:num_tasks]

    return [
        Task(
            id=f"fanoutqa.{split}.{question.id}",
            goal=question.question,
            max_steps=max_steps,
            misc={"source_id": question.id, "dataset_split": split, "categories": list(question.categories)},
            fork_strategy="task",
        )
        for question in questions
    ]
