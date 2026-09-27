"""Lazy public API for FanOutQA adapters.

Importing this package must not import the official FanOutQA package. Inference
CLI runtime overrides (including Wikipedia backend settings) are applied before
the dataset and retrieval modules are first loaded.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "FanOutQACodeExecutor": ("env", "FanOutQACodeExecutor"),
    "FanOutQAEnv": ("env", "FanOutQAEnv"),
    "FanOutQARecursiveAgent": ("agent", "FanOutQARecursiveAgent"),
    "FanOutQARecursiveCodeExecutor": ("env", "FanOutQARecursiveCodeExecutor"),
    "FanOutQARecursiveEnv": ("env", "FanOutQARecursiveEnv"),
    "FanOutQAPromptBuilder": ("agent", "FanOutQAPromptBuilder"),
    "FanOutQASingleAgent": ("agent", "FanOutQASingleAgent"),
    "FanOutQASingleCodeExecutor": ("env", "FanOutQASingleCodeExecutor"),
    "FanOutQASingleEnv": ("env", "FanOutQASingleEnv"),
    "QUESTION_CACHE": ("tasks", "QUESTION_CACHE"),
    "evaluate_answers": ("evaluation", "evaluate_answers"),
    "load_tasks": ("tasks", "load_tasks"),
    "score_answer": ("evaluation", "score_answer"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(f".{module_name}", __name__)
    return getattr(module, attribute_name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
