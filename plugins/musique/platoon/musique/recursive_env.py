"""MuSiQue search-only and recursive Platoon environments."""

from __future__ import annotations

import ast
from copy import deepcopy
from typing import Any

from platoon.agents.actions.common import finish
from platoon.agents.actions.subagent import launch_subagent
from platoon.envs.base import Task
from platoon.envs.codeact import CodeActEnv, CodeActStep, IPythonCodeExecutor, safe_asyncio
from platoon.episode.context import finish_message

from .evaluation import score_answer
from .search_tools import build_passages, render_passages, search


def agent_safe_misc(task: Task) -> dict[str, Any]:
    """Create the only metadata allowed to cross a recursive task boundary."""
    passages = build_passages(task.misc)
    return {
        "source_id": str(task.misc.get("source_id") or task.id or ""),
        "question": str(task.goal or ""),
        "context": render_passages({"retrieval_passages": passages}),
        "retrieval_passages": passages,
        "_is_subagent": True,
    }


class MuSiQueCodeExecutor(IPythonCodeExecutor):
    """Non-recursive executor exposing only local search and finish."""

    def __init__(self, task: Task):
        super().__init__(
            task,
            actions=(self.search, finish),
            detect_unawaited_async_calls=True,
            detect_while_loops=True,
            detect_interactive_input=True,
        )

    def search(self, query: str, max_results: int = 5) -> str:
        return search(query, max_results, task_misc=self.task.misc)

    async def describe_action_space(self) -> str:
        return """Available Actions (Python functions):
1. def search(query: str, max_results: int = 5) -> str
   Search the task's passages and return readable evidence as plain text.
2. def finish(message: str) -> str
   Submit the final answer and end the task.
"""

    async def reset(self) -> MuSiQueCodeExecutor:
        await super().reset()
        return self

    async def fork(self, task: Task) -> MuSiQueCodeExecutor:
        return MuSiQueCodeExecutor(task)


class MuSiQueRecursiveCodeExecutor(MuSiQueCodeExecutor):
    """Recursive executor adding the native Platoon subagent action."""

    def __init__(self, task: Task, subagent_max_steps: int = 25):
        self.subagent_max_steps = subagent_max_steps
        IPythonCodeExecutor.__init__(
            self,
            task,
            actions=(self.launch_subagent, self.search, finish, safe_asyncio),
            detect_unawaited_async_calls=True,
            detect_while_loops=True,
            detect_interactive_input=True,
        )

    async def launch_subagent(self, goal: str) -> str:
        if not isinstance(goal, str) or not goal.strip():
            return "Delegated goal is empty."
        result = await launch_subagent(
            goal=goal,
            max_steps=self.subagent_max_steps,
            task_misc=deepcopy(agent_safe_misc(self.task)),
            verbose=False,
        )
        return str(result)

    async def describe_action_space(self) -> str:
        return """Available Actions (Python functions):
1. async def launch_subagent(goal: str) -> str
   Delegate one focused, self-contained subproblem and receive plain text.
2. def search(query: str, max_results: int = 5) -> str
   Search the task's passages and return readable evidence as plain text.
3. def finish(message: str) -> str
   Submit the final answer and end the task.
"""

    async def reset(self) -> MuSiQueRecursiveCodeExecutor:
        await super().reset()
        return self

    async def fork(self, task: Task) -> MuSiQueRecursiveCodeExecutor:
        return MuSiQueRecursiveCodeExecutor(task, self.subagent_max_steps)


class MuSiQueEnv(CodeActEnv):
    """Root MuSiQue evaluator; child tasks return text without root scoring."""

    def __init__(self, task: Task):
        super().__init__(task=task, code_executor=MuSiQueCodeExecutor(task))

    async def evaluate(self) -> tuple[float, dict]:
        if not self._state.finished:
            return 0.0, {}
        if self._task.misc.get("_is_subagent"):
            return 0.0, {
                "success": False,
                "reward/success": 0.0,
                "reason": "Subagent result is returned to its parent.",
            }
        result = score_answer(str(finish_message.get() or ""), self._task.misc)
        reward_misc = {
            "answer_em": float(result["answer_em"]),
            "answer_f1": float(result["answer_f1"]),
            "reward/success": float(result["answer_f1"]),
            "prediction": str(result.get("prediction", "")),
            "reason": str(result.get("reason", "")),
        }
        return float(result["answer_f1"]), reward_misc

    async def fork(self, task: Task) -> MuSiQueEnv:
        return MuSiQueEnv(task)


class MuSiQueRecursiveEnv(MuSiQueEnv):
    """Recursive MuSiQue environment with native parent/child trajectories."""

    def __init__(self, task: Task, subagent_max_steps: int = 25):
        CodeActEnv.__init__(
            self,
            task=task,
            code_executor=MuSiQueRecursiveCodeExecutor(task, subagent_max_steps),
        )
        self.subagent_max_steps = subagent_max_steps

    async def fork(self, task: Task) -> MuSiQueRecursiveEnv:
        return MuSiQueRecursiveEnv(task, self.subagent_max_steps)

# Platoon currently calls IPython's old run_cell_async API. IPython 9 requires
# callers to pass the transformed cell explicitly; keep this compatibility
# local to the plugin instead of changing the shared framework.
import inspect


_PLATOON_RUN = IPythonCodeExecutor.run


def _ensure_ipython_compat(executor: IPythonCodeExecutor) -> None:
    shell = executor.shell
    if getattr(shell, "_musique_run_cell_compat", False):
        return
    signature = inspect.signature(shell.run_cell_async)
    if "transformed_cell" not in signature.parameters:
        return
    original = shell.run_cell_async

    async def run_cell_compat(raw_cell: str, *args, **kwargs):
        kwargs.setdefault("transformed_cell", shell.transform_cell(raw_cell))
        return await original(raw_cell, *args, **kwargs)

    shell.run_cell_async = run_cell_compat
    shell._musique_run_cell_compat = True


_PROTECTED_ACTION_NAMES = {"search", "launch_subagent", "finish"}
_MODEL_TAG_SUFFIXES = ("</｜DSML｜>", "<｜DSML｜>", "</DSML>", "<DSML>")


def _clean_generated_code(code: str) -> str:
    cleaned = code.strip()
    changed = True
    while changed:
        changed = False
        for suffix in _MODEL_TAG_SUFFIXES:
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)].rstrip()
                changed = True
    return cleaned


def _protected_action_rebinding(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name in _PROTECTED_ACTION_NAMES
        ):
            return node.name
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node.id in _PROTECTED_ACTION_NAMES
        ):
            return node.id
    return None


async def _run_with_ipython_compat(self: IPythonCodeExecutor, code: str):
    _ensure_ipython_compat(self)
    code = _clean_generated_code(code)
    protected_name = _protected_action_rebinding(code)
    if protected_name is not None:
        return CodeActStep(
            code=code,
            error=f"Action {protected_name!r} is preloaded and cannot be redefined or deleted.",
        )
    return await _PLATOON_RUN(self, code)


MuSiQueCodeExecutor.run = _run_with_ipython_compat
