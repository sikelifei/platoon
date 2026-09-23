"""MuSiQue CodeAct environment and answer evaluation."""

from __future__ import annotations

from platoon.agents.actions.common import finish
from platoon.envs.base import Task
from platoon.envs.codeact import CodeActEnv, IPythonCodeExecutor, safe_asyncio
from platoon.episode.context import finish_message

from .evaluation import score_answer


class MuSiQueCodeExecutor(IPythonCodeExecutor):
    def __init__(self, task: Task):
        self.context = str(task.misc.get("context", ""))
        self.question = task.goal or ""
        super().__init__(
            task,
            actions=(finish, safe_asyncio),
            detect_unawaited_async_calls=True,
            detect_while_loops=True,
            detect_interactive_input=True,
        )
        self._inject_bindings()

    def _inject_bindings(self) -> None:
        self.shell.user_ns["context"] = self.context
        self.shell.user_ns["question"] = self.question

    async def describe_action_space(self) -> str:
        return """Available Actions (Python functions):
1. finish(message: str) -> str
   Submit the final answer and end the task.
2. context (str) and question (str)
   Preloaded read-only variables available in the REPL.
"""

    async def reset(self) -> MuSiQueCodeExecutor:
        await super().reset()
        self._inject_bindings()
        return self


class MuSiQueEnv(CodeActEnv):
    def __init__(self, task: Task):
        super().__init__(task=task, code_executor=MuSiQueCodeExecutor(task))

    async def evaluate(self) -> tuple[float, dict]:
        if not self._state.finished:
            return 0.0, {}

        result = score_answer(str(finish_message.get() or ""), self._task.misc)
        reward_misc = {
            "answer_em": float(result["answer_em"]),
            "answer_f1": float(result["answer_f1"]),
            "reward/success": float(result["answer_f1"]),
            "prediction": str(result.get("prediction", "")),
            "reason": str(result.get("reason", "")),
        }
        return float(result["answer_f1"]), reward_misc

from .recursive_env import (  # noqa: E402
    MuSiQueCodeExecutor,
    MuSiQueRecursiveCodeExecutor,
    MuSiQueRecursiveEnv,
    MuSiQueEnv,
)
