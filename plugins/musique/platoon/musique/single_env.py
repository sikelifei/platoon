"""Standalone search-only environment for the MuSiQue single agent."""

from __future__ import annotations

from platoon.envs.base import Task
from platoon.envs.codeact import CodeActEnv

from .recursive_env import MuSiQueCodeExecutor, MuSiQueEnv


class MuSiQueSingleCodeExecutor(MuSiQueCodeExecutor):
    """Executor exposing exactly search and finish."""


class MuSiQueSingleEnv(MuSiQueEnv):
    """Root-only MuSiQue environment with no recursive action."""

    def __init__(self, task: Task):
        CodeActEnv.__init__(
            self,
            task=task,
            code_executor=MuSiQueSingleCodeExecutor(task),
        )

    async def fork(self, task: Task) -> "MuSiQueSingleEnv":
        return MuSiQueSingleEnv(task)
