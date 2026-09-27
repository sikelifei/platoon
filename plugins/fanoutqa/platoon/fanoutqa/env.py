"""Official FanOutQA retrieval and evaluation inside Platoon."""

from __future__ import annotations

import inspect
from typing import Any

from platoon.agents.actions.common import finish
from platoon.agents.actions.subagent import launch_subagent as platoon_launch_subagent
from platoon.envs.base import Task
from platoon.envs.codeact import CodeActEnv, IPythonCodeExecutor, safe_asyncio
from platoon.episode.context import budget_tracker, current_trajectory, finish_message


def _count_tool_call(name: str) -> None:
    trajectory = current_trajectory.get(None)
    if trajectory is None:
        return
    counts = trajectory.misc.setdefault("fanoutqa_analysis", {"wiki_searches": 0, "wiki_content_calls": 0})
    key = "wiki_searches" if name == "wiki_search" else "wiki_content_calls"
    counts[key] = int(counts.get(key, 0)) + 1


def _policy_task(task: Task) -> Task:
    child = bool(task.misc.get("_is_subagent"))
    allowed = (
        ("source_id", "dataset_split", "_is_subagent")
        if child
        else (
            "source_id",
            "dataset_split",
            "categories",
        )
    )
    misc: dict[str, Any] = {key: task.misc[key] for key in allowed if key in task.misc}
    if child:
        misc["_is_subagent"] = True
    return Task(
        id=task.id,
        goal=task.goal,
        max_steps=task.max_steps,
        misc=misc,
        fork_strategy="task",
    )


def _ensure_ipython_compat(executor: IPythonCodeExecutor) -> None:
    """Adapt IPython 9's run_cell_async API on this executor instance only."""
    shell = executor.shell
    if getattr(shell, "_fanoutqa_run_cell_compat", False):
        return
    if "transformed_cell" not in inspect.signature(shell.run_cell_async).parameters:
        return
    original = shell.run_cell_async

    async def run_cell_compat(raw_cell: str, *args, **kwargs):
        kwargs.setdefault("transformed_cell", shell.transform_cell(raw_cell))
        return await original(raw_cell, *args, **kwargs)

    shell.run_cell_async = run_cell_compat
    shell._fanoutqa_run_cell_compat = True


class FanOutQASingleCodeExecutor(IPythonCodeExecutor):
    def _create_shell(self):
        shell = super()._create_shell()
        from fanoutqa.models import Evidence

        shell.user_ns["Evidence"] = Evidence
        return shell

    def __init__(self, task: Task):
        super().__init__(
            _policy_task(task),
            actions=(self.wiki_search, self.wiki_content, finish),
            detect_unawaited_async_calls=True,
            detect_while_loops=True,
            detect_interactive_input=True,
        )

    async def run(self, code: str):
        _ensure_ipython_compat(self)
        return await super().run(code)

    def wiki_search(self, query: str, results: int = 10):
        _count_tool_call("wiki_search")
        from fanoutqa import wiki_search as official_wiki_search

        return official_wiki_search(query, results=results)

    def wiki_content(self, evidence):
        _count_tool_call("wiki_content")
        from fanoutqa import wiki_content as official_wiki_content

        return official_wiki_content(evidence)

    async def describe_action_space(self) -> str:
        return """Available Actions (Python functions):

1. def wiki_search(query: str, results: int = 10)
   Search Wikipedia and return a list of official FanOutQA Evidence objects.
   This function is synchronous; do not await it.

2. def wiki_content(evidence) -> str
   Retrieve page content for an Evidence object returned by wiki_search.
   Pass the returned object directly; do not reconstruct it.
   This function is synchronous; do not await it.
   Use page content as evidence for factual claims. Search results are mainly
   for locating the appropriate page.

3. def finish(message: str) -> str
   Submit the final answer and end the task.
   This function is synchronous.
"""

    async def reset(self) -> FanOutQASingleCodeExecutor:
        await super().reset()
        return self

    async def fork(self, task: Task) -> FanOutQASingleCodeExecutor:
        return FanOutQASingleCodeExecutor(task)


class FanOutQARecursiveCodeExecutor(FanOutQASingleCodeExecutor):
    def __init__(self, task: Task, subagent_max_steps: int = 15):
        self.subagent_max_steps = max(1, int(subagent_max_steps))
        IPythonCodeExecutor.__init__(
            self,
            _policy_task(task),
            actions=(self.launch_subagent, self.wiki_search, self.wiki_content, finish, safe_asyncio),
            detect_unawaited_async_calls=True,
            detect_while_loops=True,
            detect_interactive_input=True,
        )

    async def launch_subagent(self, goal: str) -> str:
        if not isinstance(goal, str) or not goal.strip():
            return "Delegated goal is empty."
        remaining = int(budget_tracker.get().remaining_budget())
        child_steps = min(self.subagent_max_steps, remaining - 1)
        if child_steps < 1:
            return "Not enough step budget to launch a subagent and process its result."
        child_misc = {
            "source_id": str(self.task.misc.get("source_id") or ""),
            "dataset_split": str(self.task.misc.get("dataset_split") or ""),
            "_is_subagent": True,
        }
        result = await platoon_launch_subagent(
            goal=goal,
            max_steps=child_steps,
            task_misc=child_misc,
            verbose=False,
        )
        return str(result)

    async def describe_action_space(self) -> str:
        return """Available Actions (Python functions):

1. async def launch_subagent(goal: str) -> str
   Delegate one focused, self-contained subproblem to a child agent.
   The child has the same Wikipedia tools and may recursively delegate.
   The child does not see the parent's previous search history, so include all
   entities and context required to solve the subproblem in `goal`.

   Use:
       result = await launch_subagent(goal)

   Independent subproblems may run concurrently:
       results = await asyncio.gather(
           launch_subagent(goal_a),
           launch_subagent(goal_b),
       )

   `asyncio` is already available.

2. def wiki_search(query: str, results: int = 10)
   Search Wikipedia and return a list of official FanOutQA Evidence objects.
   This function is synchronous; do not await it.

3. def wiki_content(evidence) -> str
   Retrieve page content for an Evidence object returned by wiki_search.
   Pass the returned object directly; do not reconstruct it.
   This function is synchronous; do not await it.

   Use page content as evidence for factual claims. Search results are mainly
   for locating the appropriate page.

4. def finish(message: str) -> str
   Submit the final answer and end the task.
   This function is synchronous.
"""

    async def reset(self) -> FanOutQARecursiveCodeExecutor:
        await IPythonCodeExecutor.reset(self)
        return self

    async def fork(self, task: Task) -> FanOutQARecursiveCodeExecutor:
        return FanOutQARecursiveCodeExecutor(task, self.subagent_max_steps)


class FanOutQASingleEnv(CodeActEnv):
    def __init__(self, task: Task):
        policy_task = _policy_task(task)
        CodeActEnv.__init__(self, task=policy_task, code_executor=FanOutQASingleCodeExecutor(policy_task))

    async def evaluate(self) -> tuple[float, dict]:
        if not self._state.finished:
            return 0.0, {}
        if self.task.misc.get("_is_subagent"):
            return 0.0, {"reward/success": 0.0, "reason": "Subagent result is returned to its parent."}
        if self.task.misc.get("dataset_split") != "dev":
            return 0.0, {"reward/success": 0.0, "reason": "Reference unavailable for this split."}
        from .evaluation import score_episode

        prediction = str(finish_message.get(None) or "")
        reward, metric = score_episode(str(self.task.misc.get("source_id") or ""), prediction)
        return reward, {"reward/success": reward, **metric}

    async def fork(self, task: Task) -> FanOutQASingleEnv:
        # Fresh context; the native CodeActEnv fork would attach parent_state.
        return FanOutQASingleEnv(task)


class FanOutQARecursiveEnv(FanOutQASingleEnv):
    def __init__(self, task: Task, subagent_max_steps: int = 15):
        self.subagent_max_steps = max(1, int(subagent_max_steps))
        policy_task = _policy_task(task)
        CodeActEnv.__init__(
            self,
            task=policy_task,
            code_executor=FanOutQARecursiveCodeExecutor(policy_task, self.subagent_max_steps),
        )

    async def fork(self, task: Task) -> FanOutQARecursiveEnv:
        return FanOutQARecursiveEnv(task, self.subagent_max_steps)


FanOutQACodeExecutor = FanOutQASingleCodeExecutor
FanOutQAEnv = FanOutQASingleEnv
