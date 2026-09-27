"""FanOutQA rollout glue for native Platoon episodes and trajectory trees."""

from __future__ import annotations

import asyncio
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from platoon.config_defs import RolloutConfig
from platoon.envs.base import Task
from platoon.episode.context import budget_tracker, current_trajectory, current_trajectory_collection
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import (
    BudgetExceededError,
    DepthAwareStepBudgetTracker,
    Trajectory,
    TrajectoryCollection,
)
from platoon.utils.llm_client import LiteLLMClient
from platoon.visualization.event_sinks import JsonlFileSink

from .agent import FanOutQAPromptBuilder, FanOutQARecursiveAgent, FanOutQASingleAgent
from .env import FanOutQARecursiveEnv, FanOutQASingleEnv

logger = logging.getLogger("platoon.fanoutqa.rollout")


@dataclass
class FanOutQATotalStepBudgetTracker(DepthAwareStepBudgetTracker):
    """Depth-aware tracker with a shared StepBudgetTracker-style subtree cap."""

    reserved_trajectory_budgets: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))

    def _iter_descendant_trajectory_ids(self, trajectory_id: str):
        collection = current_trajectory_collection.get()
        stack = [trajectory_id]
        seen: set[str] = set()
        while stack:
            parent_id = stack.pop()
            for child_id, child in collection.trajectories.items():
                info = child.parent_info
                if info is not None and info.id == parent_id and child_id not in seen:
                    seen.add(child_id)
                    yield child_id
                    stack.append(child_id)

    def used_budget_for(self, trajectory_id: str) -> float:
        collection = current_trajectory_collection.get()
        used = len(collection.trajectories[trajectory_id].steps)
        for child_id in self._iter_descendant_trajectory_ids(trajectory_id):
            used += len(collection.trajectories[child_id].steps)
        return float(used)

    def remaining_budget_for(self, trajectory_id: str) -> float:
        collection = current_trajectory_collection.get()
        task = collection.trajectories[trajectory_id].task
        allocated = task.max_steps if task is not None and task.max_steps else float("inf")
        return float(allocated - self.used_budget_for(trajectory_id) - self.reserved_trajectory_budgets[trajectory_id])

    def reserve_budget(self, requested_budget: float, raise_on_failure: bool = False) -> bool:
        # Preserve native depth validation, then reserve from this trajectory's total subtree.
        if not super().reserve_budget(requested_budget, raise_on_failure=raise_on_failure):
            return False
        trajectory_id = current_trajectory.get().id
        remaining = self.remaining_budget()
        if remaining < requested_budget:
            if raise_on_failure:
                raise BudgetExceededError(
                    f"Requested step budget {requested_budget} exceeds remaining shared budget {remaining}.",
                    reason="step_budget",
                    guidance="Request a smaller child budget or solve the subproblem directly.",
                )
            return False
        self.reserved_trajectory_budgets[trajectory_id] += requested_budget
        return True

    def release_budget(self, amount_to_release: float) -> None:
        trajectory_id = current_trajectory.get().id
        self.reserved_trajectory_budgets[trajectory_id] -= amount_to_release
        if self.reserved_trajectory_budgets[trajectory_id] < 0:
            self.reserved_trajectory_budgets[trajectory_id] = 0.0


class _ParallelBranchTracker:
    """Record observed overlap among live child episodes per parent."""

    def __init__(self, collection: TrajectoryCollection):
        self.collection = collection
        self.root_id: str | None = None
        self.active_by_parent: defaultdict[str, int] = defaultdict(int)
        self.max_by_parent: defaultdict[str, int] = defaultdict(int)
        self.finished: set[str] = set()

    def _write_metrics(self) -> None:
        if self.root_id is None:
            return
        root = self.collection.trajectories.get(self.root_id)
        if root is None:
            return
        root.misc["fanoutqa_parallel_metrics"] = {
            "parallel_branches": sum(1 for count in self.max_by_parent.values() if count > 1),
            "max_parallel_branches": max(self.max_by_parent.values(), default=0),
        }

    def on_trajectory_created(self, trajectory: Trajectory) -> None:
        if trajectory.parent_info is None:
            self.root_id = trajectory.id
        else:
            parent_id = trajectory.parent_info.id
            self.active_by_parent[parent_id] += 1
            self.max_by_parent[parent_id] = max(self.max_by_parent[parent_id], self.active_by_parent[parent_id])
        self._write_metrics()

    def on_trajectory_finished(self, trajectory: Trajectory) -> None:
        if trajectory.id in self.finished:
            return
        self.finished.add(trajectory.id)
        if trajectory.parent_info is not None:
            parent_id = trajectory.parent_info.id
            self.active_by_parent[parent_id] = max(0, self.active_by_parent[parent_id] - 1)
        self._write_metrics()

    def on_trajectory_step_added(self, trajectory: Trajectory, step: Any) -> None:
        del trajectory, step

    def on_trajectory_task_set(self, trajectory: Trajectory, task: Task | None) -> None:
        del trajectory, task


async def _run_rollout(task: Task, config: RolloutConfig, *, recursive: bool) -> dict[str, Any] | TrajectoryCollection:
    agent = env = None
    collection_none_token = current_trajectory_collection.set(None)
    trajectory_none_token = current_trajectory.set(None)
    budget_none_token = budget_tracker.set(None)
    collection_token = budget_token = None
    try:
        if config.max_steps is not None:
            task.max_steps = config.max_steps

        llm_client = LiteLLMClient(
            model=config.model_name, base_url=config.model_endpoint, api_key=config.model_api_key
        )
        prompt_builder = FanOutQAPromptBuilder(
            recursive=recursive,
            prompt_path=getattr(task, "_fanoutqa_prompt_path", None),
        )
        if recursive:
            subagent_max_steps = int(getattr(task, "_fanoutqa_subagent_max_steps", 15))
            max_depth = getattr(task, "_fanoutqa_max_recursive_depth", 4)
            env = FanOutQARecursiveEnv(task, subagent_max_steps=subagent_max_steps)
            agent = FanOutQARecursiveAgent(
                llm_client=llm_client,
                inference_params=config.inference_params,
                prompt_builder=prompt_builder,
            )
        else:
            max_depth = None
            env = FanOutQASingleEnv(task)
            agent = FanOutQASingleAgent(
                llm_client=llm_client,
                inference_params=config.inference_params,
                prompt_builder=prompt_builder,
            )

        collection = TrajectoryCollection()
        collection_token = current_trajectory_collection.set(collection)
        if recursive:
            budget_token = budget_tracker.set(FanOutQATotalStepBudgetTracker(max_depth=max_depth))

        os.makedirs(os.path.join(config.output_dir, "events"), exist_ok=True)
        events_path = os.path.join(config.output_dir, "events", f"events_{task.id}_{collection.id}.jsonl")
        collection.register_event_handlers(
            [
                JsonlFileSink(events_path, collection_id=collection.id, process_id=os.getpid()),
                _ParallelBranchTracker(collection),
            ]
        )

        if config.verbose:
            logger.info(
                "Starting FanOutQA %s rollout for task %s", "recursive" if recursive else "single-agent", task.id
            )

        episode_task = asyncio.create_task(run_episode(agent, env, timeout=config.step_timeout))
        try:
            await asyncio.wait_for(episode_task, timeout=config.timeout)
        except asyncio.TimeoutError:
            episode_task.cancel()
            try:
                await asyncio.wait_for(episode_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.warning("Timed out while cancelling FanOutQA task %s", task.id)
            raise

        if config.return_dict:
            return collection.to_dict()
        return collection
    finally:
        try:
            if agent is not None:
                await agent.close()
        finally:
            try:
                if env is not None:
                    await env.close()
            finally:
                if budget_token is not None:
                    budget_tracker.reset(budget_token)
                if collection_token is not None:
                    current_trajectory_collection.reset(collection_token)
                current_trajectory.reset(trajectory_none_token)
                current_trajectory_collection.reset(collection_none_token)
                budget_tracker.reset(budget_none_token)


async def run_rollout(task: Task, config: RolloutConfig) -> dict[str, Any] | TrajectoryCollection:
    """Run the strict single-agent baseline without a recursion action."""
    return await _run_rollout(task, config, recursive=False)


async def run_recursive_rollout(task: Task, config: RolloutConfig) -> dict[str, Any] | TrajectoryCollection:
    """Run a homogeneous recursive rollout with one shared total step budget."""
    return await _run_rollout(task, config, recursive=True)


def recursive_workflow_metrics(collection: dict[str, Any]) -> dict[str, float]:
    """Compute tree, retrieval, token, and actual parallel-branch diagnostics."""
    trajectories = collection.get("trajectories")
    if not isinstance(trajectories, dict) or not trajectories:
        return {}

    parents: dict[str, str | None] = {}
    steps_by_id: dict[str, list[Any]] = {}
    for trajectory_id, trajectory in trajectories.items():
        if not isinstance(trajectory, dict):
            continue
        info = trajectory.get("parent_info")
        parents[trajectory_id] = info.get("id") if isinstance(info, dict) else None
        raw_steps = trajectory.get("steps", [])
        steps_by_id[trajectory_id] = raw_steps if isinstance(raw_steps, list) else []

    roots = [trajectory_id for trajectory_id, parent_id in parents.items() if parent_id is None]
    if not roots:
        return {}
    root_id = roots[0]
    depth_cache: dict[str, int] = {}

    def depth_for(trajectory_id: str, visiting: set[str] | None = None) -> int:
        if trajectory_id in depth_cache:
            return depth_cache[trajectory_id]
        visiting = set() if visiting is None else visiting
        if trajectory_id in visiting:
            return 0
        visiting.add(trajectory_id)
        parent_id = parents.get(trajectory_id)
        depth = 0 if parent_id is None or parent_id not in parents else depth_for(parent_id, visiting) + 1
        visiting.remove(trajectory_id)
        depth_cache[trajectory_id] = depth
        return depth

    total_steps = 0
    root_steps = len(steps_by_id.get(root_id, []))
    child_steps = 0
    prompt_tokens = completion_tokens = total_tokens = 0
    num_searches = num_contents = max_depth = 0

    for trajectory_id, steps in steps_by_id.items():
        depth = depth_for(trajectory_id)
        max_depth = max(max_depth, depth)
        step_count = len(steps)
        total_steps += step_count
        if depth > 0:
            child_steps += step_count
        trajectory = trajectories.get(trajectory_id, {})
        trajectory_misc = trajectory.get("misc", {}) if isinstance(trajectory, dict) else {}
        tool_counts = trajectory_misc.get("fanoutqa_analysis", {}) if isinstance(trajectory_misc, dict) else {}
        if isinstance(tool_counts, dict):
            num_searches += int(tool_counts.get("wiki_searches", 0))
            num_contents += int(tool_counts.get("wiki_content_calls", 0))
        for step in steps:
            if not isinstance(step, dict):
                continue
            misc = step.get("misc", {})
            action_misc = misc.get("action_misc", {}) if isinstance(misc, dict) else {}
            usage = action_misc.get("usage", {}) if isinstance(action_misc, dict) else {}
            if isinstance(usage, dict):
                prompt_tokens += int(usage.get("prompt_tokens") or 0)
                completion_tokens += int(usage.get("completion_tokens") or 0)
                total_tokens += int(usage.get("total_tokens") or 0)

    root = trajectories.get(root_id, {})
    root_misc = root.get("misc", {}) if isinstance(root, dict) else {}
    parallel = root_misc.get("fanoutqa_parallel_metrics", {}) if isinstance(root_misc, dict) else {}
    parallel_branches = int(parallel.get("parallel_branches", 0)) if isinstance(parallel, dict) else 0
    max_parallel = int(parallel.get("max_parallel_branches", 0)) if isinstance(parallel, dict) else 0
    num_subagents = max(0, len(trajectories) - 1)
    return {
        "number_of_subagents": float(num_subagents),
        "num_subagents": float(num_subagents),
        "max_recursion_depth": float(max_depth),
        "max_depth": float(max_depth),
        "number_of_wiki_searches": float(num_searches),
        "num_wiki_searches": float(num_searches),
        "number_of_wiki_content_calls": float(num_contents),
        "num_wiki_content_calls": float(num_contents),
        "root_steps": float(root_steps),
        "child_steps": float(child_steps),
        "total_steps": float(total_steps),
        "prompt_tokens": float(prompt_tokens),
        "completion_tokens": float(completion_tokens),
        "total_tokens": float(total_tokens),
        "parallel_branches": float(parallel_branches),
        "max_parallel_branches": float(max_parallel),
    }
