"""CurAgent TextCraft-Synth rollout adapter for the AReaL backend."""

from __future__ import annotations

import asyncio
import sys
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

from platoon.config_defs import RolloutConfig
from platoon.envs.base import Task

_IMPORT_LOCK = Lock()


@dataclass
class CurAgentRolloutConfig(RolloutConfig):
    """Rollout options owned by the CurAgent harness."""

    curagent_root: str = ""
    curagent_model_config: str = ""
    prompt_file: str | None = None
    max_depth: int = 12
    max_concurrent_subagents: int = 4
    max_subagents_per_agent: int = 6
    max_run_seconds: float = 900.0
    max_observation_chars: int = 8000
    max_retries: int = 2


def task_to_curagent_sample(task: Task) -> dict[str, Any]:
    """Convert a Platoon TextCraft task into CurAgent's normalized sample shape."""

    misc = task.misc or {}
    targets = misc.get("target_items")
    inventory = misc.get("initial_inventory")
    gold_trajectory = misc.get("gold_trajectory")
    if not isinstance(targets, Mapping) or not targets:
        raise ValueError(f"TextCraft task {task.id!r} has no target_items")
    if not isinstance(inventory, Mapping):
        raise ValueError(f"TextCraft task {task.id!r} has no initial_inventory")
    if not isinstance(gold_trajectory, list) or not gold_trajectory:
        raise ValueError(f"TextCraft task {task.id!r} has no gold_trajectory")

    recipes: dict[str, list[dict[str, Any]]] = {}
    for index, step in enumerate(gold_trajectory):
        if not isinstance(step, Mapping) or step.get("action") != "craft":
            continue
        target = step.get("target")
        ingredients = step.get("ingredients")
        total_output = step.get("result_count")
        if (
            not isinstance(target, (list, tuple))
            or len(target) != 2
            or not isinstance(ingredients, Mapping)
        ):
            raise ValueError(f"Malformed craft step {index} in task {task.id!r}")

        item = str(target[0])
        executions = int(target[1])
        total_output = int(total_output)
        if executions <= 0 or total_output <= 0 or total_output % executions:
            raise ValueError(f"Invalid craft counts at step {index} in task {task.id!r}")

        per_execution: dict[str, int] = {}
        for ingredient, total_count in ingredients.items():
            total_count = int(total_count)
            if total_count <= 0 or total_count % executions:
                raise ValueError(
                    f"Ingredient count for {ingredient!r} is not divisible by "
                    f"the execution count at step {index} in task {task.id!r}"
                )
            per_execution[str(ingredient)] = total_count // executions

        recipe = {
            "ingredients": per_execution,
            "result_count": total_output // executions,
        }
        if recipe not in recipes.setdefault(item, []):
            recipes[item].append(recipe)

    if not recipes:
        raise ValueError(f"TextCraft task {task.id!r} produced no CurAgent recipes")

    split = "val" if ".val." in str(task.id) else "train"
    return {
        "id": str(task.id),
        "split": split,
        "initial_inventory": {str(key): int(value) for key, value in inventory.items()},
        "recipes": recipes,
        "targets": {str(key): int(value) for key, value in targets.items()},
        "difficulty": str(misc.get("difficulty", "unknown")),
        "crafting_depth": int(misc.get("max_depth", 0)),
        "metadata": {"source": "platoon_textcraft_synth"},
    }


def cached_completions_to_trajectory_collection(
    completion_ids: list[str],
    *,
    task_id: str,
    reward: float,
) -> dict[str, Any]:
    """Represent each cached CurAgent model call as one trainable trajectory."""

    trajectories = {}
    for index, completion_id in enumerate(completion_ids):
        trajectories[f"{task_id}:completion:{index}"] = {
            "task": None,
            "steps": [
                {
                    "action": None,
                    "observation": None,
                    "reward": reward,
                    "done": False,
                    "misc": {"action_misc": {"completion_id": completion_id}},
                }
            ],
            "reward": reward,
            "parent_info": None,
        }
    return {"id": f"curagent:{task_id}", "trajectories": trajectories}


def curagent_trace_to_trajectory_collection(
    trace: Mapping[str, Any] | None,
    completions: Mapping[str, Any],
    *,
    task_id: str,
    reward: float,
) -> dict[str, Any]:
    """Map CurAgent's agent tree and AReaL completion cache into trajectories."""

    nodes = list(_walk_curagent_trace(trace))
    expected: dict[tuple[str, str, str], deque[str]] = defaultdict(deque)
    trajectories: dict[str, dict[str, Any]] = {}

    for node in nodes:
        agent_id = str(node.get("agent_id") or f"curagent-agent-{len(trajectories)}")
        parent_id = node.get("parent_id")
        trajectories[agent_id] = {
            "task": None,
            "steps": [],
            "reward": reward,
            "parent_info": (
                {"id": str(parent_id), "fork_step": 0} if parent_id is not None else None
            ),
        }
        initial_user = (
            f"Task:\n{node.get('task', '')}"
            if parent_id is None
            else f"Delegated task:\n{node.get('task', '')}"
        )
        responses = [
            str(step.get("response", ""))
            for step in node.get("steps", [])
            if isinstance(step, Mapping)
        ]
        if node.get("forced_final_response") is not None:
            responses.append(str(node["forced_final_response"]))
        for response in responses:
            expected[(str(node.get("system_prompt", "")), initial_user, response)].append(
                agent_id
            )

    unmatched: list[str] = []
    for completion_id, interaction in completions.items():
        key = _completion_match_key(interaction)
        candidates = expected.get(key)
        if not candidates:
            unmatched.append(str(completion_id))
            continue
        agent_id = candidates.popleft()
        trajectories[agent_id]["steps"].append(
            _trajectory_step(str(completion_id), reward)
        )

    trajectories = {
        trajectory_id: trajectory
        for trajectory_id, trajectory in trajectories.items()
        if trajectory["steps"]
    }
    if unmatched:
        fallback = cached_completions_to_trajectory_collection(
            unmatched,
            task_id=f"{task_id}:unmatched",
            reward=reward,
        )
        trajectories.update(fallback["trajectories"])
    return {"id": f"curagent:{task_id}", "trajectories": trajectories}


def _walk_curagent_trace(trace: Mapping[str, Any] | None) -> Iterable[Mapping[str, Any]]:
    if not isinstance(trace, Mapping):
        return
    yield trace
    for child in trace.get("children", []) or []:
        if isinstance(child, Mapping):
            yield from _walk_curagent_trace(child)


def _completion_match_key(interaction: Any) -> tuple[str, str, str]:
    messages = getattr(interaction, "messages", None) or []
    system = str(messages[0].get("content", "")) if len(messages) > 0 else ""
    initial_user = str(messages[1].get("content", "")) if len(messages) > 1 else ""
    completion = getattr(interaction, "completion", None)
    choices = getattr(completion, "choices", None) or []
    response = ""
    if choices:
        response = str(getattr(getattr(choices[0], "message", None), "content", "") or "")
    return system, initial_user, response


def _trajectory_step(completion_id: str, reward: float) -> dict[str, Any]:
    return {
        "action": None,
        "observation": None,
        "reward": reward,
        "done": False,
        "misc": {"action_misc": {"completion_id": completion_id}},
    }


async def run_curagent_synth_rollout(
    task: Task,
    config: CurAgentRolloutConfig,
) -> dict[str, Any]:
    """Run CurAgent's recursive TextCraft harness against an AReaL proxy session."""

    _ensure_curagent_importable(config.curagent_root)
    from recursive_agent.envs import run_registered_environment

    model_config = Path(config.curagent_model_config).expanduser()
    if not model_config.is_file():
        raise FileNotFoundError(f"CurAgent model config not found: {model_config}")

    prompt = None
    if config.prompt_file:
        prompt = Path(config.prompt_file).expanduser().read_text(encoding="utf-8")

    model_name = str(config.model_name or "")
    if model_name.startswith("openai/"):
        model_name = model_name.removeprefix("openai/")

    sample = task_to_curagent_sample(task)
    environment_kwargs: dict[str, Any] = {
        "samples": [sample],
        "split": sample["split"],
        "instance_id": 0,
    }
    if prompt is not None:
        environment_kwargs["agent_prompt"] = prompt

    inference = config.inference_params
    sampling_args = {
        "temperature": inference.temperature,
        "top_p": inference.top_p,
        "max_completion_tokens": inference.max_completion_tokens,
    }
    sampling_args = {key: value for key, value in sampling_args.items() if value is not None}
    model_overrides = {
        "model_name": model_name,
        "base_url": config.model_endpoint,
        "api_key": config.model_api_key or "None",
        "timeout": float(config.step_timeout),
        "max_retries": config.max_retries,
        "sampling_args": sampling_args,
    }
    agent_kwargs = {
        "max_steps": config.max_steps or 25,
        "max_depth": config.max_depth,
        "max_concurrent_subagents": config.max_concurrent_subagents,
        "max_subagents_per_agent": config.max_subagents_per_agent,
        "max_run_seconds": config.max_run_seconds,
        "max_observation_chars": config.max_observation_chars,
    }

    run = await asyncio.to_thread(
        run_registered_environment,
        "textcraft_synth",
        model_config=model_config,
        environment_kwargs=environment_kwargs,
        agent_kwargs=agent_kwargs,
        model_overrides=model_overrides,
    )
    report = run.environment_report
    reward = float(bool(report.get("success")) and bool(report.get("finished")))
    return {
        "adapter": "curagent",
        "task_id": str(task.id),
        "reward": reward,
        "environment_report": report,
        "curagent_trace": run.to_trace_dict(),
        # The workflow replaces this placeholder with AReaL completion IDs.
        "trajectories": {},
    }


def _ensure_curagent_importable(curagent_root: str) -> None:
    if not curagent_root:
        raise ValueError("workflow_config.rollout_config.curagent_root must be set")
    root = Path(curagent_root).expanduser().resolve()
    package = root / "recursive_agent"
    if not package.is_dir():
        raise FileNotFoundError(f"CurAgent package not found under {root}")
    with _IMPORT_LOCK:
        root_string = str(root)
        if root_string not in sys.path:
            sys.path.insert(0, root_string)


__all__ = [
    "CurAgentRolloutConfig",
    "cached_completions_to_trajectory_collection",
    "curagent_trace_to_trajectory_collection",
    "run_curagent_synth_rollout",
    "task_to_curagent_sample",
]
