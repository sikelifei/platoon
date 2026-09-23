"""Single-task rollout execution for MuSiQue inference."""

from __future__ import annotations

import asyncio
import logging
import os

from platoon.config_defs import RolloutConfig
from platoon.envs.base import Task
from platoon.episode.context import current_trajectory_collection
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import TrajectoryCollection
from platoon.inference import InferenceBenchmarkConfig
from platoon.utils.llm_client import LiteLLMClient
from platoon.visualization.event_sinks import JsonlFileSink

from .single_agent import MuSiQueSingleAgent
from .single_env import MuSiQueSingleEnv

logger = logging.getLogger("platoon.musique.rollout")


async def run_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    agent = None
    env = None
    try:
        llm_client = LiteLLMClient(
            model=config.model_name,
            base_url=config.model_endpoint,
            api_key=config.model_api_key,
        )
        env = MuSiQueSingleEnv(task)
        agent = MuSiQueSingleAgent(
            llm_client=llm_client,
            inference_params=config.inference_params,
            prompt_path=task.misc.get("prompt_path"),
        )

        trajectory_collection = TrajectoryCollection()
        current_trajectory_collection.set(trajectory_collection)
        events_dir = os.path.join(config.output_dir, "events")
        os.makedirs(events_dir, exist_ok=True)
        events_path = os.path.join(
            events_dir,
            f"events_{task.id}_{trajectory_collection.id}.jsonl",
        )
        trajectory_collection.register_event_handlers(
            JsonlFileSink(
                events_path,
                collection_id=trajectory_collection.id,
                process_id=os.getpid(),
            )
        )

        if config.verbose:
            logger.info("Starting MuSiQue rollout for task %s", task.id)

        rollout_task = asyncio.create_task(
            run_episode(agent, env, timeout=config.step_timeout)
        )
        try:
            await asyncio.wait_for(rollout_task, timeout=config.timeout)
        except asyncio.TimeoutError:
            rollout_task.cancel()
            try:
                await asyncio.wait_for(rollout_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.warning("Timed out while cancelling task %s", task.id)
            raise

        if config.return_dict:
            return trajectory_collection.to_dict()
        return trajectory_collection
    finally:
        if agent is not None:
            await agent.close()
        if env is not None:
            await env.close()

from platoon.episode.context import budget_tracker
from platoon.episode.trajectory import DepthAwareStepBudgetTracker
from platoon.utils.subagent_rewards import propogate_root_success

from .recursive_agent import MuSiQueRecursiveAgent
from .recursive_env import MuSiQueRecursiveEnv


async def run_recursive_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    """Run a recursive rollout while preserving Platoon's trajectory tree."""
    agent = None
    env = None
    try:
        llm_client = LiteLLMClient(
            model=config.model_name,
            base_url=config.model_endpoint,
            api_key=config.model_api_key,
        )
        env = MuSiQueRecursiveEnv(task)
        agent = MuSiQueRecursiveAgent(
            llm_client=llm_client,
            inference_params=config.inference_params,
            prompt_path=task.misc.get("prompt_path"),
        )
        trajectory_collection = TrajectoryCollection()
        current_trajectory_collection.set(trajectory_collection)
        budget_tracker.set(DepthAwareStepBudgetTracker(max_depth=4))
        events_dir = os.path.join(config.output_dir, "events")
        os.makedirs(events_dir, exist_ok=True)
        events_path = os.path.join(
            events_dir,
            f"events_{task.id}_{trajectory_collection.id}.jsonl",
        )
        trajectory_collection.register_event_handlers(
            JsonlFileSink(
                events_path,
                collection_id=trajectory_collection.id,
                process_id=os.getpid(),
            )
        )
        rollout_task = asyncio.create_task(
            run_episode(agent, env, timeout=config.step_timeout)
        )
        try:
            await asyncio.wait_for(rollout_task, timeout=config.timeout)
        except asyncio.TimeoutError:
            rollout_task.cancel()
            try:
                await asyncio.wait_for(rollout_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.warning("Timed out while cancelling task %s", task.id)
            raise
        result = trajectory_collection.to_dict() if config.return_dict else trajectory_collection
        if config.propogate_root_success:
            result = propogate_root_success(result)
        return result
    finally:
        if agent is not None:
            await agent.close()
        if env is not None:
            await env.close()
