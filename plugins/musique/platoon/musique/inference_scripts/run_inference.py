"""Run Platoon inference over MuSiQue JSONL examples."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from platoon.inference import (
    DefaultInferenceGroupWorkflow,
    InferenceBenchmarkConfig,
    InferenceBenchmarkRunner,
)
from platoon.utils.config import load_config

from platoon.musique.evaluation import score_answer
from platoon.musique.rollout import run_recursive_rollout, run_rollout
from platoon.musique.tasks import DEFAULT_HF_DATASET, get_task, get_task_ids

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class MuSiQueInferenceConfig:
    inference: InferenceBenchmarkConfig
    dataset_variant: Literal["ans", "full"] = "ans"
    dataset_split: Literal["train", "dev", "validation", "test"] = "dev"
    data_dir: str = "/data2/zhangwenjian/agent/musique_data"
    hf_dataset_name: str = DEFAULT_HF_DATASET
    prompt_path: str = "prompts/musique.md"
    num_tasks: int = 10
    use_recursive_agent: bool = True
    task_id: str | None = None
    stage: Literal["full", "rollouts", "report"] = "full"
    shuffle_tasks: bool = False
    seed: int = 42


_RUNTIME_DATA_DIR: str | None = None
_RUNTIME_HF_DATASET = DEFAULT_HF_DATASET
_RUNTIME_PROMPT_PATH: str | None = None


def _configured_task(task_id: str):
    return get_task(
        task_id,
        data_dir=_RUNTIME_DATA_DIR,
        hf_dataset_name=_RUNTIME_HF_DATASET,
        prompt_path=_RUNTIME_PROMPT_PATH,
    )


def reward_processor(traj: dict) -> tuple[float, dict[str, float]]:
    components: dict[str, float] = {}
    for step in traj.get("steps", []):
        reward_misc = step.get("misc", {}).get("reward_misc", {})
        if not isinstance(reward_misc, dict):
            continue
        for key in ("answer_em", "answer_f1", "reward/success"):
            if key in reward_misc:
                components[key] = components.get(key, 0.0) + float(reward_misc[key])
    score = components.get("reward/success", components.get("answer_f1", 0.0))
    if not components:
        score = float(traj.get("reward", 0.0))
    return score, components


def get_dataset_task_ids(config: MuSiQueInferenceConfig) -> list[str]:
    if config.task_id is not None:
        return [config.task_id]
    task_ids = get_task_ids(
        dataset_variant=config.dataset_variant,
        split=config.dataset_split,
        num_tasks=config.num_tasks,
        shuffle=config.shuffle_tasks,
        seed=config.seed,
        data_dir=config.data_dir,
        hf_dataset_name=config.hf_dataset_name,
    )
    if not task_ids:
        raise ValueError("No MuSiQue tasks matched the requested configuration.")
    return task_ids


async def main(args: list[str]) -> None:
    default_config = Path(__file__).parent.parent / "configs" / "inference" / "musique_inference.yaml"
    config, _ = load_config(
        args=args,
        config_class=MuSiQueInferenceConfig,
        default_config_path=str(default_config),
    )

    prompt_path = Path(config.prompt_path).expanduser()
    if not prompt_path.is_absolute():
        prompt_path = (Path(__file__).parent.parent / prompt_path).resolve()

    global _RUNTIME_DATA_DIR, _RUNTIME_HF_DATASET, _RUNTIME_PROMPT_PATH
    _RUNTIME_DATA_DIR = config.data_dir
    _RUNTIME_HF_DATASET = config.hf_dataset_name
    _RUNTIME_PROMPT_PATH = str(prompt_path)
    os.environ["MUSIQUE_DATA_DIR"] = config.data_dir
    os.environ["MUSIQUE_PROMPT_PATH"] = str(prompt_path)

    if config.inference.workflow.use_subprocesses:
        raise ValueError(
            "MuSiQue inference currently uses an in-process configured task loader; "
            "set inference.workflow.use_subprocesses=false."
        )

    if config.stage == "report":
        dataset = []
    else:
        dataset = [{"task_id": task_id} for task_id in get_dataset_task_ids(config)]

    workflow = DefaultInferenceGroupWorkflow(
        rollout_fn=run_recursive_rollout if config.use_recursive_agent else run_rollout,
        get_task_fn=_configured_task,
        config=config.inference.workflow,
        model_name=config.inference.model_name,
        model_endpoint=config.inference.model_endpoint,
        model_api_key=config.inference.model_api_key,
        reward_processor=reward_processor,
    )
    runner = InferenceBenchmarkRunner(workflow=workflow, output_dir=config.inference.output_dir)
    result = await runner.arun(
        dataset=dataset,
        resume=config.inference.resume,
        run_rollouts=config.stage in {"full", "rollouts"},
        generate_report=config.stage in {"full", "report"},
    )

    if "summary" in result:
        logger.info("MuSiQue inference report saved under %s", config.inference.output_dir)
        print(json.dumps(result["summary"], indent=2))
    else:
        logger.info("MuSiQue rollout stage complete: %s", config.inference.output_dir)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
