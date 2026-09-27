"""Run FanOutQA inference and export official-format generations."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Literal

from platoon.inference import (
    DefaultInferenceGroupWorkflow,
    InferenceBenchmarkConfig,
    InferenceBenchmarkRunner,
)
from platoon.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class RecursiveConfig:
    enabled: bool = True
    max_depth: int | None = 4
    subagent_max_steps: int = 15


@dataclass
class WikipediaConfig:
    type: str | None = "kiwix"
    kiwix_base: str | None = "http://127.0.0.1:8889"
    kiwix_zimname: str | None = "wikipedia_en_all_nopic_2023-09"


@dataclass
class FanOutQAInferenceConfig:
    inference: InferenceBenchmarkConfig
    dataset_split: Literal["dev", "test"] = "dev"
    prompt_path: str | None = None
    num_tasks: int | None = 20
    use_recursive_agent: bool = True
    task_id: str | None = None
    stage: Literal["full", "rollouts", "report"] = "full"
    shuffle_tasks: bool = False
    seed: int = 42
    recursive: RecursiveConfig = field(default_factory=RecursiveConfig)
    wikipedia: WikipediaConfig = field(default_factory=WikipediaConfig)


def _safe_task_id(task_id: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in task_id)


def _resolve_prompt_path(path: str | None) -> str | None:
    if not path:
        return None
    prompt_path = Path(path).expanduser()
    if not prompt_path.is_absolute():
        prompt_path = Path(__file__).parent.parent / prompt_path
    return str(prompt_path.resolve())


def _apply_runtime_overrides(config: FanOutQAInferenceConfig) -> None:
    inference = config.inference
    inference.model_name = os.getenv("FANOUTQA_MODEL_NAME", inference.model_name)
    inference.model_endpoint = os.getenv("FANOUTQA_MODEL_ENDPOINT", inference.model_endpoint)
    inference.model_api_key = os.getenv("FANOUTQA_MODEL_API_KEY", inference.model_api_key)

    env_values = {
        "FANOUTQA_WIKIPEDIA_TYPE": config.wikipedia.type,
        "FANOUTQA_KIWIX_BASE": config.wikipedia.kiwix_base,
        "FANOUTQA_KIWIX_ZIMNAME": config.wikipedia.kiwix_zimname,
    }
    for env_name, configured_value in env_values.items():
        value = os.getenv(env_name, configured_value)
        if value is not None:
            os.environ[env_name] = value

    root_budget = inference.workflow.rollout_config.max_steps
    if root_budget is None or root_budget < 1:
        raise ValueError("inference.workflow.rollout_config.max_steps must be a positive shared budget")
    if config.recursive.subagent_max_steps < 1:
        raise ValueError("recursive.subagent_max_steps must be positive")
    if config.recursive.max_depth is not None and config.recursive.max_depth < 0:
        raise ValueError("recursive.max_depth must be non-negative or null")
    if config.stage not in {"full", "rollouts", "report"}:
        raise ValueError("stage must be full, rollouts, or report")
    if config.inference.workflow.use_subprocesses:
        raise ValueError(
            "FanOutQA inference uses in-process task loading; set inference.workflow.use_subprocesses=false"
        )
    if config.inference.workflow.num_rollouts_per_task < 1:
        raise ValueError("inference.workflow.num_rollouts_per_task must be positive")


def _select_tasks(config: FanOutQAInferenceConfig):
    from platoon.fanoutqa.tasks import load_tasks

    max_steps = config.inference.workflow.rollout_config.max_steps
    tasks = load_tasks(split=config.dataset_split, max_steps=max_steps)
    if config.task_id is not None:
        matches = [
            task for task in tasks if task.id == config.task_id or str(task.misc.get("source_id")) == config.task_id
        ]
        if not matches:
            raise ValueError(f"Unknown FanOutQA {config.dataset_split} task id: {config.task_id}")
        tasks = matches
    elif config.shuffle_tasks:
        random.Random(config.seed).shuffle(tasks)

    if config.num_tasks is not None:
        if config.num_tasks < 0:
            raise ValueError("num_tasks must be non-negative or null")
        tasks = tasks[: config.num_tasks]
    if not tasks:
        raise ValueError("No FanOutQA tasks matched this configuration.")
    return tasks


def _extract_root(collection: dict) -> dict | None:
    trajectories = collection.get("trajectories")
    if not isinstance(trajectories, dict) or not trajectories:
        return None
    for trajectory in trajectories.values():
        if isinstance(trajectory, dict):
            parent = trajectory.get("parent_info")
            if not isinstance(parent, dict) or parent.get("id") is None:
                return trajectory
    return None


def _read_collection(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not read trajectory collection %s", path)
        return None


def _collect_exports(
    tasks: list,
    output_dir: Path,
    rollout_count: int,
    model_name: str,
    split: str,
    recursive: bool,
) -> dict:
    from platoon.fanoutqa.rollout import recursive_workflow_metrics

    rollouts_dir = output_dir / "rollouts"
    trajectories_dir = output_dir / "trajectories"
    events_dir = output_dir / "events"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    events_dir.mkdir(parents=True, exist_ok=True)

    generations: list[dict[str, str]] = []
    records: list[dict] = []
    for task in tasks:
        safe_id = _safe_task_id(task.id)
        for rollout_index in range(rollout_count):
            artifact_dir = rollouts_dir / safe_id / f"rollout_{rollout_index}"
            collection_path = artifact_dir / "trajectory_collection.json"
            source_id = str(task.misc["source_id"])
            collection = _read_collection(collection_path)
            if collection is None:
                if rollout_index == 0:
                    generations.append({"id": source_id, "answer": ""})
                records.append(
                    {
                        "id": source_id,
                        "rollout_index": rollout_index,
                        "reward": 0.0,
                        "answer": "",
                        "metrics": {},
                        "error": "missing trajectory collection",
                        "has_artifact": False,
                    }
                )
                continue

            target_trajectory = trajectories_dir / f"{_safe_task_id(source_id)}_rollout_{rollout_index}.json"
            shutil.copy2(collection_path, target_trajectory)
            for event_path in (artifact_dir / "events").glob("*.jsonl"):
                shutil.copy2(event_path, events_dir / event_path.name)

            root = _extract_root(collection)
            if root is None:
                answer = ""
                reward = 0.0
                metrics = {}
                error = "trajectory collection has no root trajectory"
            else:
                answer = str(root.get("finish_message") or "")
                reward = float(root.get("reward") or 0.0)
                metrics = recursive_workflow_metrics(collection)
                error = str(root.get("error_message") or "") or None
            records.append(
                {
                    "id": source_id,
                    "rollout_index": rollout_index,
                    "reward": reward,
                    "answer": answer,
                    "metrics": metrics,
                    "error": error,
                    "has_artifact": True,
                }
            )
            # Official evaluation accepts one answer per question ID. Export rollout zero
            # for multi-rollout runs without selecting by score.
            if rollout_index == 0:
                generations.append({"id": source_id, "answer": answer})

    generations_path = output_dir / "generations.jsonl"
    with generations_path.open("w", encoding="utf-8") as stream:
        for generation in generations:
            stream.write(json.dumps(generation, ensure_ascii=False) + "\n")

    metric_keys = sorted({key for record in records for key in record["metrics"]})
    aggregate_metrics = {
        key: mean(float(record["metrics"][key]) for record in records if key in record["metrics"])
        for key in metric_keys
    }
    metrics = {
        "dataset_split": split,
        "setting": "recursive" if recursive else "single",
        "model": model_name,
        "num_requested_questions": len(tasks),
        "num_questions_with_rollouts": len({record["id"] for record in records if record["has_artifact"]}),
        "num_rollouts_requested": len(tasks) * rollout_count,
        "num_rollouts_with_artifacts": sum(1 for record in records if record["has_artifact"]),
        "num_rollouts_with_errors": sum(1 for record in records if record["error"] is not None),
        "mean_root_reward": mean([record["reward"] for record in records]) if records else 0.0,
        "workflow_metrics_mean": aggregate_metrics,
        "per_rollout": records,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "generations_path": str(generations_path),
        "metrics_path": str(metrics_path),
        "num_generations": len(generations),
        "num_rollouts": len(records),
    }


def _load_default_config_path() -> str:
    return str(Path(__file__).parent.parent / "configs" / "inference" / "fanoutqa_inference.yaml")


async def _run(args: list[str]) -> dict:
    config, _ = load_config(
        args=args,
        config_class=FanOutQAInferenceConfig,
        default_config_path=_load_default_config_path(),
    )
    _apply_runtime_overrides(config)
    from platoon.fanoutqa.rollout import recursive_workflow_metrics, run_recursive_rollout, run_rollout

    tasks = _select_tasks(config)
    prompt_path = _resolve_prompt_path(config.prompt_path)

    task_by_id = {task.id: task for task in tasks}

    def configured_task(task_id: str):
        from copy import deepcopy

        task = deepcopy(task_by_id[task_id])
        task.max_steps = config.inference.workflow.rollout_config.max_steps
        task._fanoutqa_prompt_path = prompt_path
        task._fanoutqa_subagent_max_steps = config.recursive.subagent_max_steps
        task._fanoutqa_max_recursive_depth = config.recursive.max_depth
        return task

    rollout_fn = run_recursive_rollout if config.use_recursive_agent and config.recursive.enabled else run_rollout
    workflow = DefaultInferenceGroupWorkflow(
        rollout_fn=rollout_fn,
        get_task_fn=configured_task,
        config=config.inference.workflow,
        model_name=config.inference.model_name,
        model_endpoint=config.inference.model_endpoint,
        model_api_key=config.inference.model_api_key,
        reward_processor=lambda root: (
            float(root.get("reward", 0.0)),
            {"reward/fanoutqa_official_loose": float(root.get("reward", 0.0))},
        ),
        workflow_metrics_fn=recursive_workflow_metrics,
    )
    output_dir = Path(config.inference.output_dir).expanduser().resolve()
    runner = InferenceBenchmarkRunner(workflow=workflow, output_dir=str(output_dir))
    dataset = [{"task_id": task.id} for task in tasks] if config.stage in {"full", "rollouts"} else []
    result = await runner.arun(
        dataset=dataset,
        resume=config.inference.resume,
        run_rollouts=config.stage in {"full", "rollouts"},
        generate_report=config.stage in {"full", "report"},
    )
    if config.stage == "rollouts":
        result = runner.generate_report()

    exports = _collect_exports(
        tasks=tasks,
        output_dir=output_dir,
        rollout_count=config.inference.workflow.num_rollouts_per_task,
        model_name=config.inference.model_name,
        split=config.dataset_split,
        recursive=bool(config.use_recursive_agent and config.recursive.enabled),
    )
    result["exports"] = exports
    logger.info("FanOutQA output saved under %s", output_dir)
    return result


def _evaluate_cli(args: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run the official FanOutQA evaluator on generations.jsonl.")
    parser.add_argument("--predictions", required=True, help="Official-format generations JSONL")
    parser.add_argument("--split", choices=("dev",), default="dev")
    parser.add_argument("--output", help="JSON result path (default: alongside predictions)")
    parser.add_argument("--only-score-answered", action="store_true")
    parser.add_argument("--llm-cache-key")
    options = parser.parse_args(args)

    predictions_path = Path(options.predictions).expanduser()
    predictions = []
    seen_ids: set[str] = set()
    with predictions_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if (
                set(value) != {"id", "answer"}
                or not isinstance(value["id"], str)
                or not isinstance(value["answer"], str)
            ):
                raise ValueError(
                    f"Invalid official generation on line {line_number}; expected only string id and answer."
                )
            if value["id"] in seen_ids:
                raise ValueError(f"Duplicate official question id in predictions: {value['id']}")
            seen_ids.add(value["id"])
            predictions.append(value)

    from fanoutqa import load_dev
    from platoon.fanoutqa.evaluation import evaluate_answers

    all_questions = load_dev()
    questions_by_id = {question.id: question for question in all_questions}
    missing = sorted(seen_ids - questions_by_id.keys())
    if missing:
        raise ValueError(f"Predictions contain question IDs absent from the official dev set: {missing[:5]}")

    evaluation = evaluate_answers(
        predictions,
        questions=all_questions,
        only_score_answered=options.only_score_answered,
        llm_cache_key=options.llm_cache_key,
    )
    output_path = (
        Path(options.output).expanduser() if options.output else predictions_path.with_name("official_evaluation.json")
    )
    output_path.write_text(json.dumps(asdict(evaluation), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(asdict(evaluation), indent=2, ensure_ascii=False))
    logger.info("Official FanOutQA evaluation saved to %s", output_path)
    return 0


def main(args: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if args is None else args)
    if args and args[0] == "evaluate":
        return _evaluate_cli(args[1:])
    result = asyncio.run(_run(args))
    print(json.dumps(result.get("summary", result), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
