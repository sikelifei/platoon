import json
import os
import subprocess
import sys
from pathlib import Path

from platoon.envs.base import Task
from platoon.utils.config import load_yaml_config

from platoon.fanoutqa.inference_scripts.run_inference import _collect_exports

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def test_single_and_recursive_configs_hold_all_baseline_settings_equal():
    config_root = PLUGIN_ROOT / "platoon" / "fanoutqa" / "configs"
    single = load_yaml_config(config_root / "qwen_single.yaml")
    recursive = load_yaml_config(config_root / "qwen_recursive.yaml")

    assert single["use_recursive_agent"] is False
    assert recursive["use_recursive_agent"] is True
    assert single["dataset_split"] == recursive["dataset_split"] == "dev"
    assert single["num_tasks"] == recursive["num_tasks"] == 20
    assert single["prompt_path"] == recursive["prompt_path"] is None
    assert single["inference"]["model_name"] == recursive["inference"]["model_name"]
    assert single["inference"]["model_endpoint"] == recursive["inference"]["model_endpoint"]
    assert single["inference"]["model_api_key"] == recursive["inference"]["model_api_key"]
    assert single["inference"]["workflow"]["rollout_config"] == recursive["inference"]["workflow"]["rollout_config"]
    assert single["inference"]["workflow"]["num_rollouts_per_task"] == 1
    assert single["inference"]["workflow"]["num_concurrent_workers"] == 4
    assert single["inference"]["workflow"]["rollout_config"]["max_steps"] == 30
    assert single["recursive"]["enabled"] is False
    assert recursive["recursive"]["enabled"] is True
    assert recursive["recursive"]["subagent_max_steps"] == 15
    assert recursive["recursive"]["max_depth"] == 4
    assert single["wikipedia"] == recursive["wikipedia"]


def test_exports_keep_original_ids_full_trees_events_and_missing_attempts(tmp_path):
    task_one = Task(
        id="fanoutqa.dev.q-one",
        goal="Question one",
        max_steps=30,
        misc={"source_id": "q-one", "dataset_split": "dev", "categories": ["a"]},
    )
    task_two = Task(
        id="fanoutqa.dev.q-two",
        goal="Question two",
        max_steps=30,
        misc={"source_id": "q-two", "dataset_split": "dev", "categories": ["b"]},
    )
    rollout = tmp_path / "rollouts" / "fanoutqa.dev.q-one" / "rollout_0"
    (rollout / "events").mkdir(parents=True)
    collection = {
        "id": "collection-one",
        "trajectories": {
            "root-id": {
                "id": "root-id",
                "parent_info": None,
                "finish_message": "Official answer from root",
                "error_message": "native root error retained",
                "reward": 0.5,
                "misc": {
                    "fanoutqa_parallel_metrics": {
                        "parallel_branches": 1,
                        "max_parallel_branches": 2,
                    },
                    "fanoutqa_analysis": {
                        "wiki_searches": 2,
                        "wiki_content_calls": 1,
                    },
                },
                "steps": [
                    {
                        "code": "wiki_search('sample')",
                        "misc": {
                            "action_misc": {
                                "usage": {
                                    "prompt_tokens": 7,
                                    "completion_tokens": 3,
                                    "total_tokens": 10,
                                }
                            }
                        },
                    }
                ],
            },
            "child-id": {
                "id": "child-id",
                "parent_info": {"id": "root-id", "fork_step": 1},
                "finish_message": "child result",
                "error_message": None,
                "reward": 0.0,
                "misc": {
                    "fanoutqa_analysis": {
                        "wiki_searches": 1,
                        "wiki_content_calls": 2,
                    }
                },
                "steps": [
                    {
                        "code": "wiki_content(evidence)",
                        "misc": {
                            "action_misc": {
                                "usage": {
                                    "prompt_tokens": 11,
                                    "completion_tokens": 5,
                                    "total_tokens": 16,
                                }
                            }
                        },
                    }
                ],
            },
            "grandchild-id": {
                "id": "grandchild-id",
                "parent_info": {"id": "child-id", "fork_step": 1},
                "finish_message": "grandchild result",
                "error_message": None,
                "reward": 0.0,
                "misc": {},
                "steps": [{"code": "finish('done')", "misc": {}}],
            },
        },
    }
    (rollout / "trajectory_collection.json").write_text(json.dumps(collection), encoding="utf-8")
    (rollout / "events" / "events_collection-one.jsonl").write_text('{"event":"created"}\n', encoding="utf-8")

    result = _collect_exports(
        tasks=[task_one, task_two],
        output_dir=tmp_path,
        rollout_count=1,
        model_name="openai/Qwen3-4B-Instruct-2507",
        split="dev",
        recursive=True,
    )

    generations = [
        json.loads(line) for line in (tmp_path / "generations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert generations == [
        {"id": "q-one", "answer": "Official answer from root"},
        {"id": "q-two", "answer": ""},
    ]
    assert (tmp_path / "trajectories" / "q-one_rollout_0.json").is_file()
    assert (tmp_path / "events" / "events_collection-one.jsonl").is_file()
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["num_requested_questions"] == 2
    assert metrics["num_questions_with_rollouts"] == 1
    assert metrics["num_rollouts_requested"] == 2
    assert metrics["num_rollouts_with_artifacts"] == 1
    assert metrics["num_rollouts_with_errors"] == 2
    assert metrics["per_rollout"][0]["error"] == "native root error retained"
    assert metrics["per_rollout"][1]["error"] == "missing trajectory collection"
    workflow = metrics["workflow_metrics_mean"]
    assert workflow["number_of_subagents"] == 2.0
    assert workflow["max_recursion_depth"] == 2.0
    assert workflow["root_steps"] == 1.0
    assert workflow["child_steps"] == 2.0
    assert workflow["total_steps"] == 3.0
    assert workflow["total_tokens"] == 26.0
    assert workflow["number_of_wiki_searches"] == 3.0
    assert workflow["number_of_wiki_content_calls"] == 3.0
    assert workflow["parallel_branches"] == 1.0
    assert workflow["max_parallel_branches"] == 2.0
    assert result["num_generations"] == 2


def test_cli_applies_wikipedia_config_before_official_package_import():
    environment = os.environ.copy()
    for name in (
        "FANOUTQA_WIKIPEDIA_TYPE",
        "FANOUTQA_KIWIX_BASE",
        "FANOUTQA_KIWIX_ZIMNAME",
    ):
        environment.pop(name, None)

    script = r"""
import sys
from pathlib import Path

from platoon.fanoutqa.inference_scripts.run_inference import (
    FanOutQAInferenceConfig,
    _apply_runtime_overrides,
    _select_tasks,
)
from platoon.utils.config import load_config

assert "fanoutqa" not in sys.modules
config, _ = load_config(
    args=[],
    config_class=FanOutQAInferenceConfig,
    default_config_path=str(Path("platoon/fanoutqa/configs/inference/fanoutqa_inference.yaml")),
)
config.num_tasks = 0
_apply_runtime_overrides(config)
try:
    _select_tasks(config)
except ValueError:
    pass

import fanoutqa.wiki as wiki
assert wiki.FANOUTQA_WIKIPEDIA_TYPE == "kiwix"
assert wiki.FANOUTQA_KIWIX_BASE == "http://127.0.0.1:8889"
assert wiki.FANOUTQA_KIWIX_ZIMNAME == "wikipedia_en_all_nopic_2023-09"
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=PLUGIN_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
