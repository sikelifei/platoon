from types import SimpleNamespace

from platoon.envs.base import Task

from platoon.textcraft.curagent_areal import (
    cached_completions_to_trajectory_collection,
    curagent_trace_to_trajectory_collection,
    task_to_curagent_sample,
)


def test_task_to_curagent_sample_recovers_per_execution_recipes() -> None:
    task = Task(
        id="textcraft_synth.train.7",
        misc={
            "target_items": {"final": 3},
            "initial_inventory": {"ore": 8},
            "gold_trajectory": [
                {
                    "action": "craft",
                    "target": ["part", 4],
                    "ingredients": {"ore": 8},
                    "result_count": 12,
                },
                {
                    "action": "craft",
                    "target": ["final", 3],
                    "ingredients": {"part": 6},
                    "result_count": 3,
                },
            ],
            "difficulty": "medium",
            "max_depth": 5,
        },
    )

    sample = task_to_curagent_sample(task)

    assert sample["split"] == "train"
    assert sample["recipes"]["part"] == [
        {"ingredients": {"ore": 2}, "result_count": 3}
    ]
    assert sample["recipes"]["final"] == [
        {"ingredients": {"part": 2}, "result_count": 1}
    ]


def test_cached_completions_become_independent_trainable_trajectories() -> None:
    collection = cached_completions_to_trajectory_collection(
        ["chatcmpl-a", "chatcmpl-b"],
        task_id="task-1",
        reward=1.0,
    )

    trajectories = list(collection["trajectories"].values())
    assert len(trajectories) == 2
    assert [
        trajectory["steps"][0]["misc"]["action_misc"]["completion_id"]
        for trajectory in trajectories
    ] == ["chatcmpl-a", "chatcmpl-b"]
    assert all(trajectory["reward"] == 1.0 for trajectory in trajectories)


def test_curagent_trace_restores_parent_child_trajectories() -> None:
    root_response = "```repl\nprint('root')\n```"
    child_response = "```repl\nprint('child')\n```"
    trace = {
        "agent_id": "root",
        "parent_id": None,
        "task": "craft final",
        "system_prompt": "root system",
        "steps": [{"response": root_response}],
        "children": [
            {
                "agent_id": "child",
                "parent_id": "root",
                "task": "craft part",
                "system_prompt": "child system",
                "steps": [{"response": child_response}],
                "children": [],
            }
        ],
    }
    completions = {
        "chatcmpl-root": _interaction("root system", "Task:\ncraft final", root_response),
        "chatcmpl-child": _interaction(
            "child system",
            "Delegated task:\ncraft part",
            child_response,
        ),
    }

    collection = curagent_trace_to_trajectory_collection(
        trace,
        completions,
        task_id="task-1",
        reward=1.0,
    )

    assert collection["trajectories"]["root"]["parent_info"] is None
    assert collection["trajectories"]["child"]["parent_info"] == {
        "id": "root",
        "fork_step": 0,
    }
    assert (
        collection["trajectories"]["child"]["steps"][0]["misc"]["action_misc"][
            "completion_id"
        ]
        == "chatcmpl-child"
    )


def _interaction(system: str, user: str, response: str) -> SimpleNamespace:
    return SimpleNamespace(
        messages=[{"content": system}, {"content": user}],
        completion=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response))]
        ),
    )
