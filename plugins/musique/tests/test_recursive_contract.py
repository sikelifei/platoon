import asyncio

from platoon.envs.base import Task

from platoon.musique.agent import MuSiQueRecursivePromptBuilder
from platoon.musique.env import MuSiQueCodeExecutor, MuSiQueRecursiveCodeExecutor
from platoon.musique.recursive_env import agent_safe_misc


def _task() -> Task:
    return Task(
        goal="Who designed the Analytical Engine?",
        id="toy-0",
        max_steps=10,
        fork_strategy="task",
        misc={
            "source_id": "toy-0",
            "answer": "SECRET",
            "question_decomposition": [{"answer": "HIDDEN"}],
            "paragraphs": [
                {
                    "id": "doc-a",
                    "title": "Ada Lovelace",
                    "paragraph_text": "Ada Lovelace wrote notes on the Analytical Engine.",
                    "is_supporting": True,
                },
                {
                    "id": "doc-b",
                    "title": "Analytical Engine",
                    "paragraph_text": "The Analytical Engine was designed by Charles Babbage.",
                    "is_supporting": False,
                },
            ],
        },
    )


def test_action_spaces_and_safe_child_metadata():
    task = _task()
    assert [action.__name__ for action in MuSiQueCodeExecutor(task).actions] == ["search", "finish"]
    assert [action.__name__ for action in MuSiQueRecursiveCodeExecutor(task).actions] == [
        "launch_subagent",
        "search",
        "finish",
        "asyncio",
    ]
    safe = agent_safe_misc(task)
    assert "answer" not in safe
    assert "question_decomposition" not in safe
    assert "is_supporting" not in repr(safe)


def test_search_is_plain_text_and_deterministic():
    result = MuSiQueCodeExecutor(_task()).search("who designed engine")
    assert "Charles Babbage" in result
    assert "doc-b" in result
    assert "is_supporting" not in result


def test_executor_protects_actions_and_strips_provider_suffixes():
    from platoon.musique.recursive_env import (
        _clean_generated_code,
        _protected_action_rebinding,
    )

    redefining = (
        "async def search(query):\n"
        "    return None\n"
        "result = search('engine')"
    )
    assert _protected_action_rebinding(redefining) == "search"
    assert _protected_action_rebinding("del finish") == "finish"
    assert _protected_action_rebinding('search("engine")') is None
    assert _clean_generated_code('print("ok")</｜DSML｜>') == 'print("ok")'


def test_prompt_contract():
    prompt = MuSiQueRecursivePromptBuilder().build_system_prompt(None).lower()
    for term in (
        "dependency and delegation decision:",
        "independent branches",
        "linear chain of four or more",
        "canonical answer span",
        "fork-join cues and examples:",
        "never combine truly independent anchor descriptions",
        "two subproblems are independent only if",
        "<thought>",
        "evidence",
    ):
        assert term in prompt
    for term in ("budget", "depth", "memory", "json", "xml", "\nplan:"):
        assert term not in prompt
    assert "do not delegate the complete original question" in prompt
    assert "delegation is optional" in prompt
    assert 'correct: finish("example entity")' in prompt
    assert "colin firth" not in prompt
    assert 'correct: finish("100")' not in prompt
    assert "search(...) is synchronous" in prompt
    assert "launch_subagent(...) is asynchronous" in prompt


def test_recursive_executor_fork_preserves_type():
    async def run():
        child = await MuSiQueRecursiveCodeExecutor(_task(), subagent_max_steps=7).fork(_task())
        assert isinstance(child, MuSiQueRecursiveCodeExecutor)
        assert child.subagent_max_steps == 7

    asyncio.run(run())


def test_native_recursive_trace_preserves_parent_links_and_plain_child_tasks():
    from types import SimpleNamespace

    from platoon.config_defs import InferenceParams
    from platoon.episode.context import budget_tracker, current_trajectory_collection
    from platoon.episode.loop import run_episode
    from platoon.episode.trajectory import DepthAwareStepBudgetTracker, TrajectoryCollection
    from platoon.musique.recursive_agent import MuSiQueRecursiveAgent
    from platoon.musique.recursive_env import MuSiQueRecursiveEnv

    class FakeClient:
        def __init__(self, depth=0):
            self.depth, self.calls, self.model = depth, 0, "fake"

        def fork(self):
            return FakeClient(self.depth + 1)

        async def aclose(self):
            return None

        async def async_chat_completion(self, prompt, **kwargs):
            self.calls += 1
            if self.depth == 0 and self.calls == 1:
                code = 'result = await launch_subagent("Find the designer and return evidence")\nprint(result)'
            elif self.depth == 1 and self.calls == 1:
                code = 'result = await launch_subagent("Confirm the designer from the passages")\nprint(result)'
            elif self.calls == 1:
                code = 'result = search("who designed the Analytical Engine")\nprint(result)'
            else:
                code = 'finish("Charles Babbage")'
            content = f"<thought>work</thought>\n<python>{code}</python>"
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(to_dict=lambda: {}),
                model="fake",
                id=f"fake-{self.depth}-{self.calls}",
            )

    task = _task()
    task.misc["answer"] = "Charles Babbage"
    collection = TrajectoryCollection()
    current_trajectory_collection.set(collection)
    budget_tracker.set(DepthAwareStepBudgetTracker(max_depth=4))
    root = asyncio.run(
        run_episode(
            MuSiQueRecursiveAgent(
                llm_client=FakeClient(),
                inference_params=InferenceParams(max_completion_tokens=256),
            ),
            MuSiQueRecursiveEnv(task, subagent_max_steps=4),
            timeout=30,
        )
    )
    assert len(collection.trajectories) == 3
    trajectories = list(collection.trajectories.values())
    assert root.finish_message == "Charles Babbage"
    assert trajectories[1].parent_info.id == root.id
    assert trajectories[2].parent_info.id == trajectories[1].id
    assert all("answer" not in (item.task.misc if item.task else {}) for item in trajectories[1:])
