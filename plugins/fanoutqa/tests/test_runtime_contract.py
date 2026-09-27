import asyncio

import fanoutqa
from platoon.config_defs import InferenceParams
from platoon.envs.base import Task

from platoon.fanoutqa.agent import (
    FanOutQAPromptBuilder,
    FanOutQARecursiveAgent,
)
from platoon.fanoutqa.env import (
    FanOutQARecursiveCodeExecutor,
    FanOutQARecursiveEnv,
    FanOutQASingleCodeExecutor,
)


class FakeClient:
    def fork(self):
        return FakeClient()

    async def aclose(self):
        return None


def _root_task():
    return Task(
        id="root",
        goal="Find a safe factual answer.",
        max_steps=30,
        misc={
            "source_id": "synthetic-id",
            "dataset_split": "dev",
            "categories": ["synthetic"],
            "answer": "GoldLeakCanary-8472",
            "decomposition": ["PrivateDecompositionCanary-391"],
            "necessary_evidence": ["PrivateEvidenceCanary-625"],
        },
        fork_strategy="task",
    )


def test_single_and_recursive_action_spaces_and_retrieval_identity(monkeypatch):
    task = _root_task()
    single = FanOutQASingleCodeExecutor(task)
    recursive = FanOutQARecursiveCodeExecutor(task, subagent_max_steps=15)

    single_actions = [action.__name__ for action in single.actions]
    recursive_actions = [action.__name__ for action in recursive.actions]
    assert single_actions == ["wiki_search", "wiki_content", "finish"]
    assert "launch_subagent" not in single_actions
    assert "wiki_search" in recursive_actions
    assert "wiki_content" in recursive_actions
    assert "launch_subagent" in recursive_actions
    assert "finish" in recursive_actions

    single_description = asyncio.run(single.describe_action_space())
    recursive_description = asyncio.run(recursive.describe_action_space())
    assert "launch_subagent" not in single_description
    assert "launch_subagent" in recursive_description
    assert "async def" not in single_description
    assert recursive_description.count("async def") == 1
    assert "async def launch_subagent(goal: str) -> str" in recursive_description
    assert "result = await launch_subagent(goal)" in recursive_description
    assert "await asyncio.gather(" in recursive_description
    assert "The child does not see the parent's previous search history" in recursive_description
    assert "include all\n   entities and context required" in recursive_description
    for description in (single_description, recursive_description):
        assert "def wiki_search(query: str, results: int = 10)" in description
        assert "This function is synchronous; do not await it." in description
        assert "def wiki_content(evidence) -> str" in description
        assert "Pass the returned object directly; do not reconstruct it." in description
        assert "def finish(message: str) -> str" in description
        assert "This function is synchronous." in description
        assert "await wiki_search" not in description
        assert "await wiki_content" not in description
        assert "await finish" not in description

    evidence = object()
    results = [evidence]
    page = "Official markdown page"
    calls = []

    def wiki_search(query, results=10):
        calls.append(("search", query, results))
        return results_value

    def wiki_content(argument):
        calls.append(("content", argument))
        return page

    results_value = results
    monkeypatch.setattr(fanoutqa, "wiki_search", wiki_search)
    monkeypatch.setattr(fanoutqa, "wiki_content", wiki_content)

    returned_results = single.wiki_search("  exact query  ", results=7)
    returned_page = single.wiki_content(evidence)
    assert returned_results is results
    assert returned_page is page
    assert calls == [
        ("search", "  exact query  ", 7),
        ("content", evidence),
    ]


def test_recursive_agent_and_root_child_grandchild_use_homogeneous_isolated_contexts():
    async def run():
        root = _root_task()
        root_env = FanOutQARecursiveEnv(root, subagent_max_steps=15)
        root_env.code_executor.shell.user_ns["root_only_canary"] = "GoldLeakCanary-8472"

        child_misc = {
            "source_id": "synthetic-id",
            "dataset_split": "dev",
            "_is_subagent": True,
        }
        child_task = root_env.task.fork(
            "Find one supporting fact.",
            max_steps=15,
            task_misc=child_misc,
        )
        child_env = await root_env.fork(child_task)
        child_env.code_executor.shell.user_ns["child_only_canary"] = "PrivateChildCanary-102"
        grandchild_task = child_env.task.fork(
            "Check the delegated supporting fact.",
            max_steps=5,
            task_misc=child_misc.copy(),
        )
        grandchild_env = await child_env.fork(grandchild_task)

        try:
            safe_expected = {"source_id", "dataset_split", "_is_subagent"}
            assert set(root_env.task.misc) == {"source_id", "dataset_split", "categories"}
            assert set(child_env.task.misc) == safe_expected
            assert set(grandchild_env.task.misc) == safe_expected
            for task in (root_env.task, child_env.task, grandchild_env.task):
                serialized = repr(task.misc)
                assert "GoldLeakCanary-8472" not in serialized
                assert "PrivateDecompositionCanary-391" not in serialized
                assert "PrivateEvidenceCanary-625" not in serialized

            assert "root_only_canary" not in child_env.code_executor.shell.user_ns
            assert "root_only_canary" not in grandchild_env.code_executor.shell.user_ns
            assert "child_only_canary" not in grandchild_env.code_executor.shell.user_ns
            assert "parent_state" not in child_env._state.misc
            assert "parent_state" not in grandchild_env._state.misc
            assert isinstance(child_env.code_executor, FanOutQARecursiveCodeExecutor)
            assert isinstance(grandchild_env.code_executor, FanOutQARecursiveCodeExecutor)

            root_builder = FanOutQAPromptBuilder(recursive=True)
            single_builder = FanOutQAPromptBuilder(recursive=False)
            recursive_prompt = root_builder.system_prompt
            single_prompt = single_builder.system_prompt
            headings = [
                "RESEARCH STRATEGY:",
                "DELEGATION STRATEGY:",
                "ANSWER SUBMISSION:",
                "OTHER TIPS:",
            ]
            assert [recursive_prompt.index(heading) for heading in headings] == sorted(
                recursive_prompt.index(heading) for heading in headings
            )
            assert "newly discovered entities or facts" in recursive_prompt
            assert "multiple independent branches" in recursive_prompt
            assert "Do not delegate the current task unchanged" in recursive_prompt
            assert "launch_subagent" in recursive_prompt
            assert "DELEGATION STRATEGY:" not in single_prompt
            assert "launch_subagent" not in single_prompt
            assert "specific search terms" not in recursive_prompt
            assert "wiki_search(" not in recursive_prompt
            assert "wiki_content(" not in recursive_prompt
            assert "Use the available tools according to the Action Space." in recursive_prompt
            assert "You are a deep research agent" in single_prompt
            assert "Python plus Wikipedia search tools." in single_prompt
            assert "you can delegate subproblems to subagents" not in single_prompt
            common_paragraphs = []
            for index, paragraph in enumerate(recursive_prompt.split("\n\n")):
                if index == 0:
                    paragraph = paragraph.replace(", and you can delegate subproblems to subagents", "")
                if "launch_subagent" not in paragraph.lower() and "subagent" not in paragraph.lower():
                    common_paragraphs.append(paragraph)
            assert single_prompt == "\n\n".join(common_paragraphs)

            root_agent = FanOutQARecursiveAgent(
                prompt_builder=root_builder,
                llm_client=FakeClient(),
                inference_params=InferenceParams(temperature=0.0),
            )
            child_agent = await root_agent.fork(child_task)
            grandchild_agent = await child_agent.fork(grandchild_task)
            assert type(child_agent) is FanOutQARecursiveAgent
            assert type(grandchild_agent) is FanOutQARecursiveAgent
            assert child_agent.prompt_builder is root_agent.prompt_builder
            assert grandchild_agent.prompt_builder is root_agent.prompt_builder
        finally:
            await grandchild_env.close()
            await child_env.close()
            await root_env.close()

    asyncio.run(run())


def test_custom_prompt_reaches_first_llm_request(tmp_path):
    from types import SimpleNamespace

    from platoon.envs.codeact import CodeActObservation

    from platoon.fanoutqa.agent import FanOutQARecursiveAgent

    prompt_path = tmp_path / "custom_prompt.md"
    prompt_path.write_text("CUSTOM_PROMPT_READY\nUse the available evidence.", encoding="utf-8")

    class CapturingClient:
        def __init__(self):
            self.messages = None

        async def async_chat_completion(self, messages, **kwargs):
            self.messages = messages
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="<thought>ready</thought>\n<python>finish('done')</python>")
                    )
                ],
                usage=SimpleNamespace(to_dict=lambda: {}),
                model="fake",
                id="fake-completion",
            )

        def fork(self):
            return self

        async def aclose(self):
            return None

    async def run():
        task = Task(id="prompt-test", goal="Answer safely.", max_steps=30, misc={})
        client = CapturingClient()
        agent = FanOutQARecursiveAgent(
            llm_client=client,
            prompt_builder=FanOutQAPromptBuilder(recursive=True, prompt_path=prompt_path),
            inference_params=InferenceParams(temperature=0.0),
        )
        observation = CodeActObservation(task=task, action_space="finish(message)", history=[])
        await agent.act(observation)
        assert client.messages is not None
        assert client.messages[0]["role"] == "system"
        assert "CUSTOM_PROMPT_READY" in client.messages[0]["content"]
        await agent.close()

    asyncio.run(run())


def test_official_evidence_symbol_survives_resets_and_tool_roundtrip(monkeypatch):
    from fanoutqa.models import Evidence
    from platoon.episode.context import current_trajectory, current_trajectory_collection
    from platoon.episode.trajectory import TrajectoryCollection

    evidence = Evidence(
        pageid=0,
        revid=0,
        title="Returned official page",
        url="/wikipedia_en_all_nopic_2023-09/A/Returned_page",
    )
    received_evidence = []

    def wiki_search(query, results=10):
        assert query == "synthetic lookup"
        assert results == 1
        return [evidence]

    def wiki_content(argument):
        received_evidence.append(argument)
        return "Official page content."

    monkeypatch.setattr(fanoutqa, "wiki_search", wiki_search)
    monkeypatch.setattr(fanoutqa, "wiki_content", wiki_content)

    async def run():
        task = Task(
            id="evidence-namespace",
            goal="Inspect one page.",
            max_steps=30,
            misc={"source_id": "synthetic-id", "dataset_split": "dev", "categories": []},
        )
        single = FanOutQASingleCodeExecutor(task)
        recursive = FanOutQARecursiveCodeExecutor(task, subagent_max_steps=15)
        child_task = task.fork(
            "Inspect a delegated page.",
            max_steps=15,
            task_misc={"source_id": "synthetic-id", "dataset_split": "dev", "_is_subagent": True},
        )
        child = await recursive.fork(child_task)
        grandchild_task = child_task.fork(
            "Inspect a nested page.",
            max_steps=5,
            task_misc={"source_id": "synthetic-id", "dataset_split": "dev", "_is_subagent": True},
        )
        grandchild = await child.fork(grandchild_task)
        executors = (single, recursive, child, grandchild)

        for executor in executors:
            assert executor.shell.user_ns["Evidence"] is Evidence
            await executor.reset()
            assert executor.shell.user_ns["Evidence"] is Evidence

        collection = TrajectoryCollection()
        collection_token = current_trajectory_collection.set(collection)
        trajectory = collection.create_trajectory()
        collection.set_trajectory_task(trajectory.id, task)
        trajectory_token = current_trajectory.set(trajectory)
        try:
            for executor in (single, recursive):
                code = (
                    "results = wiki_search('synthetic lookup', results=1)\n"
                    "page = wiki_content(results[0])\n"
                    f"evidence_copy = {evidence!r}\n"
                    "copy_page = wiki_content(evidence_copy)"
                )
                step = await executor.run(code)
                assert not step.error
                assert executor.shell.user_ns["Evidence"] is Evidence
                assert executor.shell.user_ns["results"][0] is evidence
                assert executor.shell.user_ns["page"] == "Official page content."
                evidence_copy = executor.shell.user_ns["evidence_copy"]
                assert type(evidence_copy) is Evidence
                assert evidence_copy == evidence
                assert executor.shell.user_ns["copy_page"] == "Official page content."
            assert len(received_evidence) == 4
            assert received_evidence[0] is evidence
            assert received_evidence[1] is single.shell.user_ns["evidence_copy"]
            assert received_evidence[2] is evidence
            assert received_evidence[3] is recursive.shell.user_ns["evidence_copy"]
        finally:
            current_trajectory.reset(trajectory_token)
            current_trajectory_collection.reset(collection_token)

    asyncio.run(run())
