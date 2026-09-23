from pathlib import Path

from platoon.envs.base import Task

from platoon.musique.single_agent import (
    MUSIQUE_SINGLE_SYSTEM_PROMPT,
    MuSiQueSingleAgent,
    MuSiQueSinglePromptBuilder,
)
from platoon.musique.single_env import MuSiQueSingleCodeExecutor, MuSiQueSingleEnv


def _task() -> Task:
    return Task(
        goal="Who designed the Analytical Engine?",
        id="single-toy-0",
        max_steps=5,
        misc={
            "answer": "Charles Babbage",
            "paragraphs": [
                {
                    "id": "doc-a",
                    "title": "Analytical Engine",
                    "paragraph_text": "The Analytical Engine was designed by Charles Babbage.",
                    "is_supporting": True,
                }
            ],
        },
    )


def test_single_action_space_has_no_recursive_action():
    executor = MuSiQueSingleCodeExecutor(_task())
    assert [action.__name__ for action in executor.actions] == ["search", "finish"]
    assert not hasattr(executor, "launch_subagent")
    assert isinstance(MuSiQueSingleEnv(_task()).code_executor, MuSiQueSingleCodeExecutor)


def test_single_prompt_contract():
    prompt = MuSiQueSinglePromptBuilder().build_system_prompt(None)
    lower = prompt.lower()
    for term in ("research strategy:", "answer submission:", "<thought>", "search(...) is synchronous"):
        assert term in lower
    for term in ("launch_subagent", "delegation strategy:", "view_webpage_content", "plan:"):
        assert term not in lower


def test_editable_single_prompt_matches_default():
    prompt_path = (
        Path(__file__).parents[1]
        / "platoon"
        / "musique"
        / "prompts"
        / "musique_single.md"
    )
    loaded = MuSiQueSinglePromptBuilder(prompt_path).build_system_prompt(None)
    assert loaded == prompt_path.read_text(encoding="utf-8").strip()
    assert loaded == MUSIQUE_SINGLE_SYSTEM_PROMPT.strip()


def test_single_agent_uses_single_prompt_builder():
    agent = MuSiQueSingleAgent(llm_client=object())
    assert isinstance(agent.prompt_builder, MuSiQueSinglePromptBuilder)
