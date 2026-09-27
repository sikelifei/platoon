"""Homogeneous CodeAct agents and prompts for FanOutQA."""

from __future__ import annotations

import re
from pathlib import Path

from platoon.agents.codeact import CodeActAgent
from platoon.agents.codeact.prompt_builder import CodeActPromptBuilder, PromptMode
from platoon.config_defs import InferenceParams
from platoon.envs.base import Task
from platoon.utils.llm_client import LLMClient

_RECURSIVE_PROMPT = Path(__file__).parent / "prompts" / "fanoutqa_recursive.md"


def _remove_delegation_prompt(prompt: str) -> str:
    """Keep the recursive research prompt intact except for delegation instructions."""
    prompt = re.sub(
        r"(?ms)^DELEGATION STRATEGY:[ \t]*\n.*?(?=^[A-Z][A-Z _-]*:[ \t]*$|\Z)",
        "",
        prompt,
    )
    prompt = re.sub(
        r",\s*and you can delegate subproblems to subagents\b",
        "",
        prompt,
        flags=re.IGNORECASE,
    )
    prompt = re.sub(
        r"\bresearch (?:or|and) delegation strategy\b",
        "research strategy",
        prompt,
        flags=re.IGNORECASE,
    )
    lines = [
        line
        for line in prompt.splitlines()
        if not re.search(
            r"\b(?:launch_subagent|subagents?|delegat\w*)\b",
            line,
            flags=re.IGNORECASE,
        )
    ]
    prompt = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return f"{prompt}\n" if prompt else ""


def _read_prompt(prompt_path: str | Path | None, recursive: bool) -> str:
    if prompt_path is None:
        path = _RECURSIVE_PROMPT
    else:
        path = Path(prompt_path).expanduser()
        if not path.is_absolute() and not path.is_file():
            path = Path(__file__).parent / path
        if not path.is_file():
            raise FileNotFoundError(f"FanOutQA prompt file not found: {prompt_path}")
    prompt = path.read_text(encoding="utf-8")
    return prompt if recursive else _remove_delegation_prompt(prompt)


class FanOutQAPromptBuilder(CodeActPromptBuilder):
    def __init__(
        self,
        recursive: bool = True,
        prompt_path: str | Path | None = None,
        prompt_mode: PromptMode = "sequence_extension",
        include_reasoning: bool = True,
    ):
        super().__init__(prompt_mode=prompt_mode, include_reasoning=include_reasoning)
        self.recursive = recursive
        self.prompt_path = str(prompt_path) if prompt_path is not None else None
        self.system_prompt = _read_prompt(prompt_path, recursive)

    def build_system_prompt(self, obs, **context) -> str:
        return self.system_prompt


class FanOutQASingleAgent(CodeActAgent):
    """Single-agent baseline without recursion awareness."""

    def __init__(
        self,
        prompt_builder: CodeActPromptBuilder | None = None,
        prompt_mode: PromptMode = "sequence_extension",
        include_reasoning: bool = True,
        llm_client: LLMClient | None = None,
        inference_params: InferenceParams | None = None,
        stuck_in_loop_threshold: int = 4,
        stuck_in_loop_window: int = 3,
        prompt_path: str | Path | None = None,
    ):
        if prompt_builder is None:
            prompt_builder = FanOutQAPromptBuilder(False, prompt_path, prompt_mode, include_reasoning)
        super().__init__(
            prompt_builder=prompt_builder,
            prompt_mode=prompt_mode,
            include_reasoning=include_reasoning,
            llm_client=llm_client,
            inference_params=inference_params,
            stuck_in_loop_threshold=stuck_in_loop_threshold,
            stuck_in_loop_window=stuck_in_loop_window,
        )

    async def fork(self, task: Task) -> FanOutQASingleAgent:
        return await super().fork(task)


class FanOutQARecursiveAgent(CodeActAgent):
    """Every descendant uses this same recursive policy and prompt."""

    def __init__(
        self,
        prompt_builder: CodeActPromptBuilder | None = None,
        prompt_mode: PromptMode = "sequence_extension",
        include_reasoning: bool = True,
        llm_client: LLMClient | None = None,
        inference_params: InferenceParams | None = None,
        stuck_in_loop_threshold: int = 4,
        stuck_in_loop_window: int = 3,
        prompt_path: str | Path | None = None,
    ):
        if prompt_builder is None:
            prompt_builder = FanOutQAPromptBuilder(True, prompt_path, prompt_mode, include_reasoning)
        super().__init__(
            prompt_builder=prompt_builder,
            prompt_mode=prompt_mode,
            include_reasoning=include_reasoning,
            llm_client=llm_client,
            inference_params=inference_params,
            stuck_in_loop_threshold=stuck_in_loop_threshold,
            stuck_in_loop_window=stuck_in_loop_window,
        )

    async def fork(self, task: Task) -> FanOutQARecursiveAgent:
        return await super().fork(task)
