"""CodeAct agent and editable prompt support for MuSiQue."""

from __future__ import annotations

from pathlib import Path

from platoon.agents.codeact import CodeActAgent, CodeActPromptBuilder, PromptMode
from platoon.envs.codeact import CodeActObservation


DEFAULT_PROMPT = """You are solving a MuSiQue multi-hop question answering task.

The task question is shown in the user message. The REPL has a preloaded string
named context containing all candidate passages. Read the passages carefully,
connect the facts needed for the question, and then submit the shortest correct
answer with finish(answer).

Do not use gold annotations or assume that passage order indicates relevance.
Return only the answer in the finish call; do not include a long explanation.
"""


def load_prompt(prompt_path: str | Path | None) -> str:
    if prompt_path is None:
        return DEFAULT_PROMPT
    path = Path(prompt_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"MuSiQue prompt file does not exist: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"MuSiQue prompt file is empty: {path}")
    return text


class MuSiQuePromptBuilder(CodeActPromptBuilder):
    def __init__(
        self,
        prompt_path: str | Path | None = None,
        prompt_mode: PromptMode = "sequence_extension",
        include_reasoning: bool = True,
    ):
        super().__init__(prompt_mode=prompt_mode, include_reasoning=include_reasoning)
        self.prompt = load_prompt(prompt_path)

    def build_system_prompt(self, obs: CodeActObservation, **context) -> str:
        del obs, context
        return self.prompt


class MuSiQueAgent(CodeActAgent):
    """Platoon CodeAct agent with a prompt loaded from a user-editable file."""

    def __init__(
        self,
        prompt_path: str | Path | None = None,
        prompt_mode: PromptMode = "sequence_extension",
        include_reasoning: bool = True,
        **kwargs,
    ):
        kwargs.setdefault(
            "prompt_builder",
            MuSiQuePromptBuilder(
                prompt_path=prompt_path,
                prompt_mode=prompt_mode,
                include_reasoning=include_reasoning,
            ),
        )
        super().__init__(
            prompt_mode=prompt_mode,
            include_reasoning=include_reasoning,
            **kwargs,
        )

    async def fork(self, task):
        return MuSiQueAgent(
            prompt_path=None,
            prompt_mode=self.prompt_builder.prompt_mode,
            include_reasoning=self.include_reasoning,
            prompt_builder=self.prompt_builder,
            llm_client=self.llm_client.fork(),
            inference_params=self.inference_params,
            stuck_in_loop_threshold=self.stuck_in_loop_threshold,
            stuck_in_loop_window=self.stuck_in_loop_window,
        )

from .recursive_agent import (  # noqa: E402
    MUSIQUE_SYSTEM_PROMPT,
    MuSiQueRecursiveAgent,
    MuSiQueRecursivePromptBuilder,
)
DEFAULT_PROMPT = MUSIQUE_SYSTEM_PROMPT

# Use the search-only policy for the legacy public class as well.
from .recursive_agent import MuSiQueAgent, MuSiQuePromptBuilder
