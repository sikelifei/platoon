"""Standalone non-recursive CodeAct agent for MuSiQue."""

from __future__ import annotations

from pathlib import Path

from platoon.agents.codeact import CodeActAgent, CodeActPromptBuilder, PromptMode
from platoon.envs.codeact import CodeActObservation


MUSIQUE_SINGLE_SYSTEM_PROMPT = """You are a deep research agent solving a factual question by searching the
task's passages. You have access to Python plus a local passage-search tool.
There are no subagents in this environment.

RESEARCH STRATEGY:
- Start broad, then refine the query based on what you learn.
- Break multi-hop questions into a small number of meaningful factual links.
- Cross-check key claims across multiple passages when possible.
- Use search(query, max_results=5) when the current evidence is insufficient.
- Avoid dumping unnecessary passage text into the notebook.
- Keep intermediate notes concise and use Python to organize findings.
- Use only the supplied passages. Do not rely on benchmark annotations or
  passage order as evidence.

ANSWER SUBMISSION:
- When every required link is supported and you are confident, call finish(...).
- Submit only the shortest answer that directly answers the question unless the
  task explicitly asks for more detail.
- Do not repeat equivalent searches after the answer is established.

TOOL RULES:
- search(...) is synchronous: call it directly and print its result.
- finish(...) is synchronous.
- The action functions are already available in the Python session. Never
  define, replace, delete, or introspect search or finish.
- Keep executable Python free of provider markup, Markdown, or closing tags.
- For each step, first reason briefly in <thought> tags, then output one Python
  cell in <python> tags. You will receive the output before the next step.
"""


def load_single_prompt(prompt_path: str | Path | None) -> str:
    if prompt_path is None:
        return MUSIQUE_SINGLE_SYSTEM_PROMPT
    path = Path(prompt_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"MuSiQue single-agent prompt does not exist: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"MuSiQue single-agent prompt is empty: {path}")
    return text


class MuSiQueSinglePromptBuilder(CodeActPromptBuilder):
    """Prompt builder for the search-only MuSiQue agent."""

    def __init__(
        self,
        prompt_path: str | Path | None = None,
        prompt_mode: PromptMode = "sequence_extension",
        include_reasoning: bool = True,
    ):
        super().__init__(prompt_mode=prompt_mode, include_reasoning=include_reasoning)
        self.prompt = load_single_prompt(prompt_path)

    def build_system_prompt(self, obs: CodeActObservation, **context) -> str:
        del obs, context
        return self.prompt


class MuSiQueSingleAgent(CodeActAgent):
    """Non-recursive MuSiQue agent with no subagent action."""

    def __init__(
        self,
        prompt_path: str | Path | None = None,
        prompt_mode: PromptMode = "sequence_extension",
        include_reasoning: bool = True,
        **kwargs,
    ):
        if "prompt_builder" not in kwargs:
            kwargs["prompt_builder"] = MuSiQueSinglePromptBuilder(
                prompt_path=prompt_path,
                prompt_mode=prompt_mode,
                include_reasoning=include_reasoning,
            )
        super().__init__(
            prompt_mode=prompt_mode,
            include_reasoning=include_reasoning,
            **kwargs,
        )

    async def fork(self, task):
        return MuSiQueSingleAgent(
            prompt_mode=self.prompt_builder.prompt_mode,
            include_reasoning=self.include_reasoning,
            prompt_builder=self.prompt_builder,
            llm_client=self.llm_client.fork(),
            inference_params=self.inference_params,
            stuck_in_loop_threshold=self.stuck_in_loop_threshold,
            stuck_in_loop_window=self.stuck_in_loop_window,
        )
