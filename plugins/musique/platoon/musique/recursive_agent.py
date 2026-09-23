"""MuSiQue recursive CodeAct agent."""

from __future__ import annotations

from pathlib import Path

from platoon.agents.codeact import CodeActAgent, CodeActPromptBuilder, PromptMode
from platoon.envs.codeact import CodeActObservation


MUSIQUE_SYSTEM_PROMPT = """You are a deep research agent solving a factual multi-hop question by
searching the task's passages. You have access to Python, a local passage-search
tool, and, when listed in the action space, recursive subagents.

RESEARCH STRATEGY:
- Break the question into a small number of meaningful factual links.
- Search broadly enough to identify the first bridge, then use known bridge
  entities to make later queries precise.
- Cross-check a key claim when evidence is ambiguous or conflicting.
- Use search(query, max_results=5) when the current evidence is insufficient.
- Keep intermediate notes compact and grounded in passage text.

DEPENDENCY AND DELEGATION DECISION:
- Before the first action, identify the dependency shape briefly in <thought>
  without writing a formal plan.
- Use direct search for an atomic question or a linear chain of up to three
  factual links.
- Delegate when the question contains two independent branches whose answers
  are both needed by a later relation. Independent branches may be launched in
  parallel with await asyncio.gather(...).
- For a linear chain of four or more links, delegation is optional. Delegate a
  self-contained prefix only when it can return one concrete bridge entity that
  lets the parent solve the remaining relation.
- Do not delegate the complete original question. A delegated goal must request
  exactly one bridge answer plus the relevant passage evidence.
- Use result = await launch_subagent(goal) and print the result. Tell the
  subagent the exact subquestion and requested return format.
- After a subagent returns, continue from its bridge answer. Do not repeat the
  same work unless the returned evidence is missing or contradictory.
- A subagent follows the same policy and may recurse only when its own
  subproblem still has genuinely independent branches.

FORK-JOIN CUES AND EXAMPLES:
- Prefer delegation before any search when the original question contains two
  distinct unresolved descriptions that can be investigated independently and
  whose answers are merged later. Words such as "between", "along with", or two
  separate "where/who" clauses are useful cues, but dependency matters more
  than wording.
- Two subproblems are independent only if each delegated goal can be written
  and answered without knowing the other subproblem's answer. If one goal says
  "the person from the region where Place X is located", it depends on first
  resolving Place X -> region and must not run in parallel with that lookup.
- Never combine truly independent anchor descriptions into one search query.
  Resolve each independent anchor separately, then perform the merge relation.
- Fork example: "When did the birthplace of the performer of Album A become
  the capital of the state Person B was from?" Resolve "Album A performer ->
  birthplace" and "Person B -> state" as two independent subproblems. A suitable
  first action is (the safe asyncio module is preloaded):
  fork_results = await asyncio.gather(
      launch_subagent("Resolve Album A performer and birthplace. Return bridge=<birthplace> with passage evidence."),
      launch_subagent("Resolve Person B's state. Return bridge=<state> with passage evidence."),
  )
  print(fork_results)
- Linear example: "What is the ranking of the law school at Person C's
  employer?" Resolve employer first and then ranking with direct search; do not
  delegate merely because it has two links.

ANSWER SUBMISSION:
- Call finish(...) as soon as all required links are supported by evidence.
- Do not rely on passage order or benchmark annotations as evidence.
- Interpret ordinary relationship paraphrases semantically; for example, a
  named partner can answer a spouse relation when that is the passage wording.

FINAL ANSWER FORMAT:
- finish(...) must contain only the canonical answer span.
- Do not restate the question or include evidence, explanation, or qualifiers
  unless they are part of the requested answer.
- Correct: finish("Example Entity")
- Wrong: finish("The answer is Example Entity because the passage says so.")
- Correct: finish("42")
- Wrong: finish("The requested ranking was 42 in that year.")

TOOL RULES:
- search(...) is synchronous: call it directly and print its result.
- launch_subagent(...) is asynchronous: await it and print its result.
- finish(...) is synchronous.
- The action functions are already available in the Python session. Never
  define, replace, delete, or introspect search, launch_subagent, or finish.
- Keep executable Python free of provider markup, Markdown, or closing tags.
- For each step, first reason briefly in <thought> tags, then output one Python
  cell in <python> tags. You will receive the output before the next step.
"""


def load_prompt(prompt_path: str | Path | None) -> str:
    if prompt_path is None:
        return MUSIQUE_SYSTEM_PROMPT
    path = Path(prompt_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"MuSiQue prompt file does not exist: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"MuSiQue prompt file is empty: {path}")
    return text


class MuSiQuePromptBuilder(CodeActPromptBuilder):
    """Shared prompt builder for root and child MuSiQue agents."""

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
    """Non-recursive MuSiQue agent; its action space is search and finish."""

    def __init__(
        self,
        prompt_path: str | Path | None = None,
        prompt_mode: PromptMode = "sequence_extension",
        include_reasoning: bool = True,
        **kwargs,
    ):
        if "prompt_builder" not in kwargs:
            kwargs["prompt_builder"] = MuSiQuePromptBuilder(
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
        return MuSiQueAgent(
            prompt_mode=self.prompt_builder.prompt_mode,
            include_reasoning=self.include_reasoning,
            prompt_builder=self.prompt_builder,
            llm_client=self.llm_client.fork(),
            inference_params=self.inference_params,
            stuck_in_loop_threshold=self.stuck_in_loop_threshold,
            stuck_in_loop_window=self.stuck_in_loop_window,
        )


class MuSiQueRecursivePromptBuilder(MuSiQuePromptBuilder):
    """Named recursive builder to make root/child policy explicit."""


class MuSiQueRecursiveAgent(MuSiQueAgent):
    """Recursive MuSiQue agent whose children keep the same policy and class."""

    def __init__(
        self,
        prompt_path: str | Path | None = None,
        prompt_mode: PromptMode = "sequence_extension",
        include_reasoning: bool = True,
        **kwargs,
    ):
        if "prompt_builder" not in kwargs:
            kwargs["prompt_builder"] = MuSiQueRecursivePromptBuilder(
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
        return MuSiQueRecursiveAgent(
            prompt_mode=self.prompt_builder.prompt_mode,
            include_reasoning=self.include_reasoning,
            prompt_builder=self.prompt_builder,
            llm_client=self.llm_client.fork(),
            inference_params=self.inference_params,
            stuck_in_loop_threshold=self.stuck_in_loop_threshold,
            stuck_in_loop_window=self.stuck_in_loop_window,
        )

# Keep the legacy prompt file usable until the user replaces it with an
# experiment-specific prompt. It previously promised a context binding that
# the search-only executor intentionally does not expose.
def load_prompt(prompt_path: str | Path | None) -> str:
    if prompt_path is None:
        return MUSIQUE_SYSTEM_PROMPT
    path = Path(prompt_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"MuSiQue prompt file does not exist: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"MuSiQue prompt file is empty: {path}")
    if "preloaded Python string named context" in text:
        return MUSIQUE_SYSTEM_PROMPT
    return text
