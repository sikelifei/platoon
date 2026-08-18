from pathlib import Path

from rubric import RubricTree, RubricTreeGenerator

from platoon.utils.prompt_retriever import get_prompt


def generate_rubric_tree(task: str) -> RubricTree:
    RUBRIC_GEN_PROMPT_CONTEXT = get_prompt("rubric-tree-judge-prompt", prompts_dir=Path(__file__).parent / "prompts")
    generator = RubricTreeGenerator()  # TODO: Add an option for explicitly passing in a custom model/llm client.
    tree = generator.generate_rubric_tree(
        task=task,
        rubric_gen_prompt_context=RUBRIC_GEN_PROMPT_CONTEXT,
        rubric_gen_generation_guidelines="Prefer generating simple and concise rubrics over complex and lengthy ones.",
        temperature=1,
        max_tokens=5000,
        scorer_types=["llm"],
    )
    return tree
