"""MuSiQue data loading and conversion into Platoon tasks."""

from __future__ import annotations

import argparse
import json
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from platoon.envs.base import Task

DatasetVariant = Literal["ans", "full"]
DatasetSplit = Literal["train", "dev", "validation", "test"]

DEFAULT_DATA_DIR = Path(
    os.environ.get("MUSIQUE_DATA_DIR", "/data2/zhangwenjian/agent/musique_data")
)
DEFAULT_HF_DATASET = "voidful/MuSiQue"
DEFAULT_HF_ENDPOINT = os.environ.get(
    "HF_ENDPOINT", "https://hf-mirror.com"
).rstrip("/")

_VARIANT_FILES = {
    "ans": "musique_ans_v1.0_{split}.jsonl",
    "full": "musique_full_v1.0_{split}.jsonl",
}
_DATA_CACHE: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}


def canonical_split(split: DatasetSplit | str) -> str:
    """Normalize the official dev/validation spelling used by configs."""
    if split == "validation":
        return "dev"
    if split not in {"train", "dev", "test"}:
        raise ValueError("split must be one of: train, dev, validation, test")
    return split


def dataset_filename(dataset_variant: DatasetVariant | str, split: DatasetSplit | str) -> str:
    if dataset_variant not in _VARIANT_FILES:
        raise ValueError("dataset_variant must be 'ans' or 'full'")
    return _VARIANT_FILES[dataset_variant].format(split=canonical_split(split))


def _find_local_file(data_dir: Path, filename: str) -> Path | None:
    candidates = (
        data_dir / filename,
        data_dir / "data" / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object on {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"MuSiQue file is empty: {path}")
    return records


def _load_from_huggingface(
    dataset_variant: DatasetVariant,
    split: str,
    hf_dataset_name: str,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The datasets package is required when the MuSiQue JSONL file is not local. "
            "Run 'uv sync' in plugins/musique."
        ) from exc

    filename = dataset_filename(dataset_variant, split)
    url = (
        f"{DEFAULT_HF_ENDPOINT}/datasets/{hf_dataset_name}/resolve/main/"
        f"{filename}?download=true"
    )
    dataset = load_dataset("json", data_files={split: url}, split=split)
    return [dict(example) for example in dataset]


def load_records(
    dataset_variant: DatasetVariant = "ans",
    split: DatasetSplit = "dev",
    data_dir: str | Path | None = None,
    hf_dataset_name: str = DEFAULT_HF_DATASET,
) -> list[dict[str, Any]]:
    """Load one official MuSiQue split from disk, or from Hugging Face if absent."""
    normalized_split = canonical_split(split)
    root = Path(data_dir or DEFAULT_DATA_DIR).expanduser()
    cache_key = (str(root.resolve()), dataset_variant, normalized_split, hf_dataset_name)
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

    filename = dataset_filename(dataset_variant, normalized_split)
    local_file = _find_local_file(root, filename)
    if local_file is not None:
        records = _read_jsonl(local_file)
    else:
        records = _load_from_huggingface(dataset_variant, normalized_split, hf_dataset_name)

    _DATA_CACHE[cache_key] = records
    return records


def _paragraph_text(paragraph: dict[str, Any]) -> str:
    for key in ("paragraph_text", "text", "paragraph", "contents"):
        value = paragraph.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def format_context(example: dict[str, Any]) -> str:
    """Format passages without exposing gold supporting-paragraph annotations."""
    paragraphs = example.get("paragraphs")
    if isinstance(paragraphs, list):
        blocks: list[str] = []
        for index, paragraph in enumerate(paragraphs, start=1):
            if not isinstance(paragraph, dict):
                continue
            title = paragraph.get("title") or f"Passage {index}"
            text = _paragraph_text(paragraph)
            if text:
                blocks.append(f"[Passage {index}] {title}\n{text}")
        if blocks:
            return "\n\n".join(blocks)

    context = example.get("context")
    if isinstance(context, str) and context.strip():
        return context.strip()
    return ""


def _example_to_task(
    example: dict[str, Any],
    dataset_variant: DatasetVariant,
    split: str,
    index: int,
    prompt_path: str | None = None,
) -> Task:
    question = example.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"MuSiQue example {index} has no question")

    task_id = f"musique.{dataset_variant}.{split}.{index}"
    task_misc = deepcopy(example)
    task_misc.update(
        {
            "context": format_context(example),
            "source_id": example.get("id"),
            "dataset_variant": dataset_variant,
            "dataset_split": split,
        }
    )
    if prompt_path:
        task_misc["prompt_path"] = prompt_path

    return Task(
        goal=question.strip(),
        id=task_id,
        max_steps=50,
        misc=task_misc,
        fork_strategy="task",
    )


def get_task_ids(
    dataset_variant: DatasetVariant = "ans",
    split: DatasetSplit = "dev",
    num_tasks: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
    data_dir: str | Path | None = None,
    hf_dataset_name: str = DEFAULT_HF_DATASET,
) -> list[str]:
    normalized_split = canonical_split(split)
    records = load_records(dataset_variant, normalized_split, data_dir, hf_dataset_name)
    task_ids = [f"musique.{dataset_variant}.{normalized_split}.{index}" for index in range(len(records))]
    if shuffle:
        random.Random(seed).shuffle(task_ids)
    if num_tasks is not None and num_tasks > 0:
        task_ids = task_ids[:num_tasks]
    return task_ids


def load_task(
    task_id: str,
    data_dir: str | Path | None = None,
    hf_dataset_name: str = DEFAULT_HF_DATASET,
    prompt_path: str | None = None,
) -> Task:
    parts = task_id.split(".")
    if len(parts) != 4 or parts[0] != "musique":
        raise ValueError(
            f"Invalid MuSiQue task id {task_id!r}; expected musique.<ans|full>.<split>.<index>"
        )
    dataset_variant = parts[1]
    split = canonical_split(parts[2])
    try:
        index = int(parts[3])
    except ValueError as exc:
        raise ValueError(f"Invalid MuSiQue task index in {task_id!r}") from exc

    records = load_records(dataset_variant, split, data_dir, hf_dataset_name)
    if index < 0 or index >= len(records):
        raise IndexError(f"Task index {index} out of range for {dataset_variant}/{split}")
    return _example_to_task(records[index], dataset_variant, split, index, prompt_path)


_TASK_CACHE: dict[tuple[str, str, str, str, str | None], Task] = {}


def get_task(
    task_id: str,
    data_dir: str | Path | None = None,
    hf_dataset_name: str = DEFAULT_HF_DATASET,
    prompt_path: str | None = None,
) -> Task:
    key = (
        task_id,
        str(data_dir or DEFAULT_DATA_DIR),
        hf_dataset_name,
        str(prompt_path) if prompt_path else "",
        "v1",
    )
    if key not in _TASK_CACHE:
        _TASK_CACHE[key] = load_task(task_id, data_dir, hf_dataset_name, prompt_path)
    return deepcopy(_TASK_CACHE[key])


def save_split(
    dataset_variant: DatasetVariant,
    split: DatasetSplit,
    output_dir: str | Path,
    data_dir: str | Path | None = None,
    hf_dataset_name: str = DEFAULT_HF_DATASET,
) -> Path:
    """Materialize a split from the configured source as local official JSONL."""
    output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / dataset_filename(dataset_variant, split)
    records = load_records(dataset_variant, split, data_dir, hf_dataset_name)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or materialize MuSiQue data.")
    parser.add_argument("--variant", choices=["ans", "full"], default="ans")
    parser.add_argument("--split", choices=["train", "dev", "validation", "test"], default="dev")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--hf-dataset-name", default=DEFAULT_HF_DATASET)
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()

    records = load_records(args.variant, args.split, args.data_dir, args.hf_dataset_name)
    print(f"Loaded {len(records)} MuSiQue {args.variant}/{canonical_split(args.split)} examples.")
    if args.materialize:
        output_path = save_split(
            args.variant, args.split, args.data_dir, args.data_dir, args.hf_dataset_name
        )
        print(f"Materialized: {output_path}")


if __name__ == "__main__":
    main()
