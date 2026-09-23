#!/usr/bin/env python3
"""Build a deterministic, hop-stratified MuSiQue difficulty slice."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
DEFAULT_QUOTAS = {2: 5, 3: 5, 4: 10}


def tokens(value: str) -> set[str]:
    return {item.lower() for item in TOKEN_RE.findall(value) if len(item) > 2}


def paragraph_text(paragraph: dict[str, Any]) -> str:
    return str(paragraph.get("paragraph_text") or paragraph.get("text") or "")


def features(index: int, example: dict[str, Any]) -> dict[str, Any]:
    question_tokens = tokens(str(example.get("question", "")))
    paragraphs = example.get("paragraphs") or []
    support_overlaps: list[int] = []
    distractor_overlaps: list[int] = []
    context_chars = 0
    for paragraph in paragraphs:
        text = paragraph_text(paragraph)
        context_chars += len(text)
        overlap = len(question_tokens & tokens(f"{paragraph.get('title', '')} {text}"))
        if paragraph.get("is_supporting"):
            support_overlaps.append(overlap)
        else:
            distractor_overlaps.append(overlap)
    hops = len(example.get("question_decomposition") or [])
    return {
        "original_index": index,
        "source_id": example.get("id"),
        "hops": hops,
        "context_chars": context_chars,
        "distractor_overlap": max(distractor_overlaps, default=0),
        "support_overlap": sum(support_overlaps),
        "question": example.get("question"),
        "answer": example.get("answer"),
    }


def spread_pick(items: list[dict[str, Any]], quota: int) -> list[dict[str, Any]]:
    ranked = sorted(
        items,
        key=lambda item: (
            item["difficulty_score"],
            item["context_chars"],
            item["original_index"],
        ),
    )
    if quota >= len(ranked):
        return ranked
    positions = [round(i * (len(ranked) - 1) / (quota - 1)) for i in range(quota)]
    return [ranked[position] for position in positions]


def build_subset(
    rows: list[dict[str, Any]],
    quotas: dict[int, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    feature_rows = [features(index, row) for index, row in enumerate(rows)]
    selected: list[dict[str, Any]] = []
    for hops, quota in quotas.items():
        group = [item for item in feature_rows if item["hops"] == hops]
        raw_scores = [
            math.log1p(item["context_chars"])
            + 0.35 * item["distractor_overlap"]
            - 0.20 * item["support_overlap"]
            for item in group
        ]
        low, high = min(raw_scores), max(raw_scores)
        for item, raw_score in zip(group, raw_scores):
            item["difficulty_score"] = (
                0.0 if high == low else round((raw_score - low) / (high - low), 6)
            )
        chosen = spread_pick(group, quota)
        for rank, item in enumerate(chosen):
            item["difficulty_band"] = (
                "easy" if rank < quota / 3 else "medium" if rank < 2 * quota / 3 else "hard"
            )
        selected.extend(chosen)

    selected.sort(key=lambda item: (item["hops"], item["difficulty_score"]))
    subset_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for subset_index, item in enumerate(selected):
        subset_rows.append(rows[item["original_index"]])
        manifest.append({"subset_index": subset_index, **item})
    return subset_rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quota-2", type=int, default=DEFAULT_QUOTAS[2])
    parser.add_argument("--quota-3", type=int, default=DEFAULT_QUOTAS[3])
    parser.add_argument("--quota-4", type=int, default=DEFAULT_QUOTAS[4])
    args = parser.parse_args()
    quotas = {2: args.quota_2, 3: args.quota_3, 4: args.quota_4}

    rows = [
        json.loads(line)
        for line in args.input.open(encoding="utf-8")
        if line.strip()
    ]
    subset_rows, manifest = build_subset(rows, quotas)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = args.output_dir / "musique_ans_v1.0_dev.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in subset_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest_path = args.output_dir / "selection_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source": str(args.input),
                "selection_method": "hop quotas plus evenly spread composite difficulty",
                "quotas": quotas,
                "examples": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(subset_rows)} examples to {dataset_path}")
    for hops in sorted(quotas):
        chosen = [item for item in manifest if item["hops"] == hops]
        bands = {band: sum(item["difficulty_band"] == band for item in chosen) for band in ("easy", "medium", "hard")}
        print(f"{hops}-hop: {len(chosen)} {bands}")


if __name__ == "__main__":
    main()
