# MuSiQue plugin

This plugin adds MuSiQue multi-hop question answering to Platoon's standard
inference runner. It supports the official JSONL files and a Hugging Face
fallback, while keeping the answer prompt in an editable file.

## Setup

From this directory:

    uv sync

The default config uses answerable development examples from:

    /data2/zhangwenjian/agent/musique_data

If the configured file is absent, the loader fetches the matching JSONL from
`https://hf-mirror.com`. Set `HF_ENDPOINT` to override that mirror. The
server currently has all official answerable/full train, dev, and test files
materialized in the directory above.

## Inference

Start an OpenAI-compatible model server, then run:

    uv run python -m platoon.musique.inference_scripts.run_inference \
      --config platoon/musique/configs/inference/musique_inference.yaml

Useful overrides:

    --dataset_split test
    --num_tasks 10
    --prompt_path /absolute/path/to/your_prompt.md
    --inference.output_dir /absolute/path/to/results

The editable prompt is:

    platoon/musique/prompts/musique.md

The recursive runner exposes only search, finish, and (in recursive mode) the native launch_subagent action. Search is deterministic and task-local; it returns plain-text passage evidence without model calls. Root-only evaluation keeps benchmark gold data out of child task metadata.
