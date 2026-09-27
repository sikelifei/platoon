# FanOutQA plugin for Platoon

This plugin wraps the official FanOutQA Open-Book benchmark in native Platoon
tasks and CodeAct episodes. It loads questions with `fanoutqa.load_dev()` or
`fanoutqa.load_test()`, calls the official `fanoutqa.wiki_search(query,
results=10)` and `fanoutqa.wiki_content(evidence)` functions, and uses
FanOutQA's official answer evaluator. It does not copy benchmark data or build
a second retrieval index.

The single-agent and recursive Qwen configs use the same model, prompt source,
Wikipedia backend, temperature, completion limit, and root step budget. The
single-agent executor exposes `wiki_search`, `wiki_content`, and `finish`.
The recursive executor adds only Platoon's native `launch_subagent` action.
Every recursive child uses the same policy class and prompt and may launch its
own children.

Root answers receive FanOutQA's official loose answer reward. Child episode
rewards stay at zero, and test-split rewards stay at zero. Gold answers are
looked up only by the benchmark-side evaluator and are not added to tool
results, task metadata, or agent history. Any future recursive credit
assignment belongs outside the environment reward.

## Setup

The plugin pins the official FanOutQA GitHub source commit
`989f4c40d9deea1ecb0897d7a17a9c0fe20d5c33`, whose package metadata is
`1.3.0`. From this directory, install its locked environment with:

```bash
uv sync
uv run python -m spacy download en_core_web_sm
```

The spaCy English model is required by the official scalar reward. The first
tokenizer use may download tiktoken's `o200k_base` vocabulary; set
`TIKTOKEN_CACHE_DIR` to a writable cache directory if you want to control
where it is stored.

The inference configs target the OpenAI-compatible Qwen service at
`http://192.168.1.134:8010/v1`. Override it without editing the config:

```bash
export FANOUTQA_MODEL_NAME=openai/Qwen3-4B-Instruct-2507
export FANOUTQA_MODEL_ENDPOINT=http://192.168.1.134:8010/v1
export FANOUTQA_MODEL_API_KEY=EMPTY
```

The default configs select FanOutQA's recommended September 2023 Kiwix
snapshot:

```bash
export FANOUTQA_WIKIPEDIA_TYPE=kiwix
export FANOUTQA_KIWIX_BASE=http://127.0.0.1:8889
export FANOUTQA_KIWIX_ZIMNAME=wikipedia_en_all_nopic_2023-09
```

Run the upstream Kiwix server and the local compatibility proxy in separate
terminals. Point Kiwix at the September 2023 snapshot:

```bash
kiwix-serve --port=8888 --address=127.0.0.1 /path/to/wikipedia_en_all_nopic_2023-09.zim
```

Then start the plugin's standard-library proxy:

```bash
uv run python -m platoon.fanoutqa.inference_scripts.kiwix_compat
```

Kiwix 3.2 returns HTML from `/search`, while the official FanOutQA 1.3
parser expects Atom. This loopback proxy changes only that wire format; the
official package, query parameters, result order, and ZIM snapshot remain
untouched. It forwards the raw search query, converts only direct article-title
links inside the results list to Atom entries, and preserves their order and
hrefs. All other paths, including article content, pass through with their
upstream status, content type, and body.

FanOutQA writes page content beneath `~/.cache/fanoutqa/kiwix` but does not
create that directory. Initialize it once before the first content request:

```bash
mkdir -p ~/.cache/fanoutqa/kiwix
```

The proxy and its response handling are covered by local HTTP tests. A manual
smoke test against the configured snapshot confirmed three ordered `Paris`
search results, successful official `wiki_content` retrieval, and an empty
result for a nonexistent query. The matched 20-question Single and Recursive runs have completed against the
same official dev IDs and September 2023 ZIM snapshot. The outputs and full
official scores are under results/fanoutqa/qwen_single/ and
results/fanoutqa/qwen_recursive/. Both runs used prompt source SHA-256
45284a5b3b466a65679dc2cdf7b0564a3f1843765ec154634876652cce806a3d.
That prompt version adds a concrete Evidence handoff example and requires
claims to be grounded in wiki_content; treat these scores as an integration
reference run, not a prompt-optimized result.

| Setting | Official loose accuracy | Strict accuracy | ROUGE-L F1 | BLEURT-20 |
| --- | ---: | ---: | ---: | ---: |
| Single | 0.4212 | 0.0500 | 0.2246 | 0.4362 |
| Recursive | 0.4155 | 0.1500 | 0.2069 | 0.3878 |

The official evaluator scored all 20 supplied prediction rows in each setting,
including explicit empty answers, so failures remain in the denominator.
Single produced 19 non-empty answers; one episode exceeded the Qwen service's
65,536-token context limit. Recursive produced 15 non-empty answers; four
episodes exceeded that context limit, and one ended with an exhausted-step
warning before producing an answer. A second exhausted-step warning retained
a partial answer. Recursive averaged 11.2 total steps and 113,676 tokens per
question, versus 7.05 steps and 43,252 tokens for Single. Five Recursive
questions launched subagents; none reached a grandchild. These are baseline
observations only; the current prompt and agent policy are not optimized.

## Run the matched Qwen baselines

From this directory, run each config against the same 20 dev examples:

```bash
uv run python -m platoon.fanoutqa.inference_scripts.run_inference \
  --config platoon/fanoutqa/configs/qwen_single.yaml

uv run python -m platoon.fanoutqa.inference_scripts.run_inference \
  --config platoon/fanoutqa/configs/qwen_recursive.yaml
```

Both configs set a root limit of 30 steps, temperature 0, top-p 1, and a
1,024-token completion ceiling. Recursive runs cap each child at 15 steps and
the tree at depth 4 (root depth is 0). Platoon's native trajectory collection
preserves parent IDs and nested branches. A plugin-local tracker extends the
native depth-aware tracker with descendant accounting and reservation, so the
root limit covers root, children, and grandchildren together. Parallel
delegation is policy-controlled; the wrapper does not create children
automatically.

For a full dev run, set `num_tasks: null` in the config. Use
`dataset_split: test` for hidden-label test inference; test rewards stay at
zero.

## Output files

The inference runner retains its per-rollout artifacts and also writes:

- `generations.jsonl`: one official-format `{"id": "...", "answer": "..."}`
  record per selected question, using the original FanOutQA ID.
- `trajectories/`: the complete native root/child/grandchild trajectory
  collections as JSON.
- `events/`: Platoon JSONL event logs.
- `metrics.json`: root/child/total steps, recursion depth, child count,
  official retrieval call counts, token counts, and parallel-branch analysis.

The parallel fields count observed concurrent child episodes. In
`parallel_branches`, a value of 1 means one parent had at least two child
episodes active at the same time. `max_parallel_branches` is the peak number
of simultaneously active direct children under one parent. Sequential sibling
calls are not counted as parallel.

For multi-rollout runs, the JSONL export contains rollout zero for each
question without choosing an answer based on its reward. The analysis files
retain all rollouts. If a rollout artifact is missing, the export still
contains that question ID with an empty answer and records the missing artifact
as an error, preserving the requested denominator.

## Official evaluation

The official evaluator consumes the generated file directly:

```bash
uv run python -m platoon.fanoutqa.inference_scripts.run_inference evaluate \
  --predictions ./results/fanoutqa/qwen_recursive/generations.jsonl \
  --only-score-answered
```

Use `--only-score-answered` for a selected sample such as the default 20
examples. Without it, the official evaluator scores the full dev set and treats
questions without a prediction as unanswered. Full FanOutQA scoring requires
its BLEURT-20 dependency and model checkpoint in addition to the plugin's
`fanoutqa[eval]` extra. The evaluator will report an import/checkpoint error
if BLEURT is unavailable; this plugin does not replace it with another metric.
The official GPT judge remains disabled unless
`FANOUTQA_OPENAI_API_KEY` is configured in the environment.

## Tests

From this directory:

```bash
uv run pytest
```
