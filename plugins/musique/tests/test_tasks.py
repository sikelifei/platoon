import json

from platoon.musique.evaluation import extract_prediction, score_answer
from platoon.musique.tasks import (
    DEFAULT_HF_ENDPOINT,
    format_context,
    get_task,
    get_task_ids,
)


def test_huggingface_fallback_defaults_to_hf_mirror():
    assert DEFAULT_HF_ENDPOINT == "https://hf-mirror.com"


def test_local_task_loading_and_context_does_not_leak_support_labels(tmp_path):
    example = {
        "id": "toy-0",
        "question": "Who wrote the novel?",
        "answer": "Alice",
        "paragraphs": [
            {"title": "Book", "paragraph_text": "The novel was written by Alice.", "is_supporting": True},
            {"title": "Noise", "paragraph_text": "This is unrelated.", "is_supporting": False},
        ],
    }
    path = tmp_path / "musique_ans_v1.0_dev.jsonl"
    path.write_text(json.dumps(example) + "\n", encoding="utf-8")

    task_ids = get_task_ids(data_dir=tmp_path, num_tasks=1)
    task = get_task(task_ids[0], data_dir=tmp_path)

    assert task.goal == example["question"]
    assert "The novel was written by Alice." in task.misc["context"]
    assert "is_supporting" not in task.misc["context"]
    assert "[Passage 1]" in format_context(example)


def test_answer_scoring_accepts_plain_and_labeled_outputs():
    example = {"answer": "Alice"}
    assert score_answer("Alice", example)["answer_em"] == 1.0
    assert score_answer("Final answer: Alice", example)["answer_f1"] == 1.0
    assert extract_prediction('{"answer": "Alice"}') == "Alice"
