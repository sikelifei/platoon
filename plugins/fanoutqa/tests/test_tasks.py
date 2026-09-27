from platoon.fanoutqa.tasks import QUESTION_CACHE, load_tasks


def test_official_dev_loader_builds_safe_platoon_tasks():
    QUESTION_CACHE.clear()

    tasks = load_tasks(split="dev", max_steps=30, num_tasks=1)

    assert len(tasks) == 1
    task = tasks[0]
    source_id = task.misc["source_id"]
    question = QUESTION_CACHE[source_id]
    assert task.id == f"fanoutqa.dev.{source_id}"
    assert task.goal == question.question
    assert task.max_steps == 30
    assert task.misc == {
        "source_id": source_id,
        "dataset_split": "dev",
        "categories": list(question.categories),
    }
    assert task.fork_strategy == "task"
    assert not {"answer", "decomposition", "necessary_evidence"} & task.misc.keys()
    assert "answer" not in task.misc
    assert "decomposition" not in task.misc
    assert "necessary_evidence" not in task.misc
