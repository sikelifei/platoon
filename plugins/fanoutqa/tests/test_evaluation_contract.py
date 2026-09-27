from __future__ import annotations

from types import SimpleNamespace

import pytest
from fanoutqa.models import DevQuestion
from fanoutqa.models import TestQuestion as FanOutQATestQuestion
from platoon.envs.base import Task
from platoon.envs.codeact import CodeActAction
from platoon.episode.context import (
    current_trajectory,
    current_trajectory_collection,
    error_message,
    finish_message,
)
from platoon.episode.trajectory import TrajectoryCollection

import platoon.fanoutqa.evaluation as evaluation
from platoon.fanoutqa.env import FanOutQARecursiveEnv, FanOutQASingleEnv
from platoon.fanoutqa.tasks import QUESTION_CACHE

GOLD_SECRET = "beta_secret_123"


def _dev_question(source_id: str = "fanoutqa-evaluation-contract") -> DevQuestion:
    return DevQuestion(
        id=source_id,
        question="Return the two requested tokens.",
        decomposition=[],
        answer=["alpha", GOLD_SECRET],
        categories=["contract"],
    )


@pytest.fixture
def empty_question_cache():
    original = QUESTION_CACHE.copy()
    QUESTION_CACHE.clear()
    try:
        yield
    finally:
        QUESTION_CACHE.clear()
        QUESTION_CACHE.update(original)


def test_score_answer_matches_official_nontrivial_partial_list_score():
    reference = ["alpha", GOLD_SECRET]
    candidate = "alpha"

    official = evaluation._official_module("fanoutqa.eval.string").answer_in_text
    expected = official(reference, candidate)
    actual = evaluation.score_answer(reference, candidate)

    assert actual == expected
    assert actual.score == pytest.approx(0.5)
    assert actual.found is False
    assert actual.missing


@pytest.mark.asyncio
async def test_root_reward_matches_official_and_does_not_leak_missing_gold(empty_question_cache):
    question = _dev_question()
    QUESTION_CACHE[question.id] = question
    task = Task(
        id=f"fanoutqa.dev.{question.id}",
        goal=question.question,
        max_steps=5,
        misc={
            "source_id": question.id,
            "dataset_split": "dev",
            "categories": list(question.categories),
        },
        fork_strategy="task",
    )
    env = FanOutQASingleEnv(task)
    collection = TrajectoryCollection()
    collection_token = current_trajectory_collection.set(collection)
    trajectory = collection.create_trajectory()
    trajectory_token = current_trajectory.set(trajectory)
    finish_token = finish_message.set(None)
    error_token = error_message.set(None)
    try:
        await env.reset()
        observation = await env.step(CodeActAction(parsed_code="finish('alpha')"))
        step = observation.history[-1]
        reward_info = step.misc["reward_misc"]
        official_result = evaluation.score_answer(question.answer, "alpha")

        assert step.reward == pytest.approx(official_result.score)
        assert step.reward == pytest.approx(0.5)
        assert reward_info["acc.loose"] == pytest.approx(official_result.score)
        assert reward_info["acc.strict"] == 0.0
        assert "missing" not in reward_info
        assert GOLD_SECRET not in repr(reward_info)
        assert GOLD_SECRET not in repr(observation.history)
        assert GOLD_SECRET not in repr(collection.to_dict())
    finally:
        await env.close()
        error_message.reset(error_token)
        finish_message.reset(finish_token)
        current_trajectory.reset(trajectory_token)
        current_trajectory_collection.reset(collection_token)


@pytest.mark.asyncio
async def test_recursive_child_gets_zero_reward_even_for_correct_text(empty_question_cache):
    question = _dev_question()
    QUESTION_CACHE[question.id] = question
    child_task = Task(
        id="fanoutqa-child",
        goal="Return both requested tokens.",
        max_steps=5,
        misc={
            "source_id": question.id,
            "dataset_split": "dev",
            "_is_subagent": True,
        },
        fork_strategy="task",
    )
    env = FanOutQARecursiveEnv(child_task)
    env._state.finished = True
    token = finish_message.set(f"alpha and {GOLD_SECRET}")
    try:
        reward, reward_info = await env.evaluate()
        assert reward == 0.0
        assert reward_info["reward/success"] == 0.0
        assert GOLD_SECRET not in repr(reward_info)
    finally:
        await env.close()
        finish_message.reset(token)


@pytest.mark.asyncio
async def test_test_question_has_no_environment_reference_reward(empty_question_cache):
    question = FanOutQATestQuestion(
        id="fanoutqa-test-no-reference",
        question="Answer the test question.",
        necessary_evidence=[],
        categories=["contract"],
    )
    task = Task(
        id=f"fanoutqa.test.{question.id}",
        goal=question.question,
        max_steps=5,
        misc={
            "source_id": question.id,
            "dataset_split": "test",
            "categories": list(question.categories),
        },
        fork_strategy="task",
    )
    env = FanOutQASingleEnv(task)
    env._state.finished = True
    token = finish_message.set("a plausible answer")
    try:
        reward, reward_info = await env.evaluate()
        assert reward == 0.0
        assert reward_info["reward/success"] == 0.0
        assert reward_info["reason"] == "Reference unavailable for this split."
    finally:
        await env.close()
        finish_message.reset(token)


def test_episode_score_reloads_official_dev_cache_for_a_fresh_worker(monkeypatch, empty_question_cache):
    question = _dev_question()
    monkeypatch.setattr(evaluation.fanoutqa, "load_dev", lambda: [question])

    reward, reward_info = evaluation.score_episode(question.id, "alpha")

    assert QUESTION_CACHE[question.id] is question
    assert reward == pytest.approx(0.5)
    assert reward_info == {"acc.loose": pytest.approx(0.5), "acc.strict": 0.0}


def test_unknown_dev_source_id_raises_after_official_cache_reload(monkeypatch, empty_question_cache):
    monkeypatch.setattr(evaluation.fanoutqa, "load_dev", lambda: [_dev_question()])

    with pytest.raises(ValueError, match="FanOutQA dev reference not found"):
        evaluation.score_episode("unknown-dev-question-id", "alpha")


def test_official_evaluator_receives_original_objects_and_returns_same_result(monkeypatch):
    questions = [_dev_question("official-id")]
    answers = [{"id": "official-id", "answer": "alpha"}]
    marker = object()
    expected_result = object()
    captured = {}

    def fake_evaluate(got_questions, got_answers, **kwargs):
        captured["questions"] = got_questions
        captured["answers"] = got_answers
        captured["kwargs"] = kwargs
        return expected_result

    monkeypatch.setattr(
        evaluation,
        "_official_module",
        lambda name: SimpleNamespace(evaluate=fake_evaluate) if name == "fanoutqa.eval" else None,
    )

    result = evaluation.evaluate_answers(
        answers,
        questions,
        only_score_answered=True,
        llm_cache_key=marker,
    )

    assert result is expected_result
    assert captured["questions"] is questions
    assert captured["answers"] is answers
    assert captured["answers"][0] is answers[0]
    assert captured["kwargs"]["only_score_answered"] is True
    assert captured["kwargs"]["llm_cache_key"] is marker
