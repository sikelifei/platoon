from platoon.envs.base import Task
from platoon.episode.context import budget_tracker, current_trajectory, current_trajectory_collection
from platoon.episode.trajectory import BudgetExceededError, TrajectoryCollection, TrajectoryStep

from platoon.fanoutqa.rollout import FanOutQATotalStepBudgetTracker, _ParallelBranchTracker


def _add_steps(trajectory, count):
    trajectory.steps.extend(TrajectoryStep() for _ in range(count))


def test_shared_budget_caps_root_parallel_children_and_grandchild():
    collection_token = current_trajectory_collection.set(None)
    trajectory_token = current_trajectory.set(None)
    tracker_token = budget_tracker.set(None)
    try:
        collection = TrajectoryCollection()
        current_trajectory_collection.set(collection)
        tracker = FanOutQATotalStepBudgetTracker(max_depth=4)
        budget_tracker.set(tracker)

        root = collection.create_trajectory()
        collection.set_trajectory_task(root.id, Task(id="root", goal="root", max_steps=30))
        current_trajectory.set(root)
        _add_steps(root, 1)

        first_child_budget = min(15, int(tracker.remaining_budget()) - 1)
        assert first_child_budget == 15
        assert tracker.reserve_budget(first_child_budget + 1)
        second_child_budget = min(15, int(tracker.remaining_budget()) - 1)
        assert second_child_budget == 12
        assert tracker.reserve_budget(second_child_budget + 1)
        assert tracker.remaining_budget() == 0

        child_one = collection.create_trajectory(parent_traj=root)
        collection.set_trajectory_task(
            child_one.id,
            Task(id="child-one", goal="child one", max_steps=first_child_budget),
        )
        child_two = collection.create_trajectory(parent_traj=root)
        collection.set_trajectory_task(
            child_two.id,
            Task(id="child-two", goal="child two", max_steps=second_child_budget),
        )

        current_trajectory.set(child_one)
        _add_steps(child_one, 1)
        grandchild_budget = min(15, int(tracker.remaining_budget()) - 1)
        assert grandchild_budget == 13
        assert tracker.reserve_budget(grandchild_budget + 1)
        grandchild = collection.create_trajectory(parent_traj=child_one)
        collection.set_trajectory_task(
            grandchild.id,
            Task(id="grandchild", goal="grandchild", max_steps=grandchild_budget),
        )
        _add_steps(grandchild, grandchild_budget)
        tracker.release_budget(grandchild_budget + 1)

        current_trajectory.set(child_two)
        _add_steps(child_two, second_child_budget)
        current_trajectory.set(root)
        tracker.release_budget(first_child_budget + 1)
        tracker.release_budget(second_child_budget + 1)
        _add_steps(root, 1)  # The root finishes processing its parallel results.

        assert tracker.used_budget_for(root.id) == 28
        assert tracker.used_budget_for(child_one.id) == 14
        assert tracker.remaining_budget_for(root.id) == 2
        assert tracker.reserve_budget(3) is False
    finally:
        budget_tracker.reset(tracker_token)
        current_trajectory.reset(trajectory_token)
        current_trajectory_collection.reset(collection_token)


def test_depth_four_rejects_a_fifth_recursive_level():
    collection_token = current_trajectory_collection.set(None)
    trajectory_token = current_trajectory.set(None)
    tracker_token = budget_tracker.set(None)
    try:
        collection = TrajectoryCollection()
        current_trajectory_collection.set(collection)
        root = collection.create_trajectory()
        collection.set_trajectory_task(root.id, Task(id="root", goal="root", max_steps=30))
        parent = root
        for depth in range(1, 5):
            child = collection.create_trajectory(parent_traj=parent)
            collection.set_trajectory_task(child.id, Task(id=f"depth-{depth}", goal="child", max_steps=15))
            parent = child
        current_trajectory.set(parent)
        tracker = FanOutQATotalStepBudgetTracker(max_depth=4)
        budget_tracker.set(tracker)
        assert tracker.reserve_budget(1) is False
        try:
            tracker.reserve_budget(1, raise_on_failure=True)
        except BudgetExceededError as exc:
            assert exc.reason == "depth"
        else:
            raise AssertionError("depth limit should reject another child")
    finally:
        budget_tracker.reset(tracker_token)
        current_trajectory.reset(trajectory_token)
        current_trajectory_collection.reset(collection_token)


def test_parallel_branch_stats_measure_overlap_not_sibling_count():
    collection = TrajectoryCollection()
    handler = _ParallelBranchTracker(collection)
    collection.register_event_handlers(handler)
    root = collection.create_trajectory()

    first = collection.create_trajectory(parent_traj=root)
    collection.finish_trajectory(first.id)
    second = collection.create_trajectory(parent_traj=root)
    collection.finish_trajectory(second.id)
    assert root.misc["fanoutqa_parallel_metrics"] == {
        "parallel_branches": 0,
        "max_parallel_branches": 1,
    }

    parallel_collection = TrajectoryCollection()
    parallel_handler = _ParallelBranchTracker(parallel_collection)
    parallel_collection.register_event_handlers(parallel_handler)
    parallel_root = parallel_collection.create_trajectory()
    child_a = parallel_collection.create_trajectory(parent_traj=parallel_root)
    child_b = parallel_collection.create_trajectory(parent_traj=parallel_root)
    parallel_collection.finish_trajectory(child_a.id)
    parallel_collection.finish_trajectory(child_b.id)
    assert parallel_root.misc["fanoutqa_parallel_metrics"] == {
        "parallel_branches": 1,
        "max_parallel_branches": 2,
    }
