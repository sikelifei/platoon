import argparse
import hashlib
import json
import pathlib
import random
from dataclasses import asdict
from typing import Literal

from platoon.envs.base import Task


class NumberSearch:
    def __init__(
        self,
        min_bound: int = 1,
        max_bound: int = 100,
    ):
        if max_bound <= min_bound:
            raise ValueError("max_bound must be greater than min_bound")
        self.min_bound = min_bound
        self.max_bound = max_bound

    def _sample_bounds_around_target(self, target: int, rng: random.Random | None = None) -> tuple[int, int]:
        # Ensure x < target < y while remaining in [min_bound, max_bound]
        rnd = rng if rng is not None else random
        if target <= self.min_bound:
            x = self.min_bound
        else:
            x = rnd.randint(self.min_bound, max(self.min_bound, target - 1))

        if target >= self.max_bound:
            y = self.max_bound
        else:
            y = rnd.randint(min(self.max_bound, target + 1), self.max_bound)

        # Fix degeneracies to maintain x < y and contain target
        if x >= target:
            x = max(self.min_bound, target - 1)
        if y <= target:
            y = min(self.max_bound, target + 1)
        if x >= y:
            # final fallback to a minimal valid window around target
            x = max(self.min_bound, target - 1)
            y = min(self.max_bound, target + 1)
            if x >= y:
                # If still invalid due to tight bounds, expand within limits
                x = self.min_bound
                y = self.max_bound
        return x, y

    def get_task(self, id: str) -> Task:
        target = random.randint(self.min_bound, self.max_bound)
        low, high = self._sample_bounds_around_target(target)
        return Task(
            goal=f"Guess the correct number between {low} and {high}.",
            max_steps=1,
            misc={"low": low, "high": high, "target": target},
            id=id,
        )


def create_number_search_datasets(
    seed: int = 42,
    num_samples: int = 50000,
    eval_size: int = 1000,
    min_bound: int = 0,
    max_bound: int = 1024,
) -> tuple[list[Task], list[Task]]:
    random.seed(seed)
    rng = random.Random(seed)
    generator = NumberSearch(min_bound=min_bound, max_bound=max_bound)

    # Deterministic split by triplet (low, target, high) using hashing
    total_needed = max(1, num_samples + eval_size)
    p_val = eval_size / total_needed

    def is_val_triplet(low: int, target: int, high: int) -> bool:
        h = int(hashlib.sha256(f"{seed}:{low}:{target}:{high}".encode()).hexdigest()[:8], 16)
        return (h / 0xFFFFFFFF) < p_val

    train_data: list[Task] = []
    val_data: list[Task] = []
    used_triplets: set[tuple[int, int, int]] = set()

    # Sample until desired sizes are reached; uniqueness enforced by triplet
    # The space is large (0..1024); this should converge quickly.
    while len(train_data) < num_samples or len(val_data) < eval_size:
        t = rng.randint(min_bound, max_bound)
        low, high = generator._sample_bounds_around_target(t, rng=rng)
        triplet = (low, t, high)
        if triplet in used_triplets:
            continue
        used_triplets.add(triplet)
        if is_val_triplet(low, t, high):
            if len(val_data) < eval_size:
                val_data.append(
                    Task(
                        goal=f"Guess the correct number between {low} and {high}.",
                        max_steps=1,
                        misc={"low": low, "high": high, "target": t},
                        id=f"number_search.val.{len(val_data)}",
                    )
                )
        else:
            if len(train_data) < num_samples:
                train_data.append(
                    Task(
                        goal=f"Guess the correct number between {low} and {high}.",
                        max_steps=1,
                        misc={"low": low, "high": high, "target": t},
                        id=f"number_search.train.{len(train_data)}",
                    )
                )

    return train_data, val_data


TRAIN_DATA: list[str] | None = None
VAL_DATA: list[str] | None = None
TASKS: dict[str, Task] = {}


def get_task_ids(
    split: Literal["train", "val"],
    num_samples_train: int = 50000,
    num_samples_val: int = 1000,
) -> list[str]:
    if split == "train":
        return [f"number_search.train.{i}" for i in range(num_samples_train)]
    if split == "val":
        return [f"number_search.val.{i}" for i in range(num_samples_val)]
    raise ValueError(f"Invalid split: {split}")


def load_task_from_disk(id: str) -> Task:
    parent = pathlib.Path(__file__).parent
    if id.startswith("number_search.train."):
        global TRAIN_DATA
        if TRAIN_DATA is None:
            file = parent / "number_search_train.jsonl"
            TRAIN_DATA = file.read_text().splitlines()
        return Task.from_dict(json.loads(TRAIN_DATA[int(id.split(".")[2])]))
    if id.startswith("number_search.val."):
        global VAL_DATA
        if VAL_DATA is None:
            file = parent / "number_search_val.jsonl"
            VAL_DATA = file.read_text().splitlines()
        return Task.from_dict(json.loads(VAL_DATA[int(id.split(".")[2])]))
    raise ValueError(f"Invalid task id: {id}")


def get_task(id: str) -> Task:
    global TASKS
    if id in TASKS:
        return TASKS[id]
    task = load_task_from_disk(id)
    TASKS[id] = task
    return task


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=50000, help="Number of training samples to generate")
    parser.add_argument("--eval_size", type=int, default=1000, help="Number of validation/test samples to generate")
    args = parser.parse_args()

    train_data, val_data = create_number_search_datasets(num_samples=args.num_samples, eval_size=args.eval_size)

    parent_dir = pathlib.Path(__file__).parent

    train_file = parent_dir / "number_search_train.jsonl"
    with open(train_file, "w") as f:
        for task in train_data:
            json.dump(asdict(task), f)
            f.write("\n")

    val_file = parent_dir / "number_search_val.jsonl"
    with open(val_file, "w") as f:
        for task in val_data:
            json.dump(asdict(task), f)
            f.write("\n")

    print(f"Saved {len(train_data)} training samples to {train_file}")
    print(f"Saved {len(val_data)} validation samples to {val_file}")
