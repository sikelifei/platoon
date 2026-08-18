from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, Tuple


def iter_dump_objects(paths: Iterable[Path]) -> Iterator[dict]:
    """Yield trajectory collection dump objects from JSON or JSONL files.

    - For ``.json`` files: the file contains a single dump object.
    - For ``.jsonl`` files: each line is a dump object.
    Lines that fail to parse are skipped.
    """
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".json":
            try:
                with path.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                continue
        elif suffix == ".jsonl":
            try:
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        if isinstance(obj, dict):
                            # Heuristic: ignore event lines (they have a "type" key)
                            if "type" in obj and "trajectories" not in obj:
                                continue
                            yield obj
            except Exception:
                continue


def is_success_for_collection(dump_obj: dict) -> Tuple[bool, float | None]:
    """Return (success, reward_used) for a trajectory collection dump.

    Success criteria:
    - reward of the first trajectory is exactly 1, OR
    - reward of the last step of the first trajectory is exactly 1.
    """
    trajectories = dump_obj.get("trajectories")
    if not isinstance(trajectories, dict) or not trajectories:
        return (False, None)

    # First trajectory in insertion order
    first_traj = next(iter(trajectories.values()))
    if not isinstance(first_traj, dict):
        return (False, None)

    # Check trajectory-level reward
    traj_reward = first_traj.get("reward")
    try:
        if traj_reward is not None and float(traj_reward) == 1.0:
            return (True, float(traj_reward))
    except Exception:
        pass

    # Check last step's reward if present
    steps = first_traj.get("steps")
    if isinstance(steps, list) and steps:
        last_step = steps[-1]
        if isinstance(last_step, dict) and "reward" in last_step:
            try:
                if float(last_step.get("reward")) == 1.0:
                    return (True, float(last_step.get("reward")))
            except Exception:
                pass

    return (False, None)


def num_steps_for_collection(dump_obj: dict) -> int:
    """Return the total number of steps across all trajectories in the collection."""
    trajectories = dump_obj.get("trajectories")
    if not isinstance(trajectories, dict) or not trajectories:
        return 0
    total_steps = 0
    for traj in trajectories.values():
        if isinstance(traj, dict):
            steps = traj.get("steps")
            if isinstance(steps, list):
                total_steps += len(steps)
    return total_steps


def discover_input_paths(dir_arg: str | None, path_args: list[str]) -> list[Path]:
    paths: list[Path] = []
    if dir_arg:
        d = Path(dir_arg)
        if d.is_dir():
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in {".json", ".jsonl"}:
                    paths.append(p)
    for p in path_args or []:
        pp = Path(p)
        if pp.exists() and pp.suffix.lower() in {".json", ".jsonl"}:
            paths.append(pp)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute successes and accuracy from trajectory collection dumps")
    parser.add_argument("paths", nargs="*", help="JSON/JSONL dump files (JSONL: each line is a dump)")
    parser.add_argument("--dir", dest="dir", default=None, help="Directory containing JSON/JSONL dumps")
    parser.add_argument(
        "--denom",
        dest="denom",
        type=int,
        default=None,
        help="Optional denominator to use for accuracy (overrides number of parsed dumps)",
    )
    args = parser.parse_args()

    input_paths = discover_input_paths(args.dir, args.paths)
    if not input_paths:
        parser.error("Provide at least one path or --dir with JSON/JSONL dump files")

    total = 0
    successes = 0
    sum_steps_across_collections = 0

    for dump_obj in iter_dump_objects(input_paths):
        total += 1
        ok, _ = is_success_for_collection(dump_obj)
        if ok:
            successes += 1
        sum_steps_across_collections += num_steps_for_collection(dump_obj)

    denom = args.denom if (args.denom is not None and args.denom > 0) else total
    accuracy = (successes / denom) if denom > 0 else 0.0
    avg_steps_per_collection = (sum_steps_across_collections / total) if total > 0 else 0.0
    print(
        json.dumps(
            {
                "total_collections": total,
                "successes": successes,
                "denominator_used": denom,
                "accuracy": accuracy,
                "total_steps": sum_steps_across_collections,
                "avg_steps_per_collection": avg_steps_per_collection,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
