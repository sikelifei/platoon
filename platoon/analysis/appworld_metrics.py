from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


def _ensure_appworld_root() -> None:
    """Best-effort to set APPWORLD_ROOT like examples/try_app_world.py."""
    if os.getenv("APPWORLD_ROOT"):
        return
    # Resolve examples/../src/platoon/envs/appworld relative to this file
    here = Path(__file__).resolve()
    candidate = (here.parent.parent.parent / "envs" / "appworld").resolve()
    if candidate.exists():
        os.environ["APPWORLD_ROOT"] = str(candidate)


def _try_import_load_task_ids():
    _ensure_appworld_root()
    try:
        from appworld import load_task_ids  # type: ignore

        return load_task_ids
    except Exception as e:
        raise RuntimeError(
            "Failed to import appworld.load_task_ids. Ensure APPWORLD_ROOT is set or appworld is installed."
        ) from e


def iter_dump_objects(paths: Iterable[Path]) -> Iterator[dict]:
    """Yield trajectory collection dump objects from JSON or JSONL files.

    - .json: file is a single dump object
    - .jsonl: each line is a dump object
    Skips malformed lines and event JSONL lines (which have 'type' but not 'trajectories').
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
                            if "type" in obj and "trajectories" not in obj:
                                continue
                            yield obj
            except Exception:
                continue


def get_first_traj_and_task_id(dump_obj: dict) -> Tuple[Optional[dict], Optional[str]]:
    trajectories = dump_obj.get("trajectories")
    if not isinstance(trajectories, dict) or not trajectories:
        return None, None
    first_traj = next(iter(trajectories.values()))
    if not isinstance(first_traj, dict):
        return None, None
    task = first_traj.get("task") if isinstance(first_traj, dict) else None
    task_id = None
    if isinstance(task, dict):
        task_id = task.get("id")
    return first_traj, task_id


def is_success_for_first_traj(first_traj: dict) -> bool:
    # trajectory-level reward
    try:
        if first_traj.get("reward") is not None and float(first_traj.get("reward")) == 1.0:
            return True
    except Exception:
        pass
    # last step reward
    steps = first_traj.get("steps")
    if isinstance(steps, list) and steps:
        last = steps[-1]
        if isinstance(last, dict) and "reward" in last:
            try:
                return float(last.get("reward")) == 1.0
            except Exception:
                return False
    return False


def num_steps_for_collection(dump_obj: dict) -> int:
    """Count total steps across all trajectories in a collection dump."""
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


def discover_input_paths(dir_arg: str | None, path_args: List[str]) -> List[Path]:
    paths: List[Path] = []
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


def difficulty_name(d: int) -> str:
    # AppWorld difficulties are 1-indexed: 1=easy, 2=medium, 3=hard
    return {1: "easy", 2: "medium", 3: "hard"}.get(d, str(d))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute AppWorld accuracy per difficulty from trajectory collection dumps"
    )
    parser.add_argument("paths", nargs="*", help="JSON/JSONL dump files (JSONL: each line is a dump)")
    parser.add_argument("--dir", dest="dir", default=None, help="Directory of JSON/JSONL dumps")
    parser.add_argument(
        "--difficulties",
        dest="difficulties",
        default="1,2,3",
        help="Comma-separated difficulties to evaluate (1=easy,2=medium,3=hard)",
    )
    parser.add_argument("--denom1", type=int, default=None, help="Override denominator for easy")
    parser.add_argument("--denom2", type=int, default=None, help="Override denominator for medium")
    parser.add_argument("--denom3", type=int, default=None, help="Override denominator for hard")

    args = parser.parse_args()

    load_task_ids = _try_import_load_task_ids()

    try:
        diffs = [int(x.strip()) for x in str(args.difficulties).split(",") if x.strip() != ""]
    except Exception:
        diffs = [1, 2, 3]

    # Build difficulty -> task_id set
    def _fetch_ids_for_diff(d: int) -> List[str]:
        # Try a few calling conventions to be robust across versions
        for caller in (
            lambda: load_task_ids(d),
            lambda: load_task_ids(difficulty=d),  # type: ignore[call-arg]
        ):
            try:
                ids = caller()
                if isinstance(ids, (list, tuple)) and ids:
                    return list(ids)
            except Exception:
                continue
        return []

    diff_to_tasks: Dict[int, set[str]] = {}
    diff_to_tasks_base: Dict[int, set[str]] = {}
    for d in diffs:
        ids = _fetch_ids_for_diff(d)
        full = set(str(x) for x in ids)
        base = set(s.split("_")[0] for s in full)
        diff_to_tasks[d] = full
        diff_to_tasks_base[d] = base

    input_paths = discover_input_paths(args.dir, args.paths)
    if not input_paths:
        parser.error("Provide at least one path or --dir with JSON/JSONL dump files")

    # Accumulators
    totals: Dict[int, int] = {d: 0 for d in diffs}
    successes: Dict[int, int] = {d: 0 for d in diffs}
    steps_totals: Dict[int, int] = {d: 0 for d in diffs}

    for dump_obj in iter_dump_objects(input_paths):
        first_traj, task_id = get_first_traj_and_task_id(dump_obj)
        if task_id is None or first_traj is None:
            continue
        # Determine difficulty by membership
        # Try exact match first, then match by base prefix before underscore
        matched_diff: Optional[int] = None
        for d in diffs:
            in_full = task_id in diff_to_tasks.get(d, set())
            in_base = task_id.split("_")[0] in diff_to_tasks_base.get(d, set())
            if in_full or in_base:
                matched_diff = d
                break
        if matched_diff is None:
            continue
        totals[matched_diff] += 1
        if is_success_for_first_traj(first_traj):
            successes[matched_diff] += 1
        steps_totals[matched_diff] += num_steps_for_collection(dump_obj)

    # Build output
    result: Dict[str, dict] = {}
    for d in diffs:
        denom_override = {1: args.denom1, 2: args.denom2, 3: args.denom3}.get(d)
        denom = denom_override if (denom_override is not None and denom_override > 0) else totals[d]
        acc = (successes[d] / denom) if denom > 0 else 0.0
        avg_steps = (steps_totals[d] / totals[d]) if totals[d] > 0 else 0.0
        result[difficulty_name(d)] = {
            "total_collections": totals[d],
            "successes": successes[d],
            "denominator_used": denom,
            "accuracy": acc,
            "total_steps": steps_totals[d],
            "avg_steps_per_collection": avg_steps,
        }

    # Also provide overall on the evaluated difficulties
    total_all = sum(totals.values())
    success_all = sum(successes.values())
    denom_all = sum(({1: args.denom1, 2: args.denom2, 3: args.denom3}.get(d) or totals[d]) for d in diffs)
    acc_all = (success_all / denom_all) if denom_all > 0 else 0.0
    total_steps_all = sum(steps_totals.values())
    avg_steps_all = (total_steps_all / total_all) if total_all > 0 else 0.0
    result["overall"] = {
        "total_collections": total_all,
        "successes": success_all,
        "denominator_used": denom_all,
        "accuracy": acc_all,
        "total_steps": total_steps_all,
        "avg_steps_per_collection": avg_steps_all,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
