from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from platoon.analysis.appworld_metrics import (
    get_first_traj_and_task_id,
)
from platoon.analysis.appworld_metrics import (
    num_steps_for_collection as num_steps_for_collection_in_dump,
)
from platoon.analysis.compute_metrics import is_success_for_collection

# ----------------------------
# Models
# ----------------------------


@dataclass
class MethodCollection:
    method: str
    source_path: Path
    collection_dump: dict
    task_id: Optional[str]
    first_traj_id: Optional[str]
    success: bool
    steps_total: int
    reward_used: Optional[float]


@dataclass
class CompareItem:
    task_id: str
    a: Optional[MethodCollection]
    b: Optional[MethodCollection]
    winner: str  # "A", "B", or "tie" or "unknown"
    rationale: str
    interesting_steps: Dict[str, List[int]]  # method -> list of step indices to inspect
    cluster_key: str  # lightweight clustering label


@dataclass
class CompareSummary:
    a_better: List[CompareItem]
    b_better: List[CompareItem]
    ties: List[CompareItem]
    unmatched: List[CompareItem]


# ----------------------------
# Diagnostics flags
# ----------------------------

_LLM_WARNED_ANALYSIS: bool = False
_LLM_WARNED_CLUSTER: bool = False

# One-time logging guard to avoid noisy stderr
_LOG_ONCE_EMITTED: Dict[str, bool] = {}


def _log_once(key: str, msg: str) -> None:
    if _LOG_ONCE_EMITTED.get(key):
        return
    try:
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()
    except Exception:
        pass
    _LOG_ONCE_EMITTED[key] = True


def _response_to_text(resp: Any) -> str:
    """Coerce various client response shapes to a plain text string."""
    try:
        if isinstance(resp, str):
            return resp.strip()
        if isinstance(resp, bytes):
            try:
                return resp.decode("utf-8", errors="ignore").strip()
            except Exception:
                return str(resp).strip()
        if isinstance(resp, dict):
            # If it's already a mapping (label -> indices), return sentinel to signal JSON-ready
            if all(isinstance(k, str) for k in resp.keys()):
                return json.dumps(resp)
            # Otherwise try common text fields
            for k in ("content", "text", "message"):
                v = resp.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        # Fallback to str()
        return str(resp).strip()
    except Exception:
        return ""


def _parse_json_mapping(text: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from text. Handles code fences and extra prose."""
    if not text:
        return None
    # Strip common code fences
    t = text.strip()
    if t.startswith("```"):
        try:
            first = t.find("\n")
            if first != -1:
                t = t[first + 1 :]
            if t.endswith("```"):
                t = t[:-3]
        except Exception:
            pass
        t = t.strip()
    # Try direct parse
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Try to extract the first {...} block
    try:
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(t[start : end + 1])
            if isinstance(obj, dict):
                return obj
    except Exception:
        pass
    return None


# ----------------------------
# I/O helpers
# ----------------------------


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


def _iter_event_records_from_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get("type"):
                yield obj


def _coerce_step_dict_with_event_reward(step: Any, event_reward: Any) -> dict:
    # Ensure we have a dict-like step and propagate event-level reward for success checks
    if isinstance(step, dict):
        step_dict = dict(step)
    else:
        try:
            step_dict = json.loads(json.dumps(step, default=str))
        except Exception:
            step_dict = {"value": str(step)}
    if "reward" not in step_dict and event_reward is not None:
        try:
            step_dict["reward"] = float(event_reward)
        except Exception:
            pass
    return step_dict


def _aggregate_events_to_collection_dumps(path: Path) -> List[dict]:
    """Aggregate a single event JSONL file into zero or more collection dump dicts.

    Groups by collection_id when present; otherwise treats the file as a single collection.
    """
    collections: Dict[str, dict] = {}
    # Fallback collection id when none provided
    default_cid = f"file::{path.stem}::{uuid.uuid4().hex}"

    for rec in _iter_event_records_from_jsonl(path):
        cid = rec.get("collection_id") or default_cid
        col = collections.get(cid)
        if col is None:
            col = {"id": cid, "trajectories": {}}
            collections[cid] = col

        t = rec.get("type")
        if t == "trajectory_created":
            traj = rec.get("trajectory") or {}
            traj_id = (traj or {}).get("id") or str(uuid.uuid4())
            # Normalize minimal structure
            traj_dump = {
                "id": traj_id,
                "task": (traj or {}).get("task"),
                "parent_info": (traj or {}).get("parent_info"),
                "steps": [],
                "reward": float((traj or {}).get("reward", 0.0) or 0.0),
                "finish_message": (traj or {}).get("finish_message"),
                "error_message": (traj or {}).get("error_message"),
                "misc": (traj or {}).get("misc", {}),
            }
            col["trajectories"][traj_dump["id"]] = traj_dump

        elif t == "trajectory_task_set":
            tid = rec.get("trajectory_id")
            if tid and tid in col["trajectories"]:
                col["trajectories"][tid]["task"] = rec.get("task")

        elif t == "trajectory_step_added":
            tid = rec.get("trajectory_id")
            if not tid:
                continue
            traj = col["trajectories"].setdefault(
                tid,
                {
                    "id": tid,
                    "task": None,
                    "parent_info": None,
                    "steps": [],
                    "reward": 0.0,
                    "finish_message": None,
                    "error_message": None,
                    "misc": {},
                },
            )
            step = rec.get("step")
            reward = rec.get("reward")
            step_dict = _coerce_step_dict_with_event_reward(step, reward)
            traj["steps"].append(step_dict)
            # Keep cumulative reward if event provides it
            try:
                if reward is not None:
                    traj["reward"] = float(reward)
            except Exception:
                pass
            if rec.get("finish_message") is not None:
                traj["finish_message"] = rec.get("finish_message")
            if rec.get("error_message") is not None:
                traj["error_message"] = rec.get("error_message")

        elif t == "trajectory_finished":
            tid = rec.get("trajectory_id")
            if tid and tid in col["trajectories"]:
                try:
                    r = rec.get("reward")
                    if r is not None:
                        col["trajectories"][tid]["reward"] = float(r)
                except Exception:
                    pass
                if rec.get("finish_message") is not None:
                    col["trajectories"][tid]["finish_message"] = rec.get("finish_message")
                if rec.get("error_message") is not None:
                    col["trajectories"][tid]["error_message"] = rec.get("error_message")

    return list(collections.values())


def iter_collection_dumps(paths: Iterable[Path]) -> Iterator[Tuple[Path, dict]]:
    """Yield (source_path, collection_dump) pairs from a mix of dumps and event JSONLs."""
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".json":
            try:
                with path.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict) and obj.get("trajectories"):
                    yield (path, obj)
            except Exception:
                continue
        elif suffix == ".jsonl":
            # First attempt to treat as dump JSONLs
            had_dump_line = False
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
                        if isinstance(obj, dict) and obj.get("trajectories"):
                            had_dump_line = True
                            yield (path, obj)
                if had_dump_line:
                    continue
            except Exception:
                pass

            # Otherwise treat as event JSONL
            try:
                for dump in _aggregate_events_to_collection_dumps(path):
                    yield (path, dump)
            except Exception:
                continue


# ----------------------------
# Comparison logic
# ----------------------------


def _first_traj_id(dump_obj: dict) -> Optional[str]:
    tr = dump_obj.get("trajectories")
    if isinstance(tr, dict) and tr:
        return next(iter(tr.keys()))
    return None


def _build_method_collections(method_name: str, paths: List[Path]) -> List[MethodCollection]:
    items: List[MethodCollection] = []
    for src_path, dump_obj in iter_collection_dumps(paths):
        first_traj, task_id = get_first_traj_and_task_id(dump_obj)
        if task_id is None:
            # Skip when not identifiable
            continue
        ok, reward_used = is_success_for_collection(dump_obj)
        steps_total = num_steps_for_collection_in_dump(dump_obj)
        items.append(
            MethodCollection(
                method=method_name,
                source_path=src_path,
                collection_dump=dump_obj,
                task_id=task_id,
                first_traj_id=_first_traj_id(dump_obj),
                success=ok,
                steps_total=steps_total,
                reward_used=reward_used,
            )
        )
    return items


def _cluster_key_for_pair(a: Optional[MethodCollection], b: Optional[MethodCollection]) -> str:
    def reason(mc: Optional[MethodCollection]) -> str:
        if mc is None:
            return "missing"
        # Try to extract a short failure/success reason
        ft = None
        if mc.first_traj_id and mc.collection_dump.get("trajectories"):
            traj = mc.collection_dump["trajectories"].get(mc.first_traj_id, {})
            ft = traj.get("finish_message") or traj.get("error_message")
        if mc.success:
            return "success"
        if not ft:
            return "failure:unknown"
        msg = str(ft).lower()
        for kw in ("timeout", "invalid", "compile", "assert", "exception", "tool", "plan"):
            if kw in msg:
                return f"failure:{kw}"
        return "failure:other"

    return f"A:{reason(a)}|B:{reason(b)}"


def _interesting_steps(mc: Optional[MethodCollection]) -> List[int]:
    if mc is None:
        return []
    ftid = mc.first_traj_id
    if not ftid:
        return []
    traj = mc.collection_dump.get("trajectories", {}).get(ftid)
    if not isinstance(traj, dict):
        return []
    steps = traj.get("steps")
    if isinstance(steps, list) and steps:
        return [len(steps) - 1]
    return []


def compare_methods(
    a_name: str,
    a_paths: List[Path],
    b_name: str,
    b_paths: List[Path],
) -> CompareSummary:
    a_items = _build_method_collections(a_name, a_paths)
    b_items = _build_method_collections(b_name, b_paths)

    # Choose best-per-task for each method (success preferred; ignore step counts per request)
    def best_per_task(items: List[MethodCollection]) -> Dict[str, MethodCollection]:
        best: Dict[str, MethodCollection] = {}
        for it in items:
            cur = best.get(it.task_id or "")
            if cur is None:
                best[it.task_id or ""] = it
            else:
                # Prefer success over failure only
                if not cur.success and it.success:
                    best[it.task_id or ""] = it
        return {k: v for k, v in best.items() if k}

    a_best = best_per_task(a_items)
    b_best = best_per_task(b_items)

    all_task_ids = set(a_best.keys()) | set(b_best.keys())

    a_better: List[CompareItem] = []
    b_better: List[CompareItem] = []
    ties: List[CompareItem] = []
    unmatched: List[CompareItem] = []

    for tid in sorted(all_task_ids):
        a = a_best.get(tid)
        b = b_best.get(tid)
        if a is None or b is None:
            winner = "unknown"
            rationale = "task present in only one method"
            item = CompareItem(
                task_id=tid,
                a=a,
                b=b,
                winner=winner,
                rationale=rationale,
                interesting_steps={"A": _interesting_steps(a), "B": _interesting_steps(b)},
                cluster_key=_cluster_key_for_pair(a, b),
            )
            unmatched.append(item)
            continue

        # Determine winner
        if a.success and not b.success:
            winner = "A"
            rationale = "A succeeded, B failed"
            target_list = a_better
        elif b.success and not a.success:
            winner = "B"
            rationale = "B succeeded, A failed"
            target_list = b_better
        elif a.success and b.success:
            winner = "tie"
            rationale = "Both succeeded"
            target_list = ties
        else:
            winner = "tie"
            rationale = "Both failed"
            target_list = ties

        item = CompareItem(
            task_id=tid,
            a=a,
            b=b,
            winner=winner,
            rationale=rationale,
            interesting_steps={"A": _interesting_steps(a), "B": _interesting_steps(b)},
            cluster_key=_cluster_key_for_pair(a, b),
        )
        target_list.append(item)

    return CompareSummary(a_better=a_better, b_better=b_better, ties=ties, unmatched=unmatched)


# ----------------------------
# Optional LLM-based clustering (minimal; enabled by flag in CLI)
# ----------------------------


def llm_cluster_compare_items(items: List[CompareItem], model: Optional[str] = None) -> Dict[str, List[CompareItem]]:
    """Group items into clusters using a lightweight LLM prompt.

    Returns mapping cluster_label -> items.
    """
    client = None
    try:
        from platoon.utils.llm_client import create_llm_client

        try:
            client = create_llm_client(model=model) if model else create_llm_client()
        except Exception:
            client = None
    except Exception:
        client = None

    if client is None:
        _log_once(
            "llm.cluster.items.client_unavailable",
            "[analyze-compare] LLM unavailable for clustering; using heuristic.",
        )
        # Fallback: group by existing cluster_key without using an LLM
        clusters: Dict[str, List[CompareItem]] = {}
        for it in items:
            clusters.setdefault(it.cluster_key, []).append(it)
        return clusters
    else:
        _log_once(
            "llm.cluster.items.client_available",
            "[analyze-compare] LLM client available for clustering.",
        )

    # Build a compact prompt listing cases; if available, enrich using AppWorld prompt builders.
    def _compact_desc(it: CompareItem) -> str:
        def mini(mc: Optional[MethodCollection]) -> str:
            if mc is None:
                return "missing"
            msg = ""
            if mc.first_traj_id and mc.collection_dump.get("trajectories"):
                traj = mc.collection_dump["trajectories"].get(mc.first_traj_id, {})
                fm = traj.get("finish_message") or traj.get("error_message")
                if fm:
                    msg = str(fm)[:200]
            return f"success={mc.success}, steps={mc.steps_total}, note={msg}"

        return f"task={it.task_id}, winner={it.winner}, hint={it.cluster_key}, A[{mini(it.a)}], B[{mini(it.b)}]"

    def _builder_snippet(mc: Optional[MethodCollection]) -> str:
        if mc is None:
            return ""
        # Attempt to use AppWorld prompt builders when available
        try:
            from platoon.agents.appworld.codeact import AppWorldCodeActPromptBuilder  # type: ignore

            builder = AppWorldCodeActPromptBuilder()
            messages = builder.build_messages_from_traj_dump(mc.collection_dump, reward_threshold=-1e9)
            # Take up to the first two conversations and extract a short user prompt excerpt
            parts: List[str] = []
            for conv in messages[:2]:
                msgs = conv.get("messages", [])  # type: ignore[assignment]
                for m in msgs:
                    if m.get("role") == "user":
                        txt = str(m.get("content", ""))
                        if txt:
                            parts.append(txt.strip().replace("\n", " ")[:300])
                            break
            if parts:
                return " | ".join(parts)
        except Exception:
            pass
        return ""

    lines: List[str] = []
    for i, it in enumerate(items):
        a_extra = _builder_snippet(it.a)
        b_extra = _builder_snippet(it.b)
        enriched = _compact_desc(it)
        if a_extra or b_extra:
            enriched += f" | A_ctx={a_extra} | B_ctx={b_extra}"
        lines.append(f"{i}. {enriched}")

    user_prompt = (
        "You are clustering A/B method comparisons by failure/success patterns.\n"
        "Given the following lines (one per task), propose 4-6 short cluster labels "
        "and assign each line to exactly one label.\n"
        "Avoid trivial singletons; group similar items together. "
        "Place leftovers in 'misc'.\n"
        "Respond as JSON object mapping label -> list of indices (0-based).\n\n" + "\n".join(lines)
    )

    try:
        _log_once(
            "llm.cluster.items.invoke",
            f"[analyze-compare] Invoking LLM clustering for {len(items)} items.",
        )
        resp = client.system_completion(
            system_prompt=("You are a concise analyst. Output strict JSON only. Labels should be short."),
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=800,
        )
        text = _response_to_text(resp)
        _log_once(
            "llm.cluster.items.response",
            f"[analyze-compare] Received LLM clustering response (chars={len(text)}).",
        )
        parsed_map = _parse_json_mapping(text)
        if parsed_map is None:
            _log_once(
                "llm.cluster.items.parse_error",
                f"[analyze-compare] LLM clustering parse error; falling back. head='{text[:120]}'",
            )
            by_key: Dict[str, List[CompareItem]] = {}
            for it in items:
                by_key.setdefault(it.cluster_key, []).append(it)
            return by_key
        clusters: Dict[str, List[CompareItem]] = {}
        for label, indices in parsed_map.items():
            try:
                idxs = [int(x) for x in indices]
            except Exception:
                continue
            clusters[label] = [items[i] for i in idxs if 0 <= i < len(items)]
        if clusters:
            # Post-process to avoid trivial singleton-heavy outputs
            try:
                total = len(items)
                singleton_labels = [lbl for lbl, its in clusters.items() if len(its) == 1]
                num_singletons = sum(len(clusters[lbl]) for lbl in singleton_labels)
                is_trivial_identity = len(clusters) == total and all(
                    lbl and len(its) == 1 for lbl, its in clusters.items()
                )
                if is_trivial_identity:
                    # Fallback to grouping by precomputed cluster_key
                    by_key: Dict[str, List[CompareItem]] = {}
                    for it in items:
                        by_key.setdefault(it.cluster_key, []).append(it)
                    return by_key
                if total >= 4 and num_singletons / max(1, total) >= 0.6:
                    # Preserve non-singleton LLM clusters, regroup the rest by cluster_key
                    final_map: Dict[str, List[CompareItem]] = {k: v for k, v in clusters.items() if len(v) >= 2}
                    singleton_set = {it.task_id for lbl in singleton_labels for it in clusters[lbl]}
                    regroup: Dict[str, List[CompareItem]] = {}
                    for it in items:
                        if it.task_id in singleton_set:
                            regroup.setdefault(it.cluster_key, []).append(it)
                    for k, v in regroup.items():
                        label = f"misc:{k}"
                        if label in final_map:
                            final_map[label].extend(v)
                        else:
                            final_map[label] = v
                    return final_map
            except Exception:
                pass
            return clusters
    except Exception as e:
        _log_once(
            "llm.cluster.items.error",
            f"[analyze-compare] LLM clustering error; using heuristic grouping. {e}",
        )
        pass

    # Fallback: group by heuristic cluster_key
    clusters: Dict[str, List[CompareItem]] = {}
    for it in items:
        clusters.setdefault(it.cluster_key, []).append(it)
    return clusters


# ----------------------------
# Utility to materialize a temporary JSONL from a dump for replay
# ----------------------------


def dump_to_temp_events_jsonl(dump_obj: dict, tmp_dir: Optional[str | os.PathLike[str]] = None) -> Path:
    from platoon.visualization.event_sinks import write_events_from_dump_to_jsonl

    base_dir = Path(tmp_dir) if tmp_dir is not None else Path(os.getenv("TMPDIR", "/tmp"))
    out = base_dir / f"trajectory_dump_events_{uuid.uuid4().hex}.jsonl"
    write_events_from_dump_to_jsonl(dump_obj, out)
    return out


__all__ = [
    "MethodCollection",
    "CompareItem",
    "CompareSummary",
    "discover_input_paths",
    "iter_collection_dumps",
    "compare_methods",
    "llm_cluster_compare_items",
    "dump_to_temp_events_jsonl",
    "explain_compare_item",
    "batch_explain_compare",
    "get_cached_explanation",
    "llm_cluster_analyses",
    "write_clusters_cache",
    "read_clusters_cache",
]


# ----------------------------
# LLM-based explanation for a single comparison with caching
# ----------------------------


def _cache_dir(default_subdir: str = "analyze_compare", override: Optional[str] = None) -> Path:
    if override is not None:
        p = Path(override)
    else:
        base = Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        p = base / "AgentEcho" / default_subdir
    p.mkdir(parents=True, exist_ok=True)
    return p


def _item_key(item: CompareItem) -> str:
    def safe(v: Any) -> str:
        try:
            return str(v)
        except Exception:
            return "?"

    a = item.a
    b = item.b
    parts = [
        item.task_id,
        item.winner,
        safe(a and a.success),
        safe(a and a.steps_total),
        safe(a and a.source_path),
        safe(b and b.success),
        safe(b and b.steps_total),
        safe(b and b.source_path),
        "v1",
    ]
    m = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return m


def explain_compare_item(
    item: CompareItem,
    *,
    model: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """Return a concise LLM analysis for the comparison; cached on disk.

    When no LLM is available, return a heuristic explanation string.
    """
    cache_path = _cache_dir(override=cache_dir) / f"{_item_key(item)}.json"
    # Load cache
    try:
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as f:
                cached = json.load(f)
            if isinstance(cached, dict) and isinstance(cached.get("analysis"), str):
                return cached["analysis"]
    except Exception:
        pass

    # Attempt LLM
    client = None
    try:
        from platoon.utils.llm_client import create_llm_client

        client = create_llm_client(model=model) if model else create_llm_client()
    except Exception:
        client = None

    if client is not None:
        _log_once(
            "llm.explain.client_available",
            "[analyze-compare] LLM client available for explanation.",
        )

        # Use raw serialized trajectory collection dumps for both methods
        def dump_or_missing(mc: Optional[MethodCollection]) -> str:
            if mc is None:
                return "(missing)"
            try:
                return json.dumps(mc.collection_dump, ensure_ascii=False)
            except Exception:
                return str(mc.collection_dump)

        user_prompt = (
            "You are analyzing two trajectory collection dumps (A and B) for the SAME "
            "task.\n"
            "Each dump is a JSON serialization of a TrajectoryCollection with one or "
            "more trajectories and steps.\n"
            "Explain succinctly WHY the winner outperformed the other, or WHY both "
            "failed.\n"
            "Cite concrete evidence from errors/finish messages or step contents when "
            "possible.\n"
            "Please also reference the steps and trajectory ids (from root and subagent "
            "trajectories) with concrete examples when possible, to contextualize the "
            "analysis.\n"
            "Text can be markdown.\n\n"
            f"Task: {item.task_id}\n"
            f"Outcome: {item.winner} ({item.rationale})\n"
            f"A dump JSON:\n{dump_or_missing(item.a)}\n\n"
            f"B dump JSON:\n{dump_or_missing(item.b)}\n"
        )
        try:
            _log_once("llm.explain.invoke", "[analyze-compare] Invoking LLM for explanation.")
            resp = client.system_completion(
                system_prompt="Be direct, specific, and helpful. No JSON, just text.",
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=1000,
            )
            text = _response_to_text(resp)
            _log_once(
                "llm.explain.response",
                f"[analyze-compare] Received LLM explanation (chars={len(text)}).",
            )
            analysis_text = text
            if analysis_text:
                try:
                    with cache_path.open("w", encoding="utf-8") as f:
                        json.dump({"analysis": analysis_text, "ts": time.time()}, f)
                except Exception:
                    pass
                return analysis_text
        except Exception:
            _log_once(
                "llm.explain.error",
                "[analyze-compare] LLM explanation error; using heuristic fallback.",
            )
            pass

    # Fallback heuristic
    global _LLM_WARNED_ANALYSIS
    if not _LLM_WARNED_ANALYSIS:
        try:
            sys.stderr.write(
                "[analyze-compare] LLM unavailable for analysis; using heuristic "
                "fallback. Set OPENAI_API_KEY and OPENAI_BASE_URL to enable.\n"
            )
            sys.stderr.flush()
        except Exception:
            pass
        _LLM_WARNED_ANALYSIS = True
    if item.winner in ("A", "B"):
        loser = "B" if item.winner == "A" else "A"
        text = (
            f"{item.winner} likely outperformed {loser}. "
            "Review finish/error messages and recent steps in the raw dumps."
        )
        try:
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump({"analysis": text, "ts": time.time()}, f)
        except Exception:
            pass
        return text
    else:
        text = "No clear winner. Investigate terminal messages and last steps in both dumps."
        try:
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump({"analysis": text, "ts": time.time()}, f)
        except Exception:
            pass
        return text


def get_cached_explanation(item: CompareItem, *, cache_dir: Optional[str] = None) -> Optional[str]:
    """Return cached analysis text if present; never invokes the LLM."""
    try:
        cache_path = _cache_dir(override=cache_dir) / f"{_item_key(item)}.json"
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as f:
                cached = json.load(f)
            if isinstance(cached, dict) and isinstance(cached.get("analysis"), str):
                return cached["analysis"]
    except Exception:
        return None
    return None


def batch_explain_compare(
    items: List[CompareItem],
    *,
    model: Optional[str] = None,
    cache_dir: Optional[str] = None,
    show_progress: bool = True,
) -> Dict[str, str]:
    """Return mapping task_id -> analysis string for the given items, with a progress bar.

    Uses rich.Progress when available; falls back to a simple stderr bar.
    """
    results: Dict[str, str] = {}
    total = len(items)

    if show_progress and total > 0:
        try:
            from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeRemainingColumn(),
                transient=True,
            )
            with progress:
                task = progress.add_task("LLM analyses", total=total)
                for it in items:
                    results[it.task_id] = explain_compare_item(it, model=model, cache_dir=cache_dir)
                    progress.advance(task)
            return results
        except Exception:
            # Fall through to simple stderr bar
            pass

    printed_any = False
    for idx, it in enumerate(items, start=1):
        results[it.task_id] = explain_compare_item(it, model=model, cache_dir=cache_dir)
        if show_progress and total > 0:
            width = 30
            pct = idx / float(total)
            filled = int(width * pct)
            bar = "#" * filled + "-" * (width - filled)
            sys.stderr.write(f"\r[{bar}] {idx}/{total} {it.task_id}")
            sys.stderr.flush()
            printed_any = True
    if printed_any:
        sys.stderr.write("\n")
        sys.stderr.flush()
    return results


# ----------------------------
# LLM hierarchical clustering of analyses
# ----------------------------


def llm_cluster_analyses(
    items: List[CompareItem],
    analyses: Dict[str, str],
    *,
    model: Optional[str] = None,
    passes: int = 2,
) -> Dict[str, List[CompareItem]]:
    global _LLM_WARNED_CLUSTER
    """Cluster compare items by their analysis texts.

    Returns mapping cluster_label -> list of CompareItem. Falls back to heuristic grouping
    by CompareItem.cluster_key when LLM unavailable.
    """
    # Build compact lines
    ordered: List[tuple[CompareItem, str]] = []
    for it in items:
        text = analyses.get(it.task_id, "")
        if not isinstance(text, str):
            continue
        stripped = text.strip()
        if not stripped:
            continue
        clean = stripped.replace("\n", " ")[:800]
        ordered.append((it, clean))

    if not ordered:
        _log_once(
            "llm.cluster.analyses.empty_items",
            "[analyze-compare] No items with LLM analyses to cluster; skipping.",
        )
        return {}

    # LLM client
    client = None
    try:
        from platoon.utils.llm_client import create_llm_client

        client = create_llm_client(model=model) if model else create_llm_client()
    except Exception:
        client = None

    if client is None:
        if not _LLM_WARNED_CLUSTER:
            try:
                sys.stderr.write("[analyze-compare] LLM unavailable for clustering; using heuristic clustering.\n")
                sys.stderr.flush()
            except Exception:
                pass
            _LLM_WARNED_CLUSTER = True
        return _heuristic_cluster_from_analyses(ordered)
    else:
        _log_once(
            "llm.cluster.analyses.client_available", "[analyze-compare] LLM client available for analyses clustering."
        )

    # Initial grouping: each item is its own group with a short title
    groups: Dict[str, List[int]] = {f"i{i}": [i] for i in range(len(ordered))}

    for _ in range(max(1, passes)):
        # Representative for each group
        lines: List[str] = []
        keys = list(groups.keys())
        for k in keys:
            idx = groups[k][0]
            it, text = ordered[idx]
            lines.append(f"{k}: task={it.task_id}; winner={it.winner}; key={it.cluster_key}; summary={text}")

        prompt = (
            "You are clustering analysis summaries into higher-level groups.\n"
            "Input is lines with a key and a short summary.\n"
            "Group semantically similar items together; do not output one label per key.\n"
            "Return JSON mapping cluster_label -> list of keys to merge. Use 4-8 concise labels.\n"
            "Avoid singletons except for truly unique cases; put leftovers into a 'misc' group.\n\n" + "\n".join(lines)
        )
        try:
            _log_once(
                "llm.cluster.analyses.invoke",
                f"[analyze-compare] Invoking LLM analyses clustering for {len(lines)} representatives.",
            )
            resp = client.system_completion(
                system_prompt="Output STRICT JSON only.",
                user_prompt=prompt,
                temperature=0.2,
                max_tokens=800,
            )
            text = _response_to_text(resp)
            _log_once(
                "llm.cluster.analyses.response",
                f"[analyze-compare] Received LLM analyses clustering response (chars={len(text)}).",
            )
            mapping = _parse_json_mapping(text)
            if mapping is None:
                _log_once(
                    "llm.cluster.analyses.parse_error",
                    f"[analyze-compare] LLM analyses clustering parse error; keeping current groups. head='{text[:120]}'",  # noqa: E501
                )
                break
            if isinstance(mapping, dict) and mapping:
                merged: Dict[str, List[int]] = {}
                for label, gkeys in mapping.items():
                    if not isinstance(gkeys, list):
                        continue
                    idxs: List[int] = []
                    for k in gkeys:
                        if k in groups:
                            idxs.extend(groups[k])
                    if idxs:
                        merged[str(label)] = sorted(set(idxs))
                if merged:
                    groups = merged
                    continue
        except Exception as e:
            _log_once(
                "llm.cluster.analyses.error",
                f"[analyze-compare] LLM analyses clustering error; keeping current groups. {e}",
            )
            break
        break

    # Convert to final mapping
    out: Dict[str, List[CompareItem]] = {}
    for label, idxs in groups.items():
        out[label] = [ordered[i][0] for i in idxs if 0 <= i < len(ordered)]

    # Detect trivial or overly singleton-heavy clustering and improve it
    try:
        total_items = len(ordered)
        singleton_labels = [label for label, items in out.items() if len(items) == 1]
        num_singletons = sum(len(out[label]) for label in singleton_labels)
        is_trivial_identity = len(out) == total_items and all(
            label.startswith("i") and len(items) == 1 for label, items in out.items()
        )

        # If everything is a singleton (identity mapping), fallback entirely to heuristic clustering
        if is_trivial_identity:
            if not _LLM_WARNED_CLUSTER:
                try:
                    sys.stderr.write(
                        "[analyze-compare] LLM clustering produced trivial singletons; using heuristic clustering.\n"
                    )
                    sys.stderr.flush()
                except Exception:
                    pass
                _LLM_WARNED_CLUSTER = True
            return _heuristic_cluster_from_analyses(ordered)

        # If too many singletons, keep LLM's non-singleton groups and heuristically group the rest
        if total_items >= 4 and num_singletons / max(1, total_items) >= 0.6:
            # Preserve non-singleton LLM clusters
            final_out: Dict[str, List[CompareItem]] = {k: v for k, v in out.items() if len(v) >= 2}

            # Heuristically re-cluster singleton items
            singleton_task_ids = {it.task_id for label in singleton_labels for it in out[label]}
            ordered_singletons: List[tuple[CompareItem, str]] = [
                (it, text) for (it, text) in ordered if it.task_id in singleton_task_ids
            ]
            if ordered_singletons:
                heur_map = _heuristic_cluster_from_analyses(ordered_singletons)
                for lbl, items in heur_map.items():
                    key = f"misc:{lbl}"
                    if key in final_out:
                        final_out[key].extend(items)
                    else:
                        final_out[key] = list(items)
            return final_out
    except Exception:
        pass
    return out


def _heuristic_cluster_from_analyses(ordered: List[tuple[CompareItem, str]]) -> Dict[str, List[CompareItem]]:
    """Group items by simple keyword buckets derived from their analysis text."""
    keywords = {
        "timeout": ["timeout", "time out", "timed out"],
        "tool": ["tool", "browser", "search", "api"],
        "compile": ["compile", "syntax", "parse error"],
        "assert": ["assert", "failed test", "unit test"],
        "exception": ["exception", "traceback", "error:"],
        "planning": ["plan", "planning", "loop", "repeated"],
        "io": ["file", "path", "read", "write", "permission"],
    }
    buckets: Dict[str, List[CompareItem]] = {k: [] for k in keywords.keys()}
    misc: List[CompareItem] = []
    for it, text in ordered:
        t = text.lower()
        placed = False
        for label, kws in keywords.items():
            if any(kw in t for kw in kws):
                buckets[label].append(it)
                placed = True
                break
        if not placed:
            misc.append(it)
    # Remove empty buckets
    out: Dict[str, List[CompareItem]] = {k: v for k, v in buckets.items() if v}
    if misc:
        out["misc"] = misc
    # If still empty (no analyses), group by existing cluster_key instead
    if not out:
        by_key: Dict[str, List[CompareItem]] = {}
        for it, _ in ordered:
            by_key.setdefault(it.cluster_key, []).append(it)
        return by_key
    return out


# ----------------------------
# Cluster caching helpers
# ----------------------------


def _clusters_cache_path(cache_dir: Optional[str]) -> Path:
    base = _cache_dir(override=cache_dir)
    return base / "clusters.json"


def write_clusters_cache(
    clusters: Dict[str, List[CompareItem]] | Dict[str, List[str]],
    *,
    cache_dir: Optional[str] = None,
) -> None:
    """Persist clusters to cache as label -> list of task_ids."""
    try:
        # Normalize to task_id lists
        mapping: Dict[str, List[str]] = {}
        for label, items in clusters.items():
            if items and isinstance(items[0], CompareItem):  # type: ignore[index]
                mapping[label] = [it.task_id for it in items]  # type: ignore[assignment]
            else:
                mapping[label] = list(items)  # type: ignore[list-item]
        path = _clusters_cache_path(cache_dir)
        with path.open("w", encoding="utf-8") as f:
            json.dump({"clusters": mapping, "ts": time.time()}, f)
    except Exception:
        pass


def read_clusters_cache(*, cache_dir: Optional[str] = None) -> Optional[Dict[str, List[str]]]:
    """Load cached cluster mapping label -> list of task_ids, if present."""
    try:
        path = _clusters_cache_path(cache_dir)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        m = obj.get("clusters") if isinstance(obj, dict) else None
        if isinstance(m, dict):
            out: Dict[str, List[str]] = {}
            for k, v in m.items():
                if isinstance(k, str) and isinstance(v, list):
                    out[k] = [str(x) for x in v]
            return out or None
    except Exception:
        return None
    return None
