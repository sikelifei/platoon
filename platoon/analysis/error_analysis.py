from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from platoon.analysis.compare import (
    _cache_dir as _cmp_cache_dir,
)
from platoon.analysis.compare import (
    _log_once as _cmp_log_once,
)
from platoon.analysis.compare import (
    _parse_json_mapping as _cmp_parse_json_mapping,
)
from platoon.analysis.compare import (
    _response_to_text as _cmp_response_to_text,
)
from platoon.analysis.compare import (
    discover_input_paths as _discover_paths,
)
from platoon.analysis.compare import (
    iter_collection_dumps,
)
from platoon.analysis.compute_metrics import is_success_for_collection


@dataclass
class ErrorIssue:
    task_id: str
    collection_id: str
    title: str
    reason: str
    step_refs: List[int]
    traj_id: Optional[str]
    source_path: Path


@dataclass
class ErrorClusters:
    # label -> issues in that cluster
    clusters: Dict[str, List[ErrorIssue]]


def discover_input_paths(dir_arg: str | None, path_args: List[str]) -> List[Path]:
    return _discover_paths(dir_arg, path_args)


def _first_traj_and_task(dump_obj: dict) -> Tuple[Optional[str], Optional[str]]:
    tr = dump_obj.get("trajectories")
    if isinstance(tr, dict) and tr:
        first_id, first = next(iter(tr.items()))
        if isinstance(first, dict):
            task = first.get("task")
            if isinstance(task, dict):
                return first_id, task.get("id")
    return None, None


def _summarize_traj_for_prompt(traj: dict) -> str:
    rid = traj.get("id")
    reward = traj.get("reward")
    fm = traj.get("finish_message") or traj.get("error_message") or ""
    steps = traj.get("steps") or []
    last_bits: List[str] = []
    if isinstance(steps, list) and steps:
        last = steps[-1]
        if isinstance(last, dict):
            for k in ("error", "output", "code", "thought"):
                v = last.get(k)
                if isinstance(v, str) and v.strip():
                    last_line = v.splitlines()[0].strip()
                    if len(last_line) > 160:
                        last_line = last_line[:157] + "..."
                    last_bits.append(f"{k}={last_line}")
    last_str = "; ".join(last_bits)
    fm_short = str(fm)[:200] if fm else ""
    return f"traj={rid}, reward={reward}, finish='{fm_short}', last=[{last_str}]"


def _heuristic_collection_issues(dump_obj: dict, source: Path) -> List[ErrorIssue]:
    issues: List[ErrorIssue] = []
    cid = str(dump_obj.get("id") or uuid.uuid4().hex)
    first_traj_id, task_id = _first_traj_and_task(dump_obj)
    task_id = task_id or "unknown"
    for tid, traj in (dump_obj.get("trajectories") or {}).items():
        if not isinstance(traj, dict):
            continue
        fm = traj.get("finish_message") or traj.get("error_message") or ""
        title = "failure" if fm else "behavior"
        step_refs: List[int] = []
        steps = traj.get("steps")
        if isinstance(steps, list) and steps:
            step_refs.append(len(steps) - 1)
        issues.append(
            ErrorIssue(
                task_id=task_id,
                collection_id=cid,
                title=title,
                reason=str(fm)[:400],
                step_refs=step_refs,
                traj_id=tid,
                source_path=source,
            )
        )
    return issues


def _maybe_create_llm(model: Optional[str]):
    try:
        from platoon.utils.llm_client import create_llm_client

        return create_llm_client(model=model) if model else create_llm_client()
    except Exception:
        return None


def llm_extract_issues_for_collection(dump_obj: dict, source: Path, *, model: Optional[str] = None) -> List[ErrorIssue]:
    client = _maybe_create_llm(model)
    if client is None:
        return _heuristic_collection_issues(dump_obj, source)

    cid = str(dump_obj.get("id") or uuid.uuid4().hex)
    first_traj_id, task_id = _first_traj_and_task(dump_obj)
    task_id = task_id or "unknown"

    lines: List[str] = []
    trajs = dump_obj.get("trajectories") or {}
    if isinstance(trajs, dict):
        for traj in trajs.values():
            if isinstance(traj, dict):
                lines.append(_summarize_traj_for_prompt(traj))

    user_prompt = (
        "You are analyzing an agent's multi-trajectory attempt for a single task.\n"
        "Summarize distinct failure/anti-patterns and noteworthy behaviors.\n"
        "Return a JSON array of objects: {title, reason, step_refs}. step_refs = list of step indices.\n\n"
        + "\n".join(lines)
    )

    try:
        resp = client.system_completion(
            system_prompt=("Return valid JSON only. Be concise but precise. Titles should be short and descriptive."),
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=900,
        )
        parsed = json.loads(resp)
        issues: List[ErrorIssue] = []
        if isinstance(parsed, list):
            for obj in parsed:
                if not isinstance(obj, dict):
                    continue
                title = str(obj.get("title") or "issue")
                reason = str(obj.get("reason") or "")[:800]
                refs = obj.get("step_refs")
                step_refs = [int(x) for x in refs] if isinstance(refs, list) else []
                issues.append(
                    ErrorIssue(
                        task_id=task_id,
                        collection_id=cid,
                        title=title,
                        reason=reason,
                        step_refs=step_refs,
                        traj_id=first_traj_id,
                        source_path=source,
                    )
                )
        if issues:
            return issues
    except Exception:
        pass
    return _heuristic_collection_issues(dump_obj, source)


def llm_hierarchical_cluster(
    issues: List[ErrorIssue], *, model: Optional[str] = None, passes: int = 2, show_progress: bool = True
) -> ErrorClusters:
    client = _maybe_create_llm(model)
    # Fallback: simple keyword grouping
    if client is None:
        groups: Dict[str, List[ErrorIssue]] = {}
        for it in issues:
            k = _normalize_label(it.title + " " + it.reason)
            groups.setdefault(k, []).append(it)
        return ErrorClusters(groups)

    current_groups: Dict[str, List[int]] = {f"i{i}": [i] for i in range(len(issues))}

    rng_passes = max(1, passes)
    use_progress = show_progress and rng_passes > 0

    progress = None
    task_id = None
    if use_progress:
        try:
            from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeRemainingColumn(),
                transient=True,
            )
            progress.__enter__()
            task_id = progress.add_task("LLM clustering passes", total=rng_passes)
        except Exception:
            progress = None
            task_id = None

    try:
        for _ in range(rng_passes):
            # Build lines per group: first issue as representative
            lines: List[str] = []
            keys = list(current_groups.keys())
            for gk in keys:
                idx = current_groups[gk][0]
                it = issues[idx]
                lines.append(f"{gk}: title={it.title}; reason={it.reason[:200]}")

            user_prompt = (
                "You are clustering error patterns.\n"
                "Input is a list of group keys with a short representative description.\n"
                "Return JSON mapping new_cluster_label -> list of group keys to merge.\n\n" + "\n".join(lines)
            )
            try:
                resp = client.system_completion(
                    system_prompt=("Return strict JSON only. Use 3-8 concise labels."),
                    user_prompt=user_prompt,
                    temperature=0.2,
                    max_tokens=800,
                )
                mapping = json.loads(resp)
                if isinstance(mapping, dict) and mapping:
                    # Merge
                    merged: Dict[str, List[int]] = {}
                    for label, gkeys in mapping.items():
                        if not isinstance(gkeys, list):
                            continue
                        idxs: List[int] = []
                        for k in gkeys:
                            if k in current_groups:
                                idxs.extend(current_groups[k])
                        if idxs:
                            merged[str(label)] = sorted(set(idxs))
                    if merged:
                        current_groups = merged
                        if progress and task_id is not None:
                            try:
                                progress.advance(task_id)
                            except Exception:
                                pass
                        continue
            except Exception:
                pass
            # If LLM merge fails, stop
            break
    finally:
        if progress is not None:
            try:
                progress.__exit__(None, None, None)
            except Exception:
                pass

    final: Dict[str, List[ErrorIssue]] = {}
    for label, idxs in current_groups.items():
        final[label] = [issues[i] for i in idxs if 0 <= i < len(issues)]
    return ErrorClusters(final)


# ----------------------------
# LLM analysis generation (cached) akin to compare.explain_compare_item
# ----------------------------


def _error_cache_dir(override: Optional[str] = None) -> Path:
    return _cmp_cache_dir(default_subdir="analyze_errors", override=override)


def _issue_key_v1(it: ErrorIssue) -> str:
    parts = [
        str(it.task_id),
        str(it.collection_id),
        str(it.traj_id or ""),
        str(it.title or ""),
        ",".join(str(x) for x in (it.step_refs or [])),
        str(it.source_path),
        "v1",
    ]
    m = uuid.uuid5(uuid.NAMESPACE_URL, "|".join(parts))
    return m.hex


def _issue_key(it: ErrorIssue) -> str:
    # Stable key that avoids volatile fields like title/step_refs to improve cache hits across runs
    parts = [
        str(it.task_id),
        str(it.collection_id),
        str(it.traj_id or ""),
        str(it.source_path),
        "v2",
    ]
    m = uuid.uuid5(uuid.NAMESPACE_URL, "|".join(parts))
    return m.hex


def _load_dump_for_issue(it: ErrorIssue) -> Optional[dict]:
    try:
        src = Path(it.source_path)
        # Try to find the matching collection id
        for _, dump in iter_collection_dumps([src]):
            cid = str(dump.get("id")) if isinstance(dump, dict) else None
            if cid and cid == str(it.collection_id):
                return dump
        # Fallback to the first dump in the file
        for _, dump in iter_collection_dumps([src]):
            return dump
    except Exception:
        return None
    return None


def explain_error_issue(
    it: ErrorIssue,
    *,
    model: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """Generate a concise analysis for a single-method error issue with caching."""
    cache_path = _error_cache_dir(override=cache_dir) / f"{_issue_key(it)}.json"
    try:
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as f:
                cached = json.load(f)
            if isinstance(cached, dict) and isinstance(cached.get("analysis"), str):
                return cached["analysis"]
    except Exception:
        pass

    client = _maybe_create_llm(model)
    dump_obj = _load_dump_for_issue(it)
    if client is not None and isinstance(dump_obj, dict):
        try:
            try:
                dump_json = json.dumps(dump_obj, ensure_ascii=False)
            except Exception:
                dump_json = str(dump_obj)
            user_prompt = (
                "You are analyzing a single method's trajectory collection for one task.\n"
                "Explain succinctly WHY the error occurred and HOW to remediate.\n"
                "Cite concrete evidence from finish/error messages and step contents.\n"
                "Reference step indices and root agent/subagent trajectory ids when possible.\n"
                "Text can be markdown.\n\n"
                f"Dump JSON:\n{dump_json}\n"
            )
            resp = client.system_completion(
                system_prompt="Be direct, specific, and helpful. No JSON, just text.",
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=900,
            )
            text = _cmp_response_to_text(resp)
            if isinstance(text, str) and text.strip():
                try:
                    with cache_path.open("w", encoding="utf-8") as f:
                        json.dump({"analysis": text.strip(), "ts": time.time()}, f)
                except Exception:
                    pass
                return text.strip()
        except Exception:
            pass

    # Fallback: use reason text or heuristic instruction
    fallback = (it.reason or "Investigate terminal messages and last steps.").strip()
    try:
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump({"analysis": fallback, "ts": time.time()}, f)
    except Exception:
        pass
    return fallback


def get_cached_error_explanation(it: ErrorIssue, *, cache_dir: Optional[str] = None) -> Optional[str]:
    try:
        base = _error_cache_dir(override=cache_dir)
        # Try v2 key first
        p2 = base / f"{_issue_key(it)}.json"
        if p2.exists():
            with p2.open("r", encoding="utf-8") as f:
                cached = json.load(f)
            if isinstance(cached, dict) and isinstance(cached.get("analysis"), str):
                return cached["analysis"]
        # Fallback to v1 key for backward compatibility
        p1 = base / f"{_issue_key_v1(it)}.json"
        if p1.exists():
            with p1.open("r", encoding="utf-8") as f:
                cached = json.load(f)
            if isinstance(cached, dict) and isinstance(cached.get("analysis"), str):
                return cached["analysis"]
    except Exception:
        return None
    return None


def batch_explain_errors(
    issues: List[ErrorIssue],
    *,
    model: Optional[str] = None,
    cache_dir: Optional[str] = None,
    show_progress: bool = True,
) -> Dict[str, str]:
    results: Dict[str, str] = {}
    total = len(issues)
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
                task = progress.add_task("LLM error analyses", total=total)
                for it in issues:
                    results[_issue_key(it)] = explain_error_issue(it, model=model, cache_dir=cache_dir)
                    progress.advance(task)
            return results
        except Exception:
            pass
    printed_any = False
    for idx, it in enumerate(issues, start=1):
        results[_issue_key(it)] = explain_error_issue(it, model=model, cache_dir=cache_dir)
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


def llm_cluster_issue_analyses(
    issues: List[ErrorIssue],
    analyses: Optional[Dict[str, str]] = None,
    *,
    model: Optional[str] = None,
    passes: int = 2,
    show_progress: bool = True,
) -> ErrorClusters:
    """Cluster issues by their LLM-generated reason texts, similar to compare.llm_cluster_analyses.

    Falls back to heuristic grouping when LLM is unavailable or returns trivial clusters.
    """
    # Build ordered list of (issue, summary_text)
    ordered: List[tuple[ErrorIssue, str]] = []
    if analyses:
        for it in issues:
            k = _issue_key(it)
            v = analyses.get(k, "")
            if isinstance(v, str) and v.strip():
                ordered.append((it, v.strip().replace("\n", " ")[:800]))
    else:
        for it in issues:
            t = (it.reason or "").strip()
            if not t:
                continue
            ordered.append((it, t.replace("\n", " ")[:800]))
    if not ordered:
        _cmp_log_once("error.cluster.analyses.empty", "[error-analysis] No analyses to cluster.")
        return ErrorClusters({})

    client = _maybe_create_llm(model)
    if client is None:
        # Heuristic fallback
        return ErrorClusters(_heuristic_cluster_from_texts(ordered))

    # Initialize groups as singletons
    groups: Dict[str, List[int]] = {f"i{i}": [i] for i in range(len(ordered))}

    rng_passes = max(1, passes)
    use_progress = show_progress and rng_passes > 0

    progress = None
    task_id = None
    if use_progress:
        try:
            from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeRemainingColumn(),
                transient=True,
            )
            progress.__enter__()
            task_id = progress.add_task("LLM analyses clustering", total=rng_passes)
        except Exception:
            progress = None
            task_id = None

    try:
        for _ in range(rng_passes):
            lines: List[str] = []
            keys = list(groups.keys())
            for k in keys:
                idx = groups[k][0]
                it, text = ordered[idx]
                lines.append(f"{k}: title={it.title}; summary={text}")

            prompt = (
                "You are clustering error analysis summaries into higher-level groups.\n"
                "Input is lines with a key and a short summary.\n"
                "Group semantically similar items together; do not output one label per key.\n"
                "Return JSON mapping cluster_label -> list of keys to merge. Use 4-8 concise labels.\n"
                "Avoid singletons except for truly unique cases; put leftovers into a 'misc' group.\n\n"
                + "\n".join(lines)
            )
            try:
                resp = client.system_completion(
                    system_prompt="Output STRICT JSON only.",
                    user_prompt=prompt,
                    temperature=0.2,
                    max_tokens=800,
                )
                text = _cmp_response_to_text(resp)
                mapping = _cmp_parse_json_mapping(text)
                if mapping is None:
                    _cmp_log_once(
                        "error.cluster.analyses.parse_error",
                        f"[error-analysis] LLM analyses clustering parse error; using current groups. head='{text[:120]}'",  # noqa: E501
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
                        if progress and task_id is not None:
                            try:
                                progress.advance(task_id)
                            except Exception:
                                pass
                        continue
            except Exception:
                _cmp_log_once(
                    "error.cluster.analyses.error", "[error-analysis] LLM analyses clustering error; stopping."
                )
                break
            break
    finally:
        if progress is not None:
            try:
                progress.__exit__(None, None, None)
            except Exception:
                pass

    # Convert to mapping
    out: Dict[str, List[ErrorIssue]] = {}
    for label, idxs in groups.items():
        out[label] = [ordered[i][0] for i in idxs if 0 <= i < len(ordered)]

    # Anti-singleton handling
    try:
        total = len(ordered)
        singleton_labels = [lbl for lbl, its in out.items() if len(its) == 1]
        num_singletons = len(singleton_labels)
        is_trivial_identity = len(out) == total and num_singletons == total
        if is_trivial_identity:
            _cmp_log_once(
                "error.cluster.analyses.trivial",
                "[error-analysis] LLM produced trivial singletons; using heuristic clustering.",
            )
            return ErrorClusters(_heuristic_cluster_from_texts(ordered))
        if total >= 4 and num_singletons / float(total) >= 0.6:
            # Keep non-singleton clusters and regroup the rest heuristically
            preserved: Dict[str, List[ErrorIssue]] = {k: v for k, v in out.items() if len(v) >= 2}
            singleton_set = {it for lbl in singleton_labels for it in out[lbl]}
            ordered_singletons = [(it, txt) for (it, txt) in ordered if it in singleton_set]
            heur = _heuristic_cluster_from_texts(ordered_singletons)
            for k, v in heur.items():
                label = f"misc:{k}"
                if label in preserved:
                    preserved[label].extend(v)
                else:
                    preserved[label] = v
            return ErrorClusters(preserved)
    except Exception:
        pass
    return ErrorClusters(out)


def _heuristic_cluster_from_texts(ordered: List[tuple[ErrorIssue, str]]) -> Dict[str, List[ErrorIssue]]:
    keywords = {
        "timeout": ["timeout", "time out", "timed out"],
        "tool": ["tool", "browser", "search", "api"],
        "compile": ["compile", "syntax", "parse error"],
        "assert": ["assert", "failed test", "unit test"],
        "exception": ["exception", "traceback", "error:"],
        "planning": ["plan", "planning", "loop", "repeated"],
        "io": ["file", "path", "read", "write", "permission"],
    }
    buckets: Dict[str, List[ErrorIssue]] = {k: [] for k in keywords.keys()}
    misc: List[ErrorIssue] = []
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
    out: Dict[str, List[ErrorIssue]] = {k: v for k, v in buckets.items() if v}
    if misc:
        out["misc"] = misc
    return out


def _normalize_label(text: str) -> str:
    t = (text or "").lower()
    for kw in ("timeout", "assert", "compile", "invalid", "exception", "tool", "plan"):
        if kw in t:
            return f"{kw}"
    return "other"


def analyze_errors(
    label: str,
    paths: List[Path],
    *,
    model: Optional[str] = None,
    passes: int = 2,
    include_successes: bool = False,
    llm_issues: bool = False,
) -> Tuple[List[ErrorIssue], ErrorClusters]:
    all_issues: List[ErrorIssue] = []
    for source_path, dump_obj in iter_collection_dumps(paths):
        try:
            # Skip successful collections unless explicitly included
            ok, _ = is_success_for_collection(dump_obj)
            if ok and not include_successes:
                continue
            # Prefer heuristic extraction unless explicitly enabled
            if llm_issues and model is not None:
                issues = llm_extract_issues_for_collection(dump_obj, source_path, model=model)
            else:
                issues = _heuristic_collection_issues(dump_obj, source_path)
            all_issues.extend(issues)
        except Exception:
            continue
    clusters = llm_hierarchical_cluster(all_issues, model=model, passes=passes)
    return all_issues, clusters


__all__ = [
    "ErrorIssue",
    "ErrorClusters",
    "discover_input_paths",
    "analyze_errors",
    "llm_extract_issues_for_collection",
    "llm_hierarchical_cluster",
]
