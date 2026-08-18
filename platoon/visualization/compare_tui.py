from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll

try:
    from textual.containers import HorizontalScroll  # type: ignore
except Exception:
    HorizontalScroll = VerticalScroll  # type: ignore
from rich.markdown import Markdown
from rich.text import Text
from textual.events import MouseDown, MouseMove, MouseUp
from textual.timer import Timer
from textual.widgets import DataTable, Footer, Header, Static

from platoon.analysis.compare import (
    CompareItem,
    CompareSummary,
    dump_to_temp_events_jsonl,
    get_cached_explanation,
    llm_cluster_analyses,
    read_clusters_cache,
)


class CompareDetails(Static):
    def __init__(self) -> None:
        super().__init__()
        self._md_base: str = ""
        self._md_full: str = ""

    def show_item(self, item: CompareItem, a_label: str, b_label: str) -> None:
        def _task_text_from_mc(mc) -> Optional[str]:
            try:
                if mc is None:
                    return None
                dump = mc.collection_dump or {}
                trajs = dump.get("trajectories") or {}
                # Prefer the first traj in insertion order
                first = next(iter(trajs.values())) if isinstance(trajs, dict) and trajs else None
                task = first.get("task") if isinstance(first, dict) else None
                if isinstance(task, dict):
                    # Prefer explicit goal field; fallback to id or full dict string
                    if "goal" in task and isinstance(task["goal"], str):
                        return task["goal"]
                    if "id" in task and isinstance(task["id"], str):
                        return task["id"]
                    return str(task)
                elif task is not None:
                    return str(task)
            except Exception:
                return None
            return None

        def _task_text(item: CompareItem) -> str:
            txt = _task_text_from_mc(item.a)
            if txt is None:
                txt = _task_text_from_mc(item.b)
            return txt or "<unknown>"

        def _source(mc):
            return f"`{mc.source_path}`" if mc else "-"

        def _summary_lines(name: str, mc):
            if mc is None:
                return [f"### {name}", "- (missing)"]
            return [
                f"### {name}",
                f"- success: {mc.success}",
                f"- steps: {mc.steps_total}",
                f"- source: {_source(mc)}",
            ]

        a_steps = ", ".join(str(x) for x in item.interesting_steps.get("A", [])) or "-"
        b_steps = ", ".join(str(x) for x in item.interesting_steps.get("B", [])) or "-"

        lines: list[str] = [
            f"## Task `{item.task_id}`",
            f"- Goal: {_task_text(item)}",
            f"- Winner: **{item.winner}** ({item.rationale})",
            f"- Cluster: `{item.cluster_key}`",
            "",
            *_summary_lines(a_label, item.a),
            f"- interesting steps: {a_steps}",
            "",
            *_summary_lines(b_label, item.b),
            f"- interesting steps: {b_steps}",
        ]
        self._md_base = "\n".join(lines)
        self._md_full = self._md_base
        self.update(Markdown(self._md_base))

    def show_analysis(self, text: str) -> None:
        md = self._md_base + "\n\n### Analysis\n\n" + (text or "")
        self._md_full = md
        self.update(Markdown(md))

    def get_markdown(self) -> str:
        return self._md_full or self._md_base


class CompareApp(App):
    CSS_PATH = None
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("o", "open_viewer", "Open Viewer"),
        Binding("g", "toggle_grouping", "Group: winner/cluster"),
        Binding("enter", "show_details", show=False),
        Binding("c", "copy_details", "Copy"),
        Binding("L", "cluster_analyses", "Cluster Analyses"),
    ]

    def __init__(
        self, summary: CompareSummary, a_label: str, b_label: str, *, analysis_cache_dir: Optional[str] = None
    ) -> None:
        super().__init__()
        self.summary = summary
        self.a_label = a_label
        self.b_label = b_label
        self.analysis_cache_dir = analysis_cache_dir
        self.table: Optional[DataTable] = None
        self.details = CompareDetails()
        self.items_flat: List[CompareItem] = []
        self._section_rows: Dict[str, int] = {}
        self.group_by_clusters: bool = False
        self.analysis_clusters: Optional[Dict[str, List[CompareItem]]] = None
        self._row_to_item_index: Dict[int, int] = {}
        self._poll_timer: Optional[Timer] = None
        self._last_cursor_row: Optional[int] = None
        # Split state
        self._split_pct: int = 60
        self._left_container: Optional[HorizontalScroll] = None
        self._right_container: Optional[VerticalScroll] = None
        self._divider: Optional[Static] = None

    def compose(self) -> ComposeResult:  # type: ignore[override]
        yield Header(show_clock=True)
        row = Horizontal()
        try:
            row.styles.gap = 0
            row.styles.padding = 0
            row.styles.margin = 0
        except Exception:
            pass
        with row:
            # Left pane
            self._left_container = HorizontalScroll(id="left_pane")
            try:
                self._left_container.styles.width = f"{self._split_pct}%"
                self._left_container.styles.min_width = 30
            except Exception:
                pass
            with self._left_container:
                self.table = DataTable(zebra_stripes=True)
                self.table.cursor_type = "row"
                try:
                    self.table.styles.min_width = 30
                except Exception:
                    pass
                yield self.table

            # Divider (draggable)
            self._divider = SplitDivider(self)
            try:
                self._divider.styles.width = 2
                self._divider.styles.min_width = 2
            except Exception:
                pass
            yield self._divider

            # Right pane
            self._right_container = VerticalScroll(id="right_pane")
            try:
                self._right_container.styles.flex = 1
                # Use wrapping instead of horizontal scroll for readability
                self._right_container.styles.overflow_x = "hidden"  # type: ignore[attr-defined]
            except Exception:
                pass
            with self._right_container:
                try:
                    self.details.styles.overflow_x = "hidden"  # type: ignore[attr-defined]
                    self.details.styles.overflow_y = "auto"  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    self.details.can_focus = True  # type: ignore[attr-defined]
                except Exception:
                    pass
                yield self.details
        yield row
        yield Footer()

    def on_mount(self) -> None:  # type: ignore[override]
        assert self.table is not None
        self.table.add_columns("Task", "Winner", self.a_label, self.b_label, "Cluster")
        # Load precomputed clusters if any
        try:
            cached = read_clusters_cache(cache_dir=self.analysis_cache_dir)
            if cached:
                # Map task_id back to items
                id_to_item: Dict[str, CompareItem] = {
                    it.task_id: it
                    for it in (
                        self.summary.a_better + self.summary.b_better + self.summary.ties + self.summary.unmatched
                    )
                }
                clusters_items: Dict[str, List[CompareItem]] = {}
                for label, ids in cached.items():
                    clusters_items[label] = [id_to_item[i] for i in ids if i in id_to_item]
                if any(clusters_items.values()):
                    self.analysis_clusters = clusters_items
        except Exception:
            pass

        # Section headers
        def _header(label: str) -> None:
            self._section_rows[label] = self.table.row_count
            self.table.add_row(label, "", "", "", "")
            try:
                self.table.add_section(label)  # visual break if supported
            except Exception:
                pass

        def _add_items(items: List[CompareItem]):
            for it in items:
                a_s = "-" if it.a is None else f"ok={it.a.success}, steps={it.a.steps_total}"
                b_s = "-" if it.b is None else f"ok={it.b.success}, steps={it.b.steps_total}"
                self.table.add_row(it.task_id, it.winner, a_s, b_s, it.cluster_key)
                self.items_flat.append(it)

        self._populate_table()

        if self.items_flat:
            self.table.focus()
            # Focus the first data row (skip headers)
            first_data_row = min(self._row_to_item_index.keys()) if self._row_to_item_index else 0
            self.table.cursor_coordinate = (first_data_row, 0)
            self._update_details_for_item_index(self._row_to_item_index.get(first_data_row, 0))

        self.table.add_row(*([" "] * 5))
        self.table.add_row("Use 'o' to open selected pair in the event viewer", "", "", "", "")

        self.table.on_cursor_move = self._on_cursor_move  # type: ignore[attr-defined]
        # Fallbacks for different Textual versions
        try:
            self.table.on_row_selected = self._on_row_selected  # type: ignore[attr-defined]
        except Exception:
            pass
        # Periodic polling fallback
        try:
            self._poll_timer = self.set_interval(0.15, self._poll_cursor)
        except Exception:
            self._poll_timer = None

    def set_split(self, pct: int) -> None:
        pct = max(10, min(90, int(pct)))
        self._split_pct = pct
        if self._left_container is not None:
            try:
                self._left_container.styles.width = f"{pct}%"
            except Exception:
                pass
        try:
            self.refresh()
        except Exception:
            pass

    def _populate_table(self) -> None:
        assert self.table is not None
        self.items_flat.clear()
        self._row_to_item_index.clear()
        try:
            self.table.clear(rows=True)
        except Exception:
            pass

        def _header(label: str) -> None:
            self._section_rows[label] = self.table.row_count
            heavy = f"════════ {label} ════════"
            self.table.add_row(heavy, "", "", "", "")
            try:
                self.table.add_section(label)
            except Exception:
                pass

        def _add_items(items: List[CompareItem]):
            for it in items:
                a_s = "-" if it.a is None else f"ok={it.a.success}, steps={it.a.steps_total}"
                b_s = "-" if it.b is None else f"ok={it.b.success}, steps={it.b.steps_total}"
                row_before = self.table.row_count
                self.table.add_row(it.task_id, it.winner, a_s, b_s, it.cluster_key)
                item_index = len(self.items_flat)
                self._row_to_item_index[row_before] = item_index
                self.items_flat.append(it)

        if not self.group_by_clusters:
            _header("A better")
            _add_items(self.summary.a_better)
            _header("B better")
            _add_items(self.summary.b_better)
            # Split ties into "both succeeded" and "both failed"
            both_succeeded = [it for it in self.summary.ties if "Both succeeded" in it.rationale]
            both_failed = [it for it in self.summary.ties if "Both failed" in it.rationale]
            if both_succeeded:
                _header("Both succeeded (tie)")
                _add_items(both_succeeded)
            if both_failed:
                _header("Both failed")
                _add_items(both_failed)
            if self.summary.unmatched:
                _header("Unmatched (only in A or B)")
                _add_items(self.summary.unmatched)
        else:
            # Group by cluster labels; rebuild from all items
            clusters: Dict[str, List[CompareItem]]
            if self.analysis_clusters is not None:
                clusters = self.analysis_clusters
            else:
                from collections import defaultdict

                clusters = defaultdict(list)
                for it in self.summary.a_better + self.summary.b_better + self.summary.ties + self.summary.unmatched:
                    clusters[it.cluster_key].append(it)
            for label in sorted(clusters.keys()):
                _header(label)
                _add_items(clusters[label])

    def action_toggle_grouping(self) -> None:
        self.group_by_clusters = not self.group_by_clusters
        self._populate_table()

    def action_cluster_analyses(self) -> None:
        # Build item list to cluster (winners by default + both-failed ties if present in cache)
        items: List[CompareItem] = []
        items.extend(self.summary.a_better)
        items.extend(self.summary.b_better)
        items.extend([it for it in self.summary.ties if "Both failed" in it.rationale])
        # Gather cached analyses
        analyses: Dict[str, str] = {}
        for it in items:
            txt = get_cached_explanation(it)
            if isinstance(txt, str) and txt.strip():
                analyses[it.task_id] = txt
        if not analyses:
            return
        # Cluster with LLM when available; fallback inside function
        try:
            # Only pass items that actually have analyses
            items_with_analyses = [it for it in items if it.task_id in analyses and analyses[it.task_id].strip()]
            if not items_with_analyses:
                return
            self.analysis_clusters = llm_cluster_analyses(items_with_analyses, analyses)
            self.group_by_clusters = True
            self._populate_table()
        except Exception:
            pass

    def _row_to_item(self, row_index: int) -> Optional[int]:
        if row_index in self._row_to_item_index:
            return self._row_to_item_index[row_index]
        # Find next data row
        higher = [r for r in self._row_to_item_index.keys() if r > row_index]
        if higher:
            return self._row_to_item_index[min(higher)]
        lower = [r for r in self._row_to_item_index.keys() if r < row_index]
        if lower:
            return self._row_to_item_index[max(lower)]
        return None

    def _on_cursor_move(self, coordinate) -> None:  # type: ignore[no-redef]
        row = coordinate.row
        idx = self._row_to_item(row)
        if idx is not None:
            self._update_details_for_item_index(idx)

    def _on_row_selected(self, row_key) -> None:  # type: ignore[no-redef]
        # Some Textual versions pass an int row index, others an object; best-effort
        try:
            row = int(getattr(row_key, "row", row_key))
        except Exception:
            row = 0
        idx = self._row_to_item(row)
        if idx is not None:
            self._update_details_for_item_index(idx)

    def _poll_cursor(self) -> None:
        if self.table is None:
            return
        coord = getattr(self.table, "cursor_coordinate", None)
        row = getattr(coord, "row", None) if coord is not None else None
        if row is None:
            return
        if self._last_cursor_row == row:
            return
        self._last_cursor_row = row
        idx = self._row_to_item(row)
        if idx is not None:
            self._update_details_for_item_index(idx)

    def _update_details_for_item_index(self, item_index: int) -> None:
        if not (0 <= item_index < len(self.items_flat)):
            return
        item = self.items_flat[item_index]
        self.details.show_item(item, self.a_label, self.b_label)
        # Load cached analysis only; heavy batch precomputation happens before UI
        try:
            analysis = get_cached_explanation(item, cache_dir=self.analysis_cache_dir)
            if isinstance(analysis, str) and analysis.strip():
                self.details.show_analysis(analysis.strip())
        except Exception:
            pass

    def action_open_viewer(self) -> None:
        if self.table is None or not self.items_flat:
            return
        row = getattr(self.table, "cursor_coordinate", None)
        row_index = getattr(row, "row", 0) if row is not None else 0
        item_index = self._row_to_item(row_index)
        if item_index is None:
            return
        item = self.items_flat[item_index]
        tmp_paths: List[Path] = []
        if item.a is not None:
            tmp_paths.append(dump_to_temp_events_jsonl(item.a.collection_dump))
        if item.b is not None:
            tmp_paths.append(dump_to_temp_events_jsonl(item.b.collection_dump))
        if not tmp_paths:
            return
        # Spawn the existing viewer in a subprocess to avoid nested Textual apps
        try:
            cmd = [
                sys.executable,
                "-m",
                "platoon.visualization.cli",
                "replay",
                *[str(p) for p in tmp_paths],
                "--delay",
                "0.0",
            ]
            # Suspend current TUI to avoid overlapping UIs in the same terminal
            suspender = getattr(self, "suspend", None)
            if callable(suspender):
                with self.suspend():  # type: ignore[attr-defined]
                    subprocess.call(cmd, env=os.environ.copy())
            else:
                subprocess.call(cmd, env=os.environ.copy())
        except Exception:
            pass

    def action_show_details(self) -> None:
        if self.table is None or not self.items_flat:
            return
        row = getattr(self.table, "cursor_coordinate", None)
        row_index = getattr(row, "row", 0) if row is not None else 0
        item_index = self._row_to_item(row_index)
        if item_index is not None:
            self._update_details_for_item_index(item_index)

    def action_copy_details(self) -> None:
        try:
            text = self.details.get_markdown()
        except Exception:
            text = ""
        if not text:
            return
        copied = False
        try:
            import pyperclip  # type: ignore

            pyperclip.copy(text)
            copied = True
        except Exception:
            copied = False
        if not copied:
            try:
                cache_dir = os.path.join(os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "AgentEcho")
                os.makedirs(cache_dir, exist_ok=True)
                path = os.path.join(cache_dir, "compare_details.md")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                # Best-effort notify by updating the details footer area
                note = f"\n\n_Saved to {path}_"
                self.details.show_analysis(text + note)
            except Exception:
                pass


__all__ = ["run_compare_ui"]


def run_compare_ui(
    summary: CompareSummary, a_label: str = "A", b_label: str = "B", *, analysis_cache_dir: Optional[str] = None
) -> None:
    app = CompareApp(summary=summary, a_label=a_label, b_label=b_label, analysis_cache_dir=analysis_cache_dir)
    app.run()


class SplitDivider(Static):
    """Draggable divider to resize left/right panes."""

    def __init__(self, viewer: "CompareApp") -> None:
        super().__init__(id="split_divider")
        self.viewer = viewer
        self._dragging: bool = False
        try:
            self.styles.cursor = "col-resize"  # type: ignore[attr-defined]
            self.styles.background = "grey23"  # type: ignore[attr-defined]
            self.styles.height = "100%"  # type: ignore[attr-defined]
        except Exception:
            pass

    def render(self) -> Text:  # type: ignore[override]
        return Text("│\n" * 2000)

    def on_mouse_down(self, event: MouseDown) -> None:  # type: ignore[override]
        self._dragging = True
        try:
            self.capture_mouse()
        except Exception:
            pass
        try:
            event.stop()
        except Exception:
            pass

    def on_mouse_up(self, event: MouseUp) -> None:  # type: ignore[override]
        self._dragging = False
        try:
            self.release_mouse()
        except Exception:
            pass
        try:
            event.stop()
        except Exception:
            pass

    def on_mouse_move(self, event: MouseMove) -> None:  # type: ignore[override]
        if not self._dragging:
            return
        try:
            row = getattr(self, "parent", None)
            region = getattr(row, "region", None)
            width = getattr(region, "width", None)
            screen_left = getattr(region, "x", 0)
            screen_x = getattr(event, "screen_x", None)
            if screen_x is None or width is None or width <= 0:
                return
            relative_x = max(0, min(screen_x - screen_left, width))
            pct = int((relative_x / float(width)) * 100)
            self.viewer.set_split(pct)
            event.stop()
        except Exception:
            pass
