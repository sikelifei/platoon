from __future__ import annotations

import os
import subprocess
import sys
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

from platoon.analysis.error_analysis import (
    ErrorClusters,
    ErrorIssue,
    _issue_key,
    get_cached_error_explanation,
    llm_cluster_issue_analyses,
)


class ErrorDetails(Static):
    def __init__(self) -> None:
        super().__init__()
        self._md_base: str = ""
        self._md_full: str = ""

    def show_issue(self, it: ErrorIssue) -> None:
        steps = ", ".join(str(x) for x in it.step_refs) if it.step_refs else "-"
        lines = [
            f"## Task `{it.task_id}`",
            f"- Collection: `{it.collection_id}`",
            f"- Title: {it.title}",
            f"- Steps: {steps}",
            f"- Source: `{it.source_path}`",
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


class ErrorApp(App):
    CSS_PATH = None
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("o", "open_viewer", "Open Viewer"),
        Binding("g", "toggle_grouping", "Group: task/cluster"),
        Binding("L", "cluster_analyses", "Cluster Analyses"),
    ]

    def __init__(
        self, issues: List[ErrorIssue], clusters: ErrorClusters, *, analysis_cache_dir: Optional[str] = None
    ) -> None:
        super().__init__()
        self.issues = issues
        self.clusters = clusters
        self.analysis_cache_dir = analysis_cache_dir
        self.table: Optional[DataTable] = None
        self.details = ErrorDetails()
        self.items_flat: List[ErrorIssue] = []
        self.group_by_clusters: bool = False
        self._row_to_item_index: Dict[int, int] = {}
        self._poll_timer: Optional[Timer] = None
        self._last_cursor_row: Optional[int] = None

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
            self._left_container = HorizontalScroll(id="left_pane")
            try:
                self._left_container.styles.width = "60%"
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
            # Divider (draggable) matching compare_tui
            self._divider = SplitDivider(self)
            try:
                self._divider.styles.width = 2
                self._divider.styles.min_width = 2
            except Exception:
                pass
            yield self._divider
            self._right_container = VerticalScroll(id="right_pane")
            try:
                self._right_container.styles.flex = 1
                self._right_container.styles.overflow_x = "hidden"  # type: ignore[attr-defined]
            except Exception:
                pass
            with self._right_container:
                try:
                    self.details.styles.overflow_x = "hidden"  # type: ignore[attr-defined]
                    self.details.styles.overflow_y = "auto"  # type: ignore[attr-defined]
                except Exception:
                    pass
                yield self.details
        yield row
        yield Footer()

    def on_mount(self) -> None:  # type: ignore[override]
        assert self.table is not None
        self.table.add_columns("Task", "Title", "Steps", "Cluster")
        self._populate_table()
        if self.items_flat:
            self.table.focus()
            first_data_row = min(self._row_to_item_index.keys()) if self._row_to_item_index else 0
            self.table.cursor_coordinate = (first_data_row, 0)
            self._update_details_for_row(first_data_row)
        self.table.on_cursor_move = self._on_cursor_move  # type: ignore[attr-defined]
        try:
            self._poll_timer = self.set_interval(0.15, self._poll_cursor)
        except Exception:
            self._poll_timer = None

    def _populate_table(self) -> None:
        assert self.table is not None
        self.items_flat.clear()
        self._row_to_item_index.clear()
        try:
            self.table.clear(rows=True)
        except Exception:
            pass

        def add_row(it: ErrorIssue, cluster_label: str) -> None:
            steps = ",".join(str(x) for x in it.step_refs) if it.step_refs else "-"
            row_before = self.table.row_count
            self.table.add_row(it.task_id, it.title, steps, cluster_label)
            item_index = len(self.items_flat)
            self._row_to_item_index[row_before] = item_index
            self.items_flat.append(it)

        if not self.group_by_clusters:
            # Group by task id
            from collections import defaultdict

            task_to_issues: Dict[str, List[ErrorIssue]] = defaultdict(list)
            for it in self.issues:
                task_to_issues[it.task_id].append(it)
            for task_id in sorted(task_to_issues.keys()):
                self.table.add_row(f"— {task_id} —", "", "", "")
                for it in task_to_issues[task_id]:
                    add_row(it, self._find_cluster_label(it))
        else:
            # Group by clusters
            for label in sorted(self.clusters.clusters.keys()):
                self.table.add_row(f"— {label} —", "", "", "")
                for it in self.clusters.clusters[label]:
                    add_row(it, label)

    def _find_cluster_label(self, it: ErrorIssue) -> str:
        for label, items in self.clusters.clusters.items():
            if it in items:
                return label
        return "-"

    def _on_cursor_move(self, coordinate) -> None:  # type: ignore[no-redef]
        row = coordinate.row
        idx = self._row_to_item(row)
        if idx is not None:
            self._update_details_for_row(row)

    def _row_to_item(self, row_index: int) -> Optional[int]:
        if row_index in self._row_to_item_index:
            return self._row_to_item_index[row_index]
        higher = [r for r in self._row_to_item_index.keys() if r > row_index]
        if higher:
            return self._row_to_item_index[min(higher)]
        lower = [r for r in self._row_to_item_index.keys() if r < row_index]
        if lower:
            return self._row_to_item_index[max(lower)]
        return None

    def _poll_cursor(self) -> None:
        if self.table is None:
            return
        coord = getattr(self.table, "cursor_coordinate", None)
        row = getattr(coord, "row", None) if coord is not None else None
        if row is None:
            return
        if getattr(self, "_last_cursor_row", None) == row:
            return
        self._last_cursor_row = row
        idx = self._row_to_item(row)
        if idx is not None:
            self._update_details_for_row(row)

    def _update_details_for_row(self, row: int) -> None:
        it = self.items_flat[self._row_to_item(row) or 0]
        self.details.show_issue(it)
        # Prefer cached analysis text; fallback to issue.reason
        try:
            txt = get_cached_error_explanation(it, cache_dir=self.analysis_cache_dir)
            if isinstance(txt, str) and txt.strip():
                self.details.show_analysis(txt.strip())
            elif isinstance(it.reason, str) and it.reason.strip():
                self.details.show_analysis(it.reason.strip())
        except Exception:
            pass

    def action_toggle_grouping(self) -> None:
        self.group_by_clusters = not self.group_by_clusters
        self._populate_table()

    def set_split(self, pct: int) -> None:
        pct = max(10, min(90, int(pct)))
        try:
            self._left_container.styles.width = f"{pct}%"
        except Exception:
            pass
        try:
            self.refresh()
        except Exception:
            pass

    def action_cluster_analyses(self) -> None:
        # Only cluster items that have analysis (cached preferred)
        items_with_analyses: List[ErrorIssue] = []
        analyses: Dict[str, str] = {}
        for it in self.items_flat:
            txt = get_cached_error_explanation(it, cache_dir=self.analysis_cache_dir)
            if isinstance(txt, str) and txt.strip():
                items_with_analyses.append(it)
                analyses[_issue_key(it)] = txt.strip()
            elif isinstance(it.reason, str) and it.reason.strip():
                items_with_analyses.append(it)
        if not items_with_analyses:
            return
        try:
            clusters = llm_cluster_issue_analyses(items_with_analyses, analyses)
            self.clusters = clusters
            self.group_by_clusters = True
            self._populate_table()
        except Exception:
            pass

    def action_open_viewer(self) -> None:
        if self.table is None or not self.items_flat:
            return
        row = getattr(self.table, "cursor_coordinate", None)
        row_index = getattr(row, "row", 0) if row is not None else 0
        item_index = self._row_to_item(row_index)
        if item_index is None or not (0 <= item_index < len(self.items_flat)):
            return
        it = self.items_flat[item_index]
        # Render the specific collection dump as events
        try:
            # The ErrorIssue contains only metadata; open source file by collection
            # We don't have the dump here, so rely on compare.dump_to_temp_events_jsonl caller when needed.
            # As a pragmatic approach, open the entire source file via show-dump fast replay
            cmd = [
                sys.executable,
                "-m",
                "platoon.visualization.cli",
                "show-dump",
                str(it.source_path),
            ]
            suspender = getattr(self, "suspend", None)
            if callable(suspender):
                with self.suspend():  # type: ignore[attr-defined]
                    subprocess.call(cmd, env=os.environ.copy())
            else:
                subprocess.call(cmd, env=os.environ.copy())
        except Exception:
            pass


def run_error_ui(
    issues: List[ErrorIssue], clusters: ErrorClusters, *, analysis_cache_dir: Optional[str] = None
) -> None:
    app = ErrorApp(issues, clusters, analysis_cache_dir=analysis_cache_dir)
    app.run()


__all__ = ["run_error_ui"]


class SplitDivider(Static):
    """Draggable divider to resize left/right panes."""

    def __init__(self, viewer: "ErrorApp") -> None:
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
