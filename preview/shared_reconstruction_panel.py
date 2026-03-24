#!/usr/bin/env python3
"""Shared reconstruction-selection widgets used by multiple GUI tabs."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk
from typing import Any

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def show_placeholder_figure(
    figure: Figure,
    canvas: FigureCanvasTkAgg,
    message: str,
    *,
    fontsize: int = 11,
    color: str = "#555555",
) -> None:
    """Render a lightweight placeholder message inside one Matplotlib figure."""

    figure.clear()
    ax = figure.subplots(1, 1)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=fontsize, color=color)
    canvas.draw_idle()


class SharedReconstructionPanel(ttk.Frame):
    """Single reconstruction selector shared by the active analysis tab."""

    def __init__(
        self,
        master,
        state: Any,
        *,
        tooltip_fn: Callable[[Any, str], None] | None = None,
    ):
        super().__init__(master)
        self.state = state
        self._default_names: list[str] = []
        self._selection_callback = None
        self._refresh_callback = None
        self._suspend_selection_callback = False

        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=8, pady=(6, 2))
        self.clean_button = ttk.Button(header, text="Clean trial outputs", command=lambda: None)
        self.clean_button.pack(side=tk.RIGHT, padx=(0, 8))
        self.refresh_button = ttk.Button(header, text="Refresh available reconstructions")
        self.refresh_button.pack(side=tk.RIGHT)

        self.tree = ttk.Treeview(
            self,
            columns=("label", "family", "frames", "reproj", "path"),
            show="headings",
            height=5,
            selectmode="extended",
        )
        self.tree.heading("label", text="Reconstruction")
        self.tree.heading("family", text="Family")
        self.tree.heading("frames", text="Frames")
        self.tree.heading("reproj", text="Reproj (px)")
        self.tree.heading("path", text="Path")
        self.tree.column("label", width=240, anchor="w")
        self.tree.column("family", width=90, anchor="w")
        self.tree.column("frames", width=70, anchor="w")
        self.tree.column("reproj", width=95, anchor="w")
        self.tree.column("path", width=540, anchor="w")
        self.tree.pack(fill=tk.X, expand=False, padx=8, pady=(0, 6))
        self.tree.bind("<<TreeviewSelect>>", self._on_selection_changed)
        if tooltip_fn is not None:
            tooltip_fn(self.tree, "Shared reconstruction selector used by the active analysis tab.")
        self.show_placeholder("Select a tab that uses reconstruction comparisons.")

    def configure_for_consumer(
        self,
        *,
        title: str,
        refresh_callback,
        selection_callback,
        selectmode: str = "extended",
    ) -> None:
        """Bind the shared panel to one tab consumer."""

        _ = title
        self._refresh_callback = refresh_callback
        self._selection_callback = selection_callback
        self.refresh_button.configure(command=refresh_callback)
        clean_callback = getattr(self.state, "clean_trial_outputs_callback", None)
        self.clean_button.configure(command=clean_callback or (lambda: None))
        self.tree.configure(selectmode=selectmode)

    def set_rows(self, rows: list[dict[str, object]], default_names: list[str] | None = None) -> None:
        """Populate the shared tree while preserving compatible selections."""

        previous = set(self.selected_names())
        self._default_names = list(default_names or [])
        self._suspend_selection_callback = True
        try:
            for item in self.tree.get_children():
                self.tree.delete(item)
            row_names = []
            for row in rows:
                name = str(row.get("name", ""))
                if not name:
                    continue
                row_names.append(name)
                reproj_mean = row.get("reproj_mean")
                self.tree.insert(
                    "",
                    "end",
                    iid=name,
                    values=(
                        str(row.get("label", name)),
                        str(row.get("family", "-")),
                        row.get("frames", "-"),
                        "-" if reproj_mean is None else f"{float(reproj_mean):.2f}",
                        str(row.get("path", "")),
                    ),
                )
            selection = [name for name in row_names if name in previous]
            if not selection:
                selection = [name for name in row_names if name in self._default_names]
            self.tree.selection_set(selection)
        finally:
            self._suspend_selection_callback = False
        self._publish_selection()

    def selected_names(self) -> list[str]:
        """Return the current shared selection or the default fallbacks."""

        selected = list(self.tree.selection())
        if selected:
            return selected
        return [name for name in self._default_names if self.tree.exists(name)]

    def show_placeholder(self, message: str) -> None:
        """Replace the table content with an empty informative state."""

        self._default_names = []
        self._selection_callback = None
        self._refresh_callback = None
        self.refresh_button.configure(command=lambda: None)
        self.clean_button.configure(command=lambda: None)
        self._suspend_selection_callback = True
        try:
            for item in self.tree.get_children():
                self.tree.delete(item)
            self.tree.insert("", "end", iid="__placeholder__", values=(message, "-", "-", "-", "-"))
            self.tree.selection_remove(self.tree.selection())
        finally:
            self._suspend_selection_callback = False
        self.state.set_shared_reconstruction_selection([])

    def _publish_selection(self) -> None:
        self.state.set_shared_reconstruction_selection(self.selected_names())

    def _on_selection_changed(self, _event=None) -> None:
        if self._suspend_selection_callback:
            return
        self._publish_selection()
        if self._selection_callback is not None:
            self._selection_callback()
