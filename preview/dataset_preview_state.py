#!/usr/bin/env python3
"""Helpers to derive GUI preview state from catalog and preview bundles."""

from __future__ import annotations

from dataclasses import dataclass

from reconstruction.reconstruction_presenter import (
    bundle_available_reconstruction_names,
    catalog_rows_for_names,
    default_selection,
)


@dataclass
class DatasetPreviewState:
    """Compact state shared by preview tabs after loading a dataset bundle."""

    rows: list[dict[str, object]]
    defaults: list[str]
    available_names: list[str]
    max_frame: int


def build_dataset_preview_state(
    *,
    catalog: list[dict[str, object]],
    bundle: dict[str, object] | None,
    preferred_names: list[str],
    fallback_count: int,
    include_3d: bool = True,
    include_q: bool = True,
    include_q_root: bool = False,
    extra_rows: list[dict[str, object]] | None = None,
) -> DatasetPreviewState:
    """Build common GUI preview state from a reconstruction catalog and bundle.

    This keeps the selection-table rows, default selections, and frame range
    logic consistent across dataset-first preview tabs.
    """
    available_names = (
        bundle_available_reconstruction_names(
            bundle,
            include_3d=include_3d,
            include_q=include_q,
            include_q_root=include_q_root,
        )
        if bundle is not None
        else []
    )
    row_names_source = available_names or [row.get("name") for row in catalog]
    rows = catalog_rows_for_names(catalog, row_names_source, extra_rows=extra_rows)
    defaults = default_selection([row.get("name") for row in rows], preferred_names, fallback_count=fallback_count)
    frame_count = len(bundle.get("frames", [])) if bundle is not None else 0
    return DatasetPreviewState(
        rows=rows,
        defaults=defaults,
        available_names=[str(name) for name in available_names],
        max_frame=max(frame_count - 1, 0),
    )
