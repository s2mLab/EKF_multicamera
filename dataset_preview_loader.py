#!/usr/bin/env python3
"""Shared dataset-first preview loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from dataset_preview_state import DatasetPreviewState, build_dataset_preview_state


@dataclass
class DatasetPreviewLoadResult:
    """Aggregated resources needed by dataset-first preview tabs."""

    sources: dict[str, str]
    catalog: list[dict[str, object]]
    bundle: dict[str, object]
    preview_state: DatasetPreviewState


def load_dataset_preview_resources(
    *,
    output_dir: Path,
    preferred_names: list[str],
    fallback_count: int,
    extra_rows: list[dict[str, object]] | None = None,
    dataset_source_paths_fn: Callable[..., dict[str, str]],
    discover_catalog_fn: Callable[[Path, Path | None], list[dict[str, object]]],
    bundle_loader_fn: Callable[[Path, Path | None, Path | None, bool], dict[str, object]],
    pose2sim_trc: Path | None = None,
    calib: Path | None = None,
    keypoints: Path | None = None,
    biomod_path: Path | None = None,
    align_root: bool = False,
) -> DatasetPreviewLoadResult:
    """Load dataset sources, catalog, preview bundle, and derived GUI state in one step."""

    sources = dataset_source_paths_fn(
        output_dir,
        calib=calib,
        keypoints=keypoints,
        pose2sim_trc=pose2sim_trc,
    )
    pose2sim_source = Path(sources["pose2sim_trc"]) if sources.get("pose2sim_trc") else None
    catalog = discover_catalog_fn(output_dir, pose2sim_trc)
    bundle = bundle_loader_fn(output_dir, biomod_path, pose2sim_source, align_root)
    preview_state = build_dataset_preview_state(
        catalog=catalog,
        bundle=bundle,
        preferred_names=preferred_names,
        fallback_count=fallback_count,
        extra_rows=extra_rows,
    )
    return DatasetPreviewLoadResult(
        sources=sources,
        catalog=catalog,
        bundle=bundle,
        preview_state=preview_state,
    )
