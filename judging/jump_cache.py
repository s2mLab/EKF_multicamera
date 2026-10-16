#!/usr/bin/env python3
"""Shared DD jump-segmentation cache helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from judging.dd_analysis import DDSessionAnalysis, analyze_dd_session
from vitpose_ekf_pipeline import KP_INDEX


def jump_segmentation_height_series(points_3d: np.ndarray | None, root_q: np.ndarray) -> np.ndarray:
    """Return the vertical series used to segment jumps."""

    if points_3d is not None:
        points_3d = np.asarray(points_3d, dtype=float)
        if points_3d.ndim == 3 and points_3d.shape[2] >= 3:
            ankles = np.asarray(points_3d[:, [KP_INDEX["left_ankle"], KP_INDEX["right_ankle"]], 2], dtype=float)
            if np.any(np.isfinite(ankles)):
                return np.nanmin(ankles, axis=1)
            all_markers_z = np.asarray(points_3d[:, :, 2], dtype=float)
            if np.any(np.isfinite(all_markers_z)):
                return np.nanmin(all_markers_z, axis=1)
    return np.asarray(root_q[:, 2], dtype=float)


def jump_analysis_cache_key(
    *,
    reconstruction_name: str,
    root_q: np.ndarray,
    points_3d: np.ndarray | None,
    full_q: np.ndarray | None,
    q_names: list[str] | None,
    fps: float,
    height_threshold: float | None,
    height_threshold_range_ratio: float,
    smoothing_window_s: float,
    min_airtime_s: float,
    min_gap_s: float,
    min_peak_prominence_m: float,
    contact_window_s: float,
    angle_mode: str,
    analysis_start_frame: int,
    require_complete_jumps: bool,
) -> tuple[object, ...]:
    """Build a stable in-memory cache key for one DD analysis request."""

    return (
        str(reconstruction_name),
        id(root_q),
        id(points_3d),
        id(full_q),
        tuple(str(name) for name in (q_names or [])),
        float(fps),
        None if height_threshold is None else float(height_threshold),
        float(height_threshold_range_ratio),
        float(smoothing_window_s),
        float(min_airtime_s),
        float(min_gap_s),
        float(min_peak_prominence_m),
        float(contact_window_s),
        str(angle_mode),
        int(analysis_start_frame),
        bool(require_complete_jumps),
    )


def get_cached_jump_analysis(
    cache: dict[tuple[object, ...], DDSessionAnalysis],
    *,
    reconstruction_name: str,
    root_q: np.ndarray,
    points_3d: np.ndarray | None,
    fps: float,
    height_threshold: float | None,
    height_threshold_range_ratio: float,
    smoothing_window_s: float,
    min_airtime_s: float,
    min_gap_s: float,
    min_peak_prominence_m: float,
    contact_window_s: float,
    full_q: np.ndarray | None = None,
    q_names: list[str] | None = None,
    angle_mode: str = "euler",
    analysis_start_frame: int = 0,
    require_complete_jumps: bool = True,
) -> DDSessionAnalysis:
    """Reuse one DD jump analysis when the inputs and parameters match."""

    key = jump_analysis_cache_key(
        reconstruction_name=reconstruction_name,
        root_q=root_q,
        points_3d=points_3d,
        full_q=full_q,
        q_names=q_names,
        fps=fps,
        height_threshold=height_threshold,
        height_threshold_range_ratio=height_threshold_range_ratio,
        smoothing_window_s=smoothing_window_s,
        min_airtime_s=min_airtime_s,
        min_gap_s=min_gap_s,
        min_peak_prominence_m=min_peak_prominence_m,
        contact_window_s=contact_window_s,
        angle_mode=angle_mode,
        analysis_start_frame=analysis_start_frame,
        require_complete_jumps=require_complete_jumps,
    )
    cached = cache.get(key)
    if cached is not None:
        return cached
    analysis = analyze_dd_session(
        np.asarray(root_q, dtype=float),
        float(fps),
        height_values=jump_segmentation_height_series(points_3d, np.asarray(root_q, dtype=float)),
        height_threshold=height_threshold,
        height_threshold_range_ratio=height_threshold_range_ratio,
        smoothing_window_s=smoothing_window_s,
        min_airtime_s=min_airtime_s,
        min_gap_s=min_gap_s,
        min_peak_prominence_m=min_peak_prominence_m,
        contact_window_s=contact_window_s,
        full_q=full_q,
        q_names=q_names,
        points_3d=points_3d,
        angle_mode=angle_mode,
        analysis_start_frame=analysis_start_frame,
        require_complete_jumps=require_complete_jumps,
    )
    cache[key] = analysis
    return analysis
