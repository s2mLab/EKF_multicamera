#!/usr/bin/env python3
"""Helpers to present reconstruction timing summaries consistently."""

from __future__ import annotations

import math


def parse_stage_timings(summary: dict[str, object]) -> list[tuple[str, float]]:
    """Extract finite stage timings from a bundle summary.

    The JSON insertion order is preserved so the display order follows the
    writing order from the reconstruction step.
    """
    timings = summary.get("stage_timings_s", {})
    if not isinstance(timings, dict):
        return []
    result: list[tuple[str, float]] = []
    for raw_name, raw_value in timings.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        result.append((str(raw_name), value))
    return result


def format_seconds_brief(value: float | None) -> str:
    if value is None:
        return "-"
    if value < 60.0:
        return f"{value:.2f} s"
    minutes = int(value // 60.0)
    seconds = value - 60.0 * minutes
    return f"{minutes:d} min {seconds:.1f} s"


def humanize_stage_name(stage_name: str) -> str:
    raw = stage_name.strip()
    if raw.endswith("_s"):
        raw = raw[:-2]
    replacements = {
        "total": "Total",
        "triangulation": "Triangulation",
        "epipolar_coherence": "Epipolar coherence",
        "model_creation": "Model creation",
        "ekf_3d": "EKF 3D",
        "ekf_2d": "EKF 2D",
        "ekf_2d_initial_state": "EKF 2D initial state",
        "ekf_2d_init": "EKF 2D init",
        "ekf_2d_loop": "EKF 2D loop",
        "ekf_2d_predict": "EKF 2D predict",
        "ekf_2d_update": "EKF 2D update",
        "ekf_2d_markers": "EKF 2D markers",
        "ekf_2d_marker_jacobians": "EKF 2D marker jacobians",
        "ekf_2d_assembly": "EKF 2D assembly",
        "ekf_2d_solve": "EKF 2D solve",
    }
    if raw in replacements:
        return replacements[raw]
    return raw.replace("_", " ").strip().title()


def compute_time_seconds(summary: dict[str, object]) -> float | None:
    for stage_name, value in parse_stage_timings(summary):
        if stage_name == "total_s":
            return value
    return None


def format_reconstruction_timing_details(summary: dict[str, object]) -> str:
    lines = [
        f"Name: {summary.get('name', '-')}",
        f"Family: {summary.get('family', '-')}",
        f"Frames: {summary.get('n_frames', '-')}",
        f"Sequence duration: {format_seconds_brief(_coerce_float(summary.get('duration_s')))}",
        f"Compute time: {format_seconds_brief(compute_time_seconds(summary))}",
        "",
        "Stage timings:",
    ]
    stage_timings = parse_stage_timings(summary)
    if not stage_timings:
        lines.append("  - no timing details available")
    else:
        for stage_name, value in stage_timings:
            lines.append(f"  - {humanize_stage_name(stage_name)}: {format_seconds_brief(value)}")
    return "\n".join(lines)


def _coerce_float(value: object) -> float | None:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_float):
        return None
    return value_float
