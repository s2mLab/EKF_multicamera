#!/usr/bin/env python3
"""Helpers to store and present reconstruction timing traces consistently."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np


def make_timing_stage(
    stage_id: str,
    label: str,
    *,
    compute_time_s: float | None,
    source: str = "computed_now",
    include_in_total: bool = True,
    cache_path: str | None = None,
    details: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build one normalized timing-stage record for bundle summaries."""

    stage: dict[str, object] = {
        "id": str(stage_id),
        "label": str(label),
        "source": str(source),
        "compute_time_s": _coerce_float(compute_time_s),
        "include_in_total": bool(include_in_total),
    }
    if cache_path:
        stage["cache_path"] = str(cache_path)
    if details:
        stage["details"] = dict(details)
    return stage


def stage_compute_time(stage: dict[str, object]) -> float | None:
    """Extract the finite compute time stored in a timing stage."""

    return _coerce_float(stage.get("compute_time_s"))


def _load_model_stage_compute_time(model_stage_path: Path) -> float | None:
    """Read the cached model generation time from one `model_stage.npz`."""

    if not model_stage_path.exists():
        return None
    try:
        with np.load(model_stage_path, allow_pickle=True) as data:
            if "compute_time_s" not in data:
                return None
            return _coerce_float(np.asarray(data["compute_time_s"]).item())
    except Exception:
        return None


def _candidate_model_stage_paths(summary: dict[str, object]) -> list[Path]:
    """Enumerate candidate `model_stage.npz` paths associated with this summary."""

    candidates: list[Path] = []
    selected_model = summary.get("selected_model")
    if isinstance(selected_model, str) and selected_model.strip():
        biomod_path = Path(selected_model)
        candidates.append(biomod_path.parent / "model_stage.npz")

    cache_paths = summary.get("cache_paths")
    if isinstance(cache_paths, dict):
        raw_model_cache = cache_paths.get("model")
        if isinstance(raw_model_cache, str) and raw_model_cache.strip():
            model_cache_path = Path(raw_model_cache)
            if model_cache_path.name == "model_stage.npz":
                candidates.append(model_cache_path)
            elif model_cache_path.suffix == ".bioMod":
                candidates.append(model_cache_path.parent / "model_stage.npz")

    seen: set[Path] = set()
    result: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(candidate)
    return result


def fallback_model_objective_seconds(summary: dict[str, object]) -> float | None:
    """Recover the model-generation cost when the bundle reuses an existing bioMod."""

    for candidate in _candidate_model_stage_paths(summary):
        value = _load_model_stage_compute_time(candidate)
        if value is not None:
            return value
    return None


def stage_objective_seconds(stage: dict[str, object], summary: dict[str, object]) -> float | None:
    """Return the objective cost of one stage, replacing cache-read times when possible."""

    value = stage_compute_time(stage)
    if str(stage.get("id", "")) != "model_creation":
        return value
    if value is not None and value > 0.0:
        return value
    fallback = fallback_model_objective_seconds(summary)
    if fallback is not None:
        return fallback
    return value


def objective_total_seconds(summary: dict[str, object]) -> float | None:
    """Return the total objective compute time, including reused cached stages."""

    pipeline = summary.get("pipeline_timing")
    if isinstance(pipeline, dict):
        stages = pipeline.get("stages")
        if isinstance(stages, list):
            total = 0.0
            used = False
            for stage in stages:
                if not isinstance(stage, dict) or not bool(stage.get("include_in_total", True)):
                    continue
                value = stage_objective_seconds(stage, summary)
                if value is None:
                    continue
                total += value
                used = True
            if used:
                return total
        explicit_total = _coerce_float(pipeline.get("objective_total_s"))
        if explicit_total is not None:
            return explicit_total
    return compute_time_seconds(summary)


def model_compute_seconds(summary: dict[str, object]) -> float | None:
    """Return the objective time attributable to the model-creation stage."""

    pipeline = summary.get("pipeline_timing")
    if isinstance(pipeline, dict):
        stages = pipeline.get("stages")
        if isinstance(stages, list):
            total = 0.0
            used = False
            for stage in stages:
                if not isinstance(stage, dict):
                    continue
                if str(stage.get("id", "")) != "model_creation":
                    continue
                value = stage_objective_seconds(stage, summary)
                if value is None:
                    continue
                total += value
                used = True
            if used and total > 0.0:
                return total
    stage_timings = dict(parse_stage_timings(summary))
    stage_time = _coerce_float(stage_timings.get("model_creation_s"))
    if stage_time is not None:
        return stage_time
    return fallback_model_objective_seconds(summary)


def reconstruction_run_seconds(summary: dict[str, object]) -> float | None:
    """Return the objective reconstruction time excluding model creation."""

    total = objective_total_seconds(summary)
    if total is None:
        return None
    model_s = model_compute_seconds(summary)
    if model_s is None:
        return total
    return max(0.0, total - model_s)


def current_run_seconds(summary: dict[str, object]) -> float | None:
    """Return the wall time spent by the current run, excluding prior cached work."""

    pipeline = summary.get("pipeline_timing")
    if not isinstance(pipeline, dict):
        return None
    return _coerce_float(pipeline.get("current_run_wall_s"))


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
        value = _coerce_float(raw_value)
        if value is None:
            continue
        result.append((str(raw_name), value))
    return result


def parse_timing_trace(summary: dict[str, object]) -> list[dict[str, object]]:
    """Extract the normalized timing trace from a bundle summary."""

    pipeline = summary.get("pipeline_timing")
    if not isinstance(pipeline, dict):
        return []
    stages = pipeline.get("stages")
    if not isinstance(stages, list):
        return []
    return [stage for stage in stages if isinstance(stage, dict)]


def format_seconds_brief(value: float | None) -> str:
    """Format short timing values for GUI summaries."""

    if value is None:
        return "-"
    if value < 60.0:
        return f"{value:.2f} s"
    minutes = int(value // 60.0)
    seconds = value - 60.0 * minutes
    return f"{minutes:d} min {seconds:.1f} s"


def humanize_stage_name(stage_name: str) -> str:
    """Turn machine stage ids into user-facing labels."""

    raw = stage_name.strip()
    if raw.endswith("_s"):
        raw = raw[:-2]
    replacements = {
        "total": "Total",
        "pose_data": "2D cleaning",
        "pose_data_variant": "2D pose variant",
        "flip_diagnostics": "Flip diagnosis",
        "flip_application": "Apply flip",
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
    """Fallback accessor for legacy total timing fields."""

    for stage_name, value in parse_stage_timings(summary):
        if stage_name == "total_s":
            return value
    return None


def build_pipeline_diagram(stages: Iterable[dict[str, object]]) -> str:
    """Render the high-level stage flow used to produce a reconstruction."""

    labels = []
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        label = str(stage.get("label") or humanize_stage_name(str(stage.get("id", ""))))
        source = str(stage.get("source", "computed_now"))
        if source == "cache":
            label = f"{label} [cache]"
        labels.append(label)
    return " -> ".join(labels) if labels else "No pipeline trace available"


def format_reconstruction_timing_details(summary: dict[str, object]) -> str:
    """Format the detailed timing panel shown in the Reconstructions tab."""

    pipeline_stages = parse_timing_trace(summary)
    objective_time = objective_total_seconds(summary)
    model_time = model_compute_seconds(summary)
    reconstruction_time = reconstruction_run_seconds(summary)
    current_time = current_run_seconds(summary)
    lines = [
        f"Name: {summary.get('name', '-')}",
        f"Family: {summary.get('family', '-')}",
        f"Frames: {summary.get('n_frames', '-')}",
        f"Source FPS: {summary.get('source_fps', summary.get('fps', '-'))}",
        f"Effective FPS: {summary.get('fps', '-')}",
        f"Sequence duration: {format_seconds_brief(_coerce_float(summary.get('duration_s')))}",
        f"Objective compute time: {format_seconds_brief(objective_time)}",
        f"Reconstruction time (excl. model): {format_seconds_brief(reconstruction_time)}",
        f"Model time: {format_seconds_brief(model_time)}",
    ]
    source_path = summary.get("source")
    if isinstance(source_path, str) and source_path.strip():
        lines.append(f"Source file: {source_path}")
    trc_rate_hz = _coerce_float(summary.get("trc_rate_hz"))
    if trc_rate_hz is not None:
        lines.append(f"TRC rate: {trc_rate_hz:.2f} Hz")
    if current_time is not None:
        lines.append(f"Current run wall time: {format_seconds_brief(current_time)}")

    lines.extend(
        [
            "",
            "Pipeline:",
            f"  {build_pipeline_diagram(pipeline_stages)}",
            "",
            "Stage timings:",
        ]
    )
    if pipeline_stages:
        for stage in pipeline_stages:
            value = stage_compute_time(stage)
            source = str(stage.get("source", "computed_now"))
            cache_path = stage.get("cache_path")
            suffix = " [cache]" if source == "cache" else ""
            label = stage.get("label", humanize_stage_name(str(stage.get("id", ""))))
            lines.append(f"  - {label}: {format_seconds_brief(value)}{suffix}")
            if cache_path:
                lines.append(f"      cache: {cache_path}")
    else:
        stage_timings = parse_stage_timings(summary)
        if not stage_timings:
            lines.append("  - no timing details available")
        else:
            for stage_name, value in stage_timings:
                lines.append(f"  - {humanize_stage_name(stage_name)}: {format_seconds_brief(value)}")
    return "\n".join(lines)


def _coerce_float(value: object) -> float | None:
    """Convert a value to a finite float or return ``None``."""

    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_float):
        return None
    return value_float
