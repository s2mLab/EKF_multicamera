#!/usr/bin/env python3
"""Helpers for jump/DD analysis from root kinematics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R
from datetime import datetime
from root_kinematics import (
    TRUNK_ROTATION_NAMES,
    TRUNK_TRANSLATION_NAMES,
    TRUNK_ROOT_ROTATION_SEQUENCE,
    reextract_euler_with_gaps,
    unwrap_with_gaps,
)
DEFAULT_HIP_DOFS = ("LEFT_THIGH:RotY", "RIGHT_THIGH:RotY")
DEFAULT_KNEE_DOFS = ("LEFT_SHANK:RotY", "RIGHT_SHANK:RotY")


def dd_debug(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[DD {timestamp}] {message}", flush=True)


@dataclass
class JumpSegment:
    start: int
    end: int
    peak_index: int


@dataclass
class DDJumpAnalysis:
    segment: JumpSegment
    somersault_turns: float
    twist_turns: float
    max_tilt_rad: float
    mean_tilt_rad: float
    classification: str
    body_shape: str | None
    code: str | None
    twists_per_salto: list[float]
    full_salto_event_indices: list[int]
    quarter_salto_event_indices: list[int]
    half_twist_event_indices: list[int]
    somersault_curve_turns: np.ndarray
    twist_curve_turns: np.ndarray
    tilt_curve_rad: np.ndarray
    angle_mode: str


@dataclass
class DDSessionAnalysis:
    root_q: np.ndarray
    height: np.ndarray
    smoothed_height: np.ndarray
    height_threshold: float
    airborne_regions: list[tuple[int, int]]
    jump_segments: list[JumpSegment]
    jumps: list[DDJumpAnalysis]


def smooth_signal(signal: np.ndarray, window_frames: int) -> np.ndarray:
    window_frames = max(1, int(window_frames))
    if window_frames <= 1:
        return np.asarray(signal, dtype=float).copy()
    if window_frames % 2 == 0:
        window_frames += 1
    kernel = np.ones(window_frames, dtype=float) / float(window_frames)
    padded = np.pad(np.asarray(signal, dtype=float), (window_frames // 2, window_frames // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def relative_height_threshold(height: np.ndarray, ratio: float) -> float:
    min_h = float(np.nanmin(height))
    max_h = float(np.nanmax(height))
    return min_h + float(ratio) * (max_h - min_h)


def contiguous_true_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(np.asarray(mask, dtype=bool)):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            regions.append((start, idx - 1))
            start = None
    if start is not None:
        regions.append((start, len(mask) - 1))
    return regions


def merge_close_regions(regions: list[tuple[int, int]], max_gap_frames: int) -> list[tuple[int, int]]:
    if not regions:
        return []
    merged = [list(regions[0])]
    for start, end in regions[1:]:
        if start - merged[-1][1] - 1 <= max_gap_frames:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def local_minimum_index(height: np.ndarray, center: int, left_limit: int, right_limit: int, window_frames: int) -> int:
    start = max(left_limit, center - window_frames)
    end = min(right_limit, center + window_frames)
    if end <= start:
        return int(center)
    return int(start + np.argmin(height[start : end + 1]))


def refine_jump_boundaries(height: np.ndarray, airborne_regions: list[tuple[int, int]], contact_window_frames: int) -> list[JumpSegment]:
    segments: list[JumpSegment] = []
    if not airborne_regions:
        return segments
    previous_end = 0
    for region_idx, (start, end) in enumerate(airborne_regions):
        next_region_start = airborne_regions[region_idx + 1][0] if region_idx + 1 < len(airborne_regions) else len(height) - 1
        jump_start = local_minimum_index(height, center=start, left_limit=previous_end, right_limit=start, window_frames=contact_window_frames)
        jump_end = local_minimum_index(height, center=end, left_limit=end, right_limit=next_region_start, window_frames=contact_window_frames)
        peak_index = int(jump_start + np.argmax(height[jump_start : jump_end + 1]))
        segments.append(JumpSegment(start=jump_start, end=jump_end, peak_index=peak_index))
        previous_end = jump_end
    return segments


def filter_jump_segments(
    height: np.ndarray,
    segments: list[JumpSegment],
    height_threshold: float,
    min_airtime_frames: int,
    min_peak_prominence_m: float,
) -> list[JumpSegment]:
    kept: list[JumpSegment] = []
    for segment in segments:
        segment_height = height[segment.start : segment.end + 1]
        airborne_frames = int(np.count_nonzero(segment_height > height_threshold))
        if airborne_frames < min_airtime_frames:
            continue
        start_h = float(height[segment.start])
        end_h = float(height[segment.end])
        peak_h = float(height[segment.peak_index])
        prominence = peak_h - max(start_h, end_h)
        if prominence < min_peak_prominence_m:
            continue
        kept.append(segment)
    return kept


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    out = np.full_like(vectors, np.nan)
    valid = np.isfinite(norms[:, 0]) & (norms[:, 0] > 1e-12)
    out[valid] = vectors[valid] / norms[valid]
    return out


def compute_angles_over_jump_from_axes(
    root_q: np.ndarray,
    start: int,
    end: int,
    rotation_sequence: str = TRUNK_ROOT_ROTATION_SEQUENCE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window = np.asarray(root_q[start : end + 1], dtype=float)
    rotations = window[:, 3:6]
    z_axes = np.full((window.shape[0], 3), np.nan, dtype=float)
    y_axes = np.full((window.shape[0], 3), np.nan, dtype=float)
    for idx, angles in enumerate(rotations):
        if not np.all(np.isfinite(angles)):
            continue
        matrix = R.from_euler(rotation_sequence, angles, degrees=False).as_matrix()
        y_axes[idx] = matrix[:, 1]
        z_axes[idx] = matrix[:, 2]

    sagittal_proj = np.array(z_axes, copy=True)
    sagittal_proj[:, 1] = 0.0
    sagittal_proj = _normalize_rows(sagittal_proj)
    somersault_angle = np.full(window.shape[0], np.nan, dtype=float)
    twist_angle = np.full(window.shape[0], np.nan, dtype=float)
    tilt_angle = np.full(window.shape[0], np.nan, dtype=float)

    valid_som = np.all(np.isfinite(sagittal_proj[:, [0, 2]]), axis=1)
    somersault_angle[valid_som] = np.arctan2(sagittal_proj[valid_som, 2], sagittal_proj[valid_som, 0])
    somersault_angle = unwrap_with_gaps(somersault_angle)

    valid_tw = np.all(np.isfinite(y_axes[:, [0, 1]]), axis=1)
    twist_angle[valid_tw] = np.arctan2(y_axes[valid_tw, 0], y_axes[valid_tw, 1])
    twist_angle = unwrap_with_gaps(twist_angle)

    vertical = np.array([0.0, 0.0, 1.0], dtype=float)
    valid_tilt = np.all(np.isfinite(z_axes), axis=1)
    if np.any(valid_tilt):
        dots = np.clip(np.sum(z_axes[valid_tilt] * vertical[np.newaxis, :], axis=1), -1.0, 1.0)
        tilt_angle[valid_tilt] = np.arccos(dots)
    return somersault_angle, twist_angle, tilt_angle


def compute_angles_over_jump_from_euler(
    root_q: np.ndarray,
    start: int,
    end: int,
    rotation_sequence: str = TRUNK_ROOT_ROTATION_SEQUENCE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window = np.asarray(root_q[start : end + 1], dtype=float)
    raw_rotations = np.asarray(window[:, 3:6], dtype=float)
    rotations = reextract_euler_with_gaps(raw_rotations, rotation_sequence)
    rotations = unwrap_with_gaps(rotations)
    somersault_angle = rotations[:, 0]
    tilt_angle = rotations[:, 1]
    twist_angle = rotations[:, 2]
    return somersault_angle, twist_angle, tilt_angle


def compute_angles_over_jump(
    root_q: np.ndarray,
    start: int,
    end: int,
    rotation_sequence: str = TRUNK_ROOT_ROTATION_SEQUENCE,
    angle_mode: str = "euler",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if angle_mode == "body_axes":
        return compute_angles_over_jump_from_axes(root_q, start, end, rotation_sequence=rotation_sequence)
    return compute_angles_over_jump_from_euler(root_q, start, end, rotation_sequence=rotation_sequence)


def round_to_nearest(value: float, step: float) -> float:
    return round(float(value) / float(step)) * float(step)


def crossing_index(cumulative_turns: np.ndarray, target: float) -> int | None:
    hits = np.where(np.asarray(cumulative_turns, dtype=float) >= float(target))[0]
    return int(hits[0]) if hits.size else None


def signed_threshold_crossing_indices(cumulative_turns: np.ndarray, step: float) -> list[int]:
    curve = np.asarray(cumulative_turns, dtype=float)
    valid = np.flatnonzero(np.isfinite(curve))
    if valid.size == 0:
        return []
    final_value = float(curve[valid[-1]])
    direction = 1.0 if final_value >= 0.0 else -1.0
    target_count = int(np.floor(abs(final_value) / float(step) + 1e-9))
    if target_count <= 0:
        return []
    directed_curve = direction * curve
    indices: list[int] = []
    previous_idx = -1
    for target_idx in range(1, target_count + 1):
        target = float(target_idx) * float(step)
        crossing = np.where(directed_curve >= target)[0]
        if crossing.size == 0:
            continue
        index = int(crossing[0])
        if index != previous_idx:
            indices.append(index)
            previous_idx = index
    return indices


def twists_per_completed_salto(
    cumulative_salto_turns: np.ndarray,
    cumulative_twist_turns: np.ndarray,
) -> tuple[list[float], list[int]]:
    twists_per_salto: list[float] = []
    full_salto_event_indices = signed_threshold_crossing_indices(cumulative_salto_turns, 1.0)
    segment_start = 0
    for end_idx in full_salto_event_indices:
        if end_idx <= segment_start:
            continue
        twist_delta = cumulative_twist_turns[end_idx] - cumulative_twist_turns[segment_start]
        twists_per_salto.append(abs(round_to_nearest(twist_delta, 0.5)))
        segment_start = end_idx
    return twists_per_salto, full_salto_event_indices


def detect_body_shape(
    q_segment: np.ndarray,
    hip_indices: list[int],
    knee_indices: list[int],
    hip_threshold_deg: float = 70.0,
    knee_tuck_threshold_deg: float = 70.0,
    knee_pike_threshold_deg: float = 20.0,
) -> str:
    hip_peak_deg = np.rad2deg(np.nanmax(np.abs(q_segment[:, hip_indices]), axis=0))
    knee_peak_deg = np.rad2deg(np.nanmax(np.abs(q_segment[:, knee_indices]), axis=0))
    hip_value = float(np.nanmax(hip_peak_deg))
    knee_value = float(np.nanmax(knee_peak_deg))
    if hip_value >= hip_threshold_deg and knee_value >= knee_tuck_threshold_deg:
        return "grouped"
    if hip_value >= hip_threshold_deg and knee_value <= knee_pike_threshold_deg:
        return "piked"
    return "straight"


def body_shape_suffix(body_shape: str) -> str:
    return {"grouped": "o", "piked": "<", "straight": "/"}.get(body_shape, "?")


def quarter_salto_count_token(total_saltos: float) -> str:
    quarter_count = int(round(float(total_saltos) * 4.0))
    return str(quarter_count)


def classify_jump(som_turns: float, twist_turns: float) -> str:
    som_round = round(float(som_turns) * 2.0) / 2.0
    tw_round = round(float(twist_turns) * 2.0) / 2.0
    if abs(som_round - 2.0) < 0.25:
        salto = "double"
    elif abs(som_round - 1.0) < 0.25:
        salto = "single"
    elif abs(som_round - 3.0) < 0.25:
        salto = "triple"
    else:
        salto = "unknown"
    direction = "forward" if som_turns > 0 else "backward"
    return f"{salto} {direction}, twist {tw_round:.1f}"


def default_body_shape_indices(q_names: list[str]) -> tuple[list[int], list[int]] | None:
    index_map = {name: idx for idx, name in enumerate(q_names)}
    if any(name not in index_map for name in DEFAULT_HIP_DOFS + DEFAULT_KNEE_DOFS):
        return None
    hip_indices = [index_map[name] for name in DEFAULT_HIP_DOFS]
    knee_indices = [index_map[name] for name in DEFAULT_KNEE_DOFS]
    return hip_indices, knee_indices


def analyze_single_jump(
    root_q: np.ndarray,
    segment: JumpSegment,
    full_q: np.ndarray | None = None,
    q_names: list[str] | None = None,
    angle_mode: str = "euler",
) -> DDJumpAnalysis:
    som, tw, tilt = compute_angles_over_jump(root_q, segment.start, segment.end, angle_mode=angle_mode)
    som_turns = float((som[-1] - som[0]) / (2.0 * np.pi)) if np.count_nonzero(np.isfinite(som)) >= 2 else float("nan")
    twist_turns = float((tw[-1] - tw[0]) / (2.0 * np.pi)) if np.count_nonzero(np.isfinite(tw)) >= 2 else float("nan")
    classification = classify_jump(som_turns, twist_turns) if np.isfinite(som_turns) and np.isfinite(twist_turns) else "unknown"

    total_saltos = abs(round_to_nearest(som_turns, 0.25)) if np.isfinite(som_turns) else float("nan")
    twists_per_salto: list[float] = []
    full_salto_event_indices: list[int] = []
    quarter_salto_event_indices: list[int] = []
    half_twist_event_indices: list[int] = []
    somersault_curve_turns = (som - som[0]) / (2.0 * np.pi) if np.count_nonzero(np.isfinite(som)) else np.full_like(som, np.nan)
    twist_curve_turns = (tw - tw[0]) / (2.0 * np.pi) if np.count_nonzero(np.isfinite(tw)) else np.full_like(tw, np.nan)
    if np.isfinite(som_turns) and np.isfinite(twist_turns):
        quarter_salto_event_indices = signed_threshold_crossing_indices(somersault_curve_turns, 0.25)
        half_twist_event_indices = signed_threshold_crossing_indices(twist_curve_turns, 0.5)
        twists_per_salto, full_salto_event_indices = twists_per_completed_salto(somersault_curve_turns, twist_curve_turns)

    body_shape = None
    code = None
    if full_q is not None and q_names is not None:
        body_indices = default_body_shape_indices(q_names)
        if body_indices is not None:
            hip_indices, knee_indices = body_indices
            q_segment = full_q[segment.start : segment.end + 1]
            body_shape = detect_body_shape(q_segment, hip_indices, knee_indices)
    if np.isfinite(total_saltos):
        twist_tokens = "".join(str(int(round(value * 2.0))) for value in twists_per_salto) if twists_per_salto else "0"
        code = f"{quarter_salto_count_token(total_saltos)}{twist_tokens}"
        if body_shape is not None:
            code = f"{code}{body_shape_suffix(body_shape)}"

    return DDJumpAnalysis(
        segment=segment,
        somersault_turns=som_turns,
        twist_turns=twist_turns,
        max_tilt_rad=float(np.nanmax(tilt)) if np.any(np.isfinite(tilt)) else float("nan"),
        mean_tilt_rad=float(np.nanmean(tilt)) if np.any(np.isfinite(tilt)) else float("nan"),
        classification=classification,
        body_shape=body_shape,
        code=code,
        twists_per_salto=twists_per_salto,
        full_salto_event_indices=full_salto_event_indices,
        quarter_salto_event_indices=quarter_salto_event_indices,
        half_twist_event_indices=half_twist_event_indices,
        somersault_curve_turns=somersault_curve_turns,
        twist_curve_turns=twist_curve_turns,
        tilt_curve_rad=tilt,
        angle_mode=angle_mode,
    )


def analyze_dd_session(
    root_q: np.ndarray,
    fps: float,
    *,
    height_values: np.ndarray | None = None,
    height_threshold: float | None = None,
    height_threshold_range_ratio: float = 0.20,
    smoothing_window_s: float = 0.15,
    min_airtime_s: float = 0.25,
    min_gap_s: float = 0.08,
    min_peak_prominence_m: float = 0.35,
    contact_window_s: float = 0.35,
    full_q: np.ndarray | None = None,
    q_names: list[str] | None = None,
    angle_mode: str = "euler",
) -> DDSessionAnalysis:
    dd_debug(
        "analyze_dd_session start "
        f"frames={len(root_q)} fps={fps:.3f} smooth={smoothing_window_s} "
        f"thr_abs={height_threshold} thr_ratio={height_threshold_range_ratio}"
    )
    if height_values is None:
        height_values = root_q[:, 2]
    height = np.asarray(height_values, dtype=float)
    window_frames = max(1, int(round(float(smoothing_window_s) * float(fps))))
    smoothed_height = smooth_signal(height, window_frames=window_frames)
    effective_threshold = float(height_threshold) if height_threshold is not None else relative_height_threshold(smoothed_height, ratio=height_threshold_range_ratio)
    airborne_mask = smoothed_height > effective_threshold
    airborne_regions = contiguous_true_regions(airborne_mask)
    airborne_regions = merge_close_regions(airborne_regions, max_gap_frames=max(0, int(round(float(min_gap_s) * float(fps)))))
    segments = refine_jump_boundaries(
        smoothed_height,
        airborne_regions,
        contact_window_frames=max(1, int(round(float(contact_window_s) * float(fps)))),
    )
    segments = filter_jump_segments(
        smoothed_height,
        segments,
        height_threshold=effective_threshold,
        min_airtime_frames=max(1, int(round(float(min_airtime_s) * float(fps)))),
        min_peak_prominence_m=float(min_peak_prominence_m),
    )
    dd_debug(
        "analyze_dd_session segmentation "
        f"airborne_regions={len(airborne_regions)} jumps={len(segments)} "
        f"threshold={effective_threshold:.4f}"
    )
    q_name_list = list(q_names) if q_names is not None else None
    jumps = [
        analyze_single_jump(root_q, segment, full_q=full_q, q_names=q_name_list, angle_mode=angle_mode)
        for segment in segments
    ]
    dd_debug(f"analyze_dd_session done jumps={len(jumps)}")
    return DDSessionAnalysis(
        root_q=np.asarray(root_q, dtype=float),
        height=height,
        smoothed_height=smoothed_height,
        height_threshold=effective_threshold,
        airborne_regions=airborne_regions,
        jump_segments=segments,
        jumps=jumps,
    )
