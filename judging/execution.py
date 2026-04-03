#!/usr/bin/env python3
"""Execution-deduction helpers used by the GUI execution-analysis tab.

The current implementation keeps a pragmatic scope:
- it reuses the DD jump segmentation to isolate complete elements;
- it computes a small set of discrete FIG-like deductions;
- it localizes each deduction to one representative frame and body region so
  the GUI can highlight the error in 3D and 2D.

This module is intentionally conservative. The resulting deductions are meant
to support inspection and iterative development of the judging workflow, not
to claim full FIG compliance yet.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

from judging.body_geometry import arm_raise_series, hip_angle_series, knee_angle_series
from judging.dd_analysis import DDSessionAnalysis, JumpSegment
from vitpose_ekf_pipeline import KP_INDEX

DEG10 = np.deg2rad(10.0)
DEG20 = np.deg2rad(20.0)
DEG45 = np.deg2rad(45.0)
DEG90 = np.deg2rad(90.0)
DEG135 = np.deg2rad(135.0)
DEG170 = np.deg2rad(170.0)

EXECUTION_HIP_DOF_NAMES = ("LEFT_THIGH:RotY", "RIGHT_THIGH:RotY")
EXECUTION_KNEE_DOF_NAMES = ("LEFT_SHANK:RotY", "RIGHT_SHANK:RotY")
ROOT_TILT_DOF_NAME = "TRUNK:RotX"
ROOT_TRANSLATION_VELOCITY_NAMES = ("TRUNK:TransX", "TRUNK:TransY", "TRUNK:TransZ")
SUPPORTED_OVERLAY_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_EXECUTION_IMAGE_INDEX_CACHE: dict[str, tuple[float, dict[tuple[str, int], Path], dict[str, set[int]]]] = {}


@dataclass
class ExecutionDeductionEvent:
    """One localized deduction ready for display in the GUI."""

    code: str
    label: str
    deduction: float
    frame_idx: int
    local_frame_idx: int
    metric_value: float
    metric_unit: str
    detail: str
    keypoint_names: tuple[str, ...]


@dataclass
class ExecutionJumpAnalysis:
    """Per-jump execution deductions and the local metrics used to derive them."""

    jump_index: int
    segment: JumpSegment
    classification: str
    total_deduction: float
    capped_deduction: float
    event_frame_idx: int
    deduction_events: list[ExecutionDeductionEvent]
    metric_time_s: np.ndarray
    metric_series: dict[str, np.ndarray]


@dataclass
class ExecutionSessionAnalysis:
    """Execution deductions over a full trial."""

    jumps: list[ExecutionJumpAnalysis]
    total_deduction: float
    execution_score: float
    time_of_flight_s: float


@dataclass
class ExecutionOverlayFrame:
    """All display data required for one execution 2D overlay frame."""

    camera_name: str
    frame_idx: int
    frame_number: int
    image_root: Path | None
    image_path: Path | None
    raw_points_2d: np.ndarray
    projected_points_2d: np.ndarray
    keypoint_names: tuple[str, ...]


def lowpass_filter(signal: np.ndarray, fs: float, fc: float = 10.0) -> np.ndarray:
    """Apply a simple low-pass filter to mimic judge perception."""

    signal = np.asarray(signal, dtype=float)
    if signal.shape[-1] < 4 or fs <= 0.0:
        return np.array(signal, copy=True)
    normalized_cutoff = float(fc) / (0.5 * float(fs))
    if not np.isfinite(normalized_cutoff) or normalized_cutoff <= 0.0:
        return np.array(signal, copy=True)
    normalized_cutoff = min(normalized_cutoff, 0.99)
    b, a = butter(2, normalized_cutoff, btype="low")
    min_length = 3 * max(len(a), len(b))
    if signal.shape[-1] <= min_length:
        return np.array(signal, copy=True)
    return filtfilt(b, a, signal, axis=-1)


def quantize_fig(value: float, thresholds: list[tuple[float, float]]) -> float:
    """Convert one continuous metric into a discrete FIG deduction."""

    for threshold, deduction in thresholds:
        if value < threshold:
            return deduction
    return thresholds[-1][1]


def deduction_form_discrete(knee_angle: np.ndarray, hip_angle: np.ndarray) -> float:
    """Compute the discrete body-form deduction from knee and hip angles."""

    deduction = 0.0
    knee_error = np.mean(np.abs(np.pi - np.asarray(knee_angle, dtype=float)))
    if knee_error > DEG10:
        deduction += 0.1
    if knee_error > DEG20:
        deduction += 0.1

    hip_error = np.mean(np.abs(np.pi - np.asarray(hip_angle, dtype=float)))
    if hip_error > DEG10:
        deduction += 0.1

    return min(0.5, deduction)


def deduction_opening_discrete(hip_angle: np.ndarray, time: np.ndarray) -> float:
    """Compute the opening deduction from the timing of the return to straight.

    This mirrors the FIG late-opening ladder as closely as possible with the
    information we currently have:
    - before roughly 60% of the element: no deduction;
    - between ~60% and ~75%: 0.1;
    - between ~75% and ~90%: 0.2;
    - later or never clearly straight: 0.3.
    """

    error = np.abs(np.asarray(hip_angle, dtype=float) - np.pi)
    idx_open = int(np.nanargmin(error)) if error.size else 0
    ratio = 0.0 if len(time) == 0 else float(idx_open) / float(len(time))
    if ratio < 0.6:
        return 0.0
    if ratio < 0.75:
        return 0.1
    if ratio < 0.9:
        return 0.2
    return 0.3


def deduction_pike_down_discrete(hip_angle: np.ndarray) -> float:
    """Compute the pike-down / kick-out deduction from the minimum hip angle."""

    min_angle = float(np.nanmin(np.asarray(hip_angle, dtype=float)))
    if min_angle > DEG170:
        return 0.0
    if min_angle > DEG135:
        return 0.1
    return 0.2


def deduction_arms_discrete(arm_angle: np.ndarray) -> float:
    """Compute the arm-position deduction."""

    if np.nanmax(np.asarray(arm_angle, dtype=float)) > DEG90:
        return 0.1
    return 0.0


def deduction_axis_discrete(root_tilt: np.ndarray) -> float:
    """Compute the axis-control deduction from tilt variability."""

    root_tilt = np.asarray(root_tilt, dtype=float)
    if not np.any(np.isfinite(root_tilt)):
        return 0.0
    variance = float(np.nanvar(root_tilt))
    return quantize_fig(
        variance,
        [
            (0.01, 0.0),
            (0.03, 0.1),
            (0.06, 0.2),
            (0.1, 0.3),
            (np.inf, 0.5),
        ],
    )


def deduction_landing_discrete(root_velocity_norm: float) -> float:
    """Compute the landing deduction from root translational speed."""

    velocity = float(root_velocity_norm)
    if velocity < 0.5:
        return 0.0
    if velocity < 1.0:
        return 0.1
    if velocity < 2.0:
        return 0.2
    return 0.3


def deduction_fall(contact_type: str) -> float:
    """Return the extra deduction associated with a fall or invalid landing."""

    if contact_type == "hands":
        return 0.5
    if contact_type in {"knees", "out"}:
        return 1.0
    return 0.0


def _safe_name_to_index(q_names: list[str] | np.ndarray) -> dict[str, int]:
    """Build a robust DoF lookup dictionary from q-name labels."""

    return {str(name): idx for idx, name in enumerate(np.asarray(q_names, dtype=object))}


def _optional_dof_indices(q_names: list[str] | np.ndarray, names: tuple[str, ...]) -> list[int]:
    """Return the subset of DoF indices present in the trajectory."""

    name_to_index = _safe_name_to_index(q_names)
    return [name_to_index[name] for name in names if name in name_to_index]


def root_tilt_series(q_series: np.ndarray, q_names: list[str] | np.ndarray) -> np.ndarray:
    """Extract the root tilt DoF from the generalized coordinates."""

    q_series = np.asarray(q_series, dtype=float)
    name_to_index = _safe_name_to_index(q_names)
    tilt_idx = name_to_index.get(ROOT_TILT_DOF_NAME)
    if tilt_idx is None:
        return np.full(q_series.shape[0], np.nan, dtype=float)
    return q_series[:, tilt_idx]


def root_translation_velocity_series(qdot_series: np.ndarray, q_names: list[str] | np.ndarray) -> np.ndarray:
    """Extract the root translational velocity norm from qdot."""

    qdot_series = np.asarray(qdot_series, dtype=float)
    translation_indices = _optional_dof_indices(q_names, ROOT_TRANSLATION_VELOCITY_NAMES)
    if len(translation_indices) != len(ROOT_TRANSLATION_VELOCITY_NAMES):
        return np.full(qdot_series.shape[0], np.nan, dtype=float)
    velocity_xyz = qdot_series[:, translation_indices]
    return np.linalg.norm(velocity_xyz, axis=1)


def root_vertical_translation_series(q_series: np.ndarray, q_names: list[str] | np.ndarray) -> np.ndarray:
    """Extract the root vertical translation used for time-of-flight scoring."""

    q_series = np.asarray(q_series, dtype=float)
    name_to_index = _safe_name_to_index(q_names)
    translation_z_idx = name_to_index.get("TRUNK:TransZ")
    if translation_z_idx is None:
        return np.full(q_series.shape[0], np.nan, dtype=float)
    return q_series[:, translation_z_idx]


def detect_contacts_velocity(Tz: np.ndarray, time: np.ndarray) -> list[int]:
    """Detect likely trampoline-contact instants from the vertical root trajectory.

    The heuristic looks for local minima where vertical velocity changes from
    descending to ascending. This keeps the implementation lightweight while
    staying robust enough for session-level time-of-flight summaries.
    """

    Tz = np.asarray(Tz, dtype=float)
    time = np.asarray(time, dtype=float)
    if Tz.ndim != 1 or time.ndim != 1 or Tz.shape[0] != time.shape[0] or Tz.shape[0] < 3:
        return []

    valid = np.isfinite(Tz) & np.isfinite(time)
    if np.count_nonzero(valid) < 3:
        return []

    valid_indices = np.flatnonzero(valid)
    tz_valid = Tz[valid]
    time_valid = time[valid]
    contact_indices: list[int] = []
    if tz_valid[0] <= tz_valid[1]:
        contact_indices.append(int(valid_indices[0]))

    delta_tz = np.diff(tz_valid)
    minima_mask = (delta_tz[:-1] < 0.0) & (delta_tz[1:] >= 0.0)
    for local_min_idx in np.flatnonzero(minima_mask) + 1:
        contact_indices.append(int(valid_indices[local_min_idx]))

    if tz_valid[-1] <= tz_valid[-2]:
        contact_indices.append(int(valid_indices[-1]))

    if not contact_indices:
        return []

    deduplicated: list[int] = [contact_indices[0]]
    for frame_idx in contact_indices[1:]:
        if frame_idx != deduplicated[-1]:
            deduplicated.append(frame_idx)
    return deduplicated


def compute_time_of_flight_robust(Tz: np.ndarray, time: np.ndarray) -> float:
    """Return the summed time between robustly detected contact instants.

    Small gaps are ignored to avoid counting micro-bounces as full elements.
    """

    contacts = detect_contacts_velocity(Tz, time)
    if len(contacts) < 2:
        return 0.0

    total = 0.0
    for contact_idx, next_contact_idx in zip(contacts[:-1], contacts[1:]):
        dt = float(time[next_contact_idx] - time[contact_idx])
        if dt > 0.2:
            total += dt
    return total


def _normalize_overlay_token(value: str) -> str:
    """Normalize camera and dataset tokens for loose filesystem matching."""

    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _extract_overlay_frame_number(path: Path) -> int | None:
    """Extract one frame number from a pragmatic camera-image filename."""

    stem = str(path.stem)
    patterns = (
        r"(?:^|[_-])frame[_-]?(\d+)(?:$|[_-])",
        r"(?:^|[_-])(\d{4,})(?:$|[_-])",
    )
    for pattern in patterns:
        match = re.search(pattern, stem, flags=re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
    return None


def _overlay_camera_tokens(path: Path, *, directory_token: str | None = None) -> set[str]:
    tokens: set[str] = set()
    if directory_token:
        tokens.add(directory_token)
    for match in re.findall(r"M\d{4,}", path.stem, flags=re.IGNORECASE):
        tokens.add(_normalize_overlay_token(match))
    return tokens


def _execution_image_index(
    images_root: Path | str | None,
) -> tuple[dict[tuple[str, int], Path], dict[str, set[int]]]:
    """Index execution images once per root to avoid repeated directory scans."""

    if images_root is None:
        return {}, {}
    root = Path(images_root)
    if not root.exists() or not root.is_dir():
        return {}, {}

    cache_key = str(root.resolve())
    try:
        root_mtime = float(root.stat().st_mtime_ns)
    except OSError:
        root_mtime = -1.0
    cached = _EXECUTION_IMAGE_INDEX_CACHE.get(cache_key)
    if cached is not None and cached[0] == root_mtime:
        return cached[1], cached[2]

    paths_by_camera_frame: dict[tuple[str, int], Path] = {}
    frames_by_camera: dict[str, set[int]] = {}

    candidate_directories: list[tuple[Path, str | None]] = [(root, None)]
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if entry.is_dir():
                    candidate_directories.append((Path(entry.path), _normalize_overlay_token(entry.name)))
    except OSError:
        pass

    for directory, directory_token in candidate_directories:
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not entry.is_file():
                        continue
                    path = Path(entry.path)
                    if path.suffix.lower() not in SUPPORTED_OVERLAY_IMAGE_EXTENSIONS:
                        continue
                    frame_number = _extract_overlay_frame_number(path)
                    if frame_number is None:
                        continue
                    for token in _overlay_camera_tokens(path, directory_token=directory_token):
                        frames_by_camera.setdefault(token, set()).add(frame_number)
                        paths_by_camera_frame.setdefault((token, frame_number), path)
        except OSError:
            continue

    _EXECUTION_IMAGE_INDEX_CACHE[cache_key] = (root_mtime, paths_by_camera_frame, frames_by_camera)
    return paths_by_camera_frame, frames_by_camera


def available_execution_image_frames(
    images_root: Path | str | None,
    camera_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, set[int]]:
    """Return available frame ids per camera without rescanning the disk repeatedly."""

    _paths_by_camera_frame, frames_by_camera = _execution_image_index(images_root)
    if camera_names is None:
        return {camera: set(frames) for camera, frames in frames_by_camera.items()}
    available: dict[str, set[int]] = {}
    for camera_name in camera_names:
        token = _normalize_overlay_token(str(camera_name))
        available[str(camera_name)] = set(frames_by_camera.get(token, set()))
    return available


def infer_execution_images_root(keypoints_path: Path | str | None) -> Path | None:
    """Infer the most likely image root for a keypoint file.

    The GUI does not expose image selection yet, so this helper looks for a few
    pragmatic directory conventions around the keypoint JSON.
    """

    if keypoints_path is None:
        return None
    keypoints_path = Path(keypoints_path)
    dataset_stem = keypoints_path.stem
    if dataset_stem.endswith("_keypoints"):
        dataset_stem = dataset_stem[: -len("_keypoints")]
    candidates = [
        keypoints_path.parent.parent / "images" / dataset_stem,
        keypoints_path.parent.parent / "frames" / dataset_stem,
        keypoints_path.parent / dataset_stem,
        keypoints_path.parent.parent / "images",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def resolve_execution_image_path(images_root: Path | str | None, camera_name: str, frame_number: int) -> Path | None:
    """Resolve one image path from a root, camera name, and frame number."""

    if images_root is None:
        return None
    images_root = Path(images_root)
    if not images_root.exists():
        return None

    frame_number = int(frame_number)
    camera_name = str(camera_name)
    normalized_camera = _normalize_overlay_token(camera_name)
    indexed_paths, _frames_by_camera = _execution_image_index(images_root)
    indexed = indexed_paths.get((normalized_camera, frame_number))
    if indexed is not None and indexed.exists():
        return indexed

    frame_tokens = (
        f"{frame_number}",
        f"{frame_number:04d}",
        f"{frame_number:05d}",
        f"{frame_number:06d}",
        f"frame_{frame_number}",
        f"frame_{frame_number:04d}",
        f"frame_{frame_number:05d}",
        f"frame_{frame_number:06d}",
    )
    camera_tokens = {camera_name, camera_name.lower(), normalized_camera}

    candidate_dirs = [images_root]
    for child in images_root.iterdir():
        if child.is_dir() and _normalize_overlay_token(child.name) in camera_tokens:
            candidate_dirs.insert(0, child)

    for directory in candidate_dirs:
        for frame_token in frame_tokens:
            for extension in SUPPORTED_OVERLAY_IMAGE_EXTENSIONS:
                direct_path = directory / f"{frame_token}{extension}"
                if direct_path.exists():
                    return direct_path
                camera_prefixed = directory / f"{camera_name}_{frame_token}{extension}"
                if camera_prefixed.exists():
                    return camera_prefixed
        for candidate in directory.iterdir():
            if not candidate.is_file() or candidate.suffix.lower() not in SUPPORTED_OVERLAY_IMAGE_EXTENSIONS:
                continue
            stem_normalized = _normalize_overlay_token(candidate.stem)
            if normalized_camera not in stem_normalized:
                continue
            candidate_frame = _extract_overlay_frame_number(candidate)
            if candidate_frame != frame_number:
                continue
            return candidate
    return None


def build_execution_overlay_frame(
    *,
    camera_name: str,
    frame_idx: int,
    frame_number: int,
    frame_points_3d: np.ndarray,
    calibrations: dict[str, object] | None,
    pose_data,
    keypoint_names: tuple[str, ...] = (),
    images_root: Path | None = None,
) -> ExecutionOverlayFrame:
    """Build the 2D overlay payload used by the execution-inspection tab."""

    frame_points_3d = np.asarray(frame_points_3d, dtype=float)
    projected_points = np.full((len(KP_INDEX), 2), np.nan, dtype=float)
    raw_points = np.full((len(KP_INDEX), 2), np.nan, dtype=float)

    if isinstance(calibrations, dict) and camera_name in calibrations:
        calibration = calibrations[camera_name]
        for point_idx, point in enumerate(frame_points_3d):
            if np.all(np.isfinite(point)):
                projected_points[point_idx] = calibration.project_point(point)

    if pose_data is not None and getattr(pose_data, "camera_names", None) is not None:
        camera_names = [str(name) for name in pose_data.camera_names]
        if camera_name in camera_names:
            cam_idx = camera_names.index(camera_name)
            pose_frames = np.asarray(pose_data.frames, dtype=int)
            pose_idx = np.where(pose_frames == int(frame_number))[0]
            if pose_idx.size:
                raw_points = np.asarray(pose_data.keypoints[cam_idx, int(pose_idx[0])], dtype=float)

    image_path = resolve_execution_image_path(images_root, camera_name, frame_number)
    return ExecutionOverlayFrame(
        camera_name=str(camera_name),
        frame_idx=int(frame_idx),
        frame_number=int(frame_number),
        image_root=None if images_root is None else Path(images_root),
        image_path=image_path,
        raw_points_2d=raw_points,
        projected_points_2d=projected_points,
        keypoint_names=tuple(str(name) for name in keypoint_names),
    )


def execution_focus_frame(jump: ExecutionJumpAnalysis) -> int:
    """Return the frame that should drive the 3D/2D execution preview."""

    if jump.deduction_events:
        return int(max(jump.deduction_events, key=lambda event: event.deduction).frame_idx)
    return int(jump.event_frame_idx)


def _event(
    *,
    code: str,
    label: str,
    deduction: float,
    segment: JumpSegment,
    local_frame_idx: int,
    metric_value: float,
    metric_unit: str,
    detail: str,
    keypoint_names: tuple[str, ...],
) -> ExecutionDeductionEvent:
    """Build one localized deduction event."""

    local_frame_idx = int(np.clip(local_frame_idx, 0, max(segment.end - segment.start, 0)))
    return ExecutionDeductionEvent(
        code=code,
        label=label,
        deduction=float(deduction),
        frame_idx=int(segment.start + local_frame_idx),
        local_frame_idx=local_frame_idx,
        metric_value=float(metric_value),
        metric_unit=str(metric_unit),
        detail=str(detail),
        keypoint_names=tuple(str(name) for name in keypoint_names),
    )


def compute_execution_jump_analysis(
    *,
    jump_index: int,
    segment: JumpSegment,
    classification: str,
    q_series: np.ndarray,
    qdot_series: np.ndarray | None,
    q_names: list[str] | np.ndarray,
    points_3d: np.ndarray,
    fs: float,
) -> ExecutionJumpAnalysis:
    """Compute localized execution deductions for one jump segment."""

    q_slice = np.asarray(q_series[segment.start : segment.end + 1], dtype=float)
    points_slice = np.asarray(points_3d[segment.start : segment.end + 1], dtype=float)
    time_local = np.arange(q_slice.shape[0], dtype=float) / max(float(fs), 1.0)
    q_filtered = lowpass_filter(q_slice.T, fs).T if q_slice.size else q_slice
    points_filtered = lowpass_filter(np.moveaxis(points_slice, 0, -1), fs)
    points_filtered = np.moveaxis(points_filtered, -1, 0)

    knee_series_rad = knee_angle_series(points_filtered)
    hip_series_rad = hip_angle_series(points_filtered)
    arm_series_rad = arm_raise_series(points_filtered)
    tilt_series_rad = root_tilt_series(q_filtered, q_names)
    landing_speed_series = (
        root_translation_velocity_series(np.asarray(qdot_series, dtype=float), q_names)
        if qdot_series is not None and np.asarray(qdot_series).size
        else np.full(q_series.shape[0], np.nan, dtype=float)
    )
    landing_speed_local = landing_speed_series[segment.start : segment.end + 1]

    metric_series = {
        "knee_error_deg": np.rad2deg(np.abs(np.pi - knee_series_rad)),
        "hip_error_deg": np.rad2deg(np.abs(np.pi - hip_series_rad)),
        "hip_angle_deg": np.rad2deg(hip_series_rad),
        "arm_raise_deg": np.rad2deg(arm_series_rad),
        "tilt_deg": np.rad2deg(tilt_series_rad),
        "landing_speed_mps": landing_speed_local,
    }

    events: list[ExecutionDeductionEvent] = []

    knee_error_series = np.abs(np.pi - knee_series_rad)
    hip_error_series = np.abs(np.pi - hip_series_rad)
    mean_knee_error = float(np.nanmean(knee_error_series))
    mean_hip_error = float(np.nanmean(hip_error_series))
    if np.isfinite(mean_knee_error) and mean_knee_error > DEG10:
        local_idx = int(np.nanargmax(knee_error_series))
        deduction = 0.2 if mean_knee_error > DEG20 else 0.1
        events.append(
            _event(
                code="form_knees",
                label="Leg form",
                deduction=deduction,
                segment=segment,
                local_frame_idx=local_idx,
                metric_value=np.rad2deg(knee_error_series[local_idx]),
                metric_unit="deg",
                detail="Knee extension error exceeds the FIG form threshold.",
                keypoint_names=("left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle"),
            )
        )
    if np.isfinite(mean_hip_error) and mean_hip_error > DEG10:
        local_idx = int(np.nanargmax(hip_error_series))
        deduction = 0.2 if mean_hip_error > DEG20 else 0.1
        events.append(
            _event(
                code="form_hips",
                label="Body straightness",
                deduction=deduction,
                segment=segment,
                local_frame_idx=local_idx,
                metric_value=np.rad2deg(hip_error_series[local_idx]),
                metric_unit="deg",
                detail="Hip angle departs from a straight body line (FIG-style body straightness deduction).",
                keypoint_names=("left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee"),
            )
        )

    if hip_series_rad.size:
        opening_idx = int(np.nanargmin(hip_error_series))
        opening_ratio = float(opening_idx) / float(max(len(time_local), 1))
        opening_deduction = deduction_opening_discrete(hip_series_rad, time_local)
        if opening_deduction > 0.0:
            events.append(
                _event(
                    code="opening",
                    label="Late opening",
                    deduction=opening_deduction,
                    segment=segment,
                    local_frame_idx=opening_idx,
                    metric_value=opening_ratio,
                    metric_unit="ratio",
                    detail="The return to a straight body happens late in the element (late opening).",
                    keypoint_names=(
                        "left_shoulder",
                        "right_shoulder",
                        "left_hip",
                        "right_hip",
                        "left_knee",
                        "right_knee",
                    ),
                )
            )

        pike_deduction = deduction_pike_down_discrete(hip_series_rad)
        if pike_deduction > 0.0:
            pike_idx = int(np.nanargmin(hip_series_rad))
            events.append(
                _event(
                    code="pike_down",
                    label="Pike down / kick-out",
                    deduction=pike_deduction,
                    segment=segment,
                    local_frame_idx=pike_idx,
                    metric_value=np.rad2deg(hip_series_rad[pike_idx]),
                    metric_unit="deg",
                    detail="The hip angle reaches a visible pike-down / kick-out configuration.",
                    keypoint_names=(
                        "left_shoulder",
                        "right_shoulder",
                        "left_hip",
                        "right_hip",
                        "left_knee",
                        "right_knee",
                    ),
                )
            )

    arm_deduction = deduction_arms_discrete(arm_series_rad)
    if arm_deduction > 0.0 and arm_series_rad.size and np.any(np.isfinite(arm_series_rad)):
        arm_idx = int(np.nanargmax(arm_series_rad))
        events.append(
            _event(
                code="arms",
                label="Arm position",
                deduction=arm_deduction,
                segment=segment,
                local_frame_idx=arm_idx,
                metric_value=np.rad2deg(arm_series_rad[arm_idx]),
                metric_unit="deg",
                detail="Arms move visibly away from the trunk line (arm-position deduction).",
                keypoint_names=(
                    "left_shoulder",
                    "left_elbow",
                    "left_wrist",
                    "right_shoulder",
                    "right_elbow",
                    "right_wrist",
                ),
            )
        )

    axis_deduction = deduction_axis_discrete(tilt_series_rad)
    if axis_deduction > 0.0 and tilt_series_rad.size and np.any(np.isfinite(tilt_series_rad)):
        tilt_idx = int(np.nanargmax(np.abs(tilt_series_rad)))
        events.append(
            _event(
                code="axis",
                label="Axis control",
                deduction=axis_deduction,
                segment=segment,
                local_frame_idx=tilt_idx,
                metric_value=np.rad2deg(np.abs(tilt_series_rad[tilt_idx])),
                metric_unit="deg",
                detail="Root tilt varies enough to trigger an axis-control deduction.",
                keypoint_names=("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
            )
        )

    if landing_speed_local.size:
        landing_local_idx = landing_speed_local.shape[0] - 1
        landing_speed = landing_speed_local[landing_local_idx]
        landing_deduction = deduction_landing_discrete(landing_speed)
        if landing_deduction > 0.0:
            events.append(
                _event(
                    code="landing",
                    label="Landing control",
                    deduction=landing_deduction,
                    segment=segment,
                    local_frame_idx=landing_local_idx,
                    metric_value=landing_speed,
                    metric_unit="m/s",
                    detail="Root translational speed remains high at landing.",
                    keypoint_names=("left_ankle", "right_ankle", "left_hip", "right_hip"),
                )
            )

    total_deduction = float(sum(event.deduction for event in events))
    capped_deduction = float(min(0.5, total_deduction))
    focus_local_idx = int(segment.peak_index - segment.start)
    if events:
        focus_local_idx = int(max(events, key=lambda event: event.deduction).local_frame_idx)
    return ExecutionJumpAnalysis(
        jump_index=int(jump_index),
        segment=segment,
        classification=str(classification),
        total_deduction=total_deduction,
        capped_deduction=capped_deduction,
        event_frame_idx=int(segment.start + focus_local_idx),
        deduction_events=events,
        metric_time_s=time_local,
        metric_series=metric_series,
    )


def analyze_execution_session(
    dd_session: DDSessionAnalysis,
    q_series: np.ndarray,
    qdot_series: np.ndarray | None,
    q_names: list[str] | np.ndarray,
    points_3d: np.ndarray,
    fs: float,
) -> ExecutionSessionAnalysis:
    """Compute execution deductions over all complete jumps in one trial."""

    q_series = np.asarray(q_series, dtype=float)
    points_3d = np.asarray(points_3d, dtype=float)
    qdot_series = None if qdot_series is None else np.asarray(qdot_series, dtype=float)
    jumps: list[ExecutionJumpAnalysis] = []
    for jump_index, segment in enumerate(dd_session.jump_segments, start=1):
        classification = (
            dd_session.jumps[jump_index - 1].classification if jump_index - 1 < len(dd_session.jumps) else "-"
        )
        jumps.append(
            compute_execution_jump_analysis(
                jump_index=jump_index,
                segment=segment,
                classification=classification,
                q_series=q_series,
                qdot_series=qdot_series,
                q_names=q_names,
                points_3d=points_3d,
                fs=fs,
            )
        )
    total_deduction = float(sum(jump.capped_deduction for jump in jumps))
    time = np.arange(q_series.shape[0], dtype=float) / max(float(fs), 1.0)
    root_tz = root_vertical_translation_series(q_series, q_names)
    return ExecutionSessionAnalysis(
        jumps=jumps,
        total_deduction=total_deduction,
        execution_score=max(0.0, 20.0 - total_deduction),
        time_of_flight_s=compute_time_of_flight_robust(root_tz, time),
    )
