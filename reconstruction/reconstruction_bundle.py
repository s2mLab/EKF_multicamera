#!/usr/bin/env python3
"""Generation de bundles standardises pour les reconstructions cinematiques."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from kinematics.root_kinematics import (
    ROOT_Q_NAMES,
    ROOT_ROTATION_SLICE,
    TRUNK_ROOT_ROTATION_SCIPY_SEQUENCE,
    TRUNK_ROOT_ROTATION_SEQUENCE,
    build_root_rotation_matrices,
    centered_finite_difference,
    extract_root_from_q,
    normalize,
    normalize_root_unwrap_mode,
    root_z_correction_angle_from_points,
    stabilize_root_rotations,
    unwrap_with_gaps,
)
from reconstruction.reconstruction_dataset import load_trc_root_kinematics_sidecar
from reconstruction.reconstruction_registry import ALGORITHM_VERSIONS, BUNDLE_SCHEMA_VERSION, latest_version_for_family
from reconstruction.reconstruction_timings import make_timing_stage
from vitpose_ekf_pipeline import (
    COCO17,
    DEFAULT_ANKLE_BED_PSEUDO_STD_M,
    DEFAULT_BIORBD_KALMAN_ERROR_FACTOR,
    DEFAULT_BIORBD_KALMAN_INIT_METHOD,
    DEFAULT_BIORBD_KALMAN_NOISE_FACTOR,
    DEFAULT_CALIB,
    DEFAULT_CAMERA_FPS,
    DEFAULT_COHERENCE_CONFIDENCE_FLOOR,
    DEFAULT_COHERENCE_METHOD,
    DEFAULT_EPIPOLAR_THRESHOLD_PX,
    DEFAULT_FLIGHT_HEIGHT_THRESHOLD_M,
    DEFAULT_FLIGHT_MIN_CONSECUTIVE_FRAMES,
    DEFAULT_FLIP_IMPROVEMENT_RATIO,
    DEFAULT_FLIP_MIN_GAIN_PX,
    DEFAULT_FLIP_MIN_OTHER_CAMERAS,
    DEFAULT_FLIP_OUTLIER_FLOOR_PX,
    DEFAULT_FLIP_OUTLIER_PERCENTILE,
    DEFAULT_FLIP_RESTRICT_TO_OUTLIERS,
    DEFAULT_FLIP_TEMPORAL_MIN_VALID_KEYPOINTS,
    DEFAULT_FLIP_TEMPORAL_TAU_PX,
    DEFAULT_FLIP_TEMPORAL_WEIGHT,
    DEFAULT_KEYPOINTS,
    DEFAULT_MEASUREMENT_NOISE_SCALE,
    DEFAULT_MIN_CAMERAS_FOR_TRIANGULATION,
    DEFAULT_MIN_FRAME_COHERENCE_FOR_UPDATE,
    DEFAULT_REPROJECTION_THRESHOLD_PX,
    DEFAULT_SUBJECT_MASS_KG,
    DEFAULT_TRIANGULATION_METHOD,
    DEFAULT_TRIANGULATION_WORKERS,
    DEFAULT_UPPER_BACK_PSEUDO_STD_RAD,
    DEFAULT_UPPER_BACK_SAGITTAL_GAIN,
    KP_INDEX,
    CameraCalibration,
    PoseData,
    ReconstructionResult,
    SegmentLengths,
    apply_left_right_flip_corrections,
    apply_left_right_flip_to_points,
    build_biomod,
    canonical_coherence_method,
    canonical_triangulation_method,
    compute_ekf2d_initial_state,
    compute_epipolar_coherence,
    detect_left_right_flip_diagnostics,
    estimate_segment_lengths,
    frame_signature,
    fundamental_matrix,
    load_calibrations,
    load_model_stage,
    load_pose_data,
    load_reconstruction_cache,
    marker_name_list,
    metadata_cache_matches,
    model_stage_metadata,
    pose_data_signature,
    reconstruction_cache_metadata,
    run_biorbd_marker_kalman_with_parameters,
    run_ekf,
    save_model_stage,
    save_reconstruction_cache,
    save_single_ekf_state,
    select_active_coherence,
    support_coherence_method_for_runtime,
    triangulate_pose2sim_like,
    triangulation_method_from_coherence_method,
)

DEFAULT_POSE2SIM_TRC = Path("inputs/trc/1_partie_0429.trc")
SUPPORTED_EKF2D_3D_SOURCE_MODES = ("full_triangulation", "first_frame_only")


@dataclass
class BundleBuildResult:
    payload: dict[str, np.ndarray]
    summary: dict[str, object]


def _jsonable_metadata(metadata: dict[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(metadata, sort_keys=True))


def cache_key(metadata: dict[str, object]) -> str:
    canonical = json.dumps(_jsonable_metadata(metadata), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


def dataset_cache_root(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    if output_dir.name in {"models", "reconstructions", "figures"}:
        dataset_dir = output_dir.parent
    elif output_dir.parent.name in {"models", "reconstructions", "figures"}:
        dataset_dir = output_dir.parent.parent
    else:
        dataset_dir = output_dir
    return dataset_dir / "_cache"


def cache_entry_dir(output_dir: Path, category: str, metadata: dict[str, object], prefix: str) -> Path:
    cache_dir = dataset_cache_root(output_dir) / category / f"{prefix}_{cache_key(metadata)}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "metadata.json").write_text(json.dumps(_jsonable_metadata(metadata), indent=2), encoding="utf-8")
    return cache_dir


def epipolar_cache_metadata(
    pose_data: PoseData,
    epipolar_threshold_px: float,
    distance_mode: str,
    pose_data_mode: str,
    pose_filter_window: int,
    pose_outlier_threshold_ratio: float,
    pose_amplitude_lower_percentile: float,
    pose_amplitude_upper_percentile: float,
) -> dict[str, object]:
    return {
        "camera_names": list(pose_data.camera_names),
        "n_frames": int(pose_data.frames.shape[0]),
        "frame_signature": frame_signature(pose_data.frames),
        "pose_data_signature": pose_data_signature(pose_data),
        "epipolar_threshold_px": float(epipolar_threshold_px),
        "distance_mode": str(distance_mode),
        "pose_data_mode": pose_data_mode,
        "pose_filter_window": int(pose_filter_window),
        "pose_outlier_threshold_ratio": float(pose_outlier_threshold_ratio),
        "pose_amplitude_lower_percentile": float(pose_amplitude_lower_percentile),
        "pose_amplitude_upper_percentile": float(pose_amplitude_upper_percentile),
    }


def save_epipolar_cache(
    cache_path: Path, epipolar_coherence: np.ndarray, metadata: dict[str, object], compute_time_s: float
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        epipolar_coherence=np.asarray(epipolar_coherence, dtype=float),
        compute_time_s=np.asarray(compute_time_s, dtype=float),
        metadata=np.asarray(json.dumps(_jsonable_metadata(metadata)), dtype=object),
    )


def load_epipolar_cache(cache_path: Path) -> tuple[np.ndarray, float]:
    with np.load(cache_path, allow_pickle=True) as data:
        epipolar_coherence = np.asarray(data["epipolar_coherence"], dtype=float)
        compute_time_s = float(np.asarray(data["compute_time_s"]).item()) if "compute_time_s" in data else 0.0
    return epipolar_coherence, compute_time_s


def load_or_compute_epipolar_cache(
    *,
    output_dir: Path,
    pose_data: PoseData,
    calibrations: dict[str, CameraCalibration],
    coherence_method: str,
    epipolar_threshold_px: float,
    pose_data_mode: str,
    pose_filter_window: int,
    pose_outlier_threshold_ratio: float,
    pose_amplitude_lower_percentile: float,
    pose_amplitude_upper_percentile: float,
) -> tuple[np.ndarray, float, Path, str]:
    coherence_method = canonical_coherence_method(coherence_method)
    support_method = support_coherence_method_for_runtime(coherence_method)
    distance_mode = "symmetric" if support_method == "epipolar_fast" else "sampson"
    metadata = epipolar_cache_metadata(
        pose_data,
        epipolar_threshold_px,
        distance_mode,
        pose_data_mode,
        pose_filter_window,
        pose_outlier_threshold_ratio,
        pose_amplitude_lower_percentile,
        pose_amplitude_upper_percentile,
    )
    cache_dir = cache_entry_dir(output_dir, "epipolar", metadata, prefix="epipolar")
    cache_path = cache_dir / "epipolar_coherence.npz"
    if metadata_cache_matches(cache_path, metadata):
        coherence, compute_time_s = load_epipolar_cache(cache_path)
        return coherence, compute_time_s, cache_path, "cache"

    ordered_calibrations = [calibrations[name] for name in pose_data.camera_names]
    fundamental_matrices = {
        (i_cam, j_cam): fundamental_matrix(ordered_calibrations[i_cam], ordered_calibrations[j_cam])
        for i_cam in range(len(ordered_calibrations))
        for j_cam in range(len(ordered_calibrations))
        if i_cam != j_cam
    }
    t0 = time.perf_counter()
    coherence = compute_epipolar_coherence(
        pose_data,
        fundamental_matrices,
        threshold_px=epipolar_threshold_px,
        distance_mode=distance_mode,
    )
    compute_time_s = time.perf_counter() - t0
    save_epipolar_cache(cache_path, coherence, metadata, compute_time_s)
    return coherence, compute_time_s, cache_path, "computed_now"


def flip_cache_metadata(
    pose_data: PoseData,
    *,
    method: str,
    pose_data_mode: str,
    pose_filter_window: int,
    pose_outlier_threshold_ratio: float,
    pose_amplitude_lower_percentile: float,
    pose_amplitude_upper_percentile: float,
    improvement_ratio: float,
    min_gain_px: float,
    min_other_cameras: int,
    restrict_to_outliers: bool,
    outlier_percentile: float,
    outlier_floor_px: float,
    tau_px: float | None,
    temporal_weight: float,
    temporal_tau_px: float,
    temporal_min_valid_keypoints: int,
) -> dict[str, object]:
    return {
        "camera_names": list(pose_data.camera_names),
        "n_frames": int(pose_data.frames.shape[0]),
        "frame_signature": frame_signature(pose_data.frames),
        "pose_data_signature": pose_data_signature(pose_data),
        "method": str(method),
        "pose_data_mode": pose_data_mode,
        "pose_filter_window": int(pose_filter_window),
        "pose_outlier_threshold_ratio": float(pose_outlier_threshold_ratio),
        "pose_amplitude_lower_percentile": float(pose_amplitude_lower_percentile),
        "pose_amplitude_upper_percentile": float(pose_amplitude_upper_percentile),
        "improvement_ratio": float(improvement_ratio),
        "min_gain_px": float(min_gain_px),
        "min_other_cameras": int(min_other_cameras),
        "restrict_to_outliers": bool(restrict_to_outliers),
        "outlier_percentile": float(outlier_percentile),
        "outlier_floor_px": float(outlier_floor_px),
        "tau_px": None if tau_px is None else float(tau_px),
        "temporal_weight": float(temporal_weight),
        "temporal_tau_px": float(temporal_tau_px),
        "temporal_min_valid_keypoints": int(temporal_min_valid_keypoints),
        "epipolar_pair_weighting": "baseline_confidence_weighted",
        "epipolar_keypoint_weighting": "torso_proximal_priority",
        "epipolar_distance_mode": (
            "symmetric" if str(method) in {"epipolar_fast", "epipolar_fast_viterbi"} else "sampson"
        ),
        "epipolar_flip_scoring_version": (
            7 if str(method) in {"epipolar", "epipolar_fast", "epipolar_viterbi", "epipolar_fast_viterbi"} else 1
        ),
        "temporal_smoothing_window": (
            5 if str(method) in {"epipolar", "epipolar_fast", "epipolar_viterbi", "epipolar_fast_viterbi"} else 1
        ),
    }


def save_flip_cache(
    cache_path: Path,
    suspect_mask: np.ndarray,
    diagnostics: dict[str, object],
    metadata: dict[str, object],
    compute_time_s: float,
    detail_arrays: dict[str, np.ndarray] | None = None,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "suspect_mask": np.asarray(suspect_mask, dtype=bool),
        "diagnostics": np.asarray(json.dumps(_jsonable_metadata(diagnostics)), dtype=object),
        "compute_time_s": np.asarray(float(compute_time_s), dtype=float),
        "metadata": np.asarray(json.dumps(_jsonable_metadata(metadata)), dtype=object),
    }
    if detail_arrays:
        for key, value in detail_arrays.items():
            payload[key] = np.asarray(value)
    np.savez(cache_path, **payload)


def load_flip_cache(cache_path: Path) -> tuple[np.ndarray, dict[str, object], float]:
    with np.load(cache_path, allow_pickle=True) as data:
        suspect_mask = np.asarray(data["suspect_mask"], dtype=bool)
        diagnostics = json.loads(str(np.asarray(data["diagnostics"]).item()))
        compute_time_s = float(np.asarray(data["compute_time_s"]).item()) if "compute_time_s" in data else 0.0
    return suspect_mask, diagnostics, compute_time_s


def load_or_compute_left_right_flip_cache(
    *,
    output_dir: Path,
    pose_data: PoseData,
    calibrations: dict[str, CameraCalibration],
    method: str,
    pose_data_mode: str,
    pose_filter_window: int,
    pose_outlier_threshold_ratio: float,
    pose_amplitude_lower_percentile: float,
    pose_amplitude_upper_percentile: float,
    improvement_ratio: float = DEFAULT_FLIP_IMPROVEMENT_RATIO,
    min_gain_px: float = DEFAULT_FLIP_MIN_GAIN_PX,
    min_other_cameras: int = DEFAULT_FLIP_MIN_OTHER_CAMERAS,
    restrict_to_outliers: bool = DEFAULT_FLIP_RESTRICT_TO_OUTLIERS,
    outlier_percentile: float = DEFAULT_FLIP_OUTLIER_PERCENTILE,
    outlier_floor_px: float = DEFAULT_FLIP_OUTLIER_FLOOR_PX,
    tau_px: float | None = None,
    temporal_weight: float = DEFAULT_FLIP_TEMPORAL_WEIGHT,
    temporal_tau_px: float = DEFAULT_FLIP_TEMPORAL_TAU_PX,
    temporal_min_valid_keypoints: int = DEFAULT_FLIP_TEMPORAL_MIN_VALID_KEYPOINTS,
) -> tuple[np.ndarray, dict[str, object], float, Path, str]:
    metadata = flip_cache_metadata(
        pose_data,
        method=method,
        pose_data_mode=pose_data_mode,
        pose_filter_window=pose_filter_window,
        pose_outlier_threshold_ratio=pose_outlier_threshold_ratio,
        pose_amplitude_lower_percentile=pose_amplitude_lower_percentile,
        pose_amplitude_upper_percentile=pose_amplitude_upper_percentile,
        improvement_ratio=improvement_ratio,
        min_gain_px=min_gain_px,
        min_other_cameras=min_other_cameras,
        restrict_to_outliers=bool(restrict_to_outliers),
        outlier_percentile=float(outlier_percentile),
        outlier_floor_px=float(outlier_floor_px),
        tau_px=tau_px,
        temporal_weight=float(temporal_weight),
        temporal_tau_px=float(temporal_tau_px),
        temporal_min_valid_keypoints=int(temporal_min_valid_keypoints),
    )
    cache_dir = cache_entry_dir(output_dir, "flip", metadata, prefix=f"flip_{method}")
    cache_path = cache_dir / "flip_diagnostics.npz"
    if metadata_cache_matches(cache_path, metadata):
        suspect_mask, diagnostics, compute_time_s = load_flip_cache(cache_path)
        return suspect_mask, diagnostics, compute_time_s, cache_path, "cache"

    t0 = time.perf_counter()
    suspect_mask, diagnostics, detail_arrays = detect_left_right_flip_diagnostics(
        pose_data,
        calibrations,
        method=method,
        improvement_ratio=improvement_ratio,
        min_gain_px=min_gain_px,
        min_other_cameras=min_other_cameras,
        restrict_to_outliers=restrict_to_outliers,
        outlier_percentile=outlier_percentile,
        outlier_floor_px=outlier_floor_px,
        geometry_tau_px=(tau_px if tau_px is not None else 1.0),
        temporal_weight=temporal_weight,
        temporal_tau_px=temporal_tau_px,
        temporal_min_valid_keypoints=temporal_min_valid_keypoints,
    )
    diagnostics = dict(diagnostics)
    diagnostics["tau_px"] = None if tau_px is None else float(tau_px)
    diagnostics["temporal_weight"] = float(temporal_weight)
    diagnostics["temporal_tau_px"] = float(temporal_tau_px)
    diagnostics["temporal_min_valid_keypoints"] = int(temporal_min_valid_keypoints)
    compute_time_s = time.perf_counter() - t0
    save_flip_cache(cache_path, suspect_mask, diagnostics, metadata, compute_time_s, detail_arrays=detail_arrays)
    return suspect_mask, diagnostics, compute_time_s, cache_path, "computed_now"


def slice_pose_data(pose_data: PoseData, frame_indices: list[int] | np.ndarray) -> PoseData:
    idx = np.asarray(frame_indices, dtype=int)
    return PoseData(
        camera_names=list(pose_data.camera_names),
        frames=np.asarray(pose_data.frames[idx], dtype=int),
        keypoints=np.asarray(pose_data.keypoints[:, idx, :, :], dtype=float),
        scores=np.asarray(pose_data.scores[:, idx, :], dtype=float),
        frame_stride=int(getattr(pose_data, "frame_stride", 1)),
        raw_keypoints=(
            None if pose_data.raw_keypoints is None else np.asarray(pose_data.raw_keypoints[:, idx, :, :], dtype=float)
        ),
        filtered_keypoints=(
            None
            if pose_data.filtered_keypoints is None
            else np.asarray(pose_data.filtered_keypoints[:, idx, :, :], dtype=float)
        ),
    )


def pose_variant_cache_metadata(
    pose_data: PoseData,
    *,
    correction_mode: str,
    pose_data_mode: str,
    pose_filter_window: int,
    pose_outlier_threshold_ratio: float,
    pose_amplitude_lower_percentile: float,
    pose_amplitude_upper_percentile: float,
    flip_method: str | None = None,
    improvement_ratio: float | None = None,
    min_gain_px: float | None = None,
    min_other_cameras: int | None = None,
    restrict_to_outliers: bool | None = None,
    outlier_percentile: float | None = None,
    outlier_floor_px: float | None = None,
    tau_px: float | None = None,
    temporal_weight: float | None = None,
    temporal_tau_px: float | None = None,
    temporal_min_valid_keypoints: int | None = None,
) -> dict[str, object]:
    metadata = {
        "camera_names": list(pose_data.camera_names),
        "n_frames": int(pose_data.frames.shape[0]),
        "frame_signature": frame_signature(pose_data.frames),
        "source_pose_data_signature": pose_data_signature(pose_data),
        "correction_mode": str(correction_mode),
        "pose_data_mode": str(pose_data_mode),
        "pose_filter_window": int(pose_filter_window),
        "pose_outlier_threshold_ratio": float(pose_outlier_threshold_ratio),
        "pose_amplitude_lower_percentile": float(pose_amplitude_lower_percentile),
        "pose_amplitude_upper_percentile": float(pose_amplitude_upper_percentile),
    }
    if flip_method is not None:
        metadata.update(
            {
                "flip_method": str(flip_method),
                "improvement_ratio": float(improvement_ratio),
                "min_gain_px": float(min_gain_px),
                "min_other_cameras": int(min_other_cameras),
                "restrict_to_outliers": bool(restrict_to_outliers),
                "outlier_percentile": float(outlier_percentile),
                "outlier_floor_px": float(outlier_floor_px),
                "temporal_weight": float(temporal_weight),
                "temporal_tau_px": float(temporal_tau_px),
                "temporal_min_valid_keypoints": int(temporal_min_valid_keypoints),
            }
        )
        if tau_px is not None:
            metadata["tau_px"] = float(tau_px)
    return metadata


def save_pose_variant_cache(
    cache_path: Path,
    pose_data: PoseData,
    metadata: dict[str, object],
    *,
    diagnostics: dict[str, object] | None = None,
    suspect_mask: np.ndarray | None = None,
    compute_time_s: float = 0.0,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "camera_names": np.asarray(list(pose_data.camera_names), dtype=object),
        "frames": np.asarray(pose_data.frames, dtype=int),
        "keypoints": np.asarray(pose_data.keypoints, dtype=float),
        "scores": np.asarray(pose_data.scores, dtype=float),
        "frame_stride": np.asarray(int(getattr(pose_data, "frame_stride", 1)), dtype=int),
        "metadata": np.asarray(json.dumps(_jsonable_metadata(metadata)), dtype=object),
        "diagnostics": np.asarray(json.dumps(_jsonable_metadata(diagnostics or {})), dtype=object),
        "compute_time_s": np.asarray(float(compute_time_s), dtype=float),
    }
    if pose_data.raw_keypoints is not None:
        payload["raw_keypoints"] = np.asarray(pose_data.raw_keypoints, dtype=float)
    if pose_data.filtered_keypoints is not None:
        payload["filtered_keypoints"] = np.asarray(pose_data.filtered_keypoints, dtype=float)
    if suspect_mask is not None:
        payload["suspect_mask"] = np.asarray(suspect_mask, dtype=bool)
    np.savez(cache_path, **payload)


def load_pose_variant_cache(cache_path: Path) -> tuple[PoseData, dict[str, object], np.ndarray | None, float]:
    with np.load(cache_path, allow_pickle=True) as data:
        pose_data = PoseData(
            camera_names=[str(name) for name in np.asarray(data["camera_names"], dtype=object).tolist()],
            frames=np.asarray(data["frames"], dtype=int),
            keypoints=np.asarray(data["keypoints"], dtype=float),
            scores=np.asarray(data["scores"], dtype=float),
            frame_stride=int(np.asarray(data["frame_stride"]).item()) if "frame_stride" in data else 1,
            raw_keypoints=np.asarray(data["raw_keypoints"], dtype=float) if "raw_keypoints" in data else None,
            filtered_keypoints=(
                np.asarray(data["filtered_keypoints"], dtype=float) if "filtered_keypoints" in data else None
            ),
        )
        diagnostics = json.loads(str(np.asarray(data["diagnostics"]).item())) if "diagnostics" in data else {}
        suspect_mask = np.asarray(data["suspect_mask"], dtype=bool) if "suspect_mask" in data else None
        compute_time_s = float(np.asarray(data["compute_time_s"]).item()) if "compute_time_s" in data else 0.0
    return pose_data, diagnostics, suspect_mask, compute_time_s


def load_or_compute_pose_data_variant_cache(
    *,
    output_dir: Path,
    pose_data: PoseData,
    calibrations: dict[str, CameraCalibration],
    pose_data_mode: str,
    pose_filter_window: int,
    pose_outlier_threshold_ratio: float,
    pose_amplitude_lower_percentile: float,
    pose_amplitude_upper_percentile: float,
    correction_mode: str = "none",
    flip_method: str | None = None,
    improvement_ratio: float = DEFAULT_FLIP_IMPROVEMENT_RATIO,
    min_gain_px: float = DEFAULT_FLIP_MIN_GAIN_PX,
    min_other_cameras: int = DEFAULT_FLIP_MIN_OTHER_CAMERAS,
    restrict_to_outliers: bool = DEFAULT_FLIP_RESTRICT_TO_OUTLIERS,
    outlier_percentile: float = DEFAULT_FLIP_OUTLIER_PERCENTILE,
    outlier_floor_px: float = DEFAULT_FLIP_OUTLIER_FLOOR_PX,
    tau_px: float | None = None,
    temporal_weight: float = DEFAULT_FLIP_TEMPORAL_WEIGHT,
    temporal_tau_px: float = DEFAULT_FLIP_TEMPORAL_TAU_PX,
    temporal_min_valid_keypoints: int = DEFAULT_FLIP_TEMPORAL_MIN_VALID_KEYPOINTS,
) -> tuple[PoseData, dict[str, object] | None, float, Path, str]:
    metadata = pose_variant_cache_metadata(
        pose_data,
        correction_mode=correction_mode,
        pose_data_mode=pose_data_mode,
        pose_filter_window=pose_filter_window,
        pose_outlier_threshold_ratio=pose_outlier_threshold_ratio,
        pose_amplitude_lower_percentile=pose_amplitude_lower_percentile,
        pose_amplitude_upper_percentile=pose_amplitude_upper_percentile,
        flip_method=flip_method,
        improvement_ratio=improvement_ratio if flip_method is not None else None,
        min_gain_px=min_gain_px if flip_method is not None else None,
        min_other_cameras=min_other_cameras if flip_method is not None else None,
        restrict_to_outliers=restrict_to_outliers if flip_method is not None else None,
        outlier_percentile=outlier_percentile if flip_method is not None else None,
        outlier_floor_px=outlier_floor_px if flip_method is not None else None,
        tau_px=tau_px if flip_method is not None else None,
        temporal_weight=temporal_weight if flip_method is not None else None,
        temporal_tau_px=temporal_tau_px if flip_method is not None else None,
        temporal_min_valid_keypoints=temporal_min_valid_keypoints if flip_method is not None else None,
    )
    cache_dir = cache_entry_dir(output_dir, "pose2d", metadata, prefix=f"pose2d_{correction_mode}")
    cache_path = cache_dir / "pose_data_variant.npz"
    if metadata_cache_matches(cache_path, metadata):
        corrected_pose_data, diagnostics, _suspect_mask, compute_time_s = load_pose_variant_cache(cache_path)
        if diagnostics:
            diagnostics = dict(diagnostics)
            diagnostics.setdefault("source", "cache")
        return corrected_pose_data, (diagnostics or None), compute_time_s, cache_path, "cache"

    if correction_mode == "none":
        t0 = time.perf_counter()
        corrected_pose_data = PoseData(
            camera_names=list(pose_data.camera_names),
            frames=np.asarray(pose_data.frames, dtype=int),
            keypoints=np.asarray(pose_data.keypoints, dtype=float),
            scores=np.asarray(pose_data.scores, dtype=float),
            frame_stride=int(getattr(pose_data, "frame_stride", 1)),
            raw_keypoints=None if pose_data.raw_keypoints is None else np.asarray(pose_data.raw_keypoints, dtype=float),
            filtered_keypoints=(
                None if pose_data.filtered_keypoints is None else np.asarray(pose_data.filtered_keypoints, dtype=float)
            ),
        )
        compute_time_s = time.perf_counter() - t0
        save_pose_variant_cache(cache_path, corrected_pose_data, metadata, compute_time_s=compute_time_s)
        return corrected_pose_data, None, compute_time_s, cache_path, "computed_now"

    if correction_mode != "flip":
        raise ValueError(f"Unsupported pose-data correction mode: {correction_mode}")
    if flip_method is None:
        raise ValueError("flip_method is required when correction_mode='flip'.")

    suspect_mask, diagnostics, flip_compute_time_s, flip_cache_path, flip_source = (
        load_or_compute_left_right_flip_cache(
            output_dir=output_dir,
            pose_data=pose_data,
            calibrations=calibrations,
            method=flip_method,
            pose_data_mode=pose_data_mode,
            pose_filter_window=pose_filter_window,
            pose_outlier_threshold_ratio=pose_outlier_threshold_ratio,
            pose_amplitude_lower_percentile=pose_amplitude_lower_percentile,
            pose_amplitude_upper_percentile=pose_amplitude_upper_percentile,
            improvement_ratio=improvement_ratio,
            min_gain_px=min_gain_px,
            min_other_cameras=min_other_cameras,
            restrict_to_outliers=restrict_to_outliers,
            outlier_percentile=outlier_percentile,
            outlier_floor_px=outlier_floor_px,
            tau_px=tau_px,
            temporal_weight=temporal_weight,
            temporal_tau_px=temporal_tau_px,
            temporal_min_valid_keypoints=temporal_min_valid_keypoints,
        )
    )
    corrected_pose_data = apply_left_right_flip_corrections(pose_data, suspect_mask)
    diagnostics = dict(diagnostics)
    diagnostics["compute_time_s"] = float(flip_compute_time_s)
    diagnostics["flip_cache_path"] = str(flip_cache_path)
    save_pose_variant_cache(
        cache_path,
        corrected_pose_data,
        metadata,
        diagnostics=diagnostics,
        suspect_mask=suspect_mask,
        compute_time_s=float(flip_compute_time_s),
    )
    diagnostics["source"] = str(flip_source)
    return corrected_pose_data, diagnostics, float(flip_compute_time_s), cache_path, "computed_now"


def reconstruction_with_full_frame_support(
    pose_data: PoseData,
    bootstrap_reconstruction: ReconstructionResult,
    epipolar_coherence: np.ndarray,
    epipolar_compute_time_s: float,
    coherence_method: str,
    bootstrap_frame_global_idx: int,
) -> ReconstructionResult:
    n_frames = pose_data.frames.shape[0]
    n_cams = len(pose_data.camera_names)
    points_3d = np.full((n_frames, len(COCO17), 3), np.nan, dtype=float)
    mean_confidence = np.full((n_frames, len(COCO17)), np.nan, dtype=float)
    reprojection_error = np.full((n_frames, len(COCO17)), np.nan, dtype=float)
    reprojection_error_per_view = np.full((n_frames, len(COCO17), n_cams), np.nan, dtype=float)
    triangulation_coherence = np.zeros((n_frames, len(COCO17), n_cams), dtype=float)
    excluded_views = np.ones((n_frames, len(COCO17), n_cams), dtype=bool)

    local_idx = 0
    points_3d[bootstrap_frame_global_idx] = bootstrap_reconstruction.points_3d[local_idx]
    mean_confidence[bootstrap_frame_global_idx] = bootstrap_reconstruction.mean_confidence[local_idx]
    reprojection_error[bootstrap_frame_global_idx] = bootstrap_reconstruction.reprojection_error[local_idx]
    reprojection_error_per_view[bootstrap_frame_global_idx] = bootstrap_reconstruction.reprojection_error_per_view[
        local_idx
    ]
    triangulation_coherence[bootstrap_frame_global_idx] = bootstrap_reconstruction.triangulation_coherence[local_idx]
    excluded_views[bootstrap_frame_global_idx] = bootstrap_reconstruction.excluded_views[local_idx]

    multiview_coherence = select_active_coherence(
        epipolar_coherence=epipolar_coherence,
        triangulation_coherence=triangulation_coherence,
        coherence_method=coherence_method,
    )
    return ReconstructionResult(
        frames=np.asarray(pose_data.frames, dtype=int),
        points_3d=points_3d,
        mean_confidence=mean_confidence,
        reprojection_error=reprojection_error,
        reprojection_error_per_view=reprojection_error_per_view,
        multiview_coherence=multiview_coherence,
        epipolar_coherence=np.asarray(epipolar_coherence, dtype=float),
        triangulation_coherence=triangulation_coherence,
        excluded_views=excluded_views,
        coherence_method=coherence_method,
        epipolar_coherence_compute_time_s=float(epipolar_compute_time_s),
        triangulation_compute_time_s=float(bootstrap_reconstruction.triangulation_compute_time_s),
    )


def estimate_segment_lengths_first_frame(reconstruction: ReconstructionResult) -> tuple[SegmentLengths, int]:
    for frame_idx in range(reconstruction.points_3d.shape[0]):
        frame_points = reconstruction.points_3d[frame_idx : frame_idx + 1]
        if np.any(np.isfinite(frame_points)):
            single_frame_reconstruction = ReconstructionResult(
                frames=reconstruction.frames[frame_idx : frame_idx + 1],
                points_3d=frame_points,
                mean_confidence=reconstruction.mean_confidence[frame_idx : frame_idx + 1],
                reprojection_error=reconstruction.reprojection_error[frame_idx : frame_idx + 1],
                reprojection_error_per_view=reconstruction.reprojection_error_per_view[frame_idx : frame_idx + 1],
                multiview_coherence=reconstruction.multiview_coherence[frame_idx : frame_idx + 1],
                epipolar_coherence=reconstruction.epipolar_coherence[frame_idx : frame_idx + 1],
                triangulation_coherence=reconstruction.triangulation_coherence[frame_idx : frame_idx + 1],
                excluded_views=reconstruction.excluded_views[frame_idx : frame_idx + 1],
                coherence_method=reconstruction.coherence_method,
                epipolar_coherence_compute_time_s=reconstruction.epipolar_coherence_compute_time_s,
                triangulation_compute_time_s=reconstruction.triangulation_compute_time_s,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                return estimate_segment_lengths(single_frame_reconstruction, fps=1.0, window_s=1.0), frame_idx
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return estimate_segment_lengths(reconstruction, fps=1.0, window_s=1.0), 0


def load_or_compute_triangulation_cache(
    *,
    output_dir: Path,
    pose_data: PoseData,
    calibrations: dict[str, CameraCalibration],
    coherence_method: str,
    triangulation_method: str,
    reprojection_threshold_px: float | None,
    min_cameras_for_triangulation: int,
    epipolar_threshold_px: float,
    triangulation_workers: int,
    pose_data_mode: str,
    pose_filter_window: int,
    pose_outlier_threshold_ratio: float,
    pose_amplitude_lower_percentile: float,
    pose_amplitude_upper_percentile: float,
) -> tuple[ReconstructionResult, Path, Path, str]:
    triangulation_method = canonical_triangulation_method(triangulation_method)
    coherence_method = canonical_coherence_method(coherence_method, triangulation_method)
    effective_triangulation_method = triangulation_method_from_coherence_method(coherence_method, triangulation_method)
    metadata = reconstruction_cache_metadata(
        pose_data,
        reprojection_threshold_px,
        min_cameras_for_triangulation,
        epipolar_threshold_px,
        effective_triangulation_method,
        pose_data_mode,
        pose_filter_window,
        pose_outlier_threshold_ratio,
        pose_amplitude_lower_percentile,
        pose_amplitude_upper_percentile,
    )
    cache_dir = cache_entry_dir(
        output_dir, "triangulation", metadata, prefix=f"triang_{effective_triangulation_method}"
    )
    cache_path = cache_dir / "reconstruction.npz"
    epipolar_coherence, epipolar_compute_time_s, epipolar_cache_path, epipolar_source = load_or_compute_epipolar_cache(
        output_dir=output_dir,
        pose_data=pose_data,
        calibrations=calibrations,
        coherence_method=coherence_method,
        epipolar_threshold_px=epipolar_threshold_px,
        pose_data_mode=pose_data_mode,
        pose_filter_window=pose_filter_window,
        pose_outlier_threshold_ratio=pose_outlier_threshold_ratio,
        pose_amplitude_lower_percentile=pose_amplitude_lower_percentile,
        pose_amplitude_upper_percentile=pose_amplitude_upper_percentile,
    )
    if metadata_cache_matches(cache_path, metadata):
        reconstruction = load_reconstruction_cache(cache_path, coherence_method=coherence_method)
        reconstruction.epipolar_coherence_compute_time_s = float(epipolar_compute_time_s)
        return reconstruction, cache_path, epipolar_cache_path, "cache"

    triangulation_start = time.perf_counter()
    reconstruction = triangulate_pose2sim_like(
        pose_data,
        calibrations,
        error_threshold_px=reprojection_threshold_px,
        min_cameras_for_triangulation=min_cameras_for_triangulation,
        coherence_method=coherence_method,
        epipolar_threshold_px=epipolar_threshold_px,
        triangulation_method=effective_triangulation_method,
        n_workers=triangulation_workers,
        precomputed_epipolar_coherence=epipolar_coherence,
        precomputed_epipolar_time_s=epipolar_compute_time_s,
    )
    reconstruction.triangulation_compute_time_s = time.perf_counter() - triangulation_start
    save_reconstruction_cache(cache_path, reconstruction, metadata)
    return reconstruction, cache_path, epipolar_cache_path, "computed_now"


def load_or_build_model_cache(
    *,
    output_dir: Path,
    reconstruction: ReconstructionResult,
    reconstruction_cache_path: Path,
    fps: float,
    subject_mass_kg: float,
    initial_rotation_correction: bool,
    lengths_mode: str,
    model_variant: str = "single_trunk",
    symmetrize_limbs: bool = True,
) -> tuple[SegmentLengths, Path, Path, int, float, str]:
    build_start = time.perf_counter()
    if lengths_mode == "first_frame_only":
        lengths, bootstrap_frame_idx = estimate_segment_lengths_first_frame(reconstruction)
    else:
        lengths = estimate_segment_lengths(reconstruction, fps)
        bootstrap_frame_idx = 0

    metadata = model_stage_metadata(
        reconstruction_cache_path,
        reconstruction,
        fps,
        subject_mass_kg,
        initial_rotation_correction,
        model_variant=model_variant,
        symmetrize_limbs=symmetrize_limbs,
    )
    metadata["lengths_mode"] = lengths_mode
    metadata["bootstrap_frame_idx"] = int(bootstrap_frame_idx)
    cache_dir = cache_entry_dir(output_dir, "model", metadata, prefix="model")
    cache_path = cache_dir / "model_stage.npz"
    biomod_cache_path = cache_dir / "vitpose_chain.bioMod"
    if metadata_cache_matches(cache_path, metadata) and biomod_cache_path.exists():
        cached_lengths, _biomod_path, compute_time_s = load_model_stage(cache_path)
        return cached_lengths, biomod_cache_path, cache_path, int(bootstrap_frame_idx), float(compute_time_s), "cache"

    build_biomod(
        lengths,
        biomod_cache_path,
        subject_mass_kg=subject_mass_kg,
        reconstruction=reconstruction,
        apply_initial_root_rotation_correction=initial_rotation_correction,
        model_variant=model_variant,
        symmetrize_limbs=symmetrize_limbs,
    )
    compute_time_s = time.perf_counter() - build_start
    save_model_stage(cache_path, lengths, biomod_cache_path, metadata, compute_time_s=compute_time_s)
    return lengths, biomod_cache_path, cache_path, int(bootstrap_frame_idx), float(compute_time_s), "computed_now"


def prepare_pose_data_for_reconstruction(
    *,
    output_dir: Path,
    pose_data: PoseData,
    calibrations: dict[str, CameraCalibration],
    coherence_method: str,
    reprojection_threshold_px: float | None,
    epipolar_threshold_px: float,
    pose_data_mode: str,
    pose_filter_window: int,
    pose_outlier_threshold_ratio: float,
    pose_amplitude_lower_percentile: float,
    pose_amplitude_upper_percentile: float,
    flip_left_right: bool,
    flip_improvement_ratio: float,
    flip_min_gain_px: float,
    flip_min_other_cameras: int,
    flip_restrict_to_outliers: bool,
    flip_outlier_percentile: float,
    flip_outlier_floor_px: float,
    flip_temporal_weight: float,
    flip_temporal_tau_px: float,
    flip_temporal_min_valid_keypoints: int,
    flip_method: str | None = None,
) -> tuple[PoseData, dict[str, object] | None, Path | None, str]:
    coherence_method = canonical_coherence_method(coherence_method)
    if not flip_left_right:
        return pose_data, None, None, "computed_now"

    active_flip_method = flip_method
    if active_flip_method is None:
        if coherence_method in {"epipolar", "epipolar_fast", "epipolar_framewise", "epipolar_fast_framewise"}:
            active_flip_method = support_coherence_method_for_runtime(coherence_method)
        else:
            active_flip_method = "triangulation_exhaustive"
    if active_flip_method == "triangulation":
        active_flip_method = "triangulation_exhaustive"
    if active_flip_method.startswith("triangulation_"):
        tau_px = reprojection_threshold_px
    else:
        tau_px = epipolar_threshold_px
    pose_data_used, flip_diagnostics, _compute_time_s, pose_variant_cache_path, pose_variant_source = (
        load_or_compute_pose_data_variant_cache(
            output_dir=output_dir,
            pose_data=pose_data,
            calibrations=calibrations,
            correction_mode="flip",
            flip_method=active_flip_method,
            pose_data_mode=pose_data_mode,
            pose_filter_window=pose_filter_window,
            pose_outlier_threshold_ratio=pose_outlier_threshold_ratio,
            pose_amplitude_lower_percentile=pose_amplitude_lower_percentile,
            pose_amplitude_upper_percentile=pose_amplitude_upper_percentile,
            improvement_ratio=flip_improvement_ratio,
            min_gain_px=flip_min_gain_px,
            min_other_cameras=flip_min_other_cameras,
            restrict_to_outliers=flip_restrict_to_outliers,
            outlier_percentile=flip_outlier_percentile,
            outlier_floor_px=flip_outlier_floor_px,
            tau_px=tau_px,
            temporal_weight=flip_temporal_weight,
            temporal_tau_px=flip_temporal_tau_px,
            temporal_min_valid_keypoints=flip_temporal_min_valid_keypoints,
        )
    )
    if flip_diagnostics is not None:
        flip_diagnostics = dict(flip_diagnostics)
        flip_diagnostics["cache_path"] = str(pose_variant_cache_path)
        flip_diagnostics["pose_data_signature"] = pose_data_signature(pose_data_used)
    return pose_data_used, flip_diagnostics, pose_variant_cache_path, str(pose_variant_source)


def with_version_info(summary: dict[str, object], family: str) -> dict[str, object]:
    latest = latest_version_for_family(family)
    enriched = dict(summary)
    enriched["bundle_schema_version"] = int(BUNDLE_SCHEMA_VERSION)
    enriched["algorithm_versions"] = dict(ALGORITHM_VERSIONS)
    enriched["family_version"] = latest
    enriched["is_latest_family_version"] = True
    return enriched


def _canonical_trc_marker_name(marker_name: str) -> str:
    """Normalize TRC marker labels so multiple header styles map consistently."""

    text = str(marker_name).strip().lower()
    text = text.replace("_", " ")
    text = " ".join(text.split())
    return text


TRC_MARKER_TO_COCO = {
    "nose": "nose",
    "l eye": "left_eye",
    "left eye": "left_eye",
    "r eye": "right_eye",
    "right eye": "right_eye",
    "l ear": "left_ear",
    "left ear": "left_ear",
    "r ear": "right_ear",
    "right ear": "right_ear",
    "l shoulder": "left_shoulder",
    "left shoulder": "left_shoulder",
    "r shoulder": "right_shoulder",
    "right shoulder": "right_shoulder",
    "l elbow": "left_elbow",
    "left elbow": "left_elbow",
    "r elbow": "right_elbow",
    "right elbow": "right_elbow",
    "l wrist": "left_wrist",
    "left wrist": "left_wrist",
    "r wrist": "right_wrist",
    "right wrist": "right_wrist",
    "l hip": "left_hip",
    "left hip": "left_hip",
    "left_hip": "left_hip",
    "r hip": "right_hip",
    "right hip": "right_hip",
    "right_hip": "right_hip",
    "l knee": "left_knee",
    "left knee": "left_knee",
    "left_knee": "left_knee",
    "r knee": "right_knee",
    "right knee": "right_knee",
    "right_knee": "right_knee",
    "l ankle": "left_ankle",
    "left ankle": "left_ankle",
    "left_ankle": "left_ankle",
    "r ankle": "right_ankle",
    "right ankle": "right_ankle",
    "right_ankle": "right_ankle",
    "left shoulder": "left_shoulder",
    "right shoulder": "right_shoulder",
    "left elbow": "left_elbow",
    "right elbow": "right_elbow",
    "left wrist": "left_wrist",
    "right wrist": "right_wrist",
}


def parse_trc_points(trc_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Parse a TRC file generated either by Pose2Sim or by the local exporter."""

    with trc_path.open("r", encoding="utf-8") as file:
        lines = [line.rstrip("\n") for line in file]

    metadata_keys = lines[1].split("\t")
    metadata_values = lines[2].split("\t")
    metadata = {key: value for key, value in zip(metadata_keys, metadata_values)}
    data_rate = float(metadata["DataRate"])

    marker_labels = [label for label in lines[3].split("\t")[2:] if label]

    rows = []
    for line in lines[5:]:
        if line.strip():
            parts = line.split("\t")
            rows.append(parts[: 2 + 3 * len(marker_labels)])

    frames = np.asarray([int(row[0]) for row in rows], dtype=int)
    time_s = np.asarray([float(row[1]) for row in rows], dtype=float)
    xyz_values = np.asarray(
        [[float(value) if value != "" else np.nan for value in row[2:]] for row in rows], dtype=float
    )
    points_3d = np.full((len(frames), len(COCO17), 3), np.nan, dtype=float)
    for marker_idx, marker_name in enumerate(marker_labels):
        coco_name = TRC_MARKER_TO_COCO.get(_canonical_trc_marker_name(marker_name))
        if coco_name is None:
            continue
        points_3d[:, KP_INDEX[coco_name], :] = xyz_values[:, 3 * marker_idx : 3 * marker_idx + 3]
    return frames, time_s, points_3d, data_rate


def q_names_from_model(model) -> np.ndarray:
    names = [
        f"{model.segment(i_seg).name().to_string()}:{model.segment(i_seg).nameDof(i_dof).to_string()}"
        for i_seg in range(model.nbSegment())
        for i_dof in range(model.segment(i_seg).nbDof())
    ]
    return np.asarray(names, dtype=object)


def should_apply_initial_rotation_correction(points_3d: np.ndarray) -> bool:
    angle = root_z_correction_angle_from_points(
        points_3d,
        left_hip_idx=KP_INDEX["left_hip"],
        right_hip_idx=KP_INDEX["right_hip"],
        left_shoulder_idx=KP_INDEX["left_shoulder"],
        right_shoulder_idx=KP_INDEX["right_shoulder"],
    )
    return abs(angle) > 1e-8


def extract_root_from_points(
    points_3d: np.ndarray,
    apply_initial_rotation_correction: bool,
    unwrap_rotations: bool,
    unwrap_mode: str | None = None,
    translation_origin: str = "pelvis",
) -> tuple[np.ndarray, bool]:
    correction_angle = root_z_correction_angle_from_points(
        points_3d,
        left_hip_idx=KP_INDEX["left_hip"],
        right_hip_idx=KP_INDEX["right_hip"],
        left_shoulder_idx=KP_INDEX["left_shoulder"],
        right_shoulder_idx=KP_INDEX["right_shoulder"],
    )
    translations, rotation_matrices = build_root_rotation_matrices(
        points_3d,
        left_hip_idx=KP_INDEX["left_hip"],
        right_hip_idx=KP_INDEX["right_hip"],
        left_shoulder_idx=KP_INDEX["left_shoulder"],
        right_shoulder_idx=KP_INDEX["right_shoulder"],
        translation_origin=translation_origin,
    )
    correction_detected = abs(correction_angle) > 1e-8
    correction_applied = bool(apply_initial_rotation_correction and correction_detected)

    root_q = np.full((points_3d.shape[0], 6), np.nan, dtype=float)
    root_q[:, :3] = translations
    for frame_idx in range(points_3d.shape[0]):
        matrix = rotation_matrices[frame_idx]
        if not np.all(np.isfinite(matrix)):
            continue
        if correction_applied:
            matrix = Rotation.from_euler("z", -correction_angle, degrees=False).as_matrix() @ matrix
        root_q[frame_idx, ROOT_ROTATION_SLICE] = Rotation.from_matrix(matrix).as_euler(
            TRUNK_ROOT_ROTATION_SCIPY_SEQUENCE,
            degrees=False,
        )

    effective_unwrap_mode = normalize_root_unwrap_mode(unwrap_mode, legacy_unwrap=unwrap_rotations)
    if effective_unwrap_mode != "off":
        root_q[:, ROOT_ROTATION_SLICE] = stabilize_root_rotations(root_q[:, ROOT_ROTATION_SLICE], effective_unwrap_mode)
    return root_q, correction_applied


def root_kinematics_from_trc(
    trc_path: Path,
    *,
    frames: np.ndarray,
    time_s: np.ndarray,
    points_3d: np.ndarray,
    fps: float,
    initial_rotation_correction: bool,
    unwrap_root: bool,
    root_unwrap_mode: str = "off",
) -> tuple[np.ndarray, np.ndarray, bool, str]:
    """Resolve root kinematics for one imported TRC file.

    When the TRC comes from the local ``Export TRC from q`` workflow, an
    optional sidecar stores the exact root trajectory. Reusing it keeps the
    round-trip consistent with the original EKF reconstruction. Generic TRC
    files still fall back to geometric root extraction from the marker cloud.
    """

    sidecar = load_trc_root_kinematics_sidecar(trc_path)
    if sidecar is not None:
        sidecar_frames = np.asarray(sidecar["frames"], dtype=int)
        sidecar_time_s = np.asarray(sidecar["time_s"], dtype=float)
        if np.array_equal(sidecar_frames, np.asarray(frames, dtype=int)) and np.allclose(
            sidecar_time_s,
            np.asarray(time_s, dtype=float),
            equal_nan=True,
        ):
            return (
                np.asarray(sidecar["q_root"], dtype=float),
                np.asarray(sidecar["qdot_root"], dtype=float),
                bool(initial_rotation_correction),
                "trc_root_kinematics_sidecar",
            )

    effective_root_unwrap_mode = normalize_root_unwrap_mode(root_unwrap_mode, legacy_unwrap=unwrap_root)
    q_root, correction_applied = extract_root_from_points(
        points_3d,
        initial_rotation_correction,
        unwrap_root=(effective_root_unwrap_mode != "off"),
        unwrap_mode=effective_root_unwrap_mode,
    )
    qdot_root = centered_finite_difference(q_root, 1.0 / fps)
    return q_root, qdot_root, correction_applied, "geometric_trc_markers"


def align_points_to_frames(points_3d: np.ndarray, point_frames: np.ndarray, target_frames: np.ndarray) -> np.ndarray:
    aligned = np.full((len(target_frames), points_3d.shape[1], 3), np.nan, dtype=float)
    frame_to_index = {int(frame): idx for idx, frame in enumerate(np.asarray(point_frames, dtype=int))}
    for out_idx, frame in enumerate(np.asarray(target_frames, dtype=int)):
        source_idx = frame_to_index.get(int(frame))
        if source_idx is None:
            continue
        aligned[out_idx] = points_3d[source_idx]
    return aligned


def compute_points_reprojection_error_per_view(
    points_3d: np.ndarray,
    point_frames: np.ndarray,
    calibrations: dict[str, CameraCalibration],
    pose_data: PoseData,
) -> np.ndarray:
    aligned_points = align_points_to_frames(points_3d, point_frames, pose_data.frames)
    errors = np.full((pose_data.frames.shape[0], len(COCO17), len(pose_data.camera_names)), np.nan, dtype=float)
    for cam_idx, cam_name in enumerate(pose_data.camera_names):
        calibration = calibrations[cam_name]
        for frame_idx in range(pose_data.frames.shape[0]):
            for kp_idx in range(len(COCO17)):
                if pose_data.scores[cam_idx, frame_idx, kp_idx] <= 0:
                    continue
                z_uv = pose_data.keypoints[cam_idx, frame_idx, kp_idx]
                point_3d = aligned_points[frame_idx, kp_idx]
                if not (np.all(np.isfinite(z_uv)) and np.all(np.isfinite(point_3d))):
                    continue
                h_uv = calibration.project_point(point_3d)
                errors[frame_idx, kp_idx, cam_idx] = float(np.linalg.norm(z_uv - h_uv))
    return errors


def compute_model_reprojection_error_per_view(
    model,
    q_trajectory: np.ndarray,
    calibrations: dict[str, CameraCalibration],
    pose_data: PoseData,
) -> np.ndarray:
    marker_names = marker_name_list(model)
    marker_kp_pairs = [(marker_name, KP_INDEX[marker_name]) for marker_name in marker_names if marker_name in KP_INDEX]
    errors = np.full((pose_data.keypoints.shape[1], len(COCO17), len(pose_data.camera_names)), np.nan, dtype=float)
    n_frames = min(q_trajectory.shape[0], pose_data.keypoints.shape[1])

    for frame_idx in range(n_frames):
        q = q_trajectory[frame_idx]
        marker_positions = {name: marker.to_array() for name, marker in zip(marker_names, model.markers(q))}
        for cam_idx, cam_name in enumerate(pose_data.camera_names):
            calibration = calibrations[cam_name]
            frame_keypoints = pose_data.keypoints[cam_idx, frame_idx]
            frame_scores = pose_data.scores[cam_idx, frame_idx]
            for marker_name, kp_idx in marker_kp_pairs:
                if frame_scores[kp_idx] <= 0:
                    continue
                z_uv = frame_keypoints[kp_idx]
                point_3d = marker_positions[marker_name]
                if not (np.all(np.isfinite(z_uv)) and np.all(np.isfinite(point_3d))):
                    continue
                h_uv = calibration.project_point(point_3d)
                errors[frame_idx, kp_idx, cam_idx] = float(np.linalg.norm(z_uv - h_uv))
    return errors


def compute_model_marker_points_3d(model, q_trajectory: np.ndarray) -> np.ndarray:
    marker_names = marker_name_list(model)
    marker_kp_pairs = [(marker_name, KP_INDEX[marker_name]) for marker_name in marker_names if marker_name in KP_INDEX]
    points_3d = np.full((q_trajectory.shape[0], len(COCO17), 3), np.nan, dtype=float)
    for frame_idx, q in enumerate(q_trajectory):
        for marker_name, marker in zip(marker_names, model.markers(q)):
            kp_idx = KP_INDEX.get(marker_name)
            if kp_idx is not None:
                points_3d[frame_idx, kp_idx, :] = marker.to_array()
    return points_3d


def summarize_reprojection_errors(errors: np.ndarray, camera_names: list[str]) -> dict[str, object]:
    finite = errors[np.isfinite(errors)]
    per_key_mean = np.nanmean(errors, axis=(0, 2))
    per_key_std = np.nanstd(errors, axis=(0, 2))
    per_camera_mean = np.nanmean(errors, axis=(0, 1))
    per_camera_std = np.nanstd(errors, axis=(0, 1))
    per_keypoint = {
        keypoint_name: {
            "mean_px": float(per_key_mean[idx]) if np.isfinite(per_key_mean[idx]) else None,
            "std_px": float(per_key_std[idx]) if np.isfinite(per_key_std[idx]) else None,
            "n_samples": int(np.isfinite(errors[:, idx, :]).sum()),
        }
        for idx, keypoint_name in enumerate(COCO17)
    }
    per_camera = {
        camera_name: {
            "mean_px": float(per_camera_mean[idx]) if np.isfinite(per_camera_mean[idx]) else None,
            "std_px": float(per_camera_std[idx]) if np.isfinite(per_camera_std[idx]) else None,
        }
        for idx, camera_name in enumerate(camera_names)
    }
    return {
        "mean_px": float(np.mean(finite)) if finite.size else None,
        "std_px": float(np.std(finite)) if finite.size else None,
        "per_keypoint": per_keypoint,
        "per_camera": per_camera,
        "per_keypoint_mean": per_key_mean,
        "per_keypoint_std": per_key_std,
        "per_camera_mean": per_camera_mean,
        "per_camera_std": per_camera_std,
    }


def summarize_view_usage(excluded_views: np.ndarray | None, camera_names: list[str]) -> dict[str, object]:
    if excluded_views is None:
        return {"included_ratio": None, "excluded_ratio": None, "per_camera": {}}
    mask = np.asarray(excluded_views, dtype=bool)
    if mask.ndim != 3 or mask.shape[2] != len(camera_names):
        return {"included_ratio": None, "excluded_ratio": None, "per_camera": {}}
    included = ~mask
    per_camera: dict[str, object] = {}
    for idx, camera_name in enumerate(camera_names):
        included_ratio = float(np.mean(included[:, :, idx])) if included[:, :, idx].size else None
        excluded_ratio = float(np.mean(mask[:, :, idx])) if mask[:, :, idx].size else None
        per_camera[camera_name] = {
            "included_ratio": included_ratio,
            "excluded_ratio": excluded_ratio,
        }
    return {
        "included_ratio": float(np.mean(included)) if included.size else None,
        "excluded_ratio": float(np.mean(mask)) if mask.size else None,
        "per_camera": per_camera,
    }


def make_reconstruction_from_points(
    frames: np.ndarray,
    points_3d: np.ndarray,
    n_cameras: int,
    coherence_method: str,
) -> ReconstructionResult:
    n_frames = points_3d.shape[0]
    return ReconstructionResult(
        frames=np.asarray(frames, dtype=int),
        points_3d=np.asarray(points_3d, dtype=float),
        mean_confidence=np.full((n_frames, len(COCO17)), np.nan, dtype=float),
        reprojection_error=np.full((n_frames, len(COCO17)), np.nan, dtype=float),
        reprojection_error_per_view=np.full((n_frames, len(COCO17), n_cameras), np.nan, dtype=float),
        multiview_coherence=np.full((n_frames, len(COCO17), n_cameras), np.nan, dtype=float),
        epipolar_coherence=np.full((n_frames, len(COCO17), n_cameras), np.nan, dtype=float),
        triangulation_coherence=np.full((n_frames, len(COCO17), n_cameras), np.nan, dtype=float),
        excluded_views=np.ones((n_frames, len(COCO17), n_cameras), dtype=bool),
        coherence_method=coherence_method,
        epipolar_coherence_compute_time_s=0.0,
        triangulation_compute_time_s=0.0,
    )


def empty_pose_data() -> PoseData:
    return PoseData(
        camera_names=[],
        frames=np.array([], dtype=int),
        keypoints=np.full((0, 0, len(COCO17), 2), np.nan, dtype=float),
        scores=np.zeros((0, 0, len(COCO17)), dtype=float),
    )


def duration_from_time(time_s: np.ndarray) -> float:
    if time_s.size == 0:
        return 0.0
    return float(time_s[-1] - time_s[0]) if time_s.size > 1 else 0.0


def build_bundle_payload(
    *,
    name: str,
    family: str,
    frames: np.ndarray,
    time_s: np.ndarray,
    camera_names: list[str],
    points_3d: np.ndarray,
    q_names: np.ndarray | None,
    q: np.ndarray | None,
    qdot: np.ndarray | None,
    qddot: np.ndarray | None,
    q_root: np.ndarray,
    qdot_root: np.ndarray,
    reprojection_errors: np.ndarray,
    summary: dict[str, object],
    support_points_3d: np.ndarray | None = None,
    excluded_views: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    if q is None:
        q = np.empty((len(frames), 0), dtype=float)
    if qdot is None:
        qdot = np.empty((len(frames), 0), dtype=float)
    if qddot is None:
        qddot = np.empty((len(frames), 0), dtype=float)
    if q_names is None:
        q_names = np.array([], dtype=object)
    reprojection_stats = summarize_reprojection_errors(reprojection_errors, camera_names)
    payload = {
        "bundle_name": np.asarray(name, dtype=object),
        "family": np.asarray(family, dtype=object),
        "frames": np.asarray(frames, dtype=int),
        "time_s": np.asarray(time_s, dtype=float),
        "camera_names": np.asarray(camera_names, dtype=object),
        "points_3d": np.asarray(points_3d, dtype=float),
        "q_names": np.asarray(q_names, dtype=object),
        "q": np.asarray(q, dtype=float),
        "qdot": np.asarray(qdot, dtype=float),
        "qddot": np.asarray(qddot, dtype=float),
        "root_q_names": np.asarray(ROOT_Q_NAMES, dtype=object),
        "q_root": np.asarray(q_root, dtype=float),
        "qdot_root": np.asarray(qdot_root, dtype=float),
        "reprojection_error_per_view": np.asarray(reprojection_errors, dtype=float),
        "reprojection_error_per_keypoint_mean": np.asarray(reprojection_stats["per_keypoint_mean"], dtype=float),
        "reprojection_error_per_keypoint_std": np.asarray(reprojection_stats["per_keypoint_std"], dtype=float),
        "reprojection_error_per_camera_mean": np.asarray(reprojection_stats["per_camera_mean"], dtype=float),
        "reprojection_error_per_camera_std": np.asarray(reprojection_stats["per_camera_std"], dtype=float),
        "bundle_summary_json": np.asarray(json.dumps(summary), dtype=object),
    }
    if excluded_views is not None:
        payload["excluded_views"] = np.asarray(excluded_views, dtype=bool)
    if support_points_3d is not None:
        payload["support_points_3d"] = np.asarray(support_points_3d, dtype=float)
    return payload


def write_bundle(output_dir: Path, payload: dict[str, np.ndarray], summary: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "reconstruction_bundle.npz", **payload)
    summary_json = json.dumps(summary, indent=2)
    (output_dir / "bundle_summary.json").write_text(summary_json, encoding="utf-8")
    (output_dir / "summary.json").write_text(summary_json, encoding="utf-8")


def save_legacy_triangulation(
    output_dir: Path,
    reconstruction: ReconstructionResult,
    pose_data: PoseData,
    *,
    triangulation_method: str,
    error_threshold_px: float | None,
    min_cameras_for_triangulation: int,
    epipolar_threshold_px: float,
    pose_data_mode: str,
    pose_filter_window: int,
    pose_outlier_threshold_ratio: float,
    pose_amplitude_lower_percentile: float,
    pose_amplitude_upper_percentile: float,
) -> Path:
    triangulation_method = canonical_triangulation_method(triangulation_method)
    if triangulation_method == "once":
        cache_name = "triangulation_pose2sim_like_once.npz"
    elif triangulation_method == "greedy":
        cache_name = "triangulation_pose2sim_like_fast.npz"
    else:
        cache_name = "triangulation_pose2sim_like.npz"
    cache_path = output_dir / cache_name
    metadata = reconstruction_cache_metadata(
        pose_data,
        error_threshold_px,
        min_cameras_for_triangulation,
        epipolar_threshold_px,
        triangulation_method,
        pose_data_mode,
        pose_filter_window,
        pose_outlier_threshold_ratio,
        pose_amplitude_lower_percentile,
        pose_amplitude_upper_percentile,
    )
    save_reconstruction_cache(cache_path, reconstruction, metadata)
    return cache_path


def save_legacy_ekf_2d(
    output_dir: Path,
    result: dict[str, np.ndarray],
    predictor: str,
    flip: bool,
    marker_points_3d: np.ndarray | None = None,
) -> None:
    save_single_ekf_state(output_dir / f"ekf_states_{'flip_' if flip else ''}{predictor}.npz", result)
    key_prefix = f"q_ekf_2d_{'flip_' if flip else ''}{predictor}"
    qdot_prefix = f"qdot_ekf_2d_{'flip_' if flip else ''}{predictor}"
    qddot_prefix = f"qddot_ekf_2d_{'flip_' if flip else ''}{predictor}"
    status_prefix = f"update_status_per_frame_ekf_2d_{'flip_' if flip else ''}{predictor}"
    payload = {
        "q": result["q"],
        "qdot": result["qdot"],
        "qddot": result["qddot"],
        "q_names": result["q_names"],
        key_prefix: result["q"],
        qdot_prefix: result["qdot"],
        qddot_prefix: result["qddot"],
        status_prefix: result["update_status_per_frame"],
    }
    if marker_points_3d is not None:
        payload[f"points_3d_ekf_2d_{'flip_' if flip else ''}{predictor}"] = np.asarray(marker_points_3d, dtype=float)
    np.savez(output_dir / "ekf_states.npz", **payload)


def save_legacy_ekf_3d(
    output_dir: Path,
    result: dict[str, np.ndarray],
    reprojection_stats: dict[str, object],
    marker_points_3d: np.ndarray | None = None,
) -> None:
    np.savez(
        output_dir / "kalman_comparison.npz",
        q_ekf_3d=result["q"],
        qdot_ekf_3d=result["qdot"],
        qddot_ekf_3d=result["qddot"],
        q_biorbd_kalman=result["q"],
        qdot_biorbd_kalman=result["qdot"],
        qddot_biorbd_kalman=result["qddot"],
        q_names=result["q_names"],
        **({"points_3d_ekf_3d": np.asarray(marker_points_3d, dtype=float)} if marker_points_3d is not None else {}),
        ekf_3d_reprojection_mean_px=np.asarray(
            reprojection_stats["mean_px"] if reprojection_stats["mean_px"] is not None else np.nan, dtype=float
        ),
        ekf_3d_reprojection_std_px=np.asarray(
            reprojection_stats["std_px"] if reprojection_stats["std_px"] is not None else np.nan, dtype=float
        ),
    )


def build_pose_data(
    *,
    keypoints_path: Path,
    calibrations: dict[str, CameraCalibration],
    max_frames: int | None,
    frame_stride: int,
    pose_data_mode: str,
    pose_filter_window: int,
    pose_outlier_threshold_ratio: float,
    pose_amplitude_lower_percentile: float,
    pose_amplitude_upper_percentile: float,
    annotations_path: Path | None = None,
) -> PoseData:
    return load_pose_data(
        keypoints_path,
        calibrations,
        max_frames=max_frames,
        frame_stride=frame_stride,
        data_mode=pose_data_mode,
        smoothing_window=pose_filter_window,
        outlier_threshold_ratio=pose_outlier_threshold_ratio,
        lower_percentile=pose_amplitude_lower_percentile,
        upper_percentile=pose_amplitude_upper_percentile,
        annotations_path=annotations_path,
    )


def pose_effective_fps(pose_data: PoseData, source_fps: float) -> float:
    """Return the effective sampling rate after frame decimation."""

    frame_stride = max(1, int(getattr(pose_data, "frame_stride", 1)))
    return float(source_fps) / float(frame_stride)


def print_step(step_idx: int, step_total: int, label: str) -> None:
    print(f"[STEP {step_idx}/{step_total}] {label}", flush=True)


def build_pose2sim_bundle(
    *,
    name: str,
    output_dir: Path,
    pose2sim_trc: Path,
    calibrations: dict[str, CameraCalibration],
    pose_data: PoseData,
    pose_data_compute_time_s: float | None = None,
    fps: float,
    initial_rotation_correction: bool,
    unwrap_root: bool,
    root_unwrap_mode: str = "off",
) -> BundleBuildResult:
    print_step(2, 2, "TRC file import + export bundle")
    t0 = time.perf_counter()
    frames, time_s, points_3d, trc_rate = parse_trc_points(pose2sim_trc)
    q_root, qdot_root, correction_applied, root_kinematics_source = root_kinematics_from_trc(
        pose2sim_trc,
        frames=frames,
        time_s=time_s,
        points_3d=points_3d,
        fps=fps,
        initial_rotation_correction=initial_rotation_correction,
        unwrap_root=unwrap_root,
        root_unwrap_mode=root_unwrap_mode,
    )
    reprojection_errors = compute_points_reprojection_error_per_view(points_3d, frames, calibrations, pose_data)
    reprojection_stats = summarize_reprojection_errors(reprojection_errors, pose_data.camera_names)
    correction_angle = root_z_correction_angle_from_points(
        points_3d,
        left_hip_idx=KP_INDEX["left_hip"],
        right_hip_idx=KP_INDEX["right_hip"],
        left_shoulder_idx=KP_INDEX["left_shoulder"],
        right_shoulder_idx=KP_INDEX["right_shoulder"],
    )
    pipeline_stages = []
    if pose_data_compute_time_s is not None:
        pipeline_stages.append(
            make_timing_stage(
                "pose_data", "2D cleaning", compute_time_s=pose_data_compute_time_s, source="computed_now"
            )
        )
    pipeline_stages.append(
        make_timing_stage("pose2sim", "Load TRC 3D", compute_time_s=time.perf_counter() - t0, source="computed_now")
    )

    summary = {
        "name": name,
        "family": "pose2sim",
        "source": str(pose2sim_trc),
        "fps": float(fps),
        "trc_rate_hz": float(trc_rate),
        "root_kinematics_source": root_kinematics_source,
        "n_frames": int(len(frames)),
        "duration_s": duration_from_time(time_s),
        "initial_rotation_correction_requested": bool(initial_rotation_correction),
        "initial_rotation_correction_detected": bool(abs(correction_angle) > 1e-8),
        "initial_rotation_correction_applied": bool(correction_applied),
        "initial_rotation_correction_angle_rad": float(correction_angle),
        "reprojection_px": {
            "mean": reprojection_stats["mean_px"],
            "std": reprojection_stats["std_px"],
            "per_keypoint": reprojection_stats["per_keypoint"],
            "per_camera": reprojection_stats["per_camera"],
        },
        "stage_timings_s": {
            "total_s": time.perf_counter() - t0,
        },
        "pipeline_timing": {
            "diagram": [str(stage["id"]) for stage in pipeline_stages],
            "stages": pipeline_stages,
            "objective_total_s": float(
                sum(
                    float(stage.get("compute_time_s") or 0.0)
                    for stage in pipeline_stages
                    if bool(stage.get("include_in_total", True))
                )
            ),
            "current_run_wall_s": float((pose_data_compute_time_s or 0.0) + (time.perf_counter() - t0)),
        },
    }
    summary = with_version_info(summary, "triangulation")
    excluded_views = np.zeros((len(frames), len(COCO17), len(pose_data.camera_names)), dtype=bool)
    payload = build_bundle_payload(
        name=name,
        family="pose2sim",
        frames=frames,
        time_s=time_s,
        camera_names=pose_data.camera_names,
        points_3d=points_3d,
        q_names=np.array([], dtype=object),
        q=None,
        qdot=None,
        qddot=None,
        q_root=q_root,
        qdot_root=qdot_root,
        reprojection_errors=reprojection_errors,
        summary=summary,
        excluded_views=excluded_views,
    )
    write_bundle(output_dir, payload, summary)
    return BundleBuildResult(payload=payload, summary=summary)


def build_triangulation_bundle(
    *,
    name: str,
    output_dir: Path,
    pose_data: PoseData,
    pose_data_compute_time_s: float | None = None,
    calibrations: dict[str, CameraCalibration],
    fps: float,
    initial_rotation_correction: bool,
    unwrap_root: bool,
    root_unwrap_mode: str = "off",
    triangulation_method: str,
    reprojection_threshold_px: float | None,
    min_cameras_for_triangulation: int,
    epipolar_threshold_px: float,
    coherence_method: str,
    triangulation_workers: int,
    pose_data_mode: str,
    pose_filter_window: int,
    pose_outlier_threshold_ratio: float,
    pose_amplitude_lower_percentile: float,
    pose_amplitude_upper_percentile: float,
    flip_left_right: bool,
    flip_improvement_ratio: float,
    flip_min_gain_px: float,
    flip_min_other_cameras: int,
    flip_restrict_to_outliers: bool,
    flip_outlier_percentile: float,
    flip_outlier_floor_px: float,
    flip_temporal_weight: float,
    flip_temporal_tau_px: float,
    flip_temporal_min_valid_keypoints: int,
    flip_method: str | None = None,
) -> tuple[BundleBuildResult, ReconstructionResult]:
    triangulation_method = canonical_triangulation_method(triangulation_method)
    coherence_method = canonical_coherence_method(coherence_method, triangulation_method)
    effective_triangulation_method = triangulation_method_from_coherence_method(coherence_method, triangulation_method)
    if flip_left_right and flip_method == "ekf_prediction_gate":
        raise ValueError("flip_method=ekf_prediction_gate is only supported for ekf_2d reconstructions.")
    effective_fps = pose_effective_fps(pose_data, fps)
    pose_data_used, flip_diagnostics, pose_variant_cache_path, pose_variant_source = (
        prepare_pose_data_for_reconstruction(
            output_dir=output_dir,
            pose_data=pose_data,
            calibrations=calibrations,
            coherence_method=coherence_method,
            reprojection_threshold_px=reprojection_threshold_px,
            epipolar_threshold_px=epipolar_threshold_px,
            pose_data_mode=pose_data_mode,
            pose_filter_window=pose_filter_window,
            pose_outlier_threshold_ratio=pose_outlier_threshold_ratio,
            pose_amplitude_lower_percentile=pose_amplitude_lower_percentile,
            pose_amplitude_upper_percentile=pose_amplitude_upper_percentile,
            flip_left_right=flip_left_right,
            flip_improvement_ratio=flip_improvement_ratio,
            flip_min_gain_px=flip_min_gain_px,
            flip_min_other_cameras=flip_min_other_cameras,
            flip_restrict_to_outliers=flip_restrict_to_outliers,
            flip_outlier_percentile=flip_outlier_percentile,
            flip_outlier_floor_px=flip_outlier_floor_px,
            flip_temporal_weight=flip_temporal_weight,
            flip_temporal_tau_px=flip_temporal_tau_px,
            flip_temporal_min_valid_keypoints=flip_temporal_min_valid_keypoints,
            flip_method=flip_method,
        )
    )

    print_step(2, 3, "Triangulation 3D")
    t0 = time.perf_counter()
    reconstruction, reconstruction_cache_path, epipolar_cache_path, triangulation_source = (
        load_or_compute_triangulation_cache(
            output_dir=output_dir,
            pose_data=pose_data_used,
            calibrations=calibrations,
            coherence_method=coherence_method,
            triangulation_method=triangulation_method,
            reprojection_threshold_px=reprojection_threshold_px,
            min_cameras_for_triangulation=min_cameras_for_triangulation,
            epipolar_threshold_px=epipolar_threshold_px,
            triangulation_workers=triangulation_workers,
            pose_data_mode=pose_data_mode,
            pose_filter_window=pose_filter_window,
            pose_outlier_threshold_ratio=pose_outlier_threshold_ratio,
            pose_amplitude_lower_percentile=pose_amplitude_lower_percentile,
            pose_amplitude_upper_percentile=pose_amplitude_upper_percentile,
        )
    )
    save_legacy_triangulation(
        output_dir,
        reconstruction,
        pose_data_used,
        triangulation_method=effective_triangulation_method,
        error_threshold_px=reprojection_threshold_px,
        min_cameras_for_triangulation=min_cameras_for_triangulation,
        epipolar_threshold_px=epipolar_threshold_px,
        pose_data_mode=pose_data_mode,
        pose_filter_window=pose_filter_window,
        pose_outlier_threshold_ratio=pose_outlier_threshold_ratio,
        pose_amplitude_lower_percentile=pose_amplitude_lower_percentile,
        pose_amplitude_upper_percentile=pose_amplitude_upper_percentile,
    )
    time_s = reconstruction.frames / float(fps)
    effective_root_unwrap_mode = normalize_root_unwrap_mode(root_unwrap_mode, legacy_unwrap=unwrap_root)
    q_root, correction_applied = extract_root_from_points(
        reconstruction.points_3d,
        initial_rotation_correction,
        unwrap_rotations=(effective_root_unwrap_mode != "off"),
        unwrap_mode=effective_root_unwrap_mode,
    )
    qdot_root = centered_finite_difference(q_root, 1.0 / effective_fps)
    reprojection_errors = reconstruction.reprojection_error_per_view
    reprojection_stats = summarize_reprojection_errors(reprojection_errors, pose_data.camera_names)
    view_usage_stats = summarize_view_usage(reconstruction.excluded_views, pose_data.camera_names)
    correction_angle = root_z_correction_angle_from_points(
        reconstruction.points_3d,
        left_hip_idx=KP_INDEX["left_hip"],
        right_hip_idx=KP_INDEX["right_hip"],
        left_shoulder_idx=KP_INDEX["left_shoulder"],
        right_shoulder_idx=KP_INDEX["right_shoulder"],
    )
    print_step(3, 3, "Export bundle triangulation")
    pipeline_stages = []
    if pose_data_compute_time_s is not None:
        pipeline_stages.append(
            make_timing_stage(
                "pose_data", "2D cleaning", compute_time_s=pose_data_compute_time_s, source="computed_now"
            )
        )
    if flip_diagnostics is not None:
        flip_method_label = str(flip_diagnostics.get("method", "unknown"))
        pipeline_stages.extend(
            [
                make_timing_stage(
                    "flip_diagnostics",
                    f"Determine flipped frames ({flip_method_label})",
                    compute_time_s=flip_diagnostics.get("compute_time_s"),
                    source=pose_variant_source,
                    cache_path=str(flip_diagnostics.get("flip_cache_path") or flip_diagnostics.get("cache_path") or ""),
                ),
                make_timing_stage(
                    "flip_application",
                    "Apply flip to 2D data",
                    compute_time_s=0.0,
                    source=pose_variant_source,
                    include_in_total=False,
                ),
            ]
        )
    pipeline_stages.append(
        make_timing_stage(
            "epipolar_coherence",
            "Epipolar coherence",
            compute_time_s=reconstruction.epipolar_coherence_compute_time_s,
            source="cache" if str(triangulation_source) == "cache" else "computed_now",
            cache_path=str(epipolar_cache_path),
        )
    )
    pipeline_stages.append(
        make_timing_stage(
            "triangulation",
            f"{triangulation_method.title()} triangulation",
            compute_time_s=reconstruction.triangulation_compute_time_s,
            source=str(triangulation_source),
            cache_path=str(reconstruction_cache_path),
        )
    )

    summary = {
        "name": name,
        "family": "triangulation",
        "fps": float(effective_fps),
        "source_fps": float(fps),
        "frame_stride": int(getattr(pose_data_used, "frame_stride", 1)),
        "n_frames": int(reconstruction.frames.shape[0]),
        "duration_s": duration_from_time(time_s),
        "initial_rotation_correction_requested": bool(initial_rotation_correction),
        "initial_rotation_correction_detected": bool(abs(correction_angle) > 1e-8),
        "initial_rotation_correction_applied": bool(correction_applied),
        "initial_rotation_correction_angle_rad": float(correction_angle),
        "flip_left_right": bool(flip_left_right),
        "triangulation_method": triangulation_method,
        "reprojection_threshold_px": reprojection_threshold_px,
        "coherence_method": coherence_method,
        "pose_data_mode": pose_data_mode,
        "left_right_flip_diagnostics": flip_diagnostics,
        "cache_paths": {
            "triangulation": str(reconstruction_cache_path),
            "epipolar": str(epipolar_cache_path),
            **({"pose_2d": str(pose_variant_cache_path)} if pose_variant_cache_path is not None else {}),
        },
        "reprojection_px": {
            "mean": reprojection_stats["mean_px"],
            "std": reprojection_stats["std_px"],
            "per_keypoint": reprojection_stats["per_keypoint"],
            "per_camera": reprojection_stats["per_camera"],
        },
        "view_usage": view_usage_stats,
        "stage_timings_s": {
            "triangulation_s": time.perf_counter() - t0,
            "epipolar_coherence_s": float(reconstruction.epipolar_coherence_compute_time_s),
            "total_s": time.perf_counter() - t0,
        },
        "pipeline_timing": {
            "diagram": [str(stage["id"]) for stage in pipeline_stages],
            "stages": pipeline_stages,
            "objective_total_s": float(
                sum(
                    float(stage.get("compute_time_s") or 0.0)
                    for stage in pipeline_stages
                    if bool(stage.get("include_in_total", True))
                )
            ),
            "current_run_wall_s": float(time.perf_counter() - t0),
        },
    }
    summary = with_version_info(summary, "triangulation")
    payload = build_bundle_payload(
        name=name,
        family="triangulation",
        frames=reconstruction.frames,
        time_s=time_s,
        camera_names=pose_data.camera_names,
        points_3d=reconstruction.points_3d,
        q_names=np.array([], dtype=object),
        q=None,
        qdot=None,
        qddot=None,
        q_root=q_root,
        qdot_root=qdot_root,
        reprojection_errors=reprojection_errors,
        summary=summary,
    )
    write_bundle(output_dir, payload, summary)
    return BundleBuildResult(payload=payload, summary=summary), reconstruction


def build_ekf_3d_bundle(
    *,
    name: str,
    output_dir: Path,
    pose_data: PoseData,
    pose_data_compute_time_s: float | None = None,
    calibrations: dict[str, CameraCalibration],
    fps: float,
    initial_rotation_correction: bool,
    unwrap_root: bool,
    root_unwrap_mode: str = "off",
    triangulation_method: str,
    reprojection_threshold_px: float | None,
    min_cameras_for_triangulation: int,
    epipolar_threshold_px: float,
    coherence_method: str,
    triangulation_workers: int,
    pose_data_mode: str,
    pose_filter_window: int,
    pose_outlier_threshold_ratio: float,
    pose_amplitude_lower_percentile: float,
    pose_amplitude_upper_percentile: float,
    flip_left_right: bool,
    flip_improvement_ratio: float,
    flip_min_gain_px: float,
    flip_min_other_cameras: int,
    flip_restrict_to_outliers: bool,
    flip_outlier_percentile: float,
    flip_outlier_floor_px: float,
    flip_temporal_weight: float,
    flip_temporal_tau_px: float,
    flip_temporal_min_valid_keypoints: int,
    flip_method: str | None = None,
    subject_mass_kg: float,
    biorbd_kalman_noise_factor: float,
    biorbd_kalman_error_factor: float,
    biorbd_kalman_init_method: str = DEFAULT_BIORBD_KALMAN_INIT_METHOD,
    biomod_path: Path | None = None,
    model_variant: str = "single_trunk",
    symmetrize_limbs: bool = True,
) -> BundleBuildResult:
    triangulation_method = canonical_triangulation_method(triangulation_method)
    coherence_method = canonical_coherence_method(coherence_method, triangulation_method)
    effective_triangulation_method = triangulation_method_from_coherence_method(coherence_method, triangulation_method)
    if flip_left_right and flip_method == "ekf_prediction_gate":
        raise ValueError("flip_method=ekf_prediction_gate is only supported for ekf_2d reconstructions.")
    effective_fps = pose_effective_fps(pose_data, fps)
    pose_data_used, flip_diagnostics, pose_variant_cache_path, pose_variant_source = (
        prepare_pose_data_for_reconstruction(
            output_dir=output_dir,
            pose_data=pose_data,
            calibrations=calibrations,
            coherence_method=coherence_method,
            reprojection_threshold_px=reprojection_threshold_px,
            epipolar_threshold_px=epipolar_threshold_px,
            pose_data_mode=pose_data_mode,
            pose_filter_window=pose_filter_window,
            pose_outlier_threshold_ratio=pose_outlier_threshold_ratio,
            pose_amplitude_lower_percentile=pose_amplitude_lower_percentile,
            pose_amplitude_upper_percentile=pose_amplitude_upper_percentile,
            flip_left_right=flip_left_right,
            flip_improvement_ratio=flip_improvement_ratio,
            flip_min_gain_px=flip_min_gain_px,
            flip_min_other_cameras=flip_min_other_cameras,
            flip_restrict_to_outliers=flip_restrict_to_outliers,
            flip_outlier_percentile=flip_outlier_percentile,
            flip_outlier_floor_px=flip_outlier_floor_px,
            flip_temporal_weight=flip_temporal_weight,
            flip_temporal_tau_px=flip_temporal_tau_px,
            flip_temporal_min_valid_keypoints=flip_temporal_min_valid_keypoints,
            flip_method=flip_method,
        )
    )

    print_step(2, 5, "Triangulation 3D")
    tri_start = time.perf_counter()
    reconstruction, reconstruction_cache_path, epipolar_cache_path, triangulation_source = (
        load_or_compute_triangulation_cache(
            output_dir=output_dir,
            pose_data=pose_data_used,
            calibrations=calibrations,
            coherence_method=coherence_method,
            triangulation_method=triangulation_method,
            reprojection_threshold_px=reprojection_threshold_px,
            min_cameras_for_triangulation=min_cameras_for_triangulation,
            epipolar_threshold_px=epipolar_threshold_px,
            triangulation_workers=triangulation_workers,
            pose_data_mode=pose_data_mode,
            pose_filter_window=pose_filter_window,
            pose_outlier_threshold_ratio=pose_outlier_threshold_ratio,
            pose_amplitude_lower_percentile=pose_amplitude_lower_percentile,
            pose_amplitude_upper_percentile=pose_amplitude_upper_percentile,
        )
    )
    triangulation_s = time.perf_counter() - tri_start
    save_legacy_triangulation(
        output_dir,
        reconstruction,
        pose_data_used,
        triangulation_method=effective_triangulation_method,
        error_threshold_px=reprojection_threshold_px,
        min_cameras_for_triangulation=min_cameras_for_triangulation,
        epipolar_threshold_px=epipolar_threshold_px,
        pose_data_mode=pose_data_mode,
        pose_filter_window=pose_filter_window,
        pose_outlier_threshold_ratio=pose_outlier_threshold_ratio,
        pose_amplitude_lower_percentile=pose_amplitude_lower_percentile,
        pose_amplitude_upper_percentile=pose_amplitude_upper_percentile,
    )
    print_step(3, 5, "Model creation")
    model_start = time.perf_counter()
    selected_biomod_path = None if biomod_path is None else Path(biomod_path)
    output_biomod_path = output_dir / "vitpose_chain.bioMod"
    if selected_biomod_path is not None:
        if not selected_biomod_path.exists():
            raise FileNotFoundError(f"Selected bioMod not found: {selected_biomod_path}")
        shutil.copy2(selected_biomod_path, output_biomod_path)
        model_cache_path = selected_biomod_path
        model_compute_time_s = 0.0
        model_source = "provided"
    else:
        _lengths, biomod_cache_path, model_cache_path, _, model_compute_time_s, model_source = (
            load_or_build_model_cache(
                output_dir=output_dir,
                reconstruction=reconstruction,
                reconstruction_cache_path=reconstruction_cache_path,
                fps=effective_fps,
                subject_mass_kg=subject_mass_kg,
                initial_rotation_correction=initial_rotation_correction,
                lengths_mode="full_triangulation",
                model_variant=model_variant,
                symmetrize_limbs=symmetrize_limbs,
            )
        )
        shutil.copy2(biomod_cache_path, output_biomod_path)
    model_s = time.perf_counter() - model_start

    import biorbd

    model = biorbd.Model(str(output_biomod_path))
    print_step(4, 5, "EKF 3D")
    ekf3d_start = time.perf_counter()
    result = run_biorbd_marker_kalman_with_parameters(
        model,
        reconstruction,
        effective_fps,
        noise_factor=biorbd_kalman_noise_factor,
        error_factor=biorbd_kalman_error_factor,
        unwrap_root=unwrap_root,
        root_unwrap_mode=root_unwrap_mode,
        initial_state_method=biorbd_kalman_init_method,
    )
    ekf3d_s = time.perf_counter() - ekf3d_start
    result["q_names"] = q_names_from_model(model)
    model_points_3d = compute_model_marker_points_3d(model, result["q"])
    q_root = extract_root_from_q(
        result["q_names"],
        result["q"],
        unwrap_rotations=unwrap_root,
        unwrap_mode=root_unwrap_mode,
    )
    qdot_root = extract_root_from_q(
        result["q_names"], result["qdot"], unwrap_rotations=False, renormalize_rotations=False
    )
    correction_angle = root_z_correction_angle_from_points(
        reconstruction.points_3d,
        left_hip_idx=KP_INDEX["left_hip"],
        right_hip_idx=KP_INDEX["right_hip"],
        left_shoulder_idx=KP_INDEX["left_shoulder"],
        right_shoulder_idx=KP_INDEX["right_shoulder"],
    )
    reprojection_errors = compute_points_reprojection_error_per_view(
        model_points_3d, reconstruction.frames, calibrations, pose_data_used
    )
    reprojection_stats = summarize_reprojection_errors(reprojection_errors, pose_data_used.camera_names)
    view_usage_stats = summarize_view_usage(reconstruction.excluded_views, pose_data_used.camera_names)
    save_legacy_ekf_3d(output_dir, result, reprojection_stats, model_points_3d)
    print_step(5, 5, "Export bundle EKF 3D")

    time_s = reconstruction.frames / float(fps)
    pipeline_stages = []
    if pose_data_compute_time_s is not None:
        pipeline_stages.append(
            make_timing_stage(
                "pose_data", "2D cleaning", compute_time_s=pose_data_compute_time_s, source="computed_now"
            )
        )
    if flip_diagnostics is not None:
        flip_method_label = str(flip_diagnostics.get("method", "unknown"))
        pipeline_stages.extend(
            [
                make_timing_stage(
                    "flip_diagnostics",
                    f"Determine flipped frames ({flip_method_label})",
                    compute_time_s=flip_diagnostics.get("compute_time_s"),
                    source=pose_variant_source,
                    cache_path=str(flip_diagnostics.get("flip_cache_path") or flip_diagnostics.get("cache_path") or ""),
                ),
                make_timing_stage(
                    "flip_application",
                    "Apply flip to 2D data",
                    compute_time_s=0.0,
                    source="cache",
                    include_in_total=False,
                ),
            ]
        )
    pipeline_stages.extend(
        [
            make_timing_stage(
                "epipolar_coherence",
                "Epipolar coherence",
                compute_time_s=reconstruction.epipolar_coherence_compute_time_s,
                source="cache" if str(triangulation_source) == "cache" else "computed_now",
                cache_path=str(epipolar_cache_path),
            ),
            make_timing_stage(
                "triangulation",
                f"{effective_triangulation_method.title()} triangulation",
                compute_time_s=reconstruction.triangulation_compute_time_s,
                source=str(triangulation_source),
                cache_path=str(reconstruction_cache_path),
            ),
            make_timing_stage(
                "model_creation",
                "Model creation",
                compute_time_s=model_compute_time_s,
                source=str(model_source),
                cache_path=str(model_cache_path),
            ),
            make_timing_stage(
                "ekf_3d",
                "EKF 3D",
                compute_time_s=ekf3d_s,
                source="computed_now",
            ),
        ]
    )

    summary = {
        "name": name,
        "family": "ekf_3d",
        "fps": float(effective_fps),
        "source_fps": float(fps),
        "frame_stride": int(getattr(pose_data_used, "frame_stride", 1)),
        "n_frames": int(reconstruction.frames.shape[0]),
        "duration_s": duration_from_time(time_s),
        "initial_rotation_correction_requested": bool(initial_rotation_correction),
        "initial_rotation_correction_detected": bool(abs(correction_angle) > 1e-8),
        "initial_rotation_correction_applied": bool(initial_rotation_correction and abs(correction_angle) > 1e-8),
        "initial_rotation_correction_angle_rad": float(correction_angle),
        "pose_data_mode": pose_data_mode,
        "flip_left_right": bool(flip_left_right),
        "triangulation_method": effective_triangulation_method,
        "reprojection_threshold_px": reprojection_threshold_px,
        "coherence_method": coherence_method,
        "cache_paths": {
            "triangulation": str(reconstruction_cache_path),
            "epipolar": str(epipolar_cache_path),
            "model": str(model_cache_path),
            **({"pose_2d": str(pose_variant_cache_path)} if pose_variant_cache_path is not None else {}),
        },
        "selected_model": None if selected_biomod_path is None else str(selected_biomod_path),
        "model_variant": str(model_variant),
        "symmetrize_limbs": bool(symmetrize_limbs),
        "filter_parameters": {
            "noise_factor": float(biorbd_kalman_noise_factor),
            "error_factor": float(biorbd_kalman_error_factor),
            "initial_state_method": str(biorbd_kalman_init_method),
        },
        "biorbd_kalman_initial_state_diagnostics": result.get("initial_state_diagnostics", {}),
        "left_right_flip_diagnostics": flip_diagnostics,
        "reprojection_px": {
            "mean": reprojection_stats["mean_px"],
            "std": reprojection_stats["std_px"],
            "per_keypoint": reprojection_stats["per_keypoint"],
            "per_camera": reprojection_stats["per_camera"],
        },
        "view_usage": view_usage_stats,
        "stage_timings_s": {
            "triangulation_s": triangulation_s,
            "epipolar_coherence_s": float(reconstruction.epipolar_coherence_compute_time_s),
            "model_creation_s": model_s,
            "ekf_3d_s": ekf3d_s,
            "total_s": triangulation_s + model_s + ekf3d_s,
        },
        "pipeline_timing": {
            "diagram": [str(stage["id"]) for stage in pipeline_stages],
            "stages": pipeline_stages,
            "objective_total_s": float(
                sum(
                    float(stage.get("compute_time_s") or 0.0)
                    for stage in pipeline_stages
                    if bool(stage.get("include_in_total", True))
                )
            ),
            "current_run_wall_s": float(triangulation_s + model_s + ekf3d_s),
        },
        "points_3d_source": "model_forward_kinematics",
    }
    summary = with_version_info(summary, "ekf_3d")
    payload = build_bundle_payload(
        name=name,
        family="ekf_3d",
        frames=reconstruction.frames,
        time_s=time_s,
        camera_names=pose_data_used.camera_names,
        points_3d=model_points_3d,
        q_names=result["q_names"],
        q=result["q"],
        qdot=result["qdot"],
        qddot=result["qddot"],
        q_root=q_root,
        qdot_root=qdot_root,
        reprojection_errors=reprojection_errors,
        summary=summary,
        support_points_3d=reconstruction.points_3d,
        excluded_views=reconstruction.excluded_views,
    )
    write_bundle(output_dir, payload, summary)
    return BundleBuildResult(payload=payload, summary=summary)


def build_ekf_2d_bundle(
    *,
    name: str,
    output_dir: Path,
    pose_data: PoseData,
    pose_data_compute_time_s: float | None = None,
    calibrations: dict[str, CameraCalibration],
    fps: float,
    initial_rotation_correction: bool,
    unwrap_root: bool,
    root_unwrap_mode: str = "off",
    triangulation_method: str,
    reprojection_threshold_px: float | None,
    min_cameras_for_triangulation: int,
    epipolar_threshold_px: float,
    coherence_method: str,
    triangulation_workers: int,
    pose_data_mode: str,
    pose_filter_window: int,
    pose_outlier_threshold_ratio: float,
    pose_amplitude_lower_percentile: float,
    pose_amplitude_upper_percentile: float,
    subject_mass_kg: float,
    predictor: str,
    ekf2d_3d_source: str,
    ekf2d_initial_state_method: str,
    ekf2d_bootstrap_passes: int,
    flip_left_right: bool,
    flip_improvement_ratio: float,
    flip_min_gain_px: float,
    flip_min_other_cameras: int,
    flip_restrict_to_outliers: bool,
    flip_outlier_percentile: float,
    flip_outlier_floor_px: float,
    flip_temporal_weight: float,
    flip_temporal_tau_px: float,
    flip_temporal_min_valid_keypoints: int,
    flip_method: str | None = None,
    enable_dof_locking: bool,
    measurement_noise_scale: float,
    process_noise_scale: float,
    coherence_confidence_floor: float,
    min_frame_coherence_for_update: float,
    skip_low_coherence_updates: bool,
    flight_height_threshold_m: float,
    flight_min_consecutive_frames: int,
    upper_back_sagittal_gain: float = DEFAULT_UPPER_BACK_SAGITTAL_GAIN,
    upper_back_pseudo_std_deg: float = np.rad2deg(DEFAULT_UPPER_BACK_PSEUDO_STD_RAD),
    ankle_bed_pseudo_obs: bool = False,
    ankle_bed_pseudo_std_m: float = DEFAULT_ANKLE_BED_PSEUDO_STD_M,
    biomod_path: Path | None = None,
    model_variant: str = "single_trunk",
    symmetrize_limbs: bool = True,
) -> BundleBuildResult:
    triangulation_method = canonical_triangulation_method(triangulation_method)
    coherence_method = canonical_coherence_method(coherence_method, triangulation_method)
    support_coherence_method = support_coherence_method_for_runtime(coherence_method)
    effective_triangulation_method = triangulation_method_from_coherence_method(coherence_method, triangulation_method)
    effective_fps = pose_effective_fps(pose_data, fps)
    use_runtime_flip_gate = bool(flip_left_right and flip_method == "ekf_prediction_gate")
    if ekf2d_3d_source not in SUPPORTED_EKF2D_3D_SOURCE_MODES:
        raise ValueError(f"Unsupported ekf2d_3d_source: {ekf2d_3d_source}")
    if ekf2d_initial_state_method not in {"triangulation_ik", "ekf_bootstrap", "root_pose_bootstrap"}:
        raise ValueError(f"Unsupported ekf2d_initial_state_method: {ekf2d_initial_state_method}")
    if ekf2d_3d_source == "first_frame_only" and coherence_method not in {
        "epipolar",
        "epipolar_fast",
        "epipolar_framewise",
        "epipolar_fast_framewise",
    }:
        raise ValueError("ekf2d_3d_source=first_frame_only requires an epipolar coherence method.")

    pose_data_used, flip_diagnostics, pose_variant_cache_path, pose_variant_source = (
        prepare_pose_data_for_reconstruction(
            output_dir=output_dir,
            pose_data=pose_data,
            calibrations=calibrations,
            coherence_method=support_coherence_method,
            reprojection_threshold_px=reprojection_threshold_px,
            epipolar_threshold_px=epipolar_threshold_px,
            pose_data_mode=pose_data_mode,
            pose_filter_window=pose_filter_window,
            pose_outlier_threshold_ratio=pose_outlier_threshold_ratio,
            pose_amplitude_lower_percentile=pose_amplitude_lower_percentile,
            pose_amplitude_upper_percentile=pose_amplitude_upper_percentile,
            flip_left_right=(flip_left_right and not use_runtime_flip_gate),
            flip_improvement_ratio=flip_improvement_ratio,
            flip_min_gain_px=flip_min_gain_px,
            flip_min_other_cameras=flip_min_other_cameras,
            flip_restrict_to_outliers=flip_restrict_to_outliers,
            flip_outlier_percentile=flip_outlier_percentile,
            flip_outlier_floor_px=flip_outlier_floor_px,
            flip_temporal_weight=flip_temporal_weight,
            flip_temporal_tau_px=flip_temporal_tau_px,
            flip_temporal_min_valid_keypoints=flip_temporal_min_valid_keypoints,
            flip_method=flip_method,
        )
    )

    print_step(2, 5, "Triangulation 3D")
    tri_start = time.perf_counter()
    if ekf2d_3d_source == "full_triangulation":
        reconstruction, reconstruction_cache_path, epipolar_cache_path, triangulation_source = (
            load_or_compute_triangulation_cache(
                output_dir=output_dir,
                pose_data=pose_data_used,
                calibrations=calibrations,
                coherence_method=support_coherence_method,
                triangulation_method=effective_triangulation_method,
                reprojection_threshold_px=reprojection_threshold_px,
                min_cameras_for_triangulation=min_cameras_for_triangulation,
                epipolar_threshold_px=epipolar_threshold_px,
                triangulation_workers=triangulation_workers,
                pose_data_mode=pose_data_mode,
                pose_filter_window=pose_filter_window,
                pose_outlier_threshold_ratio=pose_outlier_threshold_ratio,
                pose_amplitude_lower_percentile=pose_amplitude_lower_percentile,
                pose_amplitude_upper_percentile=pose_amplitude_upper_percentile,
            )
        )
        if reconstruction.coherence_method != coherence_method:
            reconstruction = replace(reconstruction, coherence_method=coherence_method)
        save_legacy_triangulation(
            output_dir,
            reconstruction,
            pose_data_used,
            triangulation_method=effective_triangulation_method,
            error_threshold_px=reprojection_threshold_px,
            min_cameras_for_triangulation=min_cameras_for_triangulation,
            epipolar_threshold_px=epipolar_threshold_px,
            pose_data_mode=pose_data_mode,
            pose_filter_window=pose_filter_window,
            pose_outlier_threshold_ratio=pose_outlier_threshold_ratio,
            pose_amplitude_lower_percentile=pose_amplitude_lower_percentile,
            pose_amplitude_upper_percentile=pose_amplitude_upper_percentile,
        )
        bootstrap_frame_idx = 0
    else:
        epipolar_coherence, epipolar_time_s, epipolar_cache_path, epipolar_source = load_or_compute_epipolar_cache(
            output_dir=output_dir,
            pose_data=pose_data_used,
            calibrations=calibrations,
            coherence_method=support_coherence_method,
            epipolar_threshold_px=epipolar_threshold_px,
            pose_data_mode=pose_data_mode,
            pose_filter_window=pose_filter_window,
            pose_outlier_threshold_ratio=pose_outlier_threshold_ratio,
            pose_amplitude_lower_percentile=pose_amplitude_lower_percentile,
            pose_amplitude_upper_percentile=pose_amplitude_upper_percentile,
        )
        bootstrap_pose_data = slice_pose_data(pose_data_used, [0])
        bootstrap_reconstruction, reconstruction_cache_path, _, triangulation_source = (
            load_or_compute_triangulation_cache(
                output_dir=output_dir,
                pose_data=bootstrap_pose_data,
                calibrations=calibrations,
                coherence_method=support_coherence_method,
                triangulation_method=effective_triangulation_method,
                reprojection_threshold_px=reprojection_threshold_px,
                min_cameras_for_triangulation=min_cameras_for_triangulation,
                epipolar_threshold_px=epipolar_threshold_px,
                triangulation_workers=triangulation_workers,
                pose_data_mode=pose_data_mode,
                pose_filter_window=pose_filter_window,
                pose_outlier_threshold_ratio=pose_outlier_threshold_ratio,
                pose_amplitude_lower_percentile=pose_amplitude_lower_percentile,
                pose_amplitude_upper_percentile=pose_amplitude_upper_percentile,
            )
        )
        bootstrap_frame_idx = 0
        reconstruction = reconstruction_with_full_frame_support(
            pose_data_used,
            bootstrap_reconstruction,
            epipolar_coherence,
            epipolar_time_s,
            coherence_method=coherence_method,
            bootstrap_frame_global_idx=bootstrap_frame_idx,
        )
    triangulation_s = time.perf_counter() - tri_start

    print_step(3, 5, "Model creation")
    model_start = time.perf_counter()
    selected_biomod_path = None if biomod_path is None else Path(biomod_path)
    output_biomod_path = output_dir / "vitpose_chain.bioMod"
    if selected_biomod_path is not None:
        if not selected_biomod_path.exists():
            raise FileNotFoundError(f"Selected bioMod not found: {selected_biomod_path}")
        shutil.copy2(selected_biomod_path, output_biomod_path)
        model_cache_path = selected_biomod_path
        model_bootstrap_frame_idx = 0
        model_compute_time_s = 0.0
        model_source = "provided"
    else:
        _lengths, biomod_cache_path, model_cache_path, model_bootstrap_frame_idx, model_compute_time_s, model_source = (
            load_or_build_model_cache(
                output_dir=output_dir,
                reconstruction=reconstruction,
                reconstruction_cache_path=reconstruction_cache_path,
                fps=effective_fps,
                subject_mass_kg=subject_mass_kg,
                initial_rotation_correction=initial_rotation_correction,
                lengths_mode=ekf2d_3d_source,
                model_variant=model_variant,
                symmetrize_limbs=symmetrize_limbs,
            )
        )
        shutil.copy2(biomod_cache_path, output_biomod_path)
    model_s = time.perf_counter() - model_start

    import biorbd

    model = biorbd.Model(str(output_biomod_path))
    initial_state_start = time.perf_counter()
    initial_state, initial_state_diagnostics = compute_ekf2d_initial_state(
        model=model,
        calibrations=calibrations,
        pose_data=pose_data_used,
        reconstruction=reconstruction,
        fps=effective_fps,
        measurement_noise_scale=measurement_noise_scale,
        process_noise_scale=process_noise_scale,
        min_frame_coherence_for_update=min_frame_coherence_for_update,
        skip_low_coherence_updates=skip_low_coherence_updates,
        coherence_confidence_floor=coherence_confidence_floor,
        epipolar_threshold_px=epipolar_threshold_px,
        enable_dof_locking=enable_dof_locking,
        method=ekf2d_initial_state_method,
        bootstrap_passes=ekf2d_bootstrap_passes,
        flip_method=("ekf_prediction_gate" if use_runtime_flip_gate else None),
        flip_improvement_ratio=flip_improvement_ratio,
        flip_min_gain_px=flip_min_gain_px,
        flip_error_threshold_px=epipolar_threshold_px,
        flip_error_delta_threshold_px=flip_min_gain_px,
        upper_back_sagittal_gain=upper_back_sagittal_gain,
        upper_back_pseudo_std_rad=np.deg2rad(float(upper_back_pseudo_std_deg)),
        ankle_bed_pseudo_obs=ankle_bed_pseudo_obs,
        ankle_bed_pseudo_std_m=ankle_bed_pseudo_std_m,
    )
    initial_state_s = time.perf_counter() - initial_state_start
    print_step(4, 5, f"EKF 2D {predictor.upper()}")
    ekf_start = time.perf_counter()
    result, ekf_timings = run_ekf(
        biomod_path=None,
        calibrations=calibrations,
        pose_data=pose_data_used,
        reconstruction=reconstruction,
        fps=effective_fps,
        measurement_noise_scale=measurement_noise_scale,
        process_noise_scale=process_noise_scale,
        min_frame_coherence_for_update=min_frame_coherence_for_update,
        skip_low_coherence_updates=skip_low_coherence_updates,
        coherence_confidence_floor=coherence_confidence_floor,
        epipolar_threshold_px=epipolar_threshold_px,
        enable_dof_locking=enable_dof_locking,
        root_flight_dynamics=(predictor in {"dyn", "dyn_history3"}),
        predictor_mode=predictor,
        flight_height_threshold_m=flight_height_threshold_m,
        flight_min_consecutive_frames=flight_min_consecutive_frames,
        unwrap_root=unwrap_root,
        root_unwrap_mode=root_unwrap_mode,
        model=model,
        initial_state=initial_state,
        flip_method=("ekf_prediction_gate" if use_runtime_flip_gate else None),
        flip_improvement_ratio=flip_improvement_ratio,
        flip_min_gain_px=flip_min_gain_px,
        flip_error_threshold_px=epipolar_threshold_px,
        flip_error_delta_threshold_px=flip_min_gain_px,
        upper_back_sagittal_gain=upper_back_sagittal_gain,
        upper_back_pseudo_std_rad=np.deg2rad(float(upper_back_pseudo_std_deg)),
        ankle_bed_pseudo_obs=ankle_bed_pseudo_obs,
        ankle_bed_pseudo_std_m=ankle_bed_pseudo_std_m,
    )
    ekf_s = time.perf_counter() - ekf_start
    model_points_3d = compute_model_marker_points_3d(model, result["q"])
    q_root = extract_root_from_q(
        result["q_names"],
        result["q"],
        unwrap_rotations=unwrap_root,
        unwrap_mode=root_unwrap_mode,
    )
    qdot_root = extract_root_from_q(
        result["q_names"], result["qdot"], unwrap_rotations=False, renormalize_rotations=False
    )
    correction_angle = root_z_correction_angle_from_points(
        reconstruction.points_3d,
        left_hip_idx=KP_INDEX["left_hip"],
        right_hip_idx=KP_INDEX["right_hip"],
        left_shoulder_idx=KP_INDEX["left_shoulder"],
        right_shoulder_idx=KP_INDEX["right_shoulder"],
    )
    reprojection_errors = compute_points_reprojection_error_per_view(
        model_points_3d, reconstruction.frames, calibrations, pose_data_used
    )
    reprojection_stats = summarize_reprojection_errors(reprojection_errors, pose_data_used.camera_names)
    view_usage_stats = summarize_view_usage(reconstruction.excluded_views, pose_data_used.camera_names)
    save_legacy_ekf_2d(output_dir, result, predictor, flip_left_right, model_points_3d)
    print_step(5, 5, "Export bundle EKF 2D")

    if use_runtime_flip_gate:
        flip_diagnostics = dict(result.get("flip_diagnostics") or {})

    time_s = reconstruction.frames / float(fps)
    pipeline_stages = []
    if pose_data_compute_time_s is not None:
        pipeline_stages.append(
            make_timing_stage(
                "pose_data", "2D cleaning", compute_time_s=pose_data_compute_time_s, source="computed_now"
            )
        )
    if flip_diagnostics is not None:
        flip_method_label = str(flip_diagnostics.get("method", "unknown"))
        pipeline_stages.extend(
            [
                make_timing_stage(
                    "flip_diagnostics",
                    f"Determine flipped frames ({flip_method_label})",
                    compute_time_s=flip_diagnostics.get("compute_time_s"),
                    source=pose_variant_source,
                    cache_path=str(flip_diagnostics.get("flip_cache_path") or flip_diagnostics.get("cache_path") or ""),
                ),
                make_timing_stage(
                    "flip_application",
                    "Apply flip to 2D data",
                    compute_time_s=0.0,
                    source="cache",
                    include_in_total=False,
                ),
            ]
        )
    if ekf2d_3d_source == "first_frame_only":
        pipeline_stages.append(
            make_timing_stage(
                "epipolar_coherence",
                "Epipolar coherence",
                compute_time_s=float(epipolar_time_s),
                source="cache" if str(triangulation_source) == "cache" else "computed_now",
                cache_path=str(epipolar_cache_path),
            )
        )
        pipeline_stages.append(
            make_timing_stage(
                "triangulation",
                f"Bootstrap {effective_triangulation_method.title()} triangulation",
                compute_time_s=bootstrap_reconstruction.triangulation_compute_time_s,
                source=str(triangulation_source),
                cache_path=str(reconstruction_cache_path),
            )
        )
    else:
        pipeline_stages.append(
            make_timing_stage(
                "epipolar_coherence",
                "Epipolar coherence",
                compute_time_s=reconstruction.epipolar_coherence_compute_time_s,
                source="cache" if str(triangulation_source) == "cache" else "computed_now",
                cache_path=str(epipolar_cache_path),
            )
        )
        pipeline_stages.append(
            make_timing_stage(
                "triangulation",
                f"{effective_triangulation_method.title()} triangulation",
                compute_time_s=reconstruction.triangulation_compute_time_s,
                source=str(triangulation_source),
                cache_path=str(reconstruction_cache_path),
            )
        )
    pipeline_stages.extend(
        [
            make_timing_stage(
                "model_creation",
                "Model creation",
                compute_time_s=model_compute_time_s,
                source=str(model_source),
                cache_path=str(model_cache_path),
            ),
            make_timing_stage(
                "ekf_2d_initial_state",
                "EKF 2D initial state",
                compute_time_s=initial_state_s,
                source="computed_now",
            ),
            make_timing_stage(
                "ekf_2d",
                "EKF 2D",
                compute_time_s=ekf_s,
                source="computed_now",
            ),
            make_timing_stage(
                "ekf_2d_init",
                "EKF 2D init",
                compute_time_s=float(ekf_timings["init_s"]),
                source="computed_now",
                include_in_total=False,
            ),
            make_timing_stage(
                "ekf_2d_loop",
                "EKF 2D loop",
                compute_time_s=float(ekf_timings["loop_s"]),
                source="computed_now",
                include_in_total=False,
            ),
            make_timing_stage(
                "ekf_2d_predict",
                "EKF 2D predict",
                compute_time_s=float(ekf_timings.get("predict_s", 0.0)),
                source="computed_now",
                include_in_total=False,
            ),
            make_timing_stage(
                "ekf_2d_update",
                "EKF 2D update",
                compute_time_s=float(ekf_timings.get("update_s", 0.0)),
                source="computed_now",
                include_in_total=False,
            ),
            make_timing_stage(
                "ekf_prediction_gate",
                "EKF prediction-gated L/R flip",
                compute_time_s=float(ekf_timings.get("flip_gate_s", 0.0)),
                source="computed_now",
                include_in_total=False,
            ),
            make_timing_stage(
                "ekf_2d_markers",
                "EKF 2D markers",
                compute_time_s=float(ekf_timings.get("markers_s", 0.0)),
                source="computed_now",
                include_in_total=False,
            ),
            make_timing_stage(
                "ekf_2d_marker_jacobians",
                "EKF 2D marker jacobians",
                compute_time_s=float(ekf_timings.get("marker_jacobians_s", 0.0)),
                source="computed_now",
                include_in_total=False,
            ),
            make_timing_stage(
                "ekf_2d_assembly",
                "EKF 2D assembly",
                compute_time_s=float(ekf_timings.get("assembly_s", 0.0)),
                source="computed_now",
                include_in_total=False,
            ),
            make_timing_stage(
                "ekf_2d_solve",
                "EKF 2D solve",
                compute_time_s=float(ekf_timings.get("solve_s", 0.0)),
                source="computed_now",
                include_in_total=False,
            ),
        ]
    )

    summary = {
        "name": name,
        "family": "ekf_2d",
        "fps": float(effective_fps),
        "source_fps": float(fps),
        "frame_stride": int(getattr(pose_data_used, "frame_stride", 1)),
        "n_frames": int(reconstruction.frames.shape[0]),
        "duration_s": duration_from_time(time_s),
        "initial_rotation_correction_requested": bool(initial_rotation_correction),
        "initial_rotation_correction_detected": bool(abs(correction_angle) > 1e-8),
        "initial_rotation_correction_applied": bool(initial_rotation_correction and abs(correction_angle) > 1e-8),
        "initial_rotation_correction_angle_rad": float(correction_angle),
        "pose_data_mode": pose_data_mode,
        "triangulation_method": effective_triangulation_method,
        "reprojection_threshold_px": reprojection_threshold_px,
        "coherence_method": coherence_method,
        "ekf2d_3d_source": ekf2d_3d_source,
        "ekf2d_initial_state_method": ekf2d_initial_state_method,
        "ekf2d_bootstrap_passes": int(ekf2d_bootstrap_passes),
        "ekf2d_initial_state_diagnostics": initial_state_diagnostics,
        "predictor": predictor,
        "flip_left_right": bool(flip_left_right),
        "dof_locking": bool(enable_dof_locking),
        "cache_paths": {
            "triangulation": str(reconstruction_cache_path),
            "epipolar": str(epipolar_cache_path),
            "model": str(model_cache_path),
            **({"pose_2d": str(pose_variant_cache_path)} if pose_variant_cache_path is not None else {}),
        },
        "selected_model": None if selected_biomod_path is None else str(selected_biomod_path),
        "model_variant": str(model_variant),
        "symmetrize_limbs": bool(symmetrize_limbs),
        "filter_parameters": {
            "measurement_noise_scale": float(measurement_noise_scale),
            "process_noise_scale": float(process_noise_scale),
            "coherence_confidence_floor": float(coherence_confidence_floor),
            "upper_back_sagittal_gain": float(upper_back_sagittal_gain),
            "upper_back_pseudo_std_deg": float(upper_back_pseudo_std_deg),
            "ankle_bed_pseudo_obs": bool(ankle_bed_pseudo_obs),
            "ankle_bed_pseudo_std_m": float(ankle_bed_pseudo_std_m),
            "min_frame_coherence_for_update": float(min_frame_coherence_for_update),
            "skip_low_coherence_updates": bool(skip_low_coherence_updates),
            "flight_height_threshold_m": float(flight_height_threshold_m),
            "flight_min_consecutive_frames": int(flight_min_consecutive_frames),
            "flip_method": str(flip_method or "epipolar"),
        },
        "bootstrap_frame_idx": int(
            model_bootstrap_frame_idx if ekf2d_3d_source == "first_frame_only" else bootstrap_frame_idx
        ),
        "update_status_counts": {str(key): int(value) for key, value in result.get("update_status_counts", {}).items()},
        "left_right_flip_diagnostics": flip_diagnostics,
        "reprojection_px": {
            "mean": reprojection_stats["mean_px"],
            "std": reprojection_stats["std_px"],
            "per_keypoint": reprojection_stats["per_keypoint"],
            "per_camera": reprojection_stats["per_camera"],
        },
        "view_usage": view_usage_stats,
        "stage_timings_s": {
            "triangulation_s": triangulation_s,
            "epipolar_coherence_s": float(reconstruction.epipolar_coherence_compute_time_s),
            "model_creation_s": model_s,
            "ekf_2d_initial_state_s": float(initial_state_s),
            "ekf_2d_init_s": float(ekf_timings["init_s"]),
            "ekf_2d_loop_s": float(ekf_timings["loop_s"]),
            "ekf_2d_s": ekf_s,
            "ekf_2d_predict_s": float(ekf_timings.get("predict_s", 0.0)),
            "ekf_2d_update_s": float(ekf_timings.get("update_s", 0.0)),
            "ekf_prediction_gate_s": float(ekf_timings.get("flip_gate_s", 0.0)),
            "ekf_2d_markers_s": float(ekf_timings.get("markers_s", 0.0)),
            "ekf_2d_marker_jacobians_s": float(ekf_timings.get("marker_jacobians_s", 0.0)),
            "ekf_2d_assembly_s": float(ekf_timings.get("assembly_s", 0.0)),
            "ekf_2d_solve_s": float(ekf_timings.get("solve_s", 0.0)),
            "total_s": triangulation_s + model_s + initial_state_s + ekf_s,
        },
        "pipeline_timing": {
            "diagram": [
                str(stage["id"])
                for stage in pipeline_stages
                if str(stage["id"])
                not in {
                    "ekf_2d_init",
                    "ekf_2d_loop",
                    "ekf_2d_predict",
                    "ekf_2d_update",
                    "ekf_2d_markers",
                    "ekf_2d_marker_jacobians",
                    "ekf_2d_assembly",
                    "ekf_2d_solve",
                }
            ],
            "stages": pipeline_stages,
            "objective_total_s": float(
                sum(
                    float(stage.get("compute_time_s") or 0.0)
                    for stage in pipeline_stages
                    if bool(stage.get("include_in_total", True))
                )
            ),
            "current_run_wall_s": float(triangulation_s + model_s + initial_state_s + ekf_s),
        },
        "points_3d_source": "model_forward_kinematics",
    }
    summary = with_version_info(summary, "ekf_2d")
    payload = build_bundle_payload(
        name=name,
        family="ekf_2d",
        frames=reconstruction.frames,
        time_s=time_s,
        camera_names=pose_data_used.camera_names,
        points_3d=model_points_3d,
        q_names=result["q_names"],
        q=result["q"],
        qdot=result["qdot"],
        qddot=result["qddot"],
        q_root=q_root,
        qdot_root=qdot_root,
        reprojection_errors=reprojection_errors,
        summary=summary,
        support_points_3d=reconstruction.points_3d,
        excluded_views=reconstruction.excluded_views,
    )
    write_bundle(output_dir, payload, summary)
    return BundleBuildResult(payload=payload, summary=summary)
