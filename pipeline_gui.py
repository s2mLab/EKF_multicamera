#!/usr/bin/env python3
"""Interface graphique légère pour piloter les scripts du projet.

L'objectif est pragmatique:
- éviter de réécrire les longues lignes de commande,
- garder les options courantes visibles,
- laisser un champ `extra args` pour les cas plus spécifiques.

Le script utilise `tkinter`, donc aucune dépendance supplémentaire n'est
nécessaire dans l'environnement courant.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

ROOT = Path(__file__).resolve().parent
LOCAL_CACHE = ROOT / ".cache"
LOCAL_MPLCONFIG = LOCAL_CACHE / "matplotlib"
LOCAL_MPLCONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_MPLCONFIG))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE))

DEFAULT_GUI_CALIB_PATH = "inputs/calibration/Calib.toml"
DEFAULT_GUI_KEYPOINTS_PATH = "inputs/keypoints/1_partie_0429_keypoints.json"
DEFAULT_GUI_TRC_PATH = "inputs/trc/1_partie_0429.trc"
DEFAULT_GUI_PROFILES_CONFIG = "reconstruction_profiles.json"

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation

from annotation.annotation_store import (
    apply_annotations_to_pose_arrays,
    clear_annotation_point,
    default_annotation_path,
    empty_annotation_payload,
    get_annotation_point,
    load_annotation_payload,
    save_annotation_payload,
    set_annotation_point,
)
from annotation.frame_navigation import (
    clamp_index_to_subset,
    fallback_annotation_filtered_indices,
    navigable_annotation_frame_local_indices,
    resolve_annotation_frame_filter_mode,
    step_frame_index_within_subset,
)
from annotation.kinematic_assist import (
    annotation_blend_q_by_relevance,
    annotation_reconstruction_from_points,
    annotation_state_from_q,
    propagate_annotation_kinematic_state,
    refine_annotation_q_with_direct_measurements,
    refine_annotation_q_with_local_ekf,
    refine_annotation_window_states,
    resolve_annotation_kinematic_state_info,
    store_annotation_kinematic_state,
)
from annotation.preview_render import (
    annotation_frame_label_text,
    render_annotation_camera_view,
)
from batch_run import (
    DEFAULT_EXCEL_OUTPUT,
    DEFAULT_KEYPOINTS_GLOB,
)
from batch_run import discover_keypoints_files as batch_discover_keypoints_files
from batch_run import infer_annotations_for_keypoints as batch_infer_annotations_for_keypoints
from batch_run import infer_pose2sim_trc_for_keypoints as batch_infer_pose2sim_trc_for_keypoints
from calibration_qc import compute_calibration_qc, frame_camera_epipolar_errors
from camera_tools.camera_metrics import compute_camera_metric_rows, suggest_best_camera_names
from camera_tools.camera_selection import format_camera_names, parse_camera_names
from judging.dd_analysis import DDSessionAnalysis, contiguous_true_regions
from judging.dd_presenter import (
    build_jump_plot_data,
    compare_dd_code_characters,
    compare_dd_to_reference,
    dd_reference_status_color,
    dd_reference_status_text,
    format_dd_summary,
    format_detected_dd_codes_with_inline_errors,
    jump_list_label_with_reference,
)
from judging.dd_reference import default_dd_reference_path, load_dd_reference_codes
from judging.execution import (
    ExecutionDeductionEvent,
    ExecutionJumpAnalysis,
    ExecutionOverlayFrame,
    ExecutionSessionAnalysis,
    analyze_execution_session,
    available_execution_image_frames,
    build_execution_overlay_frame,
    execution_focus_frame,
    infer_execution_images_root,
    resolve_execution_image_path,
)
from judging.jump_cache import get_cached_jump_analysis
from judging.jump_cache import jump_segmentation_height_series as shared_jump_segmentation_height_series
from judging.trampoline_displacement import (
    BED_X_MAX,
    BED_Y_MAX,
    TRAMPOLINE_BED_HEIGHT_M,
    TRAMPOLINE_GEOMETRY,
    X_INNER,
    X_MAX,
    Y_INNER,
    Y_MAX,
    analyze_trampoline_contacts,
    judged_trampoline_zone_xy,
    total_trampoline_penalty,
    trampoline_penalty_refined,
)
from kinematics.analysis_3d import (
    SEGMENT_LENGTH_DEFINITIONS,
    angular_momentum_plot_data,
    segment_length_series,
    valid_segment_length_samples,
)
from kinematics.root_kinematics import (
    ROOT_Q_NAMES,
    TRUNK_ROOT_ROTATION_SEQUENCE,
    TRUNK_ROTATION_NAMES,
    TRUNK_TRANSLATION_NAMES,
    centered_finite_difference,
    compute_trunk_dofs_from_points,
    extract_root_from_q,
    normalize,
    normalize_root_unwrap_mode,
    rotation_unit_label,
    rotation_unit_scale,
)
from kinematics.root_series import (
    interpolate_trunk_marker_gaps_for_root,
    quantity_unit_label,
    root_axis_display_labels,
    root_ordered_names,
    root_rotation_matrices_from_points,
    root_rotation_matrices_from_series,
    root_series_from_model_markers,
    root_series_from_points,
    root_series_from_precomputed,
    root_series_from_q,
    scale_root_series_rotations,
)
from observability.observability_analysis import compute_observability_rank_series, summarize_rank_series
from preview.dataset_preview_loader import load_dataset_preview_resources
from preview.dataset_preview_state import DatasetPreviewState, build_dataset_preview_state
from preview.frame_2d_render import (
    PointValueOverlay2D,
    SkeletonLayer2D,
    draw_point_value_overlay,
    render_camera_frame_2d,
)
from preview.preview_bundle import (
    align_to_reference,
    load_dataset_preview_bundle,
    project_points_all_cameras,
    root_center,
)
from preview.preview_navigation import clamp_frame_index, frame_from_slider_click, step_frame_index
from preview.shared_reconstruction_panel import SharedReconstructionPanel, show_placeholder_figure
from preview.two_d_view import (
    adjust_image_levels,
    apply_2d_axis_limits,
    camera_layout,
    compute_pose_crop_limits_2d,
    crop_limits_from_points,
    draw_2d_background_image,
    hide_2d_axes,
    load_camera_background_image,
    square_crop_bounds,
)
from reconstruction.reconstruction_bundle import (
    extract_root_from_points,
    load_or_compute_left_right_flip_cache,
    load_or_compute_pose_data_variant_cache,
    load_or_compute_triangulation_cache,
)
from reconstruction.reconstruction_bundle import parse_trc_points as parse_reconstruction_trc_points
from reconstruction.reconstruction_bundle import (
    slice_pose_data,
)
from reconstruction.reconstruction_dataset import (
    dataset_source_paths,
    reconstruction_color,
    reconstruction_label,
    write_trc_file,
    write_trc_root_kinematics_sidecar,
)
from reconstruction.reconstruction_presenter import (
    bundle_available_reconstruction_names,
    catalog_rows_for_names,
    default_selection,
)
from reconstruction.reconstruction_profiles import (
    SUPPORTED_COHERENCE_METHODS,
    SUPPORTED_FLIP_METHODS,
    SUPPORTED_MODEL_VARIANTS,
    SUPPORTED_TRIANGULATION_METHODS,
    ReconstructionProfile,
    build_pipeline_command,
    canonical_profile_name,
    example_profiles,
    generate_supported_profiles,
    load_profiles_json,
    profile_to_dict,
    save_profiles_json,
    validate_profile,
)
from reconstruction.reconstruction_registry import (
    DEFAULT_MODEL_SYMMETRIZE_LIMBS,
    dataset_figures_dir,
    dataset_models_dir,
    dataset_reconstructions_dir,
    default_model_stem,
    infer_dataset_name,
    latest_version_for_family,
    model_biomod_path,
    model_output_dir,
    normalize_output_root,
    scan_model_dirs,
    scan_reconstruction_dirs,
)
from reconstruction.reconstruction_timings import (
    format_reconstruction_timing_details,
    model_compute_seconds,
    objective_total_seconds,
    reconstruction_run_seconds,
)
from vitpose_ekf_pipeline import (
    COCO17,
    DEFAULT_COHERENCE_METHOD,
    DEFAULT_EPIPOLAR_THRESHOLD_PX,
    DEFAULT_FLIP_IMPROVEMENT_RATIO,
    DEFAULT_FLIP_MIN_GAIN_PX,
    DEFAULT_FLIP_MIN_OTHER_CAMERAS,
    DEFAULT_FLIP_OUTLIER_FLOOR_PX,
    DEFAULT_FLIP_OUTLIER_PERCENTILE,
    DEFAULT_FLIP_RESTRICT_TO_OUTLIERS,
    DEFAULT_FLIP_TEMPORAL_TAU_PX,
    DEFAULT_FLIP_TEMPORAL_WEIGHT,
    DEFAULT_MIN_CAMERAS_FOR_TRIANGULATION,
    DEFAULT_MODEL_VARIANT,
    DEFAULT_REPROJECTION_THRESHOLD_PX,
    PoseData,
    ReconstructionResult,
    apply_left_right_flip_to_points,
    canonicalize_model_q_rotation_branches,
    fundamental_matrix,
    initial_state_from_triangulation,
    load_calibrations,
    load_pose_data,
    load_reconstruction_cache,
    metadata_cache_matches,
    reconstruction_cache_metadata,
    swap_left_right_keypoints,
    triangulate_pose2sim_like,
    weighted_triangulation,
)

SKELETON_EDGES = [
    ("left_ear", "left_eye"),
    ("left_eye", "nose"),
    ("nose", "right_eye"),
    ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]
KP_INDEX = {name: idx for idx, name in enumerate(COCO17)}
LEFT_KEYPOINTS = {
    "left_eye",
    "left_ear",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hip",
    "left_knee",
    "left_ankle",
}
RIGHT_KEYPOINTS = {
    "right_eye",
    "right_ear",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
}
FACE_KEYPOINTS = {
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
}
BODY_ONLY_KEYPOINTS = tuple(name for name in COCO17 if name not in FACE_KEYPOINTS)
ANNOTATION_KEYPOINT_ORDER = (
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
)
SUPPORTED_MODEL_PREVIEW_VIEWERS = ("matplotlib", "pyorerun")
LOWER_LIMB_EDGES = {
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
}
RECONSTRUCTION_ORDER = [
    "pose2sim",
    "triangulation_once",
    "triangulation_adaptive",
    "triangulation_fast",
    "ekf_2d_acc",
    "ekf_2d_flip_acc",
    "ekf_2d_dyn",
    "ekf_2d_flip_dyn",
    "ekf_3d_flip",
    "ekf_3d",
]
RECONSTRUCTION_LABELS = {
    "pose2sim": "TRC file",
    "triangulation_once": "Triangulation once",
    "triangulation_adaptive": "Triangulation adaptive",
    "triangulation_fast": "Triangulation fast",
    "ekf_2d_acc": "EKF 2D ACC",
    "ekf_2d_flip_acc": "EKF 2D FLIP ACC",
    "ekf_2d_dyn": "EKF 2D DYN",
    "ekf_2d_flip_dyn": "EKF 2D FLIP DYN",
    "ekf_3d_flip": "EKF 3D FLIP",
    "ekf_3d": "EKF 3D",
}
RECONSTRUCTION_COLORS = {
    "pose2sim": "black",
    "triangulation_once": "#8c8c8c",
    "triangulation_adaptive": "#dd8452",
    "triangulation_fast": "#f2a104",
    "ekf_2d_acc": "#c44e52",
    "ekf_2d_flip_acc": "#937860",
    "ekf_2d_dyn": "#8172b3",
    "ekf_2d_flip_dyn": "#da8bc3",
    "ekf_3d_flip": "#4c9f70",
    "ekf_3d": "#55a868",
}
STEP_RE = re.compile(r"\[STEP\s+(\d+)/(\d+)\]\s*(.*)")
PROFILE_RE = re.compile(r"\[PROFILE\s+(\d+)/(\d+)\]\s*(.*)")
ANALYSIS_START_FRAME = 10


def gui_debug(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[GUI {timestamp}] {message}", flush=True)


def shared_reconstruction_order(state: "SharedAppState") -> list[str]:
    """Return the current shared reconstruction order used across analysis tabs."""

    panel = getattr(state, "shared_reconstruction_panel", None)
    if panel is not None:
        names = [str(item) for item in panel.tree.get_children("") if str(item) != "__placeholder__"]
        if names:
            return names
    return [str(name) for name in getattr(state, "shared_reconstruction_selection", [])]


def reconstruction_legend_label(state: "SharedAppState", name: str) -> str:
    """Return a compact numeric label for one reconstruction in plot legends."""

    ordered_names = shared_reconstruction_order(state)
    if name in ordered_names:
        return str(ordered_names.index(name) + 1)
    return reconstruction_label(name)


def reconstruction_display_color(state: "SharedAppState", name: str) -> str:
    """Return a color driven by the shared reconstruction order when available."""

    ordered_names = shared_reconstruction_order(state)
    if name in ordered_names:
        palette = [
            "#4c72b0",
            "#dd8452",
            "#55a868",
            "#c44e52",
            "#8172b3",
            "#937860",
            "#da8bc3",
            "#8c8c8c",
            "#64b5cd",
            "#ccb974",
        ]
        return palette[ordered_names.index(name) % len(palette)]
    return reconstruction_color(name)


def load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def finite_mean_std(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None, None
    return float(np.mean(finite)), float(np.std(finite))


def keypoint_preset_names(preset: str) -> list[str]:
    """Return the keypoint names covered by one 2D-explorer selection preset."""

    normalized = str(preset).strip().lower()
    if normalized == "none":
        return []
    if normalized == "body_only":
        return list(BODY_ONLY_KEYPOINTS)
    return list(COCO17)


def analysis_frame_slice(frame_count: int, start_frame: int = ANALYSIS_START_FRAME) -> slice:
    """Return the common analysis window used across GUI analysis tabs."""

    frame_count = max(0, int(frame_count))
    start = min(max(0, int(start_frame)), frame_count)
    return slice(start, frame_count)


def slice_analysis_series(values: np.ndarray, start_frame: int = ANALYSIS_START_FRAME) -> np.ndarray:
    """Trim a frame-based array to the shared analysis window."""

    array = np.asarray(values)
    return array[analysis_frame_slice(array.shape[0], start_frame=start_frame)]


def plt_colormap(n: int) -> list[tuple[float, float, float, float]]:
    return list(plt.cm.tab20(np.linspace(0, 1, max(n, 1))))


def compute_triangulation_reprojection_stats(npz_path: Path) -> tuple[float | None, float | None]:
    if not npz_path.exists():
        return None, None
    data = np.load(npz_path, allow_pickle=True)
    if "reprojection_error" not in data:
        return None, None
    return finite_mean_std(np.asarray(data["reprojection_error"], dtype=float))


def load_bundle_summary(output_dir: Path) -> dict[str, object]:
    return load_json_if_exists(output_dir / "bundle_summary.json")


def load_bundle_payload(output_dir: Path) -> dict[str, np.ndarray]:
    bundle_path = output_dir / "reconstruction_bundle.npz"
    if not bundle_path.exists():
        return {}
    with np.load(bundle_path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def load_flip_detail_arrays(cache_path: Path) -> dict[str, np.ndarray]:
    if not cache_path.exists():
        return {}
    with np.load(cache_path, allow_pickle=True) as data:
        return {
            key: np.asarray(data[key])
            for key in data.files
            if key
            not in {
                "suspect_mask",
                "diagnostics",
                "compute_time_s",
                "metadata",
            }
        }


def reconstruction_dirs_for_path(path: Path) -> list[Path]:
    if (path / "bundle_summary.json").exists():
        return [path]
    return scan_reconstruction_dirs(path)


def load_q_names(ekf_states_path: Path) -> np.ndarray:
    if not ekf_states_path.exists():
        return np.array([], dtype=object)
    data = np.load(ekf_states_path, allow_pickle=True)
    return np.asarray(data["q_names"]) if "q_names" in data else np.array([], dtype=object)


def read_q_variant(
    ekf_states_path: Path, kalman_comparison_path: Path, variant: str
) -> tuple[np.ndarray | None, np.ndarray | None]:
    ekf = np.load(ekf_states_path, allow_pickle=True) if ekf_states_path.exists() else None
    comparison = np.load(kalman_comparison_path, allow_pickle=True) if kalman_comparison_path.exists() else None
    if variant == "ekf_2d_acc" and ekf is not None:
        q = (
            np.asarray(ekf["q_ekf_2d_acc"], dtype=float)
            if "q_ekf_2d_acc" in ekf
            else (np.asarray(ekf["q"], dtype=float) if "q" in ekf else None)
        )
        qdot = (
            np.asarray(ekf["qdot_ekf_2d_acc"], dtype=float)
            if "qdot_ekf_2d_acc" in ekf
            else (np.asarray(ekf["qdot"], dtype=float) if "qdot" in ekf else None)
        )
        return q, qdot
    if variant == "ekf_2d_flip_acc" and ekf is not None and "q_ekf_2d_flip_acc" in ekf:
        q = np.asarray(ekf["q_ekf_2d_flip_acc"], dtype=float)
        qdot = np.asarray(ekf["qdot_ekf_2d_flip_acc"], dtype=float) if "qdot_ekf_2d_flip_acc" in ekf else None
        return q, qdot
    if variant == "ekf_2d_dyn" and ekf is not None and "q_ekf_2d_dyn" in ekf:
        q = np.asarray(ekf["q_ekf_2d_dyn"], dtype=float)
        qdot = np.asarray(ekf["qdot_ekf_2d_dyn"], dtype=float) if "qdot_ekf_2d_dyn" in ekf else None
        return q, qdot
    if variant == "ekf_2d_flip_dyn" and ekf is not None and "q_ekf_2d_flip_dyn" in ekf:
        q = np.asarray(ekf["q_ekf_2d_flip_dyn"], dtype=float)
        qdot = np.asarray(ekf["qdot_ekf_2d_flip_dyn"], dtype=float) if "qdot_ekf_2d_flip_dyn" in ekf else None
        return q, qdot
    if variant == "ekf_3d" and comparison is not None:
        q = (
            np.asarray(comparison["q_ekf_3d"], dtype=float)
            if "q_ekf_3d" in comparison
            else (np.asarray(comparison["q_biorbd_kalman"], dtype=float) if "q_biorbd_kalman" in comparison else None)
        )
        qdot = (
            np.asarray(comparison["qdot_ekf_3d"], dtype=float)
            if "qdot_ekf_3d" in comparison
            else (
                np.asarray(comparison["qdot_biorbd_kalman"], dtype=float)
                if "qdot_biorbd_kalman" in comparison
                else None
            )
        )
        return q, qdot
    return None, None


def discover_reconstruction_catalog(output_dir: Path, pose2sim_trc: Path | None = None) -> list[dict[str, object]]:
    bundle_dirs = reconstruction_dirs_for_path(output_dir)
    if bundle_dirs:
        rows: list[dict[str, object]] = []
        for recon_dir in bundle_dirs:
            summary = load_bundle_summary(recon_dir)
            if not summary:
                continue
            name = str(summary.get("name", recon_dir.name))
            family = str(summary.get("family", ""))
            reproj = summary.get("reprojection_px", {})
            family_version = summary.get("family_version")
            latest_version = latest_version_for_family(family)
            rows.append(
                {
                    "name": name,
                    "label": reconstruction_label(name),
                    "cached": True,
                    "path": str(recon_dir),
                    "frames": summary.get("n_frames"),
                    "reproj_mean": reproj.get("mean"),
                    "reproj_std": reproj.get("std"),
                    "family": family,
                    "family_version": family_version,
                    "latest_family_version": latest_version,
                    "is_latest": (family_version == latest_version) if latest_version is not None else None,
                }
            )
        order = {name: idx for idx, name in enumerate(RECONSTRUCTION_ORDER)}
        rows.sort(key=lambda row: (order.get(str(row["name"]), 999), str(row["name"])))
        return rows

    ekf_states_path = output_dir / "ekf_states.npz"
    kalman_path = output_dir / "kalman_comparison.npz"
    kalman_flip_acc_path = output_dir / "kalman_comparison_flip_acc.npz"
    q_names = load_q_names(ekf_states_path)
    bundle_summary = load_bundle_summary(output_dir)
    rows: list[dict[str, object]] = []

    pose2sim_path = pose2sim_trc or ROOT / "inputs/trc/1_partie_0429.trc"
    rows.append(
        {
            "name": "pose2sim",
            "label": reconstruction_label("pose2sim"),
            "cached": pose2sim_path.exists() or bundle_summary.get("family") == "pose2sim",
            "path": str(
                output_dir / "reconstruction_bundle.npz"
                if bundle_summary.get("family") == "pose2sim"
                else pose2sim_path
            ),
            "frames": None,
            "reproj_mean": None,
            "reproj_std": None,
        }
    )

    triang_once_path = output_dir / "triangulation_pose2sim_like_once.npz"
    mean_px, std_px = compute_triangulation_reprojection_stats(triang_once_path)
    frames = None
    if triang_once_path.exists():
        data = np.load(triang_once_path, allow_pickle=True)
        frames = int(np.asarray(data["points_3d"]).shape[0])
    rows.append(
        {
            "name": "triangulation_once",
            "label": reconstruction_label("triangulation_once"),
            "cached": triang_once_path.exists(),
            "path": str(triang_once_path),
            "frames": frames,
            "reproj_mean": mean_px,
            "reproj_std": std_px,
        }
    )

    triang_path = output_dir / "triangulation_pose2sim_like.npz"
    mean_px, std_px = compute_triangulation_reprojection_stats(triang_path)
    frames = None
    if triang_path.exists():
        data = np.load(triang_path, allow_pickle=True)
        frames = int(np.asarray(data["points_3d"]).shape[0])
    rows.append(
        {
            "name": "triangulation_adaptive",
            "label": reconstruction_label("triangulation_adaptive"),
            "cached": triang_path.exists(),
            "path": str(triang_path),
            "frames": frames,
            "reproj_mean": mean_px,
            "reproj_std": std_px,
        }
    )

    triang_fast_path = output_dir / "triangulation_pose2sim_like_fast.npz"
    mean_px, std_px = compute_triangulation_reprojection_stats(triang_fast_path)
    frames = None
    if triang_fast_path.exists():
        data = np.load(triang_fast_path, allow_pickle=True)
        frames = int(np.asarray(data["points_3d"]).shape[0])
    rows.append(
        {
            "name": "triangulation_fast",
            "label": reconstruction_label("triangulation_fast"),
            "cached": triang_fast_path.exists(),
            "path": str(triang_fast_path),
            "frames": frames,
            "reproj_mean": mean_px,
            "reproj_std": std_px,
        }
    )

    for variant, comparison_path in [
        ("ekf_2d_acc", kalman_path),
        ("ekf_2d_flip_acc", kalman_flip_acc_path),
        ("ekf_2d_dyn", kalman_path),
        ("ekf_2d_flip_dyn", output_dir / "kalman_comparison_flip_dyn.npz"),
        ("ekf_3d", kalman_path),
    ]:
        q, _ = read_q_variant(ekf_states_path, comparison_path, variant)
        reproj_mean = None
        reproj_std = None
        if comparison_path.exists():
            comp = np.load(comparison_path, allow_pickle=True)
            if variant == "ekf_3d":
                reproj_mean = (
                    float(comp["ekf_3d_reprojection_mean_px"]) if "ekf_3d_reprojection_mean_px" in comp else None
                )
                reproj_std = float(comp["ekf_3d_reprojection_std_px"]) if "ekf_3d_reprojection_std_px" in comp else None
            elif "ekf_2d_reprojection_mean_px" in comp:
                reproj_mean = float(comp["ekf_2d_reprojection_mean_px"])
                reproj_std = float(comp["ekf_2d_reprojection_std_px"]) if "ekf_2d_reprojection_std_px" in comp else None
        if bundle_summary.get("family") in ("ekf_2d", "ekf_3d"):
            bundle_reproj = bundle_summary.get("reprojection_px", {})
            expected_variant = None
            if bundle_summary.get("family") == "ekf_3d":
                expected_variant = "ekf_3d"
            elif bundle_summary.get("family") == "ekf_2d":
                predictor = str(bundle_summary.get("predictor", "acc"))
                flip = bool(bundle_summary.get("flip_left_right", False))
                expected_variant = f"ekf_2d_{'flip_' if flip else ''}{predictor}"
            if variant == expected_variant:
                reproj_mean = bundle_reproj.get("mean")
                reproj_std = bundle_reproj.get("std")
        rows.append(
            {
                "name": variant,
                "label": reconstruction_label(variant),
                "cached": q is not None,
                "path": str(comparison_path if variant == "ekf_3d" else ekf_states_path),
                "frames": None if q is None else int(q.shape[0]),
                "reproj_mean": reproj_mean,
                "reproj_std": reproj_std,
                "n_dof": None if q is None else int(q.shape[1]),
                "q_names": q_names,
            }
        )

    order = {name: idx for idx, name in enumerate(RECONSTRUCTION_ORDER)}
    rows.sort(key=lambda row: order.get(str(row["name"]), 999))
    return rows


def available_dual_show_options(output_dir: Path, pose2sim_trc: Path | None = None) -> list[str]:
    catalog = discover_reconstruction_catalog(output_dir, pose2sim_trc)
    bundle_dirs = reconstruction_dirs_for_path(output_dir)
    if bundle_dirs:
        return [str(row["name"]) for row in catalog if row["cached"]]
    mapping = {
        "pose2sim": "pose2sim",
        "triangulation_adaptive": "triangulation",
        "ekf_3d": "ekf_3d",
        "ekf_2d_acc": "ekf_2d_acc",
        "ekf_2d_flip_acc": "ekf_2d_flip_acc",
        "ekf_2d_dyn": "ekf_2d_dyn",
        "ekf_2d_flip_dyn": "ekf_2d_flip_dyn",
    }
    options = []
    for row in catalog:
        if row["cached"] and row["name"] in mapping:
            options.append(mapping[str(row["name"])])
    return options


def available_multiview_show_options(output_dir: Path, pose2sim_trc: Path | None = None) -> list[str]:
    options = ["raw"]
    options.extend(available_dual_show_options(output_dir, pose2sim_trc))
    return options


def parse_trc_points(trc_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Parse a TRC file into COCO17-like points for preview/GUI usage."""

    _frames, time_s, points_3d, data_rate = parse_reconstruction_trc_points(trc_path)
    return points_3d, time_s, data_rate


def resample_points(points: np.ndarray, source_time: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    out = np.full((len(target_time), points.shape[1], 3), np.nan)
    for marker_idx in range(points.shape[1]):
        for axis in range(3):
            values = points[:, marker_idx, axis]
            valid = np.isfinite(values)
            if np.sum(valid) < 2:
                continue
            out[:, marker_idx, axis] = np.interp(
                target_time, source_time[valid], values[valid], left=np.nan, right=np.nan
            )
    return out


def interpolate_short_nan_runs(values: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Linearly fill only short interior NaN runs along time for each column."""

    data = np.asarray(values, dtype=float)
    if data.ndim == 1:
        data = data[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False
    if data.shape[0] == 0 or max_gap_frames <= 0:
        return np.asarray(values, dtype=float)

    result = np.array(data, copy=True)
    for col_idx in range(result.shape[1]):
        column = result[:, col_idx]
        finite_idx = np.flatnonzero(np.isfinite(column))
        if finite_idx.size < 2:
            continue
        for left_idx, right_idx in zip(finite_idx[:-1], finite_idx[1:]):
            gap_start = int(left_idx) + 1
            gap_end = int(right_idx)
            gap_len = gap_end - gap_start
            if gap_len <= 0 or gap_len > int(max_gap_frames):
                continue
            if np.any(np.isfinite(column[gap_start:gap_end])):
                continue
            result[gap_start:gap_end, col_idx] = np.interp(
                np.arange(gap_start, gap_end, dtype=float),
                np.array([left_idx, right_idx], dtype=float),
                np.array([column[left_idx], column[right_idx]], dtype=float),
            )
    return result[:, 0] if squeeze else result


def fill_short_edge_nan_runs(values: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Fill short leading/trailing NaN runs with the nearest finite value."""

    data = np.asarray(values, dtype=float)
    if data.ndim == 1:
        data = data[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False
    if data.shape[0] == 0 or max_gap_frames <= 0:
        return np.asarray(values, dtype=float)

    result = np.array(data, copy=True)
    for col_idx in range(result.shape[1]):
        column = result[:, col_idx]
        finite_idx = np.flatnonzero(np.isfinite(column))
        if finite_idx.size == 0:
            continue
        first_finite = int(finite_idx[0])
        if 0 < first_finite <= int(max_gap_frames) and not np.any(np.isfinite(column[:first_finite])):
            result[:first_finite, col_idx] = column[first_finite]
        last_finite = int(finite_idx[-1])
        trailing_len = int(column.shape[0] - last_finite - 1)
        if 0 < trailing_len <= int(max_gap_frames) and not np.any(np.isfinite(column[last_finite + 1 :])):
            result[last_finite + 1 :, col_idx] = column[last_finite]
    return result[:, 0] if squeeze else result


def biorbd_markers_from_q(biomod_path: Path, q_series: np.ndarray) -> np.ndarray:
    import biorbd

    model = biorbd.Model(str(biomod_path))
    marker_names = [name.to_string() for name in model.markerNames()]
    points = np.full((q_series.shape[0], len(COCO17), 3), np.nan)
    for frame_idx, q in enumerate(q_series):
        for marker_name, marker in zip(marker_names, model.markers(q)):
            if marker_name in KP_INDEX:
                points[frame_idx, KP_INDEX[marker_name], :] = marker.to_array()
    return points


def biorbd_q_names(model) -> list[str]:
    names: list[str] = []
    for i_seg in range(model.nbSegment()):
        segment = model.segment(i_seg)
        seg_name = segment.name().to_string()
        for i_dof in range(segment.nbDof()):
            names.append(f"{seg_name}:{segment.nameDof(i_dof).to_string()}")
    return names


def biorbd_segment_frames_from_q(model, q_values: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    frames: list[tuple[str, np.ndarray, np.ndarray]] = []
    for seg_idx, rt in enumerate(model.allGlobalJCS(q_values)):
        matrix = np.asarray(rt.to_array(), dtype=float)
        if matrix.shape != (4, 4):
            continue
        segment_name = model.segment(seg_idx).name().to_string()
        origin = matrix[:3, 3]
        rotation = matrix[:3, :3]
        if np.all(np.isfinite(origin)) and np.all(np.isfinite(rotation)):
            frames.append((segment_name, origin, rotation))
    return frames


def single_frame_reconstruction(reconstruction: ReconstructionResult, frame_idx: int) -> ReconstructionResult:
    return ReconstructionResult(
        frames=np.asarray(reconstruction.frames[frame_idx : frame_idx + 1], dtype=int),
        points_3d=np.asarray(reconstruction.points_3d[frame_idx : frame_idx + 1], dtype=float),
        mean_confidence=np.asarray(reconstruction.mean_confidence[frame_idx : frame_idx + 1], dtype=float),
        reprojection_error=np.asarray(reconstruction.reprojection_error[frame_idx : frame_idx + 1], dtype=float),
        reprojection_error_per_view=np.asarray(
            reconstruction.reprojection_error_per_view[frame_idx : frame_idx + 1], dtype=float
        ),
        multiview_coherence=np.asarray(reconstruction.multiview_coherence[frame_idx : frame_idx + 1], dtype=float),
        epipolar_coherence=np.asarray(reconstruction.epipolar_coherence[frame_idx : frame_idx + 1], dtype=float),
        triangulation_coherence=np.asarray(
            reconstruction.triangulation_coherence[frame_idx : frame_idx + 1], dtype=float
        ),
        excluded_views=np.asarray(reconstruction.excluded_views[frame_idx : frame_idx + 1], dtype=bool),
        coherence_method=str(reconstruction.coherence_method),
        epipolar_coherence_compute_time_s=float(reconstruction.epipolar_coherence_compute_time_s),
    )


def pair_dof_names(q_names: np.ndarray) -> list[tuple[str, str, str | None]]:
    names = [str(name) for name in q_names]
    pairs = []
    for name in names:
        if not name.startswith("LEFT_"):
            continue
        right_name = name.replace("LEFT_", "RIGHT_", 1)
        if right_name in names:
            pair_label = name.replace("LEFT_", "", 1)
            pairs.append((pair_label, name, right_name))
    for name in names:
        if name.startswith(("UPPER_BACK:", "LOWER_TRUNK:")):
            pairs.append((name, name, None))
    pairs.sort(key=lambda item: item[0])
    return pairs


def resolve_biomod_path(biomod_path: str | Path | None) -> Path | None:
    if biomod_path is None:
        return None
    path = Path(str(biomod_path))
    if not path.is_absolute():
        path = ROOT / path
    return path


def infer_model_variant_from_biomod(biomod_path: str | Path | None) -> str:
    path = resolve_biomod_path(biomod_path)
    if path is None or not path.exists():
        return DEFAULT_MODEL_VARIANT
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return DEFAULT_MODEL_VARIANT

    current_segment_name = None
    rotations_value = ""
    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("segment\t"):
            current_segment_name = line.split("\t", 1)[1].strip()
            continue
        if current_segment_name not in {"UPPER_BACK", "LOWER_TRUNK"}:
            continue
        if line == "endsegment":
            if rotations_value:
                break
            current_segment_name = None
            continue
        if line.startswith("rotations\t"):
            rotations_value = line.split("\t", 1)[1].strip().lower().replace(" ", "")
            if current_segment_name == "LOWER_TRUNK":
                return (
                    "upper_root_back_flexion_1d"
                    if rotations_value in {"x", "rx", "y", "ry"}
                    else "upper_root_back_3dof"
                )
            if current_segment_name == "UPPER_BACK":
                return "back_flexion_1d" if rotations_value in {"x", "rx", "y", "ry"} else "back_3dof"

    if not rotations_value:
        text = "\n".join(lines)
        if "LOWER_TRUNK" in text:
            return "upper_root_back_flexion_1d"
        if "UPPER_BACK" in text:
            return "back_flexion_1d"
        return DEFAULT_MODEL_VARIANT
    return DEFAULT_MODEL_VARIANT


def biomod_supports_upper_back_options(biomod_path: str | Path | None) -> bool:
    return infer_model_variant_from_biomod(biomod_path) in {
        "back_flexion_1d",
        "back_3dof",
        "upper_root_back_flexion_1d",
        "upper_root_back_3dof",
    }


def set_equal_3d_limits(ax, points_dict: dict[str, np.ndarray], frame_idx: int | None) -> None:
    """Apply equal 3D limits either from one frame or from the full sequence."""
    pts = []
    for points in points_dict.values():
        if frame_idx is None:
            valid = points.reshape(-1, 3)
            valid = valid[np.all(np.isfinite(valid), axis=1)]
            if valid.size:
                pts.append(valid)
            continue
        if frame_idx < points.shape[0]:
            frame_points = points[frame_idx]
            valid = frame_points[np.all(np.isfinite(frame_points), axis=1)]
            if valid.size:
                pts.append(valid)
    if not pts:
        return
    flat = np.vstack(pts)
    mins = np.min(flat, axis=0)
    maxs = np.max(flat, axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(0.25, 0.6 * np.max(maxs - mins))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def keypoint_groups(points: np.ndarray) -> dict[str, np.ndarray]:
    indices = {
        "left": [KP_INDEX[name] for name in COCO17 if name in LEFT_KEYPOINTS],
        "right": [KP_INDEX[name] for name in COCO17 if name in RIGHT_KEYPOINTS],
        "center": [KP_INDEX[name] for name in COCO17 if name not in LEFT_KEYPOINTS and name not in RIGHT_KEYPOINTS],
    }
    grouped: dict[str, np.ndarray] = {}
    for side, side_indices in indices.items():
        side_points = points[side_indices]
        valid = np.all(np.isfinite(side_points), axis=1)
        grouped[side] = side_points[valid]
    return grouped


def edge_linewidth(start_name: str, end_name: str, base: float, lower_scale: float = 3.0) -> float:
    return base * lower_scale if (start_name, end_name) in LOWER_LIMB_EDGES else base


def draw_skeleton_3d(ax, frame_points: np.ndarray, color: str, label: str, marker_size: float = 22.0) -> None:
    grouped = keypoint_groups(frame_points)
    show_markers = float(marker_size) > 0.0
    if show_markers and grouped["center"].size:
        ax.scatter(
            grouped["center"][:, 0],
            grouped["center"][:, 1],
            grouped["center"][:, 2],
            s=marker_size,
            c=color,
            marker="o",
            depthshade=False,
            label=label,
        )
    if show_markers and grouped["left"].size:
        ax.scatter(
            grouped["left"][:, 0],
            grouped["left"][:, 1],
            grouped["left"][:, 2],
            s=marker_size * 1.55,
            c=color,
            marker="^",
            depthshade=False,
        )
    if show_markers and grouped["right"].size:
        ax.scatter(
            grouped["right"][:, 0],
            grouped["right"][:, 1],
            grouped["right"][:, 2],
            s=marker_size * 1.55,
            c=color,
            marker="s",
            depthshade=False,
        )
    label_drawn = bool(show_markers and grouped["center"].size)
    for start_name, end_name in SKELETON_EDGES:
        p0 = frame_points[KP_INDEX[start_name]]
        p1 = frame_points[KP_INDEX[end_name]]
        if np.all(np.isfinite(p0)) and np.all(np.isfinite(p1)):
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                color=color,
                linewidth=edge_linewidth(start_name, end_name, base=1.8),
                alpha=0.9,
                label=label if not label_drawn else None,
            )
            label_drawn = True


def draw_upper_back_preview(
    ax,
    frame_points: np.ndarray,
    segment_frames: list[tuple[str, np.ndarray, np.ndarray]] | None = None,
) -> None:
    """Draw back triangles around the mid-back origin for models with a segmented trunk."""

    geometry = segmented_back_frame_geometry(frame_points, segment_frames)
    if geometry is None:
        return
    mid_back, hip_triangle, shoulder_triangle = geometry
    ax.plot(
        hip_triangle[:, 0],
        hip_triangle[:, 1],
        hip_triangle[:, 2],
        color="#1d4e89",
        linewidth=2.4,
        alpha=0.9,
    )
    ax.plot(
        shoulder_triangle[:, 0],
        shoulder_triangle[:, 1],
        shoulder_triangle[:, 2],
        color="#1d4e89",
        linewidth=2.4,
        alpha=0.9,
    )
    ax.scatter(
        [mid_back[0]],
        [mid_back[1]],
        [mid_back[2]],
        s=44,
        c="#1d4e89",
        marker="D",
        depthshade=False,
    )


def segmented_back_frame_geometry(
    frame_points: np.ndarray,
    segment_frames: list[tuple[str, np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return mid-back and both segmented-back triangles for one 3D frame."""

    if frame_points is None:
        return None
    left_hip = np.asarray(frame_points[KP_INDEX["left_hip"]], dtype=float)
    right_hip = np.asarray(frame_points[KP_INDEX["right_hip"]], dtype=float)
    left_shoulder = np.asarray(frame_points[KP_INDEX["left_shoulder"]], dtype=float)
    right_shoulder = np.asarray(frame_points[KP_INDEX["right_shoulder"]], dtype=float)
    if not (
        np.all(np.isfinite(left_hip))
        and np.all(np.isfinite(right_hip))
        and np.all(np.isfinite(left_shoulder))
        and np.all(np.isfinite(right_shoulder))
    ):
        return None
    segment_map = (
        {
            str(name): (np.asarray(origin, dtype=float), np.asarray(rotation, dtype=float))
            for name, origin, rotation in segment_frames
        }
        if segment_frames is not None
        else {}
    )
    upper_back_frame = segment_map.get("UPPER_BACK") or segment_map.get("LOWER_TRUNK")
    hip_center = 0.5 * (left_hip + right_hip)
    upper_center = 0.5 * (left_shoulder + right_shoulder)
    mid_back = upper_back_frame[0] if upper_back_frame is not None else 0.5 * (hip_center + upper_center)
    hip_triangle = np.vstack([right_hip, left_hip, mid_back, right_hip])
    shoulder_triangle = np.vstack([right_shoulder, left_shoulder, mid_back, right_shoulder])
    return mid_back, hip_triangle, shoulder_triangle


def draw_upper_back_overlay_2d(
    ax,
    *,
    hip_triangle_2d: np.ndarray | None,
    shoulder_triangle_2d: np.ndarray | None,
    mid_back_2d: np.ndarray | None,
    color: str,
    line_width: float = 1.9,
    line_alpha: float = 0.9,
    line_style: str = "-",
    marker_size: float = 34.0,
    marker_line_width: float = 1.5,
    marker_alpha: float = 0.95,
) -> None:
    """Draw the segmented-back overlay in a 2D camera view."""

    if hip_triangle_2d is not None:
        hip_triangle_2d = np.asarray(hip_triangle_2d, dtype=float)
        if np.all(np.isfinite(hip_triangle_2d)):
            ax.plot(
                hip_triangle_2d[:, 0],
                hip_triangle_2d[:, 1],
                color=color,
                linewidth=line_width,
                alpha=line_alpha,
                linestyle=line_style,
            )
    if shoulder_triangle_2d is not None:
        shoulder_triangle_2d = np.asarray(shoulder_triangle_2d, dtype=float)
        if np.all(np.isfinite(shoulder_triangle_2d)):
            ax.plot(
                shoulder_triangle_2d[:, 0],
                shoulder_triangle_2d[:, 1],
                color=color,
                linewidth=line_width,
                alpha=line_alpha,
                linestyle=line_style,
            )
    if mid_back_2d is not None:
        mid_back_2d = np.asarray(mid_back_2d, dtype=float)
        if np.all(np.isfinite(mid_back_2d)):
            ax.scatter(
                [mid_back_2d[0]],
                [mid_back_2d[1]],
                s=marker_size,
                facecolors="none",
                edgecolors=color,
                linewidths=marker_line_width,
                marker="D",
                alpha=marker_alpha,
            )


def segmented_back_overlay_from_q(
    biomod_path: Path | str | None,
    q_series: np.ndarray,
) -> dict[str, np.ndarray] | None:
    """Compute 3D overlay trajectories for segmented-back models."""

    biomod_path = resolve_biomod_path(biomod_path)
    if biomod_path is None or not biomod_path.exists():
        return None
    if infer_model_variant_from_biomod(biomod_path) not in {
        "back_flexion_1d",
        "back_3dof",
        "upper_root_back_flexion_1d",
        "upper_root_back_3dof",
    }:
        return None

    import biorbd

    q_series = np.asarray(q_series, dtype=float)
    if q_series.ndim != 2 or q_series.size == 0:
        return None
    model = biorbd.Model(str(biomod_path))
    marker_names = [name.to_string() for name in model.markerNames()]
    required_marker_names = {"left_hip", "right_hip", "left_shoulder", "right_shoulder"}
    if not required_marker_names.issubset(set(marker_names)):
        return None

    n_frames = q_series.shape[0]
    mid_back = np.full((n_frames, 1, 3), np.nan, dtype=float)
    hip_triangle = np.full((n_frames, 4, 3), np.nan, dtype=float)
    shoulder_triangle = np.full((n_frames, 4, 3), np.nan, dtype=float)
    for frame_idx, q_values in enumerate(q_series):
        marker_points = np.full((len(COCO17), 3), np.nan, dtype=float)
        for marker_name, marker in zip(marker_names, model.markers(q_values)):
            kp_idx = KP_INDEX.get(marker_name)
            if kp_idx is not None:
                marker_points[kp_idx, :] = marker.to_array()
        geometry = segmented_back_frame_geometry(marker_points, biorbd_segment_frames_from_q(model, q_values))
        if geometry is None:
            continue
        mid_back_point, hip_triangle_points, shoulder_triangle_points = geometry
        mid_back[frame_idx, 0, :] = mid_back_point
        hip_triangle[frame_idx, :, :] = hip_triangle_points
        shoulder_triangle[frame_idx, :, :] = shoulder_triangle_points
    return {"mid_back": mid_back, "hip_triangle": hip_triangle, "shoulder_triangle": shoulder_triangle}


def has_segmented_back_visualization(
    *,
    segment_frames: list[tuple[str, np.ndarray, np.ndarray]] | None = None,
    q_names: list[str] | np.ndarray | None = None,
    summary: dict[str, object] | None = None,
) -> bool:
    """Return whether one model/reconstruction includes a dedicated back segment to visualize."""

    if segment_frames is not None:
        segment_names = {str(name) for name, _origin, _rotation in segment_frames}
        if {"UPPER_BACK", "LOWER_TRUNK"} & segment_names:
            return True
    if q_names is not None:
        q_name_set = {str(name) for name in q_names}
        if any(name.startswith(("UPPER_BACK:", "LOWER_TRUNK:")) for name in q_name_set):
            return True
    if summary:
        model_variant = str(summary.get("model_variant", "") or "")
        if model_variant in {"back_flexion_1d", "back_3dof", "upper_root_back_flexion_1d", "upper_root_back_3dof"}:
            return True
    return False


def compute_root_frame_from_points(frame_points: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    left_hip = frame_points[KP_INDEX["left_hip"]]
    right_hip = frame_points[KP_INDEX["right_hip"]]
    left_shoulder = frame_points[KP_INDEX["left_shoulder"]]
    right_shoulder = frame_points[KP_INDEX["right_shoulder"]]
    if not (
        np.all(np.isfinite(left_hip))
        and np.all(np.isfinite(right_hip))
        and np.all(np.isfinite(left_shoulder))
        and np.all(np.isfinite(right_shoulder))
    ):
        return None, None
    origin = 0.5 * (left_hip + right_hip)
    shoulder_center = 0.5 * (left_shoulder + right_shoulder)
    z_axis = normalize(shoulder_center - origin)
    y_seed = 0.5 * ((left_hip - right_hip) + (left_shoulder - right_shoulder))
    y_seed = normalize(y_seed)
    x_axis = normalize(np.cross(y_seed, z_axis))
    if not np.all(np.isfinite(x_axis)):
        return origin, None
    y_axis = normalize(np.cross(z_axis, x_axis))
    if not np.all(np.isfinite(y_axis)):
        return origin, None
    return origin, np.column_stack((x_axis, y_axis, z_axis))


def draw_coordinate_system(
    ax,
    origin: np.ndarray,
    rotation: np.ndarray,
    scale: float = 0.2,
    alpha: float = 1.0,
    prefix: str = "",
    show_labels: bool = True,
    line_width: float = 2.0,
) -> None:
    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    labels = ["X", "Y", "Z"]
    for axis_idx in range(3):
        direction = rotation[:, axis_idx]
        endpoint = origin + scale * direction
        ax.plot(
            [origin[0], endpoint[0]],
            [origin[1], endpoint[1]],
            [origin[2], endpoint[2]],
            color=colors[axis_idx],
            linewidth=line_width,
            alpha=alpha,
        )
        if show_labels:
            ax.text(
                endpoint[0], endpoint[1], endpoint[2], f"{prefix}{labels[axis_idx]}", color=colors[axis_idx], fontsize=8
            )


def draw_skeleton_2d(
    ax,
    frame_points: np.ndarray,
    color: str,
    label: str | None = None,
    marker_size: float = 12.0,
    marker_fill: bool = True,
    marker_edge_width: float = 1.4,
    line_alpha: float = 0.85,
    line_style: str = "-",
    line_width_scale: float = 1.0,
    legend_label: str | None = None,
) -> None:
    if label is None:
        label = legend_label or ""
    grouped = keypoint_groups(frame_points)
    facecolors = color if marker_fill else "none"
    edgecolors = color
    if grouped["center"].size:
        ax.scatter(
            grouped["center"][:, 0],
            grouped["center"][:, 1],
            s=marker_size,
            facecolors=facecolors,
            edgecolors=edgecolors,
            linewidths=marker_edge_width,
            marker="o",
            label=label,
            alpha=0.85,
        )
    if grouped["left"].size:
        ax.scatter(
            grouped["left"][:, 0],
            grouped["left"][:, 1],
            s=marker_size * 1.25,
            facecolors=facecolors,
            edgecolors=edgecolors,
            linewidths=marker_edge_width,
            marker="^",
            alpha=0.85,
        )
    if grouped["right"].size:
        ax.scatter(
            grouped["right"][:, 0],
            grouped["right"][:, 1],
            s=marker_size * 1.25,
            facecolors=facecolors,
            edgecolors=edgecolors,
            linewidths=marker_edge_width,
            marker="s",
            alpha=0.85,
        )
    for start_name, end_name in SKELETON_EDGES:
        p0 = frame_points[KP_INDEX[start_name]]
        p1 = frame_points[KP_INDEX[end_name]]
        if np.all(np.isfinite(p0)) and np.all(np.isfinite(p1)):
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                color=color,
                linewidth=line_width_scale * edge_linewidth(start_name, end_name, base=1.6),
                alpha=line_alpha,
                linestyle=line_style,
            )


def annotate_flip_keypoints(ax, frame_points: np.ndarray) -> None:
    highlight_names = [
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_wrist",
        "right_wrist",
        "left_ankle",
        "right_ankle",
    ]
    for kp_name in highlight_names:
        point = frame_points[KP_INDEX[kp_name]]
        if not np.all(np.isfinite(point)):
            continue
        short = kp_name.replace("left_", "L ").replace("right_", "R ").replace("_", "\n")
        ax.text(point[0] + 8.0, point[1] - 8.0, short, color="#222222", fontsize=8, alpha=0.85)


def draw_trampoline_bed(ax) -> None:
    """Draw the trampoline bed using the calibrated judging geometry."""
    ax.set_aspect("equal")
    outer = plt.Rectangle(
        (-BED_X_MAX, -BED_Y_MAX), 2 * BED_X_MAX, 2 * BED_Y_MAX, fill=False, linewidth=2.0, edgecolor="#2b6cb0"
    )
    big_rect = plt.Rectangle((-X_MAX, -Y_MAX), 2 * X_MAX, 2 * Y_MAX, fill=False, linewidth=1.6, edgecolor="#b56576")
    small_square = plt.Rectangle(
        (-X_INNER, -Y_INNER),
        2 * X_INNER,
        2 * Y_INNER,
        fill=True,
        linewidth=1.4,
        edgecolor="#b56576",
        facecolor="#f7fafc",
        alpha=0.55,
    )
    ax.add_patch(outer)
    ax.add_patch(big_rect)
    ax.add_patch(small_square)
    cross = TRAMPOLINE_GEOMETRY.cross
    ax.plot(
        [cross["left"][0], cross["right"][0]], [cross["left"][1], cross["right"][1]], color="#b56576", linewidth=1.4
    )
    ax.plot(
        [cross["bottom"][0], cross["top"][0]], [cross["bottom"][1], cross["top"][1]], color="#b56576", linewidth=1.4
    )
    ax.text(0.0, 0.0, "0.0", ha="center", va="center", fontsize=13, color="#b56576")
    ax.text(0.0, Y_INNER + 0.05, "0.2", ha="center", va="bottom", fontsize=11, color="#7b341e")
    ax.text(0.0, -(Y_INNER + 0.05), "0.2", ha="center", va="top", fontsize=11, color="#7b341e")
    ax.text(X_INNER + 0.05, 0.0, "0.1", ha="left", va="center", fontsize=11, color="#7b341e")
    ax.text(-(X_INNER + 0.05), 0.0, "0.1", ha="right", va="center", fontsize=11, color="#7b341e")
    for x_sign in (-1, 1):
        for y_sign in (-1, 1):
            ax.text(
                x_sign * (0.5 * (X_INNER + X_MAX)),
                y_sign * (0.5 * (Y_INNER + Y_MAX)),
                "0.3",
                ha="center",
                va="center",
                fontsize=11,
                color="#7b341e",
            )
    ax.set_xlim(-BED_X_MAX - 0.2, BED_X_MAX + 0.2)
    ax.set_ylim(-BED_Y_MAX - 0.2, BED_Y_MAX + 0.2)
    ax.grid(alpha=0.18)
    ax.set_xlabel("X on bed (m)")
    ax.set_ylabel("Y on bed (m)")


def draw_trampoline_bed_3d(ax, z_level: float) -> None:
    """Draw a lightweight 3D trampoline outline for animation previews."""

    outer = np.array(
        [
            [-BED_X_MAX, -BED_Y_MAX, z_level],
            [BED_X_MAX, -BED_Y_MAX, z_level],
            [BED_X_MAX, BED_Y_MAX, z_level],
            [-BED_X_MAX, BED_Y_MAX, z_level],
            [-BED_X_MAX, -BED_Y_MAX, z_level],
        ],
        dtype=float,
    )
    big_rect = np.array(
        [
            [-X_MAX, -Y_MAX, z_level],
            [X_MAX, -Y_MAX, z_level],
            [X_MAX, Y_MAX, z_level],
            [-X_MAX, Y_MAX, z_level],
            [-X_MAX, -Y_MAX, z_level],
        ],
        dtype=float,
    )
    inner_rect = np.array(
        [
            [-X_INNER, -Y_INNER, z_level],
            [X_INNER, -Y_INNER, z_level],
            [X_INNER, Y_INNER, z_level],
            [-X_INNER, Y_INNER, z_level],
            [-X_INNER, -Y_INNER, z_level],
        ],
        dtype=float,
    )
    cross = TRAMPOLINE_GEOMETRY.cross
    ax.plot(outer[:, 0], outer[:, 1], outer[:, 2], color="#2b6cb0", linewidth=1.8, alpha=0.45)
    ax.plot(big_rect[:, 0], big_rect[:, 1], big_rect[:, 2], color="#b56576", linewidth=1.3, alpha=0.4)
    ax.plot(inner_rect[:, 0], inner_rect[:, 1], inner_rect[:, 2], color="#b56576", linewidth=1.0, alpha=0.30)
    ax.plot(
        [cross["left"][0], cross["right"][0]],
        [cross["left"][1], cross["right"][1]],
        [z_level, z_level],
        color="#b56576",
        linewidth=1.0,
        alpha=0.35,
    )
    ax.plot(
        [cross["bottom"][0], cross["top"][0]],
        [cross["bottom"][1], cross["top"][1]],
        [z_level, z_level],
        color="#b56576",
        linewidth=1.0,
        alpha=0.35,
    )


def trampoline_contact_zone_xy(frame_points_list: list[np.ndarray]) -> np.ndarray | None:
    """Return the judged trampoline zone associated with the strongest visible foot contact."""

    best_penalty = -1.0
    best_xy: tuple[float, float] | None = None
    for frame_points in frame_points_list:
        for kp_name in ("left_ankle", "right_ankle"):
            point = np.asarray(frame_points[KP_INDEX[kp_name]], dtype=float)
            if np.all(np.isfinite(point[:2])):
                penalty = trampoline_penalty_refined(float(point[0]), float(point[1]))
                if np.isfinite(penalty) and penalty >= best_penalty:
                    best_penalty = float(penalty)
                    best_xy = (float(point[0]), float(point[1]))
    if best_xy is None:
        return None
    return judged_trampoline_zone_xy(*best_xy)


def draw_trampoline_contact_zone_3d(
    ax,
    polygon_xy: np.ndarray | None,
    z_level: float,
    *,
    color: str = "#ed8936",
    alpha: float = 0.30,
) -> None:
    """Draw a semi-transparent contact zone on the trampoline bed."""

    if polygon_xy is None or np.asarray(polygon_xy).shape != (4, 2):
        return
    vertices = np.column_stack((np.asarray(polygon_xy, dtype=float), np.full(4, float(z_level), dtype=float)))
    patch = Poly3DCollection([vertices], facecolors=color, edgecolors=color, linewidths=1.2, alpha=float(alpha))
    ax.add_collection3d(patch)


def compute_airborne_mask_from_points(
    points_3d: np.ndarray,
    *,
    threshold_m: float,
    min_consecutive_frames: int,
) -> np.ndarray:
    """Infer airborne frames from 3D points using the bundle flight settings."""

    above = np.all(points_3d[:, :, 2] > float(threshold_m), axis=1)
    above &= np.all(np.isfinite(points_3d[:, :, 2]), axis=1)
    mask = np.zeros(points_3d.shape[0], dtype=bool)
    consecutive = 0
    for frame_idx, is_above in enumerate(above):
        consecutive = consecutive + 1 if is_above else 0
        if consecutive >= max(1, int(min_consecutive_frames)):
            mask[frame_idx] = True
    return mask


def jump_segmentation_height_series(points_3d: np.ndarray | None, root_q: np.ndarray) -> np.ndarray:
    """Backward-compatible wrapper around the shared jump-segmentation helper."""

    return shared_jump_segmentation_height_series(points_3d, root_q)


def shared_jump_analysis(
    state: "SharedAppState",
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
    """Reuse one DD jump analysis through the shared GUI state cache."""

    cache = getattr(state, "jump_analysis_cache", None)
    if cache is None:
        cache = {}
        state.jump_analysis_cache = cache
    return get_cached_jump_analysis(
        cache,
        reconstruction_name=reconstruction_name,
        root_q=np.asarray(root_q, dtype=float),
        points_3d=None if points_3d is None else np.asarray(points_3d, dtype=float),
        fps=float(fps),
        height_threshold=height_threshold,
        height_threshold_range_ratio=height_threshold_range_ratio,
        smoothing_window_s=smoothing_window_s,
        min_airtime_s=min_airtime_s,
        min_gap_s=min_gap_s,
        min_peak_prominence_m=min_peak_prominence_m,
        contact_window_s=contact_window_s,
        full_q=None if full_q is None else np.asarray(full_q, dtype=float),
        q_names=list(q_names) if q_names is not None else None,
        angle_mode=angle_mode,
        analysis_start_frame=analysis_start_frame,
        require_complete_jumps=require_complete_jumps,
    )


def shared_jump_analysis_for_reconstruction(
    state: "SharedAppState",
    reconstruction_name: str | None,
    *,
    analysis_start_frame: int = ANALYSIS_START_FRAME,
    require_complete_jumps: bool = True,
) -> DDSessionAnalysis | None:
    """Resolve one reconstruction through the shared preview cache and reuse jump analysis."""

    name = str(reconstruction_name or "").strip()
    if not name:
        return None
    try:
        _output_dir, bundle, _preview_state = load_shared_reconstruction_preview_state(
            state,
            preferred_names=[name],
            fallback_count=1,
            include_3d=True,
            include_q=True,
            include_q_root=True,
        )
        root_q, full_q, q_names = preview_root_series_for_reconstruction(
            bundle=bundle,
            name=name,
            initial_rotation_correction=bool(state.initial_rotation_correction_var.get()),
        )
        if root_q is None:
            return None
        recon_3d = bundle.get("recon_3d", {}) if isinstance(bundle, dict) else {}
        points_3d = np.asarray(recon_3d[name], dtype=float) if name in recon_3d else None
        return shared_jump_analysis(
            state,
            reconstruction_name=name,
            root_q=np.asarray(root_q, dtype=float),
            points_3d=points_3d,
            fps=float(state.fps_var.get()),
            height_threshold=TRAMPOLINE_BED_HEIGHT_M,
            height_threshold_range_ratio=0.20,
            smoothing_window_s=0.15,
            min_airtime_s=0.25,
            min_gap_s=0.08,
            min_peak_prominence_m=0.35,
            contact_window_s=0.35,
            full_q=None if full_q is None else np.asarray(full_q, dtype=float),
            q_names=q_names,
            angle_mode="euler",
            analysis_start_frame=analysis_start_frame,
            require_complete_jumps=require_complete_jumps,
        )
    except Exception:
        return None


def preview_pose_frame_indices(pose_frames: np.ndarray, target_frames: np.ndarray) -> np.ndarray:
    """Map preview frame ids back to pose-data indices."""

    pose_frames = np.asarray(pose_frames, dtype=int)
    target_frames = np.asarray(target_frames, dtype=int)
    frame_to_idx = {int(frame): idx for idx, frame in enumerate(pose_frames)}
    missing = [int(frame) for frame in target_frames if int(frame) not in frame_to_idx]
    if missing:
        raise ValueError(
            "The preview bundle frames and the 2D detections do not share the same frame ids. "
            f"Missing raw frames for ids: {missing[:10]}"
        )
    return np.asarray([frame_to_idx[int(frame)] for frame in target_frames], dtype=int)


def compose_multiview_crop_points(
    base_points_2d: np.ndarray,
    projected_layers: dict[str, np.ndarray],
    selected_names: list[str],
) -> np.ndarray:
    """Build the 2D crop reference from detections and selected reprojections."""

    stacked = [np.asarray(base_points_2d, dtype=float)]
    for name in selected_names:
        if name == "raw":
            continue
        points_2d = projected_layers.get(str(name))
        if points_2d is None:
            continue
        stacked.append(np.asarray(points_2d, dtype=float))
    if len(stacked) == 1:
        return stacked[0]
    return np.concatenate(stacked, axis=2)


ANNOTATION_MARKER_COLORS = plt.colormaps["tab20"].resampled(len(COCO17))
ANNOTATION_DELETE_RADIUS_PX = 18.0
ANNOTATION_HOVER_RADIUS_PX = 10.0
ANNOTATION_DRAG_START_RADIUS_PX = 10.0
ANNOTATION_DRAG_ACTIVATION_PX = 3.0
ANNOTATION_SNAP_RADIUS_PX = 24.0
ANNOTATION_KINEMATIC_BOOTSTRAP_PASSES = 10
ANNOTATION_KINEMATIC_INITIAL_BOOTSTRAP_MULTIPLIER = 3
ANNOTATION_KINEMATIC_CLICK_PASSES = 3
ANNOTATION_KINEMATIC_CLICK_DIRECT_PASSES = 1
ANNOTATION_KINEMATIC_WINDOW_RADIUS = 1
ANNOTATION_KINEMATIC_WINDOW_PASSES = 2
ANNOTATION_FRAME_FILTER_OPTIONS = {
    "all": "All frames",
    "flipped": "Flipped L/R",
    "worst_reproj": "Worst reproj 5%",
}


def annotation_marker_color(keypoint_name: str) -> tuple[float, float, float, float]:
    """Return one stable color for a keypoint name."""

    kp_idx = KP_INDEX.get(str(keypoint_name), 0)
    return ANNOTATION_MARKER_COLORS(int(kp_idx))


def annotation_marker_shape(keypoint_name: str) -> str:
    """Return one side-aware marker shape for annotations."""

    if str(keypoint_name) in LEFT_KEYPOINTS:
        return "+"
    if str(keypoint_name) in RIGHT_KEYPOINTS:
        return "x"
    return "+"


def annotation_keypoint_names_for_biomod(biomod_path: str | Path | None) -> tuple[str, ...]:
    """Return the annotation keypoint order, adding segmented-back markers when available."""

    names = list(ANNOTATION_KEYPOINT_ORDER)
    if biomod_supports_upper_back_options(biomod_path) and "mid_back" not in names:
        names.insert(12, "mid_back")
    return tuple(names)


def annotation_motion_prior_center(
    point_t_minus_1: np.ndarray | None,
    point_t_minus_2: np.ndarray | None,
) -> np.ndarray | None:
    """Predict one simple constant-velocity 2D center from the two previous frames."""

    if point_t_minus_1 is None or point_t_minus_2 is None:
        return None
    pt1 = np.asarray(point_t_minus_1, dtype=float).reshape(2)
    pt2 = np.asarray(point_t_minus_2, dtype=float).reshape(2)
    if not (np.all(np.isfinite(pt1)) and np.all(np.isfinite(pt2))):
        return None
    return pt1 + (pt1 - pt2)


def annotation_epipolar_guides(
    calibrations: dict[str, object],
    source_camera_name: str,
    target_camera_name: str,
    source_point_2d: np.ndarray,
) -> np.ndarray | None:
    """Return one image-space epipolar line ``ax + by + c = 0`` for a target camera."""

    point = np.asarray(source_point_2d, dtype=float).reshape(2)
    if not np.all(np.isfinite(point)):
        return None
    if source_camera_name == target_camera_name:
        return None
    source_calibration = calibrations.get(str(source_camera_name))
    target_calibration = calibrations.get(str(target_camera_name))
    if source_calibration is None or target_calibration is None:
        return None
    fundamental = fundamental_matrix(source_calibration, target_calibration)
    point_h = np.array([point[0], point[1], 1.0], dtype=float)
    line = fundamental @ point_h
    if not np.all(np.isfinite(line)) or np.linalg.norm(line[:2]) < 1e-12:
        return None
    return line


def annotation_project_point_to_line(line: np.ndarray, point_xy: np.ndarray) -> np.ndarray | None:
    """Project one 2D point orthogonally onto an epipolar line."""

    line = np.asarray(line, dtype=float).reshape(3)
    point_xy = np.asarray(point_xy, dtype=float).reshape(2)
    if not (np.all(np.isfinite(line)) and np.all(np.isfinite(point_xy))):
        return None
    a, b, c = [float(value) for value in line]
    denom = a * a + b * b
    if denom <= 1e-12:
        return None
    distance = (a * point_xy[0] + b * point_xy[1] + c) / denom
    return np.array([point_xy[0] - a * distance, point_xy[1] - b * distance], dtype=float)


def annotation_intersect_epipolar_lines(lines: list[np.ndarray]) -> np.ndarray | None:
    """Return the least-squares intersection of multiple epipolar lines."""

    if len(lines) < 2:
        return None
    rows = []
    offsets = []
    for line in lines:
        array = np.asarray(line, dtype=float).reshape(3)
        if not np.all(np.isfinite(array)):
            continue
        normal = array[:2]
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            continue
        rows.append(normal / norm)
        offsets.append(-array[2] / norm)
    if len(rows) < 2:
        return None
    matrix = np.asarray(rows, dtype=float)
    rhs = np.asarray(offsets, dtype=float)
    try:
        solution, *_rest = np.linalg.lstsq(matrix, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    solution = np.asarray(solution, dtype=float).reshape(2)
    return solution if np.all(np.isfinite(solution)) else None


def annotation_triangulated_reprojection(
    calibrations: dict[str, object],
    *,
    target_camera_name: str,
    source_camera_names: list[str],
    source_points_2d: list[np.ndarray],
) -> np.ndarray | None:
    """Triangulate one 3D point from already placed views and reproject it to a target camera."""

    valid_projections: list[np.ndarray] = []
    valid_points: list[np.ndarray] = []
    valid_scores: list[float] = []
    for camera_name, point_2d in zip(source_camera_names, source_points_2d):
        point = np.asarray(point_2d, dtype=float).reshape(2)
        calibration = calibrations.get(str(camera_name))
        if calibration is None or not np.all(np.isfinite(point)):
            continue
        projection_matrix = getattr(calibration, "projection_matrix", None)
        if projection_matrix is None:
            projection_matrix = getattr(calibration, "P", None)
        if projection_matrix is None:
            continue
        valid_projections.append(np.asarray(projection_matrix, dtype=float))
        valid_points.append(point)
        valid_scores.append(1.0)
    if len(valid_points) < 2:
        return None
    point_3d = weighted_triangulation(valid_projections, np.asarray(valid_points), np.asarray(valid_scores))
    if not np.all(np.isfinite(point_3d)):
        return None
    target_calibration = calibrations.get(str(target_camera_name))
    if target_calibration is None:
        return None
    projected = np.asarray(target_calibration.project_point(point_3d), dtype=float)
    return projected if np.all(np.isfinite(projected)) else None


def triangulate_annotation_frame_points(
    calibrations: dict[str, object],
    *,
    camera_names: list[str],
    frame_number: int,
    annotation_payload: dict[str, object],
) -> np.ndarray:
    """Triangulate all currently annotated keypoints for one frame."""

    points_3d = np.full((len(COCO17), 3), np.nan, dtype=float)
    for keypoint_name in COCO17:
        valid_projections: list[np.ndarray] = []
        valid_points: list[np.ndarray] = []
        valid_scores: list[float] = []
        for camera_name in camera_names:
            point_xy, _score = get_annotation_point(
                annotation_payload,
                camera_name=camera_name,
                frame_number=frame_number,
                keypoint_name=keypoint_name,
            )
            if point_xy is None:
                continue
            calibration = calibrations.get(str(camera_name))
            if calibration is None:
                continue
            projection_matrix = getattr(calibration, "projection_matrix", None)
            if projection_matrix is None:
                projection_matrix = getattr(calibration, "P", None)
            if projection_matrix is None:
                continue
            point = np.asarray(point_xy, dtype=float).reshape(2)
            if not np.all(np.isfinite(point)):
                continue
            valid_projections.append(np.asarray(projection_matrix, dtype=float))
            valid_points.append(point)
            valid_scores.append(1.0)
        if len(valid_points) < 2:
            continue
        point_3d = weighted_triangulation(valid_projections, np.asarray(valid_points), np.asarray(valid_scores))
        if np.all(np.isfinite(point_3d)):
            points_3d[KP_INDEX[keypoint_name], :] = point_3d
    return points_3d


def annotation_pose_data_for_frame(
    base_pose_data: PoseData,
    *,
    camera_names: list[str],
    frame_number: int,
    annotation_payload: dict[str, object],
) -> PoseData:
    """Build one-frame PoseData from sparse annotations only."""

    base_camera_names = [str(name) for name in base_pose_data.camera_names]
    selected_camera_names = [str(name) for name in camera_names if str(name) in base_camera_names]
    if not selected_camera_names:
        raise ValueError("No selected cameras are available in the current pose data.")
    frames = np.asarray([int(frame_number)], dtype=int)
    keypoints = np.full((len(selected_camera_names), 1, len(COCO17), 2), np.nan, dtype=float)
    scores = np.zeros((len(selected_camera_names), 1, len(COCO17)), dtype=float)
    for cam_idx, camera_name in enumerate(selected_camera_names):
        for keypoint_name in COCO17:
            point_xy, _score = get_annotation_point(
                annotation_payload,
                camera_name=camera_name,
                frame_number=frame_number,
                keypoint_name=keypoint_name,
            )
            if point_xy is None:
                continue
            point = np.asarray(point_xy, dtype=float).reshape(2)
            if not np.all(np.isfinite(point)):
                continue
            kp_idx = KP_INDEX[str(keypoint_name)]
            keypoints[cam_idx, 0, kp_idx] = point
            scores[cam_idx, 0, kp_idx] = 1.0
    return PoseData(
        camera_names=selected_camera_names,
        frames=frames,
        keypoints=keypoints,
        scores=scores,
        frame_stride=int(getattr(base_pose_data, "frame_stride", 1) or 1),
        raw_keypoints=np.array(keypoints, copy=True),
        annotated_keypoints=np.array(keypoints, copy=True),
        annotated_scores=np.array(scores, copy=True),
    )


def annotation_adjust_image(
    image: np.ndarray,
    *,
    brightness: float = 1.0,
    contrast: float = 1.0,
) -> np.ndarray:
    """Backward-compatible wrapper for annotation image level adjustment."""

    return adjust_image_levels(image, brightness=brightness, contrast=contrast)


def find_annotation_frame_with_images(
    *,
    frames: np.ndarray,
    current_index: int,
    direction: int,
    camera_names: list[str],
    images_root: Path | None,
) -> int:
    """Move to the next frame that actually has one image for the selected cameras."""

    frame_array = np.asarray(frames, dtype=int)
    if frame_array.size == 0:
        return 0
    step = -1 if int(direction) < 0 else 1
    current_index = max(0, min(int(current_index), frame_array.size - 1))
    fallback_index = max(0, min(frame_array.size - 1, current_index + step))
    if images_root is None or not camera_names:
        return fallback_index
    candidate_index = fallback_index
    while 0 <= candidate_index < frame_array.size:
        frame_number = int(frame_array[candidate_index])
        for camera_name in camera_names:
            image_path = resolve_execution_image_path(images_root, camera_name, frame_number)
            if image_path is not None and image_path.exists():
                return candidate_index
        candidate_index += step
    return fallback_index


def load_preview_bundle(
    output_dir: Path,
    biomod_path: Path | None,
    pose2sim_trc: Path | None,
    align_root: bool = False,
) -> dict[str, object]:
    gui_debug(
        "load_preview_bundle start "
        f"dataset={output_dir} biomod={'yes' if biomod_path is not None else 'no'} "
        f"pose2sim={'yes' if pose2sim_trc is not None else 'no'} align_root={align_root}"
    )
    bundle_dirs = reconstruction_dirs_for_path(output_dir)
    if bundle_dirs:
        bundle = load_dataset_preview_bundle(output_dir, biomod_path, biorbd_markers_from_q)
        gui_debug(
            "load_preview_bundle done "
            f"bundle_mode=dataset recon3d={len(bundle['recon_3d'])} "
            f"reconq={len(bundle['recon_q'])} frames={len(bundle['frames'])}"
        )
        return bundle

    bundle: dict[str, object] = {
        "frames": np.array([], dtype=int),
        "time_s": np.array([], dtype=float),
        "q_names": np.array([], dtype=object),
        "recon_3d": {},
        "recon_q": {},
        "recon_qdot": {},
        "recon_q_root": {},
        "recon_qdot_root": {},
        "recon_summary": {},
    }

    triang_once_path = output_dir / "triangulation_pose2sim_like_once.npz"
    triang_path = output_dir / "triangulation_pose2sim_like.npz"
    triang_fast_path = output_dir / "triangulation_pose2sim_like_fast.npz"
    master_points = None
    master_frames = None
    if triang_once_path.exists():
        data = np.load(triang_once_path, allow_pickle=True)
        bundle["recon_3d"]["triangulation_once"] = np.asarray(data["points_3d"], dtype=float)
        if master_frames is None:
            master_points = bundle["recon_3d"]["triangulation_once"]
            master_frames = np.asarray(data["frames"], dtype=int)
    if triang_path.exists():
        data = np.load(triang_path, allow_pickle=True)
        master_points = np.asarray(data["points_3d"], dtype=float)
        master_frames = np.asarray(data["frames"], dtype=int)
        bundle["recon_3d"]["triangulation_adaptive"] = master_points
    if triang_fast_path.exists():
        data = np.load(triang_fast_path, allow_pickle=True)
        bundle["recon_3d"]["triangulation_fast"] = np.asarray(data["points_3d"], dtype=float)
        if master_frames is None:
            master_points = bundle["recon_3d"]["triangulation_fast"]
            master_frames = np.asarray(data["frames"], dtype=int)
    if master_frames is None and pose2sim_trc is not None and pose2sim_trc.exists():
        pose2sim_points, pose2sim_time, pose2sim_rate = parse_trc_points(pose2sim_trc)
        master_frames = np.arange(pose2sim_points.shape[0], dtype=int)
        master_points = pose2sim_points
        bundle["recon_3d"]["pose2sim"] = pose2sim_points
        bundle["frames"] = master_frames
        bundle["time_s"] = pose2sim_time
    else:
        bundle["frames"] = master_frames if master_frames is not None else np.array([], dtype=int)
        bundle["time_s"] = bundle["frames"] / 120.0 if master_frames is not None else np.array([], dtype=float)

    if (
        pose2sim_trc is not None
        and pose2sim_trc.exists()
        and master_frames is not None
        and "pose2sim" not in bundle["recon_3d"]
    ):
        pose2sim_points, pose2sim_time, _ = parse_trc_points(pose2sim_trc)
        pose2sim_resampled = resample_points(pose2sim_points, pose2sim_time, bundle["time_s"])
        if align_root and master_points is not None:
            pose2sim_resampled = align_to_reference(master_points, pose2sim_resampled)
        bundle["recon_3d"]["pose2sim"] = pose2sim_resampled

    ekf_states_path = output_dir / "ekf_states.npz"
    kalman_path = output_dir / "kalman_comparison.npz"
    bundle["q_names"] = load_q_names(ekf_states_path)
    if biomod_path is None:
        return bundle

    for variant in ["ekf_2d_acc", "ekf_2d_flip_acc", "ekf_2d_dyn", "ekf_2d_flip_dyn", "ekf_3d"]:
        q, qdot = read_q_variant(
            ekf_states_path,
            (
                kalman_path
                if "flip" not in variant
                else output_dir / f"kalman_comparison_{variant.replace('ekf_2d_', '')}.npz"
            ),
            variant,
        )
        if q is None:
            continue
        bundle["recon_q"][variant] = q
        if qdot is not None:
            bundle["recon_qdot"][variant] = qdot
        q_for_markers = q if master_frames is None else q[: len(bundle["frames"])]
        if q_for_markers.size == 0:
            continue
        points = biorbd_markers_from_q(biomod_path, q_for_markers)
        if align_root and master_points is not None:
            points = align_to_reference(master_points[: points.shape[0]], points)
        bundle["recon_3d"][variant] = points

    gui_debug(
        "load_preview_bundle done "
        f"bundle_mode=legacy recon3d={len(bundle['recon_3d'])} "
        f"reconq={len(bundle['recon_q'])} frames={len(bundle['frames'])}"
    )
    return bundle


def get_cached_preview_bundle(
    state: SharedAppState,
    output_dir: Path,
    biomod_path: Path | None,
    pose2sim_trc: Path | None,
    align_root: bool = False,
) -> dict[str, object]:
    cache_key = preview_bundle_cache_key(output_dir, biomod_path, pose2sim_trc, align_root)
    cached = state.preview_bundle_cache.get(cache_key)
    if cached is not None:
        gui_debug(f"preview bundle cache hit dataset={output_dir}")
        report_startup_status(state, f"Using cached preview bundle: {output_dir.name}")
        return cached
    gui_debug(f"preview bundle cache miss dataset={output_dir}")
    report_startup_status(state, f"Loading preview bundle: {output_dir.name}")
    bundle = load_preview_bundle(output_dir, biomod_path, pose2sim_trc, align_root=align_root)
    state.preview_bundle_cache[cache_key] = bundle
    return bundle


def resolve_preview_biomod(dataset_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for model_dir in scan_model_dirs(Path(dataset_dir)):
        candidates.extend(sorted(model_dir.glob("*.bioMod")))
    return candidates[0] if candidates else None


def resolve_reconstruction_biomod(dataset_dir: Path, reconstruction_name: str) -> Path | None:
    recon_dir = reconstruction_dir_by_name(dataset_dir, reconstruction_name)
    if recon_dir is not None:
        candidate = recon_dir / "vitpose_chain.bioMod"
        if candidate.exists():
            return candidate
    return resolve_preview_biomod(dataset_dir)


def preview_root_series_for_reconstruction(
    *,
    bundle: dict[str, object],
    name: str,
    initial_rotation_correction: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, list[str] | None]:
    recon_q = bundle.get("recon_q", {})
    recon_q_root = bundle.get("recon_q_root", {})
    recon_3d = bundle.get("recon_3d", {})
    q_names = np.asarray(bundle.get("q_names", np.array([], dtype=object)), dtype=object)
    if name in recon_q:
        full_q = np.asarray(recon_q[name], dtype=float)
        root_q = extract_root_from_q(q_names, full_q, unwrap_rotations=False, renormalize_rotations=True)
        return root_q, full_q, [str(q_name) for q_name in q_names]
    if name in recon_3d:
        root_q, _ = extract_root_from_points(
            np.asarray(recon_3d[name], dtype=float),
            bool(initial_rotation_correction),
            False,
        )
        return root_q, None, None
    if name in recon_q_root:
        return np.asarray(recon_q_root[name], dtype=float), None, None
    return None, None, None


class LabeledEntry(ttk.Frame):
    """Petit composant label + entry + bouton browse optionnel."""

    def __init__(
        self,
        master,
        label: str,
        default: str = "",
        browse: bool = False,
        directory: bool = False,
        readonly: bool = False,
        *,
        label_width: int = 18,
        entry_width: int = 70,
        label_padx: tuple[int, int] = (0, 6),
        filetypes: tuple[tuple[str, str], ...] | None = None,
        browse_initialdir: str | None = None,
        on_browse_selected=None,
    ):
        super().__init__(master)
        self.directory = directory
        self.filetypes = filetypes
        self.browse_initialdir = browse_initialdir
        self.on_browse_selected = on_browse_selected
        self.label_widget = ttk.Label(self, text=label, width=label_width)
        self.label_widget.pack(side=tk.LEFT, padx=label_padx)
        self.var = tk.StringVar(value=default)
        self.entry_widget = ttk.Entry(self, textvariable=self.var, width=entry_width)
        self.entry_widget.pack(side=tk.LEFT, fill=tk.X, expand=True)
        if readonly:
            self.entry_widget.state(["readonly"])
        self.browse_button: ttk.Button | None = None
        if browse:
            self.browse_button = ttk.Button(self, text="Browse", command=self._browse)
            self.browse_button.pack(side=tk.LEFT, padx=(6, 0))

    @staticmethod
    def _sanitize_filetypes(
        filetypes: tuple[tuple[str, str], ...] | list[tuple[str, str]] | None,
    ) -> tuple[tuple[str, str], ...] | None:
        """Normalize Tk filetypes and avoid macOS crashes with basename wildcards.

        Tk on macOS can crash when a filetype pattern is not a simple extension
        glob. Patterns such as ``*_DD.json`` or ``*_keypoints.json`` are useful
        for humans but can become invalid `allowedFileTypes` entries in the
        native Cocoa dialog. We therefore downgrade such filters to their
        extension-only equivalent on macOS.
        """

        if not filetypes:
            return None

        normalized: list[tuple[str, str]] = []
        for entry in filetypes:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                continue
            label, pattern = str(entry[0]).strip(), str(entry[1]).strip()
            if not label or not pattern:
                continue
            if sys.platform == "darwin":
                if pattern == "*.*":
                    pattern = "*"
                elif pattern.startswith("*") and "." in pattern and not pattern.startswith("*."):
                    suffix = pattern.rsplit(".", 1)[-1].strip()
                    pattern = f"*.{suffix}" if suffix else "*"
            normalized.append((label, pattern))

        return tuple(normalized) or None

    def _browse(self) -> None:
        current_value = self.get()
        initial_dir_path: Path | None = None
        if current_value:
            current_path = Path(current_value)
            if not current_path.is_absolute():
                current_path = ROOT / current_path
            candidate_dir = current_path if self.directory else current_path.parent
            if candidate_dir.exists():
                initial_dir_path = candidate_dir
        if initial_dir_path is None and self.browse_initialdir:
            candidate_dir = Path(self.browse_initialdir)
            if not candidate_dir.is_absolute():
                candidate_dir = ROOT / candidate_dir
            if candidate_dir.exists():
                initial_dir_path = candidate_dir
        if initial_dir_path is None:
            fallback_dir = ROOT if self.directory else ROOT / "inputs"
            initial_dir_path = fallback_dir
        initial_dir = str(initial_dir_path)
        if self.directory:
            path = filedialog.askdirectory(initialdir=initial_dir)
        else:
            kwargs = {"initialdir": initial_dir}
            sanitized_filetypes = self._sanitize_filetypes(self.filetypes)
            if sanitized_filetypes:
                kwargs["filetypes"] = sanitized_filetypes
            path = filedialog.askopenfilename(**kwargs)
        if path:
            try:
                rel = Path(path).resolve().relative_to(ROOT)
                self.var.set(str(rel))
            except Exception:
                self.var.set(path)
            if callable(self.on_browse_selected):
                self.on_browse_selected(self.get())

    def get(self) -> str:
        return self.var.get().strip()

    def set_tooltip(self, text: str) -> None:
        attach_tooltip(self.label_widget, text)
        attach_tooltip(self.entry_widget, text)
        if self.browse_button is not None:
            attach_tooltip(self.browse_button, text)


class ToolTip:
    """Info-bulle simple au survol d'un widget."""

    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text = text
        self.tipwindow: tk.Toplevel | None = None
        self.after_id: str | None = None
        self.last_x = 0
        self.last_y = 0
        self.widget.bind("<Enter>", self.schedule_show, add="+")
        self.widget.bind("<Leave>", self.hide, add="+")
        self.widget.bind("<ButtonPress>", self.hide, add="+")
        self.widget.bind("<Motion>", self.on_motion, add="+")

    def on_motion(self, event) -> None:
        self.last_x = int(event.x_root)
        self.last_y = int(event.y_root)
        if self.tipwindow is not None:
            self.tipwindow.wm_geometry(f"+{self.last_x + 16}+{self.last_y + 16}")

    def schedule_show(self, event=None) -> None:
        if not self.text:
            return
        if event is not None:
            self.last_x = int(event.x_root)
            self.last_y = int(event.y_root)
        self.cancel_scheduled_show()
        self.after_id = self.widget.after(400, self.show)

    def cancel_scheduled_show(self) -> None:
        if self.after_id is not None:
            try:
                self.widget.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

    def show(self) -> None:
        self.cancel_scheduled_show()
        if self.tipwindow is not None or not self.text:
            return
        x = self.last_x or (self.widget.winfo_rootx() + 16)
        y = self.last_y or (self.widget.winfo_rooty() + self.widget.winfo_height() + 6)
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_attributes("-topmost", True)
        tw.wm_geometry(f"+{x + 16}+{y + 16}")
        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#fffde7",
            foreground="#1f2328",
            relief=tk.SOLID,
            borderwidth=1,
            wraplength=340,
            padx=8,
            pady=6,
        )
        label.pack()

    def hide(self, _event=None) -> None:
        self.cancel_scheduled_show()
        if self.tipwindow is not None:
            self.tipwindow.destroy()
            self.tipwindow = None


def extend_listbox_selection(widget: tk.Listbox, direction: int) -> str:
    """Extend one listbox selection by one row with Shift+Up/Down semantics."""

    size = int(widget.size())
    if size <= 0:
        return "break"
    selected = [int(index) for index in widget.curselection()]
    if selected:
        target = min(selected) - 1 if direction < 0 else max(selected) + 1
    else:
        try:
            active = int(widget.index(tk.ACTIVE))
        except Exception:
            active = 0
        target = active + (1 if direction > 0 else -1)
    target = max(0, min(size - 1, int(target)))
    widget.selection_set(target)
    try:
        widget.activate(target)
    except Exception:
        pass
    try:
        widget.see(target)
    except Exception:
        pass
    return "break"


def select_all_listbox(widget: tk.Listbox) -> str:
    """Select all rows in one Tk listbox."""

    if int(widget.size()) > 0:
        widget.selection_set(0, tk.END)
        try:
            widget.activate(0)
        except Exception:
            pass
    return "break"


def bind_extended_listbox_shortcuts(widget: tk.Listbox) -> None:
    """Add additive keyboard selection shortcuts to one multi-select listbox."""

    widget.bind("<Shift-Up>", lambda _event: extend_listbox_selection(widget, -1), add="+")
    widget.bind("<Shift-Down>", lambda _event: extend_listbox_selection(widget, 1), add="+")
    for sequence in ("<Control-a>", "<Control-A>", "<Command-a>", "<Command-A>"):
        widget.bind(sequence, lambda _event: select_all_listbox(widget), add="+")


def extend_treeview_selection(tree: ttk.Treeview, direction: int) -> str:
    """Extend one Treeview selection by one row with Shift+Up/Down semantics."""

    rows = list(tree.get_children(""))
    if not rows:
        return "break"
    selected = list(tree.selection())
    if selected:
        anchor = rows.index(selected[0]) if direction < 0 else rows.index(selected[-1])
        target_index = anchor + (-1 if direction < 0 else 1)
    else:
        focus_item = tree.focus()
        focus_index = rows.index(focus_item) if focus_item in rows else 0
        target_index = focus_index + (-1 if direction < 0 else 1)
    target_index = max(0, min(len(rows) - 1, int(target_index)))
    target_item = rows[target_index]
    updated = []
    for item in selected:
        if item in rows:
            updated.append(item)
    if target_item not in updated:
        updated.append(target_item)
    tree.selection_set(tuple(updated))
    tree.focus(target_item)
    try:
        tree.see(target_item)
    except Exception:
        pass
    return "break"


def select_all_treeview(tree: ttk.Treeview) -> str:
    """Select all rows in one Treeview."""

    rows = list(tree.get_children(""))
    if rows:
        tree.selection_set(tuple(rows))
        tree.focus(rows[0])
    return "break"


def bind_extended_treeview_shortcuts(tree: ttk.Treeview) -> None:
    """Add additive keyboard selection shortcuts to one multi-select Treeview."""

    tree.bind("<Shift-Up>", lambda _event: extend_treeview_selection(tree, -1), add="+")
    tree.bind("<Shift-Down>", lambda _event: extend_treeview_selection(tree, 1), add="+")
    for sequence in ("<Control-a>", "<Control-A>", "<Command-a>", "<Command-A>"):
        tree.bind(sequence, lambda _event: select_all_treeview(tree), add="+")


def attach_tooltip(widget: tk.Widget, text: str) -> ToolTip:
    tooltip = ToolTip(widget, text)
    setattr(widget, "_tooltip_ref", tooltip)
    return tooltip


@dataclass
class SharedAppState:
    calib_var: tk.StringVar
    keypoints_var: tk.StringVar
    annotation_path_var: tk.StringVar
    pose2sim_trc_var: tk.StringVar
    fps_var: tk.StringVar
    workers_var: tk.StringVar
    pose_data_mode_var: tk.StringVar
    pose_filter_window_var: tk.StringVar
    pose_outlier_ratio_var: tk.StringVar
    pose_p_low_var: tk.StringVar
    pose_p_high_var: tk.StringVar
    flip_improvement_ratio_var: tk.StringVar
    flip_min_gain_px_var: tk.StringVar
    flip_min_other_cameras_var: tk.StringVar
    flip_outlier_percentile_var: tk.StringVar
    flip_outlier_floor_px_var: tk.StringVar
    flip_restrict_to_outliers_var: tk.BooleanVar
    flip_temporal_weight_var: tk.StringVar
    flip_temporal_tau_px_var: tk.StringVar
    calibration_correction_var: tk.StringVar
    initial_rotation_correction_var: tk.BooleanVar
    selected_camera_names_var: tk.StringVar
    output_root_var: tk.StringVar
    profiles_config_var: tk.StringVar
    profiles: list[ReconstructionProfile] = field(default_factory=list)
    profile_listeners: list[callable] = field(default_factory=list)
    reconstruction_listeners: list[callable] = field(default_factory=list)
    calibration_cache: dict[str, dict[str, object]] = field(default_factory=dict)
    pose_data_cache: dict[tuple[object, ...], object] = field(default_factory=dict)
    preview_bundle_cache: dict[tuple[object, ...], dict[str, object]] = field(default_factory=dict)
    jump_analysis_cache: dict[tuple[object, ...], DDSessionAnalysis] = field(default_factory=dict)
    shared_reconstruction_selection: list[str] = field(default_factory=list)
    shared_reconstruction_selection_listeners: list[callable] = field(default_factory=list)
    shared_reconstruction_panel: object | None = None
    active_reconstruction_consumer: object | None = None
    clean_trial_outputs_callback: callable | None = None
    clean_trial_caches_callback: callable | None = None
    startup_status_callback: callable | None = None

    def set_profiles(self, profiles: list[ReconstructionProfile]) -> None:
        self.profiles = list(profiles)
        for callback in list(self.profile_listeners):
            try:
                callback()
            except Exception:
                pass

    def register_profile_listener(self, callback) -> None:
        if callback not in self.profile_listeners:
            self.profile_listeners.append(callback)

    def notify_reconstructions_updated(self) -> None:
        self.preview_bundle_cache.clear()
        self.jump_analysis_cache.clear()
        for callback in list(self.reconstruction_listeners):
            try:
                callback()
            except Exception:
                pass

    def register_reconstruction_listener(self, callback) -> None:
        if callback not in self.reconstruction_listeners:
            self.reconstruction_listeners.append(callback)

    def set_shared_reconstruction_selection(self, names: list[str]) -> None:
        self.shared_reconstruction_selection = list(names)
        for callback in list(self.shared_reconstruction_selection_listeners):
            try:
                callback()
            except Exception:
                pass

    def register_shared_reconstruction_selection_listener(self, callback) -> None:
        if callback not in self.shared_reconstruction_selection_listeners:
            self.shared_reconstruction_selection_listeners.append(callback)


def report_startup_status(state: SharedAppState | None, message: str) -> None:
    """Send one startup-status message to the temporary splash, if active."""

    if state is None:
        return
    callback = getattr(state, "startup_status_callback", None)
    if callback is None:
        return
    try:
        callback(str(message))
    except Exception:
        pass


class BusyStatusWindow(tk.Toplevel):
    """Small transient popup shown during long synchronous computations."""

    def __init__(self, parent, title: str, message: str):
        super().__init__(parent)
        self.title(str(title))
        self.resizable(False, False)
        self.transient(parent)
        self.attributes("-topmost", True)
        body = ttk.Frame(self, padding=14)
        body.pack(fill=tk.BOTH, expand=True)
        self.message_var = tk.StringVar(value=str(message))
        ttk.Label(body, textvariable=self.message_var, justify=tk.LEFT, wraplength=320).pack(fill=tk.X)
        self.progress = ttk.Progressbar(body, mode="indeterminate", length=320)
        self.progress.pack(fill=tk.X, pady=(10, 0))
        self.progress.start(12)
        self.update_idletasks()
        try:
            parent_root_x = int(parent.winfo_rootx())
            parent_root_y = int(parent.winfo_rooty())
            parent_width = int(parent.winfo_width())
            parent_height = int(parent.winfo_height())
            width = int(self.winfo_reqwidth())
            height = int(self.winfo_reqheight())
            x_pos = parent_root_x + max((parent_width - width) // 2, 0)
            y_pos = parent_root_y + max((parent_height - height) // 2, 0)
            self.geometry(f"+{x_pos}+{y_pos}")
        except Exception:
            pass

    def set_status(self, message: str) -> None:
        self.message_var.set(str(message))
        self.update_idletasks()

    def close(self) -> None:
        try:
            self.progress.stop()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass


@contextmanager
def gui_busy_popup(parent, *, title: str, message: str):
    """Show one small popup while a long synchronous task is running."""

    delay_s = 0.5

    class _NullBusyPopup:
        def set_status(self, _message: str) -> None:
            return

        def close(self) -> None:
            return

        def update(self) -> None:
            return

    class _DelayedBusyPopup:
        def __init__(self) -> None:
            self._started_at = time.monotonic()
            self._last_message = str(message)
            self._popup = None

        def _ensure_popup(self) -> None:
            if self._popup is not None:
                return
            if (time.monotonic() - self._started_at) < delay_s:
                return
            try:
                self._popup = BusyStatusWindow(parent, title=title, message=self._last_message)
            except Exception:
                self._popup = _NullBusyPopup()

        def set_status(self, current_message: str) -> None:
            self._last_message = str(current_message)
            self._ensure_popup()
            self._popup.set_status(self._last_message) if self._popup is not None else None

        def close(self) -> None:
            if self._popup is not None:
                self._popup.close()

        def update(self) -> None:
            self._ensure_popup()
            if self._popup is not None:
                self._popup.update()

    popup = _DelayedBusyPopup()
    try:
        yield popup
    finally:
        popup.close()
        try:
            parent.update_idletasks()
        except Exception:
            pass


def current_dataset_name(state: SharedAppState) -> str:
    keypoints_path = ROOT / state.keypoints_var.get()
    trc_path = ROOT / state.pose2sim_trc_var.get() if state.pose2sim_trc_var.get().strip() else None
    return infer_dataset_name(keypoints_path=keypoints_path, pose2sim_trc=trc_path)


def current_dataset_dir(state: SharedAppState) -> Path:
    return normalize_output_root(ROOT / state.output_root_var.get()) / current_dataset_name(state)


def current_selected_camera_names(state: SharedAppState) -> list[str]:
    return parse_camera_names(state.selected_camera_names_var.get())


def current_calibration_correction_mode(state: SharedAppState) -> str:
    raw = state.calibration_correction_var.get().strip()
    return (
        raw
        if raw
        in {
            "none",
            "flip_epipolar",
            "flip_epipolar_fast",
            "flip_epipolar_viterbi",
            "flip_epipolar_fast_viterbi",
            "flip_triangulation",
        }
        else "none"
    )


def normalize_pose_correction_mode(raw: str) -> str:
    value = str(raw).strip()
    return (
        value
        if value
        in {
            "none",
            "flip_epipolar",
            "flip_epipolar_fast",
            "flip_epipolar_viterbi",
            "flip_epipolar_fast_viterbi",
            "flip_triangulation",
        }
        else "none"
    )


def shared_pose_data_kwargs(state: SharedAppState, *, data_mode: str | None = None) -> dict[str, object]:
    return {
        "data_mode": str(data_mode or state.pose_data_mode_var.get()),
        "smoothing_window": int(state.pose_filter_window_var.get()),
        "outlier_threshold_ratio": float(state.pose_outlier_ratio_var.get()),
        "lower_percentile": float(state.pose_p_low_var.get()),
        "upper_percentile": float(state.pose_p_high_var.get()),
    }


def current_models_dir(state: SharedAppState) -> Path:
    return dataset_models_dir(normalize_output_root(ROOT / state.output_root_var.get()), current_dataset_name(state))


def current_reconstructions_dir(state: SharedAppState) -> Path:
    return dataset_reconstructions_dir(
        normalize_output_root(ROOT / state.output_root_var.get()), current_dataset_name(state)
    )


def current_figures_dir(state: SharedAppState) -> Path:
    return dataset_figures_dir(normalize_output_root(ROOT / state.output_root_var.get()), current_dataset_name(state))


def current_dataset_preview_state(
    state: SharedAppState,
    *,
    bundle: dict[str, object] | None,
    preferred_names: list[str],
    fallback_count: int,
    include_3d: bool = True,
    include_q: bool = True,
    include_q_root: bool = False,
    extra_rows: list[dict[str, object]] | None = None,
) -> tuple[Path, DatasetPreviewState]:
    """Build the shared preview-selection state for the current dataset."""

    output_dir = current_dataset_dir(state)
    catalog = discover_reconstruction_catalog(output_dir, optional_root_relative_path(state.pose2sim_trc_var.get()))
    preview_state = build_dataset_preview_state(
        catalog=catalog,
        bundle=bundle,
        preferred_names=preferred_names,
        fallback_count=fallback_count,
        include_3d=include_3d,
        include_q=include_q,
        include_q_root=include_q_root,
        extra_rows=extra_rows,
    )
    return output_dir, preview_state


def load_current_dataset_bundle(
    state: SharedAppState,
    *,
    biomod_path: Path | None = None,
    pose2sim_trc: Path | None = None,
    align_root: bool = False,
) -> tuple[Path, dict[str, object]]:
    """Load the cached preview bundle for the active dataset."""

    output_dir = current_dataset_dir(state)
    bundle = get_cached_preview_bundle(state, output_dir, biomod_path, pose2sim_trc, align_root=align_root)
    return output_dir, bundle


def load_shared_reconstruction_preview_state(
    state: SharedAppState,
    *,
    preferred_names: list[str],
    fallback_count: int,
    include_3d: bool = True,
    include_q: bool = True,
    include_q_root: bool = False,
    extra_rows: list[dict[str, object]] | None = None,
    biomod_path: Path | None = None,
    pose2sim_trc: Path | None = None,
    align_root: bool = False,
) -> tuple[Path, dict[str, object], DatasetPreviewState]:
    """Load the active dataset bundle together with the shared selection state."""

    output_dir, bundle = load_current_dataset_bundle(
        state,
        biomod_path=biomod_path,
        pose2sim_trc=pose2sim_trc,
        align_root=align_root,
    )
    _, preview_state = current_dataset_preview_state(
        state,
        bundle=bundle,
        preferred_names=preferred_names,
        fallback_count=fallback_count,
        include_3d=include_3d,
        include_q=include_q,
        include_q_root=include_q_root,
        extra_rows=extra_rows,
    )
    return output_dir, bundle, preview_state


def reconstruction_dir_by_name(dataset_dir: Path, reconstruction_name: str) -> Path | None:
    for recon_dir in reconstruction_dirs_for_path(dataset_dir):
        summary = load_bundle_summary(recon_dir)
        summary_name = str(summary.get("name", recon_dir.name))
        if summary_name == reconstruction_name:
            return recon_dir
    return None


def preview_bundle_cache_key(
    output_dir: Path,
    biomod_path: Path | None,
    pose2sim_trc: Path | None,
    align_root: bool,
) -> tuple[object, ...]:
    return (
        str(output_dir.resolve()),
        None if biomod_path is None else str(biomod_path.resolve()),
        None if pose2sim_trc is None else str(pose2sim_trc.resolve()),
        bool(align_root),
    )


def synchronize_profiles_initial_rotation_correction(state: SharedAppState) -> None:
    desired = bool(state.initial_rotation_correction_var.get())
    updated = False
    profiles: list[ReconstructionProfile] = []
    for profile in state.profiles:
        if bool(profile.initial_rotation_correction) != desired:
            profile = ReconstructionProfile(**{**profile_to_dict(profile), "initial_rotation_correction": desired})
            updated = True
        profiles.append(profile)
    if updated:
        state.set_profiles(profiles)


def ensure_dataset_layout(state: SharedAppState) -> None:
    dataset_dir = current_dataset_dir(state)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    current_models_dir(state).mkdir(parents=True, exist_ok=True)
    current_reconstructions_dir(state).mkdir(parents=True, exist_ok=True)
    current_figures_dir(state).mkdir(parents=True, exist_ok=True)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def infer_pose2sim_trc_from_keypoints(keypoints_path: Path) -> Path | None:
    dataset_name = infer_dataset_name(keypoints_path=keypoints_path)
    inputs_root = ROOT / "inputs"
    candidates = [
        keypoints_path.with_name(f"{dataset_name}.trc"),
        keypoints_path.with_name(f"{keypoints_path.stem.replace('_keypoints', '')}.trc"),
        inputs_root / "trc" / f"{dataset_name}.trc",
        inputs_root / f"{dataset_name}.trc",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists():
            return candidate
    return None


def optional_root_relative_path(raw: str) -> Path | None:
    raw = raw.strip()
    if not raw:
        return None
    return ROOT / raw


def append_default_pose2sim_profile(
    selected_profiles: list[ReconstructionProfile],
    all_profiles: list[ReconstructionProfile],
    pose2sim_trc_raw: str,
) -> list[ReconstructionProfile]:
    """Append the first configured TRC-file profile when a TRC is available and none was selected."""

    if not pose2sim_trc_raw.strip() or any(profile.family == "pose2sim" for profile in selected_profiles):
        return list(selected_profiles)
    pose2sim_profile = next((profile for profile in all_profiles if profile.family == "pose2sim"), None)
    if pose2sim_profile is None:
        return list(selected_profiles)
    return [*selected_profiles, pose2sim_profile]


FLIP_METHOD_DISPLAY_NAMES = {
    "none": "None",
    "epipolar": "Epipolar (local)",
    "epipolar_fast": "Epipolar fast (local)",
    "epipolar_viterbi": "Epipolar (Viterbi)",
    "epipolar_fast_viterbi": "Epipolar fast (Viterbi)",
    "ekf_prediction_gate": "EKF prediction gate",
    "triangulation_once": "Triangulation once",
    "triangulation_greedy": "Triangulation greedy",
    "triangulation_exhaustive": "Triangulation exhaustive",
}


def flip_method_display_name(method: str) -> str:
    """Return a user-facing label for a profile flip method."""

    return FLIP_METHOD_DISPLAY_NAMES.get(str(method).strip(), str(method).strip() or "Epipolar (local)")


COHERENCE_METHOD_DISPLAY_NAMES = {
    "epipolar": "Epipolar (precomputed)",
    "epipolar_fast": "Epipolar fast (precomputed)",
    "epipolar_framewise": "Epipolar (framewise)",
    "epipolar_fast_framewise": "Epipolar fast (framewise)",
}


def coherence_method_display_name(method: str) -> str:
    """Return a user-facing label for an EKF2D coherence method."""

    return COHERENCE_METHOD_DISPLAY_NAMES.get(str(method).strip(), str(method).strip() or "Epipolar (precomputed)")


def coherence_method_from_display_name(label: str) -> str:
    """Map a UI label back to its internal coherence identifier."""

    normalized = str(label).strip()
    for method, display_name in COHERENCE_METHOD_DISPLAY_NAMES.items():
        if normalized == display_name:
            return method
    return normalized or "epipolar"


ROOT_UNWRAP_MODE_DISPLAY_NAMES = {
    "double": "Double unwrap",
    "single": "Single unwrap",
    "off": "Off",
}

REPROJECTION_THRESHOLD_DISPLAY_VALUES = ("none", "5", "10", "15", "20", "25")


def reprojection_threshold_display_value(threshold_px: float | None) -> str:
    if threshold_px is None:
        return "none"
    value = float(threshold_px)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def reprojection_threshold_from_display_value(label: str) -> float | None:
    normalized = str(label).strip().lower()
    if normalized in {"", "none", "off"}:
        return None
    return float(normalized)


def root_unwrap_mode_display_name(mode: str) -> str:
    """Return a user-facing label for one root-angle stabilization mode."""

    normalized = normalize_root_unwrap_mode(mode)
    return ROOT_UNWRAP_MODE_DISPLAY_NAMES.get(normalized, normalized)


def root_unwrap_mode_from_display_name(label: str) -> str:
    """Map one UI label back to the canonical root-angle stabilization mode."""

    normalized = str(label).strip()
    for mode, display_name in ROOT_UNWRAP_MODE_DISPLAY_NAMES.items():
        if normalized == display_name:
            return mode
    return normalize_root_unwrap_mode(normalized or "off")


def profile_root_unwrap_mode(profile) -> str:
    """Resolve one profile root-angle stabilization mode with legacy fallback."""

    if bool(getattr(profile, "no_root_unwrap", False)):
        return "off"
    return normalize_root_unwrap_mode(getattr(profile, "root_unwrap_mode", None), legacy_unwrap=True)


def write_runtime_profiles_config(state: SharedAppState) -> Path:
    """Persist the in-memory profiles to a cache file used only for command execution."""

    runtime_path = LOCAL_CACHE / "runtime_profiles.json"
    save_profiles_json(runtime_path, state.profiles)
    return runtime_path


def calibration_cache_key(calib_path: Path) -> str:
    return str(calib_path.resolve())


def pose_data_cache_key(
    *,
    keypoints_path: Path,
    calib_path: Path,
    max_frames: int | None,
    frame_start: int | None,
    frame_end: int | None,
    data_mode: str,
    smoothing_window: int,
    outlier_threshold_ratio: float,
    lower_percentile: float,
    upper_percentile: float,
    annotations_path: Path | None = None,
) -> tuple[object, ...]:
    resolved_annotations = None if annotations_path is None else Path(annotations_path)
    return (
        str(keypoints_path.resolve()),
        str(calib_path.resolve()),
        None if max_frames is None else int(max_frames),
        None if frame_start is None else int(frame_start),
        None if frame_end is None else int(frame_end),
        str(data_mode),
        int(smoothing_window),
        float(outlier_threshold_ratio),
        float(lower_percentile),
        float(upper_percentile),
        None if resolved_annotations is None else str(resolved_annotations.resolve()),
        (
            None
            if resolved_annotations is None or not resolved_annotations.exists()
            else resolved_annotations.stat().st_mtime_ns
        ),
    )


def get_cached_calibrations(state: SharedAppState, calib_path: Path) -> dict[str, object]:
    key = calibration_cache_key(calib_path)
    cached = state.calibration_cache.get(key)
    if cached is None:
        report_startup_status(state, f"Loading calibrations: {calib_path.name}")
        cached = load_calibrations(calib_path)
        state.calibration_cache[key] = cached
    else:
        report_startup_status(state, f"Using cached calibrations: {calib_path.name}")
    return cached


def get_cached_pose_data(
    state: SharedAppState,
    *,
    keypoints_path: Path,
    calib_path: Path,
    max_frames: int | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
    data_mode: str = "cleaned",
    smoothing_window: int = 9,
    outlier_threshold_ratio: float = 0.10,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
    annotations_path: Path | None = None,
):
    resolved_annotations_path = None if annotations_path is None else Path(annotations_path)
    if resolved_annotations_path is None and str(data_mode) == "annotated":
        state_annotation_path = getattr(state, "annotation_path_var", None)
        state_annotation_value = "" if state_annotation_path is None else str(state_annotation_path.get()).strip()
        if state_annotation_value:
            resolved_annotations_path = ROOT / state_annotation_value
        else:
            resolved_annotations_path = default_annotation_path(keypoints_path)
    cache_key = pose_data_cache_key(
        keypoints_path=keypoints_path,
        calib_path=calib_path,
        max_frames=max_frames,
        frame_start=frame_start,
        frame_end=frame_end,
        data_mode=data_mode,
        smoothing_window=smoothing_window,
        outlier_threshold_ratio=outlier_threshold_ratio,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        annotations_path=resolved_annotations_path,
    )
    cached = state.pose_data_cache.get(cache_key)
    if cached is not None:
        report_startup_status(state, f"Using cached 2D poses: {keypoints_path.name}")
        calibrations = get_cached_calibrations(state, calib_path)
        return calibrations, cached
    calibrations = get_cached_calibrations(state, calib_path)
    report_startup_status(state, f"Loading 2D pose data: {keypoints_path.name}")
    pose_data = load_pose_data(
        keypoints_path,
        calibrations,
        max_frames=max_frames,
        frame_start=frame_start,
        frame_end=frame_end,
        data_mode=data_mode,
        smoothing_window=smoothing_window,
        outlier_threshold_ratio=outlier_threshold_ratio,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        annotations_path=resolved_annotations_path,
    )
    state.pose_data_cache[cache_key] = pose_data
    return calibrations, pose_data


def existing_annotation_path_for_keypoints(state: SharedAppState, keypoints_path: Path) -> Path | None:
    """Return the annotation file to use for a keypoints file when it exists."""

    state_annotation_path = getattr(state, "annotation_path_var", None)
    state_annotation_value = "" if state_annotation_path is None else str(state_annotation_path.get()).strip()
    candidates: list[Path] = []
    if state_annotation_value:
        candidates.append(ROOT / state_annotation_value)
    candidates.append(default_annotation_path(keypoints_path))
    for candidate in candidates:
        try:
            if Path(candidate).exists():
                return Path(candidate)
        except Exception:
            continue
    return None


def annotation_only_pose_data(
    pose_data: PoseData,
    *,
    keypoints_path: Path,
    annotations_path: Path | None,
) -> PoseData:
    """Return one sparse pose-data view containing only manual annotations."""

    payload = load_annotation_payload(annotations_path, keypoints_path=keypoints_path)
    sparse_keypoints, sparse_scores = apply_annotations_to_pose_arrays(
        keypoints=np.full_like(np.asarray(pose_data.keypoints, dtype=float), np.nan),
        scores=np.zeros_like(np.asarray(pose_data.scores, dtype=float)),
        camera_names=list(pose_data.camera_names),
        frames=np.asarray(pose_data.frames, dtype=int),
        keypoint_names=COCO17,
        payload=payload,
    )
    return replace(pose_data, keypoints=sparse_keypoints, scores=sparse_scores)


def available_model_pose_modes(state: SharedAppState, keypoints_path: Path) -> list[str]:
    """Return the 2D source modes available in the Models tab."""

    modes = ["raw", "cleaned"]
    if existing_annotation_path_for_keypoints(state, keypoints_path) is not None:
        modes.insert(1, "annotated")
    return modes


def get_pose_data_with_correction(
    state: SharedAppState,
    *,
    keypoints_path: Path,
    calib_path: Path,
    max_frames: int | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
    data_mode: str = "cleaned",
    smoothing_window: int = 9,
    outlier_threshold_ratio: float = 0.10,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
    correction_mode: str = "none",
):
    calibrations, pose_data = get_cached_pose_data(
        state,
        keypoints_path=keypoints_path,
        calib_path=calib_path,
        max_frames=max_frames,
        frame_start=frame_start,
        frame_end=frame_end,
        data_mode=data_mode,
        smoothing_window=smoothing_window,
        outlier_threshold_ratio=outlier_threshold_ratio,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    correction_mode = normalize_pose_correction_mode(correction_mode)
    if correction_mode == "none":
        return calibrations, pose_data, None

    if correction_mode == "flip_epipolar":
        flip_method = "epipolar"
    elif correction_mode == "flip_epipolar_fast":
        flip_method = "epipolar_fast"
    elif correction_mode == "flip_epipolar_viterbi":
        flip_method = "epipolar_viterbi"
    elif correction_mode == "flip_epipolar_fast_viterbi":
        flip_method = "epipolar_fast_viterbi"
    else:
        flip_method = "triangulation"
    corrected_pose_data, diagnostics, _compute_time_s, _cache_path, _source = load_or_compute_pose_data_variant_cache(
        output_dir=current_dataset_dir(state),
        pose_data=pose_data,
        calibrations=calibrations,
        correction_mode="flip",
        flip_method=flip_method,
        pose_data_mode=str(data_mode),
        pose_filter_window=int(smoothing_window),
        pose_outlier_threshold_ratio=float(outlier_threshold_ratio),
        pose_amplitude_lower_percentile=float(lower_percentile),
        pose_amplitude_upper_percentile=float(upper_percentile),
        improvement_ratio=float(state.flip_improvement_ratio_var.get()),
        min_gain_px=float(state.flip_min_gain_px_var.get()),
        min_other_cameras=int(state.flip_min_other_cameras_var.get()),
        restrict_to_outliers=bool(state.flip_restrict_to_outliers_var.get()),
        outlier_percentile=float(state.flip_outlier_percentile_var.get()),
        outlier_floor_px=float(state.flip_outlier_floor_px_var.get()),
        tau_px=(
            DEFAULT_EPIPOLAR_THRESHOLD_PX
            if flip_method in {"epipolar", "epipolar_fast", "epipolar_viterbi", "epipolar_fast_viterbi"}
            else DEFAULT_REPROJECTION_THRESHOLD_PX
        ),
        temporal_weight=float(state.flip_temporal_weight_var.get()),
        temporal_tau_px=float(state.flip_temporal_tau_px_var.get()),
    )
    return calibrations, corrected_pose_data, diagnostics


def get_calibration_pose_data(
    state: SharedAppState,
    *,
    keypoints_path: Path,
    calib_path: Path,
    max_frames: int | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
    data_mode: str = "cleaned",
    smoothing_window: int = 9,
    outlier_threshold_ratio: float = 0.10,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
):
    return get_pose_data_with_correction(
        state,
        keypoints_path=keypoints_path,
        calib_path=calib_path,
        max_frames=max_frames,
        frame_start=frame_start,
        frame_end=frame_end,
        data_mode=data_mode,
        smoothing_window=smoothing_window,
        outlier_threshold_ratio=outlier_threshold_ratio,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        correction_mode=current_calibration_correction_mode(state),
    )


def model_preview_cache_metadata(
    *,
    biomod_path: Path,
    keypoints_path: Path,
    calib_path: Path,
    pose_data_mode: str,
    pose_correction_mode: str,
    triangulation_method: str,
    max_frames: int | None,
    frame_start: int | None,
    frame_end: int | None,
    smoothing_window: int,
    outlier_threshold_ratio: float,
    lower_percentile: float,
    upper_percentile: float,
) -> dict[str, object]:
    stat = biomod_path.stat()
    return {
        "preview_cache_version": "qt0_v2",
        "biomod_path": str(biomod_path.resolve()),
        "biomod_mtime_ns": int(stat.st_mtime_ns),
        "biomod_size": int(stat.st_size),
        "keypoints_path": str(keypoints_path.resolve()),
        "calib_path": str(calib_path.resolve()),
        "pose_data_mode": pose_data_mode,
        "pose_correction_mode": pose_correction_mode,
        "triangulation_method": triangulation_method,
        "max_frames": None if max_frames is None else int(max_frames),
        "frame_start": None if frame_start is None else int(frame_start),
        "frame_end": None if frame_end is None else int(frame_end),
        "pose_filter_window": int(smoothing_window),
        "pose_outlier_threshold_ratio": float(outlier_threshold_ratio),
        "pose_amplitude_lower_percentile": float(lower_percentile),
        "pose_amplitude_upper_percentile": float(upper_percentile),
    }


def save_model_preview_cache(
    cache_path: Path,
    *,
    q_t0: np.ndarray,
    support_points: np.ndarray,
    preview_frame_number: int,
    metadata: dict[str, object],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        q_t0=np.asarray(q_t0, dtype=float),
        support_points=np.asarray(support_points, dtype=float),
        preview_frame_number=np.asarray(int(preview_frame_number)),
        metadata=np.asarray(json.dumps(metadata), dtype=object),
    )


def load_model_preview_cache(cache_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with np.load(cache_path, allow_pickle=True) as data:
        if "q_t0" in data:
            q_t0 = np.asarray(data["q_t0"], dtype=float)
        else:
            q_t0 = np.asarray(data["q0"], dtype=float)
        support_points = np.asarray(data["support_points"], dtype=float)
        preview_frame_number = int(np.asarray(data["preview_frame_number"]).item())
    return q_t0, support_points, preview_frame_number


class CheckGroup(ttk.LabelFrame):
    """Groupe de cases à cocher pour les options `--show`."""

    def __init__(self, master, title: str, choices: list[str], defaults: list[str]):
        super().__init__(master, text=title)
        self.vars: dict[str, tk.BooleanVar] = {}
        self.buttons: list[ttk.Checkbutton] = []
        self.set_choices(choices, defaults)

    def selected(self) -> list[str]:
        return [name for name, var in self.vars.items() if var.get()]

    def set_choices(self, choices: list[str], defaults: list[str] | None = None) -> None:
        previous_state = {name: var.get() for name, var in self.vars.items()}
        for button in self.buttons:
            button.destroy()
        self.buttons = []
        self.vars = {}
        defaults = defaults or []
        for idx, choice in enumerate(choices):
            var = tk.BooleanVar(value=previous_state.get(choice, choice in defaults))
            self.vars[choice] = var
            button = ttk.Checkbutton(self, text=choice, variable=var)
            button.grid(row=idx // 3, column=idx % 3, sticky="w", padx=6, pady=2)
            self.buttons.append(button)


class SelectionTable(ttk.Frame):
    """Table multi-selection pour choisir les reconstructions disponibles."""

    def __init__(self, master, title: str, action_label: str | None = None, action_command=None):
        super().__init__(master)
        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=2, pady=(0, 2))
        ttk.Label(header, text=title).pack(side=tk.LEFT)
        if action_label is not None and action_command is not None:
            ttk.Button(header, text=action_label, command=action_command).pack(side=tk.RIGHT)
        self.tree = ttk.Treeview(
            self,
            columns=("index", "label", "family", "frames", "reproj", "path"),
            show="headings",
            height=6,
            selectmode="extended",
        )
        self.tree.heading("index", text="#")
        self.tree.heading("label", text="Reconstruction")
        self.tree.heading("family", text="Family")
        self.tree.heading("frames", text="Frames")
        self.tree.heading("reproj", text="Reproj (px)")
        self.tree.heading("path", text="Path")
        self.tree.column("index", width=42, anchor="center", stretch=False)
        self.tree.column("label", width=220, anchor="w")
        self.tree.column("family", width=90, anchor="w")
        self.tree.column("frames", width=70, anchor="w")
        self.tree.column("reproj", width=95, anchor="w")
        self.tree.column("path", width=360, anchor="w")
        self.tree.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self._default_names: list[str] = []

    def set_rows(self, rows: list[dict[str, object]], default_names: list[str] | None = None) -> None:
        previous = set(self.selected_names())
        self._default_names = list(default_names or [])
        for item in self.tree.get_children():
            self.tree.delete(item)
        row_names = []
        for row_idx, row in enumerate(rows, start=1):
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
                    str(row_idx),
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
        for name in selection:
            if self.tree.exists(name):
                self.tree.selection_add(name)

    def selected_names(self) -> list[str]:
        selected = list(self.tree.selection())
        if selected:
            return selected
        return [name for name in self._default_names if self.tree.exists(name)]


class CommandTab(ttk.Frame):
    """Base class for tabs that build and execute one command."""

    def __init__(
        self,
        master,
        title: str,
        *,
        show_default_buttons: bool = True,
        show_progress: bool = True,
        show_command_preview: bool = True,
        show_output: bool = True,
    ):
        super().__init__(master)
        self.title = title
        self.process: subprocess.Popen[str] | None = None
        self.command_preview = tk.StringVar(value="")
        self.progress_text = tk.StringVar(value="Idle")
        self._profile_total = 0
        self._profile_index = 0
        self._profile_name = ""
        self.primary_action_button: ttk.Button | None = None
        self.primary_run_text = "Run"
        self.primary_stop_text = "Stop"

        self.main = ttk.Frame(self)
        self.main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.buttons_frame: ttk.Frame | None = None
        self.preview_button: ttk.Button | None = None
        self.copy_button: ttk.Button | None = None
        self.run_button: ttk.Button | None = None
        self.stop_button: ttk.Button | None = None
        if show_default_buttons:
            self.buttons_frame = ttk.Frame(self.main)
            self.buttons_frame.pack(fill=tk.X, pady=(8, 8))
            self.preview_button = ttk.Button(self.buttons_frame, text="Preview command", command=self.update_preview)
            self.preview_button.pack(side=tk.LEFT)
            self.copy_button = ttk.Button(self.buttons_frame, text="Copy command", command=self.copy_command)
            self.copy_button.pack(side=tk.LEFT, padx=(8, 0))
            self.run_button = ttk.Button(self.buttons_frame, text="Run", command=self.run_command)
            self.run_button.pack(side=tk.LEFT, padx=(8, 0))
            self.stop_button = ttk.Button(self.buttons_frame, text="Stop", command=self.stop_command)
            self.stop_button.pack(side=tk.LEFT, padx=(8, 0))

        self.progress_row: ttk.Frame | None = None
        self.progress_bar: ttk.Progressbar | None = None
        if show_progress:
            self.progress_row = ttk.Frame(self.main)
            self.progress_row.pack(fill=tk.X, pady=(0, 8))
            self.progress_bar = ttk.Progressbar(self.progress_row, mode="determinate", maximum=1.0, value=0.0)
            self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Label(self.progress_row, textvariable=self.progress_text, width=48).pack(side=tk.LEFT, padx=(10, 0))

        self.command_preview_label: ttk.Label | None = None
        if show_command_preview:
            self.command_preview_label = ttk.Label(
                self.main, textvariable=self.command_preview, wraplength=1100, justify=tk.LEFT
            )
            self.command_preview_label.pack(fill=tk.X, pady=(0, 8))

        self.output: ScrolledText | None = None
        if show_output:
            self.output = ScrolledText(self.main, height=18, wrap=tk.WORD)
            self.output.pack(fill=tk.BOTH, expand=True)

    def build_command(self) -> list[str]:
        raise NotImplementedError

    def update_preview(self) -> None:
        self.command_preview.set(shlex.join(self.build_command()))

    def set_run_button_text(self, text: str) -> None:
        if self.run_button is not None:
            self.run_button.configure(text=text)

    def attach_primary_action_button(self, button: ttk.Button, run_text: str = "Run", stop_text: str = "Stop") -> None:
        """Register a tab-local action button that mirrors the process state."""
        self.primary_action_button = button
        self.primary_run_text = run_text
        self.primary_stop_text = stop_text
        self.update_action_button_state()

    def hide_default_command_buttons(self) -> None:
        """Hide only the shared command buttons while keeping progress and logs visible."""
        if self.buttons_frame is not None:
            self.buttons_frame.pack_forget()

    def hide_preview_copy_buttons(self) -> None:
        """Hide only preview/copy controls while keeping run/progress/output available."""

        if self.preview_button is not None:
            self.preview_button.pack_forget()
        if self.copy_button is not None:
            self.copy_button.pack_forget()

    def hide_command_controls(self) -> None:
        if self.buttons_frame is not None:
            self.buttons_frame.pack_forget()
        if self.progress_row is not None:
            self.progress_row.pack_forget()
        if self.command_preview_label is not None:
            self.command_preview_label.pack_forget()
        if self.output is not None:
            self.output.pack_forget()

    def update_action_button_state(self) -> None:
        """Keep the primary action button label in sync with the process state."""
        if self.primary_action_button is None:
            return
        is_running = self.process is not None and self.process.poll() is None
        self.primary_action_button.configure(text=self.primary_stop_text if is_running else self.primary_run_text)

    def toggle_run_command(self) -> None:
        """Use a single button for start/stop actions."""
        if self.process is not None and self.process.poll() is None:
            self.stop_command()
        else:
            self.run_command()

    def copy_command(self) -> None:
        cmd = shlex.join(self.build_command())
        self.clipboard_clear()
        self.clipboard_append(cmd)
        self.command_preview.set(cmd)

    def append_output(self, text: str) -> None:
        if self.output is None:
            return
        self.output.insert(tk.END, text)
        self.output.see(tk.END)

    def reset_progress(self) -> None:
        if self.progress_bar is not None:
            self.progress_bar.stop()
            self.progress_bar.configure(mode="determinate", maximum=1.0, value=0.0)
        self.progress_text.set("Idle")
        self._profile_total = 0
        self._profile_index = 0
        self._profile_name = ""

    def start_indeterminate_progress(self, label: str = "Running...") -> None:
        if self.progress_bar is not None:
            self.progress_bar.stop()
            self.progress_bar.configure(mode="indeterminate")
            self.progress_bar.start(12)
        self.progress_text.set(label)

    def set_progress(self, current: float, total: float, label: str) -> None:
        total = max(float(total), 1.0)
        current = min(max(float(current), 0.0), total)
        if self.progress_bar is not None:
            self.progress_bar.stop()
            self.progress_bar.configure(mode="determinate", maximum=total, value=current)
        self.progress_text.set(label)

    def prepare_progress(self) -> None:
        self.reset_progress()
        self.start_indeterminate_progress("Running...")

    def handle_output_line(self, line: str) -> None:
        stripped = line.strip()
        if not stripped:
            return
        profile_match = PROFILE_RE.search(stripped)
        if profile_match:
            self._profile_index = int(profile_match.group(1))
            self._profile_total = int(profile_match.group(2))
            self._profile_name = profile_match.group(3).strip()
            base = max(self._profile_index - 1, 0)
            label = f"Profil {self._profile_index}/{self._profile_total}: {self._profile_name}"
            self.set_progress(base, max(self._profile_total, 1), label)
            return
        step_match = STEP_RE.search(stripped)
        if step_match:
            step_idx = int(step_match.group(1))
            step_total = int(step_match.group(2))
            step_label = step_match.group(3).strip() or f"Etape {step_idx}/{step_total}"
            if self._profile_total > 0 and self._profile_index > 0:
                current = (self._profile_index - 1) + (step_idx / max(step_total, 1))
                label = f"Profil {self._profile_index}/{self._profile_total}: {self._profile_name} | {step_label}"
                self.set_progress(current, self._profile_total, label)
            else:
                self.set_progress(step_idx, step_total, step_label)

    def run_command(self) -> None:
        if self.process is not None and self.process.poll() is None:
            messagebox.showinfo("Command running", f"{self.title} est déjà en cours.")
            return
        cmd = self.build_command()
        self.update_preview()
        self.append_output(f"\n$ {shlex.join(cmd)}\n")
        self.prepare_progress()
        self.process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.update_action_button_state()
        threading.Thread(target=self._stream_output, daemon=True).start()

    def _stream_output(self) -> None:
        assert self.process is not None
        if self.process.stdout is not None:
            for line in self.process.stdout:
                self.after(0, self.append_output, line)
                self.after(0, self.handle_output_line, line)
        return_code = self.process.wait()
        self.after(0, self.append_output, f"\n[exit code {return_code}]\n")
        self.after(0, self.finish_progress, return_code)

    def finish_progress(self, return_code: int) -> None:
        if self.progress_bar is not None:
            self.progress_bar.stop()
        if return_code == 0:
            if self.progress_bar is not None:
                maximum = float(self.progress_bar.cget("maximum") or 1.0)
                self.progress_bar.configure(mode="determinate", value=maximum)
            self.progress_text.set("Done")
            try:
                self.on_command_success()
            except Exception as exc:
                self.append_output(f"\n[post-run refresh error: {exc}]\n")
        else:
            self.progress_text.set(f"Failed (exit {return_code})")
        self.update_action_button_state()

    def stop_command(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.append_output("\n[process terminated]\n")
            if self.progress_bar is not None:
                self.progress_bar.stop()
            self.progress_text.set("Stopped")
        self.update_action_button_state()

    def on_command_success(self) -> None:
        pass

    @staticmethod
    def parse_extra_args(raw: str) -> list[str]:
        return shlex.split(raw) if raw.strip() else []


class PipelineTab(CommandTab):
    def __init__(self, master):
        super().__init__(master, "Pipeline")
        form = ttk.LabelFrame(self.main, text="vitpose_ekf_pipeline.py")
        form.pack(fill=tk.X, pady=(0, 8), before=self.output)

        self.calib = LabeledEntry(
            form,
            "Calib",
            "inputs/calibration/Calib.toml",
            browse=True,
            filetypes=(("TOML calibration", "*.toml"), ("All files", "*.*")),
        )
        self.calib.pack(fill=tk.X, padx=8, pady=4)
        self.keypoints = LabeledEntry(
            form,
            "Keypoints",
            "inputs/keypoints/1_partie_0429_keypoints.json",
            browse=True,
            filetypes=(("2D keypoints JSON", "*_keypoints.json"), ("JSON files", "*.json"), ("All files", "*.*")),
        )
        self.keypoints.pack(fill=tk.X, padx=8, pady=4)
        self.output_dir = LabeledEntry(form, "Output dir", "output/vitpose_full", browse=True, directory=True)
        self.output_dir.pack(fill=tk.X, padx=8, pady=4)
        self.biomod = LabeledEntry(form, "bioMod", "output/vitpose_full/vitpose_chain.bioMod", browse=True)
        self.biomod.pack(fill=tk.X, padx=8, pady=4)
        self.reconstruction_cache = LabeledEntry(form, "Recon cache", "", browse=True)
        self.reconstruction_cache.pack(fill=tk.X, padx=8, pady=4)
        self.model_cache = LabeledEntry(form, "Model cache", "", browse=True)
        self.model_cache.pack(fill=tk.X, padx=8, pady=4)
        self.biorbd_cache = LabeledEntry(form, "EKF3D cache", "", browse=True)
        self.biorbd_cache.pack(fill=tk.X, padx=8, pady=4)

        row0 = ttk.Frame(form)
        row0.pack(fill=tk.X, padx=8, pady=4)
        self.fps = LabeledEntry(row0, "FPS", "120")
        self.fps.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        mode_label = ttk.Label(row0, text="2D mode", width=10)
        mode_label.pack(side=tk.LEFT)
        self.pose_data_mode = tk.StringVar(value="cleaned")
        pose_mode_box = ttk.Combobox(
            row0, textvariable=self.pose_data_mode, values=["raw", "annotated", "cleaned"], width=12, state="readonly"
        )
        pose_mode_box.pack(side=tk.LEFT, padx=(0, 6))
        self.subject_mass = LabeledEntry(form, "Subject mass", "55")
        self.subject_mass.pack(fill=tk.X, padx=8, pady=4)
        self.calib.set_tooltip("Fichier de calibration multivue utilise par tout le pipeline.")
        self.keypoints.set_tooltip("JSON des detections 2D a reconstruire.")
        self.output_dir.set_tooltip("Dossier de sortie principal pour les caches, modeles et reconstructions.")
        self.biomod.set_tooltip("Chemin du bioMod cible a generer ou reutiliser.")
        self.reconstruction_cache.set_tooltip(
            "Cache de reconstruction/triangulation a relire si les options correspondent."
        )
        self.model_cache.set_tooltip("Cache de construction du modele a relire si les options correspondent.")
        self.biorbd_cache.set_tooltip("Cache des etats EKF 3D biorbd.")
        self.fps.set_tooltip(
            "Frequence d'echantillonnage supposee pour les reconstructions et les derivees temporelles."
        )
        attach_tooltip(
            mode_label,
            "Version des detections 2D utilisee par le pipeline: brute ou nettoyee apres rejet des points aberrants.",
        )
        attach_tooltip(
            pose_mode_box,
            "Version des detections 2D utilisee par le pipeline: brute ou nettoyee apres rejet des points aberrants.",
        )
        self.subject_mass.set_tooltip("Masse utilisee pour les proprietes inertielles du sujet dans le modele.")

        row1 = ttk.Frame(form)
        row1.pack(fill=tk.X, padx=8, pady=4)
        self.max_frames = LabeledEntry(row1, "Max frames", "")
        self.max_frames.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.triang_workers = LabeledEntry(row1, "Triang workers", "6")
        self.triang_workers.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.min_cams = LabeledEntry(row1, "Min cams", "3")
        self.min_cams.pack(side=tk.LEFT, fill=tk.X, expand=True)

        row2 = ttk.Frame(form)
        row2.pack(fill=tk.X, padx=8, pady=4)
        self.reproj_thresh = LabeledEntry(row2, "Reproj px", "15")
        self.reproj_thresh.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.epi_thresh = LabeledEntry(row2, "Epi px", "15")
        self.epi_thresh.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.measurement_noise = LabeledEntry(row2, "Meas noise", "1.5")
        self.measurement_noise.pack(side=tk.LEFT, fill=tk.X, expand=True)

        row3 = ttk.Frame(form)
        row3.pack(fill=tk.X, padx=8, pady=4)
        self.process_noise = LabeledEntry(row3, "Proc noise", "1.0")
        self.process_noise.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.coherence_floor = LabeledEntry(row3, "Coh floor", "0.35")
        self.coherence_floor.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.min_frame_coh = LabeledEntry(row3, "Min frame coh", "0.0")
        self.min_frame_coh.pack(side=tk.LEFT, fill=tk.X, expand=True)

        row_clean = ttk.Frame(form)
        row_clean.pack(fill=tk.X, padx=8, pady=4)
        self.pose_filter_window = LabeledEntry(row_clean, "Filter window", "9")
        self.pose_filter_window.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.pose_outlier_ratio = LabeledEntry(row_clean, "Outlier ratio", "0.10")
        self.pose_outlier_ratio.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.pose_p_low = LabeledEntry(row_clean, "P low", "5")
        self.pose_p_low.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.pose_p_high = LabeledEntry(row_clean, "P high", "95")
        self.pose_p_high.pack(side=tk.LEFT, fill=tk.X, expand=True)

        row4 = ttk.Frame(form)
        row4.pack(fill=tk.X, padx=8, pady=4)
        self.flight_threshold = LabeledEntry(row4, "Flight h (m)", "1.5")
        self.flight_threshold.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.flight_frames = LabeledEntry(row4, "Flight frames", "2")
        self.flight_frames.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.biorbd_noise = LabeledEntry(row4, "EKF3D noise", "1e-8")
        self.biorbd_noise.pack(side=tk.LEFT, fill=tk.X, expand=True)

        row5 = ttk.Frame(form)
        row5.pack(fill=tk.X, padx=8, pady=4)
        self.biorbd_error = LabeledEntry(row5, "EKF3D error", "1e-4")
        self.biorbd_error.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        coherence_label = ttk.Label(row5, text="Coherence", width=12)
        coherence_label.pack(side=tk.LEFT)
        self.coherence_method = tk.StringVar(value="epipolar")
        coherence_box = ttk.Combobox(
            row5,
            textvariable=self.coherence_method,
            values=list(SUPPORTED_COHERENCE_METHODS),
            width=24,
            state="readonly",
        )
        coherence_box.pack(side=tk.LEFT, padx=(0, 6))
        triang_label = ttk.Label(row5, text="Triangulation", width=12)
        triang_label.pack(side=tk.LEFT)
        self.triang_method = tk.StringVar(value="exhaustive")
        triang_box = ttk.Combobox(
            row5, textvariable=self.triang_method, values=["once", "greedy", "exhaustive"], width=12, state="readonly"
        )
        triang_box.pack(side=tk.LEFT)

        self.reproj_thresh.set_tooltip(
            "Seuil maximal d'erreur de reprojection pour accepter une reconstruction triangulee. Plus petit = plus strict."
        )
        self.epi_thresh.set_tooltip(
            "Seuil epipolaire en pixels. Il sert a juger si deux vues 2D sont geometriquement coherentes entre elles."
        )
        self.measurement_noise.set_tooltip(
            "Bruit de mesure du filtre EKF 2D. Plus grand = moins de confiance dans les keypoints 2D."
        )
        self.process_noise.set_tooltip(
            "Bruit du modele de prediction EKF 2D. Plus grand = prediction plus souple, moins lisse."
        )
        self.coherence_floor.set_tooltip(
            "Plancher applique a la coherence multivue avant de ponderer les mesures. Evite qu'une camera devienne quasi inutile trop facilement."
        )
        self.min_frame_coh.set_tooltip(
            "Seuil moyen minimal de coherence pour autoriser une mise a jour quand le mode 'skip-low-coherence-updates' est actif."
        )
        self.pose_filter_window.set_tooltip(
            "Taille de fenetre du lissage 2D utilise pour construire une reference filtree par keypoint."
        )
        self.pose_outlier_ratio.set_tooltip(
            "Un point 2D est rejete si son ecart a la version filtree depasse ce ratio de l'amplitude robuste du mouvement."
        )
        self.pose_p_low.set_tooltip("Percentile bas utilise pour estimer l'amplitude robuste du mouvement 2D.")
        self.pose_p_high.set_tooltip("Percentile haut utilise pour estimer l'amplitude robuste du mouvement 2D.")
        self.flight_threshold.set_tooltip(
            "Seuil de hauteur minimale de tous les marqueurs pour considerer la phase aerienne."
        )
        self.flight_frames.set_tooltip("Nombre de frames consecutives au-dessus du seuil avant d'activer l'etat 'AIR'.")
        self.biorbd_noise.set_tooltip(
            "Bruit des marqueurs 3D pour l'EKF 3D. Plus grand = filtre plus lisse, moins de confiance dans les points 3D."
        )
        self.biorbd_error.set_tooltip(
            "Erreur d'etat initiale du Kalman 3D. Plus grand = plus de liberte pour corriger rapidement."
        )
        self.max_frames.set_tooltip("Nombre maximal de frames traitees; vide = toute la sequence.")
        self.triang_workers.set_tooltip("Nombre de workers paralleles utilises pour la triangulation.")
        self.min_cams.set_tooltip("Nombre minimal de cameras valides pour accepter une triangulation.")
        attach_tooltip(
            coherence_label,
            "Source de la pondération multivue pour l'EKF 2D: épipolaire, ou cohérence dérivée d'une triangulation once, greedy, ou exhaustive.",
        )
        attach_tooltip(
            coherence_box,
            "Les variantes triangulation_* réutilisent la cohérence issue de la variante de triangulation indiquée dans le nom.",
        )
        attach_tooltip(triang_label, "Choisit la variante de triangulation 3D principale: once, greedy, ou exhaustive.")
        attach_tooltip(
            triang_box,
            "once: une seule DLT pondérée. greedy: suppression gloutonne des pires vues. exhaustive: teste plus de combinaisons et est la plus robuste.",
        )

        checks = ttk.Frame(form)
        checks.pack(fill=tk.X, padx=8, pady=6)
        self.compare_var = tk.BooleanVar(value=True)
        self.flip_acc_var = tk.BooleanVar(value=True)
        self.model_only_var = tk.BooleanVar(value=False)
        self.triang_only_var = tk.BooleanVar(value=False)
        self.reuse_triang_var = tk.BooleanVar(value=False)
        self.skip_low_coh_var = tk.BooleanVar(value=False)
        self.lock_dof_var = tk.BooleanVar(value=False)
        self.animate_var = tk.BooleanVar(value=False)
        check_tooltips = {
            "compare-biorbd-kalman": "Calcule aussi la comparaison directe avec l'EKF 3D biorbd.",
            "run-ekf-2d-flip-acc": "Lance la variante EKF 2D avec correction gauche/droite et predicteur acceleration.",
            "model-only": "Construit seulement le modele sans lancer les filtres.",
            "triangulate-only": "S'arrete apres la triangulation 3D.",
            "reuse-triangulation": "Reutilise une triangulation en cache si les options correspondent.",
            "skip-low-coherence-updates": "Ignore les corrections EKF 2D quand la coherence multivue est trop faible.",
            "enable-dof-locking": "Active le verrouillage de certains DoF pour stabiliser le modele.",
            "animate": "Produit aussi les sorties d'animation du pipeline principal.",
        }
        for text, var in [
            ("compare-biorbd-kalman", self.compare_var),
            ("run-ekf-2d-flip-acc", self.flip_acc_var),
            ("model-only", self.model_only_var),
            ("triangulate-only", self.triang_only_var),
            ("reuse-triangulation", self.reuse_triang_var),
            ("skip-low-coherence-updates", self.skip_low_coh_var),
            ("enable-dof-locking", self.lock_dof_var),
            ("animate", self.animate_var),
        ]:
            widget = ttk.Checkbutton(checks, text=text, variable=var)
            widget.pack(side=tk.LEFT, padx=(0, 12))
            attach_tooltip(widget, check_tooltips[text])

        self.extra = LabeledEntry(form, "Extra args", "")
        self.extra.pack(fill=tk.X, padx=8, pady=4)
        self.extra.set_tooltip("Arguments CLI additionnels passes tels quels a vitpose_ekf_pipeline.py.")

        status_box = ttk.LabelFrame(self.main, text="Reconstructions et caches")
        status_box.pack(fill=tk.BOTH, expand=True, pady=(0, 8), before=self.output)
        top = ttk.Frame(status_box)
        top.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(top, text="Refresh cache status", command=self.refresh_status_table).pack(side=tk.LEFT)
        ttk.Label(
            top,
            text="Les reconstructions optionnelles sont pilotees par les flags ci-dessus; le tableau lit les caches actuels.",
        ).pack(side=tk.LEFT, padx=(10, 0))
        cols = ("label", "cached", "latest", "frames", "reproj_mean", "reproj_std", "path")
        self.status_tree = ttk.Treeview(status_box, columns=cols, show="headings", height=8)
        headings = {
            "label": "Reconstruction",
            "cached": "En cache",
            "latest": "A jour",
            "frames": "Frames",
            "reproj_mean": "Reproj mean (px)",
            "reproj_std": "Reproj std (px)",
            "path": "Source",
        }
        widths = {
            "label": 170,
            "cached": 70,
            "latest": 70,
            "frames": 70,
            "reproj_mean": 110,
            "reproj_std": 110,
            "path": 430,
        }
        for col in cols:
            self.status_tree.heading(col, text=headings[col])
            self.status_tree.column(col, width=widths[col], anchor="w")
        self.status_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.refresh_status_table()

    def build_command(self) -> list[str]:
        cmd = [
            sys.executable,
            "vitpose_ekf_pipeline.py",
            "--calib",
            self.calib.get(),
            "--keypoints",
            self.keypoints.get(),
            "--fps",
            self.fps.get(),
            "--pose-data-mode",
            self.pose_data_mode.get(),
            "--subject-mass-kg",
            self.subject_mass.get(),
            "--output-dir",
            self.output_dir.get(),
            "--biomod",
            self.biomod.get(),
            "--reprojection-threshold-px",
            self.reproj_thresh.get(),
            "--epipolar-threshold-px",
            self.epi_thresh.get(),
            "--min-cameras-for-triangulation",
            self.min_cams.get(),
            "--coherence-method",
            self.coherence_method.get(),
            "--triangulation-method",
            self.triang_method.get(),
            "--triangulation-workers",
            self.triang_workers.get(),
            "--measurement-noise-scale",
            self.measurement_noise.get(),
            "--process-noise-scale",
            self.process_noise.get(),
            "--pose-filter-window",
            self.pose_filter_window.get(),
            "--pose-outlier-threshold-ratio",
            self.pose_outlier_ratio.get(),
            "--pose-amplitude-lower-percentile",
            self.pose_p_low.get(),
            "--pose-amplitude-upper-percentile",
            self.pose_p_high.get(),
            "--coherence-confidence-floor",
            self.coherence_floor.get(),
            "--min-frame-coherence-for-update",
            self.min_frame_coh.get(),
            "--flight-height-threshold-m",
            self.flight_threshold.get(),
            "--flight-min-consecutive-frames",
            self.flight_frames.get(),
            "--biorbd-kalman-noise-factor",
            self.biorbd_noise.get(),
            "--biorbd-kalman-error-factor",
            self.biorbd_error.get(),
        ]
        selected_cameras = current_selected_camera_names(self.state)
        if selected_cameras:
            cmd.extend(["--camera-names", ",".join(selected_cameras)])
        if self.reconstruction_cache.get():
            cmd.extend(["--reconstruction-cache", self.reconstruction_cache.get()])
        if self.model_cache.get():
            cmd.extend(["--model-cache", self.model_cache.get()])
        if self.biorbd_cache.get():
            cmd.extend(["--biorbd-kalman-cache", self.biorbd_cache.get()])
        if self.compare_var.get():
            cmd.append("--compare-biorbd-kalman")
        if self.flip_acc_var.get():
            cmd.append("--run-ekf-2d-flip-acc")
        cmd.extend(["--root-unwrap-mode", "off"])
        if self.model_only_var.get():
            cmd.append("--model-only")
        if self.triang_only_var.get():
            cmd.append("--triangulate-only")
        if self.reuse_triang_var.get():
            cmd.append("--reuse-triangulation")
        if self.skip_low_coh_var.get():
            cmd.append("--skip-low-coherence-updates")
        if self.lock_dof_var.get():
            cmd.append("--enable-dof-locking")
        if self.animate_var.get():
            cmd.append("--animate")
        if self.max_frames.get():
            cmd.extend(["--max-frames", self.max_frames.get()])
        cmd.extend(self.parse_extra_args(self.extra.get()))
        return cmd

    def refresh_status_table(self) -> None:
        for item in self.status_tree.get_children():
            self.status_tree.delete(item)
        catalog = discover_reconstruction_catalog(ROOT / self.output_dir.get(), ROOT / "inputs/trc/1_partie_0429.trc")
        for row in catalog:
            self.status_tree.insert(
                "",
                "end",
                values=(
                    row["label"],
                    "oui" if row["cached"] else "non",
                    "-" if row.get("is_latest") is None else ("oui" if row.get("is_latest") else "non"),
                    row["frames"] if row["frames"] is not None else "-",
                    f"{row['reproj_mean']:.2f}" if row["reproj_mean"] is not None else "-",
                    f"{row['reproj_std']:.2f}" if row["reproj_std"] is not None else "-",
                    row["path"],
                ),
            )


class DualAnimationTab(CommandTab):
    def __init__(self, master, state: SharedAppState):
        super().__init__(
            master,
            "3D animation",
            show_default_buttons=False,
            show_command_preview=False,
            show_output=False,
        )
        self.state = state
        self.bundle: dict[str, object] | None = None
        self._view_state: tuple[float, float, float] | None = None
        self._dragging_frame_scale = False
        self.uses_shared_reconstruction_panel = True
        self.shared_reconstruction_selectmode = "extended"

        body = ttk.Panedwindow(self.main, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=3)
        body.add(right, weight=7)

        form = ttk.LabelFrame(left, text="animation/animate_dual_stick_comparison.py")
        form.pack(fill=tk.BOTH, expand=False, padx=(0, 8), pady=(0, 8))

        self.output_gif = LabeledEntry(form, "GIF name", self.default_gif_name())
        self.output_gif.pack(fill=tk.X, padx=8, pady=4)

        row2 = ttk.Frame(form)
        row2.pack(fill=tk.X, padx=8, pady=4)
        self.fps = LabeledEntry(row2, "GIF fps", "12", label_width=7, entry_width=5)
        self.fps.pack(side=tk.LEFT, padx=(0, 6))
        self.stride = LabeledEntry(row2, "Stride", "5", label_width=6, entry_width=4)
        self.stride.pack(side=tk.LEFT, padx=(0, 6))

        row3 = ttk.Frame(form)
        row3.pack(fill=tk.X, padx=8, pady=4)
        self.show_trunk_frames_var = tk.BooleanVar(value=False)
        show_trunk_check = ttk.Checkbutton(
            row3, text="Show trunk frames", variable=self.show_trunk_frames_var, command=self.refresh_preview
        )
        show_trunk_check.pack(side=tk.LEFT, padx=(0, 8))
        self.show_trampoline_var = tk.BooleanVar(value=False)
        show_trampoline_check = ttk.Checkbutton(
            row3, text="Show trampoline", variable=self.show_trampoline_var, command=self.refresh_preview
        )
        show_trampoline_check.pack(side=tk.LEFT, padx=(0, 8))

        row4 = ttk.Frame(form)
        row4.pack(fill=tk.X, padx=8, pady=4)
        self.crop_var = tk.BooleanVar(value=False)
        crop_check = ttk.Checkbutton(row4, text="Crop", variable=self.crop_var, command=self.refresh_preview)
        crop_check.pack(side=tk.LEFT, padx=(0, 8))
        self.marker_size = LabeledEntry(row4, "Marker size", "8", label_width=10, entry_width=4)
        self.marker_size.pack(side=tk.LEFT, padx=(0, 6))

        row5 = ttk.Frame(form)
        row5.pack(fill=tk.X, padx=8, pady=4)
        self.generate_button = ttk.Button(row5, text="GENERATE GIF", command=self.toggle_run_command)
        self.generate_button.pack(side=tk.RIGHT)
        self.attach_primary_action_button(self.generate_button, run_text="GENERATE GIF", stop_text="STOP")

        preview_box = ttk.LabelFrame(right, text="Preview 3D")
        preview_box.pack(fill=tk.BOTH, expand=True)
        preview_controls = ttk.Frame(preview_box)
        preview_controls.pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(preview_controls, text="Load preview", command=self.load_preview).pack(side=tk.LEFT)
        ttk.Button(preview_controls, text="Refresh frame", command=self.refresh_preview).pack(side=tk.LEFT, padx=(8, 0))
        self.frame_var = tk.IntVar(value=0)
        self.frame_scale = ttk.Scale(
            preview_controls,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            variable=self.frame_var,
            command=lambda _value: self.refresh_preview(),
        )
        self.frame_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(12, 8))
        self.frame_label = ttk.Label(preview_controls, text="frame 0")
        self.frame_label.pack(side=tk.LEFT)

        self.preview_figure = Figure(figsize=(8, 6))
        self.preview_canvas = FigureCanvasTkAgg(self.preview_figure, master=preview_box)
        self.preview_canvas_widget = self.preview_canvas.get_tk_widget()
        self.preview_canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.preview_toolbar = NavigationToolbar2Tk(self.preview_canvas, preview_box, pack_toolbar=False)
        self.preview_toolbar.update()
        self.preview_toolbar.pack(fill=tk.X)
        self._bind_frame_navigation(self.preview_canvas_widget)
        self._bind_frame_navigation(self.frame_scale)

        self.extra = LabeledEntry(form, "Extra args", "")
        self.extra.pack(fill=tk.X, padx=8, pady=4)
        self.output_gif.set_tooltip(
            "Nom du GIF 3D exporté. Le fichier sera enregistré automatiquement dans ./output/<trial>/figures/."
        )
        self.fps.set_tooltip("Frequence d'images du GIF exporte.")
        self.stride.set_tooltip("Une frame sur N est exportee dans le GIF.")
        self.marker_size.set_tooltip("Taille visuelle des marqueurs du squelette 3D.")
        attach_tooltip(show_trunk_check, "Affiche le repere du tronc pour chaque reconstruction selectionnee.")
        attach_tooltip(show_trampoline_check, "Affiche un contour simplifié du trampoline dans le preview et le GIF.")
        attach_tooltip(
            crop_check, "Recadre la vue sur la frame courante. Sinon, garde les limites globales de tout l'essai."
        )
        attach_tooltip(self.generate_button, "Genere le GIF 3D, puis devient STOP tant que l'export est en cours.")
        attach_tooltip(self.frame_scale, "Slider de navigation temporelle du preview 3D.")
        attach_tooltip(self.frame_label, "Index de frame actuellement affiche dans le preview 3D.")
        self.extra.set_tooltip(
            "Options CLI supplémentaires pour animation/animate_dual_stick_comparison.py, par exemple: --align-root"
        )
        self.state.keypoints_var.trace_add("write", lambda *_args: self.sync_dataset_defaults())
        self.state.output_root_var.trace_add("write", lambda *_args: self.sync_dataset_defaults())
        self.state.register_reconstruction_listener(self.refresh_available_reconstructions)
        self.refresh_available_reconstructions()

    def default_gif_name(self) -> str:
        """Return the default GIF filename for the current dataset."""

        return f"{current_dataset_name(self.state)}_3d_animation.gif"

    def resolved_output_gif_path(self) -> Path:
        """Resolve the GIF output path inside the current dataset figures directory."""

        raw_name = self.output_gif.get().strip()
        gif_name = Path(raw_name).name if raw_name else self.default_gif_name()
        if not gif_name.lower().endswith(".gif"):
            gif_name = f"{gif_name}.gif"
        return current_figures_dir(self.state) / gif_name

    def configure_shared_reconstruction_panel(self, panel: SharedReconstructionPanel) -> None:
        panel.configure_for_consumer(
            title="Reconstructions | 3D animation",
            refresh_callback=self.refresh_available_reconstructions,
            selection_callback=self._on_reconstruction_selection_changed,
            selectmode=self.shared_reconstruction_selectmode,
        )
        self.refresh_available_reconstructions()

    def selected_reconstruction_names(self) -> list[str]:
        return list(self.state.shared_reconstruction_selection)

    def _publish_reconstruction_rows(self, rows: list[dict[str, object]], defaults: list[str]) -> None:
        panel = self.state.shared_reconstruction_panel
        if panel is not None and self.state.active_reconstruction_consumer is self:
            panel.set_rows(rows, defaults)

    def _on_reconstruction_selection_changed(self) -> None:
        if self.bundle is None:
            self.load_preview()
        else:
            self.refresh_preview()

    def _bind_frame_navigation(self, widget: tk.Widget) -> None:
        if widget is self.frame_scale:
            widget.bind("<Button-1>", self._on_frame_scale_click)
            widget.bind("<B1-Motion>", self._on_frame_scale_drag)
            widget.bind("<ButtonRelease-1>", self._on_frame_scale_release)
        else:
            widget.bind("<Enter>", lambda _event: widget.focus_set())
        widget.bind("<Left>", lambda _event: self.step_frame(-1))
        widget.bind("<Right>", lambda _event: self.step_frame(1))

    def _frame_from_scale_event(self, event) -> int:
        """Map a mouse event on the ttk.Scale trough to the target frame index."""
        return frame_from_slider_click(
            x=event.x,
            width=self.frame_scale.winfo_width(),
            from_value=self.frame_scale.cget("from"),
            to_value=self.frame_scale.cget("to"),
        )

    def _on_frame_scale_click(self, event) -> str:
        self._dragging_frame_scale = True
        self.frame_scale.focus_set()
        frame = self._frame_from_scale_event(event)
        self.frame_var.set(frame)
        self.refresh_preview()
        return "break"

    def _on_frame_scale_drag(self, event) -> str:
        if not self._dragging_frame_scale:
            return "break"
        self.frame_var.set(self._frame_from_scale_event(event))
        self.refresh_preview()
        return "break"

    def _on_frame_scale_release(self, event) -> str:
        if not self._dragging_frame_scale:
            return "break"
        self._dragging_frame_scale = False
        self.frame_var.set(self._frame_from_scale_event(event))
        self.refresh_preview()
        return "break"

    def step_frame(self, delta: int) -> str:
        if self.bundle is None:
            return "break"
        recon_3d = self.bundle.get("recon_3d", {})
        available = {name: points for name, points in recon_3d.items() if points is not None}
        if not available:
            return "break"
        max_frame = min(points.shape[0] for points in available.values()) - 1
        if max_frame < 0:
            return "break"
        current = int(round(self.frame_var.get()))
        next_frame = step_frame_index(current=current, delta=delta, max_frame=max_frame)
        if next_frame != current:
            self.frame_var.set(next_frame)
            self.refresh_preview()
        return "break"

    def sync_dataset_defaults(self) -> None:
        self.output_gif.var.set(self.default_gif_name())
        self.refresh_available_reconstructions()

    def refresh_available_reconstructions(self) -> None:
        output_dir, preview_state = current_dataset_preview_state(
            self.state,
            bundle=self.bundle,
            preferred_names=["ekf_3d", "ekf_2d_flip_acc", "ekf_2d_acc", "pose2sim"],
            fallback_count=4,
        )
        self._publish_reconstruction_rows(preview_state.rows, preview_state.defaults)
        if self.bundle is not None:
            try:
                sources = dataset_source_paths(
                    output_dir,
                    pose2sim_trc=optional_root_relative_path(self.state.pose2sim_trc_var.get()),
                )
                self.bundle = get_cached_preview_bundle(
                    self.state,
                    output_dir,
                    resolve_preview_biomod(output_dir),
                    Path(sources["pose2sim_trc"]),
                    align_root=False,
                )
                _output_dir, preview_state = current_dataset_preview_state(
                    self.state,
                    bundle=self.bundle,
                    preferred_names=["ekf_3d", "ekf_2d_flip_acc", "ekf_2d_acc", "pose2sim"],
                    fallback_count=4,
                )
                self._publish_reconstruction_rows(preview_state.rows, preview_state.defaults)
                self.frame_scale.configure(to=preview_state.max_frame)
                self.refresh_preview()
            except Exception:
                pass

    def build_command(self) -> list[str]:
        available = {
            row["name"]
            for row in discover_reconstruction_catalog(
                current_dataset_dir(self.state), optional_root_relative_path(self.state.pose2sim_trc_var.get())
            )
            if row.get("cached")
        }
        cmd = [
            sys.executable,
            "animation/animate_dual_stick_comparison.py",
            "--dataset-dir",
            display_path(current_dataset_dir(self.state)),
            "--output",
            display_path(self.resolved_output_gif_path()),
            "--fps",
            self.fps.get(),
            "--stride",
            self.stride.get(),
            "--marker-size",
            self.marker_size.get(),
            "--framing",
            "tight" if self.crop_var.get() else "full",
        ]
        selected = [name for name in self.selected_reconstruction_names() if name in available]
        if selected:
            cmd.extend(["--show", *selected])
        if self.show_trunk_frames_var.get():
            cmd.append("--show-trunk-frames")
        if self.show_trampoline_var.get():
            cmd.append("--show-trampoline")
        cmd.extend(self.parse_extra_args(self.extra.get()))
        return cmd

    def load_preview(self) -> None:
        try:
            output_dir = current_dataset_dir(self.state)
            preview_load = load_dataset_preview_resources(
                output_dir=output_dir,
                preferred_names=["ekf_3d", "ekf_2d_flip_acc", "ekf_2d_acc", "pose2sim"],
                fallback_count=4,
                dataset_source_paths_fn=dataset_source_paths,
                discover_catalog_fn=discover_reconstruction_catalog,
                bundle_loader_fn=lambda dataset_dir, biomod_path, pose2sim_trc, align_root: get_cached_preview_bundle(
                    self.state,
                    dataset_dir,
                    biomod_path,
                    pose2sim_trc,
                    align_root,
                ),
                pose2sim_trc=optional_root_relative_path(self.state.pose2sim_trc_var.get()),
                biomod_path=resolve_preview_biomod(output_dir),
                align_root=False,
            )
            self.bundle = preview_load.bundle
            self._publish_reconstruction_rows(preview_load.preview_state.rows, preview_load.preview_state.defaults)
            self.frame_scale.configure(to=preview_load.preview_state.max_frame)
            self.frame_var.set(0)
            self.preview_canvas_widget.focus_set()
            self.refresh_preview()
        except Exception as exc:
            messagebox.showerror("3D preview", str(exc))

    def refresh_preview(self) -> None:
        if self.bundle is None:
            return
        recon_3d = self.bundle["recon_3d"]
        available = {name: points for name, points in recon_3d.items() if points is not None}
        if not available:
            return
        frame_idx = int(round(self.frame_var.get()))
        max_frame = min(points.shape[0] for points in available.values()) - 1
        frame_idx = clamp_frame_index(frame_idx, max_frame)
        self.frame_var.set(frame_idx)
        self.frame_label.configure(text=f"frame {frame_idx}")

        show_names = []
        for raw_name in self.selected_reconstruction_names():
            mapped = "ekf_2d_acc" if raw_name == "ekf_2d" else raw_name
            mapped = "ekf_3d" if raw_name == "biorbd_kalman" else mapped
            if mapped == "triangulation":
                mapped = "triangulation_adaptive"
            if mapped in available:
                show_names.append(mapped)
        if not show_names:
            show_names = [next(iter(available.keys()))]

        previous_axes = self.preview_figure.axes[0] if self.preview_figure.axes else None
        if previous_axes is not None and hasattr(previous_axes, "elev") and hasattr(previous_axes, "azim"):
            self._view_state = (
                float(getattr(previous_axes, "elev", 30.0)),
                float(getattr(previous_axes, "azim", -60.0)),
                float(getattr(previous_axes, "roll", 0.0)),
            )

        self.preview_figure.clear()
        ax = self.preview_figure.add_subplot(111, projection="3d")
        ax.mouse_init()
        if self._view_state is not None:
            ax.view_init(elev=self._view_state[0], azim=self._view_state[1], roll=self._view_state[2])
        points_dict = {name: available[name] for name in show_names}
        contact_zone_xy = None
        trampoline_z = None
        if self.show_trampoline_var.get():
            stacked = np.concatenate([points.reshape(-1, 3) for points in points_dict.values()], axis=0)
            valid_points = stacked[np.all(np.isfinite(stacked), axis=1)]
            if valid_points.size:
                trampoline_z = float(np.nanpercentile(valid_points[:, 2], 5))
                reference_name = show_names[0]
                reference_points = np.asarray(points_dict[reference_name], dtype=float)
                summary = self.bundle.get("recon_summary", {}).get(reference_name, {})
                airborne_mask = compute_airborne_mask_from_points(
                    reference_points,
                    threshold_m=float(summary.get("flight_height_threshold_m", trampoline_z + 0.12)),
                    min_consecutive_frames=int(summary.get("flight_min_consecutive_frames", 1)),
                )
                if frame_idx < airborne_mask.shape[0] and not airborne_mask[frame_idx]:
                    contact_zone_xy = trampoline_contact_zone_xy(
                        [np.asarray(points_dict[name][frame_idx], dtype=float) for name in show_names]
                    )
        for name in show_names:
            frame_points = available[name][frame_idx]
            draw_skeleton_3d(
                ax,
                frame_points,
                reconstruction_display_color(self.state, name),
                reconstruction_legend_label(self.state, name),
                marker_size=float(self.marker_size.get()),
            )
            summary = self.bundle.get("recon_summary", {}).get(name, {})
            q_names = self.bundle.get("q_names", [])
            if has_segmented_back_visualization(q_names=q_names, summary=summary):
                draw_upper_back_preview(ax, frame_points)
            if self.show_trunk_frames_var.get():
                origin, rotation = compute_root_frame_from_points(frame_points)
                if origin is not None and rotation is not None:
                    draw_coordinate_system(
                        ax,
                        origin,
                        rotation,
                        scale=0.18,
                        alpha=0.95,
                        prefix=f"{reconstruction_legend_label(self.state, name)}_",
                        show_labels=False,
                        line_width=2.2,
                    )
        set_equal_3d_limits(ax, points_dict, frame_idx if self.crop_var.get() else None)
        if self.show_trampoline_var.get():
            if trampoline_z is not None:
                draw_trampoline_bed_3d(ax, trampoline_z)
                draw_trampoline_contact_zone_3d(ax, contact_zone_xy, trampoline_z)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Vue 3D des reconstructions")
        ax.legend(loc="upper right", fontsize=8)
        self.preview_figure.tight_layout()
        self.preview_canvas.draw_idle()
        self._view_state = (
            float(getattr(ax, "elev", 30.0)),
            float(getattr(ax, "azim", -60.0)),
            float(getattr(ax, "roll", 0.0)),
        )


class MultiViewTab(CommandTab):
    def __init__(self, master, state: SharedAppState):
        super().__init__(
            master,
            "2D multiview",
            show_default_buttons=False,
            show_command_preview=False,
            show_output=False,
        )
        self.state = state
        self.pose_data = None
        self.calibrations = None
        self.preview_bundle = None
        self.projected_layers: dict[str, np.ndarray] = {}
        self.segmented_back_projected_layers: dict[str, dict[str, np.ndarray]] = {}
        self.reconstruction_payloads: dict[str, dict[str, np.ndarray]] = {}
        self.preview_frame_numbers = np.array([], dtype=int)
        self.preview_raw_points = None
        self.preview_pose_points = None
        self.crop_limits_cache: dict[str, np.ndarray] = {}
        self.crop_limits_key: tuple[object, ...] | None = None
        self.images_root: Path | None = None
        self._dragging_frame_scale = False
        self.uses_shared_reconstruction_panel = True
        self.shared_reconstruction_selectmode = "extended"

        body = ttk.Panedwindow(self.main, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=3)
        body.add(right, weight=7)

        form = ttk.LabelFrame(left, text="animation/animate_multiview_2d_comparison.py")
        form.pack(fill=tk.BOTH, expand=False, padx=(0, 8), pady=(0, 8))

        self.output_gif = LabeledEntry(form, "GIF name", self.default_gif_name(), browse=False, directory=False)
        self.output_gif.pack(fill=tk.X, padx=8, pady=4)

        row2 = ttk.Frame(form)
        row2.pack(fill=tk.X, padx=8, pady=4)
        self.gif_fps = LabeledEntry(row2, "GIF fps", "10", label_width=7, entry_width=6)
        self.gif_fps.pack(side=tk.LEFT, padx=(0, 6))
        self.stride = LabeledEntry(row2, "Stride", "5", label_width=6, entry_width=4)
        self.stride.pack(side=tk.LEFT, padx=(0, 6))

        row3 = ttk.Frame(form)
        row3.pack(fill=tk.X, padx=8, pady=4)
        self.crop_var = tk.BooleanVar(value=True)
        crop_check = ttk.Checkbutton(row3, text="Crop", variable=self.crop_var, command=self.refresh_preview)
        crop_check.pack(side=tk.LEFT, padx=(0, 8))
        self.marker_size = LabeledEntry(row3, "Marker size", "18", label_width=10, entry_width=5)
        self.marker_size.pack(side=tk.LEFT, padx=(0, 6))

        row4 = ttk.Frame(form)
        row4.pack(fill=tk.X, padx=8, pady=4)
        self.generate_button = ttk.Button(row4, text="GENERATE GIF", command=self.toggle_run_command)
        self.generate_button.pack(side=tk.RIGHT)
        self.attach_primary_action_button(self.generate_button, run_text="GENERATE GIF", stop_text="STOP")

        lower_left = ttk.Frame(left)
        lower_left.pack(fill=tk.BOTH, expand=True, padx=(0, 8))

        cameras_box = ttk.LabelFrame(lower_left, text="Cameras")
        cameras_box.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        cameras_header = ttk.Frame(cameras_box)
        cameras_header.pack(fill=tk.X, padx=8, pady=(6, 2))
        self.multiview_cameras_summary = tk.StringVar(value="Cameras (n=0/0)")
        ttk.Label(cameras_header, textvariable=self.multiview_cameras_summary, anchor="w").pack(side=tk.LEFT)
        cameras_body = ttk.Frame(cameras_box, height=120)
        cameras_body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        cameras_body.pack_propagate(False)
        self.multiview_cameras_list = tk.Listbox(
            cameras_body,
            selectmode=tk.EXTENDED,
            exportselection=False,
            height=5,
        )
        self.multiview_cameras_list.pack(fill=tk.BOTH, expand=True)
        bind_extended_listbox_shortcuts(self.multiview_cameras_list)
        self.multiview_cameras_list.bind(
            "<<ListboxSelect>>", lambda _event: self.on_multiview_camera_selection_changed()
        )

        images_box = ttk.LabelFrame(lower_left, text="Images")
        images_box.pack(fill=tk.X, expand=False)
        row_images_1 = ttk.Frame(images_box)
        row_images_1.pack(fill=tk.X, padx=8, pady=4)
        self.show_images_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row_images_1, text="Show images", variable=self.show_images_var, command=self.refresh_preview
        ).pack(side=tk.LEFT)
        ttk.Label(row_images_1, text="QA").pack(side=tk.LEFT, padx=(12, 4))
        self.qa_overlay_var = tk.StringVar(value="none")
        self.qa_overlay_box = ttk.Combobox(
            row_images_1,
            textvariable=self.qa_overlay_var,
            values=["none", "2D epipolar", "3D reproj", "3D excluded"],
            width=14,
            state="readonly",
        )
        self.qa_overlay_box.pack(side=tk.LEFT)
        self.images_root_entry = LabeledEntry(images_box, "Images root", "", browse=True, directory=True)
        self.images_root_entry.pack(fill=tk.X, padx=8, pady=(0, 6))

        preview_box = ttk.LabelFrame(right, text="Preview 2D multivues")
        preview_box.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        preview_controls = ttk.Frame(preview_box)
        preview_controls.pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(preview_controls, text="Load preview", command=self.load_preview).pack(side=tk.LEFT)
        ttk.Button(preview_controls, text="Refresh frame", command=self.refresh_preview).pack(side=tk.LEFT, padx=(8, 0))
        self.frame_var = tk.IntVar(value=0)
        self.frame_scale = ttk.Scale(
            preview_controls,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            variable=self.frame_var,
            command=lambda _value: self.refresh_preview(),
        )
        self.frame_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(12, 8))
        self.frame_label = ttk.Label(preview_controls, text="frame 0")
        self.frame_label.pack(side=tk.LEFT)

        self.preview_figure = Figure(figsize=(11, 8))
        self.preview_canvas = FigureCanvasTkAgg(self.preview_figure, master=preview_box)
        self.preview_canvas_widget = self.preview_canvas.get_tk_widget()
        self.preview_canvas_widget.pack(fill=tk.BOTH, expand=True)
        self._bind_frame_navigation(self.preview_canvas_widget)
        self._bind_frame_navigation(self.frame_scale)

        self.extra = LabeledEntry(form, "Extra args", "")
        self.extra.pack(fill=tk.X, padx=8, pady=4)
        self.output_gif.set_tooltip(
            "Nom du GIF 2D exporté. Le fichier sera enregistré automatiquement dans ./output/<trial>/figures/."
        )
        self.gif_fps.set_tooltip("Frequence d'images du GIF 2D exporte.")
        self.stride.set_tooltip("Une frame sur N est exportee dans le GIF 2D.")
        attach_tooltip(crop_check, "Recadre chaque vue autour de la pose. Décochez pour garder le champ complet.")
        self.marker_size.set_tooltip("Taille visuelle des marqueurs 2D dans le preview et le GIF.")
        attach_tooltip(
            self.generate_button, "Genere le GIF 2D multivues, puis devient STOP tant que l'export est en cours."
        )
        attach_tooltip(self.multiview_cameras_list, "Choisissez les caméras visibles dans le preview et le GIF.")
        self.images_root_entry.set_tooltip("Dossier d'images utilisé comme fond des vues 2D, par caméra si disponible.")
        attach_tooltip(
            self.qa_overlay_box,
            "Overlay QA rapide dans le preview 2D: erreur épipolaire 2D ou diagnostics 3D de la reconstruction sélectionnée.",
        )
        attach_tooltip(self.frame_scale, "Slider de navigation temporelle du preview 2D.")
        attach_tooltip(self.frame_label, "Index de frame actuellement affiche dans le preview 2D.")
        self.extra.set_tooltip("Options CLI supplémentaires pour animation/animate_multiview_2d_comparison.py.")
        self.state.keypoints_var.trace_add("write", lambda *_args: self.sync_dataset_defaults())
        self.state.output_root_var.trace_add("write", lambda *_args: self.sync_dataset_defaults())
        self.state.register_reconstruction_listener(self.refresh_available_reconstructions)
        self.qa_overlay_var.trace_add("write", lambda *_args: self.refresh_preview())
        self.refresh_available_reconstructions()

    def default_gif_name(self) -> str:
        """Return the default GIF filename for the current dataset."""

        return f"{current_dataset_name(self.state)}_2d_multiview.gif"

    def selected_multiview_camera_names(self) -> list[str]:
        if self.pose_data is None:
            return []
        all_camera_names = [str(name) for name in self.pose_data.camera_names]
        indices = [int(index) for index in self.multiview_cameras_list.curselection()]
        if not indices:
            return list(all_camera_names)
        return [all_camera_names[index] for index in indices if 0 <= index < len(all_camera_names)]

    def _set_multiview_camera_selection(self, camera_names: list[str] | None) -> None:
        requested = set(camera_names or [])
        self.multiview_cameras_list.selection_clear(0, tk.END)
        for index in range(self.multiview_cameras_list.size()):
            if self.multiview_cameras_list.get(index) in requested:
                self.multiview_cameras_list.selection_set(index)

    def update_multiview_camera_summary(self) -> None:
        total_count = self.multiview_cameras_list.size()
        selected_count = len(self.multiview_cameras_list.curselection()) if total_count else 0
        if total_count and selected_count == 0:
            selected_count = total_count
        self.multiview_cameras_summary.set(f"Cameras (n={selected_count}/{total_count})")

    def refresh_multiview_camera_choices(self) -> None:
        selected_before = self.selected_multiview_camera_names()
        self.multiview_cameras_list.delete(0, tk.END)
        camera_names: list[str] = []
        if self.pose_data is not None and getattr(self.pose_data, "camera_names", None) is not None:
            camera_names = [str(name) for name in self.pose_data.camera_names]
        elif self.calibrations:
            camera_names = [str(name) for name in self.calibrations.keys()]
        for camera_name in camera_names:
            self.multiview_cameras_list.insert(tk.END, camera_name)
        default_selection = selected_before if selected_before else camera_names
        self._set_multiview_camera_selection(default_selection if camera_names else None)
        self.update_multiview_camera_summary()

    def on_multiview_camera_selection_changed(self) -> None:
        self.update_multiview_camera_summary()
        self.refresh_preview()

    def resolved_output_gif_path(self) -> Path:
        """Resolve the GIF output path inside the current dataset figures directory."""

        raw_name = self.output_gif.get().strip()
        gif_name = Path(raw_name).name if raw_name else self.default_gif_name()
        if not gif_name.lower().endswith(".gif"):
            gif_name = f"{gif_name}.gif"
        return current_figures_dir(self.state) / gif_name

    def configure_shared_reconstruction_panel(self, panel: SharedReconstructionPanel) -> None:
        panel.configure_for_consumer(
            title="Reconstructions | 2D multiview",
            refresh_callback=self.refresh_available_reconstructions,
            selection_callback=self._on_reconstruction_selection_changed,
            selectmode=self.shared_reconstruction_selectmode,
        )
        self.refresh_available_reconstructions()

    def selected_reconstruction_names(self) -> list[str]:
        return list(self.state.shared_reconstruction_selection)

    def _publish_reconstruction_rows(self, rows: list[dict[str, object]], defaults: list[str]) -> None:
        panel = self.state.shared_reconstruction_panel
        if panel is not None and self.state.active_reconstruction_consumer is self:
            panel.set_rows(rows, defaults)

    def _on_reconstruction_selection_changed(self) -> None:
        if self.pose_data is None or self.preview_bundle is None:
            self.load_preview()
        else:
            self.refresh_preview()

    def _bind_frame_navigation(self, widget: tk.Widget) -> None:
        if widget is self.frame_scale:
            widget.bind("<Button-1>", self._on_frame_scale_click)
            widget.bind("<B1-Motion>", self._on_frame_scale_drag)
            widget.bind("<ButtonRelease-1>", self._on_frame_scale_release)
        else:
            widget.bind("<Enter>", lambda _event: widget.focus_set())
        widget.bind("<Left>", lambda _event: self.step_frame(-1))
        widget.bind("<Right>", lambda _event: self.step_frame(1))

    def _frame_from_scale_event(self, event) -> int:
        return frame_from_slider_click(
            x=event.x,
            width=self.frame_scale.winfo_width(),
            from_value=self.frame_scale.cget("from"),
            to_value=self.frame_scale.cget("to"),
        )

    def _on_frame_scale_click(self, event) -> str:
        self._dragging_frame_scale = True
        self.frame_scale.focus_set()
        frame = self._frame_from_scale_event(event)
        self.frame_var.set(frame)
        self.refresh_preview()
        return "break"

    def _on_frame_scale_drag(self, event) -> str:
        if not self._dragging_frame_scale:
            return "break"
        self.frame_var.set(self._frame_from_scale_event(event))
        self.refresh_preview()
        return "break"

    def _on_frame_scale_release(self, event) -> str:
        if not self._dragging_frame_scale:
            return "break"
        self._dragging_frame_scale = False
        self.frame_var.set(self._frame_from_scale_event(event))
        self.refresh_preview()
        return "break"

    def step_frame(self, delta: int) -> str:
        if self.pose_data is None:
            return "break"
        n_frames = len(self.pose_data.frames)
        if self.preview_bundle is not None and len(self.preview_bundle["frames"]):
            n_frames = min(n_frames, len(self.preview_bundle["frames"]))
        if n_frames <= 0:
            return "break"
        current = int(round(self.frame_var.get()))
        next_frame = step_frame_index(current=current, delta=delta, max_frame=n_frames - 1)
        if next_frame != current:
            self.frame_var.set(next_frame)
            self.refresh_preview()
        return "break"

    def sync_dataset_defaults(self) -> None:
        self.output_gif.var.set(self.default_gif_name())
        inferred_images_root = infer_execution_images_root(ROOT / self.state.keypoints_var.get())
        self.images_root = inferred_images_root
        self.images_root_entry.var.set("" if inferred_images_root is None else display_path(inferred_images_root))
        self.refresh_available_reconstructions()

    def refresh_available_reconstructions(self) -> None:
        _output_dir, preview_state = current_dataset_preview_state(
            self.state,
            bundle=self.preview_bundle,
            preferred_names=["raw", "pose2sim"],
            fallback_count=2,
            extra_rows=[
                {"name": "raw", "label": "Raw 2D", "family": "2d", "frames": "-", "reproj_mean": None, "path": "-"}
            ],
        )
        self._publish_reconstruction_rows(preview_state.rows, preview_state.defaults)
        if self.pose_data is not None and self.calibrations is not None:
            try:
                self.load_preview()
            except Exception:
                pass

    def build_command(self) -> list[str]:
        available = {
            row["name"]
            for row in discover_reconstruction_catalog(
                current_dataset_dir(self.state), optional_root_relative_path(self.state.pose2sim_trc_var.get())
            )
            if row.get("cached")
        }
        available.add("raw")
        cmd = [
            sys.executable,
            "animation/animate_multiview_2d_comparison.py",
            "--dataset-dir",
            display_path(current_dataset_dir(self.state)),
            "--output",
            display_path(self.resolved_output_gif_path()),
            "--data-fps",
            self.state.fps_var.get(),
            "--gif-fps",
            self.gif_fps.get(),
            "--stride",
            self.stride.get(),
            "--workers",
            self.state.workers_var.get(),
            "--marker-size",
            self.marker_size.get(),
            "--crop-mode",
            ("pose" if self.crop_var.get() else "full"),
            "--crop-margin",
            "0.1",
        ]
        selected = [name for name in self.selected_reconstruction_names() if name in available]
        if selected:
            cmd.extend(["--show", *selected])
        selected_cameras = self.selected_multiview_camera_names()
        if selected_cameras:
            cmd.extend(["--camera-names", ",".join(selected_cameras)])
        images_root = self.images_root_entry.get().strip()
        if self.show_images_var.get():
            cmd.append("--show-images")
            if images_root:
                cmd.extend(["--images-root", images_root])
        cmd.extend(self.parse_extra_args(self.extra.get()))
        return cmd

    def load_preview(self) -> None:
        try:
            output_dir = current_dataset_dir(self.state)
            preview_load = load_dataset_preview_resources(
                output_dir=output_dir,
                preferred_names=["raw", "pose2sim"],
                fallback_count=2,
                extra_rows=[
                    {"name": "raw", "label": "Raw 2D", "family": "2d", "frames": "-", "reproj_mean": None, "path": "-"}
                ],
                dataset_source_paths_fn=dataset_source_paths,
                discover_catalog_fn=discover_reconstruction_catalog,
                bundle_loader_fn=lambda dataset_dir, biomod_path, pose2sim_trc, align_root: get_cached_preview_bundle(
                    self.state,
                    dataset_dir,
                    biomod_path,
                    pose2sim_trc,
                    align_root,
                ),
                calib=ROOT / self.state.calib_var.get(),
                keypoints=ROOT / self.state.keypoints_var.get(),
                pose2sim_trc=optional_root_relative_path(self.state.pose2sim_trc_var.get()),
                biomod_path=resolve_preview_biomod(output_dir),
                align_root=False,
            )
            sources = preview_load.sources
            self.calibrations, self.pose_data = get_cached_pose_data(
                self.state,
                keypoints_path=Path(sources["keypoints"]),
                calib_path=Path(sources["calib"]),
                data_mode=self.state.pose_data_mode_var.get(),
                smoothing_window=int(self.state.pose_filter_window_var.get()),
                outlier_threshold_ratio=float(self.state.pose_outlier_ratio_var.get()),
                lower_percentile=float(self.state.pose_p_low_var.get()),
                upper_percentile=float(self.state.pose_p_high_var.get()),
            )
            self.preview_bundle = preview_load.bundle
            self.refresh_multiview_camera_choices()
            inferred_images_root = infer_execution_images_root(ROOT / self.state.keypoints_var.get())
            if not self.images_root_entry.get().strip() or inferred_images_root is not None:
                self.images_root = inferred_images_root
                self.images_root_entry.var.set(
                    "" if inferred_images_root is None else display_path(inferred_images_root)
                )
            self._publish_reconstruction_rows(preview_load.preview_state.rows, preview_load.preview_state.defaults)
            bundle_frames = (
                np.asarray(self.preview_bundle["frames"], dtype=int)
                if self.preview_bundle is not None and "frames" in self.preview_bundle
                else np.asarray(self.pose_data.frames, dtype=int)
            )
            pose_frame_indices = preview_pose_frame_indices(np.asarray(self.pose_data.frames, dtype=int), bundle_frames)
            self.preview_frame_numbers = np.asarray(bundle_frames, dtype=int)
            self.preview_pose_points = np.asarray(self.pose_data.keypoints[:, pose_frame_indices], dtype=float)
            self.preview_raw_points = np.asarray(
                (
                    self.pose_data.raw_keypoints[:, pose_frame_indices]
                    if self.pose_data.raw_keypoints is not None
                    else self.pose_data.keypoints[:, pose_frame_indices]
                ),
                dtype=float,
            )
            camera_names = list(self.pose_data.camera_names)
            self.projected_layers = {}
            self.segmented_back_projected_layers = {}
            self.reconstruction_payloads = {}
            for name, points_3d in self.preview_bundle["recon_3d"].items():
                self.projected_layers[name] = project_points_all_cameras(points_3d, self.calibrations, camera_names)
                recon_dir = reconstruction_dir_by_name(output_dir, name)
                if recon_dir is not None:
                    self.reconstruction_payloads[name] = load_bundle_payload(recon_dir)
                q_series = self.preview_bundle.get("recon_q", {}).get(name)
                biomod_path = resolve_reconstruction_biomod(output_dir, name)
                if q_series is not None and biomod_path is not None and biomod_path.exists():
                    overlay_3d = segmented_back_overlay_from_q(biomod_path, np.asarray(q_series, dtype=float))
                    if overlay_3d:
                        self.segmented_back_projected_layers[name] = {
                            overlay_name: project_points_all_cameras(points, self.calibrations, camera_names)
                            for overlay_name, points in overlay_3d.items()
                        }
            self.crop_limits_cache = {}
            self.crop_limits_key = None
            n_frames = min(
                (
                    self.preview_pose_points.shape[1]
                    if self.preview_pose_points is not None
                    else len(self.pose_data.frames)
                ),
                (
                    preview_load.preview_state.max_frame + 1
                    if self.preview_bundle is not None
                    else len(self.pose_data.frames)
                ),
            )
            self.frame_scale.configure(to=max(n_frames - 1, 0))
            self.frame_var.set(0)
            self.preview_canvas_widget.focus_set()
            self.refresh_preview()
        except Exception as exc:
            messagebox.showerror("2D multiview", str(exc))

    def refresh_preview(self) -> None:
        if (
            self.pose_data is None
            or self.calibrations is None
            or self.preview_pose_points is None
            or self.preview_raw_points is None
        ):
            return
        n_frames = self.preview_pose_points.shape[1]
        if n_frames == 0:
            return
        frame_idx = clamp_frame_index(int(round(self.frame_var.get())), n_frames - 1)
        self.frame_var.set(frame_idx)
        self.frame_label.configure(text=f"frame {frame_idx}")

        self.preview_figure.clear()
        all_camera_names = [str(name) for name in self.pose_data.camera_names]
        camera_names = self.selected_multiview_camera_names()
        camera_indices = [all_camera_names.index(name) for name in camera_names if name in all_camera_names]
        if not camera_indices:
            camera_names = list(all_camera_names)
            camera_indices = list(range(len(camera_names)))
        nrows, ncols = camera_layout(len(camera_names))
        axes = self.preview_figure.subplots(nrows, ncols)
        axes = np.atleast_1d(axes).ravel()
        selected = self.selected_reconstruction_names()
        raw_points = np.asarray(self.preview_raw_points[camera_indices], dtype=float)
        pose_points = np.asarray(self.preview_pose_points[camera_indices], dtype=float)
        projected_layers = {
            name: np.asarray(points[camera_indices], dtype=float) for name, points in self.projected_layers.items()
        }
        crop_points = compose_multiview_crop_points(pose_points, projected_layers, selected)
        crop_mode = "pose" if self.crop_var.get() else "full"
        crop_margin = 0.1
        crop_limits = (
            self._ensure_crop_limits(crop_points, camera_names, crop_margin, tuple(selected))
            if crop_mode == "pose"
            else {}
        )
        frame_number = int(self.preview_frame_numbers[frame_idx]) if self.preview_frame_numbers.size else int(frame_idx)
        images_root = (
            Path(self.images_root_entry.get().strip()) if self.images_root_entry.get().strip() else self.images_root
        )

        for ax_idx, ax in enumerate(axes):
            if ax_idx >= len(camera_names):
                ax.axis("off")
                continue
            cam_name = camera_names[ax_idx]
            width, height = self.calibrations[cam_name].image_size
            background_image = (
                load_camera_background_image(
                    images_root,
                    cam_name,
                    frame_number,
                    image_reader=plt.imread,
                )
                if self.show_images_var.get()
                else None
            )
            layers: list[SkeletonLayer2D] = []
            if "raw" in selected:
                layers.append(
                    SkeletonLayer2D(
                        points=raw_points[ax_idx, frame_idx],
                        color=("white" if background_image is not None else "#444444"),
                        label="Raw",
                        marker_size=float(self.marker_size.get()),
                    )
                )
            for raw_name in selected:
                mapped = "ekf_2d_acc" if raw_name == "ekf_2d" else raw_name
                mapped = "triangulation_adaptive" if raw_name == "triangulation" else mapped
                if mapped == "biorbd_kalman":
                    mapped = "ekf_3d"
                if mapped == "raw" or mapped not in projected_layers:
                    continue
                points_2d = projected_layers[mapped][ax_idx, frame_idx]
                layers.append(
                    SkeletonLayer2D(
                        points=points_2d,
                        color=reconstruction_display_color(self.state, mapped),
                        label=reconstruction_legend_label(self.state, mapped),
                        marker_size=float(self.marker_size.get()),
                    )
                )
            render_camera_frame_2d(
                ax,
                width=width,
                height=height,
                title=cam_name.replace("Camera", ""),
                layers=layers,
                draw_skeleton_fn=draw_skeleton_2d,
                background_image=background_image,
                draw_background_fn=draw_2d_background_image,
                crop_mode=crop_mode,
                crop_limits=crop_limits,
                cam_name=cam_name,
                frame_idx=frame_idx,
                apply_axis_limits_fn=apply_2d_axis_limits,
                hide_axes=True,
                hide_axes_fn=hide_2d_axes,
            )
            overlay_label, overlay_values, overlay_mask, overlay_cmap = self._qa_overlay_data(cam_name, frame_idx)
            overlay_scatter = draw_point_value_overlay(
                ax,
                PointValueOverlay2D(
                    label=overlay_label,
                    points=raw_points[ax_idx, frame_idx],
                    values=overlay_values,
                    mask=overlay_mask,
                    cmap=overlay_cmap,
                    size=16.0,
                    excluded_size=max(18.0, float(self.marker_size.get()) * 0.75),
                ),
            )
            if overlay_scatter is not None:
                self.preview_figure.colorbar(overlay_scatter, ax=ax, fraction=0.042, pad=0.02, label=overlay_label)
            for raw_name in selected:
                mapped = "ekf_2d_acc" if raw_name == "ekf_2d" else raw_name
                mapped = "triangulation_adaptive" if raw_name == "triangulation" else mapped
                if mapped == "biorbd_kalman":
                    mapped = "ekf_3d"
                if mapped == "raw" or mapped not in projected_layers:
                    continue
                points_2d = projected_layers[mapped][ax_idx, frame_idx]
                segmented_back_layers = self.segmented_back_projected_layers.get(mapped, {})
                draw_upper_back_overlay_2d(
                    ax,
                    hip_triangle_2d=(
                        segmented_back_layers.get("hip_triangle", np.empty((0, 0, 0)))[ax_idx, frame_idx]
                        if "hip_triangle" in segmented_back_layers
                        else None
                    ),
                    shoulder_triangle_2d=(
                        segmented_back_layers.get("shoulder_triangle", np.empty((0, 0, 0)))[ax_idx, frame_idx]
                        if "shoulder_triangle" in segmented_back_layers
                        else None
                    ),
                    mid_back_2d=(
                        segmented_back_layers.get("mid_back", np.empty((0, 0, 0)))[ax_idx, frame_idx, 0]
                        if "mid_back" in segmented_back_layers
                        else None
                    ),
                    color=reconstruction_display_color(self.state, mapped),
                )
                self._draw_excluded_reprojection_points(
                    ax=ax,
                    reconstruction_name=mapped,
                    camera_name=cam_name,
                    frame_number=frame_number,
                    points_2d=points_2d,
                )
            ax.tick_params(labelsize=8, length=0)

        handles, labels = axes[0].get_legend_handles_labels() if axes.size else ([], [])
        if handles:
            uniq = {}
            for handle, label in zip(handles, labels):
                uniq[label] = handle
            self.preview_figure.legend(
                list(uniq.values()),
                list(uniq.keys()),
                loc="upper center",
                bbox_to_anchor=(0.5, 0.985),
                ncol=min(5, len(uniq)),
                fontsize=8,
            )
        self.preview_figure.subplots_adjust(left=0.035, right=0.995, bottom=0.035, top=0.93, wspace=0.08, hspace=0.18)
        self.preview_canvas.draw_idle()

    def _ensure_crop_limits(
        self, crop_points: np.ndarray, camera_names: list[str], crop_margin: float, selected_layers: tuple[str, ...]
    ) -> dict[str, np.ndarray]:
        cache_key = (
            id(self.pose_data),
            id(self.calibrations),
            tuple(camera_names),
            tuple(selected_layers),
            tuple(crop_points.shape),
            float(crop_margin),
            float(np.nanmin(crop_points)) if np.any(np.isfinite(crop_points)) else np.nan,
            float(np.nanmax(crop_points)) if np.any(np.isfinite(crop_points)) else np.nan,
        )
        if self.crop_limits_key != cache_key:
            gui_debug(
                "2D multiview compute crop limits "
                f"frames={crop_points.shape[1]} cams={len(camera_names)} margin={crop_margin}"
            )
            self.crop_limits_cache = compute_pose_crop_limits_2d(
                crop_points, self.calibrations, camera_names, crop_margin
            )
            self.crop_limits_key = cache_key
        return self.crop_limits_cache

    def _qa_overlay_data(
        self, camera_name: str, frame_local_idx: int
    ) -> tuple[str, np.ndarray | None, np.ndarray | None, str | None]:
        if self.pose_data is None or self.calibrations is None:
            return "none", None, None, None
        mode = str(self.qa_overlay_var.get()).strip().lower()
        if mode == "2d epipolar":
            cam_idx = list(self.pose_data.camera_names).index(camera_name)
            values = frame_camera_epipolar_errors(
                self.pose_data, self.calibrations, frame_idx=frame_local_idx, camera_idx=cam_idx
            )
            return "2D epipolar", values, None, "turbo"
        reference_name = (self._selected_reconstruction() or "").strip()
        if not reference_name:
            return "none", None, None, None
        payload = self.reconstruction_payloads.get(reference_name, {})
        if mode == "3d reproj":
            errors = payload.get("reprojection_error_per_view")
            if errors is None:
                return "3D reproj", None, None, None
            errors = np.asarray(errors, dtype=float)
            cam_idx = list(self.pose_data.camera_names).index(camera_name)
            if errors.ndim == 3 and frame_local_idx < errors.shape[0] and cam_idx < errors.shape[2]:
                return "3D reproj", np.asarray(errors[frame_local_idx, :, cam_idx], dtype=float), None, "turbo"
        if mode == "3d excluded":
            excluded = payload.get("excluded_views")
            if excluded is None:
                return "3D excluded", None, None, None
            excluded = np.asarray(excluded, dtype=bool)
            cam_idx = list(self.pose_data.camera_names).index(camera_name)
            if excluded.ndim == 3 and frame_local_idx < excluded.shape[0] and cam_idx < excluded.shape[2]:
                return "3D excluded", None, np.asarray(excluded[frame_local_idx, :, cam_idx], dtype=bool), None
        return "none", None, None, None

    def _draw_excluded_reprojection_points(
        self,
        *,
        ax,
        reconstruction_name: str,
        camera_name: str,
        frame_number: int,
        points_2d: np.ndarray,
    ) -> None:
        payload = self.reconstruction_payloads.get(str(reconstruction_name), {})
        excluded_views = payload.get("excluded_views")
        payload_frames = payload.get("frames")
        payload_camera_names = payload.get("camera_names")
        if excluded_views is None or payload_frames is None or payload_camera_names is None:
            return
        excluded_views = np.asarray(excluded_views, dtype=bool)
        payload_frames = np.asarray(payload_frames, dtype=int)
        payload_camera_names = [str(name) for name in np.asarray(payload_camera_names, dtype=object)]
        if excluded_views.ndim != 3 or frame_number not in set(int(frame) for frame in payload_frames):
            return
        try:
            frame_idx = int(np.where(payload_frames == int(frame_number))[0][0])
            cam_idx = payload_camera_names.index(str(camera_name))
        except Exception:
            return
        if frame_idx >= excluded_views.shape[0] or cam_idx >= excluded_views.shape[2]:
            return
        excluded_mask = np.asarray(excluded_views[frame_idx, :, cam_idx], dtype=bool)
        if excluded_mask.shape[0] != points_2d.shape[0] or not np.any(excluded_mask):
            return
        excluded_points = np.asarray(points_2d[excluded_mask], dtype=float)
        valid = np.all(np.isfinite(excluded_points), axis=1)
        if not np.any(valid):
            return
        ax.scatter(
            excluded_points[valid, 0],
            excluded_points[valid, 1],
            s=max(18.0, float(self.marker_size.get()) * 0.8),
            facecolors="none",
            edgecolors="#111111",
            linewidths=1.6,
            marker="x",
            alpha=0.9,
        )


class AnnotationTab(ttk.Frame):
    def __init__(self, master, state: SharedAppState):
        super().__init__(master)
        self.state = state
        self.uses_shared_reconstruction_panel = True
        self.shared_reconstruction_selectmode = "browse"
        self.calibrations = None
        self.pose_data = None
        self.annotation_payload = empty_annotation_payload()
        self.annotation_path: Path | None = None
        self.images_root: Path | None = None
        self.crop_limits_cache: dict[str, np.ndarray] = {}
        self.crop_limits_key: tuple[object, ...] | None = None
        self._navigable_frame_cache: dict[tuple[object, ...], list[int]] = {}
        self._axis_to_camera: dict[object, str] = {}
        self._current_frame_idx = 0
        self._pan_state: dict[str, object] | None = None
        self._drag_annotation_state: dict[str, object] | None = None
        self._dragging_frame_scale = False
        self._cursor_artists: dict[object, tuple[object, ...]] = {}
        self._annotation_hover_entries: dict[object, list[dict[str, object]]] = {}
        self._annotation_view_limits: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
        self._pending_reprojection_points: dict[tuple[str, int, str], np.ndarray] = {}
        self.kinematic_model_choices: dict[str, Path] = {}
        self.kinematic_frame_states: dict[tuple[str, int], np.ndarray] = {}
        self.kinematic_q_current: np.ndarray | None = None
        self.kinematic_state_current: np.ndarray | None = None
        self.kinematic_q_names: list[str] = []
        self.kinematic_projected_points: np.ndarray | None = None
        self.kinematic_segmented_back_projected: dict[str, np.ndarray] = {}
        self.annotation_jump_analysis: DDSessionAnalysis | None = None

        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=1)
        body.add(right, weight=9)

        controls = ttk.LabelFrame(left, text="Annotation controls")
        controls.pack(fill=tk.BOTH, expand=True, padx=(0, 8), pady=(0, 8))

        row0 = ttk.Frame(controls)
        row0.pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(row0, text="Load data", command=self.load_resources).pack(side=tk.LEFT)
        ttk.Button(row0, text="Save annotations", command=self.save_annotations).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(row0, text="Refresh frame", command=self.refresh_preview).pack(side=tk.LEFT, padx=(8, 0))
        self.reproject_button_var = tk.StringVar(value="Reproject")
        self.reproject_button = ttk.Button(
            row0, textvariable=self.reproject_button_var, command=self.on_reproject_button
        )
        self.reproject_button.pack(side=tk.LEFT, padx=(8, 0))

        row1 = ttk.Frame(controls)
        row1.pack(fill=tk.X, padx=8, pady=4)
        self.show_images_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row1, text="Show images", variable=self.show_images_var, command=self.refresh_preview).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        self.crop_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row1, text="Crop +20%", variable=self.crop_var, command=self.refresh_preview).pack(
            side=tk.LEFT, padx=(0, 8)
        )

        row2 = ttk.Frame(controls)
        row2.pack(fill=tk.X, padx=8, pady=4)
        self.advance_marker_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            row2,
            text="Advance marker on click",
            variable=self.advance_marker_var,
        ).pack(side=tk.LEFT, padx=(0, 8))
        self.show_epipolar_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            row2, text="Epipolar guide", variable=self.show_epipolar_var, command=self.refresh_preview
        ).pack(side=tk.LEFT, padx=(0, 8))
        self.show_triangulated_hint_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            row2,
            text="Triangulated reproj",
            variable=self.show_triangulated_hint_var,
            command=self.refresh_preview,
        ).pack(side=tk.LEFT)
        self.show_reference_reprojection_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row2,
            text="Selected recon reproj",
            variable=self.show_reference_reprojection_var,
            command=self.refresh_preview,
        ).pack(side=tk.LEFT, padx=(8, 0))

        row3 = ttk.Frame(controls)
        row3.pack(fill=tk.X, padx=8, pady=4)
        self.snap_reprojection_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row3,
            text="Snap reproj",
            variable=self.snap_reprojection_var,
        ).pack(side=tk.LEFT, padx=(0, 8))
        self.snap_epipolar_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row3,
            text="Snap epipolar",
            variable=self.snap_epipolar_var,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(row3, text=f"radius {int(ANNOTATION_SNAP_RADIUS_PX)} px", foreground="#4f5b66").pack(side=tk.LEFT)

        row4 = ttk.Frame(controls)
        row4.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(row4, text="Frame set", width=10).pack(side=tk.LEFT, padx=(0, 6))
        self.frame_filter_var = tk.StringVar(value=ANNOTATION_FRAME_FILTER_OPTIONS["all"])
        self.frame_filter_box = ttk.Combobox(
            row4,
            textvariable=self.frame_filter_var,
            values=list(ANNOTATION_FRAME_FILTER_OPTIONS.values()),
            width=18,
            state="readonly",
        )
        self.frame_filter_box.pack(side=tk.LEFT, padx=(0, 10))

        row5 = ttk.Frame(controls)
        row5.pack(fill=tk.X, padx=8, pady=4)
        self.show_motion_prior_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            row5,
            text="Velocity prior zone",
            variable=self.show_motion_prior_var,
            command=self.refresh_preview,
        ).pack(side=tk.LEFT, padx=(0, 8))
        self.motion_prior_diameter = LabeledEntry(row5, "Diameter", "15", label_width=7, entry_width=4)
        self.motion_prior_diameter.pack(side=tk.LEFT)

        row6 = ttk.Frame(controls)
        row6.pack(fill=tk.X, padx=8, pady=4)
        self.image_brightness_var = tk.DoubleVar(value=1.0)
        ttk.Label(row6, text="Brightness", width=10).pack(side=tk.LEFT)
        ttk.Scale(
            row6,
            from_=0.2,
            to=2.0,
            orient=tk.HORIZONTAL,
            variable=self.image_brightness_var,
            command=lambda _value: self.refresh_preview(),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.image_contrast_var = tk.DoubleVar(value=1.0)
        ttk.Label(row6, text="Contrast", width=8).pack(side=tk.LEFT)
        ttk.Scale(
            row6,
            from_=0.2,
            to=2.0,
            orient=tk.HORIZONTAL,
            variable=self.image_contrast_var,
            command=lambda _value: self.refresh_preview(),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.images_root_entry = LabeledEntry(controls, "Images root", "", browse=True, directory=True)
        self.images_root_entry.pack(fill=tk.X, padx=8, pady=4)
        self.images_root_entry.var.trace_add("write", lambda *_args: self.refresh_preview())
        self.annotations_path_entry = LabeledEntry(controls, "Annotations", "", browse=True, directory=False)
        self.annotations_path_entry.pack(fill=tk.X, padx=8, pady=4)
        self.annotations_path_entry.var = self.state.annotation_path_var
        self.annotations_path_entry.entry_widget.configure(textvariable=self.state.annotation_path_var)
        self.annotations_path_entry.on_browse_selected = lambda _value: self.load_resources()
        self.state.annotation_path_var.trace_add("write", lambda *_args: self.load_resources())

        kinematic_row = ttk.Frame(controls)
        kinematic_row.pack(fill=tk.X, padx=8, pady=4)
        self.kinematic_assist_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            kinematic_row, text="Kinematic assist", variable=self.kinematic_assist_var, command=self.refresh_preview
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(kinematic_row, text="Model", width=6).pack(side=tk.LEFT)
        self.kinematic_model_var = tk.StringVar(value="")
        self.kinematic_model_box = ttk.Combobox(
            kinematic_row, textvariable=self.kinematic_model_var, values=[], width=28, state="readonly"
        )
        self.kinematic_model_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.kinematic_model_var.trace_add(
            "write",
            lambda *_args: (
                self.refresh_annotation_keypoint_choices(),
                self._clear_kinematic_assist_preview(),
                self.refresh_preview(),
            ),
        )
        ttk.Button(kinematic_row, text="Estimate q", command=self.estimate_kinematic_assist_state).pack(side=tk.LEFT)
        self.kinematic_status_var = tk.StringVar(value="")
        ttk.Label(controls, textvariable=self.kinematic_status_var, anchor="w", foreground="#4f5b66").pack(
            fill=tk.X, padx=8, pady=(0, 4)
        )
        self.jump_context_var = tk.StringVar(value="")
        ttk.Label(controls, textvariable=self.jump_context_var, anchor="w", foreground="#4f5b66").pack(
            fill=tk.X, padx=8, pady=(0, 4)
        )

        lists_row = ttk.Frame(controls)
        lists_row.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

        cameras_box = ttk.LabelFrame(lists_row, text="Cameras")
        cameras_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.annotation_cameras_summary = tk.StringVar(value="Cameras (n=0/0)")
        ttk.Label(cameras_box, textvariable=self.annotation_cameras_summary, anchor="w").pack(
            fill=tk.X, padx=6, pady=(6, 2)
        )
        cameras_body = ttk.Frame(cameras_box, height=145, width=145)
        cameras_body.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        cameras_body.pack_propagate(False)
        self.annotation_cameras_list = tk.Listbox(
            cameras_body, selectmode=tk.EXTENDED, exportselection=False, height=8, width=12
        )
        self.annotation_cameras_list.pack(fill=tk.BOTH, expand=True)
        bind_extended_listbox_shortcuts(self.annotation_cameras_list)
        self.annotation_cameras_list.bind("<<ListboxSelect>>", lambda _event: self.on_camera_selection_changed())

        keypoints_box = ttk.LabelFrame(lists_row, text="Markers")
        keypoints_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        self.current_marker_var = tk.StringVar(value="")
        ttk.Label(keypoints_box, textvariable=self.current_marker_var, anchor="w").pack(fill=tk.X, padx=6, pady=(6, 2))
        keypoints_body = ttk.Frame(keypoints_box, height=145, width=155)
        keypoints_body.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        keypoints_body.pack_propagate(False)
        self.annotation_keypoints_list = tk.Listbox(keypoints_body, exportselection=False, height=8, width=14)
        self.annotation_keypoints_list.pack(fill=tk.BOTH, expand=True)
        self.annotation_keypoints_list.bind("<<ListboxSelect>>", lambda _event: self.on_keypoint_selection_changed())
        self.annotation_keypoints_list.bind("<Up>", lambda event: self._step_annotation_keypoint(-1, event))
        self.annotation_keypoints_list.bind("<Down>", lambda event: self._step_annotation_keypoint(1, event))
        self.refresh_annotation_keypoint_choices()

        preview_box = ttk.LabelFrame(right, text="Annotation preview")
        preview_box.pack(fill=tk.BOTH, expand=True)
        preview_controls = ttk.Frame(preview_box)
        preview_controls.pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(preview_controls, text="Prev frame", command=lambda: self.step_frame(-1)).pack(side=tk.LEFT)
        ttk.Button(preview_controls, text="Next frame", command=lambda: self.step_frame(1)).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        self.frame_var = tk.IntVar(value=0)
        self.frame_scale = ttk.Scale(
            preview_controls,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            variable=self.frame_var,
            command=self.on_frame_scale_changed,
        )
        self.frame_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(12, 8))
        self.frame_label = ttk.Label(preview_controls, text="frame 0")
        self.frame_label.pack(side=tk.LEFT)

        self.preview_figure = Figure(figsize=(11, 8))
        self.preview_canvas = FigureCanvasTkAgg(self.preview_figure, master=preview_box)
        self.preview_canvas_widget = self.preview_canvas.get_tk_widget()
        self.preview_canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.preview_canvas_widget.configure(cursor="crosshair")
        self.preview_canvas.mpl_connect("button_press_event", self.on_preview_click)
        self.preview_canvas.mpl_connect("scroll_event", self.on_preview_scroll)
        self.preview_canvas.mpl_connect("motion_notify_event", self.on_preview_motion)
        self.preview_canvas.mpl_connect("button_release_event", self.on_preview_release)
        self._bind_frame_navigation(self.preview_canvas_widget)
        self._bind_annotation_keypoint_navigation(self)
        self._bind_cancel_pending_reprojection(controls)
        for clickable_parent in (self, body, left, right, preview_box, preview_controls):
            clickable_parent.bind("<Button-1>", self._cancel_pending_reprojection_from_tk_event, add="+")
        self._bind_frame_navigation(self.frame_scale)

        attach_tooltip(
            self.annotation_keypoints_list,
            "Liste des marqueurs annotés. Si l’avance automatique est activée, un clic place le point puis passe au suivant.",
        )
        attach_tooltip(
            self.annotation_cameras_list,
            "Caméras visibles dans l’annotation. Les vues sélectionnées sont affichées simultanément.",
        )
        self.annotations_path_entry.set_tooltip(
            "Fichier JSON sparse d’annotations 2D. Par défaut: inputs/annotations/<dataset>_annotations.json"
        )
        self.images_root_entry.set_tooltip("Dossier d’images utilisé comme fond des vues 2D pendant l’annotation.")
        self.motion_prior_diameter.set_tooltip(
            "Diamètre du cercle de confiance basé sur la vitesse estimée entre t-2 et t-1."
        )
        attach_tooltip(
            self.frame_filter_box,
            "Choisit un sous-ensemble de frames: toutes, celles marquées comme flip L/R, ou les 5% pires erreurs de reprojection pour la reconstruction sélectionnée.",
        )
        attach_tooltip(
            row3,
            "Snaps the placed point to the nearest reprojection or epipolar guide when the corresponding option is enabled.",
        )
        attach_tooltip(
            self.preview_canvas.get_tk_widget(),
            "Souris: clic gauche place un point, clic droit efface, molette zoome, clic molette + glisser déplace.",
        )

        self.state.keypoints_var.trace_add("write", lambda *_args: self.sync_dataset_defaults())
        self.state.output_root_var.trace_add("write", lambda *_args: self.sync_dataset_defaults())
        self.frame_filter_var.trace_add("write", lambda *_args: self.on_frame_filter_changed())
        self.state.register_reconstruction_listener(lambda: self.after_idle(self.refresh_available_reconstructions))
        self.sync_dataset_defaults()
        self.after_idle(self.refresh_available_reconstructions)

    def configure_shared_reconstruction_panel(self, panel: SharedReconstructionPanel) -> None:
        panel.configure_for_consumer(
            title="Reconstructions | Annotation",
            refresh_callback=self.refresh_available_reconstructions,
            selection_callback=self._on_reconstruction_selection_changed,
            selectmode=self.shared_reconstruction_selectmode,
        )
        self.refresh_available_reconstructions()

    def selected_reconstruction_names(self) -> list[str]:
        return list(self.state.shared_reconstruction_selection)

    def _publish_reconstruction_rows(self, rows: list[dict[str, object]], defaults: list[str]) -> None:
        panel = self.state.shared_reconstruction_panel
        if panel is not None and self.state.active_reconstruction_consumer is self:
            panel.set_rows(rows, defaults)

    def _on_reconstruction_selection_changed(self) -> None:
        self.on_frame_filter_changed()

    def refresh_available_reconstructions(self) -> None:
        try:
            _output_dir, _bundle, preview_state = load_shared_reconstruction_preview_state(
                self.state,
                preferred_names=["triangulation", "ekf_2d", "ekf_3d"],
                fallback_count=3,
                include_3d=True,
                include_q=False,
                include_q_root=False,
            )
            self._publish_reconstruction_rows(preview_state.rows, preview_state.defaults[:1])
        except Exception as exc:
            gui_debug(f"Annotation refresh_available_reconstructions error: {exc}")
            self._publish_reconstruction_rows([], [])

    def sync_dataset_defaults(self) -> None:
        keypoints_path = ROOT / self.state.keypoints_var.get()
        self.annotation_path = default_annotation_path(keypoints_path)
        self.state.annotation_path_var.set(display_path(self.annotation_path))
        inferred_images_root = infer_execution_images_root(keypoints_path)
        self.images_root = inferred_images_root
        self.images_root_entry.var.set("" if inferred_images_root is None else display_path(inferred_images_root))
        self.refresh_kinematic_model_choices()
        self.load_resources()

    def refresh_kinematic_model_choices(self) -> None:
        dataset_dir = current_dataset_dir(self.state)
        biomod_paths: list[Path] = []
        for model_dir in scan_model_dirs(dataset_dir):
            biomod_paths.extend(sorted(model_dir.glob("*.bioMod")))
        choices: dict[str, Path] = {}
        labels: list[str] = []
        for biomod_path in biomod_paths:
            label = display_path(biomod_path)
            if label in choices:
                continue
            choices[label] = biomod_path
            labels.append(label)
        self.kinematic_model_choices = choices
        self.kinematic_model_box.configure(values=labels)
        current = str(self.kinematic_model_var.get()).strip()
        if current not in choices:
            self.kinematic_model_var.set(labels[0] if labels else "")
        self.refresh_annotation_keypoint_choices()
        self.kinematic_status_var.set("" if labels else "No existing bioMod available for kinematic assist.")

    def _selected_kinematic_biomod_path(self) -> Path | None:
        if not hasattr(self, "kinematic_model_var"):
            return None
        label = str(self.kinematic_model_var.get()).strip()
        return self.kinematic_model_choices.get(label)

    def annotation_keypoint_names(self) -> tuple[str, ...]:
        biomod_path = self._selected_kinematic_biomod_path()
        return annotation_keypoint_names_for_biomod(biomod_path)

    def refresh_annotation_keypoint_choices(self) -> None:
        if not hasattr(self, "annotation_keypoints_list"):
            return
        previous_name = None
        selection = self.annotation_keypoints_list.curselection()
        if selection:
            previous_name = str(self.annotation_keypoints_list.get(selection[0]))
        names = list(self.annotation_keypoint_names())
        self.annotation_keypoints_list.delete(0, tk.END)
        for keypoint_name in names:
            self.annotation_keypoints_list.insert(tk.END, keypoint_name)
        if not names:
            self.current_marker_var.set("Current marker:")
            return
        target_name = previous_name if previous_name in names else names[0]
        target_index = names.index(target_name)
        self.annotation_keypoints_list.selection_set(target_index)
        self.annotation_keypoints_list.activate(target_index)
        self.annotation_keypoints_list.see(target_index)
        self.current_marker_var.set(f"Current marker: {target_name}")

    def _frame_annotation_measurement_count(self, frame_number: int, camera_names: list[str]) -> int:
        if self.pose_data is None:
            return 0
        annotation_pose_data = annotation_pose_data_for_frame(
            self.pose_data,
            camera_names=camera_names,
            frame_number=int(frame_number),
            annotation_payload=self.annotation_payload,
        )
        return int(np.sum(np.asarray(annotation_pose_data.scores, dtype=float) > 0.0))

    def selected_annotation_camera_names(self) -> list[str]:
        if self.pose_data is None:
            return []
        all_camera_names = [str(name) for name in self.pose_data.camera_names]
        indices = [int(index) for index in self.annotation_cameras_list.curselection()]
        if not indices:
            return list(all_camera_names)
        return [all_camera_names[index] for index in indices if 0 <= index < len(all_camera_names)]

    def update_camera_summary(self) -> None:
        total_count = self.annotation_cameras_list.size()
        selected_count = len(self.annotation_cameras_list.curselection()) if total_count else 0
        if total_count and selected_count == 0:
            selected_count = total_count
        self.annotation_cameras_summary.set(f"Cameras (n={selected_count}/{total_count})")

    def on_camera_selection_changed(self) -> None:
        self._clear_pending_reprojection()
        self._clear_kinematic_assist_preview()
        self.update_camera_summary()
        self.refresh_preview()

    def selected_keypoint_name(self) -> str:
        selection = self.annotation_keypoints_list.curselection()
        index = int(selection[0]) if selection else 0
        return str(self.annotation_keypoints_list.get(index))

    def on_keypoint_selection_changed(self) -> None:
        self._clear_pending_reprojection()
        self.current_marker_var.set(f"Current marker: {self.selected_keypoint_name()}")
        self.refresh_preview()

    def _store_current_annotation_view_limits(self) -> None:
        if not hasattr(self, "preview_figure") or self.preview_figure is None:
            return
        for ax in list(getattr(self.preview_figure, "axes", [])):
            camera_name = self._axis_to_camera.get(ax)
            if camera_name is None:
                continue
            self._annotation_view_limits[str(camera_name)] = (
                tuple(float(value) for value in ax.get_xlim()),
                tuple(float(value) for value in ax.get_ylim()),
            )

    def _annotation_view_limits_for_camera(
        self, camera_name: str
    ) -> tuple[tuple[float, float], tuple[float, float]] | tuple[None, None]:
        limits = getattr(self, "_annotation_view_limits", {}).get(str(camera_name))
        if limits is None:
            return None, None
        return limits

    def _step_annotation_keypoint(self, delta: int, event=None) -> str:
        if not hasattr(self, "annotation_keypoints_list"):
            return "break"
        size = int(self.annotation_keypoints_list.size())
        if size <= 0:
            return "break"
        selection = self.annotation_keypoints_list.curselection()
        current_index = int(selection[0]) if selection else 0
        next_index = int((current_index + int(delta)) % size)
        if next_index != current_index or not selection:
            self.annotation_keypoints_list.selection_clear(0, tk.END)
            self.annotation_keypoints_list.selection_set(next_index)
            self.annotation_keypoints_list.activate(next_index)
            self.annotation_keypoints_list.see(next_index)
            self.on_keypoint_selection_changed()
        return "break"

    def _bind_annotation_keypoint_navigation(self, widget) -> None:
        widget.bind("<Up>", lambda event: self._step_annotation_keypoint(-1, event), add="+")
        widget.bind("<Down>", lambda event: self._step_annotation_keypoint(1, event), add="+")
        for child in widget.winfo_children():
            self._bind_annotation_keypoint_navigation(child)

    def _cancel_pending_reprojection_from_tk_event(self, event) -> None:
        if not self._pending_reprojection_points:
            return
        widget = getattr(event, "widget", None)
        if widget is None:
            return
        if hasattr(self, "reproject_button") and widget is self.reproject_button:
            return
        self._clear_pending_reprojection()
        self.refresh_preview()

    def _bind_cancel_pending_reprojection(self, widget) -> None:
        widget.bind("<Button-1>", self._cancel_pending_reprojection_from_tk_event, add="+")
        for child in widget.winfo_children():
            self._bind_cancel_pending_reprojection(child)

    def current_frame_number(self) -> int:
        if self.pose_data is None or len(self.pose_data.frames) == 0:
            return 0
        frame_idx = max(0, min(len(self.pose_data.frames) - 1, int(round(self.frame_var.get()))))
        return int(self.pose_data.frames[frame_idx])

    def _current_images_root(self) -> Path | None:
        images_value = self.images_root_entry.get().strip()
        if images_value:
            return ROOT / images_value
        return self.images_root

    def _set_frame_index(self, frame_idx: int) -> None:
        clamped = max(0, int(frame_idx))
        if self.pose_data is not None and len(self.pose_data.frames) > 0:
            clamped = min(clamped, len(self.pose_data.frames) - 1)
        if clamped != int(self._current_frame_idx):
            self.save_annotations()
            self._clear_pending_reprojection()
            self._clear_kinematic_assist_preview()
        self._current_frame_idx = clamped
        self.frame_var.set(clamped)

    def _bind_frame_navigation(self, widget: tk.Widget) -> None:
        if widget is self.frame_scale:
            widget.bind("<Button-1>", self._on_frame_scale_click)
            widget.bind("<B1-Motion>", self._on_frame_scale_drag)
            widget.bind("<ButtonRelease-1>", self._on_frame_scale_release)
        else:
            widget.bind("<Enter>", lambda _event: widget.focus_set())
        widget.bind("<Left>", lambda _event: self.step_frame(-1))
        widget.bind("<Right>", lambda _event: self.step_frame(1))

    def _frame_from_scale_event(self, event) -> int:
        return frame_from_slider_click(
            x=event.x,
            width=self.frame_scale.winfo_width(),
            from_value=self.frame_scale.cget("from"),
            to_value=self.frame_scale.cget("to"),
        )

    def _on_frame_scale_click(self, event) -> str:
        self._dragging_frame_scale = True
        self.frame_scale.focus_set()
        frame = self._frame_from_scale_event(event)
        self._set_frame_index(frame)
        self.refresh_preview()
        return "break"

    def _on_frame_scale_drag(self, event) -> str:
        if not self._dragging_frame_scale:
            return "break"
        self._set_frame_index(self._frame_from_scale_event(event))
        self.refresh_preview()
        return "break"

    def _on_frame_scale_release(self, event) -> str:
        if not self._dragging_frame_scale:
            return "break"
        self._dragging_frame_scale = False
        self._set_frame_index(self._frame_from_scale_event(event))
        self.refresh_preview()
        return "break"

    def on_frame_scale_changed(self, _value) -> None:
        self._set_frame_index(int(round(self.frame_var.get())))
        self.refresh_preview()

    def _clear_pending_reprojection(self) -> None:
        self._pending_reprojection_points = {}
        if hasattr(self, "reproject_button_var"):
            self.reproject_button_var.set("Reproject")

    def _reference_projected_points(
        self, camera_name: str, frame_number: int
    ) -> tuple[np.ndarray | None, str | None, str]:
        selected = self.selected_reconstruction_names()
        if not selected:
            return None, None, "#6c5ce7"
        reference_name = str(selected[-1]).strip()
        if not reference_name or reference_name == "raw":
            return None, None, "#6c5ce7"
        recon_dir = reconstruction_dir_by_name(current_dataset_dir(self.state), reference_name)
        if recon_dir is None:
            return None, None, reconstruction_display_color(self.state, reference_name)
        payload = load_bundle_payload(recon_dir)
        points_3d = np.asarray(payload.get("points_3d"), dtype=float) if "points_3d" in payload else None
        bundle_frames = np.asarray(payload.get("frames"), dtype=int) if "frames" in payload else None
        bundle_camera_names = (
            [str(name) for name in np.asarray(payload.get("camera_names"), dtype=object).tolist()]
            if "camera_names" in payload
            else list(self.pose_data.camera_names if self.pose_data is not None else [])
        )
        if points_3d is None or points_3d.ndim != 3 or bundle_frames is None:
            return (
                None,
                reconstruction_legend_label(self.state, reference_name),
                reconstruction_display_color(self.state, reference_name),
            )
        matches = np.flatnonzero(bundle_frames == int(frame_number))
        if matches.size == 0 or camera_name not in bundle_camera_names:
            return (
                None,
                reconstruction_legend_label(self.state, reference_name),
                reconstruction_display_color(self.state, reference_name),
            )
        frame_idx = int(matches[0])
        calibration = self.calibrations.get(str(camera_name)) if self.calibrations is not None else None
        if calibration is None:
            return (
                None,
                reconstruction_legend_label(self.state, reference_name),
                reconstruction_display_color(self.state, reference_name),
            )
        projected = np.full((points_3d.shape[1], 2), np.nan, dtype=float)
        for kp_idx, point_3d in enumerate(np.asarray(points_3d[frame_idx], dtype=float)):
            if np.all(np.isfinite(point_3d)):
                projected[kp_idx] = calibration.project_point(point_3d)
        return (
            projected,
            reconstruction_legend_label(self.state, reference_name),
            reconstruction_display_color(self.state, reference_name),
        )

    def _compute_pending_reprojection_points(self) -> dict[tuple[str, int, str], np.ndarray]:
        if self.pose_data is None or self.calibrations is None:
            return {}
        frame_number = self.current_frame_number()
        camera_names = self.selected_annotation_camera_names()
        pending: dict[tuple[str, int, str], np.ndarray] = {}
        for target_camera_name in camera_names:
            for keypoint_name in self.annotation_keypoint_names():
                existing_xy = self._annotation_xy(target_camera_name, frame_number, keypoint_name)
                if existing_xy is None:
                    continue
                source_camera_names: list[str] = []
                source_points_2d: list[np.ndarray] = []
                for source_camera_name in camera_names:
                    if source_camera_name == target_camera_name:
                        continue
                    source_xy = self._annotation_xy(source_camera_name, frame_number, keypoint_name)
                    if source_xy is None:
                        continue
                    source_camera_names.append(source_camera_name)
                    source_points_2d.append(source_xy)
                reprojection_xy = annotation_triangulated_reprojection(
                    self.calibrations,
                    target_camera_name=target_camera_name,
                    source_camera_names=source_camera_names,
                    source_points_2d=source_points_2d,
                )
                if reprojection_xy is None or not np.all(np.isfinite(reprojection_xy)):
                    continue
                pending[(str(target_camera_name), int(frame_number), str(keypoint_name))] = np.asarray(
                    reprojection_xy, dtype=float
                )
        return pending

    def _confirm_pending_reprojection(self) -> None:
        for (camera_name, frame_number, keypoint_name), xy in self._pending_reprojection_points.items():
            set_annotation_point(
                self.annotation_payload,
                camera_name=camera_name,
                frame_number=int(frame_number),
                keypoint_name=keypoint_name,
                xy=[float(xy[0]), float(xy[1])],
            )
        self.save_annotations()
        self._clear_pending_reprojection()

    def on_reproject_button(self) -> None:
        if self._pending_reprojection_points:
            self._confirm_pending_reprojection()
            self.refresh_preview()
            return
        self._pending_reprojection_points = self._compute_pending_reprojection_points()
        self.reproject_button_var.set("Confirm" if self._pending_reprojection_points else "Reproject")
        self.refresh_preview()

    def step_frame(self, delta: int) -> None:
        if self.pose_data is None or len(self.pose_data.frames) == 0:
            return
        current_index = int(round(self.frame_var.get()))
        current_frame_number = int(self.pose_data.frames[current_index])
        try:
            selected_camera_names = self.selected_annotation_camera_names()
        except Exception:
            selected_camera_names = []
        candidates = self._navigable_annotation_frame_local_indices()
        if candidates:
            next_frame = step_frame_index_within_subset(current_index, int(delta), candidates)
        else:
            next_frame = find_annotation_frame_with_images(
                frames=self.pose_data.frames,
                current_index=current_index,
                direction=int(delta),
                camera_names=selected_camera_names,
                images_root=self._current_images_root(),
            )
        if next_frame is None:
            return
        if (
            self.pose_data is not None
            and bool(getattr(self, "kinematic_assist_var", None) and self.kinematic_assist_var.get())
            and next_frame != current_index
        ):
            biomod_path = self._selected_kinematic_biomod_path()
            if biomod_path is not None and biomod_path.exists():
                try:
                    import biorbd

                    model = biorbd.Model(str(biomod_path))
                    current_state, _source_frame, _is_exact = self._selected_or_nearest_kinematic_state_info(
                        current_frame_number, model
                    )
                    if current_state is not None:
                        next_frame_number = int(self.pose_data.frames[int(next_frame)])
                        frame_delta = int(next_frame_number) - int(current_frame_number)
                        propagated_state = propagate_annotation_kinematic_state(
                            model,
                            current_state,
                            dt=1.0 / float(self.state.fps_var.get()),
                            frame_delta=frame_delta,
                        )
                        self.kinematic_frame_states[self._kinematic_state_key(next_frame_number)] = propagated_state
                except Exception:
                    pass
        self._set_frame_index(next_frame)
        if bool(getattr(self, "kinematic_assist_var", None) and self.kinematic_assist_var.get()):
            try:
                next_frame_number = int(self.pose_data.frames[int(next_frame)])
                if self._frame_annotation_measurement_count(next_frame_number, selected_camera_names) > 0:
                    self._estimate_kinematic_q()
            except Exception:
                pass
        self.refresh_preview()

    def load_resources(self) -> None:
        try:
            with gui_busy_popup(self, title="Annotation", message="Chargement des données d'annotation...") as popup:
                keypoints_path = ROOT / self.state.keypoints_var.get()
                calib_path = ROOT / self.state.calib_var.get()
                popup.set_status("Chargement des calibrations et des 2D bruts...")
                self.calibrations, self.pose_data = get_cached_pose_data(
                    self.state,
                    keypoints_path=keypoints_path,
                    calib_path=calib_path,
                    data_mode="raw",
                    smoothing_window=int(self.state.pose_filter_window_var.get()),
                    outlier_threshold_ratio=float(self.state.pose_outlier_ratio_var.get()),
                    lower_percentile=float(self.state.pose_p_low_var.get()),
                    upper_percentile=float(self.state.pose_p_high_var.get()),
                )
                annotation_path_value = self.annotations_path_entry.get().strip()
                self.annotation_path = (
                    (ROOT / annotation_path_value) if annotation_path_value else default_annotation_path(keypoints_path)
                )
                popup.set_status("Chargement des annotations et des modèles existants...")
                self.annotation_payload = load_annotation_payload(self.annotation_path, keypoints_path=keypoints_path)
                self.refresh_kinematic_model_choices()
                self.annotation_cameras_list.delete(0, tk.END)
                for camera_name in self.pose_data.camera_names:
                    self.annotation_cameras_list.insert(tk.END, str(camera_name))
                    self.annotation_cameras_list.selection_set(tk.END)
                self.update_camera_summary()
                self.current_marker_var.set(f"Current marker: {self.selected_keypoint_name()}")
                self.frame_scale.configure(to=max(len(self.pose_data.frames) - 1, 0))
                self._set_frame_index(min(int(self.frame_var.get()), max(len(self.pose_data.frames) - 1, 0)))
                self.crop_limits_cache = {}
                self.crop_limits_key = None
                self._navigable_frame_cache = {}
                self._clear_kinematic_assist_preview()
                self._clear_pending_reprojection()
                self.on_frame_filter_changed()
                self.refresh_preview()
        except Exception as exc:
            messagebox.showerror("Annotation", str(exc))

    def save_annotations(self) -> None:
        if self.annotation_path is None:
            return
        save_annotation_payload(self.annotation_path, self.annotation_payload)

    def _clear_kinematic_assist_preview(self) -> None:
        self.kinematic_q_current = None
        self.kinematic_state_current = None
        self.kinematic_q_names = []
        self.kinematic_projected_points = None
        self.kinematic_segmented_back_projected = {}

    def _kinematic_state_key(self, frame_number: int) -> tuple[str, int]:
        return (str(self.kinematic_model_var.get()).strip(), int(frame_number))

    def _selected_or_nearest_kinematic_state_info(
        self, frame_number: int, model
    ) -> tuple[np.ndarray | None, int | None, bool]:
        model_label = str(self.kinematic_model_var.get()).strip()
        info = resolve_annotation_kinematic_state_info(
            getattr(self, "kinematic_frame_states", {}),
            model_label=model_label,
            frame_number=int(frame_number),
            model=model,
        )
        return info.state, info.source_frame, info.is_exact

    def _selected_or_nearest_kinematic_state(self, frame_number: int, model) -> np.ndarray | None:
        state, _source_frame, _is_exact = self._selected_or_nearest_kinematic_state_info(frame_number, model)
        return state

    def _annotation_window_frame_numbers(self, frame_number: int) -> list[int]:
        if self.pose_data is None or len(self.pose_data.frames) == 0:
            return [int(frame_number)]
        frame_values = np.asarray(self.pose_data.frames, dtype=int)
        if frame_values.size == 0:
            return [int(frame_number)]
        current_matches = np.flatnonzero(frame_values == int(frame_number))
        if current_matches.size > 0:
            current_index = int(current_matches[0])
        else:
            current_index = int(np.argmin(np.abs(frame_values - int(frame_number))))
        start = max(0, current_index - ANNOTATION_KINEMATIC_WINDOW_RADIUS)
        end = min(frame_values.size, current_index + ANNOTATION_KINEMATIC_WINDOW_RADIUS + 1)
        return [int(value) for value in frame_values[start:end]]

    def _annotation_pose_data_by_frame(
        self,
        frame_numbers: list[int],
        camera_names: list[str],
    ) -> dict[int, PoseData]:
        if self.pose_data is None:
            return {}
        pose_data_by_frame: dict[int, PoseData] = {}
        for frame_number in frame_numbers:
            if self._frame_annotation_measurement_count(int(frame_number), camera_names) <= 0:
                continue
            pose_data_by_frame[int(frame_number)] = annotation_pose_data_for_frame(
                self.pose_data,
                camera_names=camera_names,
                frame_number=int(frame_number),
                annotation_payload=self.annotation_payload,
            )
        return pose_data_by_frame

    def _set_kinematic_preview_from_q(self, biomod_path: Path, camera_names: list[str], q_values: np.ndarray) -> None:
        q_values = np.asarray(q_values, dtype=float).reshape(-1)
        q_series = q_values.reshape(1, -1)
        self.kinematic_q_current = q_values
        self.kinematic_state_current = None
        self.kinematic_projected_points = project_points_all_cameras(
            biorbd_markers_from_q(biomod_path, q_series), self.calibrations, camera_names
        )
        segmented_overlay = segmented_back_overlay_from_q(biomod_path, q_series)
        self.kinematic_segmented_back_projected = (
            {
                overlay_name: project_points_all_cameras(points, self.calibrations, camera_names)
                for overlay_name, points in segmented_overlay.items()
            }
            if segmented_overlay
            else {}
        )

    def _estimate_kinematic_q(
        self,
        *,
        keypoint_name: str | None = None,
    ) -> np.ndarray:
        biomod_path = self._selected_kinematic_biomod_path()
        if biomod_path is None or not biomod_path.exists():
            raise ValueError("Choose an existing bioMod first.")
        camera_names = self.selected_annotation_camera_names()
        if not camera_names:
            raise ValueError("Select at least one camera.")
        frame_number = self.current_frame_number()
        import biorbd

        model = biorbd.Model(str(biomod_path))
        self.kinematic_q_names = biorbd_q_names(model)
        previous_state, source_frame, is_exact = self._selected_or_nearest_kinematic_state_info(frame_number, model)
        previous_q = None if previous_state is None else np.asarray(previous_state[: model.nbQ()], dtype=float)
        seed_source = "nearest state"
        estimated_state = None
        n_triangulated = 0
        if previous_state is not None:
            estimated_state = np.array(previous_state, copy=True)
            seed_source = "current state" if is_exact else f"nearest frame {source_frame}"
        else:
            triangulated_points = triangulate_annotation_frame_points(
                self.calibrations,
                camera_names=camera_names,
                frame_number=frame_number,
                annotation_payload=self.annotation_payload,
            )
            n_triangulated = int(np.sum(np.all(np.isfinite(triangulated_points), axis=1)))
            if n_triangulated < 2:
                raise ValueError("Not enough annotated support to initialize q. Add points on more views first.")
            reconstruction = annotation_reconstruction_from_points(
                triangulated_points, frame_number=frame_number, n_cameras=len(camera_names)
            )
            triangulation_state = initial_state_from_triangulation(model, reconstruction)
            estimated_state = annotation_state_from_q(
                model, np.asarray(triangulation_state[: model.nbQ()], dtype=float)
            )
            seed_source = f"triangulation ({n_triangulated} markers)"
        if keypoint_name is not None:
            estimated_state[model.nbQ() : 3 * model.nbQ()] = 0.0
        bootstrap_summary = " | local ekf skipped"
        direct_summary = ""
        annotation_pose_data = None
        try:
            annotation_pose_data = annotation_pose_data_for_frame(
                self.pose_data,
                camera_names=camera_names,
                frame_number=frame_number,
                annotation_payload=self.annotation_payload,
            )
            valid_measurements = int(np.sum(np.asarray(annotation_pose_data.scores) > 0.0))
            min_measurements = 1 if previous_state is not None else 2
            if valid_measurements >= min_measurements:
                passes = (
                    ANNOTATION_KINEMATIC_CLICK_PASSES
                    if keypoint_name is not None
                    else ANNOTATION_KINEMATIC_BOOTSTRAP_PASSES
                    * (ANNOTATION_KINEMATIC_INITIAL_BOOTSTRAP_MULTIPLIER if previous_state is None else 1)
                )
                refined_state, bootstrap_diagnostics = refine_annotation_q_with_local_ekf(
                    model=model,
                    calibrations=self.calibrations,
                    pose_data=annotation_pose_data,
                    frame_number=frame_number,
                    seed_state=estimated_state,
                    fps=float(self.state.fps_var.get()),
                    passes=passes,
                    measurement_noise_scale=1.0,
                    process_noise_scale=1.0,
                    epipolar_threshold_px=DEFAULT_EPIPOLAR_THRESHOLD_PX,
                    q_names=self.kinematic_q_names,
                    keypoint_name=None,
                )
                estimated_state = np.asarray(refined_state, dtype=float)
                bootstrap_summary = f" | local ekf {int(bootstrap_diagnostics.get('completed_passes', 0))}/" f"{passes}"
                if bool(bootstrap_diagnostics.get("used_fallback")):
                    bootstrap_summary += " fallback"
            else:
                bootstrap_summary = f" | local ekf skipped ({valid_measurements} meas)"
        except Exception:
            bootstrap_summary = " | local ekf fallback"
        if keypoint_name is not None and annotation_pose_data is not None:
            try:
                direct_state, direct_diagnostics = refine_annotation_q_with_direct_measurements(
                    model=model,
                    calibrations=self.calibrations,
                    pose_data=annotation_pose_data,
                    seed_state=estimated_state,
                    passes=ANNOTATION_KINEMATIC_CLICK_DIRECT_PASSES,
                    measurement_std_px=2.0,
                    q_names=self.kinematic_q_names,
                    keypoint_name=None,
                )
                if not bool(direct_diagnostics.get("used_fallback")):
                    estimated_state = np.asarray(direct_state, dtype=float)
                    direct_summary = (
                        f" | direct fit {int(direct_diagnostics.get('completed_passes', 0))}/"
                        f"{ANNOTATION_KINEMATIC_CLICK_DIRECT_PASSES}"
                    )
                else:
                    direct_summary = " | direct fit fallback"
            except Exception:
                direct_summary = " | direct fit fallback"
        if not hasattr(self, "kinematic_frame_states") or self.kinematic_frame_states is None:
            self.kinematic_frame_states = {}
        temporal_summary = ""
        if keypoint_name is None:
            try:
                pose_data_by_frame = self._annotation_pose_data_by_frame(
                    self._annotation_window_frame_numbers(frame_number),
                    camera_names,
                )
                if len(pose_data_by_frame) > 1:
                    refined_window_states, window_diagnostics = refine_annotation_window_states(
                        model=model,
                        calibrations=self.calibrations,
                        pose_data_by_frame=pose_data_by_frame,
                        center_frame_number=frame_number,
                        seed_state=estimated_state,
                        fps=float(self.state.fps_var.get()),
                        passes=ANNOTATION_KINEMATIC_WINDOW_PASSES,
                        epipolar_threshold_px=DEFAULT_EPIPOLAR_THRESHOLD_PX,
                        q_names=self.kinematic_q_names,
                    )
                    for saved_frame_number, saved_state in refined_window_states.items():
                        store_annotation_kinematic_state(
                            self.kinematic_frame_states,
                            model_label=str(self.kinematic_model_var.get()).strip(),
                            frame_number=int(saved_frame_number),
                            model=model,
                            state=np.asarray(saved_state, dtype=float),
                        )
                    if frame_number in refined_window_states:
                        estimated_state = np.asarray(refined_window_states[frame_number], dtype=float)
                    temporal_summary = (
                        f" | local window {int(window_diagnostics.get('completed_frames', 0))}/"
                        f"{len(pose_data_by_frame)}"
                    )
            except Exception:
                temporal_summary = ""
        estimated_state = store_annotation_kinematic_state(
            self.kinematic_frame_states,
            model_label=str(self.kinematic_model_var.get()).strip(),
            frame_number=frame_number,
            model=model,
            state=estimated_state,
        )
        self.kinematic_state_current = np.asarray(estimated_state, dtype=float)
        self._set_kinematic_preview_from_q(biomod_path, camera_names, estimated_state[: model.nbQ()])
        self.kinematic_state_current = np.asarray(estimated_state, dtype=float)
        if keypoint_name is None:
            self.kinematic_status_var.set(f"Estimated q from {seed_source}{bootstrap_summary}{temporal_summary}.")
        else:
            warm_start_text = ""
            if previous_state is not None and source_frame is not None:
                warm_start_text = " current frame" if is_exact else f" nearest frame {source_frame}"
                warm_start_text = f" from{warm_start_text}"
            self.kinematic_status_var.set(
                f"Updated model after {keypoint_name} using {seed_source}{warm_start_text}{bootstrap_summary}"
                f"{direct_summary}."
            )
        return np.asarray(estimated_state[: model.nbQ()], dtype=float)

    def estimate_kinematic_assist_state(self) -> None:
        if self.pose_data is None or self.calibrations is None:
            return
        try:
            self._estimate_kinematic_q()
            self.refresh_preview()
        except Exception as exc:
            self._clear_kinematic_assist_preview()
            self.kinematic_status_var.set("")
            messagebox.showerror("Annotation", str(exc))

    def _ensure_crop_limits(self, camera_names: list[str]) -> dict[str, np.ndarray]:
        if self.pose_data is None:
            return {}
        all_camera_names = [str(name) for name in self.pose_data.camera_names]
        camera_indices = [all_camera_names.index(str(name)) for name in camera_names if str(name) in all_camera_names]
        if not camera_indices:
            return {}
        raw_points = np.asarray(
            self.pose_data.raw_keypoints if self.pose_data.raw_keypoints is not None else self.pose_data.keypoints,
            dtype=float,
        )
        raw_points = raw_points[camera_indices]
        cache_key = (
            id(self.pose_data),
            tuple(camera_indices),
            tuple(camera_names),
            0.2,
        )
        if self.crop_limits_key != cache_key:
            self.crop_limits_cache = compute_pose_crop_limits_2d(raw_points, self.calibrations, camera_names, 0.2)
            self.crop_limits_key = cache_key
        return self.crop_limits_cache

    def _annotation_xy(self, camera_name: str, frame_number: int, keypoint_name: str) -> np.ndarray | None:
        xy, _score = get_annotation_point(
            self.annotation_payload,
            camera_name=camera_name,
            frame_number=frame_number,
            keypoint_name=keypoint_name,
        )
        return None if xy is None else np.asarray(xy, dtype=float)

    def _annotation_support_camera_names(self) -> list[str]:
        if self.pose_data is None:
            return []
        return [str(name) for name in self.pose_data.camera_names]

    def _reference_projected_keypoint(
        self,
        camera_name: str,
        frame_number: int,
        keypoint_name: str,
    ) -> np.ndarray | None:
        if keypoint_name not in KP_INDEX:
            return None
        projected_points, _label, _color = self._reference_projected_points(camera_name, frame_number)
        if projected_points is None:
            return None
        xy = np.asarray(projected_points[KP_INDEX[keypoint_name]], dtype=float)
        return xy if np.all(np.isfinite(xy)) else None

    def _triangulated_hint_for_keypoint(
        self,
        camera_name: str,
        frame_number: int,
        keypoint_name: str,
    ) -> np.ndarray | None:
        support_camera_names = []
        support_points = []
        for other_camera_name in self._annotation_support_camera_names():
            if other_camera_name == str(camera_name):
                continue
            other_xy = self._annotation_xy(other_camera_name, frame_number, keypoint_name)
            if other_xy is None:
                continue
            support_camera_names.append(str(other_camera_name))
            support_points.append(np.asarray(other_xy, dtype=float))
        return annotation_triangulated_reprojection(
            self.calibrations,
            target_camera_name=str(camera_name),
            source_camera_names=support_camera_names,
            source_points_2d=support_points,
        )

    def _epipolar_lines_for_keypoint(
        self,
        target_camera_name: str,
        frame_number: int,
        keypoint_name: str,
    ) -> list[np.ndarray]:
        lines: list[np.ndarray] = []
        for other_camera_name in self._annotation_support_camera_names():
            if str(other_camera_name) == str(target_camera_name):
                continue
            other_xy = self._annotation_xy(other_camera_name, frame_number, keypoint_name)
            if other_xy is None:
                continue
            line = annotation_epipolar_guides(
                self.calibrations,
                str(other_camera_name),
                str(target_camera_name),
                other_xy,
            )
            if line is not None:
                lines.append(np.asarray(line, dtype=float))
        return lines

    def _snap_annotation_xy(
        self,
        *,
        camera_name: str,
        frame_number: int,
        keypoint_name: str,
        pointer_xy: np.ndarray,
    ) -> np.ndarray:
        pointer_xy = np.asarray(pointer_xy, dtype=float).reshape(2)
        candidates: list[np.ndarray] = []
        if bool(getattr(self, "snap_reprojection_var", None) and self.snap_reprojection_var.get()):
            triangulated = self._triangulated_hint_for_keypoint(camera_name, frame_number, keypoint_name)
            if triangulated is not None and np.all(np.isfinite(triangulated)):
                candidates.append(np.asarray(triangulated, dtype=float))
            reference_xy = self._reference_projected_keypoint(camera_name, frame_number, keypoint_name)
            if reference_xy is not None:
                candidates.append(np.asarray(reference_xy, dtype=float))
        if bool(getattr(self, "snap_epipolar_var", None) and self.snap_epipolar_var.get()):
            lines = self._epipolar_lines_for_keypoint(camera_name, frame_number, keypoint_name)
            intersection = annotation_intersect_epipolar_lines(lines)
            if intersection is not None:
                candidates.append(np.asarray(intersection, dtype=float))
            for line in lines:
                projected = annotation_project_point_to_line(line, pointer_xy)
                if projected is not None:
                    candidates.append(np.asarray(projected, dtype=float))
        if not candidates:
            return np.array(pointer_xy, copy=True)
        distances = np.array([np.linalg.norm(candidate - pointer_xy) for candidate in candidates], dtype=float)
        if not np.any(np.isfinite(distances)):
            return np.array(pointer_xy, copy=True)
        best_index = int(np.nanargmin(distances))
        if float(distances[best_index]) > ANNOTATION_SNAP_RADIUS_PX:
            return np.array(pointer_xy, copy=True)
        return np.asarray(candidates[best_index], dtype=float)

    def _advance_to_next_keypoint(self) -> None:
        if not self.advance_marker_var.get():
            return
        size = int(self.annotation_keypoints_list.size())
        if size <= 0:
            return
        selection = self.annotation_keypoints_list.curselection()
        index = int(selection[0]) if selection else 0
        next_index = int((index + 1) % size)
        self.annotation_keypoints_list.selection_clear(0, tk.END)
        self.annotation_keypoints_list.selection_set(next_index)
        self.annotation_keypoints_list.activate(next_index)
        self.annotation_keypoints_list.see(next_index)
        self.on_keypoint_selection_changed()

    def _selected_reconstruction(self) -> str | None:
        selected = list(getattr(self.state, "shared_reconstruction_selection", []))
        return str(selected[-1]) if selected else None

    def _annotation_jump_context(self) -> str:
        selected_name = self._selected_reconstruction()
        if not selected_name or self.pose_data is None:
            self.annotation_jump_analysis = None
            return ""
        analysis = shared_jump_analysis_for_reconstruction(self.state, selected_name)
        self.annotation_jump_analysis = analysis
        if analysis is None:
            return ""
        frame_number = self.current_frame_number()
        for jump_index, jump in enumerate(analysis.jumps, start=1):
            if int(jump.segment.start) <= int(frame_number) <= int(jump.segment.end):
                return f"Jump context: S{jump_index} | {jump.classification} | frames {jump.segment.start}-{jump.segment.end}"
        return "Jump context: between jumps"

    def _frame_filter_mode(self) -> str:
        return resolve_annotation_frame_filter_mode(self.frame_filter_var.get(), ANNOTATION_FRAME_FILTER_OPTIONS)

    def _pose_data_mode_for_annotation_filters(self) -> str:
        value = str(self.state.pose_data_mode_var.get()).strip().lower()
        return value if value in {"raw", "cleaned"} else "cleaned"

    def _annotation_flip_frame_local_indices(self) -> list[int]:
        if self.pose_data is None or self.calibrations is None:
            return []
        correction_mode = current_calibration_correction_mode(self.state)
        correction_to_method = {
            "flip_epipolar": "epipolar",
            "flip_epipolar_fast": "epipolar_fast",
            "flip_epipolar_viterbi": "epipolar_viterbi",
            "flip_epipolar_fast_viterbi": "epipolar_fast_viterbi",
            "flip_triangulation": "triangulation_exhaustive",
        }
        method = correction_to_method.get(correction_mode)
        if method is None:
            return []
        keypoints_path = ROOT / self.state.keypoints_var.get()
        calib_path = ROOT / self.state.calib_var.get()
        pose_data_mode = self._pose_data_mode_for_annotation_filters()
        _calibrations, pose_data_for_flip = get_cached_pose_data(
            self.state,
            keypoints_path=keypoints_path,
            calib_path=calib_path,
            data_mode=pose_data_mode,
            smoothing_window=int(self.state.pose_filter_window_var.get()),
            outlier_threshold_ratio=float(self.state.pose_outlier_ratio_var.get()),
            lower_percentile=float(self.state.pose_p_low_var.get()),
            upper_percentile=float(self.state.pose_p_high_var.get()),
        )
        suspect_mask, _diagnostics, _compute_time_s, _cache_path, _source = load_or_compute_left_right_flip_cache(
            output_dir=current_dataset_dir(self.state),
            pose_data=pose_data_for_flip,
            calibrations=self.calibrations,
            method=method,
            pose_data_mode=pose_data_mode,
            pose_filter_window=int(self.state.pose_filter_window_var.get()),
            pose_outlier_threshold_ratio=float(self.state.pose_outlier_ratio_var.get()),
            pose_amplitude_lower_percentile=float(self.state.pose_p_low_var.get()),
            pose_amplitude_upper_percentile=float(self.state.pose_p_high_var.get()),
            improvement_ratio=float(self.state.flip_improvement_ratio_var.get()),
            min_gain_px=float(self.state.flip_min_gain_px_var.get()),
            min_other_cameras=int(self.state.flip_min_other_cameras_var.get()),
            restrict_to_outliers=bool(self.state.flip_restrict_to_outliers_var.get()),
            outlier_percentile=float(self.state.flip_outlier_percentile_var.get()),
            outlier_floor_px=float(self.state.flip_outlier_floor_px_var.get()),
            tau_px=(
                DEFAULT_EPIPOLAR_THRESHOLD_PX
                if method in {"epipolar", "epipolar_fast", "epipolar_viterbi", "epipolar_fast_viterbi"}
                else DEFAULT_REPROJECTION_THRESHOLD_PX
            ),
            temporal_weight=float(self.state.flip_temporal_weight_var.get()),
            temporal_tau_px=float(self.state.flip_temporal_tau_px_var.get()),
        )
        selected_camera_names = set(self.selected_annotation_camera_names())
        camera_indices = [
            idx for idx, name in enumerate(pose_data_for_flip.camera_names) if str(name) in selected_camera_names
        ]
        if not camera_indices:
            camera_indices = list(range(len(pose_data_for_flip.camera_names)))
        flagged_local_indices = np.flatnonzero(np.any(np.asarray(suspect_mask)[camera_indices], axis=0))
        frame_to_local = {int(frame): idx for idx, frame in enumerate(np.asarray(self.pose_data.frames, dtype=int))}
        return sorted(
            frame_to_local[int(frame)]
            for frame in np.asarray(pose_data_for_flip.frames, dtype=int)[flagged_local_indices]
            if int(frame) in frame_to_local
        )

    def _annotation_worst_reprojection_frame_local_indices(self) -> list[int]:
        if self.pose_data is None:
            return []
        selected_name = self._selected_reconstruction()
        if not selected_name:
            return []
        recon_dir = reconstruction_dir_by_name(current_dataset_dir(self.state), selected_name)
        if recon_dir is None:
            return []
        payload = load_bundle_payload(recon_dir)
        if "reprojection_error_per_view" not in payload or "frames" not in payload:
            return []
        errors = np.asarray(payload["reprojection_error_per_view"], dtype=float)
        bundle_frames = np.asarray(payload["frames"], dtype=int)
        bundle_camera_names = (
            [str(name) for name in np.asarray(payload["camera_names"], dtype=object).tolist()]
            if "camera_names" in payload
            else list(self.pose_data.camera_names)
        )
        selected_camera_names = set(self.selected_annotation_camera_names())
        camera_indices = [idx for idx, name in enumerate(bundle_camera_names) if name in selected_camera_names]
        if not camera_indices:
            camera_indices = list(range(errors.shape[2]))
        frame_errors = np.nanmean(errors[:, :, camera_indices], axis=(1, 2))
        finite_indices = np.flatnonzero(np.isfinite(frame_errors))
        if finite_indices.size == 0:
            return []
        worst_count = max(1, int(math.ceil(0.05 * float(finite_indices.size))))
        ranked_indices = finite_indices[np.argsort(frame_errors[finite_indices])]
        worst_bundle_indices = ranked_indices[-worst_count:]
        frame_to_local = {int(frame): idx for idx, frame in enumerate(np.asarray(self.pose_data.frames, dtype=int))}
        return sorted(
            frame_to_local[int(frame)] for frame in bundle_frames[worst_bundle_indices] if int(frame) in frame_to_local
        )

    def _filtered_annotation_frame_local_indices(self) -> list[int]:
        if self.pose_data is None:
            return []
        mode = self._frame_filter_mode()
        if mode == "flipped":
            filtered = self._annotation_flip_frame_local_indices()
        elif mode == "worst_reproj":
            filtered = self._annotation_worst_reprojection_frame_local_indices()
        else:
            filtered = list(range(len(self.pose_data.frames)))
        return fallback_annotation_filtered_indices(len(self.pose_data.frames), filtered)

    def _navigable_annotation_frame_local_indices(self) -> list[int]:
        if self.pose_data is None:
            return []
        filtered = self._filtered_annotation_frame_local_indices()
        camera_names = self.selected_annotation_camera_names()
        images_root = self._current_images_root()
        if images_root is None or not camera_names:
            return filtered
        cache_key = (
            tuple(int(value) for value in filtered),
            tuple(str(name) for name in camera_names),
            str(images_root.resolve()) if images_root.exists() else str(images_root),
        )
        cached = self._navigable_frame_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        resolved = navigable_annotation_frame_local_indices(
            np.asarray(self.pose_data.frames, dtype=int),
            filtered,
            camera_names,
            images_root,
        )
        self._navigable_frame_cache[cache_key] = list(resolved)
        return resolved

    def on_frame_filter_changed(self) -> None:
        if self.pose_data is None or len(self.pose_data.frames) == 0:
            return
        self._clear_pending_reprojection()
        candidates = self._filtered_annotation_frame_local_indices()
        clamped_index = clamp_index_to_subset(int(round(self.frame_var.get())), candidates)
        if clamped_index is not None and clamped_index != int(round(self.frame_var.get())):
            self._set_frame_index(clamped_index)
        self.refresh_preview()

    def _ensure_cursor_artists(self, ax) -> tuple[object, ...]:
        artists = self._cursor_artists.get(ax)
        if artists is not None:
            return artists
        artists = []
        for _ in range(4):
            line = ax.plot([], [], color="#f8f8f8", linewidth=1.2, zorder=30, solid_capstyle="butt")[0]
            line.set_path_effects([path_effects.Stroke(linewidth=2.6, foreground="black"), path_effects.Normal()])
            line.set_visible(False)
            artists.append(line)
        hover_text = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#111111",
            zorder=31,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#fff8dc", "edgecolor": "#333333", "alpha": 0.92},
        )
        hover_text.set_visible(False)
        artists.append(hover_text)
        self._cursor_artists[ax] = tuple(artists)
        return self._cursor_artists[ax]

    def _nearest_annotation_hover_entry(self, ax, x: float, y: float) -> dict[str, object] | None:
        entries = getattr(self, "_annotation_hover_entries", {}).get(ax, [])
        best_entry = None
        best_distance = None
        for entry in entries:
            point = np.asarray(entry.get("xy"), dtype=float).reshape(2)
            if not np.all(np.isfinite(point)):
                continue
            distance = float(np.linalg.norm(point - np.array([x, y], dtype=float)))
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_entry = entry
        if best_entry is None or best_distance is None or best_distance > ANNOTATION_HOVER_RADIUS_PX:
            return None
        return best_entry

    def _nearest_annotated_drag_entry(self, ax, x: float, y: float) -> dict[str, object] | None:
        hover_entry = self._nearest_annotation_hover_entry(ax, x, y)
        if hover_entry is None or str(hover_entry.get("source")) != "annotated":
            return None
        point = np.asarray(hover_entry.get("xy"), dtype=float).reshape(2)
        distance = float(np.linalg.norm(point - np.array([x, y], dtype=float)))
        if not np.isfinite(distance) or distance > ANNOTATION_DRAG_START_RADIUS_PX:
            return None
        return hover_entry

    def _annotation_hover_label(self, entry: dict[str, object]) -> str:
        keypoint_name = str(entry.get("keypoint_name", ""))
        source = str(entry.get("source", "")).strip()
        try:
            selected_keypoint = self.selected_keypoint_name()
        except Exception:
            selected_keypoint = ""
        current_suffix = " | current" if keypoint_name == selected_keypoint else ""
        if source:
            return f"{keypoint_name} | {source}{current_suffix}"
        return f"{keypoint_name}{current_suffix}"

    def _update_preview_cursor(self, event) -> None:
        if not hasattr(self, "preview_canvas_widget"):
            return
        active_ax = event.inaxes if event is not None else None
        has_position = active_ax is not None and event.xdata is not None and event.ydata is not None
        self.preview_canvas_widget.configure(cursor=("crosshair" if has_position else "arrow"))
        for ax, artists in list(self._cursor_artists.items()):
            visible = has_position and ax is active_ax
            hover_text = artists[4] if len(artists) >= 5 else None
            if not visible:
                for artist in artists:
                    if artist is not None:
                        artist.set_visible(False)
                continue
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            x = float(event.xdata)
            y = float(event.ydata)
            gap_x = max(3.0, 0.015 * abs(float(x1) - float(x0)))
            gap_y = max(3.0, 0.015 * abs(float(y1) - float(y0)))
            segments = (
                ([x0, x - gap_x], [y, y]),
                ([x + gap_x, x1], [y, y]),
                ([x, x], [y0, y - gap_y]),
                ([x, x], [y + gap_y, y1]),
            )
            for artist, (xs, ys) in zip(artists[:4], segments):
                if artist is None:
                    continue
                artist.set_data(xs, ys)
                artist.set_visible(True)
            if hover_text is not None:
                hover_entry = self._nearest_annotation_hover_entry(ax, x, y)
                if hover_entry is None:
                    hover_text.set_visible(False)
                else:
                    hover_text.set_text(self._annotation_hover_label(hover_entry))
                    hover_text.set_visible(True)
        self.preview_canvas.draw_idle()

    def _delete_nearest_annotation(self, camera_name: str, frame_number: int, xy: np.ndarray) -> bool:
        nearest_name = None
        nearest_distance = None
        point = np.asarray(xy, dtype=float).reshape(2)
        for keypoint_name in self.annotation_keypoint_names():
            annotated_xy = self._annotation_xy(camera_name, frame_number, keypoint_name)
            if annotated_xy is None:
                continue
            distance = float(np.linalg.norm(annotated_xy - point))
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_name = str(keypoint_name)
        if nearest_name is None or nearest_distance is None or nearest_distance > ANNOTATION_DELETE_RADIUS_PX:
            return False
        clear_annotation_point(
            self.annotation_payload,
            camera_name=camera_name,
            frame_number=frame_number,
            keypoint_name=nearest_name,
        )
        return True

    def on_preview_click(self, event) -> None:
        if self._pending_reprojection_points and (event.inaxes is None or event.xdata is None or event.ydata is None):
            self._clear_pending_reprojection()
            self.refresh_preview()
            return
        if self.pose_data is None or event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        if int(getattr(event, "button", 1)) == 2:
            self._pan_state = {
                "axes": event.inaxes,
                "xdata": float(event.xdata),
                "ydata": float(event.ydata),
                "xlim": tuple(event.inaxes.get_xlim()),
                "ylim": tuple(event.inaxes.get_ylim()),
            }
            return
        camera_name = self._axis_to_camera.get(event.inaxes)
        if camera_name is None:
            return
        keypoint_name = self.selected_keypoint_name()
        frame_number = self.current_frame_number()
        event_key = str(getattr(event, "key", "") or "").lower()
        delete_request = int(getattr(event, "button", 1)) == 3 or ("control" in event_key or "ctrl" in event_key)
        pointer_xy = np.array([float(event.xdata), float(event.ydata)], dtype=float)
        if delete_request:
            self._delete_nearest_annotation(camera_name, frame_number, pointer_xy)
        else:
            hover_entry = self._nearest_annotated_drag_entry(event.inaxes, float(event.xdata), float(event.ydata))
            self._drag_annotation_state = {
                "camera_name": str(camera_name),
                "frame_number": int(frame_number),
                "selected_keypoint_name": str(keypoint_name),
                "pressed_xy": np.array(pointer_xy, copy=True),
                "did_drag": False,
                "existing_keypoint_name": (
                    str(hover_entry.get("keypoint_name", "")) if hover_entry is not None else None
                ),
            }
            return
        self._clear_kinematic_assist_preview()
        if bool(getattr(self, "kinematic_assist_var", None) and self.kinematic_assist_var.get()):
            try:
                self._estimate_kinematic_q(keypoint_name=None if delete_request else keypoint_name)
            except Exception as exc:
                self._clear_kinematic_assist_preview()
                self.kinematic_status_var.set(f"Kinematic assist update skipped: {exc}")
        self.save_annotations()
        self.refresh_preview()

    def on_preview_scroll(self, event) -> None:
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        scale = 0.9 if str(getattr(event, "button", "")).lower() == "up" else 1.1
        x0, x1 = event.inaxes.get_xlim()
        y0, y1 = event.inaxes.get_ylim()
        cx = float(event.xdata)
        cy = float(event.ydata)
        event.inaxes.set_xlim(cx + (x0 - cx) * scale, cx + (x1 - cx) * scale)
        event.inaxes.set_ylim(cy + (y0 - cy) * scale, cy + (y1 - cy) * scale)
        camera_name = self._axis_to_camera.get(event.inaxes)
        if camera_name is not None:
            self._annotation_view_limits[str(camera_name)] = (
                tuple(float(value) for value in event.inaxes.get_xlim()),
                tuple(float(value) for value in event.inaxes.get_ylim()),
            )
        self.preview_canvas.draw_idle()

    def on_preview_motion(self, event) -> None:
        if self._drag_annotation_state is not None:
            if event.inaxes is None or event.xdata is None or event.ydata is None:
                self._update_preview_cursor(event)
                return
            camera_name = self._axis_to_camera.get(event.inaxes)
            if camera_name == self._drag_annotation_state.get("camera_name"):
                pointer_xy = np.array([float(event.xdata), float(event.ydata)], dtype=float)
                pressed_xy = np.asarray(self._drag_annotation_state.get("pressed_xy"), dtype=float).reshape(2)
                if not bool(self._drag_annotation_state.get("did_drag")):
                    drag_distance = float(np.linalg.norm(pointer_xy - pressed_xy))
                    if drag_distance < ANNOTATION_DRAG_ACTIVATION_PX:
                        self._update_preview_cursor(event)
                        return
                    self._drag_annotation_state["did_drag"] = True
                dragged_keypoint_name = str(
                    self._drag_annotation_state.get("existing_keypoint_name")
                    or self._drag_annotation_state.get("selected_keypoint_name")
                    or ""
                )
                snapped_xy = self._snap_annotation_xy(
                    camera_name=str(self._drag_annotation_state["camera_name"]),
                    frame_number=int(self._drag_annotation_state["frame_number"]),
                    keypoint_name=dragged_keypoint_name,
                    pointer_xy=pointer_xy,
                )
                set_annotation_point(
                    self.annotation_payload,
                    camera_name=str(self._drag_annotation_state["camera_name"]),
                    frame_number=int(self._drag_annotation_state["frame_number"]),
                    keypoint_name=dragged_keypoint_name,
                    xy=snapped_xy,
                )
                self.refresh_preview()
                return
        if self._pan_state is None or event.inaxes is None or event.xdata is None or event.ydata is None:
            self._update_preview_cursor(event)
            return
        if event.inaxes is not self._pan_state.get("axes"):
            self._update_preview_cursor(event)
            return
        x_press = float(self._pan_state["xdata"])
        y_press = float(self._pan_state["ydata"])
        xlim0 = tuple(self._pan_state["xlim"])
        ylim0 = tuple(self._pan_state["ylim"])
        dx = float(event.xdata) - x_press
        dy = float(event.ydata) - y_press
        event.inaxes.set_xlim(xlim0[0] - dx, xlim0[1] - dx)
        event.inaxes.set_ylim(ylim0[0] - dy, ylim0[1] - dy)
        camera_name = self._axis_to_camera.get(event.inaxes)
        if camera_name is not None:
            self._annotation_view_limits[str(camera_name)] = (
                tuple(float(value) for value in event.inaxes.get_xlim()),
                tuple(float(value) for value in event.inaxes.get_ylim()),
            )
        self.preview_canvas.draw_idle()
        self._update_preview_cursor(None)

    def on_preview_release(self, event) -> None:
        if self._drag_annotation_state is not None:
            state = dict(self._drag_annotation_state)
            self._drag_annotation_state = None
            camera_name = str(state.get("camera_name", ""))
            frame_number = int(state.get("frame_number", self.current_frame_number()))
            selected_keypoint_name = str(state.get("selected_keypoint_name", self.selected_keypoint_name()))
            did_drag = bool(state.get("did_drag"))
            if (
                not did_drag
                and event is not None
                and event.inaxes is not None
                and event.xdata is not None
                and event.ydata is not None
            ):
                release_camera_name = self._axis_to_camera.get(event.inaxes)
                if release_camera_name == camera_name:
                    snapped_xy = self._snap_annotation_xy(
                        camera_name=camera_name,
                        frame_number=frame_number,
                        keypoint_name=selected_keypoint_name,
                        pointer_xy=np.array([float(event.xdata), float(event.ydata)], dtype=float),
                    )
                    set_annotation_point(
                        self.annotation_payload,
                        camera_name=camera_name,
                        frame_number=frame_number,
                        keypoint_name=selected_keypoint_name,
                        xy=snapped_xy,
                    )
                    if self.advance_marker_var.get():
                        self._advance_to_next_keypoint()
            updated_keypoint = str(state.get("existing_keypoint_name")) if did_drag else selected_keypoint_name
            self._clear_kinematic_assist_preview()
            if bool(getattr(self, "kinematic_assist_var", None) and self.kinematic_assist_var.get()):
                try:
                    self._estimate_kinematic_q(keypoint_name=updated_keypoint)
                except Exception as exc:
                    self._clear_kinematic_assist_preview()
                    self.kinematic_status_var.set(f"Kinematic assist update skipped: {exc}")
            self.save_annotations()
            self.refresh_preview()
        self._pan_state = None

    def refresh_preview(self) -> None:
        if self.pose_data is None or self.calibrations is None or len(self.pose_data.frames) == 0:
            return
        if not hasattr(self, "_annotation_view_limits"):
            self._annotation_view_limits = {}
        self._store_current_annotation_view_limits()
        all_camera_names = [str(name) for name in self.pose_data.camera_names]
        camera_names = self.selected_annotation_camera_names()
        if not camera_names:
            camera_names = list(all_camera_names)
        support_camera_names = list(all_camera_names)
        frame_idx = max(0, min(len(self.pose_data.frames) - 1, int(round(self.frame_var.get()))))
        self._current_frame_idx = frame_idx
        self.frame_var.set(frame_idx)
        frame_number = int(self.pose_data.frames[frame_idx])
        mode = self._frame_filter_mode()
        filtered_indices = self._filtered_annotation_frame_local_indices()
        self.frame_label.configure(
            text=annotation_frame_label_text(
                frame_idx=frame_idx,
                frame_number=frame_number,
                mode=mode,
                filtered_indices=filtered_indices,
                mode_labels=ANNOTATION_FRAME_FILTER_OPTIONS,
            )
        )
        jump_context_var = getattr(self, "jump_context_var", None)
        if jump_context_var is not None:
            jump_context_var.set(self._annotation_jump_context())
        crop_limits = self._ensure_crop_limits(camera_names) if self.crop_var.get() else {}
        current_marker = self.selected_keypoint_name()
        current_color = annotation_marker_color(current_marker)
        if (
            bool(getattr(self, "kinematic_assist_var", None) and self.kinematic_assist_var.get())
            and self.kinematic_projected_points is None
        ):
            biomod_path = self._selected_kinematic_biomod_path()
            if biomod_path is not None and biomod_path.exists():
                try:
                    import biorbd

                    model = biorbd.Model(str(biomod_path))
                    cached_state, source_frame, is_exact = self._selected_or_nearest_kinematic_state_info(
                        frame_number, model
                    )
                    if cached_state is None:
                        raise ValueError("No cached state")
                    self._set_kinematic_preview_from_q(biomod_path, camera_names, cached_state[: model.nbQ()])
                    self.kinematic_state_current = np.asarray(cached_state, dtype=float)
                    if is_exact:
                        self.kinematic_status_var.set(f"Using saved q for frame {frame_number}.")
                    elif source_frame is not None:
                        self.kinematic_status_var.set(
                            f"Using nearest saved q from frame {source_frame} for frame {frame_number}."
                        )
                except Exception:
                    self._clear_kinematic_assist_preview()
        self.preview_figure.clear()
        self._cursor_artists = {}
        self._annotation_hover_entries = {}
        nrows, ncols = camera_layout(len(camera_names))
        axes = np.atleast_1d(self.preview_figure.subplots(nrows, ncols)).ravel()
        self._axis_to_camera = {}
        images_root = self._current_images_root()

        for ax_idx, ax in enumerate(axes):
            if ax_idx >= len(camera_names):
                ax.axis("off")
                continue
            camera_name = camera_names[ax_idx]
            self._axis_to_camera[ax] = camera_name
            width, height = self.calibrations[camera_name].image_size
            background_image = (
                load_camera_background_image(
                    images_root,
                    camera_name,
                    frame_number,
                    image_reader=plt.imread,
                    brightness=float(self.image_brightness_var.get()),
                    contrast=float(self.image_contrast_var.get()),
                )
                if self.show_images_var.get()
                else None
            )
            x_limits, y_limits = self._annotation_view_limits_for_camera(camera_name)
            reference_projected_points = None
            reference_projected_label = None
            reference_projected_color = "#6c5ce7"
            if bool(
                getattr(self, "show_reference_reprojection_var", None) and self.show_reference_reprojection_var.get()
            ):
                (
                    reference_projected_points,
                    reference_projected_label,
                    reference_projected_color,
                ) = self._reference_projected_points(camera_name, frame_number)
            self._annotation_hover_entries[ax] = render_annotation_camera_view(
                ax,
                ax_idx=ax_idx,
                camera_name=camera_name,
                frame_idx=frame_idx,
                frame_number=frame_number,
                width=width,
                height=height,
                crop_mode=("pose" if self.crop_var.get() else "full"),
                crop_limits=crop_limits,
                background_image=background_image,
                current_marker=current_marker,
                current_color=current_color,
                keypoint_names=COCO17,
                kp_index=KP_INDEX,
                annotation_xy_getter=self._annotation_xy,
                pending_reprojection_points=self._pending_reprojection_points,
                marker_color_getter=annotation_marker_color,
                marker_shape_getter=annotation_marker_shape,
                draw_background_fn=draw_2d_background_image,
                apply_axis_limits_fn=apply_2d_axis_limits,
                hide_axes_fn=hide_2d_axes,
                draw_skeleton_fn=draw_skeleton_2d,
                draw_upper_back_fn=draw_upper_back_overlay_2d,
                kinematic_projected_points=(
                    self.kinematic_projected_points
                    if bool(getattr(self, "kinematic_assist_var", None) and self.kinematic_assist_var.get())
                    else None
                ),
                kinematic_segmented_back_projected=self.kinematic_segmented_back_projected,
                reference_projected_points=reference_projected_points,
                reference_projected_label=reference_projected_label,
                reference_projected_color=reference_projected_color,
                motion_prior_enabled=bool(self.show_motion_prior_var.get()),
                motion_prior_diameter=float(self.motion_prior_diameter.get()),
                motion_prior_center_fn=annotation_motion_prior_center,
            )
            if x_limits is not None and y_limits is not None:
                ax.set_xlim(*x_limits)
                ax.set_ylim(*y_limits)

            other_camera_names: list[str] = []
            other_points: list[np.ndarray] = []
            for other_camera_name in support_camera_names:
                if other_camera_name == camera_name:
                    continue
                other_xy = self._annotation_xy(other_camera_name, frame_number, current_marker)
                if other_xy is None:
                    continue
                other_camera_names.append(other_camera_name)
                other_points.append(other_xy)
                if self.show_epipolar_var.get():
                    line = annotation_epipolar_guides(self.calibrations, other_camera_name, camera_name, other_xy)
                    if line is not None:
                        x0, x1 = ax.get_xlim()
                        y0, y1 = ax.get_ylim()
                        xs = np.array([x0, x1], dtype=float)
                        if abs(line[1]) > 1e-8:
                            ys = -(line[0] * xs + line[2]) / line[1]
                            ax.plot(xs, ys, color=current_color, linewidth=1.0, alpha=0.6, linestyle=":")
                        elif abs(line[0]) > 1e-8:
                            x_const = -line[2] / line[0]
                            ax.plot(
                                [x_const, x_const],
                                [y0, y1],
                                color=current_color,
                                linewidth=1.0,
                                alpha=0.6,
                                linestyle=":",
                            )
            if self.show_triangulated_hint_var.get():
                triangulated_hint = annotation_triangulated_reprojection(
                    self.calibrations,
                    target_camera_name=camera_name,
                    source_camera_names=other_camera_names,
                    source_points_2d=other_points,
                )
                if triangulated_hint is not None:
                    ax.scatter(
                        [triangulated_hint[0]],
                        [triangulated_hint[1]],
                        s=90,
                        facecolors="none",
                        edgecolors=[current_color],
                        marker="o",
                        linewidths=1.9,
                        zorder=6,
                    )
        self.preview_figure.subplots_adjust(left=0.03, right=0.995, bottom=0.035, top=0.96, wspace=0.05, hspace=0.12)
        self.preview_canvas.draw_idle()


class FiguresTab(CommandTab):
    def __init__(self, master):
        super().__init__(master, "Figures")
        form = ttk.LabelFrame(self.main, text="analysis/plot_kinematic_comparison.py")
        form.pack(fill=tk.X, pady=(0, 8), before=self.output)

        self.input_dir = LabeledEntry(form, "Input dir", "output/vitpose_full", browse=True, directory=True)
        self.input_dir.pack(fill=tk.X, padx=8, pady=4)
        self.output_dir = LabeledEntry(form, "Output dir", "output/vitpose_full/figures", browse=True, directory=True)
        self.output_dir.pack(fill=tk.X, padx=8, pady=4)

        row = ttk.Frame(form)
        row.pack(fill=tk.X, padx=8, pady=4)
        self.fps = LabeledEntry(row, "FPS", "120")
        self.fps.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.top_dofs = LabeledEntry(row, "Top DoFs", "10")
        self.top_dofs.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        variant_label = ttk.Label(row, text="EKF 2D variant", width=16)
        variant_label.pack(side=tk.LEFT)
        self.variant = tk.StringVar(value="acc")
        variant_box = ttk.Combobox(
            row, textvariable=self.variant, values=["acc", "flip_acc", "both"], width=12, state="readonly"
        )
        variant_box.pack(side=tk.LEFT)

        self.extra = LabeledEntry(form, "Extra args", "")
        self.extra.pack(fill=tk.X, padx=8, pady=4)
        self.input_dir.set_tooltip("Dossier contenant les sorties numeriques a comparer.")
        self.output_dir.set_tooltip("Dossier de destination des figures exportees.")
        self.fps.set_tooltip("Frequence d'echantillonnage utilisee pour l'axe temporel.")
        self.top_dofs.set_tooltip("Nombre de DoF mis en avant dans les figures de comparaison.")
        attach_tooltip(variant_label, "Choisit quelle variante EKF 2D comparer a l'EKF 3D.")
        attach_tooltip(variant_box, "Choisit quelle variante EKF 2D comparer a l'EKF 3D.")
        self.extra.set_tooltip("Arguments CLI additionnels passes a analysis/plot_kinematic_comparison.py.")

    def build_command(self) -> list[str]:
        cmd = [
            sys.executable,
            "analysis/plot_kinematic_comparison.py",
            "--input-dir",
            self.input_dir.get(),
            "--output-dir",
            self.output_dir.get(),
            "--fps",
            self.fps.get(),
            "--top-dofs",
            self.top_dofs.get(),
            "--ekf-2d-variant",
            self.variant.get(),
        ]
        cmd.extend(self.parse_extra_args(self.extra.get()))
        return cmd


class AnalysisTab(CommandTab):
    def __init__(self, master):
        super().__init__(master, "Analyses")
        self.script = tk.StringVar(value="analysis/analyze_trampoline_jumps.py")

        form = ttk.LabelFrame(self.main, text="Scripts d'analyse")
        form.pack(fill=tk.X, pady=(0, 8), before=self.output)

        top = ttk.Frame(form)
        top.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(top, text="Script", width=18).pack(side=tk.LEFT, padx=(0, 6))
        script_box = ttk.Combobox(
            top,
            textvariable=self.script,
            values=[
                "analysis/analyze_trampoline_jumps.py",
                "analysis/plot_triangulation_view_usage.py",
                "analysis/plot_triangulated_marker_trajectories.py",
                "analysis/plot_3d_posture_snapshots.py",
            ],
            state="readonly",
            width=42,
        )
        script_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
        script_box.bind("<<ComboboxSelected>>", lambda _event: self._sync_defaults())

        self.entry_a = LabeledEntry(form, "Input A", "")
        self.entry_a.pack(fill=tk.X, padx=8, pady=4)
        self.entry_b = LabeledEntry(form, "Input B", "")
        self.entry_b.pack(fill=tk.X, padx=8, pady=4)
        self.entry_c = LabeledEntry(form, "Output", "")
        self.entry_c.pack(fill=tk.X, padx=8, pady=4)

        row = ttk.Frame(form)
        row.pack(fill=tk.X, padx=8, pady=4)
        self.opt_1 = LabeledEntry(row, "FPS", "120")
        self.opt_1.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.opt_2 = LabeledEntry(row, "Option 2", "")
        self.opt_2.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.opt_3 = LabeledEntry(row, "Option 3", "")
        self.opt_3.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.interactive_var = tk.BooleanVar(value=False)
        interactive_check = ttk.Checkbutton(form, text="interactive", variable=self.interactive_var)
        interactive_check.pack(anchor="w", padx=8, pady=4)

        self.extra = LabeledEntry(form, "Extra args", "")
        self.extra.pack(fill=tk.X, padx=8, pady=4)
        attach_tooltip(script_box, "Choisit quel script d'analyse lancer dans cet onglet.")
        self.entry_a.set_tooltip("Premier chemin/argument principal du script selectionne.")
        self.entry_b.set_tooltip("Deuxieme chemin/argument principal du script selectionne.")
        self.entry_c.set_tooltip("Chemin de sortie principal du script selectionne.")
        self.opt_1.set_tooltip("Premier parametre optionnel du script selectionne.")
        self.opt_2.set_tooltip("Deuxieme parametre optionnel du script selectionne.")
        self.opt_3.set_tooltip("Troisieme parametre optionnel du script selectionne.")
        attach_tooltip(
            interactive_check, "Si coche, le script d'analyse est lance en mode interactif quand il le supporte."
        )
        self.extra.set_tooltip("Arguments CLI additionnels passes au script d'analyse.")
        self._sync_defaults()

    def _sync_defaults(self) -> None:
        script = self.script.get()
        self.interactive_var.set(False)
        if script == "analysis/analyze_trampoline_jumps.py":
            self.entry_a.var.set("output/vitpose_full/ekf_states.npz")
            self.entry_b.var.set("output/vitpose_full/jump_segmentation.png")
            self.entry_c.var.set("output/vitpose_full/jump_rotations.png")
            self.opt_1.var.set("120")
            self.opt_2.var.set("0.20")
            self.opt_3.var.set("0.15")
        elif script == "analysis/plot_triangulation_view_usage.py":
            self.entry_a.var.set("output/vitpose_full/triangulation_pose2sim_like.npz")
            self.entry_b.var.set("")
            self.entry_c.var.set("output/vitpose_full/triangulation_view_usage.png")
            self.opt_1.var.set("120")
            self.opt_2.var.set("used")
            self.opt_3.var.set("")
        elif script == "analysis/plot_triangulated_marker_trajectories.py":
            self.entry_a.var.set("output/vitpose_full/triangulation_pose2sim_like.npz")
            self.entry_b.var.set("output/vitpose_full/summary.json")
            self.entry_c.var.set("output/vitpose_full/triangulated_marker_trajectories.png")
            self.opt_1.var.set("120")
            self.opt_2.var.set("1.5")
            self.opt_3.var.set("2")
        else:
            self.entry_a.var.set("output/vitpose_full/triangulation_pose2sim_like.npz")
            self.entry_b.var.set("inputs/calibration/Calib.toml")
            self.entry_c.var.set("output/vitpose_full/posture_snapshots_3d.png")
            self.opt_1.var.set("120")
            self.opt_2.var.set("7")
            self.opt_3.var.set("output/vitpose_full/first_frame_root_coordinate_system.png")

    def build_command(self) -> list[str]:
        script = self.script.get()
        cmd = [sys.executable, script]
        if script == "analysis/analyze_trampoline_jumps.py":
            cmd.extend(
                [
                    "--states",
                    self.entry_a.get(),
                    "--fps",
                    self.opt_1.get(),
                    "--figure",
                    self.entry_b.get(),
                    "--rotation-figure",
                    self.entry_c.get(),
                    "--height-threshold-range-ratio",
                    self.opt_2.get(),
                    "--smoothing-window-s",
                    self.opt_3.get(),
                ]
            )
        elif script == "analysis/plot_triangulation_view_usage.py":
            cmd.extend(
                [
                    "--triangulation",
                    self.entry_a.get(),
                    "--output",
                    self.entry_c.get(),
                    "--fps",
                    self.opt_1.get(),
                    "--detail-mode",
                    self.opt_2.get() or "used",
                ]
            )
        elif script == "analysis/plot_triangulated_marker_trajectories.py":
            cmd.extend(
                [
                    "--triangulation",
                    self.entry_a.get(),
                    "--summary",
                    self.entry_b.get(),
                    "--output",
                    self.entry_c.get(),
                    "--fps",
                    self.opt_1.get(),
                    "--flight-height-threshold-m",
                    self.opt_2.get(),
                    "--flight-min-consecutive-frames",
                    self.opt_3.get(),
                ]
            )
            if self.interactive_var.get():
                cmd.append("--interactive")
        else:
            cmd.extend(
                [
                    "--triangulation",
                    self.entry_a.get(),
                    "--calib",
                    self.entry_b.get(),
                    "--output",
                    self.entry_c.get(),
                    "--fps",
                    self.opt_1.get(),
                    "--n-postures",
                    self.opt_2.get(),
                    "--first-frame-output",
                    self.opt_3.get(),
                ]
            )
        cmd.extend(self.parse_extra_args(self.extra.get()))
        return cmd


class DataExplorer2DTab(ttk.Frame):
    def __init__(self, master, state: SharedAppState):
        super().__init__(master)
        self.state = state
        self.pose_data = None
        self.calibrations = None
        self.flip_masks: dict[str, np.ndarray] = {}
        self.flip_diagnostics: dict[str, dict[str, object]] = {}
        self.trc_status_var = tk.StringVar(value="")
        self.dataset_summary_var = tk.StringVar(value="")
        self.flip_status_var = tk.StringVar(value="")

        controls = ttk.LabelFrame(self, text="Sources partagees + exploration 2D")
        controls.pack(fill=tk.X, padx=10, pady=10)

        self.calib = LabeledEntry(
            controls,
            "Calib",
            browse=True,
            filetypes=(("TOML calibration", "*.toml"), ("All files", "*.*")),
            browse_initialdir="inputs/calibration",
        )
        self.calib.var = state.calib_var
        self.calib.entry_widget.configure(textvariable=self.calib.var)
        self.calib.pack(fill=tk.X, padx=8, pady=4)

        row_sources = ttk.Frame(controls)
        row_sources.pack(fill=tk.X, padx=8, pady=4)
        self.keypoints = LabeledEntry(
            row_sources,
            "Keypoints",
            browse=True,
            label_width=10,
            entry_width=34,
            filetypes=(("2D keypoints JSON", "*_keypoints.json"), ("JSON files", "*.json"), ("All files", "*.*")),
            browse_initialdir="inputs/keypoints",
        )
        self.keypoints.var = state.keypoints_var
        self.keypoints.entry_widget.configure(textvariable=self.keypoints.var)
        self.keypoints.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.pose2sim_trc = LabeledEntry(
            row_sources,
            "TRC file",
            browse=True,
            label_width=12,
            entry_width=32,
            filetypes=(("TRC files", "*.trc"), ("All files", "*.*")),
            browse_initialdir="inputs/trc",
        )
        self.pose2sim_trc.var = state.pose2sim_trc_var
        self.pose2sim_trc.entry_widget.configure(textvariable=self.pose2sim_trc.var)
        self.pose2sim_trc.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.trc_status_label = ttk.Label(
            controls, textvariable=self.trc_status_var, foreground="#8a5a00", justify=tk.LEFT
        )
        self.trc_status_label.pack(fill=tk.X, padx=8, pady=(0, 4))

        row_shared = ttk.Frame(controls)
        row_shared.pack(fill=tk.X, padx=8, pady=4)
        self.fps = LabeledEntry(row_shared, "FPS", label_width=4, entry_width=6)
        self.fps.var = state.fps_var
        self.fps.entry_widget.configure(textvariable=self.fps.var)
        self.fps.pack(side=tk.LEFT, padx=(0, 6))
        self.workers = LabeledEntry(row_shared, "Workers", label_width=8, entry_width=6)
        self.workers.var = state.workers_var
        self.workers.entry_widget.configure(textvariable=self.workers.var)
        self.workers.pack(side=tk.LEFT)
        self.initial_rotation_correction_var = state.initial_rotation_correction_var
        self.root_rotfix_check = ttk.Checkbutton(
            row_shared, text="Root rot-fix", variable=self.initial_rotation_correction_var
        )
        self.root_rotfix_check.pack(side=tk.LEFT, padx=(8, 0))
        calib_correction_label = ttk.Label(row_shared, text="Calib 2D corr", width=12)
        calib_correction_label.pack(side=tk.LEFT, padx=(8, 0))
        self.calibration_correction = state.calibration_correction_var
        self.calibration_correction_box = ttk.Combobox(
            row_shared,
            textvariable=self.calibration_correction,
            values=[
                "none",
                "flip_epipolar",
                "flip_epipolar_fast",
                "flip_epipolar_viterbi",
                "flip_epipolar_fast_viterbi",
                "flip_triangulation",
            ],
            width=18,
            state="readonly",
        )
        self.calibration_correction_box.pack(side=tk.LEFT, padx=(0, 6))
        self.selected_cameras_label_var = tk.StringVar(value="Cameras: all")
        ttk.Label(row_shared, textvariable=self.selected_cameras_label_var, foreground="#4f5b66").pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Label(controls, textvariable=self.dataset_summary_var, foreground="#4f5b66", justify=tk.LEFT).pack(
            fill=tk.X, padx=8, pady=(0, 4)
        )

        row_display = ttk.Frame(controls)
        component_label = ttk.Label(row_display, text="Composante", width=12)
        component_label.pack(side=tk.LEFT)
        self.component = tk.StringVar(value="y")
        component_box = ttk.Combobox(
            row_display, textvariable=self.component, values=["x", "y"], width=6, state="readonly"
        )
        component_box.pack(side=tk.LEFT, padx=(0, 8))
        view_mode_label = ttk.Label(row_display, text="2D mode", width=10)
        view_mode_label.pack(side=tk.LEFT)
        self.view_mode = state.pose_data_mode_var
        self.view_mode_menu = ttk.Combobox(
            row_display,
            textvariable=self.view_mode,
            values=["raw", "cleaned"],
            width=10,
            state="readonly",
        )
        self.view_mode_menu.pack(side=tk.LEFT, padx=(0, 8))
        flip_mode_label = ttk.Label(row_display, text="Correction L/R", width=14)
        flip_mode_label.pack(side=tk.LEFT)
        self.flip_mode = tk.StringVar(value="none")
        self.flip_mode_menu = ttk.Combobox(
            row_display,
            textvariable=self.flip_mode,
            values=["none", "epipolar", "epipolar_fast", "epipolar_viterbi", "epipolar_fast_viterbi", "triangulation"],
            width=22,
            state="readonly",
        )
        self.flip_mode_menu.pack(side=tk.LEFT, padx=(0, 8))

        row_clean = ttk.Frame(controls)
        row_clean.pack(fill=tk.X, padx=8, pady=4)
        self.pose_filter_window = LabeledEntry(row_clean, "Filter window", "9", label_width=10, entry_width=4)
        self.pose_filter_window.var = state.pose_filter_window_var
        self.pose_filter_window.entry_widget.configure(textvariable=self.pose_filter_window.var)
        self.pose_filter_window.pack(side=tk.LEFT, padx=(0, 6))
        self.pose_outlier_ratio = LabeledEntry(row_clean, "Outlier ratio", "0.10", label_width=10, entry_width=5)
        self.pose_outlier_ratio.var = state.pose_outlier_ratio_var
        self.pose_outlier_ratio.entry_widget.configure(textvariable=self.pose_outlier_ratio.var)
        self.pose_outlier_ratio.pack(side=tk.LEFT, padx=(0, 6))
        self.pose_p_low = LabeledEntry(row_clean, "P_low", "5", label_width=5, entry_width=4, label_padx=(0, 2))
        self.pose_p_low.var = state.pose_p_low_var
        self.pose_p_low.entry_widget.configure(textvariable=self.pose_p_low.var)
        self.pose_p_low.pack(side=tk.LEFT, padx=(0, 6))
        self.pose_p_high = LabeledEntry(row_clean, "P_high", "95", label_width=6, entry_width=4, label_padx=(0, 2))
        self.pose_p_high.var = state.pose_p_high_var
        self.pose_p_high.entry_widget.configure(textvariable=self.pose_p_high.var)
        self.pose_p_high.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(row_clean, text="Load 2D data", command=self.load_data).pack(side=tk.LEFT)
        ttk.Button(row_clean, text="Refresh", command=self.reload_data).pack(side=tk.LEFT, padx=(8, 0))
        row_flip = ttk.Frame(controls)
        row_flip.pack(fill=tk.X, padx=8, pady=4)
        self.flip_restrict_var = state.flip_restrict_to_outliers_var
        flip_restrict_check = ttk.Checkbutton(
            row_flip, text="Flip: test only outliers", variable=self.flip_restrict_var
        )
        flip_restrict_check.pack(side=tk.LEFT, padx=(0, 8))
        self.flip_outlier_percentile = LabeledEntry(
            row_flip, "Flip Q pct", str(DEFAULT_FLIP_OUTLIER_PERCENTILE), label_width=9, entry_width=4
        )
        self.flip_outlier_percentile.var = state.flip_outlier_percentile_var
        self.flip_outlier_percentile.entry_widget.configure(textvariable=self.flip_outlier_percentile.var)
        self.flip_outlier_percentile.pack(side=tk.LEFT, padx=(0, 6))
        self.flip_outlier_floor = LabeledEntry(
            row_flip, "Flip floor", str(DEFAULT_FLIP_OUTLIER_FLOOR_PX), label_width=9, entry_width=4
        )
        self.flip_outlier_floor.var = state.flip_outlier_floor_px_var
        self.flip_outlier_floor.entry_widget.configure(textvariable=self.flip_outlier_floor.var)
        self.flip_outlier_floor.pack(side=tk.LEFT, padx=(0, 6))
        self.flip_improvement_ratio = LabeledEntry(
            row_flip, "Flip ratio", str(DEFAULT_FLIP_IMPROVEMENT_RATIO), label_width=9, entry_width=5
        )
        self.flip_improvement_ratio.var = state.flip_improvement_ratio_var
        self.flip_improvement_ratio.entry_widget.configure(textvariable=self.flip_improvement_ratio.var)
        self.flip_improvement_ratio.pack(side=tk.LEFT, padx=(0, 6))
        self.flip_min_gain = LabeledEntry(
            row_flip, "Flip gain", str(DEFAULT_FLIP_MIN_GAIN_PX), label_width=8, entry_width=4
        )
        self.flip_min_gain.var = state.flip_min_gain_px_var
        self.flip_min_gain.entry_widget.configure(textvariable=self.flip_min_gain.var)
        self.flip_min_gain.pack(side=tk.LEFT, padx=(0, 6))

        row_flip_timing = ttk.Frame(controls)
        row_flip_timing.pack(fill=tk.X, padx=8, pady=4)
        self.flip_min_cameras = LabeledEntry(
            row_flip_timing, "Min cams", str(DEFAULT_FLIP_MIN_OTHER_CAMERAS), label_width=8, entry_width=4
        )
        self.flip_min_cameras.var = state.flip_min_other_cameras_var
        self.flip_min_cameras.entry_widget.configure(textvariable=self.flip_min_cameras.var)
        self.flip_min_cameras.pack(side=tk.LEFT, padx=(0, 10))
        triang_flip_label = ttk.Label(row_flip_timing, text="Triang flip", width=9)
        triang_flip_label.pack(side=tk.LEFT)
        self.triang_flip_method = tk.StringVar(value="once")
        self.triang_flip_method_box = ttk.Combobox(
            row_flip_timing,
            textvariable=self.triang_flip_method,
            values=["once", "greedy", "exhaustive"],
            width=10,
            state="readonly",
        )
        self.triang_flip_method_box.pack(side=tk.LEFT, padx=(0, 10))
        self.flip_temporal_weight = LabeledEntry(
            row_flip_timing, "Temp w", str(DEFAULT_FLIP_TEMPORAL_WEIGHT), label_width=7, entry_width=4
        )
        self.flip_temporal_weight.var = state.flip_temporal_weight_var
        self.flip_temporal_weight.entry_widget.configure(textvariable=self.flip_temporal_weight.var)
        self.flip_temporal_weight.pack(side=tk.LEFT, padx=(0, 6))
        self.flip_temporal_tau = LabeledEntry(
            row_flip_timing, "Temp tau", str(DEFAULT_FLIP_TEMPORAL_TAU_PX), label_width=8, entry_width=4
        )
        self.flip_temporal_tau.var = state.flip_temporal_tau_px_var
        self.flip_temporal_tau.entry_widget.configure(textvariable=self.flip_temporal_tau.var)
        self.flip_temporal_tau.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(row_flip_timing, textvariable=self.flip_status_var, foreground="#4f5b66", justify=tk.LEFT).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

        row_display.pack(fill=tk.X, padx=8, pady=4)

        content = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        keypoint_box = ttk.LabelFrame(content, text="Keypoints")
        figure_box = ttk.LabelFrame(content, text="Courbes temporelles par camera")
        content.add(keypoint_box, weight=1)
        content.add(figure_box, weight=4)

        keypoint_actions = ttk.Frame(keypoint_box)
        keypoint_actions.pack(fill=tk.X, padx=8, pady=(8, 0))
        ttk.Button(keypoint_actions, text="All", command=self.select_all_keypoints).pack(side=tk.LEFT)
        ttk.Button(keypoint_actions, text="None", command=self.clear_keypoints).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(keypoint_actions, text="Body only", command=self.select_body_keypoints).pack(
            side=tk.LEFT, padx=(6, 0)
        )

        self.keypoint_list = tk.Listbox(keypoint_box, selectmode=tk.MULTIPLE, exportselection=False, height=18)
        for name in COCO17:
            self.keypoint_list.insert(tk.END, name)
        for idx in range(len(COCO17)):
            self.keypoint_list.selection_set(idx)
        self.keypoint_list.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        bind_extended_listbox_shortcuts(self.keypoint_list)

        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, master=figure_box)
        self.toolbar = NavigationToolbar2Tk(self.canvas, figure_box, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill=tk.X, padx=8, pady=(4, 0))
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.calib.set_tooltip("Fichier de calibration partagé par tous les onglets.")
        self.keypoints.set_tooltip("Fichier JSON des détections 2D. Son nom détermine aussi le nom du dataset.")
        self.pose2sim_trc.set_tooltip("TRC file used for direct 3D import. It is auto-detected from the 2D JSON.")
        self.fps.set_tooltip("FPS partage pour les derives temporelles et les animations.")
        self.workers.set_tooltip("Nombre de workers partage pour les rendus et les calculs paralleles.")
        attach_tooltip(
            self.root_rotfix_check,
            "Si coché, la racine est réalignée en lacet autour de Z à partir de l'axe médio-latéral du tronc à t0. L'angle est arrondi au multiple de pi/2 le plus proche, puis partagé par le modèle, les reconstructions et les analyses.",
        )
        attach_tooltip(
            calib_correction_label,
            "Choisit quelle variante 2D sera utilisée par les outils de calibration/caméras: aucune correction, flip local épipolaire, flip local épipolaire rapide, variantes Viterbi explicites, ou flip détecté par triangulation.",
        )
        attach_tooltip(
            self.calibration_correction_box,
            "Choisit quelle variante 2D sera utilisée par les outils de calibration/caméras: aucune correction, flip local épipolaire, flip local épipolaire rapide, variantes Viterbi explicites, ou flip détecté par triangulation.",
        )
        attach_tooltip(component_label, "Composante 2D affichée sur les courbes temporelles.")
        attach_tooltip(component_box, "Composante 2D affichée sur les courbes temporelles.")
        attach_tooltip(view_mode_label, "Traitement affiché: brut ou nettoyé après rejet des outliers.")
        attach_tooltip(self.view_mode_menu, "Traitement affiché: brut ou nettoyé après rejet des outliers.")
        attach_tooltip(
            flip_mode_label,
            "Applique visuellement une correction gauche/droite basée sur le diagnostic choisi. Les variantes sans suffixe utilisent une décision locale frame par frame; les variantes *_viterbi appliquent un lissage explicite par Viterbi.",
        )
        attach_tooltip(
            self.flip_mode_menu,
            "Applique visuellement une correction gauche/droite basée sur le diagnostic choisi. Les variantes sans suffixe utilisent une décision locale frame par frame; les variantes *_viterbi appliquent un lissage explicite par Viterbi.",
        )
        self.pose_filter_window.set_tooltip("Fenêtre du lissage utilisé pour construire la référence filtrée 2D.")
        self.pose_outlier_ratio.set_tooltip("Seuil de rejet des points 2D trop éloignés de la référence filtrée.")
        self.pose_p_low.set_tooltip("Percentile bas utilisé pour définir l'amplitude robuste du mouvement 2D.")
        self.pose_p_high.set_tooltip("Percentile haut utilisé pour définir l'amplitude robuste du mouvement 2D.")
        attach_tooltip(
            flip_restrict_check,
            "Si coché, le swap gauche/droite n'est testé que pour les camera-frames dont le coût nominal est déjà un outlier.",
        )
        self.flip_outlier_percentile.set_tooltip(
            "Percentile utilisé pour définir les outliers de coût nominal par caméra avant de tester un flip L/R."
        )
        self.flip_outlier_floor.set_tooltip(
            "Plancher en pixels pour tester un flip L/R: seuil = max(plancher, percentile)."
        )
        self.flip_improvement_ratio.set_tooltip(
            "Le flip est accepté si le coût swappé devient inférieur à ce ratio du coût nominal."
        )
        self.flip_min_gain.set_tooltip(
            "Gain minimal en pixels requis entre coût nominal et coût swappé pour valider un flip L/R."
        )
        self.flip_min_cameras.set_tooltip(
            "Nombre minimal d'autres caméras valides pour évaluer un flip L/R en mode triangulation."
        )
        attach_tooltip(
            triang_flip_label,
            "Choisit la variante de triangulation utilisée pour le diagnostic flip L/R: once, greedy, ou exhaustive.",
        )
        attach_tooltip(
            self.triang_flip_method_box,
            "once: une triangulation pondérée par keypoint. greedy: suppression gloutonne des pires vues. exhaustive: test des combinaisons de vues le plus robuste mais plus coûteux.",
        )
        self.flip_temporal_weight.set_tooltip(
            "Poids du coût temporel par caméra dans le coût combiné de décision du flip. Mettre 0 pour rester purement géométrique."
        )
        self.flip_temporal_tau.set_tooltip(
            "Echelle en pixels du coût temporel 2D utilisée pour normaliser sa contribution avant combinaison avec le coût épipolaire ou triangulation."
        )
        attach_tooltip(
            keypoint_actions,
            "Quick keypoint-selection presets for the 2D explorer. 'Body only' keeps trunk and limb keypoints while removing eyes and ears.",
        )
        self.component.trace_add("write", lambda *_args: self.refresh_plot())
        self.view_mode.trace_add("write", lambda *_args: self.refresh_plot())
        self.flip_mode.trace_add("write", lambda *_args: self.refresh_plot())
        self.keypoint_list.bind("<<ListboxSelect>>", lambda _event: self.refresh_plot())
        self.state.keypoints_var.trace_add("write", lambda *_args: self.on_keypoints_changed())
        self.state.output_root_var.trace_add("write", lambda *_args: self.update_dataset_summary())
        self.state.flip_restrict_to_outliers_var.trace_add("write", lambda *_args: self.on_flip_settings_changed())
        self.state.flip_outlier_percentile_var.trace_add("write", lambda *_args: self.on_flip_settings_changed())
        self.state.flip_outlier_floor_px_var.trace_add("write", lambda *_args: self.on_flip_settings_changed())
        self.state.flip_improvement_ratio_var.trace_add("write", lambda *_args: self.on_flip_settings_changed())
        self.state.flip_min_gain_px_var.trace_add("write", lambda *_args: self.on_flip_settings_changed())
        self.state.flip_min_other_cameras_var.trace_add("write", lambda *_args: self.on_flip_settings_changed())
        self.state.flip_temporal_weight_var.trace_add("write", lambda *_args: self.on_flip_settings_changed())
        self.state.flip_temporal_tau_px_var.trace_add("write", lambda *_args: self.on_flip_settings_changed())
        self.triang_flip_method.trace_add("write", lambda *_args: self.on_flip_settings_changed())
        self.state.initial_rotation_correction_var.trace_add(
            "write", lambda *_args: synchronize_profiles_initial_rotation_correction(self.state)
        )
        self.state.selected_camera_names_var.trace_add("write", lambda *_args: self.update_dataset_summary())
        self.state.clean_trial_outputs_callback = self.clean_trial_outputs
        self.state.clean_trial_caches_callback = self.clean_trial_caches
        self.on_keypoints_changed()

    def selected_triangulation_flip_method(self) -> str:
        value = str(self.triang_flip_method.get()).strip().lower()
        if value == "raw":
            value = "once"
        return value if value in {"once", "greedy", "exhaustive"} else "once"

    def on_keypoints_changed(self) -> None:
        keypoints_path = ROOT / self.keypoints.get()
        trc_path = infer_pose2sim_trc_from_keypoints(keypoints_path) if keypoints_path.exists() else None
        if trc_path is not None:
            rel = display_path(trc_path)
            if self.state.pose2sim_trc_var.get() != rel:
                self.state.pose2sim_trc_var.set(rel)
            self.trc_status_var.set(f"TRC file auto-detected: {rel}")
        else:
            if self.state.pose2sim_trc_var.get():
                self.state.pose2sim_trc_var.set("")
            self.trc_status_var.set(
                f"Aucun fichier TRC correspondant n'a été trouvé pour {keypoints_path.name}. "
                "The TRC-file reconstruction will stay unavailable until a matching TRC file is provided."
            )
        self.update_dataset_summary()
        self.update_flip_status_text()

    def on_flip_settings_changed(self) -> None:
        self.flip_masks = {
            key: value
            for key, value in self.flip_masks.items()
            if key in {"epipolar", "epipolar_fast", "epipolar_viterbi", "epipolar_fast_viterbi"}
        }
        self.flip_diagnostics = {
            key: value
            for key, value in self.flip_diagnostics.items()
            if key in {"epipolar", "epipolar_fast", "epipolar_viterbi", "epipolar_fast_viterbi"}
        }
        self.update_flip_status_text()
        if self.pose_data is not None and self.calibrations is not None:
            self.refresh_plot()

    def update_flip_status_text(self) -> None:
        try:
            restrict = bool(self.state.flip_restrict_to_outliers_var.get())
            percentile = float(self.state.flip_outlier_percentile_var.get())
            floor_px = float(self.state.flip_outlier_floor_px_var.get())
            improvement_ratio = float(self.state.flip_improvement_ratio_var.get())
            min_gain_px = float(self.state.flip_min_gain_px_var.get())
            min_other_cameras = int(self.state.flip_min_other_cameras_var.get())
            temporal_weight = float(self.state.flip_temporal_weight_var.get())
            temporal_tau_px = float(self.state.flip_temporal_tau_px_var.get())
        except Exception:
            self.flip_status_var.set("")
            return
        gating = (
            f"test if nominal >= max({floor_px:.1f}px, Q{percentile:.0f})" if restrict else "test all camera-frames"
        )
        self.flip_status_var.set(
            f"{gating} | accept if swapped < {improvement_ratio:.2f}*nominal and gain >= {min_gain_px:.1f}px"
            f" | tau_epi={DEFAULT_EPIPOLAR_THRESHOLD_PX:.1f}px | tau_triang={DEFAULT_REPROJECTION_THRESHOLD_PX:.1f}px"
            f" | triang={self.selected_triangulation_flip_method()} | temp w={temporal_weight:.2f} | temp tau={temporal_tau_px:.1f}px | min cams={min_other_cameras}"
        )

    def clean_trial_outputs(self) -> None:
        dataset_dir = current_dataset_dir(self.state)
        trial_name = current_dataset_name(self.state)
        if not dataset_dir.exists():
            messagebox.showinfo(
                "Clean trial outputs", f"No outputs found for this dataset:\n{display_path(dataset_dir)}"
            )
            return
        confirmed = messagebox.askyesno(
            "Clean trial outputs",
            f"Delete all generated outputs for trial '{trial_name}'?\n\n"
            f"{display_path(dataset_dir)}\n\n"
            "This will remove models, reconstructions, figures, caches, and generated files for this trial.\n\n"
            "This action cannot be undone.",
            icon=messagebox.WARNING,
        )
        if not confirmed:
            return
        try:
            shutil.rmtree(dataset_dir)
            self.state.pose_data_cache.clear()
            self.state.calibration_cache.clear()
            self.state.notify_reconstructions_updated()
            self.update_dataset_summary()
            panel = getattr(self.state, "shared_reconstruction_panel", None)
            refresh_callback = getattr(panel, "_refresh_callback", None)
            if callable(refresh_callback):
                self.after_idle(refresh_callback)
            messagebox.showinfo("Clean trial outputs", f"Deleted outputs for:\n{display_path(dataset_dir)}")
        except Exception as exc:
            messagebox.showerror("Clean trial outputs", str(exc))

    def clean_trial_caches(self) -> None:
        dataset_dir = current_dataset_dir(self.state)
        trial_name = current_dataset_name(self.state)
        cache_root = dataset_dir / "_cache"
        preview_cache_paths = list(dataset_dir.glob("models/**/preview_q0_cache.npz")) if dataset_dir.exists() else []
        if not cache_root.exists() and not preview_cache_paths:
            messagebox.showinfo("Clean trial caches", f"No caches found for this dataset:\n{display_path(dataset_dir)}")
            return
        confirmed = messagebox.askyesno(
            "Clean trial caches",
            f"Delete generated caches for trial '{trial_name}'?\n\n"
            f"{display_path(dataset_dir)}\n\n"
            "This removes cached intermediate files only. Models, reconstructions, and figures are kept.",
            icon=messagebox.WARNING,
        )
        if not confirmed:
            return
        try:
            if cache_root.exists():
                shutil.rmtree(cache_root)
            for cache_path in preview_cache_paths:
                if cache_path.exists():
                    cache_path.unlink()
            self.state.pose_data_cache.clear()
            self.state.calibration_cache.clear()
            self.state.notify_reconstructions_updated()
            self.update_dataset_summary()
            panel = getattr(self.state, "shared_reconstruction_panel", None)
            refresh_callback = getattr(panel, "_refresh_callback", None)
            if callable(refresh_callback):
                self.after_idle(refresh_callback)
            messagebox.showinfo("Clean trial caches", f"Deleted caches for:\n{display_path(dataset_dir)}")
        except Exception as exc:
            messagebox.showerror("Clean trial caches", str(exc))

    def update_dataset_summary(self) -> None:
        ensure_dataset_layout(self.state)
        self.dataset_summary_var.set("")
        selected_cameras = current_selected_camera_names(self.state)
        self.selected_cameras_label_var.set(
            "Cameras: all" if not selected_cameras else f"Cameras: {format_camera_names(selected_cameras)}"
        )

    def selected_keypoints(self) -> list[str]:
        indices = self.keypoint_list.curselection()
        if not indices:
            return COCO17[:]
        return [COCO17[idx] for idx in indices]

    def _apply_keypoint_preset(self, preset: str) -> None:
        """Apply one named keypoint preset to the listbox selection."""

        selected_names = set(keypoint_preset_names(preset))
        self.keypoint_list.selection_clear(0, tk.END)
        for idx, name in enumerate(COCO17):
            if name in selected_names:
                self.keypoint_list.selection_set(idx)
        self.refresh_plot()

    def select_all_keypoints(self) -> None:
        """Select every available COCO keypoint in the 2D explorer."""

        self._apply_keypoint_preset("all")

    def clear_keypoints(self) -> None:
        """Clear the 2D-explorer keypoint selection."""

        self._apply_keypoint_preset("none")

    def select_body_keypoints(self) -> None:
        """Keep trunk and limb keypoints while removing face-side details."""

        self._apply_keypoint_preset("body_only")

    def load_data(self) -> None:
        self.reload_data(show_errors=True)

    def reload_data(self, show_errors: bool = True) -> None:
        try:
            self.calibrations, self.pose_data = get_cached_pose_data(
                self.state,
                keypoints_path=ROOT / self.keypoints.get(),
                calib_path=ROOT / self.calib.get(),
                data_mode="cleaned",
                smoothing_window=int(self.pose_filter_window.get()),
                outlier_threshold_ratio=float(self.pose_outlier_ratio.get()),
                lower_percentile=float(self.pose_p_low.get()),
                upper_percentile=float(self.pose_p_high.get()),
            )
            self.flip_masks = {}
            self.flip_diagnostics = {}
            self.ensure_flip_diagnostics()
            self.refresh_plot()
        except Exception as exc:
            if show_errors:
                messagebox.showerror("2D explorer", str(exc))

    def ensure_flip_diagnostics(self) -> None:
        if self.pose_data is None or self.calibrations is None:
            return
        output_dir = current_dataset_dir(self.state)
        improvement_ratio = float(self.state.flip_improvement_ratio_var.get())
        min_gain_px = float(self.state.flip_min_gain_px_var.get())
        min_other_cameras = int(self.state.flip_min_other_cameras_var.get())
        restrict_to_outliers = bool(self.state.flip_restrict_to_outliers_var.get())
        outlier_percentile = float(self.state.flip_outlier_percentile_var.get())
        outlier_floor_px = float(self.state.flip_outlier_floor_px_var.get())
        temporal_weight = float(self.state.flip_temporal_weight_var.get())
        temporal_tau_px = float(self.state.flip_temporal_tau_px_var.get())
        selected_triangulation_method = f"triangulation_{self.selected_triangulation_flip_method()}"
        for method in (
            "epipolar",
            "epipolar_fast",
            "epipolar_viterbi",
            "epipolar_fast_viterbi",
            selected_triangulation_method,
        ):
            if method in self.flip_masks:
                continue
            suspect_mask, diagnostics, _compute_time_s, _cache_path, _flip_source = (
                load_or_compute_left_right_flip_cache(
                    output_dir=output_dir,
                    pose_data=self.pose_data,
                    calibrations=self.calibrations,
                    method=method,
                    pose_data_mode="cleaned",
                    pose_filter_window=int(self.pose_filter_window.get()),
                    pose_outlier_threshold_ratio=float(self.pose_outlier_ratio.get()),
                    pose_amplitude_lower_percentile=float(self.pose_p_low.get()),
                    pose_amplitude_upper_percentile=float(self.pose_p_high.get()),
                    improvement_ratio=improvement_ratio,
                    min_gain_px=min_gain_px,
                    min_other_cameras=min_other_cameras,
                    restrict_to_outliers=restrict_to_outliers,
                    outlier_percentile=outlier_percentile,
                    outlier_floor_px=outlier_floor_px,
                    tau_px=(
                        DEFAULT_EPIPOLAR_THRESHOLD_PX
                        if method in {"epipolar", "epipolar_fast", "epipolar_viterbi", "epipolar_fast_viterbi"}
                        else DEFAULT_REPROJECTION_THRESHOLD_PX
                    ),
                    temporal_weight=temporal_weight,
                    temporal_tau_px=temporal_tau_px,
                )
            )
            self.flip_masks[method] = suspect_mask
            self.flip_diagnostics[method] = diagnostics

    def refresh_plot(self) -> None:
        if self.pose_data is None or self.calibrations is None:
            return
        self.ensure_flip_diagnostics()
        keypoint_names = self.selected_keypoints()
        keypoint_indices = [KP_INDEX[name] for name in keypoint_names]
        if self.view_mode.get() == "raw":
            points = np.asarray(
                self.pose_data.raw_keypoints if self.pose_data.raw_keypoints is not None else self.pose_data.keypoints,
                dtype=float,
            )
        elif self.view_mode.get() == "filtered":
            points = np.asarray(
                (
                    self.pose_data.filtered_keypoints
                    if self.pose_data.filtered_keypoints is not None
                    else self.pose_data.keypoints
                ),
                dtype=float,
            )
        else:
            points = np.asarray(self.pose_data.keypoints, dtype=float)
        correction_mode = self.flip_mode.get()
        if (
            correction_mode in {"epipolar", "epipolar_fast", "epipolar_viterbi", "epipolar_fast_viterbi"}
            and correction_mode in self.flip_masks
        ):
            points = apply_left_right_flip_to_points(points, self.flip_masks[correction_mode])
        elif correction_mode == "triangulation":
            triangulation_method = f"triangulation_{self.selected_triangulation_flip_method()}"
            if triangulation_method in self.flip_masks:
                points = apply_left_right_flip_to_points(points, self.flip_masks[triangulation_method])
        t = np.asarray(self.pose_data.frames, dtype=float) / float(self.fps.get())

        self.figure.clear()
        n_cams = len(self.pose_data.camera_names)
        nrows, ncols = camera_layout(n_cams)
        axes = self.figure.subplots(nrows, ncols, sharex=True)
        axes = np.atleast_1d(axes).ravel()
        colors = {name: plt_color for name, plt_color in zip(COCO17, plt_colormap(len(COCO17)))}

        for ax_idx, ax in enumerate(axes):
            if ax_idx >= n_cams:
                ax.axis("off")
                continue
            cam_name = self.pose_data.camera_names[ax_idx]
            width, height = self.calibrations[cam_name].image_size
            for kp_name, kp_idx in zip(keypoint_names, keypoint_indices):
                values = points[ax_idx, :, kp_idx, 0 if self.component.get() == "x" else 1]
                valid = np.isfinite(values)
                if not np.any(valid):
                    continue
                ax.plot(
                    t[valid],
                    values[valid],
                    linestyle="None",
                    marker="o",
                    markersize=2.2,
                    alpha=0.8,
                    color=colors[kp_name],
                    label=kp_name,
                )
            if "epipolar" in self.flip_masks:
                for frame_idx in np.flatnonzero(self.flip_masks["epipolar"][ax_idx]):
                    ax.axvline(t[frame_idx], color="#c44e52", linestyle="--", linewidth=0.9, alpha=0.28)
            if "epipolar_fast" in self.flip_masks:
                for frame_idx in np.flatnonzero(self.flip_masks["epipolar_fast"][ax_idx]):
                    ax.axvline(t[frame_idx], color="#dd8452", linestyle="-.", linewidth=0.9, alpha=0.26)
            if "epipolar_viterbi" in self.flip_masks:
                for frame_idx in np.flatnonzero(self.flip_masks["epipolar_viterbi"][ax_idx]):
                    ax.axvline(t[frame_idx], color="#8c564b", linestyle="--", linewidth=0.9, alpha=0.20)
            if "epipolar_fast_viterbi" in self.flip_masks:
                for frame_idx in np.flatnonzero(self.flip_masks["epipolar_fast_viterbi"][ax_idx]):
                    ax.axvline(t[frame_idx], color="#e377c2", linestyle="-.", linewidth=0.9, alpha=0.20)
            triangulation_method = f"triangulation_{self.selected_triangulation_flip_method()}"
            if triangulation_method in self.flip_masks:
                for frame_idx in np.flatnonzero(self.flip_masks[triangulation_method][ax_idx]):
                    ax.axvline(t[frame_idx], color="#4c72b0", linestyle=":", linewidth=1.0, alpha=0.30)
            ax.set_title(cam_name.replace("Camera", ""))
            ax.grid(alpha=0.2)
            if self.component.get() == "x":
                ax.set_ylim(0, width)
                ax.set_ylabel("x (px)")
            else:
                ax.set_ylim(height, 0)
                ax.set_ylabel("y (px)")
            if ax_idx >= (nrows - 1) * ncols:
                ax.set_xlabel("Temps (s)")

        handles, labels = axes[0].get_legend_handles_labels() if axes.size else ([], [])
        if handles:
            self.figure.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)), fontsize=8)
        self.figure.suptitle(
            f"2D {self.view_mode.get()} | composante {self.component.get()} | correction L/R {self.flip_mode.get()} | "
            f"tau epi {DEFAULT_EPIPOLAR_THRESHOLD_PX:.1f}px | tau triang {DEFAULT_REPROJECTION_THRESHOLD_PX:.1f}px | "
            f"triang mode {self.selected_triangulation_flip_method()} | flips: epi rouge --, epi fast orange -., epi Viterbi brun --, epiF Viterbi rose -., triang bleu :",
            y=0.98,
        )
        self.figure.tight_layout()
        self.canvas.draw_idle()


class ModelTab(CommandTab):
    def __init__(self, master, state: SharedAppState):
        super().__init__(master, "Model", show_command_preview=False, show_output=False)
        self.state = state
        self.preview_points = None
        self.preview_support_points = None
        self.preview_q0: np.ndarray | None = None
        self.preview_q_t0: np.ndarray | None = None
        self.preview_q_current: np.ndarray | None = None
        self.preview_q_names: list[str] = []
        self.preview_viewer = tk.StringVar(value="matplotlib")
        self.preview_model = None
        self.preview_marker_names: list[str] = []
        self.preview_segment_frames: list[tuple[str, np.ndarray, np.ndarray]] = []
        self.preview_metadata: dict[str, object] = {}
        self._updating_dof_controls = False
        self._auto_frame_range: tuple[str, str] | None = None
        self._syncing_frame_defaults = False
        self.set_run_button_text("Generate model")
        self.content_pane = ttk.Panedwindow(self.main, orient=tk.HORIZONTAL)
        self.left_panel = ttk.Frame(self.content_pane)
        self.right_panel = ttk.Frame(self.content_pane)
        self.content_pane.add(self.left_panel, weight=1)
        self.content_pane.add(self.right_panel, weight=2)

        form = ttk.LabelFrame(self.left_panel, text="Construction du modèle")

        row = ttk.Frame(form)
        row.pack(fill=tk.X, padx=8, pady=4)
        self.subject_mass = LabeledEntry(row, "Subject mass", "55", label_width=10, entry_width=6)
        self.subject_mass.pack(side=tk.LEFT, padx=(0, 8))
        self.model_variant = tk.StringVar(value=DEFAULT_MODEL_VARIANT)
        structure_label = ttk.Label(row, text="Structure", width=10)
        structure_label.pack(side=tk.LEFT)
        structure_box = ttk.Combobox(
            row,
            textvariable=self.model_variant,
            values=list(SUPPORTED_MODEL_VARIANTS),
            width=14,
            state="readonly",
        )
        structure_box.pack(side=tk.LEFT, padx=(0, 8))

        row1b = ttk.Frame(form)
        row1b.pack(fill=tk.X, padx=8, pady=4)
        self.symmetrize_limbs_var = tk.BooleanVar(value=DEFAULT_MODEL_SYMMETRIZE_LIMBS)
        symmetrize_check = ttk.Checkbutton(
            row1b,
            text="Symmetrize limbs",
            variable=self.symmetrize_limbs_var,
        )
        symmetrize_check.pack(side=tk.LEFT, padx=(0, 12))
        self.triang_method = tk.StringVar(value="exhaustive")
        triang_label = ttk.Label(row1b, text="Triangulation", width=12)
        triang_label.pack(side=tk.LEFT)
        triang_box = ttk.Combobox(
            row1b,
            textvariable=self.triang_method,
            values=["once", "greedy", "exhaustive"],
            width=12,
            state="readonly",
        )
        triang_box.pack(side=tk.LEFT, padx=(0, 8))

        row2 = ttk.Frame(form)
        row2.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(row2, text="Frames", width=10).pack(side=tk.LEFT)
        self.frame_start = LabeledEntry(row2, "Start", "", label_width=5, entry_width=6)
        self.frame_start.pack(side=tk.LEFT, padx=(0, 8))
        self.frame_end = LabeledEntry(row2, "End", "", label_width=4, entry_width=6)
        self.frame_end.pack(side=tk.LEFT, padx=(0, 8))
        self.max_frames = LabeledEntry(row2, "Nb", "", label_width=3, entry_width=6)
        self.max_frames.pack(side=tk.LEFT, padx=(0, 8))

        row2b = ttk.Frame(form)
        row2b.pack(fill=tk.X, padx=8, pady=4)
        default_model_pose_mode = state.pose_data_mode_var.get().strip()
        if default_model_pose_mode not in ("raw", "annotated", "cleaned"):
            default_model_pose_mode = "cleaned"
        self.pose_mode_box = None
        self.pose_data_mode = tk.StringVar(value=default_model_pose_mode)
        pose_mode_label = ttk.Label(row2b, text="2D source", width=10)
        pose_mode_label.pack(side=tk.LEFT)
        self.pose_mode_box = ttk.Combobox(
            row2b, textvariable=self.pose_data_mode, values=["raw", "cleaned"], width=10, state="readonly"
        )
        self.pose_mode_box.pack(side=tk.LEFT, padx=(0, 8))

        row2c = ttk.Frame(form)
        row2c.pack(fill=tk.X, padx=8, pady=4)
        default_pose_correction_mode = current_calibration_correction_mode(state)
        self.pose_correction_mode = tk.StringVar(value=default_pose_correction_mode)
        pose_correction_label = ttk.Label(row2c, text="L/R corr", width=8)
        pose_correction_label.pack(side=tk.LEFT)
        pose_correction_box = ttk.Combobox(
            row2c,
            textvariable=self.pose_correction_mode,
            values=[
                "none",
                "flip_epipolar",
                "flip_epipolar_fast",
                "flip_epipolar_viterbi",
                "flip_epipolar_fast_viterbi",
                "flip_triangulation",
            ],
            width=17,
            state="readonly",
        )
        pose_correction_box.pack(side=tk.LEFT, padx=(0, 8))
        self.model_info_var = tk.StringVar(value="")
        ttk.Label(row2c, textvariable=self.model_info_var, foreground="#4f5b66").pack(side=tk.LEFT, padx=(6, 0))

        row3 = ttk.Frame(form)
        row3.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(row3, text="Root orientation", width=14).pack(side=tk.LEFT)
        self.initial_rot_var = state.initial_rotation_correction_var
        initial_rot_check = ttk.Checkbutton(
            row3,
            text="Align root to global",
            variable=self.initial_rot_var,
        )
        initial_rot_check.pack(side=tk.LEFT)

        self.details_var = tk.StringVar(value="")
        details = ttk.Label(
            form,
            textvariable=self.details_var,
            foreground="#4f5b66",
            justify=tk.LEFT,
        )
        details.pack(fill=tk.X, padx=8, pady=(0, 4))
        ttk.Button(form, text="Load first frame preview", command=self.load_preview).pack(
            anchor="w", padx=8, pady=(0, 4)
        )

        self.subject_mass.set_tooltip("Masse du sujet utilisée pour les paramètres inertiels du modèle.")
        attach_tooltip(
            structure_label,
            "Topologie du bioMod: single_trunk garde le modèle actuel; back_flexion_1d/back_3dof gardent la racine au bassin; upper_root_back_* place la racine au haut du tronc et met le dos mobile vers le bassin.",
        )
        attach_tooltip(
            structure_box,
            "single_trunk: modèle actuel. back_flexion_1d/back_3dof: UPPER_BACK entre bassin et épaules. upper_root_back_flexion_1d/upper_root_back_3dof: racine au haut du tronc, LOWER_TRUNK mobile vers le bassin.",
        )
        attach_tooltip(
            symmetrize_check,
            "Si coché, les longueurs bras/jambe gauche et droite sont moyennées pour construire un bioMod symétrique. Sinon, le modèle conserve les longueurs latéralisées estimées.",
        )
        attach_tooltip(
            triang_label, "Méthode de triangulation utilisée pour construire le modèle: once, greedy, ou exhaustive."
        )
        attach_tooltip(
            triang_box,
            "once: une seule triangulation pondérée. greedy: suppression gloutonne des pires vues. exhaustive: la plus robuste mais plus coûteuse.",
        )
        attach_tooltip(
            pose_mode_label,
            "Choix de la version de base des 2D utilisées pour construire le modèle: raw, cleaned, ou annotated si un fichier d'annotations existe.",
        )
        attach_tooltip(
            self.pose_mode_box,
            "Choix de la version de base des 2D utilisées pour construire le modèle. `annotated` n'est proposé que si un fichier d'annotations existe pour l'essai courant.",
        )
        attach_tooltip(
            pose_correction_label, "Correction optionnelle des labels gauche/droite avant la triangulation du modèle."
        )
        attach_tooltip(
            pose_correction_box,
            "Utilise les 2D telles quelles (`none`), ou une version corrigée des flips L/R estimée par l'approche épipolaire Sampson, par la distance épipolaire symétrique rapide, ou par triangulation/reprojection.",
        )
        attach_tooltip(
            initial_rot_check,
            "Estime l'orientation horizontale de l'axe y du tronc à t0, l'arrondit au multiple de pi/2 le plus proche, puis applique cette correction autour de Z dans le bioMod.",
        )
        self.max_frames.set_tooltip(
            "Nombre de frames a prendre uniformement entre Start et End. Laisser vide pour garder toute la plage."
        )
        self.frame_start.set_tooltip("Première frame incluse pour construire le modèle.")
        self.frame_end.set_tooltip("Dernière frame incluse pour construire le modèle.")

        self.state.calib_var.trace_add("write", lambda *_args: self.sync_paths_from_state())
        self.state.keypoints_var.trace_add("write", lambda *_args: self.sync_paths_from_state())
        self.state.annotation_path_var.trace_add("write", lambda *_args: self.sync_paths_from_state())
        self.state.output_root_var.trace_add("write", lambda *_args: self.sync_paths_from_state())
        self.state.register_reconstruction_listener(self.refresh_existing_models)
        self.model_variant.trace_add("write", lambda *_args: self.update_details())
        self.model_variant.trace_add("write", lambda *_args: self.refresh_existing_models())
        self.preview_viewer.trace_add("write", lambda *_args: self.update_preview_viewer_controls())
        self.symmetrize_limbs_var.trace_add("write", lambda *_args: self.update_details())
        self.symmetrize_limbs_var.trace_add("write", lambda *_args: self.refresh_existing_models())
        self.triang_method.trace_add("write", lambda *_args: self.update_details())
        self.pose_data_mode.trace_add("write", lambda *_args: self.update_details())
        self.pose_correction_mode.trace_add("write", lambda *_args: self.update_details())
        self.initial_rot_var.trace_add("write", lambda *_args: self.update_details())
        self.state.pose_filter_window_var.trace_add("write", lambda *_args: self.refresh_existing_models())
        self.state.pose_outlier_ratio_var.trace_add("write", lambda *_args: self.refresh_existing_models())
        self.state.pose_p_low_var.trace_add("write", lambda *_args: self.refresh_existing_models())
        self.state.pose_p_high_var.trace_add("write", lambda *_args: self.refresh_existing_models())
        self.state.pose_filter_window_var.trace_add("write", lambda *_args: self.sync_paths_from_state())
        self.state.pose_outlier_ratio_var.trace_add("write", lambda *_args: self.sync_paths_from_state())
        self.state.pose_p_low_var.trace_add("write", lambda *_args: self.sync_paths_from_state())
        self.state.pose_p_high_var.trace_add("write", lambda *_args: self.sync_paths_from_state())
        self.max_frames.var.trace_add("write", lambda *_args: self.update_details())
        self.frame_start.var.trace_add("write", lambda *_args: self.update_details())
        self.frame_end.var.trace_add("write", lambda *_args: self.update_details())
        self.subject_mass.var.trace_add("write", lambda *_args: self.update_details())

        # Layout: controls and model list on the left, preview on the right.
        self.hide_preview_copy_buttons()

        existing_box = ttk.LabelFrame(self.left_panel, text="Existing models")
        existing_controls = ttk.Frame(existing_box)
        existing_controls.pack(fill=tk.X, padx=8, pady=(8, 0))
        ttk.Button(existing_controls, text="Refresh models", command=self.refresh_existing_models).pack(side=tk.LEFT)
        ttk.Button(existing_controls, text="Clear models", command=self.clear_models).pack(side=tk.LEFT, padx=(8, 0))
        self.model_tree = ttk.Treeview(existing_box, columns=("name", "match", "path"), show="headings", height=12)
        self.model_tree.heading("name", text="Model")
        self.model_tree.heading("match", text="Match")
        self.model_tree.heading("path", text="bioMod path")
        self.model_tree.column("name", width=160, anchor="w")
        self.model_tree.column("match", width=80, anchor="center")
        self.model_tree.column("path", width=360, anchor="w")
        self.model_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.model_tree.bind("<<TreeviewSelect>>", lambda _event: self.load_preview(use_selected_model=True))
        attach_tooltip(
            self.model_tree,
            "Liste des modeles detectes pour le trial courant. La colonne Match indique s'ils correspondent aux options 2D courantes.",
        )

        preview_box = ttk.LabelFrame(self.right_panel, text="Première frame triangulée / modèle")
        preview_controls_top = ttk.Frame(preview_box)
        preview_controls_top.pack(fill=tk.X, padx=8, pady=(8, 2))
        self.show_triangulation_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            preview_controls_top,
            text="Show triangulation at t0",
            variable=self.show_triangulation_var,
            command=self.refresh_preview,
        ).pack(side=tk.LEFT, padx=(0, 12))
        self.show_local_frames_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            preview_controls_top,
            text="Show local frames",
            variable=self.show_local_frames_var,
            command=self.refresh_preview,
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(preview_controls_top, text="Viewer", width=7).pack(side=tk.LEFT)
        self.preview_viewer_box = ttk.Combobox(
            preview_controls_top,
            textvariable=self.preview_viewer,
            values=list(SUPPORTED_MODEL_PREVIEW_VIEWERS),
            width=11,
            state="readonly",
        )
        self.preview_viewer_box.pack(side=tk.LEFT, padx=(0, 8))
        self.open_preview_viewer_button = ttk.Button(
            preview_controls_top, text="Open pyorerun", command=self.open_preview_in_viewer
        )
        self.open_preview_viewer_button.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(preview_controls_top, text="Pose", width=6).pack(side=tk.LEFT)
        self.preview_pose_mode = tk.StringVar(value="q=0")
        self.preview_pose_box = ttk.Combobox(
            preview_controls_top,
            textvariable=self.preview_pose_mode,
            values=["q=0", "q(t0)"],
            width=10,
            state="readonly",
        )
        self.preview_pose_box.pack(side=tk.LEFT, padx=(0, 12))
        self.preview_pose_box.bind("<<ComboboxSelected>>", lambda _event: self.on_preview_pose_mode_changed())

        preview_controls_bottom = ttk.Frame(preview_box)
        preview_controls_bottom.pack(fill=tk.X, padx=8, pady=(0, 4))
        ttk.Label(preview_controls_bottom, text="DoF", width=6).pack(side=tk.LEFT)
        self.preview_dof_var = tk.StringVar(value="")
        self.preview_dof_box = ttk.Combobox(
            preview_controls_bottom, textvariable=self.preview_dof_var, values=[], width=26, state="disabled"
        )
        self.preview_dof_box.pack(side=tk.LEFT, padx=(0, 8))
        self.preview_dof_box.bind("<<ComboboxSelected>>", lambda _event: self.on_preview_dof_selected())
        self.preview_dof_value_var = tk.StringVar(value="")
        ttk.Label(preview_controls_bottom, textvariable=self.preview_dof_value_var, width=12).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        self.preview_dof_slider = ttk.Scale(
            preview_controls_bottom,
            from_=-2.0 * math.pi,
            to=2.0 * math.pi,
            orient=tk.HORIZONTAL,
            command=self.on_preview_dof_slider,
            state="disabled",
        )
        self.preview_dof_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        attach_tooltip(
            self.preview_pose_box,
            "Choisit la pose de référence utilisée pour afficher le modèle: neutre q=0 ou q(t0) estimé à partir de la première frame triangulée valide.",
        )
        attach_tooltip(
            self.preview_viewer_box,
            "Choisit entre le preview intégré matplotlib et une ouverture externe dans pyorerun.",
        )
        attach_tooltip(
            self.open_preview_viewer_button,
            "Ouvre le modèle courant dans pyorerun avec la pose actuellement affichée. Le preview matplotlib reste disponible dans l'onglet.",
        )
        attach_tooltip(self.preview_dof_box, "Choisit le DoF du modele a modifier manuellement dans le preview.")
        attach_tooltip(self.preview_dof_slider, "Fait varier le DoF selectionne entre -2pi et 2pi.")
        self.preview_figure = Figure(figsize=(8, 6))
        self.preview_canvas = FigureCanvasTkAgg(self.preview_figure, master=preview_box)
        self.preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.extra = LabeledEntry(form, "Extra args", "")
        self.extra.pack(fill=tk.X, padx=8, pady=4)

        form.pack(fill=tk.X, pady=(0, 8))
        existing_box.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        preview_box.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        self.content_pane.pack(fill=tk.BOTH, expand=True)

        self.update_details()
        self.sync_paths_from_state()
        self.refresh_existing_models()
        self.update_preview_viewer_controls()

    def current_pose_correction_mode(self) -> str:
        return normalize_pose_correction_mode(self.pose_correction_mode.get())

    def _available_model_pose_modes(self) -> list[str]:
        keypoints_value = str(self.state.keypoints_var.get()).strip()
        if not keypoints_value:
            return ["raw", "cleaned"]
        return available_model_pose_modes(self.state, ROOT / keypoints_value)

    def _sync_available_pose_modes(self) -> None:
        modes = self._available_model_pose_modes()
        if hasattr(self, "pose_mode_box") and self.pose_mode_box is not None:
            self.pose_mode_box.configure(values=modes)
        current = str(self.pose_data_mode.get()).strip()
        if current not in modes:
            fallback = "annotated" if "annotated" in modes else ("cleaned" if "cleaned" in modes else modes[0])
            self.pose_data_mode.set(fallback)

    def current_pose_source_label(self) -> str:
        correction_mode = self.current_pose_correction_mode()
        if correction_mode == "none":
            return self.pose_data_mode.get()
        return f"{self.pose_data_mode.get()} + {correction_mode}"

    def _model_min_cameras_for_triangulation(self) -> int:
        """Allow sparse annotated frames to bootstrap model creation with two views."""

        if str(self.pose_data_mode.get()).strip() == "annotated":
            return 2
        return DEFAULT_MIN_CAMERAS_FOR_TRIANGULATION

    def update_details(self) -> None:
        max_frames = self.max_frames.get() or "all"
        frame_start = self.frame_start.get() or "-"
        frame_end = self.frame_end.get() or "-"
        self.details_var.set(
            f"Model creation will use: {self.current_pose_source_label()} 2D data, "
            f"frames {frame_start} -> {frame_end}, nb {max_frames}, "
            f"structure {self.model_variant.get()}, "
            f"{'sym' if self.symmetrize_limbs_var.get() else 'asym'}, "
            f"triangulation {self.triang_method.get()}, "
            f"root rot-fix {'on' if self.initial_rot_var.get() else 'off'}, "
            f"subject mass {self.subject_mass.get()} kg."
        )
        self.sync_paths_from_state()

    def _available_pose_frame_bounds(self) -> tuple[int, int] | None:
        """Return the full frame range available in the current 2D source."""

        keypoints_path = ROOT / self.state.keypoints_var.get()
        calib_path = ROOT / self.state.calib_var.get()
        if not keypoints_path.exists() or not calib_path.exists():
            return None
        _calibrations, pose_data = get_cached_pose_data(
            self.state,
            keypoints_path=keypoints_path,
            calib_path=calib_path,
            max_frames=None,
            frame_start=None,
            frame_end=None,
            data_mode=self.pose_data_mode.get(),
            smoothing_window=int(self.state.pose_filter_window_var.get()),
            outlier_threshold_ratio=float(self.state.pose_outlier_ratio_var.get()),
            lower_percentile=float(self.state.pose_p_low_var.get()),
            upper_percentile=float(self.state.pose_p_high_var.get()),
        )
        if pose_data.frames.size == 0:
            return None
        return int(pose_data.frames[0]), int(pose_data.frames[-1])

    def _sync_frame_range_defaults(self) -> None:
        """Populate Start/End with the available 2D frame range without overwriting manual edits."""

        if self._syncing_frame_defaults:
            return
        try:
            bounds = self._available_pose_frame_bounds()
        except Exception:
            return
        if bounds is None:
            return
        next_range = (str(bounds[0]), str(bounds[1]))
        current_range = (self.frame_start.get().strip(), self.frame_end.get().strip())
        previous_auto_range = self._auto_frame_range
        should_update = (
            not current_range[0]
            or not current_range[1]
            or previous_auto_range is None
            or current_range == previous_auto_range
        )
        if not should_update:
            return
        if current_range == next_range:
            self._auto_frame_range = next_range
            return
        self._syncing_frame_defaults = True
        try:
            self.frame_start.var.set(next_range[0])
            self.frame_end.var.set(next_range[1])
            self._auto_frame_range = next_range
        finally:
            self._syncing_frame_defaults = False

    def sync_paths_from_state(self) -> None:
        self._sync_available_pose_modes()
        self._sync_frame_range_defaults()
        try:
            subject_mass_kg = float(self.subject_mass.get())
            max_frames = int(self.max_frames.get()) if self.max_frames.get() else None
            frame_start = int(self.frame_start.get()) if self.frame_start.get() else None
            frame_end = int(self.frame_end.get()) if self.frame_end.get() else None
            pose_filter_window = int(self.state.pose_filter_window_var.get())
            pose_outlier_threshold_ratio = float(self.state.pose_outlier_ratio_var.get())
            pose_amplitude_lower_percentile = float(self.state.pose_p_low_var.get())
            pose_amplitude_upper_percentile = float(self.state.pose_p_high_var.get())
        except ValueError:
            return
        dataset_name = current_dataset_name(self.state)
        output_root = normalize_output_root(ROOT / self.state.output_root_var.get())
        model_dir = model_output_dir(
            output_root,
            dataset_name,
            pose_data_mode=self.pose_data_mode.get(),
            triangulation_method=self.triang_method.get(),
            model_variant=self.model_variant.get(),
            symmetrize_limbs=self.symmetrize_limbs_var.get(),
            pose_correction_mode=self.current_pose_correction_mode(),
            initial_rotation_correction=self.initial_rot_var.get(),
            max_frames=max_frames,
            frame_start=frame_start,
            frame_end=frame_end,
            subject_mass_kg=subject_mass_kg,
            pose_filter_window=pose_filter_window,
            pose_outlier_threshold_ratio=pose_outlier_threshold_ratio,
            pose_amplitude_lower_percentile=pose_amplitude_lower_percentile,
            pose_amplitude_upper_percentile=pose_amplitude_upper_percentile,
        )
        biomod_path = model_biomod_path(
            output_root,
            dataset_name,
            pose_data_mode=self.pose_data_mode.get(),
            triangulation_method=self.triang_method.get(),
            model_variant=self.model_variant.get(),
            symmetrize_limbs=self.symmetrize_limbs_var.get(),
            pose_correction_mode=self.current_pose_correction_mode(),
            initial_rotation_correction=self.initial_rot_var.get(),
            max_frames=max_frames,
            frame_start=frame_start,
            frame_end=frame_end,
            subject_mass_kg=subject_mass_kg,
            pose_filter_window=pose_filter_window,
            pose_outlier_threshold_ratio=pose_outlier_threshold_ratio,
            pose_amplitude_lower_percentile=pose_amplitude_lower_percentile,
            pose_amplitude_upper_percentile=pose_amplitude_upper_percentile,
        )
        self.model_info_var.set(f"{model_dir.name} -> {display_path(biomod_path)}")
        self.refresh_existing_models()

    def on_command_success(self) -> None:
        """Refresh the matching model list after a successful model generation."""

        self.refresh_existing_models()

    def update_preview_viewer_controls(self) -> None:
        if not hasattr(self, "open_preview_viewer_button"):
            return
        viewer = self.preview_viewer.get().strip()
        if viewer == "pyorerun":
            self.open_preview_viewer_button.configure(state="normal", text="Open pyorerun")
        else:
            self.open_preview_viewer_button.configure(state="disabled", text="Built-in viewer")

    def _pyorerun_states_path(self, biomod_path: Path) -> Path:
        return biomod_path.parent / "preview_pyorerun_states.npz"

    def _write_pyorerun_preview_states(self, states_path: Path) -> bool:
        q_values = self.preview_q_current
        if q_values is None or np.asarray(q_values, dtype=float).size == 0:
            return False
        q_row = np.asarray(q_values, dtype=float).reshape(1, -1)
        if not np.all(np.isfinite(q_row)):
            return False
        q_series = np.repeat(q_row, 2, axis=0)
        states_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(states_path, q=q_series)
        return True

    def open_preview_in_viewer(self) -> None:
        viewer = self.preview_viewer.get().strip()
        if viewer != "pyorerun":
            return
        try:
            biomod_path = self._preview_model_path(use_selected_model=True)
            if not biomod_path.exists():
                biomod_path = self._preview_model_path(use_selected_model=False)
            if not biomod_path.exists():
                raise ValueError("No bioMod available to open in pyorerun.")
            states_path = self._pyorerun_states_path(biomod_path)
            has_states = self._write_pyorerun_preview_states(states_path)
            cmd = [
                sys.executable,
                "tools/show_biomod_pyorerun.py",
                "--biomod",
                display_path(biomod_path),
                "--fps",
                self.state.fps_var.get(),
            ]
            if has_states:
                cmd.extend(["--mode", "trajectory", "--states", display_path(states_path)])
            else:
                cmd.extend(["--mode", "neutral"])
            subprocess.Popen(cmd, cwd=ROOT)
        except Exception as exc:
            messagebox.showerror("pyorerun", str(exc))

    def refresh_existing_models(self) -> None:
        for item in self.model_tree.get_children():
            self.model_tree.delete(item)
        dataset_dir = current_dataset_dir(self.state)
        biomod_paths: list[Path] = []
        for model_dir in scan_model_dirs(dataset_dir):
            biomod_paths.extend(sorted(model_dir.glob("*.bioMod")))
        expected_mode = self.pose_data_mode.get()
        expected_correction = self.current_pose_correction_mode()
        expected_model_variant = self.model_variant.get() if hasattr(self, "model_variant") else DEFAULT_MODEL_VARIANT
        expected_symmetrize_limbs = self.symmetrize_limbs_var.get() if hasattr(self, "symmetrize_limbs_var") else True
        expected_window = int(self.state.pose_filter_window_var.get())
        expected_ratio = float(self.state.pose_outlier_ratio_var.get())
        expected_p_low = float(self.state.pose_p_low_var.get())
        expected_p_high = float(self.state.pose_p_high_var.get())
        seen: set[Path] = set()
        found = 0
        for biomod_path in biomod_paths:
            if biomod_path in seen:
                continue
            seen.add(biomod_path)
            matches = self._model_matches_selected_2d_data(
                biomod_path.parent,
                expected_mode,
                expected_correction,
                expected_model_variant,
                expected_symmetrize_limbs,
                expected_window,
                expected_ratio,
                expected_p_low,
                expected_p_high,
            )
            parent_name = biomod_path.parent.name
            biomod_display_path = display_path(biomod_path)
            self.model_tree.insert(
                "",
                "end",
                iid=str(biomod_path),
                values=(parent_name, "yes" if matches else "no", biomod_display_path),
            )
            found += 1
        if found == 0:
            self.model_tree.insert(
                "",
                "end",
                iid="__no_model__",
                values=(
                    "-",
                    "-",
                    f"No existing bioMod matches current 2D data settings ({self.current_pose_source_label()})",
                ),
            )

    def clear_models(self) -> None:
        selected = self.model_tree.selection()
        paths: list[Path] = []

        def parse_displayed_path(raw: str) -> Path:
            path = Path(raw)
            return path if path.is_absolute() else ROOT / path

        if selected:
            for item in selected:
                values = self.model_tree.item(item, "values")
                if len(values) >= 3 and values[2] and not str(values[2]).startswith("No existing"):
                    paths.append(parse_displayed_path(str(values[2])))
        else:
            for item in self.model_tree.get_children():
                values = self.model_tree.item(item, "values")
                if len(values) >= 3 and values[2] and not str(values[2]).startswith("No existing"):
                    paths.append(parse_displayed_path(str(values[2])))

        model_dirs = sorted({path.parent for path in paths if path.exists()})
        if not model_dirs:
            messagebox.showinfo("Models", "Aucun modèle à supprimer pour les réglages 2D courants.")
            return

        target_label = "les modèles sélectionnés" if selected else "tous les modèles affichés"
        confirmed = messagebox.askyesno(
            "Clear models",
            f"Supprimer {len(model_dirs)} dossier(s) de modèle pour {target_label} ?",
            icon="warning",
        )
        if not confirmed:
            return

        errors: list[str] = []
        for model_dir in model_dirs:
            try:
                shutil.rmtree(model_dir)
            except Exception as exc:
                errors.append(f"{model_dir.name}: {exc}")

        self.sync_paths_from_state()
        self.refresh_existing_models()
        if errors:
            messagebox.showerror("Models", "Certaines suppressions ont échoué:\n\n" + "\n".join(errors))
        else:
            messagebox.showinfo("Models", f"{len(model_dirs)} dossier(s) de modèle supprimé(s).")

    @staticmethod
    def _cache_metadata(path: Path) -> dict[str, object]:
        if not path.exists():
            return {}
        try:
            with np.load(path, allow_pickle=True) as data:
                if "metadata" not in data:
                    return {}
                return json.loads(data["metadata"].item())
        except Exception:
            return {}

    def _model_matches_selected_2d_data(
        self,
        model_dir: Path,
        expected_mode: str,
        expected_correction: str,
        expected_model_variant: str,
        expected_symmetrize_limbs: bool,
        expected_window: int,
        expected_ratio: float,
        expected_p_low: float,
        expected_p_high: float,
    ) -> bool:
        model_stage_path = model_dir / "model_stage.npz"
        stage_metadata = self._cache_metadata(model_stage_path)
        reconstruction_cache_path = self._metadata_path(model_dir, stage_metadata.get("reconstruction_cache_path"))
        if reconstruction_cache_path is None or not reconstruction_cache_path.exists():
            return False
        reconstruction_metadata = self._cache_metadata(reconstruction_cache_path)
        if not reconstruction_metadata:
            return False
        return (
            reconstruction_metadata.get("pose_data_mode") == expected_mode
            and str(reconstruction_metadata.get("pose_correction_mode", "none")) == expected_correction
            and str(stage_metadata.get("model_variant", DEFAULT_MODEL_VARIANT)) == expected_model_variant
            and bool(stage_metadata.get("symmetrize_limbs", DEFAULT_MODEL_SYMMETRIZE_LIMBS))
            == bool(expected_symmetrize_limbs)
            and int(reconstruction_metadata.get("pose_filter_window", -1)) == expected_window
            and math.isclose(
                float(reconstruction_metadata.get("pose_outlier_threshold_ratio", math.nan)),
                expected_ratio,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            and math.isclose(
                float(reconstruction_metadata.get("pose_amplitude_lower_percentile", math.nan)),
                expected_p_low,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            and math.isclose(
                float(reconstruction_metadata.get("pose_amplitude_upper_percentile", math.nan)),
                expected_p_high,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        )

    @staticmethod
    def _metadata_path(base_dir: Path, raw_path: object) -> Path | None:
        """Resolve a cache path stored in metadata.

        Older metadata may store a path relative to the workspace root, while
        some local tools may emit paths relative to the model directory.
        """

        if raw_path is None:
            return None
        text = str(raw_path).strip()
        if not text:
            return None
        path = Path(text)
        if path.is_absolute():
            return path
        root_relative = ROOT / path
        if root_relative.exists():
            return root_relative
        model_relative = base_dir / path
        if model_relative.exists():
            return model_relative
        return root_relative

    def build_command(self) -> list[str]:
        cmd = [
            sys.executable,
            "vitpose_ekf_pipeline.py",
            "--model-only",
            "--calib",
            self.state.calib_var.get(),
            "--keypoints",
            self.state.keypoints_var.get(),
            "--pose-data-mode",
            self.pose_data_mode.get(),
            "--pose-correction-mode",
            self.current_pose_correction_mode(),
            "--pose-filter-window",
            self.state.pose_filter_window_var.get(),
            "--pose-outlier-threshold-ratio",
            self.state.pose_outlier_ratio_var.get(),
            "--pose-amplitude-lower-percentile",
            self.state.pose_p_low_var.get(),
            "--pose-amplitude-upper-percentile",
            self.state.pose_p_high_var.get(),
            "--fps",
            self.state.fps_var.get(),
            "--triangulation-workers",
            self.state.workers_var.get(),
            "--subject-mass-kg",
            self.subject_mass.get(),
            "--triangulation-method",
            self.triang_method.get(),
            "--model-variant",
            self.model_variant.get(),
            "--output-dir",
            display_path(self.derived_model_dir()),
            "--biomod",
            self.derived_biomod_path(),
        ]
        selected_cameras = current_selected_camera_names(self.state)
        if selected_cameras:
            cmd.extend(["--camera-names", ",".join(selected_cameras)])
        if not self.symmetrize_limbs_var.get():
            cmd.append("--no-symmetrize-limbs")
        if self.frame_start.get():
            cmd.extend(["--frame-start", self.frame_start.get()])
        if self.frame_end.get():
            cmd.extend(["--frame-end", self.frame_end.get()])
        if self.max_frames.get():
            cmd.extend(["--max-frames", self.max_frames.get()])
        if self.initial_rot_var.get():
            cmd.append("--initial-rotation-correction")
        min_cameras = self._model_min_cameras_for_triangulation()
        if min_cameras != DEFAULT_MIN_CAMERAS_FOR_TRIANGULATION:
            cmd.extend(["--min-cameras-for-triangulation", str(min_cameras)])
        cmd.extend(self.parse_extra_args(self.extra.get()))
        return cmd

    def derived_model_dir(self) -> Path:
        dataset_name = current_dataset_name(self.state)
        output_root = normalize_output_root(ROOT / self.state.output_root_var.get())
        return model_output_dir(
            output_root,
            dataset_name,
            pose_data_mode=self.pose_data_mode.get(),
            triangulation_method=self.triang_method.get(),
            model_variant=self.model_variant.get(),
            symmetrize_limbs=self.symmetrize_limbs_var.get(),
            pose_correction_mode=self.current_pose_correction_mode(),
            initial_rotation_correction=self.initial_rot_var.get(),
            max_frames=int(self.max_frames.get()) if self.max_frames.get() else None,
            frame_start=int(self.frame_start.get()) if self.frame_start.get() else None,
            frame_end=int(self.frame_end.get()) if self.frame_end.get() else None,
            subject_mass_kg=float(self.subject_mass.get()),
            pose_filter_window=int(self.state.pose_filter_window_var.get()),
            pose_outlier_threshold_ratio=float(self.state.pose_outlier_ratio_var.get()),
            pose_amplitude_lower_percentile=float(self.state.pose_p_low_var.get()),
            pose_amplitude_upper_percentile=float(self.state.pose_p_high_var.get()),
        )

    def derived_biomod_path(self) -> str:
        dataset_name = current_dataset_name(self.state)
        output_root = normalize_output_root(ROOT / self.state.output_root_var.get())
        return display_path(
            model_biomod_path(
                output_root,
                dataset_name,
                pose_data_mode=self.pose_data_mode.get(),
                triangulation_method=self.triang_method.get(),
                model_variant=self.model_variant.get(),
                symmetrize_limbs=self.symmetrize_limbs_var.get(),
                pose_correction_mode=self.current_pose_correction_mode(),
                initial_rotation_correction=self.initial_rot_var.get(),
                max_frames=int(self.max_frames.get()) if self.max_frames.get() else None,
                frame_start=int(self.frame_start.get()) if self.frame_start.get() else None,
                frame_end=int(self.frame_end.get()) if self.frame_end.get() else None,
                subject_mass_kg=float(self.subject_mass.get()),
                pose_filter_window=int(self.state.pose_filter_window_var.get()),
                pose_outlier_threshold_ratio=float(self.state.pose_outlier_ratio_var.get()),
                pose_amplitude_lower_percentile=float(self.state.pose_p_low_var.get()),
                pose_amplitude_upper_percentile=float(self.state.pose_p_high_var.get()),
            )
        )

    def selected_biomod_path(self) -> Path | None:
        selected = self.model_tree.selection()
        if not selected:
            return None
        values = self.model_tree.item(selected[0], "values")
        if len(values) < 3 or not values[2] or str(values[2]).startswith("No existing"):
            return None
        path = Path(str(values[2]))
        return path if path.is_absolute() else ROOT / path

    def _preview_model_path(self, use_selected_model: bool = False) -> Path:
        biomod_path = self.selected_biomod_path() if use_selected_model else None
        if biomod_path is None:
            biomod_path = ROOT / self.derived_biomod_path()
        return biomod_path

    def _preview_cache_path(self, biomod_path: Path) -> Path:
        return biomod_path.parent / "preview_q0_cache.npz"

    def _preview_cache_metadata(self, biomod_path: Path) -> dict[str, object]:
        return model_preview_cache_metadata(
            biomod_path=biomod_path,
            keypoints_path=ROOT / self.state.keypoints_var.get(),
            calib_path=ROOT / self.state.calib_var.get(),
            pose_data_mode=self.pose_data_mode.get(),
            pose_correction_mode=self.current_pose_correction_mode(),
            triangulation_method=self.triang_method.get(),
            max_frames=int(self.max_frames.get()) if self.max_frames.get() else None,
            frame_start=int(self.frame_start.get()) if self.frame_start.get() else None,
            frame_end=int(self.frame_end.get()) if self.frame_end.get() else None,
            smoothing_window=int(self.state.pose_filter_window_var.get()),
            outlier_threshold_ratio=float(self.state.pose_outlier_ratio_var.get()),
            lower_percentile=float(self.state.pose_p_low_var.get()),
            upper_percentile=float(self.state.pose_p_high_var.get()),
        )

    def _load_cached_preview(self, biomod_path: Path) -> tuple[np.ndarray, np.ndarray, int] | None:
        cache_path = self._preview_cache_path(biomod_path)
        if not cache_path.exists():
            return None
        metadata = self._preview_cache_metadata(biomod_path)
        if not metadata_cache_matches(cache_path, metadata):
            return None
        return load_model_preview_cache(cache_path)

    def _try_load_existing_triangulation_cache(self, pose_data, model_dir: Path):
        cache_name = (
            "triangulation_pose2sim_like_fast.npz"
            if self.triang_method.get() == "greedy"
            else "triangulation_pose2sim_like.npz"
        )
        cache_path = model_dir / cache_name
        if not cache_path.exists():
            return None
        metadata = reconstruction_cache_metadata(
            pose_data=pose_data,
            error_threshold_px=DEFAULT_REPROJECTION_THRESHOLD_PX,
            min_cameras_for_triangulation=self._model_min_cameras_for_triangulation(),
            epipolar_threshold_px=DEFAULT_EPIPOLAR_THRESHOLD_PX,
            triangulation_method=self.triang_method.get(),
            pose_data_mode=self.pose_data_mode.get(),
            pose_filter_window=int(self.state.pose_filter_window_var.get()),
            pose_outlier_threshold_ratio=float(self.state.pose_outlier_ratio_var.get()),
            pose_amplitude_lower_percentile=float(self.state.pose_p_low_var.get()),
            pose_amplitude_upper_percentile=float(self.state.pose_p_high_var.get()),
            pose_correction_mode=self.current_pose_correction_mode(),
        )
        if not metadata_cache_matches(cache_path, metadata):
            return None
        return load_reconstruction_cache(cache_path, coherence_method=DEFAULT_COHERENCE_METHOD)

    def _first_valid_preview_reconstruction(self, calibrations, pose_data, model_dir: Path):
        existing = self._try_load_existing_triangulation_cache(pose_data, model_dir)
        if existing is not None:
            for frame_idx in range(existing.points_3d.shape[0]):
                if np.any(np.isfinite(existing.points_3d[frame_idx])):
                    return existing, frame_idx
        for frame_idx in range(pose_data.frames.shape[0]):
            single_frame_pose_data = slice_pose_data(pose_data, [frame_idx])
            single_frame_reconstruction, _cache_path, _epipolar_cache_path, _triangulation_source = (
                load_or_compute_triangulation_cache(
                    output_dir=model_dir,
                    pose_data=single_frame_pose_data,
                    calibrations=calibrations,
                    coherence_method=DEFAULT_COHERENCE_METHOD,
                    triangulation_method=self.triang_method.get(),
                    reprojection_threshold_px=DEFAULT_REPROJECTION_THRESHOLD_PX,
                    min_cameras_for_triangulation=self._model_min_cameras_for_triangulation(),
                    epipolar_threshold_px=DEFAULT_EPIPOLAR_THRESHOLD_PX,
                    triangulation_workers=max(1, int(self.state.workers_var.get() or "1")),
                    pose_data_mode=self.pose_data_mode.get(),
                    pose_filter_window=int(self.state.pose_filter_window_var.get()),
                    pose_outlier_threshold_ratio=float(self.state.pose_outlier_ratio_var.get()),
                    pose_amplitude_lower_percentile=float(self.state.pose_p_low_var.get()),
                    pose_amplitude_upper_percentile=float(self.state.pose_p_high_var.get()),
                )
            )
            if np.any(np.isfinite(single_frame_reconstruction.points_3d[0])):
                return single_frame_reconstruction, 0
        return None, None

    def _set_preview_dof_controls(self, enabled: bool, values: list[str] | None = None) -> None:
        state = "readonly" if enabled else "disabled"
        self.preview_dof_box.configure(state=state)
        self.preview_dof_slider.configure(state="normal" if enabled else "disabled")
        self.preview_dof_box["values"] = values or []
        if not enabled:
            self.preview_dof_var.set("")
            self.preview_dof_value_var.set("")

    def _current_preview_dof_index(self) -> int | None:
        if self.preview_q_current is None or not self.preview_q_names:
            return None
        selected = self.preview_dof_var.get()
        if selected not in self.preview_q_names:
            return None
        return self.preview_q_names.index(selected)

    def _preview_reference_q(self) -> np.ndarray | None:
        if self.preview_model is None:
            return None
        if self.preview_pose_mode.get() == "q(t0)" and self.preview_q_t0 is not None:
            return np.asarray(self.preview_q_t0, dtype=float)
        if self.preview_q0 is not None:
            return np.asarray(self.preview_q0, dtype=float)
        return np.zeros(self.preview_model.nbQ(), dtype=float)

    def _apply_preview_pose_mode(self) -> None:
        reference_q = self._preview_reference_q()
        if reference_q is None:
            return
        self.preview_q_current = np.array(reference_q, copy=True)
        self.preview_points = self._markers_from_preview_q(self.preview_q_current)
        self.preview_segment_frames = self._segment_frames_from_preview_q(self.preview_q_current)

    def on_preview_pose_mode_changed(self) -> None:
        if self.preview_model is None:
            return
        self._apply_preview_pose_mode()
        self.on_preview_dof_selected()
        self.refresh_preview()

    def _preview_dof_unit(self, dof_name: str) -> str:
        return "m" if ":Trans" in dof_name else "rad"

    def _markers_from_preview_q(self, q_values: np.ndarray) -> np.ndarray:
        if self.preview_model is None:
            raise ValueError("No preview model is loaded.")
        points = np.full((len(COCO17), 3), np.nan)
        for marker_name, marker in zip(self.preview_marker_names, self.preview_model.markers(q_values)):
            if marker_name in KP_INDEX:
                points[KP_INDEX[marker_name], :] = marker.to_array()
        return points

    def _segment_frames_from_preview_q(self, q_values: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
        if self.preview_model is None:
            return []
        return biorbd_segment_frames_from_q(self.preview_model, q_values)

    def on_preview_dof_selected(self) -> None:
        if self._updating_dof_controls:
            return
        idx = self._current_preview_dof_index()
        if idx is None or self.preview_q_current is None:
            self.preview_dof_value_var.set("")
            return
        self._updating_dof_controls = True
        try:
            value = float(self.preview_q_current[idx])
            self.preview_dof_slider.set(value)
            self.preview_dof_value_var.set(f"{value:.3f} {self._preview_dof_unit(self.preview_q_names[idx])}")
        finally:
            self._updating_dof_controls = False

    def on_preview_dof_slider(self, raw_value: str) -> None:
        if self._updating_dof_controls:
            return
        idx = self._current_preview_dof_index()
        if idx is None or self.preview_q_current is None:
            return
        value = float(raw_value)
        self.preview_q_current[idx] = value
        self.preview_dof_value_var.set(f"{value:.3f} {self._preview_dof_unit(self.preview_q_names[idx])}")
        try:
            self.preview_points = self._markers_from_preview_q(self.preview_q_current)
            self.preview_segment_frames = self._segment_frames_from_preview_q(self.preview_q_current)
            self.refresh_preview()
        except Exception:
            pass

    def load_preview(self, use_selected_model: bool = False) -> None:
        try:
            biomod_path = self._preview_model_path(use_selected_model=use_selected_model)
            cached_preview = self._load_cached_preview(biomod_path) if biomod_path.exists() else None
            if cached_preview is not None:
                cached_q_t0, support_points_cached, preview_frame_number = cached_preview
                self.preview_support_points = support_points_cached
                self.preview_model = None
                self.preview_marker_names = []
                self.preview_q_names = []
                self.preview_q0 = None
                self.preview_q_t0 = None
                self.preview_q_current = None
                self.preview_segment_frames = []
                if biomod_path.exists():
                    import biorbd

                    model = biorbd.Model(str(biomod_path))
                    self.preview_model = model
                    self.preview_marker_names = [name.to_string() for name in model.markerNames()]
                    self.preview_q_names = biorbd_q_names(model)
                    self.preview_q0 = np.zeros(model.nbQ(), dtype=float)
                    self.preview_q_t0 = np.asarray(cached_q_t0, dtype=float)
                    if self.preview_q_t0.shape[0] != model.nbQ() or not np.all(np.isfinite(self.preview_q_t0)):
                        self.preview_q_t0 = np.zeros(model.nbQ(), dtype=float)
                    self._apply_preview_pose_mode()
                    self._set_preview_dof_controls(True, self.preview_q_names)
                    if self.preview_q_names:
                        self._updating_dof_controls = True
                        try:
                            self.preview_dof_var.set(self.preview_q_names[0])
                        finally:
                            self._updating_dof_controls = False
                        self.on_preview_dof_selected()
                    preview_kind = "model_cached"
                else:
                    self._set_preview_dof_controls(False)
                    self.preview_points = self.preview_support_points
                    preview_kind = "triangulation_cached"
                self.preview_metadata = {
                    "n_frames": "?",
                    "pose_data_mode": self.current_pose_source_label(),
                    "triangulation_method": self.triang_method.get(),
                    "preview_frame_idx": int(preview_frame_number),
                    "preview_kind": preview_kind,
                    "biomod_name": biomod_path.name if biomod_path.exists() else "",
                }
                self.refresh_preview()
                return

            calibrations, pose_data, _diagnostics = get_pose_data_with_correction(
                self.state,
                keypoints_path=ROOT / self.state.keypoints_var.get(),
                calib_path=ROOT / self.state.calib_var.get(),
                max_frames=int(self.max_frames.get()) if self.max_frames.get() else None,
                frame_start=int(self.frame_start.get()) if self.frame_start.get() else None,
                frame_end=int(self.frame_end.get()) if self.frame_end.get() else None,
                data_mode=self.pose_data_mode.get(),
                smoothing_window=int(self.state.pose_filter_window_var.get()),
                outlier_threshold_ratio=float(self.state.pose_outlier_ratio_var.get()),
                lower_percentile=float(self.state.pose_p_low_var.get()),
                upper_percentile=float(self.state.pose_p_high_var.get()),
                correction_mode=self.current_pose_correction_mode(),
            )
            reconstruction, preview_local_idx = self._first_valid_preview_reconstruction(
                calibrations, pose_data, biomod_path.parent
            )
            if reconstruction is None or preview_local_idx is None:
                raise ValueError("Aucune frame triangulable n'a ete trouvee pour le preview du modele.")
            preview_frame_number = int(reconstruction.frames[preview_local_idx])
            self.preview_support_points = reconstruction.points_3d[preview_local_idx]
            if biomod_path.exists():
                import biorbd

                model = biorbd.Model(str(biomod_path))
                self.preview_model = model
                self.preview_marker_names = [name.to_string() for name in model.markerNames()]
                self.preview_q_names = biorbd_q_names(model)
                self.preview_q0 = np.zeros(model.nbQ(), dtype=float)
                initial_state = initial_state_from_triangulation(model, reconstruction)
                self.preview_q_t0 = np.asarray(initial_state[: model.nbQ()], dtype=float)
                if self.preview_q_t0.shape[0] != model.nbQ() or not np.all(np.isfinite(self.preview_q_t0)):
                    self.preview_q_t0 = np.zeros(model.nbQ(), dtype=float)
                self._apply_preview_pose_mode()
                save_model_preview_cache(
                    self._preview_cache_path(biomod_path),
                    q_t0=self.preview_q_t0,
                    support_points=self.preview_support_points,
                    preview_frame_number=preview_frame_number,
                    metadata=self._preview_cache_metadata(biomod_path),
                )
                self._set_preview_dof_controls(True, self.preview_q_names)
                if self.preview_q_names:
                    self._updating_dof_controls = True
                    try:
                        self.preview_dof_var.set(self.preview_q_names[0])
                    finally:
                        self._updating_dof_controls = False
                    self.on_preview_dof_selected()
                preview_kind = "model"
            else:
                self.preview_model = None
                self.preview_marker_names = []
                self.preview_q_names = []
                self.preview_q0 = None
                self.preview_q_t0 = None
                self.preview_q_current = None
                self.preview_segment_frames = []
                self._set_preview_dof_controls(False)
                self.preview_points = self.preview_support_points
                preview_kind = "triangulation_only"
            self.preview_metadata = {
                "n_frames": int(pose_data.frames.shape[0]),
                "pose_data_mode": self.current_pose_source_label(),
                "triangulation_method": self.triang_method.get(),
                "preview_frame_idx": int(preview_frame_number),
                "preview_kind": preview_kind,
                "biomod_name": biomod_path.name if biomod_path.exists() else "",
            }
            self.refresh_preview()
        except Exception as exc:
            messagebox.showerror("Modèle", str(exc))

    def refresh_preview(self) -> None:
        if self.preview_points is None:
            return
        self.preview_figure.clear()
        ax = self.preview_figure.add_subplot(111, projection="3d")
        points_dict = {}
        if self.show_triangulation_var.get() and self.preview_support_points is not None:
            draw_skeleton_3d(ax, self.preview_support_points, "#b8c4d6", "Triangulation")
            points_dict["triangulation"] = self.preview_support_points[np.newaxis, :, :]
        draw_skeleton_3d(ax, self.preview_points, "#4c72b0", "Model")
        if has_segmented_back_visualization(segment_frames=self.preview_segment_frames, q_names=self.preview_q_names):
            draw_upper_back_preview(ax, self.preview_points, self.preview_segment_frames)
        points_dict["model"] = self.preview_points[np.newaxis, :, :]
        valid = self.preview_points[np.all(np.isfinite(self.preview_points), axis=1)]
        if valid.size:
            set_equal_3d_limits(ax, points_dict, 0)
        global_origin = np.zeros(3)
        draw_coordinate_system(ax, global_origin, np.eye(3), scale=0.2, alpha=1.0, prefix="G_")
        root_origin, root_rotation = compute_root_frame_from_points(self.preview_points)
        if root_origin is not None:
            draw_coordinate_system(ax, root_origin, np.eye(3), scale=0.18, alpha=0.35, prefix="g_")
        if root_origin is not None and root_rotation is not None:
            draw_coordinate_system(ax, root_origin, root_rotation, scale=0.18, alpha=1.0, prefix="R_")
        if self.show_local_frames_var.get() and self.preview_segment_frames:
            all_points = [self.preview_points]
            if self.show_triangulation_var.get() and self.preview_support_points is not None:
                all_points.append(self.preview_support_points)
            stacked = (
                np.vstack(
                    [
                        pts[np.all(np.isfinite(pts), axis=1)]
                        for pts in all_points
                        if pts is not None and np.any(np.isfinite(pts))
                    ]
                )
                if any(pts is not None and np.any(np.isfinite(pts)) for pts in all_points)
                else np.empty((0, 3))
            )
            if stacked.size:
                span = np.nanmax(stacked, axis=0) - np.nanmin(stacked, axis=0)
                local_scale = max(0.04, 0.08 * float(np.nanmax(span)))
            else:
                local_scale = 0.08
            for segment_name, segment_origin, segment_rotation in self.preview_segment_frames:
                draw_coordinate_system(
                    ax,
                    segment_origin,
                    segment_rotation,
                    scale=local_scale,
                    alpha=0.45,
                    show_labels=False,
                    line_width=1.3,
                )
                ax.text(
                    segment_origin[0],
                    segment_origin[1],
                    segment_origin[2],
                    segment_name.replace("_", "\n"),
                    color="#5b6570",
                    fontsize=6,
                    alpha=0.75,
                )
        n_frames = self.preview_metadata.get("n_frames", "?")
        pose_data_mode = self.preview_metadata.get("pose_data_mode", self.current_pose_source_label())
        triangulation_method = self.preview_metadata.get("triangulation_method", self.triang_method.get())
        preview_frame_idx = self.preview_metadata.get("preview_frame_idx", 0)
        preview_kind = self.preview_metadata.get("preview_kind", "model")
        biomod_name = self.preview_metadata.get("biomod_name", "")
        if preview_kind in ("model", "model_cached"):
            title_suffix = f"model {self.preview_pose_mode.get()}"
        elif preview_kind == "triangulation_only":
            title_suffix = "triangulation fallback"
        else:
            title_suffix = preview_kind.replace("_", " ")
        biomod_suffix = f" | {biomod_name}" if biomod_name else ""
        dof_suffix = ""
        dof_idx = self._current_preview_dof_index()
        if dof_idx is not None and self.preview_q_current is not None:
            dof_suffix = f" | {self.preview_q_names[dof_idx]}={self.preview_q_current[dof_idx]:.3f} {self._preview_dof_unit(self.preview_q_names[dof_idx])}"
        ax.set_title(
            f"Frame {preview_frame_idx} | {title_suffix}{biomod_suffix} | {pose_data_mode} | {triangulation_method} | {n_frames} frames{dof_suffix}"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend(loc="upper right", fontsize=8)
        self.preview_figure.tight_layout()
        self.preview_canvas.draw_idle()


class ProfilesTab(CommandTab):
    def __init__(self, master, state: SharedAppState):
        super().__init__(master, "Profiles")
        self.state = state
        self._updating_profile_name = False
        self._profile_model_choices: dict[str, str | None] = {"auto": None}
        self.flip_method_label_var = tk.StringVar(value=flip_method_display_name("none"))

        form = ttk.LabelFrame(self.main, text="Profils de reconstruction")
        form.pack(fill=tk.X, pady=(0, 8), before=self.output)

        self.config_path = LabeledEntry(
            form,
            "Config JSON",
            browse=True,
            on_browse_selected=self._on_profiles_path_browsed,
        )
        self.config_path.var = state.profiles_config_var
        self.config_path.entry_widget.configure(textvariable=self.config_path.var)
        self.config_path.entry_widget.bind("<Return>", lambda _event: self.load_profiles_from_json())
        self.config_path.pack(fill=tk.X, padx=8, pady=4)
        self.config_path.set_tooltip("Fichier JSON dans lequel charger ou sauvegarder les profils.")

        info = ttk.Label(
            form,
            text="Les chemins source, le FPS et les workers sont repris depuis le 1er onglet.",
        )
        info.pack(fill=tk.X, padx=8, pady=(0, 6))

        header = ttk.Frame(form)
        header.pack(fill=tk.X, padx=8, pady=4)
        self.profile_name = LabeledEntry(header, "Name", "")
        self.profile_name.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        family_label = ttk.Label(header, text="Family", width=8)
        family_label.pack(side=tk.LEFT)
        self.family = tk.StringVar(value="ekf_2d")
        family_box = ttk.Combobox(
            header,
            textvariable=self.family,
            values=["pose2sim", "triangulation", "ekf_3d", "ekf_2d"],
            width=14,
            state="readonly",
        )
        family_box.pack(side=tk.LEFT)

        self.cameras_frame = ttk.Frame(form)
        cameras_header = ttk.Frame(self.cameras_frame)
        cameras_header.pack(fill=tk.X)
        self.profile_cameras_summary = tk.StringVar(value="Cameras (n=0/0)")
        cameras_label = ttk.Label(cameras_header, textvariable=self.profile_cameras_summary, width=22, anchor="w")
        cameras_label.pack(side=tk.LEFT)
        self.profile_source_row = ttk.Frame(self.cameras_frame)
        self.profile_source_row.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        cameras_body = ttk.Frame(self.profile_source_row, width=470, height=62)
        cameras_body.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cameras_body.pack_propagate(False)
        self.profile_cameras_list = tk.Listbox(
            cameras_body,
            selectmode="extended",
            exportselection=False,
            height=5,
            width=40,
        )
        self.profile_cameras_list.pack(fill=tk.BOTH, expand=True)
        bind_extended_listbox_shortcuts(self.profile_cameras_list)

        self.pose_mode_frame = ttk.Frame(form)
        mode_label = ttk.Label(self.pose_mode_frame, text="2D mode", width=10)
        mode_label.pack(side=tk.LEFT)
        self.pose_data_mode = tk.StringVar(value="cleaned")
        pose_mode_box = ttk.Combobox(
            self.pose_mode_frame,
            textvariable=self.pose_data_mode,
            values=["raw", "annotated", "cleaned"],
            width=12,
            state="readonly",
        )
        pose_mode_box.pack(side=tk.LEFT, padx=(0, 8))
        self.pose_mode_info = ttk.Label(
            self.pose_mode_frame,
            text="clean settings follow 2D explorer",
            foreground="#5a6570",
        )
        self.pose_mode_info.pack(side=tk.LEFT, padx=(0, 12))
        stride_label = ttk.Label(self.pose_mode_frame, text="Downsampling", width=12)
        stride_label.pack(side=tk.LEFT)
        self.frame_stride = tk.StringVar(value="1")
        stride_box = ttk.Combobox(
            self.pose_mode_frame, textvariable=self.frame_stride, values=["1", "2", "3", "4"], width=4, state="readonly"
        )
        stride_box.pack(side=tk.LEFT, padx=(0, 8))
        self.common_frame = ttk.Frame(form)
        self.common_frame.pack(fill=tk.X, padx=8, pady=4)
        self.initial_rot_var = state.initial_rotation_correction_var
        initial_rot_check = ttk.Checkbutton(
            self.common_frame, text="initial-rotation-correction", variable=self.initial_rot_var
        )
        initial_rot_check.pack(side=tk.LEFT)
        reproj_threshold_label = ttk.Label(self.common_frame, text="Reproj px", width=9)
        reproj_threshold_label.pack(side=tk.LEFT, padx=(12, 0))
        self.reprojection_threshold_var = tk.StringVar(
            value=reprojection_threshold_display_value(DEFAULT_REPROJECTION_THRESHOLD_PX)
        )
        reproj_threshold_box = ttk.Combobox(
            self.common_frame,
            textvariable=self.reprojection_threshold_var,
            values=list(REPROJECTION_THRESHOLD_DISPLAY_VALUES),
            width=7,
            state="readonly",
        )
        reproj_threshold_box.pack(side=tk.LEFT, padx=(4, 0))

        self.triang_frame = ttk.Frame(form)
        triang_label = ttk.Label(self.triang_frame, text="Triangulation", width=12)
        triang_label.pack(side=tk.LEFT)
        self.triang_method = tk.StringVar(value="exhaustive")
        triang_box = ttk.Combobox(
            self.triang_frame,
            textvariable=self.triang_method,
            values=["once", "greedy", "exhaustive"],
            width=12,
            state="readonly",
        )
        triang_box.pack(side=tk.LEFT, padx=(0, 8))

        self.flip_frame = ttk.Frame(form)
        flip_method_label = ttk.Label(self.flip_frame, text="Flip left/right method", width=20)
        flip_method_label.pack(side=tk.LEFT)
        self.flip_method = tk.StringVar(value="none")
        self.flip_method_button = ttk.Menubutton(
            self.flip_frame,
            textvariable=self.flip_method_label_var,
            width=28,
            direction="below",
        )
        self.flip_method_button.pack(side=tk.LEFT, padx=(0, 8))
        self.flip_method_menu = tk.Menu(self.flip_method_button, tearoff=False)
        for method in ("none", *SUPPORTED_FLIP_METHODS):
            self.flip_method_menu.add_radiobutton(
                label=flip_method_display_name(method),
                value=method,
                variable=self.flip_method,
                command=self.on_flip_method_changed,
            )
        self.flip_method_button.configure(menu=self.flip_method_menu)

        self.ekf2d_frame = ttk.Frame(form)
        predictor_label = ttk.Label(self.ekf2d_frame, text="Predictor", width=10)
        predictor_label.pack(side=tk.LEFT)
        self.predictor = tk.StringVar(value="acc")
        predictor_box = ttk.Combobox(
            self.ekf2d_frame, textvariable=self.predictor, values=["acc", "dyn"], width=8, state="readonly"
        )
        predictor_box.pack(side=tk.LEFT, padx=(0, 8))
        self.measurement_noise = LabeledEntry(
            self.ekf2d_frame,
            "EKF2D meas",
            "1.5",
            label_width=8,
            entry_width=3,
        )
        self.measurement_noise.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.process_noise = LabeledEntry(
            self.ekf2d_frame,
            "EKF2D proc",
            "1.0",
            label_width=8,
            entry_width=3,
        )
        self.process_noise.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.upper_back_pseudo_std_deg = LabeledEntry(
            self.ekf2d_frame,
            "3D pseudo-obs",
            "10",
            label_width=11,
            entry_width=3,
        )
        self.upper_back_pseudo_std_deg.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.ekf2d_observation_frame = ttk.Frame(form)
        ekf2d_flip_method_label = ttk.Label(self.ekf2d_observation_frame, text="Flip left/right method", width=20)
        ekf2d_flip_method_label.pack(side=tk.LEFT)
        self.ekf2d_flip_method_button = ttk.Menubutton(
            self.ekf2d_observation_frame,
            textvariable=self.flip_method_label_var,
            width=28,
            direction="below",
        )
        self.ekf2d_flip_method_button.pack(side=tk.LEFT, padx=(0, 8))
        self.ekf2d_flip_method_button.configure(menu=self.flip_method_menu)
        coherence_label = ttk.Label(self.ekf2d_observation_frame, text="Coherence", width=10)
        coherence_label.pack(side=tk.LEFT)
        self.coherence_method = tk.StringVar(value=coherence_method_display_name("epipolar"))
        self.coherence_box = ttk.Combobox(
            self.ekf2d_observation_frame,
            textvariable=self.coherence_method,
            values=[
                coherence_method_display_name("epipolar"),
                coherence_method_display_name("epipolar_fast"),
                coherence_method_display_name("epipolar_framewise"),
                coherence_method_display_name("epipolar_fast_framewise"),
            ],
            width=26,
            state="readonly",
        )
        self.coherence_box.pack(side=tk.LEFT, padx=(0, 8))
        self.coherence_floor = LabeledEntry(
            self.ekf2d_observation_frame,
            "Conf floor",
            "0.35",
            label_width=9,
            entry_width=4,
        )
        self.coherence_floor.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.ekf2d_params_frame = ttk.Frame(form)
        q0_method_label = ttk.Label(self.ekf2d_params_frame, text="q0 init", width=10)
        q0_method_label.pack(side=tk.LEFT)
        self.ekf2d_initial_state_method = tk.StringVar(value="ekf_bootstrap")
        q0_method_box = ttk.Combobox(
            self.ekf2d_params_frame,
            textvariable=self.ekf2d_initial_state_method,
            values=["ekf_bootstrap", "root_pose_bootstrap", "triangulation_ik"],
            width=16,
            state="readonly",
        )
        q0_method_box.pack(side=tk.LEFT, padx=(0, 8))
        self.ekf2d_bootstrap_passes = LabeledEntry(
            self.ekf2d_params_frame,
            "Boot passes",
            "5",
            label_width=9,
            entry_width=4,
        )
        self.ekf2d_bootstrap_passes.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.upper_back_sagittal_gain = LabeledEntry(
            self.ekf2d_params_frame,
            "Back gain",
            "0.2",
            label_width=8,
            entry_width=4,
        )
        self.upper_back_sagittal_gain.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.lock_var = tk.BooleanVar(value=False)
        self.ekf2d_lock_check = ttk.Checkbutton(self.ekf2d_params_frame, text="dof_locking", variable=self.lock_var)
        self.ekf2d_lock_check.pack(side=tk.LEFT, padx=(0, 8))
        self.ekf2d_initial_frame_info = ttk.Label(
            self.ekf2d_params_frame,
            text="q0 support: first valid frame only",
            foreground="#5a6570",
        )
        self.ekf2d_initial_frame_info.pack(side=tk.LEFT, padx=(10, 0))

        self.ekf3d_frame = ttk.Frame(form)
        ekf3d_init_label = ttk.Label(self.ekf3d_frame, text="q0 init", width=10)
        ekf3d_init_label.pack(side=tk.LEFT)
        self.biorbd_kalman_init_method = tk.StringVar(value="triangulation_ik_root_translation")
        ekf3d_init_box = ttk.Combobox(
            self.ekf3d_frame,
            textvariable=self.biorbd_kalman_init_method,
            values=[
                "triangulation_ik_root_translation",
                "root_pose_zero_rest",
                "root_translation_zero_rest",
                "triangulation_ik",
                "none",
            ],
            width=24,
            state="readonly",
        )
        ekf3d_init_box.pack(side=tk.LEFT, padx=(0, 8))
        self.biorbd_noise = LabeledEntry(
            self.ekf3d_frame,
            "EKF3D noise",
            "1e-8",
            label_width=10,
            entry_width=6,
        )
        self.biorbd_noise.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.biorbd_error = LabeledEntry(
            self.ekf3d_frame,
            "EKF3D error",
            "1e-4",
            label_width=10,
            entry_width=6,
        )
        self.biorbd_error.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.ekf_model_frame = ttk.Frame(self.profile_source_row)
        model_header = ttk.Frame(self.ekf_model_frame)
        model_header.pack(fill=tk.X)
        ekf_model_label = ttk.Label(model_header, text="Existing bioMod", width=22, anchor="w")
        ekf_model_label.pack(side=tk.LEFT)
        self.profile_models_summary = tk.StringVar(value="")
        ttk.Label(model_header, textvariable=self.profile_models_summary, foreground="#4f5b66", anchor="w").pack(
            side=tk.LEFT, padx=(8, 0)
        )
        models_body = ttk.Frame(self.ekf_model_frame, width=470, height=62)
        models_body.pack(fill=tk.BOTH, pady=(4, 0), expand=True)
        models_body.pack_propagate(False)
        self.profile_models_list = tk.Listbox(
            models_body, selectmode="browse", exportselection=False, height=5, width=40
        )
        self.profile_models_list.pack(fill=tk.BOTH, expand=True)
        self.ekf_model_info_var = tk.StringVar(value="used by EKF profiles only")
        ttk.Label(self.ekf_model_frame, textvariable=self.ekf_model_info_var, foreground="#4f5b66").pack(
            side=tk.TOP,
            fill=tk.X,
            expand=True,
            anchor="w",
            pady=(4, 0),
        )
        self.ekf_model_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0))

        self.config_path.set_tooltip(
            "Fichier JSON contenant les profils de reconstruction sauvegardes. Browse charge le fichier immédiatement; appuyez sur Entrée après une modification manuelle du chemin."
        )
        self.profile_name.set_tooltip("Nom lisible du profil de reconstruction.")
        attach_tooltip(family_label, "Famille d'algorithme a configurer dans le profil.")
        attach_tooltip(family_box, "Famille d'algorithme a configurer dans le profil.")
        attach_tooltip(
            cameras_label,
            "Sous-ensemble de caméras sauvegardé dans le profil. Laisser vide pour utiliser toutes les caméras.",
        )
        attach_tooltip(
            self.profile_cameras_list,
            "Sélectionnez les caméras à utiliser pour ce profil. Le nom canonique du profil inclura ce choix.",
        )
        attach_tooltip(mode_label, "Version des donnees 2D consommee par la reconstruction.")
        attach_tooltip(pose_mode_box, "Version des donnees 2D consommee par la reconstruction.")
        attach_tooltip(
            stride_label,
            "Traite une image sur N dans la sequence 2D. Avec 120 Hz source, 1/2/3/4 correspond a 120/60/40/30 Hz effectifs.",
        )
        attach_tooltip(
            stride_box,
            "Traite une image sur N dans la sequence 2D. Le pipeline adapte alors le dt de l'EKF et l'axe temporel du bundle.",
        )
        attach_tooltip(
            initial_rot_check,
            "Active l'alignement horizontal de la racine autour de Z avant generation du modele ou extraction geometrique.",
        )
        attach_tooltip(triang_label, "Choisit la variante de triangulation 3D: once, greedy, ou exhaustive.")
        attach_tooltip(
            triang_box,
            "once: une seule DLT pondérée. greedy: suppression gloutonne. exhaustive: la plus robuste mais plus coûteuse.",
        )
        attach_tooltip(
            reproj_threshold_label,
            "Seuil de rejet par erreur de reprojection pour la triangulation. 'none' désactive ce rejet final en mode once.",
        )
        attach_tooltip(
            reproj_threshold_box,
            "Seuil de rejet par erreur de reprojection pour la triangulation. 'none' désactive ce rejet final en mode once.",
        )
        attach_tooltip(
            flip_method_label,
            "Technique de flip L/R. Choisissez 'None' pour désactiver la correction. ekf_prediction_gate agit dans l'update EKF 2D; les variantes triangulation_* coûtent nettement plus cher que les variantes épipolaires.",
        )
        attach_tooltip(
            self.flip_method_button,
            "none: pas de correction. epipolar: Sampson local frame-by-frame. epipolar_fast: distance symétrique locale. *_viterbi: mêmes coûts, mais avec décodage temporel explicite. ekf_prediction_gate: test raw vs swapped contre la projection prédite par l'EKF 2D. triangulation_once/greedy/exhaustive: validation 3D croissante en coût.",
        )
        attach_tooltip(predictor_label, "Choisit le predicteur dynamique de l'EKF 2D.")
        attach_tooltip(predictor_box, "Choisit le predicteur dynamique de l'EKF 2D.")
        attach_tooltip(
            coherence_label,
            "Pondération multivue de l'EKF 2D pendant la boucle du filtre. Les modes precomputed sont calculés sur toute la séquence; les modes framewise recalculent la cohérence à chaque frame.",
        )
        attach_tooltip(
            self.coherence_box,
            "Epipolar (precomputed): cohérence Sampson précalculée sur la séquence. Epipolar fast (precomputed): distance symétrique précalculée. Epipolar (framewise): Sampson recalculé à chaque frame. Epipolar fast (framewise): distance symétrique recalculée à chaque frame.",
        )
        attach_tooltip(self.ekf2d_lock_check, "Verrouille certains DoF pour stabiliser l'EKF 2D.")
        attach_tooltip(
            q0_method_label,
            "Methode pour trouver q0: IK 3D sur la triangulation, bootstrap EKF classique, ou bootstrap depuis une pose racine geometrique extraite des hanches/epaules.",
        )
        attach_tooltip(
            q0_method_box,
            "Methode pour trouver q0: IK 3D sur la triangulation ou corrections EKF 2D repetees en remettant qdot/qddot a zero.",
        )
        self.ekf2d_bootstrap_passes.set_tooltip(
            "Nombre de passes EKF 2D utilisées pour affiner q0 sur la première frame valide quand le bootstrap est actif."
        )
        self.upper_back_sagittal_gain.set_tooltip(
            "Fraction de la flexion moyenne des hanches utilisée comme cible douce pour le DoF sagittal du dos (UPPER_BACK:RotY ou LOWER_TRUNK:RotY selon le modèle)."
        )
        self.upper_back_pseudo_std_deg.set_tooltip(
            "Ecart-type angulaire (en degrés) de la pseudo-observation du dos. Plus petit = contrainte plus forte."
        )
        self.measurement_noise.set_tooltip(
            "Bruit de mesure de l'EKF 2D. Plus grand = moins de confiance dans les keypoints 2D."
        )
        self.process_noise.set_tooltip("Bruit du modèle de prédiction de l'EKF 2D.")
        self.coherence_floor.set_tooltip("Plancher appliqué à la cohérence avant pondération des mesures 2D.")
        attach_tooltip(
            ekf3d_init_label,
            "Initialisation de l'EKF 3D biorbd: IK sur triangulation, translation racine seule, ou pose racine géométrique complète avec le reste du corps à zéro.",
        )
        attach_tooltip(
            ekf3d_init_box,
            "root_pose_zero_rest reproduit l'idée du q0 2D amélioré: racine géométrique depuis hanches/épaules, reste du corps à q=0.",
        )
        self.biorbd_noise.set_tooltip("Bruit des marqueurs 3D pour l'EKF 3D.")
        self.biorbd_error.set_tooltip("Erreur d'état initiale du Kalman 3D.")
        attach_tooltip(
            ekf_model_label,
            "Choisit un bioMod existant pour EKF 2D/3D. 'auto' reconstruit le modèle à partir des données 2D; choisir un modèle existant évite cette étape et réduit le temps de calcul.",
        )
        attach_tooltip(
            self.profile_models_list,
            "Choisit un bioMod existant pour EKF 2D/3D. 'auto' reconstruit le modèle à partir des données 2D; choisir un modèle existant évite cette étape et réduit le temps de calcul.",
        )
        actions = ttk.Frame(form)
        actions.pack(fill=tk.X, padx=8, pady=6)
        self.add_profile_button = ttk.Button(actions, text="Add profile", command=self.add_current_profile)
        self.add_profile_button.pack(side=tk.LEFT)
        ttk.Button(actions, text="Delete profile", command=self.remove_selected_profiles).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(actions, text="Generate examples", command=self.generate_examples).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(actions, text="Generate all supported", command=self.generate_all_supported).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(actions, text="Save JSON", command=self.save_profiles_to_json).pack(side=tk.LEFT, padx=(8, 0))

        cols = ("enabled", "name", "family", "mode", "triang", "flip", "flags")
        self.profile_tree = ttk.Treeview(form, columns=cols, show="headings", height=8, selectmode="extended")
        headings = {
            "enabled": "Use",
            "name": "Name",
            "family": "Family",
            "mode": "2D mode",
            "triang": "Triang",
            "flip": "Flip",
            "flags": "Flags",
        }
        widths = {"enabled": 50, "name": 240, "family": 90, "mode": 90, "triang": 100, "flip": 150, "flags": 260}
        for col in cols:
            self.profile_tree.heading(col, text=headings[col])
            self.profile_tree.column(col, width=widths[col], anchor="w")
        self.profile_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        bind_extended_treeview_shortcuts(self.profile_tree)
        self.profile_tree.bind("<Delete>", lambda _event: self.remove_selected_profiles())
        self.profile_tree.bind("<BackSpace>", lambda _event: self.remove_selected_profiles())
        self.profile_tree.bind("<Double-1>", self.load_selected_profile_from_tree)

        self.family.trace_add("write", lambda *_args: self.update_family_controls())
        self.family.trace_add("write", lambda *_args: self.sync_profile_name())
        self.family.trace_add("write", lambda *_args: self.refresh_profile_model_choices())
        self.pose_data_mode.trace_add("write", lambda *_args: self.sync_profile_name())
        self.frame_stride.trace_add("write", lambda *_args: self.sync_profile_name())
        self.reprojection_threshold_var.trace_add("write", lambda *_args: self.sync_profile_name())
        self.triang_method.trace_add("write", lambda *_args: self.on_profile_triangulation_method_changed())
        self.coherence_method.trace_add("write", lambda *_args: self.sync_profile_name())
        self.predictor.trace_add("write", lambda *_args: self.sync_profile_name())
        self.ekf2d_initial_state_method.trace_add("write", lambda *_args: self.sync_profile_name())
        self.biorbd_kalman_init_method.trace_add("write", lambda *_args: self.sync_profile_name())
        self.ekf2d_bootstrap_passes.var.trace_add("write", lambda *_args: self.sync_profile_name())
        self.upper_back_sagittal_gain.var.trace_add("write", lambda *_args: self.sync_profile_name())
        self.upper_back_pseudo_std_deg.var.trace_add("write", lambda *_args: self.sync_profile_name())
        self.flip_method.trace_add("write", lambda *_args: self.sync_profile_name())
        self.lock_var.trace_add("write", lambda *_args: self.sync_profile_name())
        self.initial_rot_var.trace_add("write", lambda *_args: self.sync_profile_name())
        self.state.flip_improvement_ratio_var.trace_add("write", lambda *_args: self.sync_profile_name())
        self.state.flip_min_gain_px_var.trace_add("write", lambda *_args: self.sync_profile_name())
        self.state.flip_min_other_cameras_var.trace_add("write", lambda *_args: self.sync_profile_name())
        self.state.flip_restrict_to_outliers_var.trace_add("write", lambda *_args: self.sync_profile_name())
        self.state.flip_outlier_percentile_var.trace_add("write", lambda *_args: self.sync_profile_name())
        self.state.flip_outlier_floor_px_var.trace_add("write", lambda *_args: self.sync_profile_name())
        self.state.flip_temporal_weight_var.trace_add("write", lambda *_args: self.sync_profile_name())
        self.state.flip_temporal_tau_px_var.trace_add("write", lambda *_args: self.sync_profile_name())
        self.state.calib_var.trace_add("write", lambda *_args: self.refresh_profile_camera_choices())
        self.state.keypoints_var.trace_add("write", lambda *_args: self.refresh_profile_model_choices())
        self.state.output_root_var.trace_add("write", lambda *_args: self.refresh_profile_model_choices())
        self.profile_cameras_list.bind("<<ListboxSelect>>", lambda _event: self.on_profile_camera_selection_changed())
        self.profile_models_list.bind("<<ListboxSelect>>", lambda _event: self.on_profile_model_changed())
        self.state.register_profile_listener(self.refresh_profile_tree)
        self.refresh_profile_camera_choices()
        self.refresh_profile_model_choices()
        self.update_family_controls()
        self.on_flip_method_changed()
        self.sync_profile_name()
        self.refresh_profile_tree()
        self.update_add_profile_button_state()
        self.hide_command_controls()

    def on_flip_method_changed(self) -> None:
        """Sync the menu button label with the selected flip method."""

        self.flip_method_label_var.set(flip_method_display_name(self.flip_method.get()))

    def on_profile_triangulation_method_changed(self) -> None:
        if self.triang_method.get() != "once" and self.reprojection_threshold_var.get().strip().lower() == "none":
            self.reprojection_threshold_var.set(reprojection_threshold_display_value(DEFAULT_REPROJECTION_THRESHOLD_PX))
        self.sync_profile_name()

    def selected_profile_flip_method(self) -> str | None:
        """Return the selected flip method or ``None`` when flip correction is disabled."""

        method = self.flip_method.get().strip()
        return None if method == "none" else method

    def update_family_controls(self) -> None:
        for frame in [
            self.cameras_frame,
            self.pose_mode_frame,
            self.triang_frame,
            self.flip_frame,
            self.ekf2d_frame,
            self.ekf2d_observation_frame,
            self.ekf2d_params_frame,
            self.ekf3d_frame,
        ]:
            frame.pack_forget()
        family = self.family.get()
        self.cameras_frame.pack(fill=tk.X, padx=8, pady=4)
        if family in ("triangulation", "ekf_3d", "ekf_2d"):
            self.pose_mode_frame.pack(fill=tk.X, padx=8, pady=4)
        if family in ("triangulation", "ekf_3d"):
            self.triang_frame.pack(fill=tk.X, padx=8, pady=4)
        if family in ("triangulation", "ekf_3d"):
            self.flip_frame.pack(fill=tk.X, padx=8, pady=4)
        if family == "ekf_2d":
            self.ekf2d_frame.pack(fill=tk.X, padx=8, pady=4)
            self.ekf2d_observation_frame.pack(fill=tk.X, padx=8, pady=4)
            self.ekf2d_params_frame.pack(fill=tk.X, padx=8, pady=4)
        if family == "ekf_3d":
            self.ekf3d_frame.pack(fill=tk.X, padx=8, pady=4)
        self.update_upper_back_option_visibility()
        self.update_profile_model_info()
        self.update_add_profile_button_state()

    def selected_profile_camera_names(self) -> list[str] | None:
        indices = [int(index) for index in self.profile_cameras_list.curselection()]
        if not indices:
            return None
        camera_names = [str(self.profile_cameras_list.get(index)) for index in indices]
        return camera_names or None

    def profile_uses_all_cameras(self) -> bool:
        selected = self.selected_profile_camera_names() or []
        if not hasattr(self, "profile_cameras_list"):
            return False
        available = self.profile_cameras_list.size()
        return available > 0 and len(selected) == available

    def _set_profile_camera_selection(self, camera_names: list[str] | None) -> None:
        requested = set(camera_names or [])
        self.profile_cameras_list.selection_clear(0, tk.END)
        for index in range(self.profile_cameras_list.size()):
            if str(self.profile_cameras_list.get(index)) in requested:
                self.profile_cameras_list.selection_set(index)
        self.update_profile_camera_summary()

    def refresh_profile_camera_choices(self) -> None:
        selected_before = self.selected_profile_camera_names()
        camera_names: list[str] = []
        calib_raw = self.state.calib_var.get().strip()
        if calib_raw:
            try:
                camera_names = list(load_calibrations(ROOT / calib_raw).keys())
            except Exception:
                camera_names = []
        self.profile_cameras_list.delete(0, tk.END)
        for camera_name in camera_names:
            self.profile_cameras_list.insert(tk.END, camera_name)
        default_selection = selected_before if selected_before else camera_names
        self._set_profile_camera_selection(default_selection if camera_names else None)
        self.update_add_profile_button_state()
        self.sync_profile_name()

    def on_tab_activated(self) -> None:
        self.refresh_profile_model_choices()

    def refresh_profile_model_choices(self) -> None:
        selected_value = self.selected_profile_model_label()
        dataset_dir = current_dataset_dir(self.state)
        if self.family.get() == "ekf_2d":
            choices: list[tuple[str, str | None]] = []
        else:
            choices = [("auto", None)]
        biomod_paths: list[Path] = []
        models_root = current_models_dir(self.state)
        if models_root.exists():
            biomod_paths.extend(sorted(models_root.glob("**/*.bioMod")))
        if not biomod_paths:
            for model_dir in scan_model_dirs(dataset_dir):
                biomod_paths.extend(sorted(model_dir.glob("*.bioMod")))
        unique_biomods = sorted({path.resolve(): path for path in biomod_paths}.values(), key=lambda path: str(path))
        label_counts: dict[str, int] = {}
        for biomod_path in unique_biomods:
            label_counts[biomod_path.parent.name] = label_counts.get(biomod_path.parent.name, 0) + 1
        for biomod_path in unique_biomods:
            display = display_path(biomod_path)
            short_label = biomod_path.parent.name
            label = display if label_counts.get(short_label, 0) > 1 else short_label
            choices.append((label, display))
        self._profile_model_choices = dict(choices)
        self.profile_models_list.delete(0, tk.END)
        for label, _value in choices:
            self.profile_models_list.insert(tk.END, label)
        if self.family.get() == "ekf_2d":
            fallback_value = next((label for label, value in choices if value is not None), None)
        else:
            fallback_value = "auto"
        target_value = selected_value if selected_value in self._profile_model_choices else fallback_value
        self._set_profile_model_selection_by_label(target_value)
        self.update_profile_model_summary()
        self.update_upper_back_option_visibility()
        self.update_profile_model_info()
        self.update_add_profile_button_state()
        self.sync_profile_name()

    def selected_profile_model_label(self) -> str | None:
        selection = self.profile_models_list.curselection()
        if not selection:
            return None
        return str(self.profile_models_list.get(selection[0]))

    def selected_profile_model_path(self) -> str | None:
        value = self.selected_profile_model_label()
        if value is None:
            return None
        return self._profile_model_choices.get(value)

    def _set_profile_model_selection_by_label(self, label: str | None) -> None:
        self.profile_models_list.selection_clear(0, tk.END)
        if label is None:
            return
        for index in range(self.profile_models_list.size()):
            if str(self.profile_models_list.get(index)) == label:
                self.profile_models_list.selection_set(index)
                self.profile_models_list.see(index)
                break

    def _set_profile_model_selection_by_path(self, model_path: str | None) -> None:
        if model_path is None:
            self._set_profile_model_selection_by_label(None)
            return
        requested = str(model_path)
        requested_resolved = str(Path(requested).resolve())
        for label, value in self._profile_model_choices.items():
            if value is None:
                continue
            if str(value) == requested:
                self._set_profile_model_selection_by_label(label)
                return
            try:
                if str(Path(str(value)).resolve()) == requested_resolved:
                    self._set_profile_model_selection_by_label(label)
                    return
            except Exception:
                continue
        self._set_profile_model_selection_by_label(None)

    def update_profile_model_summary(self) -> None:
        selected_label = self.selected_profile_model_label()
        if selected_label is None:
            self.profile_models_summary.set("")
            return
        if selected_label == "auto":
            self.profile_models_summary.set("auto-build")
            return
        selected_path = self.selected_profile_model_path()
        if selected_path:
            self.profile_models_summary.set(Path(selected_path).stem)
            return
        self.profile_models_summary.set(str(selected_label))

    def update_profile_model_info(self) -> None:
        family = self.family.get()
        if family not in ("ekf_2d", "ekf_3d"):
            self.ekf_model_info_var.set("available models for EKF profiles")
            return
        selected_model = self.selected_profile_model_path()
        if selected_model:
            variant = infer_model_variant_from_biomod(selected_model)
            self.ekf_model_info_var.set(f"reuse existing {variant} model (faster)")
        else:
            self.ekf_model_info_var.set("auto-build single_trunk model from current 2D data (slower)")

    def selected_profile_model_variant(self) -> str:
        return infer_model_variant_from_biomod(self.selected_profile_model_path())

    def update_upper_back_option_visibility(self) -> None:
        if not hasattr(self, "upper_back_pseudo_std_deg") or not hasattr(self, "upper_back_sagittal_gain"):
            return
        if not hasattr(self, "profile_models_list"):
            return
        supports_upper_back = self.family.get() == "ekf_2d" and biomod_supports_upper_back_options(
            self.selected_profile_model_path()
        )
        upper_back_widgets = [
            (self.upper_back_pseudo_std_deg, {"side": tk.LEFT, "fill": tk.X, "expand": True}),
            (
                self.upper_back_sagittal_gain,
                {
                    "side": tk.LEFT,
                    "fill": tk.X,
                    "expand": True,
                    "padx": (0, 6),
                    "before": getattr(self, "ekf2d_lock_check", None),
                },
            ),
        ]
        for widget, pack_kwargs in upper_back_widgets:
            if not hasattr(widget, "pack_info") or not hasattr(widget, "pack_forget"):
                continue
            visible = bool(widget.winfo_manager())
            if supports_upper_back and not visible:
                pack_kwargs = {key: value for key, value in pack_kwargs.items() if value is not None}
                widget.pack(**pack_kwargs)
            elif not supports_upper_back and visible:
                widget.pack_forget()

    def on_profile_model_changed(self) -> None:
        self.update_profile_model_summary()
        self.update_upper_back_option_visibility()
        self.update_profile_model_info()
        self.update_add_profile_button_state()
        self.sync_profile_name()

    def update_profile_camera_summary(self) -> None:
        selected = self.selected_profile_camera_names()
        total_count = self.profile_cameras_list.size()
        selected_count = total_count if not selected and total_count else len(selected or [])
        self.profile_cameras_summary.set(f"Cameras (n={selected_count}/{total_count})")

    def use_current_camera_selection(self) -> None:
        self._set_profile_camera_selection(current_selected_camera_names(self.state))
        self.sync_profile_name()

    def clear_profile_camera_selection(self) -> None:
        self._set_profile_camera_selection(None)
        self.sync_profile_name()

    def on_profile_camera_selection_changed(self) -> None:
        self.update_profile_camera_summary()
        self.update_add_profile_button_state()
        self.sync_profile_name()

    def update_add_profile_button_state(self) -> None:
        if not hasattr(self, "add_profile_button") or not hasattr(self, "profile_cameras_list"):
            return
        camera_count = len(self.selected_profile_camera_names() or [])
        has_enough_cameras = camera_count >= 2
        has_required_model = True
        if self.family.get() == "ekf_2d":
            has_required_model = self.selected_profile_model_path() is not None
        button_state = "normal" if has_enough_cameras and has_required_model else "disabled"
        try:
            self.add_profile_button.configure(state=button_state)
        except Exception:
            pass

    def current_profile(self, *, include_name: bool = True) -> ReconstructionProfile:
        family = self.family.get()
        selected_model_path = self.selected_profile_model_path() if family in ("ekf_2d", "ekf_3d") else None
        if family == "ekf_2d" and not selected_model_path:
            raise ValueError("EKF 2D requires selecting an existing bioMod.")
        model_variant = (
            infer_model_variant_from_biomod(selected_model_path) if selected_model_path else DEFAULT_MODEL_VARIANT
        )
        profile = ReconstructionProfile(
            name=self.profile_name.get() if include_name else "",
            family=family,
            camera_names=None if self.profile_uses_all_cameras() else self.selected_profile_camera_names(),
            use_all_cameras=self.profile_uses_all_cameras(),
            ekf_model_path=selected_model_path,
            model_variant=model_variant if family in ("ekf_2d", "ekf_3d") else DEFAULT_MODEL_VARIANT,
            predictor=self.predictor.get() if family == "ekf_2d" else None,
            ekf2d_3d_source="first_frame_only" if family == "ekf_2d" else "full_triangulation",
            ekf2d_initial_state_method=self.ekf2d_initial_state_method.get() if family == "ekf_2d" else "ekf_bootstrap",
            ekf2d_bootstrap_passes=int(self.ekf2d_bootstrap_passes.get()) if family == "ekf_2d" else 5,
            flip=(
                bool(self.selected_profile_flip_method()) if family in ("triangulation", "ekf_2d", "ekf_3d") else False
            ),
            flip_method=(
                self.selected_profile_flip_method() or "epipolar"
                if family in ("triangulation", "ekf_2d", "ekf_3d")
                else "epipolar"
            ),
            flip_improvement_ratio=float(self.state.flip_improvement_ratio_var.get()),
            flip_min_gain_px=float(self.state.flip_min_gain_px_var.get()),
            flip_min_other_cameras=int(self.state.flip_min_other_cameras_var.get()),
            flip_restrict_to_outliers=bool(self.state.flip_restrict_to_outliers_var.get()),
            flip_outlier_percentile=float(self.state.flip_outlier_percentile_var.get()),
            flip_outlier_floor_px=float(self.state.flip_outlier_floor_px_var.get()),
            flip_temporal_weight=float(self.state.flip_temporal_weight_var.get()),
            flip_temporal_tau_px=float(self.state.flip_temporal_tau_px_var.get()),
            dof_locking=self.lock_var.get() if family == "ekf_2d" else False,
            initial_rotation_correction=self.initial_rot_var.get(),
            pose_data_mode=self.pose_data_mode.get() if family in ("triangulation", "ekf_3d", "ekf_2d") else "cleaned",
            frame_stride=int(self.frame_stride.get()) if family in ("triangulation", "ekf_3d", "ekf_2d") else 1,
            triangulation_method=(
                self.triang_method.get() if family in ("triangulation", "ekf_3d", "ekf_2d") else "exhaustive"
            ),
            reprojection_threshold_px=(
                reprojection_threshold_from_display_value(self.reprojection_threshold_var.get())
                if family in ("triangulation", "ekf_3d", "ekf_2d")
                else DEFAULT_REPROJECTION_THRESHOLD_PX
            ),
            coherence_method=(
                coherence_method_from_display_name(self.coherence_method.get()) if family == "ekf_2d" else "epipolar"
            ),
            no_root_unwrap=True,
            root_unwrap_mode="off",
            biorbd_kalman_noise_factor=float(self.biorbd_noise.get()),
            biorbd_kalman_error_factor=float(self.biorbd_error.get()),
            biorbd_kalman_init_method=(
                self.biorbd_kalman_init_method.get() if family == "ekf_3d" else "triangulation_ik_root_translation"
            ),
            measurement_noise_scale=float(self.measurement_noise.get()),
            process_noise_scale=float(self.process_noise.get()),
            coherence_confidence_floor=float(self.coherence_floor.get()),
            upper_back_sagittal_gain=(
                float(self.upper_back_sagittal_gain.get()) if hasattr(self, "upper_back_sagittal_gain") else 0.2
            ),
            upper_back_pseudo_std_deg=(
                float(self.upper_back_pseudo_std_deg.get()) if hasattr(self, "upper_back_pseudo_std_deg") else 10.0
            ),
            pose_filter_window=int(self.state.pose_filter_window_var.get()),
            pose_outlier_threshold_ratio=float(self.state.pose_outlier_ratio_var.get()),
            pose_amplitude_lower_percentile=float(self.state.pose_p_low_var.get()),
            pose_amplitude_upper_percentile=float(self.state.pose_p_high_var.get()),
        )
        return validate_profile(profile)

    def sync_profile_name(self) -> None:
        if self._updating_profile_name:
            return
        try:
            canonical_name = canonical_profile_name(self.current_profile(include_name=False))
        except Exception:
            return
        self._updating_profile_name = True
        try:
            self.profile_name.var.set(canonical_name)
        finally:
            self._updating_profile_name = False

    def refresh_profile_tree(self) -> None:
        for item in self.profile_tree.get_children():
            self.profile_tree.delete(item)
        for idx, profile in enumerate(self.state.profiles):
            flags = []
            if profile.predictor:
                flags.append(profile.predictor)
            if getattr(profile, "ekf2d_3d_source", "full_triangulation") == "first_frame_only":
                flags.append("bootstrap1")
            if getattr(profile, "ekf2d_initial_state_method", "ekf_bootstrap") == "triangulation_ik":
                flags.append("ikq0")
            elif getattr(profile, "ekf2d_initial_state_method", "ekf_bootstrap") == "root_pose_bootstrap":
                flags.append("rootq0")
            elif int(getattr(profile, "ekf2d_bootstrap_passes", 5)) != 5:
                flags.append(f"boot{int(getattr(profile, 'ekf2d_bootstrap_passes', 5))}")
            if abs(float(getattr(profile, "upper_back_sagittal_gain", 0.2)) - 0.2) > 1e-9:
                flags.append(f"ubg:{float(getattr(profile, 'upper_back_sagittal_gain', 0.2)):.2f}")
            reprojection_threshold_px = getattr(profile, "reprojection_threshold_px", DEFAULT_REPROJECTION_THRESHOLD_PX)
            if reprojection_threshold_px is None:
                flags.append("tau:none")
            elif abs(float(reprojection_threshold_px) - float(DEFAULT_REPROJECTION_THRESHOLD_PX)) > 1e-9:
                flags.append(f"tau:{float(reprojection_threshold_px):g}")
            if profile.family == "ekf_3d":
                init_method = getattr(profile, "biorbd_kalman_init_method", "triangulation_ik_root_translation")
                if init_method == "triangulation_ik":
                    flags.append("ikq0")
                elif init_method == "root_translation_zero_rest":
                    flags.append("roottransq0")
                elif init_method == "root_pose_zero_rest":
                    flags.append("rootq0")
            if getattr(profile, "ekf_model_path", None):
                flags.append(f"mdl:{Path(str(profile.ekf_model_path)).stem}")
            elif getattr(profile, "model_variant", DEFAULT_MODEL_VARIANT) != DEFAULT_MODEL_VARIANT:
                flags.append(f"mdl:{getattr(profile, 'model_variant', DEFAULT_MODEL_VARIANT)}")
            if profile.initial_rotation_correction:
                flags.append("rotfix")
            if profile.flip:
                if getattr(profile, "flip_method", "epipolar") != "epipolar":
                    flags.append(f"flip:{getattr(profile, 'flip_method', 'epipolar')}")
                flags.append("flip")
            if profile.dof_locking:
                flags.append("lock")
            if getattr(profile, "use_all_cameras", False):
                flags.append("cams[all]")
            elif getattr(profile, "camera_names", None):
                flags.append(f"cams[{format_camera_names(profile.camera_names)}]")
            if int(getattr(profile, "frame_stride", 1)) != 1:
                flags.append(f"1/{int(getattr(profile, 'frame_stride', 1))}")
            if profile.family in ("triangulation", "ekf_3d", "ekf_2d") and profile.pose_data_mode != "cleaned":
                flags.append(profile.pose_data_mode)
            mode_value = profile.pose_data_mode if profile.family in ("triangulation", "ekf_3d", "ekf_2d") else "-"
            triang_value = (
                profile.triangulation_method if profile.family in ("triangulation", "ekf_3d", "ekf_2d") else "-"
            )
            flip_value = (
                flip_method_display_name(getattr(profile, "flip_method", "epipolar")) if profile.flip else "None"
            )
            self.profile_tree.insert(
                "",
                "end",
                iid=str(idx),
                values=(
                    "yes" if profile.enabled else "no",
                    profile.name,
                    profile.family,
                    mode_value,
                    triang_value,
                    flip_value,
                    ",".join(flags),
                ),
            )

    def load_selected_profile_from_tree(self, event=None) -> str | None:
        row_id = None
        if event is not None:
            try:
                row_id = self.profile_tree.identify_row(event.y)
            except Exception:
                row_id = None
        selected = self.profile_tree.selection()
        if row_id:
            try:
                self.profile_tree.selection_set((row_id,))
                self.profile_tree.focus(row_id)
            except Exception:
                pass
        elif selected:
            row_id = selected[0]
        if not row_id:
            return None
        try:
            profile = self.state.profiles[int(row_id)]
        except Exception:
            return None
        self.load_profile_into_form(profile)
        return "break"

    def load_profile_into_form(self, profile: ReconstructionProfile) -> None:
        self._updating_profile_name = True
        try:
            self.family.set(profile.family)
            self.refresh_profile_camera_choices()
            self.refresh_profile_model_choices()

            self.pose_data_mode.set(profile.pose_data_mode)
            self.frame_stride.set(str(int(getattr(profile, "frame_stride", 1))))
            self.triang_method.set(getattr(profile, "triangulation_method", "exhaustive"))
            self.reprojection_threshold_var.set(
                reprojection_threshold_display_value(
                    getattr(profile, "reprojection_threshold_px", DEFAULT_REPROJECTION_THRESHOLD_PX)
                )
            )
            self.predictor.set(getattr(profile, "predictor", "acc") or "acc")
            self.coherence_method.set(coherence_method_display_name(getattr(profile, "coherence_method", "epipolar")))
            self.ekf2d_initial_state_method.set(getattr(profile, "ekf2d_initial_state_method", "ekf_bootstrap"))
            self.ekf2d_bootstrap_passes.var.set(str(int(getattr(profile, "ekf2d_bootstrap_passes", 5))))
            self.upper_back_sagittal_gain.var.set(f"{float(getattr(profile, 'upper_back_sagittal_gain', 0.2)):g}")
            self.upper_back_pseudo_std_deg.var.set(f"{float(getattr(profile, 'upper_back_pseudo_std_deg', 10.0)):g}")
            self.flip_method.set(getattr(profile, "flip_method", "epipolar") if profile.flip else "none")
            self.on_flip_method_changed()
            self.lock_var.set(bool(getattr(profile, "dof_locking", False)))
            self.initial_rot_var.set(bool(getattr(profile, "initial_rotation_correction", False)))
            self.state.pose_filter_window_var.set(str(int(getattr(profile, "pose_filter_window", 9))))
            self.state.pose_outlier_ratio_var.set(f"{float(getattr(profile, 'pose_outlier_threshold_ratio', 0.10)):g}")
            self.state.pose_p_low_var.set(f"{float(getattr(profile, 'pose_amplitude_lower_percentile', 5.0)):g}")
            self.state.pose_p_high_var.set(f"{float(getattr(profile, 'pose_amplitude_upper_percentile', 95.0)):g}")
            self.state.flip_improvement_ratio_var.set(f"{float(getattr(profile, 'flip_improvement_ratio', 0.7)):g}")
            self.state.flip_min_gain_px_var.set(f"{float(getattr(profile, 'flip_min_gain_px', 3.0)):g}")
            self.state.flip_min_other_cameras_var.set(str(int(getattr(profile, "flip_min_other_cameras", 2))))
            self.state.flip_restrict_to_outliers_var.set(bool(getattr(profile, "flip_restrict_to_outliers", True)))
            self.state.flip_outlier_percentile_var.set(f"{float(getattr(profile, 'flip_outlier_percentile', 85.0)):g}")
            self.state.flip_outlier_floor_px_var.set(f"{float(getattr(profile, 'flip_outlier_floor_px', 5.0)):g}")
            self.state.flip_temporal_weight_var.set(f"{float(getattr(profile, 'flip_temporal_weight', 0.35)):g}")
            self.state.flip_temporal_tau_px_var.set(f"{float(getattr(profile, 'flip_temporal_tau_px', 20.0)):g}")
            self.biorbd_noise.var.set(f"{float(getattr(profile, 'biorbd_kalman_noise_factor', 1e-8)):g}")
            self.biorbd_error.var.set(f"{float(getattr(profile, 'biorbd_kalman_error_factor', 1e-4)):g}")
            self.biorbd_kalman_init_method.set(
                getattr(profile, "biorbd_kalman_init_method", "triangulation_ik_root_translation")
            )
            self.measurement_noise.var.set(f"{float(getattr(profile, 'measurement_noise_scale', 1.5)):g}")
            self.process_noise.var.set(f"{float(getattr(profile, 'process_noise_scale', 1.0)):g}")
            self.coherence_floor.var.set(f"{float(getattr(profile, 'coherence_confidence_floor', 0.35)):g}")

            if getattr(profile, "use_all_cameras", False) or getattr(profile, "camera_names", None) is None:
                self._set_profile_camera_selection(
                    [str(self.profile_cameras_list.get(i)) for i in range(self.profile_cameras_list.size())]
                )
            else:
                self._set_profile_camera_selection(list(profile.camera_names or []))

            selected_model_path = getattr(profile, "ekf_model_path", None)
            if selected_model_path:
                self._set_profile_model_selection_by_path(str(selected_model_path))
            elif profile.family != "ekf_2d":
                self._set_profile_model_selection_by_label("auto")
            else:
                self._set_profile_model_selection_by_label(None)
        finally:
            self._updating_profile_name = False

        self.profile_name.var.set(profile.name)
        self.update_family_controls()
        self.update_profile_camera_summary()
        self.update_profile_model_summary()
        self.update_upper_back_option_visibility()
        self.update_profile_model_info()
        self.update_add_profile_button_state()

    def add_current_profile(self) -> None:
        try:
            profile = self.current_profile()
        except Exception as exc:
            messagebox.showerror("Profiles", str(exc))
            return
        profiles = [existing for existing in self.state.profiles if existing.name != profile.name]
        profiles.append(profile)
        profiles.sort(key=lambda item: item.name)
        self.state.set_profiles(profiles)

    def remove_selected_profiles(self) -> None:
        selected = self.profile_tree.selection()
        if not selected:
            return
        indices = sorted((int(item) for item in selected), reverse=True)
        profiles = list(self.state.profiles)
        for idx in indices:
            if 0 <= idx < len(profiles):
                del profiles[idx]
        self.state.set_profiles(profiles)

    def generate_examples(self) -> None:
        self.state.set_profiles(example_profiles())
        synchronize_profiles_initial_rotation_correction(self.state)

    def generate_all_supported(self) -> None:
        self.state.set_profiles(generate_supported_profiles())
        synchronize_profiles_initial_rotation_correction(self.state)

    def load_profiles_from_json(self) -> None:
        try:
            self.state.set_profiles(load_profiles_json(ROOT / self.config_path.get()))
            synchronize_profiles_initial_rotation_correction(self.state)
        except Exception as exc:
            messagebox.showerror("Profiles", str(exc))

    def _on_profiles_path_browsed(self, _path: str) -> None:
        self.load_profiles_from_json()

    def save_profiles_to_json(self) -> None:
        try:
            synchronize_profiles_initial_rotation_correction(self.state)
            save_profiles_json(ROOT / self.config_path.get(), self.state.profiles)
        except Exception as exc:
            messagebox.showerror("Profiles", str(exc))

    def selected_profiles(self) -> list[ReconstructionProfile]:
        selected = self.profile_tree.selection()
        if not selected:
            profiles = [profile for profile in self.state.profiles if profile.enabled]
            return append_default_pose2sim_profile(profiles, self.state.profiles, self.state.pose2sim_trc_var.get())
        return [self.state.profiles[int(item)] for item in selected]

    def build_command(self) -> list[str]:
        selected_profiles = self.selected_profiles()
        runtime_config_path = write_runtime_profiles_config(self.state)
        cmd = [
            sys.executable,
            "run_reconstruction_profiles.py",
            "--config",
            display_path(runtime_config_path),
            "--output-root",
            display_path(normalize_output_root(self.state.output_root_var.get())),
            "--dataset-name",
            current_dataset_name(self.state),
            "--calib",
            self.state.calib_var.get(),
            "--keypoints",
            self.state.keypoints_var.get(),
            "--fps",
            self.state.fps_var.get(),
            "--triangulation-workers",
            self.state.workers_var.get(),
        ]
        selected_cameras = current_selected_camera_names(self.state)
        if selected_cameras:
            cmd.extend(["--camera-names", ",".join(selected_cameras)])
        if self.state.pose2sim_trc_var.get().strip():
            cmd.extend(["--trc-file", self.state.pose2sim_trc_var.get()])
        annotation_var = getattr(self.state, "annotation_path_var", None)
        annotations_path = "" if annotation_var is None else str(annotation_var.get()).strip()
        if annotations_path and any(profile.pose_data_mode == "annotated" for profile in selected_profiles):
            cmd.extend(["--annotations-path", annotations_path])
        for profile in selected_profiles:
            cmd.extend(["--profile", profile.name])
        return cmd

    def on_command_success(self) -> None:
        self.state.notify_reconstructions_updated()


class BatchTab(CommandTab):
    def __init__(self, master, state: SharedAppState):
        super().__init__(master, "Batch", show_command_preview=False, show_output=True)
        self.state = state
        self.batch_profiles: list[ReconstructionProfile] = []

        form = ttk.LabelFrame(self.main, text="Batch reconstruction runner")
        form.pack(fill=tk.BOTH, expand=False, pady=(0, 8), before=self.output)

        paths_box = ttk.Frame(form)
        paths_box.pack(fill=tk.X, padx=8, pady=(8, 4))

        self.keypoints_glob_entry = LabeledEntry(
            paths_box,
            "Source keypoints",
            DEFAULT_KEYPOINTS_GLOB,
            label_width=18,
            entry_width=64,
        )
        self.keypoints_glob_entry.pack(fill=tk.X, pady=(0, 4))
        self.keypoints_glob_entry.set_tooltip("Glob or explicit path for source keypoints JSON files.")

        self.config_path = LabeledEntry(
            paths_box,
            "Profiles JSON",
            state.profiles_config_var.get(),
            browse=True,
            label_width=18,
            entry_width=64,
            filetypes=(("Profiles JSON", "*_profiles.json"), ("JSON files", "*.json"), ("All files", "*.*")),
            on_browse_selected=lambda _value: self.load_profiles_from_config(),
        )
        self.config_path.pack(fill=tk.X, pady=(0, 4))

        self.excel_output_entry = LabeledEntry(
            paths_box,
            "Excel summary",
            str(DEFAULT_EXCEL_OUTPUT),
            browse=True,
            label_width=18,
            entry_width=64,
            filetypes=(("Excel workbooks", "*.xlsx"), ("All files", "*.*")),
        )
        self.excel_output_entry.pack(fill=tk.X, pady=(0, 4))

        batch_row = ttk.Frame(paths_box)
        batch_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(batch_row, text="Batch name", width=18).pack(side=tk.LEFT, padx=(0, 6))
        self.batch_name_entry = ttk.Entry(batch_row, width=32)
        self.batch_name_entry.pack(side=tk.LEFT)
        self.continue_on_error_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(batch_row, text="Continue on error", variable=self.continue_on_error_var).pack(
            side=tk.LEFT, padx=(12, 0)
        )
        self.export_only_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(batch_row, text="Export only", variable=self.export_only_var).pack(side=tk.LEFT, padx=(12, 0))

        controls = ttk.Frame(form)
        controls.pack(fill=tk.X, padx=8, pady=(0, 4))
        ttk.Button(controls, text="Scan keypoints", command=self.scan_keypoints_files).pack(side=tk.LEFT)
        ttk.Button(controls, text="Reload profiles", command=self.load_profiles_from_config).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        body = ttk.Panedwindow(form, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        keypoints_box = ttk.LabelFrame(body, text="Datasets")
        profiles_box = ttk.LabelFrame(body, text="Profiles")
        body.add(keypoints_box, weight=3)
        body.add(profiles_box, weight=2)

        self.keypoints_tree = ttk.Treeview(
            keypoints_box,
            columns=("dataset", "keypoints", "annotations", "trc"),
            show="headings",
            height=8,
            selectmode="extended",
        )
        for col, label, width in [
            ("dataset", "Dataset", 130),
            ("keypoints", "Keypoints", 280),
            ("annotations", "Annotations", 180),
            ("trc", "TRC", 150),
        ]:
            self.keypoints_tree.heading(col, text=label)
            self.keypoints_tree.column(col, width=width, anchor="w")
        self.keypoints_tree.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        bind_extended_treeview_shortcuts(self.keypoints_tree)

        self.batch_profile_tree = ttk.Treeview(
            profiles_box,
            columns=("name", "family", "mode"),
            show="headings",
            height=8,
            selectmode="extended",
        )
        for col, label, width in [
            ("name", "Name", 220),
            ("family", "Family", 90),
            ("mode", "2D mode", 90),
        ]:
            self.batch_profile_tree.heading(col, text=label)
            self.batch_profile_tree.column(col, width=width, anchor="w")
        self.batch_profile_tree.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        bind_extended_treeview_shortcuts(self.batch_profile_tree)

        attach_tooltip(self.keypoints_tree, "Select one or more datasets for the batch run. Empty selection means all.")
        attach_tooltip(self.batch_profile_tree, "Select one or more profiles. Empty selection means all profiles.")

        self.hide_preview_copy_buttons()
        self.config_path.entry_widget.bind("<Return>", lambda _event: self.load_profiles_from_config())
        self.keypoints_glob_entry.entry_widget.bind("<Return>", lambda _event: self.scan_keypoints_files())
        self.load_profiles_from_config()
        self.scan_keypoints_files()

    def load_profiles_from_config(self) -> None:
        raw_path = self.config_path.get()
        config_path = Path(raw_path)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        profiles: list[ReconstructionProfile] = []
        if config_path.exists():
            try:
                profiles = load_profiles_json(config_path)
            except Exception as exc:
                messagebox.showerror("Batch", f"Unable to load profiles:\n{config_path}\n\n{exc}")
                profiles = []
        self.batch_profiles = profiles
        self.refresh_batch_profile_tree()

    def refresh_batch_profile_tree(self) -> None:
        previous = set(self.batch_profile_tree.selection())
        for item in self.batch_profile_tree.get_children():
            self.batch_profile_tree.delete(item)
        for idx, profile in enumerate(self.batch_profiles):
            iid = str(idx)
            self.batch_profile_tree.insert(
                "", "end", iid=iid, values=(profile.name, profile.family, profile.pose_data_mode)
            )
            if iid in previous:
                self.batch_profile_tree.selection_add(iid)

    def scan_keypoints_files(self) -> None:
        raw_pattern = self.keypoints_glob_entry.get().strip()
        patterns = [part.strip() for part in raw_pattern.split(";") if part.strip()] or [DEFAULT_KEYPOINTS_GLOB]
        discovered = batch_discover_keypoints_files(patterns, root=ROOT)
        previous = set(self.keypoints_tree.selection())
        for item in self.keypoints_tree.get_children():
            self.keypoints_tree.delete(item)
        for keypoints_path in discovered:
            dataset_name = infer_dataset_name(keypoints_path=keypoints_path)
            iid = str(keypoints_path)
            annotations_path = batch_infer_annotations_for_keypoints(keypoints_path)
            trc_path = batch_infer_pose2sim_trc_for_keypoints(keypoints_path)
            self.keypoints_tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    dataset_name,
                    display_path(keypoints_path),
                    "-" if annotations_path is None else display_path(annotations_path),
                    "-" if trc_path is None else display_path(trc_path),
                ),
            )
            if iid in previous:
                self.keypoints_tree.selection_add(iid)

    def selected_batch_keypoints_paths(self) -> list[Path]:
        selected = list(self.keypoints_tree.selection())
        if selected:
            return [Path(item) for item in selected]
        return [Path(item) for item in self.keypoints_tree.get_children("")]

    def selected_batch_profiles(self) -> list[ReconstructionProfile]:
        selected = list(self.batch_profile_tree.selection())
        if selected:
            return [self.batch_profiles[int(item)] for item in selected]
        return list(self.batch_profiles)

    def build_command(self) -> list[str]:
        cmd = [
            sys.executable,
            "batch_run.py",
            "--config",
            self.config_path.get(),
            "--output-root",
            self.state.output_root_var.get(),
            "--calib",
            self.state.calib_var.get(),
            "--fps",
            self.state.fps_var.get(),
            "--triangulation-workers",
            self.state.workers_var.get(),
            "--excel-output",
            self.excel_output_entry.get(),
        ]
        batch_name = self.batch_name_entry.get().strip()
        if batch_name:
            cmd.extend(["--batch-name", batch_name])
        if self.continue_on_error_var.get():
            cmd.append("--continue-on-error")
        if self.export_only_var.get():
            cmd.append("--export-only")

        selected_keypoints = self.selected_batch_keypoints_paths()
        if selected_keypoints:
            for keypoints_path in selected_keypoints:
                cmd.extend(["--keypoints-glob", display_path(keypoints_path)])
        else:
            cmd.extend(["--keypoints-glob", self.keypoints_glob_entry.get()])

        for profile in self.selected_batch_profiles():
            cmd.extend(["--profile", profile.name])
        return cmd

    def on_command_success(self) -> None:
        self.state.notify_reconstructions_updated()


class ReconstructionsTab(CommandTab):
    def __init__(self, master, state: SharedAppState):
        super().__init__(master, "Reconstructions", show_command_preview=False, show_output=False)
        self.state = state
        self.status_summaries: dict[str, dict[str, object]] = {}
        self.uses_shared_reconstruction_panel = True
        self.shared_reconstruction_selectmode = "browse"

        form = ttk.LabelFrame(self.main, text="Lancer les reconstructions depuis les profils")
        form.pack(fill=tk.X, pady=(0, 8))

        controls = ttk.Frame(form)
        controls.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(controls, text="Refresh profiles", command=self.refresh_profile_tree).pack(side=tk.LEFT)
        ttk.Button(controls, text="Refresh caches", command=self.refresh_status_rows).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(controls, text="Clear reconstructions", command=self.clear_reconstructions).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        self.export_trc_button = ttk.Button(controls, text="Export TRC from q", command=self.export_selected_trc_from_q)
        self.export_trc_button.pack(side=tk.LEFT, padx=(8, 0))
        self.export_pseudo_root_button = ttk.Button(
            controls,
            text="Export pseudo q root",
            command=self.export_selected_pseudo_root_from_points,
        )
        self.export_pseudo_root_button.pack(side=tk.LEFT, padx=(8, 0))

        self.profile_tree = ttk.Treeview(
            form,
            columns=("name", "family", "mode", "flip", "flags"),
            show="headings",
            height=6,
            selectmode="extended",
        )
        for col, label, width in [
            ("name", "Name", 260),
            ("family", "Family", 90),
            ("mode", "2D mode", 90),
            ("flip", "Flip", 150),
            ("flags", "Flags", 300),
        ]:
            self.profile_tree.heading(col, text=label)
            self.profile_tree.column(col, width=width, anchor="w")
        self.profile_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        bind_extended_treeview_shortcuts(self.profile_tree)
        attach_tooltip(self.profile_tree, "Selectionnez les profils a executer pour le dataset courant.")
        attach_tooltip(
            self.export_trc_button,
            "Reconstruit les marqueurs du modèle depuis q pour la reconstruction sélectionnée et écrit un fichier TRC dans son dossier.",
        )
        attach_tooltip(
            self.export_pseudo_root_button,
            "Exporte les pseudo q de la racine obtenus géométriquement depuis les marqueurs du tronc de la reconstruction sélectionnée.",
        )

        timing_box = ttk.LabelFrame(self.main, text="Durées détaillées de la reconstruction sélectionnée")
        timing_box.pack(fill=tk.BOTH, expand=False, pady=(0, 8))
        self.timing_details = ScrolledText(timing_box, height=10, wrap=tk.WORD)
        self.timing_details.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.timing_details.insert("1.0", "Select a reconstruction to inspect timing details.")
        self.timing_details.configure(state=tk.DISABLED)

        self.state.register_profile_listener(self.refresh_profile_tree)
        self.state.register_reconstruction_listener(self.refresh_status_rows)
        self.state.calib_var.trace_add("write", lambda *_args: self.refresh_status_rows())
        self.state.keypoints_var.trace_add("write", lambda *_args: self.refresh_status_rows())
        self.state.output_root_var.trace_add("write", lambda *_args: self.refresh_status_rows())
        self.refresh_profile_tree()
        self.refresh_status_rows()
        self.hide_preview_copy_buttons()

    def configure_shared_reconstruction_panel(self, panel: SharedReconstructionPanel) -> None:
        panel.configure_for_consumer(
            title="Reconstructions",
            refresh_callback=self.refresh_status_rows,
            selection_callback=self.refresh_timing_details,
            selectmode=self.shared_reconstruction_selectmode,
        )
        self.refresh_status_rows()

    def _publish_reconstruction_rows(self, rows: list[dict[str, object]], defaults: list[str]) -> None:
        panel = self.state.shared_reconstruction_panel
        if panel is not None and self.state.active_reconstruction_consumer is self:
            panel.set_rows(rows, defaults)

    def _selected_reconstruction_dir(self) -> Path | None:
        selected = list(self.state.shared_reconstruction_selection)
        if not selected:
            return None
        recon_name = selected[-1]
        recon_dir = current_reconstructions_dir(self.state) / recon_name
        return recon_dir if recon_dir.exists() else None

    def refresh_profile_tree(self) -> None:
        for item in self.profile_tree.get_children():
            self.profile_tree.delete(item)
        for idx, profile in enumerate(self.state.profiles):
            flags = []
            if profile.predictor:
                flags.append(profile.predictor)
            if getattr(profile, "ekf2d_3d_source", "full_triangulation") == "first_frame_only":
                flags.append("bootstrap1")
            if getattr(profile, "ekf2d_initial_state_method", "ekf_bootstrap") == "triangulation_ik":
                flags.append("ikq0")
            elif int(getattr(profile, "ekf2d_bootstrap_passes", 5)) != 5:
                flags.append(f"boot{int(getattr(profile, 'ekf2d_bootstrap_passes', 5))}")
            if profile.flip:
                flags.append("flip")
            if profile.dof_locking:
                flags.append("lock")
            if profile.initial_rotation_correction:
                flags.append("rotfix")
            if int(getattr(profile, "frame_stride", 1)) != 1:
                flags.append(f"1/{int(getattr(profile, 'frame_stride', 1))}")
            flip_value = (
                flip_method_display_name(getattr(profile, "flip_method", "epipolar")) if profile.flip else "None"
            )
            self.profile_tree.insert(
                "",
                "end",
                iid=str(idx),
                values=(profile.name, profile.family, profile.pose_data_mode, flip_value, ",".join(flags)),
            )

    def refresh_status_rows(self) -> None:
        self.status_summaries = {}
        dataset_dir = current_dataset_dir(self.state)
        rows: list[dict[str, object]] = []
        for recon_dir in reconstruction_dirs_for_path(dataset_dir):
            summary = load_bundle_summary(recon_dir)
            if not summary:
                continue
            reproj = summary.get("reprojection_px", {})
            latest = summary.get("is_latest_family_version")
            compute_s = objective_total_seconds(summary)
            model_s = model_compute_seconds(summary)
            reconstruction_s = reconstruction_run_seconds(summary)
            recon_name = str(summary.get("name", recon_dir.name))
            self.status_summaries[recon_name] = summary
            rows.append(
                {
                    "name": recon_name,
                    "label": recon_name,
                    "family": summary.get("family", "-"),
                    "frames": summary.get("n_frames", "-"),
                    "reproj_mean": reproj.get("mean"),
                    "path": str(recon_dir),
                    "compute_s": reconstruction_s if reconstruction_s is not None else compute_s,
                    "model_compute_s": model_s,
                    "reproj_std": reproj.get("std"),
                    "is_latest": latest,
                }
            )
        default_names = [rows[0]["name"]] if rows else []
        self._publish_reconstruction_rows(rows, default_names)
        self.refresh_timing_details()

    def export_selected_trc_from_q(self) -> None:
        """Export the selected q-based reconstruction as a TRC marker trajectory."""

        recon_dir = self._selected_reconstruction_dir()
        if recon_dir is None:
            messagebox.showinfo("Export TRC from q", "Select one reconstruction first.")
            return
        bundle_path = recon_dir / "reconstruction_bundle.npz"
        if not bundle_path.exists():
            messagebox.showerror("Export TRC from q", f"Bundle not found:\n{bundle_path}")
            return

        data = np.load(bundle_path, allow_pickle=True)
        if "q" not in data:
            messagebox.showinfo(
                "Export TRC from q", "The selected reconstruction does not contain generalized coordinates q."
            )
            return

        q = np.asarray(data["q"], dtype=float)
        q_names = np.asarray(data["q_names"], dtype=object) if "q_names" in data else np.empty(0, dtype=object)
        frames = np.asarray(data["frames"], dtype=int) if "frames" in data else np.arange(q.shape[0], dtype=int)
        time_s = (
            np.asarray(data["time_s"], dtype=float)
            if "time_s" in data
            else np.arange(q.shape[0], dtype=float) / max(float(self.state.fps_var.get()), 1.0)
        )
        biomod_path = resolve_reconstruction_biomod(
            current_dataset_dir(self.state), self.status_summaries.get(recon_dir.name, {}).get("name", recon_dir.name)
        )
        if biomod_path is None or not biomod_path.exists():
            messagebox.showerror("Export TRC from q", f"No bioMod found for:\n{recon_dir.name}")
            return

        import biorbd

        model = biorbd.Model(str(biomod_path))
        marker_names = [name.to_string() for name in model.markerNames()]
        marker_points = np.full((q.shape[0], len(marker_names), 3), np.nan, dtype=float)
        for frame_idx, q_values in enumerate(q):
            for marker_idx, marker in enumerate(model.markers(q_values)):
                marker_points[frame_idx, marker_idx] = marker.to_array()

        if time_s.shape[0] > 1 and np.all(np.isfinite(time_s)):
            dt = np.diff(time_s)
            positive_dt = dt[np.isfinite(dt) & (dt > 0)]
            data_rate = float(1.0 / np.median(positive_dt)) if positive_dt.size else float(self.state.fps_var.get())
        else:
            data_rate = float(self.state.fps_var.get())

        output_path = recon_dir / f"{recon_dir.name}_markers_from_q.trc"
        write_trc_file(output_path, marker_names, marker_points, frames, time_s, data_rate=data_rate, units="m")
        q_root = (
            np.asarray(data["q_root"], dtype=float)
            if "q_root" in data and np.asarray(data["q_root"]).shape[0] == q.shape[0]
            else extract_root_from_q(q_names, q, unwrap_rotations=False, renormalize_rotations=True)
        )
        if "qdot_root" in data and np.asarray(data["qdot_root"]).shape == q_root.shape:
            qdot_root = np.asarray(data["qdot_root"], dtype=float)
        else:
            qdot_root = centered_finite_difference(q_root, 1.0 / max(float(data_rate), 1.0))
        write_trc_root_kinematics_sidecar(output_path, q_root, qdot_root, frames, time_s)
        messagebox.showinfo("Export TRC from q", f"TRC written to:\n{output_path}")

    def export_selected_pseudo_root_from_points(self) -> None:
        """Export root pseudo-q derived geometrically from trunk markers."""

        recon_dir = self._selected_reconstruction_dir()
        if recon_dir is None:
            messagebox.showinfo("Export pseudo q root", "Select one reconstruction first.")
            return
        bundle_path = recon_dir / "reconstruction_bundle.npz"
        if not bundle_path.exists():
            messagebox.showerror("Export pseudo q root", f"Bundle not found:\n{bundle_path}")
            return

        data = np.load(bundle_path, allow_pickle=True)
        if "points_3d" not in data:
            messagebox.showinfo(
                "Export pseudo q root",
                "The selected reconstruction does not contain 3D markers/points.",
            )
            return

        points_3d = np.asarray(data["points_3d"], dtype=float)
        frames = np.asarray(data["frames"], dtype=int) if "frames" in data else np.arange(points_3d.shape[0], dtype=int)
        fps = max(float(self.state.fps_var.get()), 1.0)
        time_s = (
            np.asarray(data["time_s"], dtype=float)
            if "time_s" in data
            else np.arange(points_3d.shape[0], dtype=float) / fps
        )
        if time_s.shape[0] > 1 and np.all(np.isfinite(time_s)):
            dt = np.diff(time_s)
            positive_dt = dt[np.isfinite(dt) & (dt > 0)]
            fps = float(1.0 / np.median(positive_dt)) if positive_dt.size else fps

        summary = self.status_summaries.get(recon_dir.name, {})
        initial_rotation_correction = bool(
            summary.get(
                "initial_rotation_correction_applied",
                summary.get("initial_rotation_correction_requested", self.state.initial_rotation_correction_var.get()),
            )
        )
        max_interp_gap_frames = max(1, int(round(0.1 * float(fps))))
        trunk_points = np.array(points_3d, copy=True)
        trunk_marker_indices = [
            KP_INDEX["left_hip"],
            KP_INDEX["right_hip"],
            KP_INDEX["left_shoulder"],
            KP_INDEX["right_shoulder"],
        ]
        trunk_marker_values = trunk_points[:, trunk_marker_indices, :].reshape(points_3d.shape[0], -1)
        trunk_marker_values = interpolate_short_nan_runs(trunk_marker_values, max_interp_gap_frames)
        trunk_marker_values = fill_short_edge_nan_runs(trunk_marker_values, max_interp_gap_frames)
        trunk_points[:, trunk_marker_indices, :] = trunk_marker_values.reshape(points_3d.shape[0], 4, 3)
        q_root, correction_applied = extract_root_from_points(
            trunk_points,
            initial_rotation_correction,
            False,
        )
        qdot_root = centered_finite_difference(q_root, 1.0 / max(fps, 1.0))
        qddot_root = centered_finite_difference(qdot_root, 1.0 / max(fps, 1.0))
        output_path = recon_dir / f"{recon_dir.name}_pseudo_root_q.npz"
        np.savez(
            output_path,
            name=recon_dir.name,
            family="pseudo_root",
            source="geometric_trunk_markers",
            q=q_root,
            qdot=qdot_root,
            qddot=qddot_root,
            q_names=np.asarray(ROOT_Q_NAMES, dtype=object),
            frames=frames,
            time_s=time_s,
            initial_rotation_correction_applied=correction_applied,
        )
        messagebox.showinfo("Export pseudo q root", f"Pseudo root q written to:\n{output_path}")

    def refresh_timing_details(self) -> None:
        text = "Select a reconstruction to inspect timing details."
        selected = list(self.state.shared_reconstruction_selection)
        if selected:
            summary = self.status_summaries.get(selected[-1], {})
            if summary:
                text = format_reconstruction_timing_details(summary)
        self.timing_details.configure(state=tk.NORMAL)
        self.timing_details.delete("1.0", tk.END)
        self.timing_details.insert("1.0", text)
        self.timing_details.configure(state=tk.DISABLED)

    def clear_reconstructions(self) -> None:
        selected_names = [str(name) for name in getattr(self.state, "shared_reconstruction_selection", []) if str(name)]
        if not selected_names:
            messagebox.showinfo("Reconstructions", "Select one or more reconstructions in the top panel first.")
            return
        dataset_dir = current_dataset_dir(self.state)
        available_dirs = {recon_dir.name: recon_dir for recon_dir in reconstruction_dirs_for_path(dataset_dir)}
        recon_dirs = [available_dirs[name] for name in selected_names if name in available_dirs]
        if not recon_dirs:
            messagebox.showinfo("Reconstructions", "The selected reconstructions are no longer available.")
            return
        confirmed = messagebox.askyesno(
            "Clear reconstructions",
            "Supprimer les reconstruction(s) sélectionnée(s) ?\n\n"
            + "\n".join(f"- {recon_dir.name}" for recon_dir in recon_dirs),
            icon="warning",
        )
        if not confirmed:
            return
        errors: list[str] = []
        for recon_dir in recon_dirs:
            try:
                shutil.rmtree(recon_dir)
            except Exception as exc:
                errors.append(f"{recon_dir.name}: {exc}")
        self.state.shared_reconstruction_selection = [
            name for name in getattr(self.state, "shared_reconstruction_selection", []) if name not in selected_names
        ]
        self.refresh_status_rows()
        self.state.notify_reconstructions_updated()
        if errors:
            messagebox.showerror("Reconstructions", "Certaines suppressions ont échoué:\n\n" + "\n".join(errors))
        else:
            messagebox.showinfo("Reconstructions", f"{len(recon_dirs)} reconstruction(s) supprimée(s).")

    def selected_profiles(self) -> list[ReconstructionProfile]:
        selected = self.profile_tree.selection()
        if not selected:
            profiles = [profile for profile in self.state.profiles if profile.enabled]
            return append_default_pose2sim_profile(profiles, self.state.profiles, self.state.pose2sim_trc_var.get())
        return [self.state.profiles[int(item)] for item in selected]

    def build_command(self) -> list[str]:
        selected_profiles = self.selected_profiles()
        runtime_config_path = write_runtime_profiles_config(self.state)
        cmd = [
            sys.executable,
            "run_reconstruction_profiles.py",
            "--config",
            display_path(runtime_config_path),
            "--output-root",
            self.state.output_root_var.get(),
            "--dataset-name",
            current_dataset_name(self.state),
            "--calib",
            self.state.calib_var.get(),
            "--keypoints",
            self.state.keypoints_var.get(),
            "--fps",
            self.state.fps_var.get(),
            "--triangulation-workers",
            self.state.workers_var.get(),
        ]
        if self.state.pose2sim_trc_var.get().strip():
            cmd.extend(["--trc-file", self.state.pose2sim_trc_var.get()])
        annotation_var = getattr(self.state, "annotation_path_var", None)
        annotations_path = "" if annotation_var is None else str(annotation_var.get()).strip()
        if annotations_path and any(profile.pose_data_mode == "annotated" for profile in selected_profiles):
            cmd.extend(["--annotations-path", annotations_path])
        for profile in selected_profiles:
            cmd.extend(["--profile", profile.name])
        return cmd

    def on_command_success(self) -> None:
        self.refresh_status_rows()
        self.state.notify_reconstructions_updated()


class RootKinematicsTab(ttk.Frame):
    def __init__(self, master, state: SharedAppState):
        super().__init__(master)
        self.state = state
        self.bundle = None
        self.model_marker_points_cache: dict[tuple[str, str, str], np.ndarray] = {}
        self.uses_shared_reconstruction_panel = True
        self.shared_reconstruction_selectmode = "extended"

        controls = ttk.LabelFrame(self, text="Cinématiques de la racine")
        controls.pack(fill=tk.X, padx=10, pady=10)

        row = ttk.Frame(controls)
        row.pack(fill=tk.X, padx=8, pady=4)
        family_label = ttk.Label(row, text="Famille", width=10)
        family_label.pack(side=tk.LEFT)
        self.family = tk.StringVar(value="rotations")
        family_box = ttk.Combobox(
            row, textvariable=self.family, values=["translations", "rotations"], width=14, state="readonly"
        )
        family_box.pack(side=tk.LEFT, padx=(0, 6))
        quantity_label = ttk.Label(row, text="Quantité", width=10)
        quantity_label.pack(side=tk.LEFT)
        self.quantity = tk.StringVar(value="q")
        quantity_box = ttk.Combobox(row, textvariable=self.quantity, values=["q", "qdot"], width=10, state="readonly")
        quantity_box.pack(side=tk.LEFT, padx=(0, 6))
        repr_label = ttk.Label(row, text="Rot plot", width=10)
        repr_label.pack(side=tk.LEFT)
        self.rotation_plot_mode = tk.StringVar(value="euler")
        repr_box = ttk.Combobox(
            row, textvariable=self.rotation_plot_mode, values=["euler", "matrix"], width=10, state="readonly"
        )
        repr_box.pack(side=tk.LEFT, padx=(0, 6))
        rot_unit_label = ttk.Label(row, text="Rot unit", width=10)
        rot_unit_label.pack(side=tk.LEFT)
        self.rotation_unit = tk.StringVar(value="rad")
        rotation_unit_box = ttk.Combobox(
            row, textvariable=self.rotation_unit, values=["rad", "deg", "turns"], width=10, state="readonly"
        )
        rotation_unit_box.pack(side=tk.LEFT, padx=(0, 6))
        root_unwrap_label = ttk.Label(row, text="Root unwrap", width=11)
        root_unwrap_label.pack(side=tk.LEFT)
        self.root_unwrap_mode_var = tk.StringVar(value=root_unwrap_mode_display_name("off"))
        root_unwrap_box = ttk.Combobox(
            row,
            textvariable=self.root_unwrap_mode_var,
            values=[root_unwrap_mode_display_name(mode) for mode in ("off", "single", "double")],
            width=14,
            state="readonly",
        )
        root_unwrap_box.pack(side=tk.LEFT)

        row2 = ttk.Frame(controls)
        row2.pack(fill=tk.X, padx=8, pady=4)
        self.reextract_var = tk.BooleanVar(value=True)
        self.fd_qdot_var = tk.BooleanVar(value=True)
        self.matrix_ignore_alpha_var = tk.BooleanVar(value=False)
        self.common_geometric_root_var = tk.BooleanVar(value=False)
        self.interpolate_root_var = tk.BooleanVar(value=False)
        reextract_check = ttk.Checkbutton(
            row2, text="recalcul matrice + re-extraction Euler", variable=self.reextract_var
        )
        reextract_check.pack(side=tk.LEFT)
        fd_qdot_check = ttk.Checkbutton(row2, text="qdot par différence finie", variable=self.fd_qdot_var)
        fd_qdot_check.pack(side=tk.LEFT, padx=(12, 0))
        matrix_ignore_alpha_check = ttk.Checkbutton(
            row2, text="matrix without alpha correction", variable=self.matrix_ignore_alpha_var
        )
        matrix_ignore_alpha_check.pack(side=tk.LEFT, padx=(12, 0))
        common_geometric_root_check = ttk.Checkbutton(
            row2,
            text="common geometric root from model markers",
            variable=self.common_geometric_root_var,
        )
        common_geometric_root_check.pack(side=tk.LEFT, padx=(12, 0))
        interpolate_root_check = ttk.Checkbutton(row2, text="Interpolation", variable=self.interpolate_root_var)
        interpolate_root_check.pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(row2, text="Load / refresh", command=self.refresh_plot).pack(side=tk.LEFT, padx=(12, 0))

        attach_tooltip(family_label, "Choisit si l'on compare les translations ou les rotations de la racine.")
        attach_tooltip(family_box, "Choisit si l'on compare les translations ou les rotations de la racine.")
        attach_tooltip(quantity_label, "Choisit entre positions q et vitesses qdot.")
        attach_tooltip(quantity_box, "Choisit entre positions q et vitesses qdot.")
        attach_tooltip(
            repr_label,
            "Pour les rotations de racine, affiche soit les angles Euler, soit les 9 composantes de la matrice de rotation.",
        )
        attach_tooltip(
            repr_box, "Mode matrix: trace les composantes de R frame par frame. Ce mode s'applique aux rotations en q."
        )
        attach_tooltip(rot_unit_label, "Unite d'affichage des rotations de racine.")
        attach_tooltip(
            rotation_unit_box, "Unité d'affichage des trois rotations de racine. Les translations restent en m ou m/s."
        )
        attach_tooltip(
            root_unwrap_label,
            "Stabilise seulement l'affichage des rotations de racine: off, single, ou double unwrap.",
        )
        attach_tooltip(
            root_unwrap_box,
            "Cette option n'affecte pas les reconstructions; elle sert uniquement à l'affichage des rotations de racine.",
        )
        attach_tooltip(
            reextract_check, "Recalcule les angles Euler via la matrice de rotation du tronc avant affichage."
        )
        attach_tooltip(
            fd_qdot_check, "Recalcule qdot par difference finie sur q au lieu d'utiliser qdot deja sauvegarde."
        )
        attach_tooltip(
            matrix_ignore_alpha_check,
            "En mode matrix, affiche la matrice brute issue des marqueurs sans retirer la correction alpha. Cette option n'affecte que les reconstructions géométriques.",
        )
        attach_tooltip(
            common_geometric_root_check,
            "Pour les reconstructions basées sur q, reconstruit d'abord les marqueurs du modèle puis ré-extrait la racine géométrique avec la même méthode que triangulation/TRC file.",
        )
        attach_tooltip(
            interpolate_root_check,
            "En affichage seulement, interpole les trous courts de NaN jusqu'à environ 0.1 s (0.1 * fps).",
        )

        plot_box = ttk.LabelFrame(self, text="Comparaison racine")
        plot_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.figure = Figure(figsize=(10, 7))
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_box)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.state.keypoints_var.trace_add("write", lambda *_args: self.sync_dataset_dir())
        self.state.output_root_var.trace_add("write", lambda *_args: self.sync_dataset_dir())
        self.state.register_reconstruction_listener(lambda: self.after_idle(self.refresh_available_reconstructions))
        self.family.trace_add("write", lambda *_args: self.refresh_plot())
        self.quantity.trace_add("write", lambda *_args: self.refresh_plot())
        self.rotation_plot_mode.trace_add("write", lambda *_args: self.refresh_plot())
        self.rotation_unit.trace_add("write", lambda *_args: self.refresh_plot())
        self.root_unwrap_mode_var.trace_add("write", lambda *_args: self.refresh_plot())
        self.reextract_var.trace_add("write", lambda *_args: self.refresh_plot())
        self.fd_qdot_var.trace_add("write", lambda *_args: self.refresh_plot())
        self.matrix_ignore_alpha_var.trace_add("write", lambda *_args: self.refresh_plot())
        self.common_geometric_root_var.trace_add("write", lambda *_args: self.refresh_plot())
        self.interpolate_root_var.trace_add("write", lambda *_args: self.refresh_plot())
        self.after_idle(self.refresh_available_reconstructions)

    def configure_shared_reconstruction_panel(self, panel: SharedReconstructionPanel) -> None:
        panel.configure_for_consumer(
            title="Reconstructions | Racine",
            refresh_callback=self.refresh_available_reconstructions,
            selection_callback=self.refresh_plot,
            selectmode=self.shared_reconstruction_selectmode,
        )
        self.refresh_available_reconstructions()

    def selected_reconstruction_names(self) -> list[str]:
        return list(self.state.shared_reconstruction_selection)

    def _publish_reconstruction_rows(self, rows: list[dict[str, object]], defaults: list[str]) -> None:
        panel = self.state.shared_reconstruction_panel
        if panel is not None and self.state.active_reconstruction_consumer is self:
            panel.set_rows(rows, defaults)

    def _on_reconstruction_selection_changed(self) -> None:
        self.refresh_plot()

    def _show_empty_plot(self, message: str) -> None:
        """Render a lightweight placeholder when no root plot can be produced."""

        show_placeholder_figure(self.figure, self.canvas, message)

    def sync_dataset_dir(self) -> None:
        self.refresh_available_reconstructions()

    def _model_marker_points_for_root(self, reconstruction_name: str, q_series: np.ndarray) -> np.ndarray | None:
        """Return cached model-marker trajectories used for common geometric-root comparisons."""

        dataset_dir = current_dataset_dir(self.state)
        biomod_path = resolve_reconstruction_biomod(dataset_dir, reconstruction_name)
        if biomod_path is None or not biomod_path.exists():
            return None
        cache_key = (
            str(dataset_dir.resolve()),
            str(reconstruction_name),
            str(biomod_path.resolve()),
        )
        cached = self.model_marker_points_cache.get(cache_key)
        if cached is not None and cached.shape[0] == np.asarray(q_series).shape[0]:
            return cached
        marker_points = biorbd_markers_from_q(biomod_path, np.asarray(q_series, dtype=float))
        self.model_marker_points_cache[cache_key] = marker_points
        return marker_points

    def refresh_available_reconstructions(self) -> None:
        try:
            _output_dir, bundle, preview_state = load_shared_reconstruction_preview_state(
                self.state,
                preferred_names=[
                    "triangulation_exhaustive",
                    "triangulation_greedy",
                    "pose2sim",
                    "ekf_2d_acc",
                    "ekf_3d",
                ],
                fallback_count=4,
                include_3d=True,
                include_q=True,
                include_q_root=False,
            )
            available_names = preview_state.available_names
            if available_names:
                self._publish_reconstruction_rows(preview_state.rows, preview_state.defaults)
                self.bundle = bundle
                self.refresh_plot()
            else:
                self._publish_reconstruction_rows([], [])
                self.bundle = bundle
                self._show_empty_plot("No reconstruction available for root exploration.")
        except Exception:
            pass

    def refresh_plot(self) -> None:
        try:
            self.bundle = get_cached_preview_bundle(
                self.state, current_dataset_dir(self.state), None, None, align_root=False
            )
            available_names = bundle_available_reconstruction_names(
                self.bundle, include_3d=True, include_q=True, include_q_root=False
            )
            if not available_names:
                self._show_empty_plot("No reconstruction available for root exploration.")
                return
            selected_names = self.selected_reconstruction_names()
            if not selected_names:
                self._show_empty_plot("Select at least one reconstruction to compare root kinematics.")
                return
            recon_3d = self.bundle["recon_3d"]
            recon_q = self.bundle["recon_q"]
            recon_qdot = self.bundle["recon_qdot"]
            recon_q_root = self.bundle.get("recon_q_root", {})
            recon_qdot_root = self.bundle.get("recon_qdot_root", {})
            recon_summary = self.bundle.get("recon_summary", {})
            q_names = self.bundle["q_names"]
            dt = 1.0 / float(self.state.fps_var.get())
            max_interp_gap_frames = max(1, int(round(0.1 * float(self.state.fps_var.get()))))
            apply_interpolation = bool(self.interpolate_root_var.get())
            family_is_translation = self.family.get() == "translations"
            matrix_mode = (not family_is_translation) and self.rotation_plot_mode.get() == "matrix"
            root_unwrap_mode = (
                root_unwrap_mode_from_display_name(self.root_unwrap_mode_var.get())
                if not family_is_translation
                else "off"
            )
            apply_root_unwrap = (not family_is_translation) and root_unwrap_mode != "off"
            family_slice = slice(0, 3) if family_is_translation else slice(3, 6)
            axis_display_labels = root_axis_display_labels(self.family.get())
            quantity = self.quantity.get()
            rotation_unit = self.rotation_unit.get()
            unit_label = quantity_unit_label(quantity, family_is_translation, rotation_unit)

            self.figure.clear()
            effective_quantity = "q" if matrix_mode else quantity
            if matrix_mode:
                axes = np.asarray(self.figure.subplots(3, 3, sharex=True))
            else:
                axes = self.figure.subplots(3, 1, sharex=True)
                if not isinstance(axes, np.ndarray):
                    axes = np.array([axes])
            component_labels = [("R11", "R12", "R13"), ("R21", "R22", "R23"), ("R31", "R32", "R33")]
            plotted_series_count = 0

            for name in selected_names:
                series = None
                interpolation_applied_before_extraction = False
                summary = recon_summary.get(name, {}) if isinstance(recon_summary, dict) else {}
                summary_family = str(summary.get("family", ""))
                geometric_family = summary_family in {"pose2sim", "triangulation"}
                geometric_rotfix_mismatch = False
                model_marker_points = None
                model_biomod_path = None
                if name in recon_3d:
                    applied = summary.get("initial_rotation_correction_applied")
                    if applied is not None:
                        geometric_rotfix_mismatch = bool(applied) != bool(
                            self.state.initial_rotation_correction_var.get()
                        )
                if bool(self.common_geometric_root_var.get()) and name in recon_q:
                    model_marker_points = self._model_marker_points_for_root(
                        name, np.asarray(recon_q[name], dtype=float)
                    )
                    if model_marker_points is not None:
                        model_biomod_path = resolve_reconstruction_biomod(current_dataset_dir(self.state), name)
                        root_translation_origin = (
                            "upper_trunk"
                            if infer_model_variant_from_biomod(model_biomod_path).startswith("upper_root_")
                            else "pelvis"
                        )
                        series, model_marker_points = root_series_from_model_markers(
                            np.asarray(recon_q[name], dtype=float),
                            biomod_path=model_biomod_path,
                            marker_builder=biorbd_markers_from_q,
                            marker_points=model_marker_points,
                            quantity=effective_quantity,
                            dt=dt,
                            initial_rotation_correction=bool(self.state.initial_rotation_correction_var.get()),
                            unwrap_rotations=apply_root_unwrap,
                            unwrap_mode=root_unwrap_mode,
                            translation_origin=root_translation_origin,
                            interpolate_gap_frames=(max_interp_gap_frames if apply_interpolation else None),
                        )
                        interpolation_applied_before_extraction = bool(apply_interpolation)
                        geometric_family = True
                if series is None and geometric_family and name in recon_3d:
                    model_biomod_path = resolve_reconstruction_biomod(current_dataset_dir(self.state), name)
                    series = root_series_from_points(
                        np.asarray(recon_3d[name], dtype=float),
                        quantity=effective_quantity,
                        dt=dt,
                        initial_rotation_correction=bool(self.state.initial_rotation_correction_var.get()),
                        unwrap_rotations=apply_root_unwrap,
                        unwrap_mode=root_unwrap_mode,
                        translation_origin=(
                            "upper_trunk"
                            if infer_model_variant_from_biomod(model_biomod_path).startswith("upper_root_")
                            else "pelvis"
                        ),
                        interpolate_gap_frames=(max_interp_gap_frames if apply_interpolation else None),
                    )
                    interpolation_applied_before_extraction = bool(apply_interpolation)
                elif series is None and name in recon_q:
                    series = root_series_from_q(
                        q_names,
                        recon_q[name],
                        quantity=effective_quantity,
                        dt=dt,
                        qdot=recon_qdot.get(name),
                        fd_qdot=bool(self.fd_qdot_var.get()),
                        unwrap_rotations=apply_root_unwrap,
                        renormalize_rotations=bool(self.reextract_var.get()),
                        unwrap_mode=root_unwrap_mode,
                    )
                elif series is None and name in recon_3d:
                    model_biomod_path = resolve_reconstruction_biomod(current_dataset_dir(self.state), name)
                    series = root_series_from_points(
                        np.asarray(recon_3d[name], dtype=float),
                        quantity=effective_quantity,
                        dt=dt,
                        initial_rotation_correction=bool(self.state.initial_rotation_correction_var.get()),
                        unwrap_rotations=apply_root_unwrap,
                        unwrap_mode=root_unwrap_mode,
                        translation_origin=(
                            "upper_trunk"
                            if infer_model_variant_from_biomod(model_biomod_path).startswith("upper_root_")
                            else "pelvis"
                        ),
                        interpolate_gap_frames=(max_interp_gap_frames if apply_interpolation else None),
                    )
                    interpolation_applied_before_extraction = bool(apply_interpolation)
                elif series is None and name in recon_q_root and not geometric_rotfix_mismatch:
                    series = root_series_from_precomputed(
                        np.asarray(recon_q_root[name], dtype=float),
                        quantity=effective_quantity,
                        dt=dt,
                        qdot_root=recon_qdot_root.get(name),
                        fd_qdot=bool(self.fd_qdot_var.get()),
                    )
                if series is None:
                    continue
                series_array = np.asarray(series, dtype=float)
                if apply_interpolation and not interpolation_applied_before_extraction:
                    series_array = interpolate_short_nan_runs(series_array, max_interp_gap_frames)
                    series_array = fill_short_edge_nan_runs(series_array, max_interp_gap_frames)
                frame_slice = analysis_frame_slice(np.asarray(series).shape[0])
                if frame_slice.start >= series_array.shape[0]:
                    continue
                frame_indices = np.arange(series_array.shape[0], dtype=float)[frame_slice]
                t = frame_indices * dt
                if matrix_mode:
                    if model_marker_points is not None:
                        matrices = root_rotation_matrices_from_points(
                            model_marker_points,
                            initial_rotation_correction=(
                                bool(self.state.initial_rotation_correction_var.get())
                                and not bool(self.matrix_ignore_alpha_var.get())
                            ),
                        )
                    elif geometric_family and name in recon_3d:
                        matrix_points = np.asarray(recon_3d[name], dtype=float)
                        if apply_interpolation:
                            matrix_points = interpolate_trunk_marker_gaps_for_root(matrix_points, max_interp_gap_frames)
                        matrices = root_rotation_matrices_from_points(
                            matrix_points,
                            initial_rotation_correction=(
                                bool(self.state.initial_rotation_correction_var.get())
                                and not bool(self.matrix_ignore_alpha_var.get())
                            ),
                        )
                    else:
                        initial_rotation_correction_angle_rad = 0.0
                        if bool(self.matrix_ignore_alpha_var.get()) and bool(
                            summary.get("initial_rotation_correction_applied")
                        ):
                            raw_angle = summary.get("initial_rotation_correction_angle_rad")
                            try:
                                initial_rotation_correction_angle_rad = float(raw_angle)
                            except (TypeError, ValueError):
                                initial_rotation_correction_angle_rad = 0.0
                        matrices = root_rotation_matrices_from_series(
                            series_array,
                            initial_rotation_correction_angle_rad=initial_rotation_correction_angle_rad,
                        )
                    if apply_interpolation:
                        matrices = interpolate_short_nan_runs(
                            np.asarray(matrices, dtype=float).reshape(matrices.shape[0], -1),
                            max_interp_gap_frames,
                        ).reshape(matrices.shape)
                    matrices = matrices[frame_slice]
                    for row_idx in range(3):
                        for col_idx in range(3):
                            ax = axes[row_idx, col_idx]
                            ax.plot(
                                t,
                                matrices[:, row_idx, col_idx],
                                color=reconstruction_display_color(self.state, name),
                                linewidth=1.5,
                                label=reconstruction_legend_label(self.state, name),
                            )
                            ax.set_title(component_labels[row_idx][col_idx], fontsize=9)
                            ax.grid(alpha=0.25)
                else:
                    series_to_plot = scale_root_series_rotations(series_array, family_is_translation, rotation_unit)[
                        frame_slice
                    ]
                    for axis_idx, ax in enumerate(axes):
                        ax.plot(
                            t,
                            series_to_plot[:, family_slice.start + axis_idx],
                            color=reconstruction_display_color(self.state, name),
                            linewidth=1.7,
                            label=reconstruction_legend_label(self.state, name),
                        )
                        ax.set_ylabel(f"{axis_display_labels[axis_idx]} ({unit_label})")
                        ax.grid(alpha=0.25)
                plotted_series_count += 1
            if plotted_series_count == 0:
                self._show_empty_plot(f"No analysis frames are available after frame {ANALYSIS_START_FRAME - 1}.")
                return
            legend_axis = axes[0, 0] if matrix_mode else axes[0]
            handles, labels = legend_axis.get_legend_handles_labels()
            if handles:
                uniq = {}
                for handle, label in zip(handles, labels):
                    uniq[label] = handle
                legend_axis.legend(list(uniq.values()), list(uniq.keys()), loc="upper right", fontsize=8)
            if matrix_mode:
                for row_idx in range(3):
                    axes[row_idx, 0].set_ylabel("value")
                for col_idx in range(3):
                    axes[-1, col_idx].set_xlabel("Temps (s)")
            else:
                axes[-1].set_xlabel("Temps (s)")
            if family_is_translation:
                family_label = "translations"
            elif matrix_mode:
                family_label = "rotation matrix"
                if bool(self.matrix_ignore_alpha_var.get()):
                    family_label += " | raw marker frame"
            else:
                family_label = f"rotations ({rotation_unit})"
            displayed_quantity = effective_quantity if matrix_mode else quantity
            self.figure.suptitle(f"Cinématiques racine | {family_label} | {displayed_quantity}")
            self.figure.tight_layout()
            self.canvas.draw_idle()
        except Exception as exc:
            messagebox.showerror("Cinématiques racine", str(exc))


class JointKinematicsTab(ttk.Frame):
    def __init__(self, master, state: SharedAppState):
        super().__init__(master)
        self.state = state
        self.bundle = None
        self.q_names = np.array([], dtype=object)
        self.uses_shared_reconstruction_panel = True
        self.shared_reconstruction_selectmode = "extended"

        controls = ttk.LabelFrame(self, text="Autres DoF")
        controls.pack(fill=tk.X, padx=10, pady=10)

        row = ttk.Frame(controls)
        row.pack(fill=tk.X, padx=8, pady=4)
        quantity_label = ttk.Label(row, text="Quantité", width=10)
        quantity_label.pack(side=tk.LEFT)
        self.quantity = tk.StringVar(value="q")
        quantity_box = ttk.Combobox(row, textvariable=self.quantity, values=["q", "qdot"], width=10, state="readonly")
        quantity_box.pack(side=tk.LEFT, padx=(0, 6))
        rot_unit_label = ttk.Label(row, text="Angle unit", width=10)
        rot_unit_label.pack(side=tk.LEFT)
        self.rotation_unit = tk.StringVar(value="rad")
        rotation_unit_box = ttk.Combobox(
            row, textvariable=self.rotation_unit, values=["rad", "deg"], width=8, state="readonly"
        )
        rotation_unit_box.pack(side=tk.LEFT, padx=(0, 6))
        self.fd_qdot_var = tk.BooleanVar(value=False)
        fd_qdot_check = ttk.Checkbutton(row, text="qdot par différence finie", variable=self.fd_qdot_var)
        fd_qdot_check.pack(side=tk.LEFT)
        ttk.Button(row, text="Load / refresh", command=self.refresh_plot).pack(side=tk.LEFT, padx=(12, 0))

        self.pair_list = tk.Listbox(controls, selectmode=tk.MULTIPLE, exportselection=False, height=6)
        self.pair_list.pack(fill=tk.X, padx=8, pady=4)
        bind_extended_listbox_shortcuts(self.pair_list)
        attach_tooltip(quantity_label, "Choisit entre positions q et vitesses qdot pour les autres DoF.")
        attach_tooltip(quantity_box, "Choisit entre positions q et vitesses qdot pour les autres DoF.")
        attach_tooltip(rot_unit_label, "Choisit l'unité d'affichage des DoF de rotation.")
        attach_tooltip(rotation_unit_box, "Choisit l'unité d'affichage des DoF de rotation.")
        attach_tooltip(fd_qdot_check, "Recalcule qdot par difference finie sur q.")
        attach_tooltip(self.pair_list, "Choisissez les paires gauche/droite de DoF a comparer sur les graphes.")
        self.quantity.trace_add("write", lambda *_args: self.refresh_plot())
        self.rotation_unit.trace_add("write", lambda *_args: self.refresh_plot())
        self.fd_qdot_var.trace_add("write", lambda *_args: self.refresh_plot())
        self.pair_list.bind("<<ListboxSelect>>", self._on_pair_selection_changed)

        plot_box = ttk.LabelFrame(self, text="Gauche / droite sur le même graphe")
        plot_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.figure = Figure(figsize=(10, 7))
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_box)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.state.keypoints_var.trace_add("write", lambda *_args: self.sync_dataset_dir())
        self.state.output_root_var.trace_add("write", lambda *_args: self.sync_dataset_dir())
        self.state.register_reconstruction_listener(lambda: self.after_idle(self.refresh_available_reconstructions))
        self.after_idle(self.refresh_available_reconstructions)

    def configure_shared_reconstruction_panel(self, panel: SharedReconstructionPanel) -> None:
        panel.configure_for_consumer(
            title="Reconstructions | Autres DoF",
            refresh_callback=self.refresh_available_reconstructions,
            selection_callback=self.refresh_plot,
            selectmode=self.shared_reconstruction_selectmode,
        )
        self.refresh_available_reconstructions()

    def selected_reconstruction_names(self) -> list[str]:
        return list(self.state.shared_reconstruction_selection)

    def _publish_reconstruction_rows(self, rows: list[dict[str, object]], defaults: list[str]) -> None:
        panel = self.state.shared_reconstruction_panel
        if panel is not None and self.state.active_reconstruction_consumer is self:
            panel.set_rows(rows, defaults)

    def _on_reconstruction_selection_changed(self) -> None:
        self.refresh_plot()

    def _on_pair_selection_changed(self, _event=None) -> None:
        """Refresh the joint-DoF plot when the selected DoF pairs change."""

        self.refresh_plot()

    def _show_empty_plot(self, message: str) -> None:
        """Render a placeholder when no joint-kinematics plot can be produced."""

        show_placeholder_figure(self.figure, self.canvas, message)

    @staticmethod
    def _is_rotational_dof(dof_name: str) -> bool:
        """Return whether the given DoF represents a rotation."""

        return ":Rot" in str(dof_name)

    def _scale_joint_dof_series(self, values: np.ndarray, dof_name: str) -> np.ndarray:
        """Convert rotational DoF series to the requested display unit."""

        if not self._is_rotational_dof(dof_name):
            return np.asarray(values, dtype=float)
        return np.asarray(values, dtype=float) * rotation_unit_scale(self.rotation_unit.get())

    def _joint_dof_unit_label(self, dof_name: str) -> str:
        """Return the display unit for one joint DoF curve."""

        if self._is_rotational_dof(dof_name):
            return rotation_unit_label(self.rotation_unit.get(), self.quantity.get())
        return "m" if self.quantity.get() == "q" else "m/s"

    @staticmethod
    def _upper_back_target_series(
        series: np.ndarray, name_to_index: dict[str, int], dof_name: str
    ) -> np.ndarray | None:
        """Return the default pseudo-observation target for one upper-back DoF."""

        if dof_name.endswith(":RotY") and dof_name.startswith(("UPPER_BACK:", "LOWER_TRUNK:")):
            hip_candidates = (
                "LEFT_THIGH:RotY",
                "RIGHT_THIGH:RotY",
                "LEFT_THIGH:RotX",
                "RIGHT_THIGH:RotX",
            )
            hip_indices = [name_to_index[name] for name in hip_candidates if name in name_to_index]
            if len(hip_indices) < 2:
                return None
            hip_values = np.asarray(series[:, hip_indices], dtype=float)
            if hip_values.ndim != 2 or hip_values.shape[1] < 2:
                return None
            with np.errstate(invalid="ignore"):
                return 0.2 * np.nanmean(hip_values, axis=1)
        if dof_name.endswith(":RotX") or dof_name.endswith(":RotZ"):
            if dof_name.startswith(("UPPER_BACK:", "LOWER_TRUNK:")):
                return np.zeros(series.shape[0], dtype=float)
        return None

    def sync_dataset_dir(self) -> None:
        self.refresh_available_reconstructions()

    def refresh_available_reconstructions(self) -> None:
        try:
            _output_dir, bundle, preview_state = load_shared_reconstruction_preview_state(
                self.state,
                preferred_names=["ekf_2d_acc", "ekf_2d_flip_acc", "ekf_2d_dyn", "ekf_2d_flip_dyn", "ekf_3d"],
                fallback_count=3,
                include_3d=False,
                include_q=True,
                include_q_root=False,
            )
            available_q = preview_state.available_names
            if available_q:
                self._publish_reconstruction_rows(preview_state.rows, preview_state.defaults)
                self.bundle = bundle
                self.refresh_plot()
            else:
                self.bundle = bundle
                self._publish_reconstruction_rows([], [])
                self.q_names = np.array([], dtype=object)
                self._show_empty_plot("No reconstruction with q is available for the joint-DoF view.")
        except Exception:
            pass

    def refresh_plot(self) -> None:
        try:
            self.bundle = get_cached_preview_bundle(
                self.state, current_dataset_dir(self.state), None, None, align_root=False
            )
            available_q = bundle_available_reconstruction_names(
                self.bundle, include_3d=False, include_q=True, include_q_root=False
            )
            if not available_q:
                self.q_names = np.array([], dtype=object)
                self._show_empty_plot("No reconstruction with q is available for the joint-DoF view.")
                return
            selected_names = self.selected_reconstruction_names()
            if not selected_names:
                self._show_empty_plot("Select at least one reconstruction to compare the joint DoFs.")
                return

            self.q_names = self.bundle["q_names"]
            pairs = pair_dof_names(self.q_names)
            if not pairs:
                self._show_empty_plot("No left/right DoF pairs are available for the selected reconstructions.")
                return
            current_labels = [self.pair_list.get(idx) for idx in range(self.pair_list.size())]
            new_labels = [pair_label for pair_label, _, _ in pairs]
            if current_labels != new_labels:
                self.pair_list.delete(0, tk.END)
                for pair_label in new_labels:
                    self.pair_list.insert(tk.END, pair_label)
                for idx in range(min(4, len(pairs))):
                    self.pair_list.selection_set(idx)

            selected_labels = [self.pair_list.get(idx) for idx in self.pair_list.curselection()]
            if not selected_labels:
                selected_labels = [pair[0] for pair in pairs[:1]]
            selected_pairs = [pair for pair in pairs if pair[0] in selected_labels]
            if not selected_pairs:
                self._show_empty_plot("Select at least one left/right DoF pair to plot.")
                return

            self.figure.clear()
            axes = self.figure.subplots(len(selected_pairs), 1, sharex=True)
            axes = np.atleast_1d(axes)
            dt = 1.0 / float(self.state.fps_var.get())
            name_to_index = {str(name): idx for idx, name in enumerate(self.q_names)}
            plotted_series = False
            side_styles = {
                "left": {"linestyle": "-", "marker": "o"},
                "right": {"linestyle": "--", "marker": "s"},
            }

            for ax, (pair_label, left_name, right_name) in zip(axes, selected_pairs):
                for recon_name in selected_names:
                    q = self.bundle["recon_q"].get(recon_name)
                    qdot = self.bundle["recon_qdot"].get(recon_name)
                    if q is None:
                        continue
                    if self.quantity.get() == "q":
                        series = q
                    else:
                        if self.fd_qdot_var.get() or qdot is None:
                            series = centered_finite_difference(q, dt)
                        else:
                            series = qdot
                    frame_slice = analysis_frame_slice(series.shape[0])
                    if frame_slice.start >= series.shape[0]:
                        continue
                    series = np.asarray(series, dtype=float)[frame_slice]
                    frame_indices = np.arange(frame_slice.start, frame_slice.stop, dtype=float)
                    time_s = frame_indices * dt
                    left_idx = name_to_index[left_name]
                    color = reconstruction_display_color(self.state, recon_name)
                    legend_label = reconstruction_legend_label(self.state, recon_name)
                    left_style = side_styles["left"]
                    left_values = self._scale_joint_dof_series(series[:, left_idx], left_name)
                    ax.plot(
                        time_s,
                        left_values,
                        color=color,
                        linewidth=1.7,
                        linestyle=left_style["linestyle"],
                        marker=left_style["marker"],
                        markevery=max(1, len(time_s) // 24) if len(time_s) else 1,
                        markersize=3.0,
                        label=f"{legend_label} | L" if right_name is not None else legend_label,
                    )
                    if right_name is not None:
                        right_idx = name_to_index[right_name]
                        right_style = side_styles["right"]
                        right_values = self._scale_joint_dof_series(series[:, right_idx], right_name)
                        ax.plot(
                            time_s,
                            right_values,
                            color=color,
                            linewidth=1.7,
                            linestyle=right_style["linestyle"],
                            marker=right_style["marker"],
                            markevery=max(1, len(time_s) // 24) if len(time_s) else 1,
                            markersize=3.0,
                            label=f"{legend_label} | R",
                        )
                    elif left_name.startswith(("UPPER_BACK:", "LOWER_TRUNK:")):
                        target_values = self._upper_back_target_series(series, name_to_index, left_name)
                        if target_values is not None:
                            ax.plot(
                                time_s,
                                self._scale_joint_dof_series(target_values, left_name),
                                color="#202020",
                                linewidth=1.2,
                                linestyle=":",
                                label=f"{legend_label} | target",
                            )
                    plotted_series = True
                ax.set_title(pair_label)
                ax.grid(alpha=0.25)
                ax.set_ylabel(self._joint_dof_unit_label(left_name))
            if not plotted_series:
                self._show_empty_plot(f"No analysis frames are available after frame {ANALYSIS_START_FRAME - 1}.")
                return
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                uniq = {}
                for handle, label in zip(handles, labels):
                    uniq[label] = handle
                axes[0].legend(list(uniq.values()), list(uniq.keys()), loc="upper right", fontsize=8, ncol=2)
            axes[-1].set_xlabel("Temps (s)")
            self.figure.tight_layout()
            self.canvas.draw_idle()
        except Exception as exc:
            messagebox.showerror("Autres DoF", str(exc))


class ObservabilityTab(ttk.Frame):
    def __init__(self, master, state: SharedAppState):
        super().__init__(master)
        self.state = state
        self.bundle = None
        self.uses_shared_reconstruction_panel = True
        self.shared_reconstruction_selectmode = "browse"

        controls = ttk.LabelFrame(self, text="Observabilité du modèle")
        controls.pack(fill=tk.X, padx=10, pady=10)

        row = ttk.Frame(controls)
        row.pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(row, text="Load / refresh", command=self.refresh_plot).pack(side=tk.LEFT)
        self.camera_info_var = tk.StringVar(value="Cameras: all available")
        ttk.Label(row, textvariable=self.camera_info_var, foreground="#4f5b66").pack(side=tk.LEFT, padx=(12, 0))

        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=3)
        body.add(right, weight=2)

        plot_box = ttk.LabelFrame(left, text="Rang des jacobiennes")
        plot_box.pack(fill=tk.BOTH, expand=True)
        self.figure = Figure(figsize=(10, 7))
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_box)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_box, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill=tk.X)

        summary_box = ttk.LabelFrame(right, text="Résumé")
        summary_box.pack(fill=tk.BOTH, expand=True)
        self.summary_text = ScrolledText(summary_box, wrap=tk.WORD, height=20)
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        attach_tooltip(self.summary_text, "Résumé des rangs min/médiane/max et du pourcentage de frames de plein rang.")

        self.state.keypoints_var.trace_add("write", lambda *_args: self.sync_dataset_dir())
        self.state.output_root_var.trace_add("write", lambda *_args: self.sync_dataset_dir())
        self.state.selected_camera_names_var.trace_add("write", lambda *_args: self.refresh_plot())
        self.state.register_reconstruction_listener(lambda: self.after_idle(self.refresh_available_reconstructions))
        self.after_idle(self.refresh_available_reconstructions)

    def configure_shared_reconstruction_panel(self, panel: SharedReconstructionPanel) -> None:
        panel.configure_for_consumer(
            title="Reconstructions | Observabilité",
            refresh_callback=self.refresh_available_reconstructions,
            selection_callback=self.refresh_plot,
            selectmode=self.shared_reconstruction_selectmode,
        )
        self.refresh_available_reconstructions()

    def _publish_reconstruction_rows(self, rows: list[dict[str, object]], defaults: list[str]) -> None:
        panel = self.state.shared_reconstruction_panel
        if panel is not None and self.state.active_reconstruction_consumer is self:
            panel.set_rows(rows, defaults)

    def _on_reconstruction_selection_changed(self) -> None:
        self.refresh_plot()

    def _show_empty_plot(self, message: str) -> None:
        """Render a lightweight placeholder when no observability plot can be produced."""

        show_placeholder_figure(self.figure, self.canvas, message)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", message)

    def sync_dataset_dir(self) -> None:
        self.refresh_available_reconstructions()

    def refresh_available_reconstructions(self) -> None:
        try:
            _output_dir, bundle, preview_state = load_shared_reconstruction_preview_state(
                self.state,
                preferred_names=["ekf_2d_acc", "ekf_2d_flip_acc", "ekf_2d_dyn", "ekf_2d_flip_dyn", "ekf_3d"],
                fallback_count=1,
                include_3d=False,
                include_q=True,
                include_q_root=False,
            )
            available_q = preview_state.available_names
            if available_q:
                self._publish_reconstruction_rows(preview_state.rows, preview_state.defaults[:1])
                self.bundle = bundle
                self.refresh_plot()
            else:
                self.bundle = bundle
                self._publish_reconstruction_rows([], [])
                self._show_empty_plot("No reconstruction with q is available for observability analysis.")
        except Exception:
            pass

    def refresh_plot(self) -> None:
        try:
            dataset_dir = current_dataset_dir(self.state)
            self.bundle = get_cached_preview_bundle(self.state, dataset_dir, None, None, align_root=False)
            available_q = bundle_available_reconstruction_names(
                self.bundle, include_3d=False, include_q=True, include_q_root=False
            )
            if not available_q:
                self._show_empty_plot("No reconstruction with q is available for observability analysis.")
                return

            selected_names = list(self.state.shared_reconstruction_selection)
            selected_name = selected_names[-1] if selected_names else None
            if selected_name is None:
                self._show_empty_plot("Select one reconstruction with q to inspect observability.")
                return

            self.figure.clear()
            self.summary_text.delete("1.0", tk.END)

            q = self.bundle.get("recon_q", {}).get(selected_name)
            if q is None:
                raise ValueError(f"Aucune trajectoire q disponible pour {selected_name}.")

            biomod_path = resolve_reconstruction_biomod(dataset_dir, selected_name)
            if biomod_path is None or not biomod_path.exists():
                raise ValueError(f"Aucun bioMod associé trouvé pour {selected_name}.")

            import biorbd

            model = biorbd.Model(str(biomod_path))
            calibrations = get_cached_calibrations(self.state, ROOT / self.state.calib_var.get())
            selected_camera_names = [name for name in current_selected_camera_names(self.state) if name in calibrations]
            if not selected_camera_names:
                selected_camera_names = list(calibrations.keys())
            camera_calibrations = [calibrations[name] for name in selected_camera_names]
            self.camera_info_var.set(
                "Cameras: all available"
                if len(selected_camera_names) == len(calibrations)
                else f"Cameras: {format_camera_names(selected_camera_names)}"
            )

            rank_series = compute_observability_rank_series(model, np.asarray(q, dtype=float), camera_calibrations)
            time_s = np.asarray(
                self.bundle.get(
                    "time_s", np.arange(np.asarray(q).shape[0]) / max(float(self.state.fps_var.get()), 1.0)
                ),
                dtype=float,
            )
            if time_s.shape[0] != np.asarray(q).shape[0]:
                time_s = np.arange(np.asarray(q).shape[0], dtype=float) / max(float(self.state.fps_var.get()), 1.0)
            frame_slice = analysis_frame_slice(time_s.shape[0])
            if frame_slice.start >= time_s.shape[0]:
                self._show_empty_plot(f"No analysis frames are available after frame {ANALYSIS_START_FRAME - 1}.")
                return
            time_s = time_s[frame_slice]
            marker_rank = np.asarray(rank_series.marker_rank, dtype=float)[frame_slice]
            observation_rank = np.asarray(rank_series.observation_rank, dtype=float)[frame_slice]
            marker_summary = summarize_rank_series(marker_rank, rank_series.marker_full_rank)
            observation_summary = summarize_rank_series(observation_rank, rank_series.observation_full_rank)

            axes = self.figure.subplots(2, 1, sharex=True)
            axes = np.atleast_1d(axes)
            axes[0].plot(time_s, marker_rank, color="#1f77b4", linewidth=1.7, label="rank(J_markers_3D)")
            axes[0].axhline(rank_series.marker_full_rank, color="#1f77b4", linestyle="--", alpha=0.6, label="full rank")
            axes[0].set_ylabel("3D rank")
            axes[0].grid(alpha=0.25)
            axes[0].legend(loc="upper right", fontsize=8)

            axes[1].plot(time_s, observation_rank, color="#d62728", linewidth=1.7, label="rank(J_obs_2D)")
            axes[1].axhline(
                rank_series.observation_full_rank, color="#d62728", linestyle="--", alpha=0.6, label="full rank"
            )
            axes[1].set_ylabel("2D rank")
            axes[1].set_xlabel("Temps (s)")
            axes[1].grid(alpha=0.25)
            axes[1].legend(loc="upper right", fontsize=8)
            self.figure.suptitle(f"Observabilité | {selected_name}")
            self.figure.tight_layout()
            self.canvas.draw_idle()

            self.summary_text.insert(
                "1.0",
                "\n".join(
                    [
                        f"reconstruction={selected_name}",
                        f"biomod={display_path(biomod_path)}",
                        f"nq={int(model.nbQ())}",
                        f"cameras={format_camera_names(selected_camera_names)}",
                        "",
                        "J_markers_3D(q)",
                        f"  full rank target={rank_series.marker_full_rank}",
                        f"  min={marker_summary['min']:.0f} | median={marker_summary['median']:.1f} | max={marker_summary['max']:.0f}",
                        f"  full-rank frames={100.0 * marker_summary['full_rank_ratio']:.1f}%",
                        "",
                        "J_obs_2D(q)",
                        f"  full rank target={rank_series.observation_full_rank}",
                        f"  min={observation_summary['min']:.0f} | median={observation_summary['median']:.1f} | max={observation_summary['max']:.0f}",
                        f"  full-rank frames={100.0 * observation_summary['full_rank_ratio']:.1f}%",
                        "",
                        "Interpretation:",
                        "  J_markers_3D montre si les marqueurs 3D du modèle contraignent tous les DoF.",
                        "  J_obs_2D ajoute la projection caméra et montre la perte de rang induite par les observations image.",
                    ]
                ),
            )
        except Exception as exc:
            messagebox.showerror("Observabilité", str(exc))


class Analysis3DTab(ttk.Frame):
    """Compare 3D segment lengths and angular momentum across reconstructions."""

    def __init__(self, master, state: SharedAppState):
        super().__init__(master)
        self.state = state
        self.bundle = None
        self.uses_shared_reconstruction_panel = True
        self.shared_reconstruction_selectmode = "extended"

        controls = ttk.LabelFrame(self, text="Analyse 3D")
        controls.pack(fill=tk.X, padx=10, pady=10)

        row = ttk.Frame(controls)
        row.pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(row, text="Load / refresh", command=self.refresh_plot).pack(side=tk.LEFT)
        ttk.Label(
            row,
            text="Segment lengths always use 3D points; angular momentum uses q/qdot + bioMod when available.",
            foreground="#4f5b66",
        ).pack(side=tk.LEFT, padx=(12, 0))

        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=3)
        body.add(right, weight=2)

        plot_box = ttk.LabelFrame(left, text="Longueurs segmentaires + moment cinétique")
        plot_box.pack(fill=tk.BOTH, expand=True)
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_box)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_box, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill=tk.X)

        summary_box = ttk.LabelFrame(right, text="Résumé")
        summary_box.pack(fill=tk.BOTH, expand=True)
        self.summary_text = ScrolledText(summary_box, wrap=tk.WORD, height=20)
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.state.keypoints_var.trace_add("write", lambda *_args: self.sync_dataset_dir())
        self.state.output_root_var.trace_add("write", lambda *_args: self.sync_dataset_dir())
        self.state.pose2sim_trc_var.trace_add("write", lambda *_args: self.sync_dataset_dir())
        self.state.register_reconstruction_listener(lambda: self.after_idle(self.refresh_available_reconstructions))
        self.after_idle(self.refresh_available_reconstructions)

    def configure_shared_reconstruction_panel(self, panel: SharedReconstructionPanel) -> None:
        panel.configure_for_consumer(
            title="Reconstructions | 3D analysis",
            refresh_callback=self.refresh_available_reconstructions,
            selection_callback=self.refresh_plot,
            selectmode=self.shared_reconstruction_selectmode,
        )
        self.refresh_available_reconstructions()

    def _publish_reconstruction_rows(self, rows: list[dict[str, object]], defaults: list[str]) -> None:
        panel = self.state.shared_reconstruction_panel
        if panel is not None and self.state.active_reconstruction_consumer is self:
            panel.set_rows(rows, defaults)

    def _on_reconstruction_selection_changed(self) -> None:
        self.refresh_plot()

    def _show_empty_plot(self, message: str) -> None:
        """Render one placeholder when the 3D analysis cannot be displayed."""

        show_placeholder_figure(self.figure, self.canvas, message)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", message)

    def sync_dataset_dir(self) -> None:
        self.refresh_available_reconstructions()

    def refresh_available_reconstructions(self) -> None:
        try:
            _output_dir, bundle, preview_state = load_shared_reconstruction_preview_state(
                self.state,
                preferred_names=[
                    "ekf_2d_acc",
                    "ekf_3d",
                    "pose2sim",
                    "triangulation_exhaustive",
                    "triangulation_greedy",
                ],
                fallback_count=3,
                include_3d=True,
                include_q=True,
                include_q_root=False,
            )
            available_names = preview_state.available_names
            if available_names:
                self._publish_reconstruction_rows(preview_state.rows, preview_state.defaults)
                self.bundle = bundle
                self.refresh_plot()
            else:
                self.bundle = bundle
                self._publish_reconstruction_rows([], [])
                self._show_empty_plot("No reconstruction with 3D data is available for 3D analysis.")
        except Exception:
            pass

    def refresh_plot(self) -> None:
        try:
            dataset_dir = current_dataset_dir(self.state)
            self.bundle = get_cached_preview_bundle(self.state, dataset_dir, None, None, align_root=False)
            available_names = bundle_available_reconstruction_names(
                self.bundle, include_3d=True, include_q=True, include_q_root=False
            )
            if not available_names:
                self._show_empty_plot("No reconstruction with 3D data is available for 3D analysis.")
                return
            selected_names = list(self.state.shared_reconstruction_selection)
            if not selected_names:
                self._show_empty_plot("Select at least one reconstruction for 3D analysis.")
                return

            self.figure.clear()
            self.summary_text.delete("1.0", tk.END)
            axes = self.figure.subplot_mosaic([["box", "box"], ["comp", "norm"]])
            box_ax = axes["box"]
            comp_ax = axes["comp"]
            norm_ax = axes["norm"]

            segment_labels = [definition[0] for definition in SEGMENT_LENGTH_DEFINITIONS]
            base_positions = np.arange(len(segment_labels), dtype=float)
            plotted_lengths = 0
            summary_lines = []
            offsets = np.linspace(-0.30, 0.30, num=max(len(selected_names), 1))
            width = min(0.18, 0.65 / max(len(selected_names), 1))

            for offset, recon_name in zip(offsets, selected_names):
                points_3d = self.bundle.get("recon_3d", {}).get(recon_name)
                if points_3d is None:
                    continue
                points_3d = slice_analysis_series(np.asarray(points_3d, dtype=float))
                if points_3d.shape[0] == 0:
                    continue
                length_samples = valid_segment_length_samples(segment_length_series(points_3d))
                box_data = [length_samples.get(label, np.array([np.nan], dtype=float)) for label in segment_labels]
                color = reconstruction_display_color(self.state, recon_name)
                boxplot = box_ax.boxplot(
                    box_data,
                    positions=base_positions + offset,
                    widths=width,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                )
                for patch in boxplot["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.35)
                    patch.set_edgecolor(color)
                for median in boxplot["medians"]:
                    median.set_color(color)
                    median.set_linewidth(1.4)
                box_ax.plot(
                    [],
                    [],
                    color=color,
                    linewidth=6,
                    alpha=0.35,
                    label=reconstruction_legend_label(self.state, recon_name),
                )
                plotted_lengths += 1
                summary_lines.append(f"{reconstruction_label(recon_name)} lengths:")
                for label in segment_labels:
                    values = length_samples.get(label)
                    if values is None or values.size == 0:
                        continue
                    summary_lines.append(
                        f"  {label}: median={float(np.median(values)):.3f} m | IQR={float(np.percentile(values, 75) - np.percentile(values, 25)):.3f} m"
                    )

            if plotted_lengths:
                box_ax.set_xticks(base_positions)
                box_ax.set_xticklabels(segment_labels, rotation=30, ha="right")
                box_ax.set_ylabel("Length (m)")
                box_ax.set_title("Segment-length distribution from 3D points")
                box_ax.grid(alpha=0.20, axis="y")
                handles, labels = box_ax.get_legend_handles_labels()
                if handles:
                    box_ax.legend(handles, labels, loc="upper right", fontsize=8)
            else:
                box_ax.axis("off")
                box_ax.text(
                    0.5,
                    0.5,
                    "No 3D points available for the selected reconstructions.",
                    ha="center",
                    va="center",
                    transform=box_ax.transAxes,
                )

            momentum_plotted = 0
            time_s_bundle = np.asarray(self.bundle.get("time_s", np.array([], dtype=float)), dtype=float)
            for recon_name in selected_names:
                q = self.bundle.get("recon_q", {}).get(recon_name)
                if q is None:
                    summary_lines.append(f"{reconstruction_label(recon_name)} angular momentum: unavailable (no q).")
                    continue
                biomod_path = resolve_reconstruction_biomod(dataset_dir, recon_name)
                if biomod_path is None or not biomod_path.exists():
                    summary_lines.append(
                        f"{reconstruction_label(recon_name)} angular momentum: unavailable (no bioMod)."
                    )
                    continue
                q = np.asarray(q, dtype=float)
                qdot = self.bundle.get("recon_qdot", {}).get(recon_name)
                if qdot is not None:
                    qdot = np.asarray(qdot, dtype=float)
                if time_s_bundle.shape[0] == q.shape[0]:
                    time_s = time_s_bundle
                else:
                    dt = 1.0 / max(float(self.state.fps_var.get()), 1.0)
                    time_s = np.arange(q.shape[0], dtype=float) * dt
                frame_slice = analysis_frame_slice(q.shape[0])
                if frame_slice.start >= q.shape[0]:
                    continue
                q = q[frame_slice]
                if qdot is not None:
                    qdot = qdot[frame_slice]
                time_s = time_s[frame_slice]
                import biorbd

                model = biorbd.Model(str(biomod_path))
                plot_data = angular_momentum_plot_data(model, q, qdot, time_s)
                color = reconstruction_display_color(self.state, recon_name)
                label = reconstruction_legend_label(self.state, recon_name)
                component_styles = [("-", "Hx"), ("--", "Hy"), (":", "Hz")]
                for component_idx, (linestyle, axis_label) in enumerate(component_styles):
                    comp_ax.plot(
                        plot_data.time_s,
                        plot_data.components[:, component_idx],
                        color=color,
                        linestyle=linestyle,
                        linewidth=1.5,
                        label=f"{label} | {axis_label}",
                    )
                norm_ax.plot(plot_data.time_s, plot_data.norm, color=color, linewidth=1.7, label=label)
                momentum_plotted += 1
                finite_norm = plot_data.norm[np.isfinite(plot_data.norm)]
                if finite_norm.size:
                    summary_lines.append(
                        f"{label} angular momentum: peak={float(np.max(finite_norm)):.3f} kg.m^2/s | median={float(np.median(finite_norm)):.3f}"
                    )

            if momentum_plotted:
                comp_ax.set_title("3D angular momentum components")
                comp_ax.set_ylabel("kg.m²/s")
                comp_ax.grid(alpha=0.25)
                comp_ax.legend(loc="upper right", fontsize=8, ncol=2)
                norm_ax.set_title("3D angular momentum norm")
                norm_ax.set_ylabel("kg.m²/s")
                norm_ax.grid(alpha=0.25)
                norm_ax.legend(loc="upper right", fontsize=8)
                comp_ax.set_xlabel("Time (s)")
                norm_ax.set_xlabel("Time (s)")
            else:
                comp_ax.axis("off")
                comp_ax.text(
                    0.5,
                    0.5,
                    "Angular momentum is only available for reconstructions with q/qdot and a bioMod.",
                    ha="center",
                    va="center",
                    transform=comp_ax.transAxes,
                )
                norm_ax.axis("off")

            self.figure.tight_layout()
            self.canvas.draw_idle()
            self.summary_text.insert("1.0", "\n".join(summary_lines) if summary_lines else "No 3D analysis available.")
        except Exception as exc:
            messagebox.showerror("3D analysis", str(exc))


class ExecutionTab(ttk.Frame):
    """Inspect localized execution deductions on complete jumps."""

    def __init__(self, master, state: SharedAppState):
        super().__init__(master)
        self.state = state
        self.bundle = None
        self.execution_analysis: ExecutionSessionAnalysis | None = None
        self.dd_analysis: DDSessionAnalysis | None = None
        self.current_reconstruction_name: str | None = None
        self.calibrations = None
        self.pose_data = None
        self.images_root: Path | None = None
        self._suspend_selection_callbacks = False
        self.uses_shared_reconstruction_panel = True
        self.shared_reconstruction_selectmode = "browse"

        controls = ttk.LabelFrame(self, text="Analyse d'execution")
        controls.pack(fill=tk.X, padx=10, pady=10)

        row = ttk.Frame(controls)
        row.pack(fill=tk.X, padx=8, pady=4)
        camera_label = ttk.Label(row, text="Camera", width=10)
        camera_label.pack(side=tk.LEFT)
        self.camera_name = tk.StringVar(value="")
        self.camera_box = ttk.Combobox(row, textvariable=self.camera_name, values=[], width=18, state="readonly")
        self.camera_box.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(row, text="Analyze / refresh", command=self.refresh_analysis).pack(side=tk.LEFT)
        self.image_layer_status_var = tk.StringVar(
            value="2D view currently shows 2D detections + reprojection. Image overlay is ready to plug in later."
        )
        ttk.Label(row, textvariable=self.image_layer_status_var, foreground="#4f5b66").pack(side=tk.LEFT, padx=(12, 0))
        attach_tooltip(camera_label, "Camera used for the 2D localization preview.")
        attach_tooltip(self.camera_box, "Camera used for the 2D localization preview.")

        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        left = ttk.Frame(body, width=430)
        right = ttk.Frame(body)
        body.add(left, weight=2)
        body.add(right, weight=3)

        jumps_box = ttk.LabelFrame(left, text="Complete jumps")
        jumps_box.pack(fill=tk.BOTH, expand=False, pady=(0, 8))
        self.jump_list = tk.Listbox(jumps_box, exportselection=False, height=7)
        self.jump_list.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.jump_list.bind("<<ListboxSelect>>", self._on_jump_selected)

        deductions_box = ttk.LabelFrame(left, text="Localized deductions")
        deductions_box.pack(fill=tk.BOTH, expand=False, pady=(0, 8))
        self.deduction_tree = ttk.Treeview(
            deductions_box,
            columns=("label", "deduction", "frame", "detail"),
            show="headings",
            height=7,
            selectmode="browse",
        )
        self.deduction_tree.heading("label", text="Deduction")
        self.deduction_tree.heading("deduction", text="Pts")
        self.deduction_tree.heading("frame", text="Frame")
        self.deduction_tree.heading("detail", text="Detail")
        self.deduction_tree.column("label", width=140, anchor="w")
        self.deduction_tree.column("deduction", width=50, anchor="center")
        self.deduction_tree.column("frame", width=60, anchor="center")
        self.deduction_tree.column("detail", width=260, anchor="w")
        self.deduction_tree.tag_configure("mild", foreground="#b36b00")
        self.deduction_tree.tag_configure("strong", foreground="#b22222")
        self.deduction_tree.tag_configure("neutral", foreground="#666666")
        self.deduction_tree.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.deduction_tree.bind("<<TreeviewSelect>>", self._on_deduction_selected)

        summary_box = ttk.LabelFrame(left, text="Summary")
        summary_box.pack(fill=tk.BOTH, expand=True)
        self.summary_text = ScrolledText(summary_box, wrap=tk.WORD, height=18)
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        figure_box = ttk.LabelFrame(right, text="Localization")
        figure_box.pack(fill=tk.BOTH, expand=True)
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, master=figure_box)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, figure_box, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill=tk.X)

        self.state.keypoints_var.trace_add("write", lambda *_args: self.sync_dataset_dir())
        self.state.output_root_var.trace_add("write", lambda *_args: self.sync_dataset_dir())
        self.state.pose2sim_trc_var.trace_add("write", lambda *_args: self.sync_dataset_dir())
        self.state.fps_var.trace_add("write", lambda *_args: self.refresh_analysis())
        self.state.initial_rotation_correction_var.trace_add("write", lambda *_args: self.refresh_analysis())
        self.camera_name.trace_add("write", lambda *_args: self.refresh_plot())
        self.state.register_reconstruction_listener(lambda: self.after_idle(self.refresh_available_reconstructions))
        self.after_idle(self.refresh_available_reconstructions)

    def configure_shared_reconstruction_panel(self, panel: SharedReconstructionPanel) -> None:
        panel.configure_for_consumer(
            title="Reconstructions | Execution",
            refresh_callback=self.refresh_available_reconstructions,
            selection_callback=self.refresh_analysis,
            selectmode=self.shared_reconstruction_selectmode,
        )
        self.refresh_available_reconstructions()

    def _publish_reconstruction_rows(self, rows: list[dict[str, object]], defaults: list[str]) -> None:
        panel = self.state.shared_reconstruction_panel
        if panel is not None and self.state.active_reconstruction_consumer is self:
            panel.set_rows(rows, defaults)

    def _selected_reconstruction(self) -> str | None:
        selected = list(self.state.shared_reconstruction_selection)
        return selected[-1] if selected else None

    def _on_reconstruction_selection_changed(self) -> None:
        self.refresh_analysis()

    def sync_dataset_dir(self) -> None:
        self.refresh_available_reconstructions()

    def refresh_available_reconstructions(self) -> None:
        try:
            dataset_dir = current_dataset_dir(self.state)
            biomod_path = resolve_preview_biomod(dataset_dir)
            pose2sim_trc = (
                ROOT / self.state.pose2sim_trc_var.get() if self.state.pose2sim_trc_var.get().strip() else None
            )
            _output_dir, bundle, preview_state = load_shared_reconstruction_preview_state(
                self.state,
                preferred_names=["ekf_2d_acc", "ekf_3d"],
                fallback_count=2,
                include_3d=True,
                include_q=True,
                include_q_root=False,
                biomod_path=biomod_path,
                pose2sim_trc=pose2sim_trc,
            )
            available_q = preview_state.available_names
            self._publish_reconstruction_rows(preview_state.rows, preview_state.defaults[:1])
            self.bundle = bundle
            if available_q:
                self.refresh_analysis()
            else:
                self.execution_analysis = None
                self._show_empty_plot("No q-based reconstruction is available for execution analysis.")
        except Exception:
            pass

    def _show_empty_plot(self, message: str) -> None:
        """Render a placeholder when execution localization cannot be shown."""

        show_placeholder_figure(self.figure, self.canvas, message)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", message)

    def _update_camera_choices(self) -> None:
        camera_names: list[str] = []
        if self.pose_data is not None and getattr(self.pose_data, "camera_names", None) is not None:
            camera_names = [str(name) for name in self.pose_data.camera_names]
        elif isinstance(self.calibrations, dict):
            camera_names = [str(name) for name in self.calibrations.keys()]
        current = self.camera_name.get().strip()
        self.camera_box.configure(values=camera_names)
        if camera_names:
            self.camera_name.set(current if current in camera_names else camera_names[0])
        else:
            self.camera_name.set("")

    def refresh_analysis(self) -> None:
        try:
            with gui_busy_popup(self, title="Execution", message="Calcul des déductions d'exécution...") as popup:
                dataset_dir = current_dataset_dir(self.state)
                biomod_path = resolve_preview_biomod(dataset_dir)
                pose2sim_trc = (
                    ROOT / self.state.pose2sim_trc_var.get() if self.state.pose2sim_trc_var.get().strip() else None
                )
                popup.set_status("Chargement du bundle partagé...")
                self.bundle = get_cached_preview_bundle(
                    self.state, dataset_dir, biomod_path, pose2sim_trc, align_root=False
                )
                selected_name = self._selected_reconstruction()
                self.current_reconstruction_name = selected_name
                if selected_name is None:
                    self.execution_analysis = None
                    self._populate_jump_list()
                    self._populate_deduction_tree()
                    self._show_empty_plot("Select one reconstruction to inspect execution deductions.")
                    return
                recon_q = self.bundle.get("recon_q", {})
                recon_points = self.bundle.get("recon_3d", {})
                q = recon_q.get(selected_name)
                points_3d = recon_points.get(selected_name)
                if q is None or points_3d is None:
                    self.execution_analysis = None
                    self._populate_jump_list()
                    self._populate_deduction_tree()
                    self._show_empty_plot(
                        "Execution analysis requires q and 3D markers for the selected reconstruction."
                    )
                    return
                q = np.asarray(q, dtype=float)
                points_3d = np.asarray(points_3d, dtype=float)
                qdot = self.bundle.get("recon_qdot", {}).get(selected_name)
                if qdot is not None:
                    qdot = np.asarray(qdot, dtype=float)
                root_q, _full_q, q_name_list = preview_root_series_for_reconstruction(
                    bundle=self.bundle,
                    name=selected_name,
                    initial_rotation_correction=bool(self.state.initial_rotation_correction_var.get()),
                )
                if root_q is None or q_name_list is None:
                    self.execution_analysis = None
                    self._populate_jump_list()
                    self._populate_deduction_tree()
                    self._show_empty_plot("Unable to derive root kinematics for execution analysis.")
                    return
                fps = float(self.state.fps_var.get())
                popup.set_status("Segmentation partagée des sauts...")
                self.dd_analysis = shared_jump_analysis(
                    self.state,
                    reconstruction_name=selected_name,
                    root_q=np.asarray(root_q, dtype=float),
                    points_3d=points_3d,
                    fps=fps,
                    height_threshold=TRAMPOLINE_BED_HEIGHT_M,
                    height_threshold_range_ratio=0.20,
                    smoothing_window_s=0.15,
                    min_airtime_s=0.25,
                    min_gap_s=0.08,
                    min_peak_prominence_m=0.35,
                    contact_window_s=0.35,
                    full_q=None,
                    q_names=q_name_list,
                    angle_mode="euler",
                    analysis_start_frame=ANALYSIS_START_FRAME,
                    require_complete_jumps=True,
                )
                popup.set_status("Calcul des déductions localisées...")
                self.execution_analysis = analyze_execution_session(
                    self.dd_analysis,
                    q,
                    qdot,
                    q_name_list,
                    points_3d,
                    fps,
                )
                try:
                    self.calibrations, self.pose_data, _diagnostics = get_calibration_pose_data(
                        self.state,
                        keypoints_path=ROOT / self.state.keypoints_var.get(),
                        calib_path=ROOT / self.state.calib_var.get(),
                        **shared_pose_data_kwargs(self.state),
                    )
                except Exception:
                    self.calibrations, self.pose_data = None, None
                self.images_root = infer_execution_images_root(ROOT / self.state.keypoints_var.get())
                self._update_camera_choices()
                self._populate_jump_list()
                self._populate_deduction_tree()
                self.render_summary()
                self.refresh_plot()
        except Exception as exc:
            messagebox.showerror("Execution", str(exc))

    def _populate_jump_list(self) -> None:
        self._suspend_selection_callbacks = True
        try:
            previous = self._selected_jump_index()
            self.jump_list.delete(0, tk.END)
            if self.execution_analysis is None:
                return
            for jump in self.execution_analysis.jumps:
                self.jump_list.insert(
                    tk.END,
                    (
                        f"S{jump.jump_index} | {jump.classification} | "
                        f"ded {jump.capped_deduction:.1f} | "
                        f"{len(jump.deduction_events)} localized events"
                    ),
                )
            if self.execution_analysis.jumps:
                next_index = previous if previous is not None and previous < len(self.execution_analysis.jumps) else 0
                self.jump_list.selection_set(next_index)
        finally:
            self._suspend_selection_callbacks = False

    def _populate_deduction_tree(self) -> None:
        self._suspend_selection_callbacks = True
        try:
            previous = self._selected_deduction_index()
            for item in self.deduction_tree.get_children():
                self.deduction_tree.delete(item)
            jump = self._current_jump()
            if jump is None:
                return
            for event_idx, event in enumerate(jump.deduction_events):
                tag = "strong" if event.deduction >= 0.2 else "mild"
                self.deduction_tree.insert(
                    "",
                    "end",
                    iid=str(event_idx),
                    values=(
                        event.label,
                        f"{event.deduction:.1f}",
                        event.frame_idx,
                        event.detail,
                    ),
                    tags=(tag,),
                )
            if jump.deduction_events:
                next_index = previous if previous is not None and previous < len(jump.deduction_events) else 0
                self.deduction_tree.selection_set(str(next_index))
            elif not self.deduction_tree.get_children():
                self.deduction_tree.insert(
                    "",
                    "end",
                    iid="none",
                    values=("No deduction", "-", "-", "No discrete execution deduction on this jump."),
                    tags=("neutral",),
                )
        finally:
            self._suspend_selection_callbacks = False

    def _selected_jump_index(self) -> int | None:
        selection = self.jump_list.curselection()
        return int(selection[0]) if selection else None

    def _selected_deduction_index(self) -> int | None:
        selection = self.deduction_tree.selection()
        if not selection or selection[0] == "none":
            return None
        return int(selection[0])

    def _current_jump(self) -> ExecutionJumpAnalysis | None:
        if self.execution_analysis is None:
            return None
        jump_index = self._selected_jump_index()
        if jump_index is None or jump_index >= len(self.execution_analysis.jumps):
            return None
        return self.execution_analysis.jumps[jump_index]

    def _current_event(self) -> ExecutionDeductionEvent | None:
        jump = self._current_jump()
        if jump is None:
            return None
        deduction_index = self._selected_deduction_index()
        if deduction_index is None or deduction_index >= len(jump.deduction_events):
            return None
        return jump.deduction_events[deduction_index]

    def _on_jump_selected(self, _event=None) -> None:
        if self._suspend_selection_callbacks:
            return
        self._populate_deduction_tree()
        self.render_summary()
        self.refresh_plot()

    def _on_deduction_selected(self, _event=None) -> None:
        if self._suspend_selection_callbacks:
            return
        self.render_summary()
        self.refresh_plot()

    def render_summary(self) -> None:
        self.summary_text.delete("1.0", tk.END)
        if self.execution_analysis is None or self.current_reconstruction_name is None:
            self.summary_text.insert("1.0", "No execution analysis yet.")
            return
        lines = [
            f"Reconstruction: {reconstruction_label(self.current_reconstruction_name)}",
            f"Complete jumps analyzed: {len(self.execution_analysis.jumps)}",
            f"Total deduction: {self.execution_analysis.total_deduction:.1f}",
            f"Execution score: {self.execution_analysis.execution_score:.1f}",
            f"Time of flight: {self.execution_analysis.time_of_flight_s:.2f} s",
            "",
        ]
        jump = self._current_jump()
        if jump is not None:
            lines.extend(
                [
                    f"Selected jump: S{jump.jump_index}",
                    f"Classification: {jump.classification}",
                    f"Segment: frames {jump.segment.start}-{jump.segment.end}",
                    f"Jump deduction: {jump.capped_deduction:.1f} (raw {jump.total_deduction:.1f})",
                    "",
                ]
            )
        event = self._current_event()
        if event is not None:
            lines.extend(
                [
                    f"Focused deduction: {event.label}",
                    f"Points deducted: {event.deduction:.1f}",
                    f"Frame: {event.frame_idx}",
                    f"Metric: {event.metric_value:.2f} {event.metric_unit}",
                    f"Body region: {', '.join(event.keypoint_names)}",
                    f"Detail: {event.detail}",
                ]
            )
        self.summary_text.insert("1.0", "\n".join(lines))

    @staticmethod
    def _highlight_keypoints_3d(
        ax, frame_points: np.ndarray, keypoint_names: tuple[str, ...], color: str = "#d62728"
    ) -> None:
        """Overlay large markers on the keypoints implicated in one deduction."""

        for keypoint_name in keypoint_names:
            point = frame_points[KP_INDEX[keypoint_name]]
            if np.all(np.isfinite(point)):
                ax.scatter(point[0], point[1], point[2], s=90, c=color, marker="o", depthshade=False)

    @staticmethod
    def _highlight_keypoints_2d(
        ax, frame_points: np.ndarray, keypoint_names: tuple[str, ...], color: str = "#d62728"
    ) -> None:
        """Overlay 2D circles on the keypoints implicated in one deduction."""

        for keypoint_name in keypoint_names:
            point = frame_points[KP_INDEX[keypoint_name]]
            if np.all(np.isfinite(point)):
                ax.scatter(point[0], point[1], s=120, facecolors="none", edgecolors=color, linewidths=2.0, marker="o")

    def refresh_plot(self) -> None:
        jump = self._current_jump()
        if (
            self.execution_analysis is None
            or jump is None
            or self.current_reconstruction_name is None
            or self.bundle is None
        ):
            self._show_empty_plot("No execution analysis yet.")
            return
        points_3d = self.bundle.get("recon_3d", {}).get(self.current_reconstruction_name)
        if points_3d is None:
            self._show_empty_plot("No 3D markers are available for the selected reconstruction.")
            return
        event = self._current_event()
        focus_frame_idx = int(event.frame_idx) if event is not None else execution_focus_frame(jump)
        points_3d = np.asarray(points_3d, dtype=float)
        if focus_frame_idx >= points_3d.shape[0]:
            self._show_empty_plot("The focused execution frame is outside the available 3D trajectory.")
            return
        frame_points_3d = points_3d[focus_frame_idx]
        keypoint_names = event.keypoint_names if event is not None else tuple()
        bundle_frames = np.asarray(self.bundle.get("frames", np.arange(points_3d.shape[0], dtype=int)), dtype=int)
        frame_number = int(bundle_frames[min(focus_frame_idx, len(bundle_frames) - 1)])
        overlay_frame = build_execution_overlay_frame(
            camera_name=self.camera_name.get().strip(),
            frame_idx=focus_frame_idx,
            frame_number=frame_number,
            frame_points_3d=frame_points_3d,
            calibrations=self.calibrations,
            pose_data=self.pose_data,
            keypoint_names=keypoint_names,
            images_root=self.images_root,
        )

        self.figure.clear()
        metrics_ax = self.figure.add_subplot(2, 2, (1, 2))
        view_3d_ax = self.figure.add_subplot(2, 2, 3, projection="3d")
        view_2d_ax = self.figure.add_subplot(2, 2, 4)

        for metric_name, label, color in (
            ("knee_error_deg", "Knee error", "#4c72b0"),
            ("hip_error_deg", "Hip error", "#dd8452"),
            ("arm_raise_deg", "Arm raise", "#55a868"),
            ("tilt_deg", "Tilt", "#c44e52"),
        ):
            values = np.asarray(jump.metric_series.get(metric_name, np.array([], dtype=float)), dtype=float)
            if values.size:
                metrics_ax.plot(jump.metric_time_s, values, color=color, linewidth=1.6, label=label)
        for deduction_event in jump.deduction_events:
            local_time = jump.metric_time_s[deduction_event.local_frame_idx]
            color = "#d62728" if deduction_event.deduction >= 0.2 else "#b36b00"
            metrics_ax.axvline(local_time, color=color, linestyle="--", linewidth=1.1, alpha=0.7)
            metrics_ax.text(
                local_time,
                metrics_ax.get_ylim()[1] if metrics_ax.get_ylim()[1] != metrics_ax.get_ylim()[0] else 0.0,
                deduction_event.code,
                color=color,
                rotation=90,
                va="top",
                ha="right",
                fontsize=8,
            )
        metrics_ax.set_title(f"S{jump.jump_index} localized execution metrics")
        metrics_ax.set_xlabel("Time within jump (s)")
        metrics_ax.set_ylabel("Angle (deg)")
        metrics_ax.grid(alpha=0.25)
        handles, labels = metrics_ax.get_legend_handles_labels()
        if handles:
            unique = {}
            for handle, label in zip(handles, labels):
                unique[label] = handle
            metrics_ax.legend(list(unique.values()), list(unique.keys()), loc="upper right", fontsize=8)

        draw_skeleton_3d(view_3d_ax, frame_points_3d, "#7a7a7a", "3D markers")
        if keypoint_names:
            self._highlight_keypoints_3d(view_3d_ax, frame_points_3d, keypoint_names)
        set_equal_3d_limits(view_3d_ax, {"focus": points_3d}, focus_frame_idx)
        view_3d_ax.set_title(f"3D localization | frame {focus_frame_idx}")

        image_path = overlay_frame.image_path
        raw_points = overlay_frame.raw_points_2d
        projected_points = overlay_frame.projected_points_2d
        if image_path is not None and image_path.exists():
            image = plt.imread(str(image_path))
            view_2d_ax.imshow(image)
            self.image_layer_status_var.set(f"Image overlay: {display_path(image_path)}")
        elif overlay_frame.image_root is not None:
            self.image_layer_status_var.set(
                f"No image found for {overlay_frame.camera_name} frame {overlay_frame.frame_number} under {display_path(overlay_frame.image_root)}."
            )
        else:
            self.image_layer_status_var.set(
                "No image directory detected yet. 2D view shows 2D detections + reprojection."
            )
        if np.any(np.isfinite(raw_points)):
            draw_skeleton_2d(
                view_2d_ax,
                raw_points,
                "black",
                "2D data",
                marker_size=18.0,
                marker_fill=False,
                line_style="--",
                line_alpha=0.35,
                line_width_scale=0.9,
            )
        if np.any(np.isfinite(projected_points)):
            draw_skeleton_2d(
                view_2d_ax,
                projected_points,
                reconstruction_display_color(self.state, self.current_reconstruction_name),
                "reprojection",
                marker_size=12.0,
            )
        if keypoint_names:
            if np.any(np.isfinite(raw_points)):
                self._highlight_keypoints_2d(view_2d_ax, raw_points, keypoint_names, color="#d62728")
            if np.any(np.isfinite(projected_points)):
                self._highlight_keypoints_2d(view_2d_ax, projected_points, keypoint_names, color="#ff7f0e")
        all_points_2d = np.vstack(
            [
                raw_points[np.all(np.isfinite(raw_points), axis=1)],
                projected_points[np.all(np.isfinite(projected_points), axis=1)],
            ]
        )
        if all_points_2d.size:
            mins = np.min(all_points_2d, axis=0)
            maxs = np.max(all_points_2d, axis=0)
            margin = max(20.0, 0.1 * float(np.max(maxs - mins)))
            view_2d_ax.set_xlim(mins[0] - margin, maxs[0] + margin)
            view_2d_ax.set_ylim(maxs[1] + margin, mins[1] - margin)
        view_2d_ax.set_aspect("equal", adjustable="box")
        view_2d_ax.grid(alpha=0.20)
        view_2d_ax.set_title(f"2D localization | {overlay_frame.camera_name or 'no camera'}")
        handles, labels = view_2d_ax.get_legend_handles_labels()
        if handles:
            unique = {}
            for handle, label in zip(handles, labels):
                unique[label] = handle
            view_2d_ax.legend(list(unique.values()), list(unique.keys()), loc="best", fontsize=8)
        event_suffix = f"{event.label} | {event.detail}" if event is not None else "No discrete deduction on this jump"
        self.figure.suptitle(
            f"Execution analysis | {reconstruction_label(self.current_reconstruction_name)} | S{jump.jump_index} | {event_suffix}"
        )
        self.figure.tight_layout()
        self.canvas.draw_idle()


class CalibrationTab(ttk.Frame):
    def __init__(self, master, state: SharedAppState):
        super().__init__(master)
        self.state = state
        self.pose_data = None
        self.calibrations = None
        self.qc = None
        self.current_reconstruction_name = None
        self.current_reconstruction_payload: dict[str, np.ndarray] = {}
        self.current_reconstruction_summary: dict[str, object] = {}
        self.jump_analysis: DDSessionAnalysis | None = None
        self.uses_shared_reconstruction_panel = True
        self.shared_reconstruction_selectmode = "browse"

        controls = ttk.LabelFrame(self, text="Calibration QA")
        controls.pack(fill=tk.X, padx=10, pady=10)
        row = ttk.Frame(controls)
        row.pack(fill=tk.X, padx=8, pady=4)
        source_label = ttk.Label(row, text="2D source", width=10)
        source_label.pack(side=tk.LEFT)
        default_pose_mode = state.pose_data_mode_var.get().strip()
        if default_pose_mode not in ("raw", "annotated", "cleaned"):
            default_pose_mode = "cleaned"
        self.pose_data_mode = tk.StringVar(value=default_pose_mode)
        self.pose_mode_box = ttk.Combobox(
            row,
            textvariable=self.pose_data_mode,
            values=["raw", "cleaned"],
            width=10,
            state="readonly",
        )
        self.pose_mode_box.pack(side=tk.LEFT, padx=(0, 8))
        trim_label = ttk.Label(row, text="Trim worst 2D %", width=16)
        trim_label.pack(side=tk.LEFT)
        self.trim_fraction_var = tk.StringVar(value="15")
        self.trim_fraction_box = ttk.Combobox(
            row,
            textvariable=self.trim_fraction_var,
            values=["0", "5", "10", "15", "20"],
            width=6,
            state="readonly",
        )
        self.trim_fraction_box.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(row, text="Analyze / refresh", command=self.refresh_analysis).pack(side=tk.LEFT)
        self.status_var = tk.StringVar(
            value="2D: pairwise epipolar error after trimming the worst samples. 3D: reprojection + spatial uniformity."
        )
        ttk.Label(controls, textvariable=self.status_var, foreground="#4f5b66", justify=tk.LEFT).pack(
            fill=tk.X, padx=8, pady=(0, 4)
        )

        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=1)
        body.add(right, weight=2)

        summary_box = ttk.LabelFrame(left, text="Summary")
        summary_box.pack(fill=tk.BOTH, expand=True)
        self.summary = ScrolledText(summary_box, height=24, wrap=tk.WORD)
        self.summary.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        worst_box = ttk.LabelFrame(left, text="Worst frames")
        worst_box.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        ttk.Button(worst_box, text="Open in Cameras", command=self.open_selected_frame_in_cameras).pack(
            anchor="w", padx=6, pady=(6, 4)
        )
        self.worst_frame_list = tk.Listbox(worst_box, exportselection=False, height=10)
        self.worst_frame_list.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        figure_box = ttk.LabelFrame(right, text="2D / 3D calibration quality")
        figure_box.pack(fill=tk.BOTH, expand=True)
        self.figure = Figure(figsize=(11, 7))
        self.canvas = FigureCanvasTkAgg(self.figure, master=figure_box)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, figure_box, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill=tk.X)

        attach_tooltip(
            source_label,
            "Version of the 2D observations used for calibration analysis: raw, cleaned, or annotated when available.",
        )
        attach_tooltip(
            self.pose_mode_box,
            "Version of the 2D observations used for calibration analysis. `annotated` is available only when an annotation file exists for the current trial.",
        )
        attach_tooltip(
            trim_label,
            "Globally exclude the worst pairwise 2D epipolar samples before summarizing calibration quality.",
        )
        attach_tooltip(
            self.trim_fraction_box,
            "Globally exclude the worst pairwise 2D epipolar samples before summarizing calibration quality.",
        )
        attach_tooltip(
            self.summary,
            "Compact synthesis of pairwise epipolar consistency, worst frames, reprojection quality, and spatial non-uniformity.",
        )
        attach_tooltip(
            self.worst_frame_list,
            "Frames most likely to reveal calibration problems. Double-click to inspect them in Cameras.",
        )

        self.state.keypoints_var.trace_add("write", lambda *_args: self.sync_dataset_dir())
        self.state.output_root_var.trace_add("write", lambda *_args: self.sync_dataset_dir())
        self.state.calibration_correction_var.trace_add("write", lambda *_args: self.load_resources())
        self.pose_data_mode.trace_add("write", lambda *_args: self.load_resources())
        self.trim_fraction_var.trace_add("write", lambda *_args: self.refresh_analysis())
        self.state.register_reconstruction_listener(lambda: self.after_idle(self.refresh_available_reconstructions))
        self.state.register_shared_reconstruction_selection_listener(self._on_reconstruction_selection_changed)
        self.worst_frame_list.bind("<Double-Button-1>", lambda _event: self.open_selected_frame_in_cameras())
        self.sync_dataset_dir()

    def configure_shared_reconstruction_panel(self, panel: SharedReconstructionPanel) -> None:
        panel.configure_for_consumer(
            title="Reconstructions | Calibration",
            refresh_callback=self.refresh_available_reconstructions,
            selection_callback=self.refresh_analysis,
            selectmode=self.shared_reconstruction_selectmode,
        )
        self.refresh_available_reconstructions()

    def _publish_reconstruction_rows(self, rows: list[dict[str, object]], defaults: list[str]) -> None:
        panel = self.state.shared_reconstruction_panel
        if panel is not None and self.state.active_reconstruction_consumer is self:
            panel.set_rows(rows, defaults)

    def _on_reconstruction_selection_changed(self) -> None:
        self.refresh_analysis()

    def _selected_reconstruction(self) -> str | None:
        selected = list(self.state.shared_reconstruction_selection)
        return selected[-1] if selected else None

    def sync_dataset_dir(self) -> None:
        self.refresh_pose_mode_choices()
        self.refresh_available_reconstructions()
        self.load_resources()

    def _available_pose_modes(self) -> list[str]:
        keypoints_value = self.state.keypoints_var.get().strip()
        if not keypoints_value:
            return ["raw", "cleaned"]
        return available_model_pose_modes(self.state, ROOT / keypoints_value)

    def refresh_pose_mode_choices(self) -> None:
        modes = self._available_pose_modes()
        self.pose_mode_box.configure(values=modes)
        current = str(self.pose_data_mode.get()).strip()
        if current not in modes:
            fallback = "annotated" if "annotated" in modes else ("cleaned" if "cleaned" in modes else modes[0])
            self.pose_data_mode.set(fallback)

    def refresh_available_reconstructions(self) -> None:
        try:
            _output_dir, _bundle, preview_state = load_shared_reconstruction_preview_state(
                self.state,
                preferred_names=["triangulation_exhaustive", "triangulation_greedy", "ekf_3d", "ekf_2d_acc"],
                fallback_count=4,
                include_3d=True,
                include_q=True,
                include_q_root=True,
            )
            self._publish_reconstruction_rows(preview_state.rows, preview_state.defaults[:1])
        except Exception:
            pass

    def load_resources(self) -> None:
        try:
            self.refresh_pose_mode_choices()
            keypoints_path = ROOT / self.state.keypoints_var.get()
            self.calibrations, pose_data = get_cached_pose_data(
                self.state,
                keypoints_path=keypoints_path,
                calib_path=ROOT / self.state.calib_var.get(),
                **shared_pose_data_kwargs(self.state, data_mode=self.pose_data_mode.get()),
            )
            if str(self.pose_data_mode.get()).strip() == "annotated":
                pose_data = annotation_only_pose_data(
                    pose_data,
                    keypoints_path=keypoints_path,
                    annotations_path=existing_annotation_path_for_keypoints(self.state, keypoints_path),
                )
            self.pose_data = pose_data
            self.refresh_analysis()
        except Exception as exc:
            messagebox.showerror("Calibration", str(exc))

    def refresh_analysis(self) -> None:
        if self.pose_data is None or self.calibrations is None:
            return
        with gui_busy_popup(self, title="Calibration", message="Analyse de la qualité de calibration...") as popup:
            self.current_reconstruction_name = self._selected_reconstruction()
            payload = {}
            summary = {}
            if self.current_reconstruction_name:
                recon_dir = reconstruction_dir_by_name(
                    current_dataset_dir(self.state), self.current_reconstruction_name
                )
                if recon_dir is not None:
                    payload = load_bundle_payload(recon_dir)
                    summary = load_bundle_summary(recon_dir)
            selected_pose_mode = str(self.pose_data_mode.get()).strip()
            reconstruction_pose_mode = str(summary.get("pose_data_mode") or "").strip()
            if payload and reconstruction_pose_mode and reconstruction_pose_mode != selected_pose_mode:
                payload = {}
            self.current_reconstruction_payload = payload
            self.current_reconstruction_summary = summary
            self.jump_analysis = shared_jump_analysis_for_reconstruction(self.state, self.current_reconstruction_name)
            trim_fraction = max(0.0, float(self.trim_fraction_var.get() or "0")) / 100.0
            popup.set_status("Agrégation 2D/3D des métriques de calibration...")
            self.qc = compute_calibration_qc(
                self.pose_data,
                self.calibrations,
                reconstruction_payload=payload or None,
                trim_fraction=trim_fraction,
                spatial_bins=3,
            )
            self.status_var.set(
                f"Source: {self.pose_data_mode.get()} | 2D trim: "
                f"{int(round(self.qc.two_d.trim_fraction * 100.0))}% | Reconstruction: "
                f"{self.current_reconstruction_name or 'none selected'}"
                + (
                    ""
                    if not self.current_reconstruction_name
                    or not reconstruction_pose_mode
                    or reconstruction_pose_mode == selected_pose_mode
                    else f" | 3D hidden: reconstruction uses {reconstruction_pose_mode}"
                )
            )
            self.render_summary()
            self.refresh_worst_frame_list()
            self.refresh_plot()

    def refresh_worst_frame_list(self) -> None:
        self.worst_frame_list.delete(0, tk.END)
        if self.qc is None or self.pose_data is None:
            return
        for frame_number, value in self._worst_frames(self.qc.two_d.per_frame_mean_px, count=5):
            self.worst_frame_list.insert(tk.END, f"2D | frame {frame_number} | {value:.2f} px")
        if self.qc.three_d is not None:
            for frame_number, value in self._worst_frames(self.qc.three_d.per_frame_mean_px, count=5):
                self.worst_frame_list.insert(tk.END, f"3D | frame {frame_number} | {value:.2f} px")

    def _selected_worst_frame(self) -> tuple[str, int] | None:
        selection = self.worst_frame_list.curselection()
        if not selection:
            return None
        text = str(self.worst_frame_list.get(selection[0]))
        match = re.search(r"^(2D|3D)\s+\|\s+frame\s+(\d+)", text)
        if match is None:
            return None
        return match.group(1), int(match.group(2))

    def open_selected_frame_in_cameras(self) -> None:
        selected = self._selected_worst_frame()
        if selected is None or self.pose_data is None:
            return
        metric_kind, frame_number = selected
        camera_name = self._worst_camera_for_frame(metric_kind, frame_number)
        notebook = self.master if isinstance(self.master, ttk.Notebook) else None
        if notebook is None:
            return
        target_tab = None
        for tab_id in notebook.tabs():
            if notebook.tab(tab_id, "text") == "Cameras":
                target_tab = notebook.nametowidget(tab_id)
                notebook.select(tab_id)
                break
        if target_tab is not None and hasattr(target_tab, "show_specific_frame"):
            target_tab.show_specific_frame(frame_number=frame_number, camera_name=camera_name)

    def _worst_camera_for_frame(self, metric_kind: str, frame_number: int) -> str | None:
        if self.pose_data is None:
            return None
        camera_names = list(self.pose_data.camera_names)
        if frame_number not in set(int(frame) for frame in self.pose_data.frames):
            return None
        frame_idx = int(np.where(np.asarray(self.pose_data.frames, dtype=int) == int(frame_number))[0][0])
        if metric_kind == "3D" and self.current_reconstruction_payload:
            errors = np.asarray(self.current_reconstruction_payload.get("reprojection_error_per_view"), dtype=float)
            if errors.ndim == 3 and frame_idx < errors.shape[0]:
                per_camera = np.nanmean(errors[frame_idx], axis=0)
                finite = np.isfinite(per_camera)
                if np.any(finite):
                    return camera_names[int(np.nanargmax(per_camera))]
        per_camera_values = []
        for cam_idx, camera_name in enumerate(camera_names):
            values = frame_camera_epipolar_errors(
                self.pose_data,
                self.calibrations,
                frame_idx=frame_idx,
                camera_idx=cam_idx,
            )
            per_camera_values.append(np.nanmean(values))
        per_camera_values = np.asarray(per_camera_values, dtype=float)
        finite = np.isfinite(per_camera_values)
        if not np.any(finite):
            return camera_names[0] if camera_names else None
        return camera_names[int(np.nanargmax(per_camera_values))]

    def render_summary(self) -> None:
        self.summary.delete("1.0", tk.END)
        if self.qc is None or self.pose_data is None:
            self.summary.insert(tk.END, "No calibration analysis available.\n")
            return
        two_d = self.qc.two_d
        lines = [
            "2D calibration quality",
            f"- Trimmed worst 2D fraction: {int(round(two_d.trim_fraction * 100.0))}%",
            (
                "- Trim threshold: "
                + ("none" if two_d.trim_threshold_px is None else f"{two_d.trim_threshold_px:.2f} px")
            ),
            f"- Kept pairwise 2D samples: {two_d.kept_ratio * 100.0:.1f}%",
        ]
        worst_pairs = self._worst_camera_pairs(two_d.pairwise_median_px)
        if worst_pairs:
            lines.append("- Worst camera pairs (median epipolar px):")
            lines.extend(f"  - {cam_a} / {cam_b}: {value:.2f} px" for cam_a, cam_b, value in worst_pairs[:3])
        worst_frames_2d = self._worst_frames(two_d.per_frame_mean_px, count=5)
        if worst_frames_2d:
            lines.append("- Worst 2D frames (mean epipolar px):")
            lines.extend(f"  - frame {frame}: {value:.2f} px" for frame, value in worst_frames_2d)

        if self.qc.three_d is not None:
            three_d = self.qc.three_d
            reproj_mean = three_d.reprojection_summary.get("mean_px")
            reproj_std = three_d.reprojection_summary.get("std_px")
            lines.extend(
                [
                    "",
                    "3D calibration quality",
                    "- Reprojection mean/std: "
                    + (
                        "none"
                        if reproj_mean is None or reproj_std is None
                        else f"{float(reproj_mean):.2f} +/- {float(reproj_std):.2f} px"
                    ),
                    "- Spatial non-uniformity (CV / range): "
                    + (
                        "none"
                        if three_d.spatial_uniformity_cv is None or three_d.spatial_uniformity_range_px is None
                        else f"{three_d.spatial_uniformity_cv:.3f} / {three_d.spatial_uniformity_range_px:.2f} px"
                    ),
                ]
            )
            worst_frames_3d = self._worst_frames(three_d.per_frame_mean_px, count=5)
            if worst_frames_3d:
                lines.append("- Worst 3D frames (mean reprojection px):")
                lines.extend(f"  - frame {frame}: {value:.2f} px" for frame, value in worst_frames_3d)
        elif self.current_reconstruction_name and self.current_reconstruction_summary:
            reconstruction_pose_mode = str(self.current_reconstruction_summary.get("pose_data_mode") or "").strip()
            if reconstruction_pose_mode and reconstruction_pose_mode != str(self.pose_data_mode.get()).strip():
                lines.extend(
                    [
                        "",
                        "3D calibration quality",
                        (
                            "- Hidden because the selected reconstruction uses "
                            f"`{reconstruction_pose_mode}` while the tab uses `{self.pose_data_mode.get()}`."
                        ),
                    ]
                )

        self.summary.insert(tk.END, "\n".join(lines) + "\n")

    def refresh_plot(self) -> None:
        self.figure.clear()
        axes = self.figure.subplots(2, 2)
        if self.qc is None or self.pose_data is None:
            for ax in axes.flat:
                ax.axis("off")
            self.canvas.draw_idle()
            return
        two_d = self.qc.two_d
        camera_names = list(self.pose_data.camera_names)

        ax = axes[0, 0]
        matrix = np.asarray(two_d.pairwise_median_px, dtype=float)
        if np.isfinite(matrix).any():
            image = ax.imshow(matrix, cmap="magma")
            ax.set_xticks(range(len(camera_names)), camera_names, rotation=45, ha="right")
            ax.set_yticks(range(len(camera_names)), camera_names)
            ax.set_title("Pairwise epipolar median (px)")
            self.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "No finite pairwise epipolar data", ha="center", va="center", transform=ax.transAxes)

        ax = axes[0, 1]
        x = np.arange(len(camera_names))
        width = 0.38
        ax.bar(x - width / 2, np.nan_to_num(two_d.per_camera_median_px, nan=0.0), width=width, label="2D epi median")
        if self.qc.three_d is not None:
            per_camera = self.qc.three_d.reprojection_summary.get("per_camera", {})
            reproj_values = np.array(
                [float(per_camera.get(name, {}).get("mean_px") or 0.0) for name in camera_names],
                dtype=float,
            )
            ax.bar(x + width / 2, reproj_values, width=width, label="3D reproj mean")
        ax.set_xticks(x, camera_names, rotation=45, ha="right")
        ax.set_ylabel("px")
        ax.set_title("Per-camera quality")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.25, axis="y")

        ax = axes[1, 0]
        frames = np.asarray(self.pose_data.frames, dtype=int)
        plot_start_idx = min(5, len(frames))
        plot_frames = frames[plot_start_idx:]
        ax.plot(
            plot_frames,
            np.asarray(two_d.per_frame_mean_px, dtype=float)[plot_start_idx:],
            label="2D epipolar mean",
            linewidth=1.2,
        )
        if self.qc.three_d is not None:
            ax.plot(
                plot_frames,
                np.asarray(self.qc.three_d.per_frame_mean_px, dtype=float)[plot_start_idx:],
                label="3D reproj mean",
                linewidth=1.2,
            )
        if self.jump_analysis is not None:
            for idx, segment in enumerate(self.jump_analysis.jump_segments):
                if int(segment.end) < int(plot_frames[0]) or int(segment.start) > int(plot_frames[-1]):
                    continue
                ax.axvspan(
                    max(int(segment.start), int(plot_frames[0])),
                    min(int(segment.end), int(plot_frames[-1])),
                    color="#4c72b0",
                    alpha=0.06,
                    label="Detected jumps" if idx == 0 else None,
                )
        ax.set_title("Worst frames over time")
        ax.set_xlabel("Frame")
        ax.set_ylabel("px")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)

        ax = axes[1, 1]
        if self.qc.three_d is None or self.qc.three_d.point_positions.size == 0:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "Select a reconstruction with 3D points to inspect spatial calibration quality.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            cv_text = (
                "n/a"
                if self.qc.three_d.spatial_uniformity_cv is None
                else f"{self.qc.three_d.spatial_uniformity_cv:.3f}"
            )
            spatial_map = np.asarray(self.qc.three_d.spatial_xz_mean_px, dtype=float)
            if np.isfinite(spatial_map).any():
                image = ax.imshow(spatial_map.T, origin="lower", cmap="turbo", aspect="auto")
                ax.set_title("Spatial reprojection map (X/Z bins)\n" f"CV={cv_text}")
                ax.set_xlabel("X bins")
                ax.set_ylabel("Z bins")
                ax.set_xticks(range(spatial_map.shape[0]))
                ax.set_yticks(range(spatial_map.shape[1]))
                mean_value = float(np.nanmean(spatial_map)) if np.isfinite(spatial_map).any() else np.nan
                for x_idx in range(spatial_map.shape[0]):
                    for z_idx in range(spatial_map.shape[1]):
                        value = float(spatial_map[x_idx, z_idx])
                        count = int(self.qc.three_d.spatial_xz_count[x_idx, z_idx])
                        if np.isfinite(value) and count > 0:
                            ax.text(
                                x_idx,
                                z_idx,
                                f"{value:.1f}\n(n={count})",
                                ha="center",
                                va="center",
                                fontsize=7,
                                color=("white" if np.isfinite(mean_value) and value > mean_value else "black"),
                            )
                self.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="px")
            else:
                ax.axis("off")
                ax.text(0.5, 0.5, "No occupied spatial bins", ha="center", va="center", transform=ax.transAxes)

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _worst_camera_pairs(self, matrix: np.ndarray) -> list[tuple[str, str, float]]:
        if self.pose_data is None:
            return []
        pairs = []
        camera_names = list(self.pose_data.camera_names)
        for i in range(len(camera_names)):
            for j in range(i + 1, len(camera_names)):
                value = float(matrix[i, j])
                if np.isfinite(value):
                    pairs.append((camera_names[i], camera_names[j], value))
        pairs.sort(key=lambda item: item[2], reverse=True)
        return pairs

    def _worst_frames(self, values: np.ndarray, *, count: int) -> list[tuple[int, float]]:
        if self.pose_data is None:
            return []
        array = np.asarray(values, dtype=float)
        finite_idx = np.flatnonzero(np.isfinite(array))
        if finite_idx.size == 0:
            return []
        order = finite_idx[np.argsort(array[finite_idx])[::-1]]
        return [(int(self.pose_data.frames[idx]), float(array[idx])) for idx in order[:count]]


class CameraToolsTab(ttk.Frame):
    def __init__(self, master, state: SharedAppState):
        super().__init__(master)
        self.state = state
        self.base_pose_data = None
        self.pose_data = None
        self.calibrations = None
        self.metrics_rows = []
        self.flip_masks: dict[str, np.ndarray] = {}
        self.flip_diagnostics: dict[str, dict[str, object]] = {}
        self.flip_detail_arrays: dict[str, dict[str, np.ndarray]] = {}
        self.flip_frame_local_indices: list[int] = []
        self.images_root: Path | None = None
        self.uses_shared_reconstruction_panel = True
        self.shared_reconstruction_selectmode = "browse"

        controls = ttk.LabelFrame(self, text="Sélection de caméras + inspection flip L/R")
        controls.pack(fill=tk.X, padx=10, pady=10)

        row = ttk.Frame(controls)
        row.pack(fill=tk.X, padx=8, pady=4)
        best_n_label = ttk.Label(row, text="Best N", width=8)
        best_n_label.pack(side=tk.LEFT)
        self.best_n_var = tk.StringVar(value="4")
        best_n_entry = ttk.Entry(row, textvariable=self.best_n_var, width=5)
        best_n_entry.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(row, text="Load / refresh", command=self.load_resources).pack(side=tk.LEFT)
        ttk.Button(row, text="Use selected cameras", command=self.apply_selected_cameras).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(row, text="Select best", command=self.select_best_cameras).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(row, text="Clear camera filter", command=self.clear_camera_filter).pack(side=tk.LEFT, padx=(8, 0))

        self.camera_filter_status = tk.StringVar(value="Reconstructions will use all cameras.")
        self.calibration_pose_status_var = tk.StringVar(value="Calibration 2D data: none")
        ttk.Label(controls, textvariable=self.camera_filter_status, foreground="#4f5b66", justify=tk.LEFT).pack(
            fill=tk.X, padx=8, pady=(0, 4)
        )
        ttk.Label(controls, textvariable=self.calibration_pose_status_var, foreground="#4f5b66", justify=tk.LEFT).pack(
            fill=tk.X, padx=8, pady=(0, 2)
        )
        ttk.Label(
            controls,
            text=(
                "Scores shown: valid 2D coverage, detector confidence, epipolar coherence, confidence x coherence, "
                "reprojection quality, triangulation usage, epipolar decision support, and flip rates.\n"
                "Additional useful literature criteria: calibration uncertainty, baseline diversity, occlusion persistence, "
                "view angle to the motion plane, and temporal stability of 2D tracks."
            ),
            foreground="#4f5b66",
            justify=tk.LEFT,
            wraplength=1200,
        ).pack(fill=tk.X, padx=8, pady=(0, 4))

        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=3)
        body.add(right, weight=2)

        metrics_box = ttk.LabelFrame(left, text="Scores caméra")
        metrics_box.pack(fill=tk.BOTH, expand=True)
        self.metrics_tree = ttk.Treeview(
            metrics_box,
            columns=(
                "camera",
                "valid",
                "score",
                "epi",
                "weighted",
                "reproj",
                "good",
                "usage",
                "epi_decision",
                "epi_decision_smoothed",
                "flip_epi",
                "flip_epif",
                "flip_tri",
            ),
            show="headings",
            selectmode="extended",
            height=10,
        )
        headings = {
            "camera": "Camera",
            "valid": "Valid %",
            "score": "Score",
            "epi": "Epi",
            "weighted": "Conf x Epi",
            "reproj": "Reproj px",
            "good": "Good reproj %",
            "usage": "Triang use %",
            "epi_decision": "Epi score",
            "epi_decision_smoothed": "Epi score sm",
            "flip_epi": "Flip epi %",
            "flip_epif": "Flip epiF %",
            "flip_tri": "Flip tri %",
        }
        widths = {
            "camera": 120,
            "valid": 80,
            "score": 70,
            "epi": 70,
            "weighted": 95,
            "reproj": 85,
            "good": 95,
            "usage": 95,
            "epi_decision": 80,
            "epi_decision_smoothed": 90,
            "flip_epi": 85,
            "flip_epif": 90,
            "flip_tri": 85,
        }
        for column in self.metrics_tree["columns"]:
            self.metrics_tree.heading(column, text=headings[column])
            self.metrics_tree.column(column, width=widths[column], anchor="center")
        self.metrics_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        inspector_box = ttk.LabelFrame(right, text="Flip inspector")
        inspector_box.pack(fill=tk.BOTH, expand=True)
        inspector_controls = ttk.Frame(inspector_box)
        inspector_controls.pack(fill=tk.X, padx=8, pady=4)
        method_label = ttk.Label(inspector_controls, text="Method", width=9)
        method_label.pack(side=tk.LEFT)
        self.flip_method_var = tk.StringVar(value="epipolar")
        self.flip_method_box = ttk.Combobox(
            inspector_controls,
            textvariable=self.flip_method_var,
            values=[
                "epipolar",
                "epipolar_fast",
                "epipolar_viterbi",
                "epipolar_fast_viterbi",
                "triangulation_once",
                "triangulation_greedy",
                "triangulation_exhaustive",
            ],
            width=24,
            state="readonly",
        )
        self.flip_method_box.pack(side=tk.LEFT, padx=(0, 8))
        camera_label = ttk.Label(inspector_controls, text="Camera", width=8)
        camera_label.pack(side=tk.LEFT)
        self.flip_camera_var = tk.StringVar(value="")
        self.flip_camera_box = ttk.Combobox(
            inspector_controls, textvariable=self.flip_camera_var, width=18, state="readonly"
        )
        self.flip_camera_box.pack(side=tk.LEFT, padx=(0, 8))
        self.flip_applied_var = tk.BooleanVar(value=False)
        self.flip_check = ttk.Checkbutton(
            inspector_controls,
            text="Flip raw left/right",
            variable=self.flip_applied_var,
            command=self.render_flip_preview,
        )
        self.flip_check.pack(side=tk.LEFT, padx=(0, 8))
        self.flip_status_var = tk.StringVar(value="Press F to swap the raw 2D labels.")
        ttk.Label(inspector_controls, textvariable=self.flip_status_var, foreground="#4f5b66").pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

        image_controls = ttk.Frame(inspector_box)
        image_controls.pack(fill=tk.X, padx=8, pady=(0, 4))
        self.show_images_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            image_controls,
            text="Show images",
            variable=self.show_images_var,
            command=self.render_flip_preview,
        ).pack(side=tk.LEFT, padx=(0, 8))
        self.images_root_entry = LabeledEntry(image_controls, "Images root", "", browse=True, directory=True)
        self.images_root_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        overlay_label = ttk.Label(image_controls, text="QA overlay", width=10)
        overlay_label.pack(side=tk.LEFT, padx=(8, 0))
        self.qa_overlay_var = tk.StringVar(value="none")
        self.qa_overlay_box = ttk.Combobox(
            image_controls,
            textvariable=self.qa_overlay_var,
            values=["none", "2D epipolar", "3D reproj", "3D excluded"],
            width=14,
            state="readonly",
        )
        self.qa_overlay_box.pack(side=tk.LEFT, padx=(0, 8))

        inspector_body = ttk.Panedwindow(inspector_box, orient=tk.HORIZONTAL)
        inspector_body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        frame_panel = ttk.Frame(inspector_body)
        preview_panel = ttk.Frame(inspector_body)
        inspector_body.add(frame_panel, weight=1)
        inspector_body.add(preview_panel, weight=2)

        ttk.Label(frame_panel, text="Frames suspectes / candidates").pack(anchor="w", pady=(0, 4))
        self.flip_frame_list = tk.Listbox(frame_panel, exportselection=False, height=12)
        self.flip_frame_list.pack(fill=tk.BOTH, expand=True)
        self.flip_details = ScrolledText(frame_panel, height=9, wrap=tk.WORD)
        self.flip_details.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self.flip_figure = Figure(figsize=(7, 6))
        self.flip_canvas = FigureCanvasTkAgg(self.flip_figure, master=preview_panel)
        self.flip_canvas_widget = self.flip_canvas.get_tk_widget()
        self.flip_canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.flip_toolbar = NavigationToolbar2Tk(self.flip_canvas, preview_panel, pack_toolbar=False)
        self.flip_toolbar.update()
        self.flip_toolbar.pack(fill=tk.X)

        attach_tooltip(best_n_label, "Nombre de caméras à présélectionner automatiquement selon le classement courant.")
        attach_tooltip(best_n_entry, "Nombre de caméras à présélectionner automatiquement selon le classement courant.")
        attach_tooltip(
            self.metrics_tree,
            "Scores comparatifs pour choisir un sous-ensemble de caméras plus stable pour la reconstruction, y compris le support moyen des décisions de flip épipolaires.",
        )
        attach_tooltip(
            method_label,
            "Méthode de diagnostic de flip L/R: cohérence épipolaire Sampson, cohérence épipolaire rapide par distance symétrique, ou triangulation/reprojection.",
        )
        attach_tooltip(
            self.flip_method_box,
            "Méthode de diagnostic de flip L/R: cohérence épipolaire Sampson, cohérence épipolaire rapide par distance symétrique, ou triangulation/reprojection.",
        )
        attach_tooltip(camera_label, "Caméra isolée à inspecter pour les frames suspectes.")
        attach_tooltip(self.flip_camera_box, "Caméra isolée à inspecter pour les frames suspectes.")
        attach_tooltip(
            self.flip_check, "Permute gauche/droite sur les données 2D brutes affichées. Raccourci clavier: F."
        )
        attach_tooltip(self.images_root_entry, "Dossier d'images utilisé comme fond du preview caméra, si disponible.")
        attach_tooltip(
            self.qa_overlay_box,
            "Overlay calibration QA on the 2D image: local 2D epipolar error, selected 3D reprojection error, or 3D excluded keypoints.",
        )
        attach_tooltip(self.flip_frame_list, "Frames suspectes ou candidates pour la caméra et la méthode choisies.")
        attach_tooltip(
            self.flip_details, "Détails des coûts géométriques, temporels et combinés pour la frame sélectionnée."
        )

        self.flip_method_var.trace_add("write", lambda *_args: self.refresh_flip_frame_list())
        self.flip_camera_var.trace_add("write", lambda *_args: self.refresh_flip_frame_list())
        self.qa_overlay_var.trace_add("write", lambda *_args: self.render_flip_preview())
        self.flip_frame_list.bind("<<ListboxSelect>>", lambda _event: self.render_flip_preview())
        for widget in (self.flip_frame_list, self.flip_canvas_widget, self.flip_method_box, self.flip_camera_box):
            widget.bind("<KeyPress-f>", self.toggle_flip_current_frame)
            widget.bind("<Enter>", lambda _event, w=widget: w.focus_set())
        self.state.keypoints_var.trace_add("write", lambda *_args: self.sync_dataset_dir())
        self.state.output_root_var.trace_add("write", lambda *_args: self.sync_dataset_dir())
        self.state.pose_data_mode_var.trace_add("write", lambda *_args: self.load_resources())
        self.state.calibration_correction_var.trace_add("write", lambda *_args: self.load_resources())
        self.state.register_reconstruction_listener(lambda: self.after_idle(self.refresh_available_reconstructions))
        self.state.register_shared_reconstruction_selection_listener(self._on_reconstruction_selection_changed)
        self.state.selected_camera_names_var.trace_add("write", lambda *_args: self.update_camera_filter_status())
        self.sync_dataset_dir()

    def toggle_flip_current_frame(self, _event=None) -> str:
        self.flip_applied_var.set(not self.flip_applied_var.get())
        self.render_flip_preview()
        return "break"

    def sync_dataset_dir(self) -> None:
        self.images_root = infer_execution_images_root(ROOT / self.state.keypoints_var.get())
        self.images_root_entry.var.set("" if self.images_root is None else display_path(self.images_root))
        self.refresh_available_reconstructions()
        self.update_camera_filter_status()
        self.load_resources()

    def update_camera_filter_status(self) -> None:
        selected = current_selected_camera_names(self.state)
        self.camera_filter_status.set(
            "Reconstructions will use all cameras."
            if not selected
            else f"Reconstructions will use: {format_camera_names(selected)}"
        )
        for item in self.metrics_tree.selection():
            self.metrics_tree.selection_remove(item)
        for camera_name in selected:
            if self.metrics_tree.exists(camera_name):
                self.metrics_tree.selection_add(camera_name)

    def _on_reconstruction_selection_changed(self) -> None:
        if self.pose_data is None:
            return
        self.refresh_metrics()
        self.refresh_flip_frame_list()

    def configure_shared_reconstruction_panel(self, panel: SharedReconstructionPanel) -> None:
        panel.configure_for_consumer(
            title="Reconstructions | Caméras",
            refresh_callback=self.refresh_available_reconstructions,
            selection_callback=self._on_reconstruction_selection_changed,
            selectmode=self.shared_reconstruction_selectmode,
            allow_empty_selection=True,
        )
        self.refresh_available_reconstructions()

    def refresh_available_reconstructions(self) -> None:
        try:
            _output_dir, _bundle, preview_state = load_shared_reconstruction_preview_state(
                self.state,
                preferred_names=["triangulation_exhaustive", "triangulation_greedy", "pose2sim", "ekf_3d"],
                fallback_count=1,
                include_3d=True,
                include_q=True,
                include_q_root=True,
            )
            panel = self.state.shared_reconstruction_panel
            if panel is not None and self.state.active_reconstruction_consumer is self:
                panel.set_rows(preview_state.rows, preview_state.defaults[:1])
        except Exception:
            pass

    def _selected_reconstruction(self) -> str | None:
        selected = list(self.state.shared_reconstruction_selection)
        return selected[-1] if selected else None

    def load_resources(self) -> None:
        try:
            self.calibrations, self.base_pose_data = get_cached_pose_data(
                self.state,
                keypoints_path=ROOT / self.state.keypoints_var.get(),
                calib_path=ROOT / self.state.calib_var.get(),
                **shared_pose_data_kwargs(self.state),
            )
            self.calibrations, self.pose_data, correction_diagnostics = get_calibration_pose_data(
                self.state,
                keypoints_path=ROOT / self.state.keypoints_var.get(),
                calib_path=ROOT / self.state.calib_var.get(),
                **shared_pose_data_kwargs(self.state),
            )
            correction_mode = current_calibration_correction_mode(self.state)
            if correction_mode == "none":
                self.calibration_pose_status_var.set("Calibration 2D data: none")
            else:
                if correction_mode == "flip_epipolar":
                    method = "epipolar"
                elif correction_mode == "flip_epipolar_fast":
                    method = "epipolar_fast"
                elif correction_mode == "flip_epipolar_viterbi":
                    method = "epipolar_viterbi"
                elif correction_mode == "flip_epipolar_fast_viterbi":
                    method = "epipolar_fast_viterbi"
                else:
                    method = "triangulation"
                suspect_count = (
                    int(correction_diagnostics.get("n_camera_frame_flip_suspects", 0)) if correction_diagnostics else 0
                )
                self.calibration_pose_status_var.set(
                    f"Calibration 2D data: flip {method} ({suspect_count} camera-frames suspectes)"
                )
            self.ensure_flip_diagnostics()
            self.refresh_metrics()
            self.refresh_flip_controls()
        except Exception as exc:
            messagebox.showerror("Caméras", str(exc))

    def ensure_flip_diagnostics(self) -> None:
        if self.base_pose_data is None or self.calibrations is None:
            return
        dataset_dir = current_dataset_dir(self.state)
        pose_kwargs = shared_pose_data_kwargs(self.state)
        for method in (
            "epipolar",
            "epipolar_fast",
            "epipolar_viterbi",
            "epipolar_fast_viterbi",
            "triangulation_once",
            "triangulation_greedy",
            "triangulation_exhaustive",
        ):
            suspect_mask, diagnostics, _compute_time_s, cache_path, _flip_source = (
                load_or_compute_left_right_flip_cache(
                    output_dir=dataset_dir,
                    pose_data=self.base_pose_data,
                    calibrations=self.calibrations,
                    method=method,
                    pose_data_mode=str(pose_kwargs["data_mode"]),
                    pose_filter_window=int(pose_kwargs["smoothing_window"]),
                    pose_outlier_threshold_ratio=float(pose_kwargs["outlier_threshold_ratio"]),
                    pose_amplitude_lower_percentile=float(pose_kwargs["lower_percentile"]),
                    pose_amplitude_upper_percentile=float(pose_kwargs["upper_percentile"]),
                    improvement_ratio=float(self.state.flip_improvement_ratio_var.get()),
                    min_gain_px=float(self.state.flip_min_gain_px_var.get()),
                    min_other_cameras=int(self.state.flip_min_other_cameras_var.get()),
                    restrict_to_outliers=bool(self.state.flip_restrict_to_outliers_var.get()),
                    outlier_percentile=float(self.state.flip_outlier_percentile_var.get()),
                    outlier_floor_px=float(self.state.flip_outlier_floor_px_var.get()),
                    tau_px=(
                        DEFAULT_EPIPOLAR_THRESHOLD_PX
                        if method in {"epipolar", "epipolar_fast", "epipolar_viterbi", "epipolar_fast_viterbi"}
                        else DEFAULT_REPROJECTION_THRESHOLD_PX
                    ),
                    temporal_weight=float(self.state.flip_temporal_weight_var.get()),
                    temporal_tau_px=float(self.state.flip_temporal_tau_px_var.get()),
                )
            )
            self.flip_masks[method] = suspect_mask
            self.flip_diagnostics[method] = diagnostics
            self.flip_detail_arrays[method] = load_flip_detail_arrays(cache_path)
        self.flip_masks["triangulation"] = self.flip_masks.get("triangulation_exhaustive")
        self.flip_diagnostics["triangulation"] = self.flip_diagnostics.get("triangulation_exhaustive", {})
        self.flip_detail_arrays["triangulation"] = self.flip_detail_arrays.get("triangulation_exhaustive", {})

    def _reference_payload(self) -> dict[str, np.ndarray]:
        reference_name = (self._selected_reconstruction() or "").strip()
        if not reference_name:
            return {}
        recon_dir = reconstruction_dir_by_name(current_dataset_dir(self.state), reference_name)
        return {} if recon_dir is None else load_bundle_payload(recon_dir)

    def refresh_metrics(self) -> None:
        if self.pose_data is None:
            return
        payload = self._reference_payload()
        self.metrics_rows = compute_camera_metric_rows(
            self.pose_data,
            epipolar_coherence=payload.get("epipolar_coherence"),
            reprojection_error_per_view=payload.get("reprojection_error_per_view"),
            excluded_views=payload.get("excluded_views"),
            flip_masks=self.flip_masks,
            flip_detail_arrays=self.flip_detail_arrays,
            good_reprojection_threshold_px=DEFAULT_REPROJECTION_THRESHOLD_PX,
        )
        previous = set(self.metrics_tree.selection())
        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)
        for row in self.metrics_rows:
            self.metrics_tree.insert(
                "",
                "end",
                iid=row.camera_name,
                values=(
                    row.camera_name,
                    self._fmt_pct(row.valid_ratio),
                    self._fmt_float(row.mean_score),
                    self._fmt_float(row.mean_epipolar_coherence),
                    self._fmt_float(row.weighted_confidence),
                    self._fmt_float(row.reprojection_mean_px),
                    self._fmt_pct(row.reprojection_good_frame_ratio),
                    self._fmt_pct(row.triangulation_usage_ratio),
                    self._fmt_float(row.epipolar_decision_score),
                    self._fmt_float(row.epipolar_decision_score_smoothed),
                    self._fmt_pct(row.flip_rate_epipolar),
                    self._fmt_pct(row.flip_rate_epipolar_fast),
                    self._fmt_pct(row.flip_rate_triangulation),
                ),
            )
        for camera_name in previous:
            if self.metrics_tree.exists(camera_name):
                self.metrics_tree.selection_add(camera_name)
        self.update_camera_filter_status()

    def apply_selected_cameras(self) -> None:
        self.state.selected_camera_names_var.set(",".join(self.metrics_tree.selection()))

    def clear_camera_filter(self) -> None:
        self.state.selected_camera_names_var.set("")

    def select_best_cameras(self) -> None:
        try:
            count = int(self.best_n_var.get())
        except ValueError:
            count = 4
        best_names = suggest_best_camera_names(self.metrics_rows, count)
        for item in self.metrics_tree.selection():
            self.metrics_tree.selection_remove(item)
        for camera_name in best_names:
            if self.metrics_tree.exists(camera_name):
                self.metrics_tree.selection_add(camera_name)

    def refresh_flip_controls(self) -> None:
        if self.base_pose_data is None:
            return
        camera_names = list(self.base_pose_data.camera_names)
        self.flip_camera_box.configure(values=camera_names)
        if self.flip_camera_var.get() not in camera_names:
            self.flip_camera_var.set(camera_names[0] if camera_names else "")
        self.refresh_flip_frame_list()

    def refresh_flip_frame_list(self) -> None:
        self.flip_frame_local_indices = []
        self.flip_frame_list.delete(0, tk.END)
        self.flip_details.delete("1.0", tk.END)
        if self.base_pose_data is None:
            self.render_flip_preview()
            return
        method = self.flip_method_var.get()
        camera_name = self.flip_camera_var.get()
        if method not in self.flip_masks or camera_name not in self.base_pose_data.camera_names:
            self.render_flip_preview()
            return
        cam_idx = list(self.base_pose_data.camera_names).index(camera_name)
        detail_arrays = self.flip_detail_arrays.get(method, {})
        candidate_mask = (
            np.asarray(detail_arrays.get("candidate_mask"), dtype=bool) if "candidate_mask" in detail_arrays else None
        )
        suspect_mask = self.flip_masks[method]
        if np.any(suspect_mask[cam_idx]):
            local_indices = np.flatnonzero(suspect_mask[cam_idx]).tolist()
        elif candidate_mask is not None and candidate_mask.ndim == 2 and np.any(candidate_mask[cam_idx]):
            local_indices = np.flatnonzero(candidate_mask[cam_idx]).tolist()
        else:
            local_indices = []
        self.flip_frame_local_indices = local_indices
        for local_idx in local_indices:
            frame_number = int(self.base_pose_data.frames[local_idx])
            nominal = self._flip_cost(detail_arrays, "nominal_combined_costs", cam_idx, local_idx)
            swapped = self._flip_cost(detail_arrays, "swapped_combined_costs", cam_idx, local_idx)
            decision_score = self._flip_cost(detail_arrays, "decision_scores", cam_idx, local_idx)
            smoothed_score = self._flip_cost(detail_arrays, "decision_scores_smoothed", cam_idx, local_idx)
            label = (
                f"{'flip' if suspect_mask[cam_idx, local_idx] else 'candidate'} | frame {frame_number} | "
                f"{self._fmt_float(nominal)} -> {self._fmt_float(swapped)} | "
                f"dec {self._fmt_float(decision_score)} / sm {self._fmt_float(smoothed_score)}"
            )
            self.flip_frame_list.insert(tk.END, label)
        if local_indices:
            self.flip_frame_list.selection_set(0)
        self.render_flip_preview()

    def _selected_flip_frame_local_idx(self) -> int | None:
        selection = self.flip_frame_list.curselection()
        if selection:
            idx = int(selection[0])
            if idx < len(self.flip_frame_local_indices):
                return self.flip_frame_local_indices[idx]
        return self.flip_frame_local_indices[0] if self.flip_frame_local_indices else None

    def show_specific_frame(self, *, frame_number: int, camera_name: str | None = None) -> None:
        if self.base_pose_data is None:
            return
        frames = np.asarray(self.base_pose_data.frames, dtype=int)
        matches = np.flatnonzero(frames == int(frame_number))
        if matches.size == 0:
            return
        if camera_name and camera_name in self.base_pose_data.camera_names:
            self.flip_camera_var.set(camera_name)
        local_idx = int(matches[0])
        self.flip_frame_local_indices = [local_idx]
        self.flip_frame_list.delete(0, tk.END)
        selected_camera = self.flip_camera_var.get() or (
            self.base_pose_data.camera_names[0] if self.base_pose_data.camera_names else ""
        )
        self.flip_frame_list.insert(tk.END, f"manual | frame {int(frame_number)} | {selected_camera}")
        self.flip_frame_list.selection_set(0)
        self.render_flip_preview()

    def _qa_overlay_data(
        self, camera_name: str, frame_local_idx: int
    ) -> tuple[str, np.ndarray | None, np.ndarray | None, str | None]:
        mode = str(getattr(getattr(self, "qa_overlay_var", None), "get", lambda: "none")()).strip().lower()
        if mode == "2d epipolar":
            cam_idx = list(self.pose_data.camera_names).index(camera_name)
            values = frame_camera_epipolar_errors(
                self.pose_data, self.calibrations, frame_idx=frame_local_idx, camera_idx=cam_idx
            )
            return "2D epipolar", values, None, "turbo"
        if mode not in {"3d reproj", "3d excluded"}:
            return "none", None, None, None
        payload = self._reference_payload()
        if mode == "3d reproj":
            errors = payload.get("reprojection_error_per_view")
            if errors is None:
                return "3D reproj", None, None, None
            errors = np.asarray(errors, dtype=float)
            cam_idx = list(self.pose_data.camera_names).index(camera_name)
            if errors.ndim == 3 and frame_local_idx < errors.shape[0] and cam_idx < errors.shape[2]:
                return "3D reproj", np.asarray(errors[frame_local_idx, :, cam_idx], dtype=float), None, "turbo"
        if mode == "3d excluded":
            excluded = payload.get("excluded_views")
            if excluded is None:
                return "3D excluded", None, None, None
            excluded = np.asarray(excluded, dtype=bool)
            cam_idx = list(self.pose_data.camera_names).index(camera_name)
            if excluded.ndim == 3 and frame_local_idx < excluded.shape[0] and cam_idx < excluded.shape[2]:
                return "3D excluded", None, np.asarray(excluded[frame_local_idx, :, cam_idx], dtype=bool), None
        return "none", None, None, None

    def _reference_projection(self, camera_name: str, frame_local_idx: int) -> tuple[np.ndarray | None, str, str]:
        reference_name = (self._selected_reconstruction() or "").strip()
        if not reference_name:
            return None, "none", "#444444"
        reference_label = reconstruction_legend_label(self.state, reference_name)
        recon_dir = reconstruction_dir_by_name(current_dataset_dir(self.state), reference_name)
        if recon_dir is None:
            return None, reference_label, reconstruction_display_color(self.state, reference_name)
        payload = load_bundle_payload(recon_dir)
        points_3d = np.asarray(payload.get("points_3d"), dtype=float) if "points_3d" in payload else None
        if points_3d is None or points_3d.ndim != 3 or frame_local_idx >= points_3d.shape[0]:
            return None, reference_label, reconstruction_display_color(self.state, reference_name)
        projected = np.full((points_3d.shape[1], 2), np.nan, dtype=float)
        calibration = self.calibrations[camera_name]
        for kp_idx, point_3d in enumerate(points_3d[frame_local_idx]):
            if np.all(np.isfinite(point_3d)):
                projected[kp_idx] = calibration.project_point(point_3d)
        return projected, reference_label, reconstruction_display_color(self.state, reference_name)

    def render_flip_preview(self) -> None:
        self.flip_figure.clear()
        if self.base_pose_data is None or self.calibrations is None:
            ax = self.flip_figure.subplots(1, 1)
            ax.text(0.5, 0.5, "No 2D data loaded", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self.flip_canvas.draw_idle()
            return
        method = self.flip_method_var.get()
        camera_name = self.flip_camera_var.get()
        frame_local_idx = self._selected_flip_frame_local_idx()
        if frame_local_idx is None or camera_name not in self.base_pose_data.camera_names:
            ax = self.flip_figure.subplots(1, 1)
            ax.text(
                0.5, 0.5, "No flagged frame for this camera/method", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_axis_off()
            self.flip_canvas.draw_idle()
            return
        cam_idx = list(self.base_pose_data.camera_names).index(camera_name)
        frame_number = int(self.base_pose_data.frames[frame_local_idx])
        raw_points = np.asarray(self.base_pose_data.keypoints[cam_idx, frame_local_idx], dtype=float)
        display_raw_points = swap_left_right_keypoints(raw_points) if self.flip_applied_var.get() else raw_points
        projected_points, projected_label, projected_color = self._reference_projection(camera_name, frame_local_idx)
        width, height = self.calibrations[camera_name].image_size
        finite_raw = display_raw_points[np.all(np.isfinite(display_raw_points), axis=1)]
        finite_projected = (
            projected_points[np.all(np.isfinite(projected_points), axis=1)]
            if projected_points is not None
            else np.empty((0, 2))
        )
        finite = (
            np.vstack([arr for arr in (finite_raw, finite_projected) if arr.size])
            if (finite_raw.size or finite_projected.size)
            else np.empty((0, 2))
        )
        x_limits, y_limits = crop_limits_from_points(finite, width=float(width), height=float(height), margin=0.2)

        ax = self.flip_figure.subplots(1, 1)
        images_root = (
            Path(self.images_root_entry.get().strip()) if self.images_root_entry.get().strip() else self.images_root
        )
        background_image = (
            load_camera_background_image(
                images_root,
                camera_name,
                frame_number,
                image_reader=plt.imread,
            )
            if self.show_images_var.get()
            else None
        )
        layers = [
            SkeletonLayer2D(
                points=display_raw_points,
                color=("white" if background_image is not None else "#000000"),
                label="Raw 2D",
                marker_size=28.0,
                marker_fill=False,
                marker_edge_width=1.9,
                line_alpha=0.55,
                line_style=(0, (2.0, 2.2)),
                line_width_scale=0.6,
            )
        ]
        if projected_points is not None:
            layers.append(
                SkeletonLayer2D(
                    points=projected_points,
                    color=projected_color,
                    label=projected_label,
                    marker_size=20.0,
                )
            )
        render_camera_frame_2d(
            ax,
            width=width,
            height=height,
            title=f"Raw {'(swapped)' if self.flip_applied_var.get() else ''} + reprojection",
            layers=layers,
            draw_skeleton_fn=draw_skeleton_2d,
            background_image=background_image,
            draw_background_fn=draw_2d_background_image,
            x_limits=x_limits,
            y_limits=y_limits,
            hide_axes=False,
            show_grid=True,
            grid_alpha=0.18,
            xlabel="x (px)",
            ylabel="y (px)",
        )
        overlay_label, overlay_values, overlay_mask, overlay_cmap = self._qa_overlay_data(camera_name, frame_local_idx)
        overlay_scatter = draw_point_value_overlay(
            ax,
            PointValueOverlay2D(
                label=overlay_label,
                points=display_raw_points,
                values=overlay_values,
                mask=overlay_mask,
                cmap=overlay_cmap,
            ),
        )
        if overlay_scatter is not None:
            self.flip_figure.colorbar(overlay_scatter, ax=ax, fraction=0.046, pad=0.04, label=overlay_label)
        side_handles = [
            plt.Line2D(
                [],
                [],
                color="#666666",
                marker="^",
                linestyle="None",
                markersize=7,
                markerfacecolor="none",
                markeredgewidth=1.4,
                label="Left side",
            ),
            plt.Line2D(
                [],
                [],
                color="#666666",
                marker="s",
                linestyle="None",
                markersize=7,
                markerfacecolor="none",
                markeredgewidth=1.4,
                label="Right side",
            ),
        ]
        ax.legend(side_handles, ["Left side", "Right side"], loc="best", fontsize=8)
        detail_arrays = self.flip_detail_arrays.get(method, {})
        suspect = bool(self.flip_masks.get(method, np.zeros((0, 0), dtype=bool))[cam_idx, frame_local_idx])
        self.flip_figure.suptitle(
            f"{camera_name} | frame {frame_number} | {method} | suspect={'yes' if suspect else 'no'} | "
            f"reference={projected_label if projected_points is not None else 'none'}"
        )
        self.flip_figure.tight_layout()
        self.flip_canvas.draw_idle()
        self.flip_status_var.set("Press F to swap the raw 2D labels and compare them to the reprojection.")
        self.flip_details.delete("1.0", tk.END)
        self.flip_details.insert(
            "1.0",
            "\n".join(
                [
                    f"camera={camera_name}",
                    f"frame={frame_number}",
                    f"method={method}",
                    f"raw_swapped={'yes' if self.flip_applied_var.get() else 'no'}",
                    f"reference={projected_label if projected_points is not None else 'none'}",
                    f"qa_overlay={overlay_label}",
                    f"suspect={'yes' if suspect else 'no'}",
                    f"candidate={'yes' if self._flip_flag(detail_arrays, 'candidate_mask', cam_idx, frame_local_idx) else 'no'}",
                    f"temporal_support={'yes' if self._flip_flag(detail_arrays, 'temporal_support_mask', cam_idx, frame_local_idx) else 'no'}",
                    f"nominal geometric={self._fmt_float(self._flip_cost(detail_arrays, 'nominal_geometric_costs', cam_idx, frame_local_idx))}",
                    f"swapped geometric={self._fmt_float(self._flip_cost(detail_arrays, 'swapped_geometric_costs', cam_idx, frame_local_idx))}",
                    f"nominal temporal={self._fmt_float(self._flip_cost(detail_arrays, 'nominal_temporal_costs', cam_idx, frame_local_idx))}",
                    f"swapped temporal={self._fmt_float(self._flip_cost(detail_arrays, 'swapped_temporal_costs', cam_idx, frame_local_idx))}",
                    f"nominal combined={self._fmt_float(self._flip_cost(detail_arrays, 'nominal_combined_costs', cam_idx, frame_local_idx))}",
                    f"swapped combined={self._fmt_float(self._flip_cost(detail_arrays, 'swapped_combined_costs', cam_idx, frame_local_idx))}",
                    f"decision score={self._fmt_float(self._flip_cost(detail_arrays, 'decision_scores', cam_idx, frame_local_idx))}",
                    f"decision score smoothed={self._fmt_float(self._flip_cost(detail_arrays, 'decision_scores_smoothed', cam_idx, frame_local_idx))}",
                ]
            ),
        )

    @staticmethod
    def _flip_cost(detail_arrays: dict[str, np.ndarray], key: str, cam_idx: int, frame_idx: int) -> float | None:
        values = detail_arrays.get(key)
        if values is None or values.ndim != 2 or cam_idx >= values.shape[0] or frame_idx >= values.shape[1]:
            return None
        value = float(values[cam_idx, frame_idx])
        return value if np.isfinite(value) else None

    @staticmethod
    def _flip_flag(detail_arrays: dict[str, np.ndarray], key: str, cam_idx: int, frame_idx: int) -> bool:
        values = detail_arrays.get(key)
        if values is None or values.ndim != 2 or cam_idx >= values.shape[0] or frame_idx >= values.shape[1]:
            return False
        return bool(values[cam_idx, frame_idx])

    @staticmethod
    def _fmt_float(value: float | None) -> str:
        return "-" if value is None else f"{value:.2f}"

    @staticmethod
    def _fmt_pct(value: float | None) -> str:
        return "-" if value is None else f"{100.0 * value:.1f}%"


class TrampolineTab(ttk.Frame):
    def __init__(self, master, state: SharedAppState):
        super().__init__(master)
        self.state = state
        self.bundle = None
        self.analysis: DDSessionAnalysis | None = None
        self.contacts = []
        self.current_reconstruction_name: str | None = None
        self.uses_shared_reconstruction_panel = True
        self.shared_reconstruction_selectmode = "browse"

        controls = ttk.LabelFrame(self, text="Déplacement dans la toile")
        controls.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(
            controls,
            text=(
                "Première version: la position sur la toile est estimée à partir des deux pieds "
                "(chevilles gauche/droite), et la pénalité retenue correspond au pied le plus pénalisant "
                "pendant chaque intervalle de contact entre deux sauts détectés par l'onglet DD."
            ),
            foreground="#4f5b66",
            justify=tk.LEFT,
            wraplength=1200,
        ).pack(fill=tk.X, padx=8, pady=(0, 4))

        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=1)
        body.add(right, weight=2)

        ttk.Button(left, text="Analyze / refresh", command=self.refresh_analysis).pack(anchor="w", pady=(0, 8))

        summary_box = ttk.LabelFrame(left, text="Résumé")
        summary_box.pack(fill=tk.BOTH, expand=True)
        self.summary = ScrolledText(summary_box, height=22, wrap=tk.WORD)
        self.summary.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        figure_box = ttk.LabelFrame(right, text="Toile + contacts")
        figure_box.pack(fill=tk.BOTH, expand=True)
        self.figure = Figure(figsize=(11, 7))
        self.canvas = FigureCanvasTkAgg(self.figure, master=figure_box)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, figure_box, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill=tk.X)

        self.state.keypoints_var.trace_add("write", lambda *_args: self.sync_dataset_dir())
        self.state.output_root_var.trace_add("write", lambda *_args: self.sync_dataset_dir())
        self.state.pose2sim_trc_var.trace_add("write", lambda *_args: self.sync_dataset_dir())
        self.state.fps_var.trace_add("write", lambda *_args: self.refresh_analysis())
        self.state.initial_rotation_correction_var.trace_add("write", lambda *_args: self.refresh_analysis())
        self.state.register_reconstruction_listener(self.refresh_available_reconstructions)
        self.sync_dataset_dir()

    def configure_shared_reconstruction_panel(self, panel: SharedReconstructionPanel) -> None:
        panel.configure_for_consumer(
            title="Reconstructions | Toile",
            refresh_callback=self.refresh_available_reconstructions,
            selection_callback=self.refresh_analysis,
            selectmode=self.shared_reconstruction_selectmode,
        )
        self.refresh_available_reconstructions()

    def _publish_reconstruction_rows(self, rows: list[dict[str, object]], defaults: list[str]) -> None:
        panel = self.state.shared_reconstruction_panel
        if panel is not None and self.state.active_reconstruction_consumer is self:
            panel.set_rows(rows, defaults)

    def _on_reconstruction_selection_changed(self) -> None:
        self.refresh_analysis()

    def sync_dataset_dir(self) -> None:
        self.refresh_available_reconstructions()

    def refresh_available_reconstructions(self) -> None:
        try:
            _output_dir, _bundle, preview_state = load_shared_reconstruction_preview_state(
                self.state,
                preferred_names=[
                    "ekf_2d_acc",
                    "ekf_3d",
                    "pose2sim",
                    "triangulation_exhaustive",
                    "triangulation_greedy",
                ],
                fallback_count=4,
                include_3d=True,
                include_q=True,
                include_q_root=True,
            )
            self._publish_reconstruction_rows(preview_state.rows, preview_state.defaults[:1])
        except Exception:
            pass

    def _selected_reconstruction(self) -> str | None:
        selected = list(self.state.shared_reconstruction_selection)
        return selected[-1] if selected else None

    def refresh_analysis(self) -> None:
        try:
            with gui_busy_popup(self, title="Toile", message="Analyse du déplacement sur la toile...") as popup:
                self.bundle = get_cached_preview_bundle(
                    self.state, current_dataset_dir(self.state), None, None, align_root=False
                )
                self.current_reconstruction_name = self._selected_reconstruction()
                if self.current_reconstruction_name is None:
                    self.analysis = None
                    self.contacts = []
                    self.render_summary()
                    self.refresh_plot()
                    return
                root_q, full_q, q_names = preview_root_series_for_reconstruction(
                    bundle=self.bundle,
                    name=self.current_reconstruction_name,
                    initial_rotation_correction=bool(self.state.initial_rotation_correction_var.get()),
                )
                if root_q is None:
                    raise ValueError(f"Aucune cinématique racine disponible pour {self.current_reconstruction_name}.")
                fps = float(self.state.fps_var.get())
                recon_3d = self.bundle.get("recon_3d", {}) if isinstance(self.bundle, dict) else {}
                points_3d = (
                    np.asarray(recon_3d[self.current_reconstruction_name], dtype=float)
                    if self.current_reconstruction_name in recon_3d
                    else None
                )
                popup.set_status("Segmentation partagée des sauts...")
                self.analysis = shared_jump_analysis(
                    self.state,
                    reconstruction_name=self.current_reconstruction_name,
                    root_q=np.asarray(root_q, dtype=float),
                    points_3d=points_3d,
                    fps=fps,
                    height_threshold=TRAMPOLINE_BED_HEIGHT_M,
                    height_threshold_range_ratio=0.20,
                    smoothing_window_s=0.15,
                    min_airtime_s=0.25,
                    min_gap_s=0.08,
                    min_peak_prominence_m=0.35,
                    contact_window_s=0.35,
                    full_q=None if full_q is None else np.asarray(full_q, dtype=float),
                    q_names=q_names,
                    angle_mode="euler",
                    analysis_start_frame=ANALYSIS_START_FRAME,
                    require_complete_jumps=True,
                )
                contact_series = points_3d if points_3d is not None else np.asarray(root_q[:, :2], dtype=float)
                self.contacts = analyze_trampoline_contacts(self.analysis, contact_series)
                self.render_summary()
                self.refresh_plot()
        except Exception as exc:
            messagebox.showerror("Déplacement toile", str(exc))

    def render_summary(self) -> None:
        self.summary.delete("1.0", tk.END)
        if self.analysis is None or self.current_reconstruction_name is None:
            self.summary.insert("1.0", "No trampoline analysis yet.")
            return
        lines = [
            f"Reconstruction: {reconstruction_label(self.current_reconstruction_name)}",
            "Contact proxy: ankles on the bed (max penalty of left/right foot)",
            f"Detected jumps: {len(self.analysis.jumps)}",
            f"Contact intervals between jumps: {len(self.contacts)}",
            f"Total penalty: {total_trampoline_penalty(self.contacts):.2f}",
            "",
        ]
        for contact in self.contacts:
            lines.append(
                f"C{contact.index}: frames {contact.start}-{contact.end} | center {contact.center_frame} | "
                f"L=({contact.left_x:.3f}, {contact.left_y:.3f}) m | "
                f"R=({contact.right_x:.3f}, {contact.right_y:.3f}) m | "
                f"applied=({contact.x:.3f}, {contact.y:.3f}) m | penalty={'-' if contact.penalty is None else f'{contact.penalty:.2f}'}"
            )
        self.summary.insert("1.0", "\n".join(lines))

    def refresh_plot(self) -> None:
        self.figure.clear()
        if self.analysis is None or self.current_reconstruction_name is None:
            ax = self.figure.subplots(1, 1)
            ax.text(0.5, 0.5, "No trampoline analysis yet", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self.canvas.draw_idle()
            return
        root_q = np.asarray(self.analysis.root_q, dtype=float)
        frame_slice = analysis_frame_slice(root_q.shape[0])
        display_root_q = root_q[frame_slice]
        if display_root_q.shape[0] == 0:
            ax = self.figure.subplots(1, 1)
            ax.text(
                0.5,
                0.5,
                f"No analysis frames are available after frame {ANALYSIS_START_FRAME - 1}.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            self.canvas.draw_idle()
            return
        t = np.arange(root_q.shape[0], dtype=float) / float(self.state.fps_var.get())
        display_t = t[frame_slice]
        bed_ax, time_ax = np.atleast_1d(self.figure.subplots(1, 2))
        draw_trampoline_bed(bed_ax)
        bed_ax.plot(
            display_root_q[:, 0], display_root_q[:, 1], color="#4f5b66", linewidth=1.0, alpha=0.25, label="root XY path"
        )
        penalty_colors = {0.0: "#55a868", 0.1: "#dd8452", 0.3: "#c44e52"}
        for contact in self.contacts:
            color = penalty_colors.get(contact.penalty, "#8172b3")
            if np.isfinite(contact.left_x) and np.isfinite(contact.left_y):
                bed_ax.scatter(contact.left_x, contact.left_y, color="#4c72b0", s=36, zorder=3, marker="^")
            if np.isfinite(contact.right_x) and np.isfinite(contact.right_y):
                bed_ax.scatter(contact.right_x, contact.right_y, color="#c44e52", s=36, zorder=3, marker="s")
            if np.isfinite(contact.x) and np.isfinite(contact.y):
                bed_ax.scatter(contact.x, contact.y, color=color, s=64, zorder=4, marker="o")
                bed_ax.text(contact.x, contact.y + 0.08, f"C{contact.index}", ha="center", va="bottom", fontsize=9)
        bed_ax.set_title("Bed coordinates")
        bed_ax.legend(loc="upper right", fontsize=8)

        time_ax.plot(display_t, display_root_q[:, 0], color="#4c72b0", linewidth=1.6, label="TRUNK:TransX")
        time_ax.plot(display_t, display_root_q[:, 1], color="#c44e52", linewidth=1.6, label="TRUNK:TransY")
        label_y = float(np.nanmax(display_root_q[:, :2])) if np.any(np.isfinite(display_root_q[:, :2])) else 0.0
        for jump in self.analysis.jump_segments:
            time_ax.axvspan(t[jump.start], t[jump.end], color="#4c72b0", alpha=0.06)
        for contact in self.contacts:
            color = penalty_colors.get(contact.penalty, "#8172b3")
            time_ax.axvspan(t[contact.start], t[contact.end], color=color, alpha=0.10)
            time_ax.text(
                t[contact.center_frame], label_y, f"C{contact.index}", color=color, fontsize=8, ha="center", va="bottom"
            )
        time_ax.set_title("Contact windows between jumps")
        time_ax.set_xlabel("Time (s)")
        time_ax.set_ylabel("Bed proxy (m)")
        time_ax.grid(alpha=0.25)
        time_ax.legend(loc="upper right", fontsize=8)
        time_ax.set_xlim(display_t[0], display_t[-1])
        self.figure.tight_layout()
        self.canvas.draw_idle()


class DDTab(ttk.Frame):
    def __init__(self, master, state: SharedAppState):
        super().__init__(master)
        self.state = state
        self.bundle = None
        self.analysis: DDSessionAnalysis | None = None
        self.analysis_by_name: dict[str, DDSessionAnalysis] = {}
        self.current_reconstruction_name: str | None = None
        self.expected_dd_codes: dict[int, str] = {}
        self._suspend_refresh = False
        self.uses_shared_reconstruction_panel = True
        self.shared_reconstruction_selectmode = "browse"

        controls = ttk.LabelFrame(self, text="Analyse DD")
        controls.pack(fill=tk.X, padx=10, pady=10)
        self.dd_reference_path = LabeledEntry(
            controls,
            "DD JSON",
            "",
            browse=True,
            filetypes=(("DD JSON", "*_DD.json"), ("JSON files", "*.json"), ("All files", "*.*")),
            browse_initialdir="inputs/dd",
        )
        self.dd_reference_path.pack(fill=tk.X, padx=8, pady=(0, 4))

        row1 = ttk.Frame(controls)
        row1.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(row1, text="Height DoF", width=12).pack(side=tk.LEFT)
        self.height_dof = tk.StringVar(value="TRUNK:TransZ")
        self.height_dof_box = ttk.Combobox(
            row1, textvariable=self.height_dof, values=["TRUNK:TransZ"], width=24, state="readonly"
        )
        self.height_dof_box.pack(side=tk.LEFT, padx=(0, 8))
        smooth_label = ttk.Label(row1, text="Smooth (s)", width=10)
        smooth_label.pack(side=tk.LEFT)
        self.smoothing_window_s = tk.StringVar(value="0.15")
        smooth_entry = ttk.Entry(row1, textvariable=self.smoothing_window_s, width=8)
        smooth_entry.pack(side=tk.LEFT, padx=(0, 8))
        thr_ratio_label = ttk.Label(row1, text="Thr ratio", width=10)
        thr_ratio_label.pack(side=tk.LEFT)
        self.height_threshold_ratio = tk.StringVar(value="0.20")
        thr_ratio_entry = ttk.Entry(row1, textvariable=self.height_threshold_ratio, width=8)
        thr_ratio_entry.pack(side=tk.LEFT, padx=(0, 8))
        thr_abs_label = ttk.Label(row1, text="Thr abs", width=8)
        thr_abs_label.pack(side=tk.LEFT)
        self.height_threshold_abs = tk.StringVar(value="")
        thr_abs_entry = ttk.Entry(row1, textvariable=self.height_threshold_abs, width=8)
        thr_abs_entry.pack(side=tk.LEFT, padx=(0, 8))
        airtime_label = ttk.Label(row1, text="Airtime (s)", width=11)
        airtime_label.pack(side=tk.LEFT)
        self.min_airtime_s = tk.StringVar(value="0.25")
        airtime_entry = ttk.Entry(row1, textvariable=self.min_airtime_s, width=8)
        airtime_entry.pack(side=tk.LEFT)

        row2 = ttk.Frame(controls)
        row2.pack(fill=tk.X, padx=8, pady=4)
        min_gap_label = ttk.Label(row2, text="Min gap (s)", width=12)
        min_gap_label.pack(side=tk.LEFT)
        self.min_gap_s = tk.StringVar(value="0.08")
        min_gap_entry = ttk.Entry(row2, textvariable=self.min_gap_s, width=8)
        min_gap_entry.pack(side=tk.LEFT, padx=(0, 8))
        prominence_label = ttk.Label(row2, text="Prominence (m)", width=13)
        prominence_label.pack(side=tk.LEFT)
        self.min_peak_prominence_m = tk.StringVar(value="0.35")
        prominence_entry = ttk.Entry(row2, textvariable=self.min_peak_prominence_m, width=8)
        prominence_entry.pack(side=tk.LEFT, padx=(0, 8))
        contact_label = ttk.Label(row2, text="Contact win (s)", width=13)
        contact_label.pack(side=tk.LEFT)
        self.contact_window_s = tk.StringVar(value="0.35")
        contact_entry = ttk.Entry(row2, textvariable=self.contact_window_s, width=8)
        contact_entry.pack(side=tk.LEFT, padx=(0, 8))
        angle_mode_label = ttk.Label(row2, text="Angles", width=8)
        angle_mode_label.pack(side=tk.LEFT)
        self.angle_mode = tk.StringVar(value="euler")
        angle_mode_box = ttk.Combobox(
            row2, textvariable=self.angle_mode, values=["euler", "body_axes"], width=12, state="readonly"
        )
        angle_mode_box.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(row2, text="Analyze / refresh", command=self.refresh_analysis).pack(side=tk.LEFT, padx=(12, 0))

        attach_tooltip(
            self.height_dof_box, "DoF utilisee pour segmenter les sauts. Par défaut, la hauteur de la racine."
        )
        attach_tooltip(smooth_label, "Fenetre de lissage temporel appliquee a la hauteur avant la segmentation.")
        attach_tooltip(smooth_entry, "Fenetre de lissage temporel appliquee a la hauteur avant la segmentation.")
        attach_tooltip(thr_ratio_label, "Seuil relatif de hauteur pour detecter les phases aeriennes.")
        attach_tooltip(thr_ratio_entry, "Seuil relatif de hauteur pour detecter les phases aeriennes.")
        attach_tooltip(thr_abs_label, "Seuil absolu de hauteur. Laisser vide pour utiliser le seuil relatif.")
        attach_tooltip(thr_abs_entry, "Seuil absolu de hauteur. Laisser vide pour utiliser le seuil relatif.")
        attach_tooltip(airtime_label, "Duree minimale au-dessus du seuil pour conserver un saut.")
        attach_tooltip(airtime_entry, "Duree minimale au-dessus du seuil pour conserver un saut.")
        attach_tooltip(min_gap_label, "Ecart maximal entre deux regions aeriennes avant fusion.")
        attach_tooltip(min_gap_entry, "Ecart maximal entre deux regions aeriennes avant fusion.")
        attach_tooltip(prominence_label, "Prominence minimale du pic de hauteur pour conserver un saut.")
        attach_tooltip(prominence_entry, "Prominence minimale du pic de hauteur pour conserver un saut.")
        attach_tooltip(contact_label, "Fenetre de recherche des minima de contact avant et apres la phase aerienne.")
        attach_tooltip(contact_entry, "Fenetre de recherche des minima de contact avant et apres la phase aerienne.")
        attach_tooltip(angle_mode_label, "Choisit la methode utilisee pour calculer salto, vrille et angles du corps.")
        attach_tooltip(
            angle_mode_box,
            "euler: ré-extrait simplement RotX/RotY/RotZ. body_axes: calcule des angles fonctionnels à partir des axes du corps.",
        )
        self.dd_reference_path.set_tooltip(
            "Fichier JSON des codes attendus par saut. S'il existe un fichier '*_DD.json' associé au fichier keypoints, il est chargé automatiquement."
        )

        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        left = ttk.Frame(body, width=460)
        right = ttk.Frame(body)
        body.add(left, weight=2)
        body.add(right, weight=3)

        comparison_box = ttk.LabelFrame(left, text="Comparaison DD par reconstruction")
        comparison_box.pack(fill=tk.BOTH, expand=False, pady=(0, 8))
        self.comparison_tree = ttk.Treeview(
            comparison_box,
            columns=("reconstruction", "match", "status", "codes"),
            show="headings",
            height=6,
            selectmode="browse",
        )
        self.comparison_tree.heading("reconstruction", text="Reconstruction")
        self.comparison_tree.heading("match", text="Match")
        self.comparison_tree.heading("status", text="Status")
        self.comparison_tree.heading("codes", text="Detected codes")
        self.comparison_tree.column("reconstruction", width=190, anchor="w")
        self.comparison_tree.column("match", width=70, anchor="center")
        self.comparison_tree.column("status", width=70, anchor="center")
        self.comparison_tree.column("codes", width=370, anchor="w")
        self.comparison_tree.tag_configure("ok", foreground="#1f7a1f")
        self.comparison_tree.tag_configure("partial", foreground="#b36b00")
        self.comparison_tree.tag_configure("bad", foreground="#b22222")
        self.comparison_tree.tag_configure("neutral", foreground="#666666")
        self.comparison_tree.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.comparison_tree.bind("<<TreeviewSelect>>", self._on_comparison_selected)
        attach_tooltip(
            self.comparison_tree,
            "Compares each reconstruction against the DD reference JSON. The detected-code column highlights mismatching characters with square brackets.",
        )

        jumps_box = ttk.LabelFrame(left, text="Sauts détectés")
        jumps_box.pack(fill=tk.BOTH, expand=False, pady=(0, 8))
        self.jump_list = tk.Listbox(jumps_box, exportselection=False, height=8)
        self.jump_list.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        summary_box = ttk.LabelFrame(left, text="Résumé DD")
        summary_box.pack(fill=tk.BOTH, expand=True)
        self.summary = ScrolledText(summary_box, height=18, wrap=tk.WORD)
        self.summary.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.summary.tag_configure("dd_role_somersault", foreground="#4c72b0")
        self.summary.tag_configure("dd_role_twist", foreground="#dd8452")
        self.summary.tag_configure("dd_role_body", foreground="#8172b3")
        self.summary.tag_configure("dd_match", background="#e8f5e9")
        self.summary.tag_configure("dd_mismatch", background="#fdecea")
        self.summary.tag_configure("dd_missing", foreground="#666666", background="#fff5d6")
        self.summary.tag_configure("dd_legend", foreground="#4f5b66")

        figure_box = ttk.LabelFrame(right, text="Segmentation et rotations")
        figure_box.pack(fill=tk.BOTH, expand=True)
        self.figure = Figure(figsize=(11, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, master=figure_box)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, figure_box, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill=tk.X)

        self.jump_list.bind("<<ListboxSelect>>", self._on_jump_selected)
        self.state.keypoints_var.trace_add("write", lambda *_args: self.sync_dataset_dir())
        self.state.output_root_var.trace_add("write", lambda *_args: self.sync_dataset_dir())
        self.state.pose2sim_trc_var.trace_add("write", lambda *_args: self.sync_dataset_dir())
        self.dd_reference_path.var.trace_add("write", lambda *_args: self.refresh_analysis())
        self.state.fps_var.trace_add("write", lambda *_args: self.refresh_plot())
        self.state.initial_rotation_correction_var.trace_add("write", lambda *_args: self.refresh_analysis())
        self.state.register_reconstruction_listener(self.refresh_available_reconstructions)
        self.after_idle(self.sync_dataset_dir)

    def _on_reconstruction_selected(self, _event=None) -> None:
        if self._suspend_refresh:
            gui_debug("DD reconstruction selection ignored during suspended refresh")
            return
        selected_name = self._selected_reconstruction()
        if (
            selected_name is not None
            and selected_name == self.current_reconstruction_name
            and self.analysis is not None
        ):
            gui_debug(f"DD reconstruction selection ignored unchanged={selected_name}")
            return
        self.refresh_analysis()

    def _on_jump_selected(self, _event=None) -> None:
        if self._suspend_refresh:
            gui_debug("DD jump selection ignored during suspended refresh")
            return
        self.refresh_plot()

    def _on_comparison_selected(self, _event=None) -> None:
        if self._suspend_refresh:
            return
        selection = self.comparison_tree.selection()
        if not selection:
            return
        recon_name = selection[-1]
        panel = self.state.shared_reconstruction_panel
        if panel is not None and self.state.active_reconstruction_consumer is self and panel.tree.exists(recon_name):
            panel._suspend_selection_callback = True
            try:
                panel.tree.selection_set((recon_name,))
                panel.tree.focus(recon_name)
            finally:
                panel._suspend_selection_callback = False
            panel._publish_selection()
        self._on_reconstruction_selected()

    def configure_shared_reconstruction_panel(self, panel: SharedReconstructionPanel) -> None:
        panel.configure_for_consumer(
            title="Reconstructions | DD",
            refresh_callback=self.refresh_available_reconstructions,
            selection_callback=self._on_reconstruction_selected,
            selectmode=self.shared_reconstruction_selectmode,
        )
        self.refresh_available_reconstructions()

    def _publish_reconstruction_rows(self, rows: list[dict[str, object]], defaults: list[str]) -> None:
        panel = self.state.shared_reconstruction_panel
        if panel is not None and self.state.active_reconstruction_consumer is self:
            panel.set_rows(rows, defaults)

    def _set_jump_selection(self, index: int) -> None:
        self._suspend_refresh = True
        try:
            self.jump_list.selection_clear(0, tk.END)
            if self.jump_list.size() > 0:
                self.jump_list.selection_set(index)
        finally:
            self._suspend_refresh = False

    def sync_dataset_dir(self) -> None:
        gui_debug("DD sync_dataset_dir")
        self.sync_dd_reference_path()
        self.refresh_available_reconstructions()

    def sync_dd_reference_path(self) -> None:
        """Auto-select the `*_DD.json` file matching the current keypoints file when it exists."""

        raw_keypoints_path = self.state.keypoints_var.get().strip()
        if not raw_keypoints_path:
            self.dd_reference_path.var.set("")
            return
        keypoints_path = ROOT / raw_keypoints_path
        candidate_path = default_dd_reference_path(keypoints_path)
        if candidate_path is not None and candidate_path.exists():
            try:
                rel = candidate_path.resolve().relative_to(ROOT)
                self.dd_reference_path.var.set(str(rel))
            except Exception:
                self.dd_reference_path.var.set(str(candidate_path))
        else:
            self.dd_reference_path.var.set("")

    def refresh_available_reconstructions(self) -> None:
        try:
            gui_debug(f"DD refresh_available_reconstructions start dataset={current_dataset_dir(self.state)}")
            _output_dir, _bundle, preview_state = load_shared_reconstruction_preview_state(
                self.state,
                preferred_names=[
                    "ekf_2d_acc",
                    "ekf_3d",
                    "pose2sim",
                    "triangulation_exhaustive",
                    "triangulation_greedy",
                ],
                fallback_count=5,
                include_3d=True,
                include_q=True,
                include_q_root=True,
            )
            self._publish_reconstruction_rows(preview_state.rows, preview_state.defaults[:1])
            gui_debug(
                "DD refresh_available_reconstructions done "
                f"available={len(preview_state.available_names)} rows={len(preview_state.rows)}"
            )
        except Exception:
            pass

    def _selected_reconstruction(self) -> str | None:
        names = list(self.state.shared_reconstruction_selection)
        return names[-1] if names else None

    def _root_series_for_reconstruction(
        self,
        name: str,
        recon_q: dict[str, np.ndarray],
        recon_q_root: dict[str, np.ndarray],
        recon_3d: dict[str, np.ndarray],
        q_names: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None, list[str] | None]:
        _ = (recon_q, recon_q_root, recon_3d, q_names)
        return preview_root_series_for_reconstruction(
            bundle=self.bundle or {},
            name=name,
            initial_rotation_correction=bool(self.state.initial_rotation_correction_var.get()),
        )

    def _update_height_dof_choices(self, q_name_list: list[str] | None) -> None:
        values: list[str] = []
        for dof_name in TRUNK_TRANSLATION_NAMES:
            if dof_name not in values:
                values.append(dof_name)
        if q_name_list is not None:
            for dof_name in q_name_list:
                if dof_name not in values:
                    values.append(dof_name)
        current = self.height_dof.get().strip()
        self.height_dof_box.configure(values=values)
        if current not in values:
            self.height_dof.set("TRUNK:TransZ" if "TRUNK:TransZ" in values else values[0])

    def _height_series(
        self, root_q: np.ndarray, full_q: np.ndarray | None, q_name_list: list[str] | None
    ) -> np.ndarray:
        dof_name = self.height_dof.get().strip() or "TRUNK:TransZ"
        if q_name_list is not None and full_q is not None and dof_name in q_name_list:
            return np.asarray(full_q[:, q_name_list.index(dof_name)], dtype=float)
        ordered = [str(name) for name in root_ordered_names()]
        if dof_name in ordered:
            return np.asarray(root_q[:, ordered.index(dof_name)], dtype=float)
        return np.asarray(root_q[:, 2], dtype=float)

    def _selected_jump_index(self) -> int:
        selection = self.jump_list.curselection()
        if selection:
            return int(selection[0])
        return 0

    def refresh_analysis(self) -> None:
        try:
            with gui_busy_popup(self, title="DD", message="Analyse des sauts et des codes DD...") as popup:
                gui_debug(f"DD refresh_analysis start dataset={current_dataset_dir(self.state)}")
                self.bundle = get_cached_preview_bundle(
                    self.state, current_dataset_dir(self.state), None, None, align_root=False
                )
                available_names = bundle_available_reconstruction_names(
                    self.bundle, include_3d=True, include_q=True, include_q_root=True
                )
                selected_name = self._selected_reconstruction()
                self.current_reconstruction_name = selected_name
                gui_debug(f"DD selected reconstruction={selected_name}")
                self.expected_dd_codes = self._load_expected_dd_codes()
                if selected_name is None:
                    self.analysis = None
                    self.analysis_by_name = {}
                    self.render_comparison_table(available_names)
                    self.render_summary()
                    self.refresh_plot()
                    return

                recon_q = self.bundle.get("recon_q", {})
                recon_q_root = self.bundle.get("recon_q_root", {})
                recon_3d = self.bundle.get("recon_3d", {})
                q_names = np.asarray(self.bundle.get("q_names", np.array([], dtype=object)), dtype=object)
                root_q, full_q, q_name_list = self._root_series_for_reconstruction(
                    selected_name, recon_q, recon_q_root, recon_3d, q_names
                )
                if root_q is None:
                    raise ValueError(f"Aucune cinématique racine disponible pour {selected_name}.")
                gui_debug(
                    "DD root series ready "
                    f"name={selected_name} root_shape={root_q.shape} "
                    f"full_q={'yes' if full_q is not None else 'no'}"
                )
                self._update_height_dof_choices(q_name_list)
                fps = float(self.state.fps_var.get())
                height_threshold_abs = self.height_threshold_abs.get().strip()
                popup.set_status("Segmentation partagée et classification DD...")
                gui_debug(
                    "DD analyze_dd_session "
                    f"fps={fps} height_dof={self.height_dof.get()} "
                    f"smooth={self.smoothing_window_s.get()} thr_ratio={self.height_threshold_ratio.get()} "
                    f"thr_abs={height_threshold_abs or '-'}"
                )
                self.analysis_by_name = {}
                for name in available_names:
                    name_root_q, name_full_q, name_q_name_list = self._root_series_for_reconstruction(
                        name, recon_q, recon_q_root, recon_3d, q_names
                    )
                    if name_root_q is None:
                        continue
                    try:
                        points_3d = np.asarray(recon_3d[name], dtype=float) if name in recon_3d else None
                        self.analysis_by_name[name] = shared_jump_analysis(
                            self.state,
                            reconstruction_name=name,
                            root_q=np.asarray(name_root_q, dtype=float),
                            points_3d=points_3d,
                            fps=fps,
                            height_threshold=(
                                float(height_threshold_abs) if height_threshold_abs else TRAMPOLINE_BED_HEIGHT_M
                            ),
                            height_threshold_range_ratio=float(self.height_threshold_ratio.get()),
                            smoothing_window_s=float(self.smoothing_window_s.get()),
                            min_airtime_s=float(self.min_airtime_s.get()),
                            min_gap_s=float(self.min_gap_s.get()),
                            min_peak_prominence_m=float(self.min_peak_prominence_m.get()),
                            contact_window_s=float(self.contact_window_s.get()),
                            full_q=name_full_q,
                            q_names=name_q_name_list,
                            angle_mode=self.angle_mode.get(),
                            analysis_start_frame=ANALYSIS_START_FRAME,
                            require_complete_jumps=True,
                        )
                    except Exception:
                        continue
                self.analysis = self.analysis_by_name.get(selected_name)
                self.render_comparison_table(available_names)
                if self.analysis is None:
                    raise ValueError(f"Aucune analyse DD disponible pour {selected_name}.")
                gui_debug(
                    "DD analyze_dd_session done "
                    f"jumps={len(self.analysis.jumps)} threshold={self.analysis.height_threshold:.4f}"
                )
                self.render_jump_list()
                self.render_summary()
                self.refresh_plot()
                gui_debug("DD refresh_analysis done")
        except Exception as exc:
            gui_debug(f"DD refresh_analysis error: {exc}")
            messagebox.showerror("Analyse DD", str(exc))

    def _load_expected_dd_codes(self) -> dict[int, str]:
        """Load the optional expected-DD reference file selected in the tab."""

        raw_path = self.dd_reference_path.get()
        if not raw_path:
            return {}
        path = optional_root_relative_path(raw_path)
        if path is None or not path.exists():
            return {}
        return load_dd_reference_codes(path)

    def render_jump_list(self) -> None:
        gui_debug("DD render_jump_list")
        previous = self._selected_jump_index() if self.jump_list.size() else 0
        self._suspend_refresh = True
        try:
            self.jump_list.delete(0, tk.END)
            if self.analysis is None:
                return
            for idx, jump in enumerate(self.analysis.jumps, start=1):
                self.jump_list.insert(
                    tk.END,
                    jump_list_label_with_reference(idx, jump, self.expected_dd_codes.get(idx)),
                )
            if self.analysis.jumps:
                self._set_jump_selection(min(previous, len(self.analysis.jumps) - 1))
        finally:
            self._suspend_refresh = False

    def render_comparison_table(self, available_names: list[str]) -> None:
        """Populate the per-reconstruction DD comparison table with color-coded statuses."""

        selected_name = self.current_reconstruction_name
        self._suspend_refresh = True
        try:
            for item in self.comparison_tree.get_children():
                self.comparison_tree.delete(item)
            for name in available_names:
                analysis = self.analysis_by_name.get(name)
                comparison = compare_dd_to_reference(analysis, self.expected_dd_codes)
                detected_codes = format_detected_dd_codes_with_inline_errors(comparison)
                tag = dd_reference_status_color(comparison)
                self.comparison_tree.insert(
                    "",
                    "end",
                    iid=name,
                    values=(
                        reconstruction_label(name),
                        dd_reference_status_text(comparison),
                        comparison.status.replace("_", " "),
                        detected_codes,
                    ),
                    tags=(tag,),
                )
            if selected_name is not None and self.comparison_tree.exists(selected_name):
                self.comparison_tree.selection_set((selected_name,))
                self.comparison_tree.focus(selected_name)
        finally:
            self._suspend_refresh = False

    def render_summary(self) -> None:
        gui_debug("DD render_summary")
        self.summary.delete("1.0", tk.END)
        summary_text = format_dd_summary(
            self.analysis,
            reconstruction_label_text=(
                reconstruction_label(self.current_reconstruction_name)
                if self.current_reconstruction_name is not None
                else None
            ),
            height_dof=self.height_dof.get(),
            angle_mode=self.angle_mode.get(),
            fps=float(self.state.fps_var.get()),
            expected_codes_by_jump=self.expected_dd_codes,
        )
        self.summary.insert(tk.END, summary_text)
        self._render_dd_code_comparison_details()

    def _insert_role_colored_code(
        self,
        code_comparison: list,
        *,
        expected: bool,
    ) -> None:
        """Insert one role-colored DD code into the summary panel."""

        last_role = None
        for char_info in code_comparison:
            if char_info.role != last_role and last_role is not None:
                self.summary.insert(tk.END, " ")
            role_tag = {
                "somersault": "dd_role_somersault",
                "twist": "dd_role_twist",
                "body": "dd_role_body",
            }.get(char_info.role, "")
            character = char_info.expected_char if expected else char_info.detected_char
            tags = [role_tag] if role_tag else []
            if character == "_":
                tags.append("dd_missing")
            elif not expected and not char_info.matches:
                tags.append("dd_mismatch")
            elif not expected and char_info.matches:
                tags.append("dd_match")
            self.summary.insert(tk.END, character, tuple(tag for tag in tags if tag))
            last_role = char_info.role

    def _render_dd_code_comparison_details(self) -> None:
        """Append a character-level, role-colored DD comparison to the summary."""

        if self.analysis is None or not self.expected_dd_codes:
            return
        self.summary.insert(tk.END, "\nRole-colored DD comparison\n", ("dd_legend",))
        self.summary.insert(
            tk.END,
            "Blue=somersault block | orange=twists | purple=body shape | red background=mismatch\n\n",
            ("dd_legend",),
        )
        for jump_idx, jump in enumerate(self.analysis.jumps, start=1):
            expected_code = self.expected_dd_codes.get(jump_idx)
            if not expected_code:
                continue
            detected_code = str(jump.code or "-")
            code_comparison = compare_dd_code_characters(expected_code, detected_code)
            match_label = "match" if code_comparison and all(item.matches for item in code_comparison) else "diff"
            self.summary.insert(tk.END, f"S{jump_idx}: ", ())
            self.summary.insert(tk.END, "expected ", ("dd_legend",))
            self._insert_role_colored_code(code_comparison, expected=True)
            self.summary.insert(tk.END, " | ", ())
            self.summary.insert(tk.END, "detected ", ("dd_legend",))
            self._insert_role_colored_code(code_comparison, expected=False)
            self.summary.insert(tk.END, f" | {match_label}\n", ("dd_legend",))

    def refresh_plot(self) -> None:
        gui_debug("DD refresh_plot start")
        self.figure.clear()
        if self.analysis is None or self.current_reconstruction_name is None:
            ax = self.figure.subplots(1, 1)
            ax.text(0.5, 0.5, "No DD analysis yet", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self.canvas.draw_idle()
            gui_debug("DD refresh_plot done empty")
            return

        fps = float(self.state.fps_var.get())
        t = np.arange(self.analysis.height.shape[0], dtype=float) / fps
        jump_idx = min(self._selected_jump_index(), max(len(self.analysis.jumps) - 1, 0))
        selected_jump = self.analysis.jumps[jump_idx] if self.analysis.jumps else None
        axes = self.figure.subplots(3, 1, sharex=False)
        axes = np.atleast_1d(axes)

        color = reconstruction_display_color(self.state, self.current_reconstruction_name)
        axes[0].plot(t, self.analysis.height, color=color, linewidth=1.1, alpha=0.45, label="height")
        axes[0].plot(t, self.analysis.smoothed_height, color=color, linewidth=2.0, label="smoothed")
        axes[0].axhline(
            self.analysis.height_threshold, color="#c44e52", linestyle="--", linewidth=1.4, label="threshold"
        )
        for idx, (start, end) in enumerate(self.analysis.airborne_regions):
            axes[0].axvspan(t[start], t[end], color="#dd8452", alpha=0.10, label="airborne" if idx == 0 else None)
        for idx, segment in enumerate(self.analysis.jump_segments):
            axes[0].axvline(t[segment.start], color="#55a868", linewidth=1.4, label="jump bounds" if idx == 0 else None)
            axes[0].axvline(t[segment.end], color="#55a868", linewidth=1.4)
            axes[0].scatter(
                t[segment.peak_index],
                self.analysis.height[segment.peak_index],
                color="#8172b3",
                s=28,
                zorder=3,
                label="peak" if idx == 0 else None,
            )
            axes[0].text(
                0.5 * (t[segment.start] + t[segment.end]),
                self.analysis.smoothed_height[segment.peak_index] + 0.03,
                f"S{idx + 1}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        if selected_jump is not None:
            axes[0].axvspan(
                t[selected_jump.segment.start],
                t[selected_jump.segment.end],
                color="#4c72b0",
                alpha=0.08,
                label="selected jump",
            )
        axes[0].set_title(f"Segmentation DD | {reconstruction_label(self.current_reconstruction_name)}")
        axes[0].set_ylabel("Height (m)")
        axes[0].grid(alpha=0.25)
        axes[0].legend(loc="best", fontsize=8)
        analysis_start_time_s = ANALYSIS_START_FRAME / fps
        axes[0].set_xlim(analysis_start_time_s, t[-1] if t.size else analysis_start_time_s)

        if selected_jump is not None:
            plot_data = build_jump_plot_data(selected_jump, fps)
            for event_idx, event_time in enumerate(plot_data.full_salto_times):
                axes[1].axvline(
                    event_time,
                    color="#777777",
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.8,
                    label="full salto" if event_idx == 0 else None,
                )
                axes[2].axvline(
                    event_time,
                    color="#777777",
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.8,
                )
            axes[1].plot(
                plot_data.local_t,
                selected_jump.somersault_curve_turns,
                color="#4c72b0",
                linewidth=1.8,
                label="somersault",
            )
            axes[1].plot(
                plot_data.local_t, selected_jump.twist_curve_turns, color="#c44e52", linewidth=1.8, label="twist"
            )
            if plot_data.quarter_salto_times.size:
                axes[1].scatter(
                    plot_data.quarter_salto_times,
                    plot_data.quarter_salto_values,
                    marker="s",
                    s=26,
                    facecolors="white",
                    edgecolors="#4c72b0",
                    linewidths=1.2,
                    zorder=4,
                    label="1/4 salto",
                )
            if plot_data.half_twist_times.size:
                axes[1].scatter(
                    plot_data.half_twist_times,
                    plot_data.half_twist_values,
                    marker="o",
                    s=24,
                    facecolors="white",
                    edgecolors="#c44e52",
                    linewidths=1.2,
                    zorder=4,
                    label="1/2 twist",
                )
            axes[1].set_title(f"S{jump_idx + 1} rotations ({selected_jump.angle_mode})")
            axes[1].set_ylabel("Turns")
            axes[1].grid(alpha=0.25)
            axes[1].legend(loc="best", fontsize=8)

            axes[2].plot(
                plot_data.local_t,
                np.rad2deg(selected_jump.hip_flex_curve_rad),
                color="#55a868",
                linewidth=1.8,
                label="hip flex",
            )
            axes[2].plot(
                plot_data.local_t,
                np.rad2deg(selected_jump.knee_flex_curve_rad),
                color="#8172b3",
                linewidth=1.8,
                label="knee flex",
            )
            for phase_name, mask, color_fill in (
                ("groupé", selected_jump.grouped_mask, "#dd8452"),
                ("carpé", selected_jump.piked_mask, "#4c72b0"),
            ):
                phase_regions = contiguous_true_regions(mask)
                for region_idx, (start_idx, end_idx) in enumerate(phase_regions):
                    axes[2].axvspan(
                        plot_data.local_t[start_idx],
                        plot_data.local_t[end_idx],
                        color=color_fill,
                        alpha=0.14,
                        label=phase_name if region_idx == 0 else None,
                    )
            axes[2].set_title(f"S{jump_idx + 1} hip/knee flexion ({selected_jump.angle_mode})")
            axes[2].set_ylabel("deg")
            axes[2].set_xlabel("Time within jump (s)")
            axes[2].grid(alpha=0.25)
            axes[2].legend(loc="best", fontsize=8)
        else:
            axes[1].text(0.5, 0.5, "No jump detected", ha="center", va="center", transform=axes[1].transAxes)
            axes[1].set_axis_off()
            axes[2].set_axis_off()

        self.figure.tight_layout()
        self.canvas.draw_idle()
        gui_debug("DD refresh_plot done " f"selected_jump={jump_idx + 1 if selected_jump is not None else 0}")


class StartupStatusWindow(tk.Toplevel):
    """Small splash window shown while the main GUI is being prepared."""

    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.title("Starting GUI")
        self.resizable(False, False)
        self.transient(master)
        self.protocol("WM_DELETE_WINDOW", lambda: None)

        self.status_var = tk.StringVar(value="Starting...")
        self.progress_var = tk.DoubleVar(value=0.0)

        body = ttk.Frame(self, padding=16)
        body.pack(fill=tk.BOTH, expand=True)
        ttk.Label(body, text="Preparing VitPose / EKF launcher", font=("", 14, "bold")).pack(anchor="w")
        ttk.Label(body, text="Please wait while the GUI loads caches and initial data.").pack(anchor="w", pady=(6, 10))
        ttk.Label(body, textvariable=self.status_var, wraplength=420).pack(anchor="w")
        self.progress = ttk.Progressbar(body, mode="determinate", length=420, maximum=1.0, variable=self.progress_var)
        self.progress.pack(fill=tk.X, pady=(12, 0))
        self.geometry("470x150")

    def set_status(self, message: str, progress_ratio: float | None = None) -> None:
        self.status_var.set(str(message))
        if progress_ratio is not None:
            self.progress_var.set(float(np.clip(progress_ratio, 0.0, 1.0)))
        self.update_idletasks()


class LauncherApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.withdraw()
        self.title("VitPose / EKF launcher")
        self.geometry("1450x950")
        self.startup_window = StartupStatusWindow(self)
        tab_specs = [
            ("2D analysis", DataExplorer2DTab),
            ("Cameras", CameraToolsTab),
            ("Calibration", CalibrationTab),
            ("Annotation", AnnotationTab),
            ("Models", ModelTab),
            ("Profiles", ProfilesTab),
            ("Reconstructions", ReconstructionsTab),
            ("Batch", BatchTab),
            ("3D animation", DualAnimationTab),
            ("2D multiview", MultiViewTab),
            ("Execution", ExecutionTab),
            ("DD", DDTab),
            ("Toile", TrampolineTab),
            ("Racine", RootKinematicsTab),
            ("Autres DoF", JointKinematicsTab),
            ("3D analysis", Analysis3DTab),
            ("Observabilité", ObservabilityTab),
        ]
        self._startup_total_steps = 4 + len(tab_specs) + 1
        self._startup_step = 0

        def advance_startup(message: str) -> None:
            self._startup_step += 1
            self._set_startup_status(message, progress_ratio=self._startup_step / self._startup_total_steps)

        advance_startup("Creating shared application state")

        state = SharedAppState(
            calib_var=tk.StringVar(value=DEFAULT_GUI_CALIB_PATH),
            keypoints_var=tk.StringVar(value=DEFAULT_GUI_KEYPOINTS_PATH),
            annotation_path_var=tk.StringVar(
                value=display_path(default_annotation_path(ROOT / DEFAULT_GUI_KEYPOINTS_PATH))
            ),
            pose2sim_trc_var=tk.StringVar(value=DEFAULT_GUI_TRC_PATH),
            fps_var=tk.StringVar(value="120"),
            workers_var=tk.StringVar(value="6"),
            pose_data_mode_var=tk.StringVar(value="cleaned"),
            pose_filter_window_var=tk.StringVar(value="9"),
            pose_outlier_ratio_var=tk.StringVar(value="0.10"),
            pose_p_low_var=tk.StringVar(value="5"),
            pose_p_high_var=tk.StringVar(value="95"),
            flip_improvement_ratio_var=tk.StringVar(value=str(DEFAULT_FLIP_IMPROVEMENT_RATIO)),
            flip_min_gain_px_var=tk.StringVar(value=str(DEFAULT_FLIP_MIN_GAIN_PX)),
            flip_min_other_cameras_var=tk.StringVar(value=str(DEFAULT_FLIP_MIN_OTHER_CAMERAS)),
            flip_outlier_percentile_var=tk.StringVar(value=str(DEFAULT_FLIP_OUTLIER_PERCENTILE)),
            flip_outlier_floor_px_var=tk.StringVar(value=str(DEFAULT_FLIP_OUTLIER_FLOOR_PX)),
            flip_restrict_to_outliers_var=tk.BooleanVar(value=DEFAULT_FLIP_RESTRICT_TO_OUTLIERS),
            flip_temporal_weight_var=tk.StringVar(value=str(DEFAULT_FLIP_TEMPORAL_WEIGHT)),
            flip_temporal_tau_px_var=tk.StringVar(value=str(DEFAULT_FLIP_TEMPORAL_TAU_PX)),
            calibration_correction_var=tk.StringVar(value="none"),
            initial_rotation_correction_var=tk.BooleanVar(value=True),
            selected_camera_names_var=tk.StringVar(value=""),
            output_root_var=tk.StringVar(value="output"),
            profiles_config_var=tk.StringVar(value=DEFAULT_GUI_PROFILES_CONFIG),
        )
        state.startup_status_callback = self._set_startup_status
        state.output_root_var.set(display_path(normalize_output_root(state.output_root_var.get())))
        advance_startup("Loading saved reconstruction profiles")
        profiles_path = ROOT / state.profiles_config_var.get()
        if profiles_path.exists():
            try:
                state.set_profiles(load_profiles_json(profiles_path))
            except Exception:
                state.set_profiles(example_profiles())
        else:
            state.set_profiles(example_profiles())
        synchronize_profiles_initial_rotation_correction(state)
        self.state = state

        advance_startup("Building main window layout")
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True)

        advance_startup("Preparing shared reconstruction selector")
        self.shared_reconstruction_panel = SharedReconstructionPanel(container, state, tooltip_fn=attach_tooltip)
        self.shared_reconstruction_panel.pack(fill=tk.X, padx=10, pady=(10, 0))
        self.state.shared_reconstruction_panel = self.shared_reconstruction_panel

        self.notebook = ttk.Notebook(container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        for tab_label, tab_cls in tab_specs:
            advance_startup(f"Preparing {tab_label} tab")
            self.notebook.add(tab_cls(self.notebook, state), text=tab_label)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        advance_startup("Finalizing GUI")
        self.after_idle(self._refresh_active_reconstruction_panel)
        self.after_idle(self._finish_startup)

    def _set_startup_status(self, message: str, progress_ratio: float | None = None) -> None:
        startup_window = getattr(self, "startup_window", None)
        if startup_window is None or not startup_window.winfo_exists():
            return
        startup_window.set_status(message, progress_ratio=progress_ratio)
        self.update_idletasks()

    def _finish_startup(self) -> None:
        self.state.startup_status_callback = None
        startup_window = getattr(self, "startup_window", None)
        if startup_window is not None and startup_window.winfo_exists():
            startup_window.destroy()
        self.deiconify()

    def _refresh_active_reconstruction_panel(self) -> None:
        """Bind the top reconstruction selector to the active tab when supported."""

        current_tab_id = self.notebook.select()
        current_tab = self.nametowidget(current_tab_id) if current_tab_id else None
        self.state.active_reconstruction_consumer = current_tab
        if current_tab is not None and hasattr(current_tab, "on_tab_activated"):
            current_tab.on_tab_activated()
        if current_tab is not None and getattr(current_tab, "uses_shared_reconstruction_panel", False):
            current_tab.configure_shared_reconstruction_panel(self.shared_reconstruction_panel)
        else:
            self.shared_reconstruction_panel.show_placeholder(
                "This tab does not use the shared reconstruction selector."
            )

    def _on_tab_changed(self, _event=None) -> None:
        self._refresh_active_reconstruction_panel()


def main() -> None:
    app = LauncherApp()
    app.mainloop()


if __name__ == "__main__":
    main()
