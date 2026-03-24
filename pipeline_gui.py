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
import tkinter as tk
from dataclasses import dataclass, field
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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.spatial.transform import Rotation

from camera_tools.camera_metrics import compute_camera_metric_rows, suggest_best_camera_names
from camera_tools.camera_selection import format_camera_names, parse_camera_names
from judging.dd_analysis import DDSessionAnalysis, analyze_dd_session
from judging.execution import (
    ExecutionDeductionEvent,
    ExecutionJumpAnalysis,
    ExecutionSessionAnalysis,
    analyze_execution_session,
    execution_focus_frame,
)
from judging.dd_presenter import (
    build_jump_plot_data,
    compare_dd_to_reference,
    dd_reference_status_color,
    dd_reference_status_text,
    format_dd_summary,
    jump_list_label_with_reference,
)
from judging.dd_reference import default_dd_reference_path, load_dd_reference_codes
from preview.dataset_preview_loader import load_dataset_preview_resources
from preview.dataset_preview_state import build_dataset_preview_state
from observability.observability_analysis import compute_observability_rank_series, summarize_rank_series
from preview.preview_bundle import (
    align_to_reference,
    load_dataset_preview_bundle,
    project_points_all_cameras,
    root_center,
)
from preview.preview_navigation import clamp_frame_index, frame_from_slider_click, step_frame_index
from reconstruction.reconstruction_presenter import (
    bundle_available_reconstruction_names,
    catalog_rows_for_names,
    default_selection,
)
from reconstruction.reconstruction_bundle import (
    extract_root_from_points,
    load_or_compute_left_right_flip_cache,
    load_or_compute_pose_data_variant_cache,
    load_or_compute_triangulation_cache,
    parse_trc_points as parse_reconstruction_trc_points,
    slice_pose_data,
)
from reconstruction.reconstruction_profiles import (
    ReconstructionProfile,
    build_pipeline_command,
    canonical_profile_name,
    example_profiles,
    generate_supported_profiles,
    load_profiles_json,
    profile_to_dict,
    save_profiles_json,
    scan_variant_output_dirs,
    validate_profile,
)
from reconstruction.reconstruction_dataset import (
    dataset_source_paths,
    reconstruction_color,
    reconstruction_label,
    write_trc_file,
)
from reconstruction.reconstruction_registry import (
    dataset_figures_dir,
    dataset_models_dir,
    dataset_reconstructions_dir,
    default_model_stem,
    infer_dataset_name,
    latest_version_for_family,
    model_biomod_path,
    model_output_dir,
    scan_model_dirs,
    scan_reconstruction_dirs,
)
from reconstruction.reconstruction_timings import format_reconstruction_timing_details, objective_total_seconds
from kinematics.root_kinematics import (
    TRUNK_ROOT_ROTATION_SEQUENCE,
    TRUNK_ROTATION_NAMES,
    TRUNK_TRANSLATION_NAMES,
    centered_finite_difference,
    compute_trunk_dofs_from_points,
    extract_root_from_q,
    normalize,
)
from kinematics.analysis_3d import (
    SEGMENT_LENGTH_DEFINITIONS,
    angular_momentum_plot_data,
    segment_length_series,
    valid_segment_length_samples,
)
from kinematics.root_series import (
    quantity_unit_label,
    root_axis_display_labels,
    root_ordered_names,
    root_rotation_matrices_from_points,
    root_rotation_matrices_from_series,
    root_series_from_points,
    root_series_from_precomputed,
    root_series_from_q,
    scale_root_series_rotations,
)
from judging.trampoline_displacement import (
    BED_X_MAX,
    BED_Y_MAX,
    TRAMPOLINE_GEOMETRY,
    X_INNER,
    X_MAX,
    Y_INNER,
    Y_MAX,
    analyze_trampoline_contacts,
    total_trampoline_penalty,
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
    DEFAULT_REPROJECTION_THRESHOLD_PX,
    ReconstructionResult,
    apply_left_right_flip_to_points,
    initial_state_from_triangulation,
    load_calibrations,
    load_reconstruction_cache,
    load_pose_data,
    metadata_cache_matches,
    reconstruction_cache_metadata,
    swap_left_right_keypoints,
    triangulate_pose2sim_like,
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
    "pose2sim": "Pose2Sim",
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


def pair_dof_names(q_names: np.ndarray) -> list[tuple[str, str, str]]:
    names = [str(name) for name in q_names]
    pairs = []
    for name in names:
        if not name.startswith("LEFT_"):
            continue
        right_name = name.replace("LEFT_", "RIGHT_", 1)
        if right_name in names:
            pair_label = name.replace("LEFT_", "", 1)
            pairs.append((pair_label, name, right_name))
    pairs.sort(key=lambda item: item[0])
    return pairs


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


def draw_skeleton_3d(ax, frame_points: np.ndarray, color: str, label: str) -> None:
    grouped = keypoint_groups(frame_points)
    if grouped["center"].size:
        ax.scatter(
            grouped["center"][:, 0],
            grouped["center"][:, 1],
            grouped["center"][:, 2],
            s=22,
            c=color,
            marker="o",
            depthshade=False,
            label=label,
        )
    if grouped["left"].size:
        ax.scatter(
            grouped["left"][:, 0],
            grouped["left"][:, 1],
            grouped["left"][:, 2],
            s=34,
            c=color,
            marker="^",
            depthshade=False,
        )
    if grouped["right"].size:
        ax.scatter(
            grouped["right"][:, 0],
            grouped["right"][:, 1],
            grouped["right"][:, 2],
            s=34,
            c=color,
            marker="s",
            depthshade=False,
        )
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
            )


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
    label: str,
    marker_size: float = 12.0,
    marker_fill: bool = True,
    marker_edge_width: float = 1.4,
    line_alpha: float = 0.85,
    line_style: str = "-",
    line_width_scale: float = 1.0,
) -> None:
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


def camera_layout(n_cameras: int) -> tuple[int, int]:
    ncols = 4
    nrows = int(math.ceil(n_cameras / ncols))
    return nrows, ncols


def compute_pose_crop_limits_2d(
    raw_2d: np.ndarray,
    calibrations: dict,
    camera_names: list[str],
    margin: float,
) -> dict[str, np.ndarray]:
    """Calcule un cadrage par frame et par camera a partir de la sequence 2D."""
    limits: dict[str, np.ndarray] = {}
    for cam_idx, cam_name in enumerate(camera_names):
        width, height = calibrations[cam_name].image_size
        n_frames = raw_2d.shape[1]
        camera_limits = np.full((n_frames, 4), np.nan, dtype=float)
        for frame_idx in range(n_frames):
            points = raw_2d[cam_idx, frame_idx]
            valid = np.all(np.isfinite(points), axis=1)
            if not np.any(valid):
                continue
            xy = points[valid]
            xmin, ymin = np.min(xy, axis=0)
            xmax, ymax = np.max(xy, axis=0)
            dx = max(10.0, float(xmax - xmin) * margin)
            dy = max(10.0, float(ymax - ymin) * margin)
            camera_limits[frame_idx] = np.array(
                [
                    max(0.0, float(xmin - dx)),
                    min(float(width), float(xmax + dx)),
                    min(float(height), float(ymax + dy)),
                    max(0.0, float(ymin - dy)),
                ],
                dtype=float,
            )
        valid_frames = np.flatnonzero(np.all(np.isfinite(camera_limits), axis=1))
        if valid_frames.size == 0:
            camera_limits[:] = np.array([0.0, float(width), float(height), 0.0], dtype=float)
        else:
            first_valid = int(valid_frames[0])
            last_valid = int(valid_frames[-1])
            for frame_idx in range(0, first_valid):
                camera_limits[frame_idx] = camera_limits[first_valid]
            for frame_idx in range(first_valid + 1, n_frames):
                if not np.all(np.isfinite(camera_limits[frame_idx])):
                    camera_limits[frame_idx] = camera_limits[frame_idx - 1]
            for frame_idx in range(last_valid + 1, n_frames):
                camera_limits[frame_idx] = camera_limits[last_valid]
        limits[cam_name] = camera_limits
    return limits


def apply_2d_axis_limits(
    ax,
    *,
    crop_mode: str,
    crop_limits: dict[str, np.ndarray],
    cam_name: str,
    frame_idx: int,
    width: float,
    height: float,
) -> None:
    """Applique un cadrage 2D fixe et desactive l'autoscale matplotlib."""
    if crop_mode == "pose":
        x0, x1, y1, y0 = crop_limits[cam_name][frame_idx]
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
    else:
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
    ax.set_autoscale_on(False)


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
    if master_frames is None and pose2sim_trc.exists():
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
        return cached
    gui_debug(f"preview bundle cache miss dataset={output_dir}")
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
        filetypes: tuple[tuple[str, str], ...] | None = None,
        browse_initialdir: str | None = None,
    ):
        super().__init__(master)
        self.directory = directory
        self.filetypes = filetypes
        self.browse_initialdir = browse_initialdir
        self.label_widget = ttk.Label(self, text=label, width=label_width)
        self.label_widget.pack(side=tk.LEFT, padx=(0, 6))
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


def attach_tooltip(widget: tk.Widget, text: str) -> ToolTip:
    tooltip = ToolTip(widget, text)
    setattr(widget, "_tooltip_ref", tooltip)
    return tooltip


@dataclass
class SharedAppState:
    calib_var: tk.StringVar
    keypoints_var: tk.StringVar
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
    shared_reconstruction_selection: list[str] = field(default_factory=list)
    shared_reconstruction_selection_listeners: list[callable] = field(default_factory=list)
    shared_reconstruction_panel: object | None = None
    active_reconstruction_consumer: object | None = None

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


def current_dataset_name(state: SharedAppState) -> str:
    keypoints_path = ROOT / state.keypoints_var.get()
    trc_path = ROOT / state.pose2sim_trc_var.get() if state.pose2sim_trc_var.get().strip() else None
    return infer_dataset_name(keypoints_path=keypoints_path, pose2sim_trc=trc_path)


def current_dataset_dir(state: SharedAppState) -> Path:
    return ROOT / state.output_root_var.get() / current_dataset_name(state)


def current_selected_camera_names(state: SharedAppState) -> list[str]:
    return parse_camera_names(state.selected_camera_names_var.get())


def current_calibration_correction_mode(state: SharedAppState) -> str:
    raw = state.calibration_correction_var.get().strip()
    return raw if raw in {"none", "flip_epipolar", "flip_epipolar_fast", "flip_triangulation"} else "none"


def normalize_pose_correction_mode(raw: str) -> str:
    value = str(raw).strip()
    return value if value in {"none", "flip_epipolar", "flip_epipolar_fast", "flip_triangulation"} else "none"


def shared_pose_data_kwargs(state: SharedAppState, *, data_mode: str | None = None) -> dict[str, object]:
    return {
        "data_mode": str(data_mode or state.pose_data_mode_var.get()),
        "smoothing_window": int(state.pose_filter_window_var.get()),
        "outlier_threshold_ratio": float(state.pose_outlier_ratio_var.get()),
        "lower_percentile": float(state.pose_p_low_var.get()),
        "upper_percentile": float(state.pose_p_high_var.get()),
    }


def current_models_dir(state: SharedAppState) -> Path:
    return dataset_models_dir(ROOT / state.output_root_var.get(), current_dataset_name(state))


def current_reconstructions_dir(state: SharedAppState) -> Path:
    return dataset_reconstructions_dir(ROOT / state.output_root_var.get(), current_dataset_name(state))


def current_figures_dir(state: SharedAppState) -> Path:
    return dataset_figures_dir(ROOT / state.output_root_var.get(), current_dataset_name(state))


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


def show_placeholder_figure(
    figure: Figure,
    canvas: FigureCanvasTkAgg,
    message: str,
    *,
    fontsize: int = 11,
    color: str = "#555555",
) -> None:
    """Render a lightweight placeholder message in one Matplotlib figure."""

    figure.clear()
    ax = figure.subplots(1, 1)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=fontsize, color=color)
    canvas.draw_idle()


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
) -> tuple[object, ...]:
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
    )


def get_cached_calibrations(state: SharedAppState, calib_path: Path) -> dict[str, object]:
    key = calibration_cache_key(calib_path)
    cached = state.calibration_cache.get(key)
    if cached is None:
        cached = load_calibrations(calib_path)
        state.calibration_cache[key] = cached
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
):
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
    )
    cached = state.pose_data_cache.get(cache_key)
    if cached is not None:
        calibrations = get_cached_calibrations(state, calib_path)
        return calibrations, cached
    calibrations = get_cached_calibrations(state, calib_path)
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
    )
    state.pose_data_cache[cache_key] = pose_data
    return calibrations, pose_data


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
            if flip_method in {"epipolar", "epipolar_fast"}
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
            columns=("label", "family", "frames", "reproj", "path"),
            show="headings",
            height=6,
            selectmode="extended",
        )
        self.tree.heading("label", text="Reconstruction")
        self.tree.heading("family", text="Family")
        self.tree.heading("frames", text="Frames")
        self.tree.heading("reproj", text="Reproj (px)")
        self.tree.heading("path", text="Path")
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
        for name in selection:
            if self.tree.exists(name):
                self.tree.selection_add(name)

    def selected_names(self) -> list[str]:
        selected = list(self.tree.selection())
        if selected:
            return selected
        return [name for name in self._default_names if self.tree.exists(name)]


class SharedReconstructionPanel(ttk.LabelFrame):
    """Single reconstruction selector shared by the active analysis tab."""

    def __init__(self, master, state: SharedAppState):
        super().__init__(master, text="Reconstructions")
        self.state = state
        self._default_names: list[str] = []
        self._selection_callback = None
        self._refresh_callback = None
        self._suspend_selection_callback = False

        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=8, pady=(6, 2))
        self.title_var = tk.StringVar(value="Reconstructions")
        ttk.Label(header, textvariable=self.title_var).pack(side=tk.LEFT)
        self.refresh_button = ttk.Button(header, text="Refresh available")
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
        attach_tooltip(self.tree, "Shared reconstruction selector used by the active analysis tab.")
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

        self.title_var.set(title)
        self._refresh_callback = refresh_callback
        self._selection_callback = selection_callback
        self.refresh_button.configure(command=refresh_callback)
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
        selected = list(self.tree.selection())
        if selected:
            return selected
        return [name for name in self._default_names if self.tree.exists(name)]

    def show_placeholder(self, message: str) -> None:
        """Replace the table content with an empty informative state."""

        self.title_var.set("Reconstructions")
        self._default_names = []
        self._selection_callback = None
        self._refresh_callback = None
        self.refresh_button.configure(command=lambda: None)
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


class CommandTab(ttk.Frame):
    """Base pour un onglet qui construit et lance une commande."""

    def __init__(self, master, title: str):
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

        self.buttons_frame = ttk.Frame(self.main)
        self.buttons_frame.pack(fill=tk.X, pady=(8, 8))
        ttk.Button(self.buttons_frame, text="Preview command", command=self.update_preview).pack(side=tk.LEFT)
        ttk.Button(self.buttons_frame, text="Copy command", command=self.copy_command).pack(side=tk.LEFT, padx=(8, 0))
        self.run_button = ttk.Button(self.buttons_frame, text="Run", command=self.run_command)
        self.run_button.pack(side=tk.LEFT, padx=(8, 0))
        self.stop_button = ttk.Button(self.buttons_frame, text="Stop", command=self.stop_command)
        self.stop_button.pack(side=tk.LEFT, padx=(8, 0))

        self.progress_row = ttk.Frame(self.main)
        self.progress_row.pack(fill=tk.X, pady=(0, 8))
        self.progress_bar = ttk.Progressbar(self.progress_row, mode="determinate", maximum=1.0, value=0.0)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(self.progress_row, textvariable=self.progress_text, width=48).pack(side=tk.LEFT, padx=(10, 0))

        self.command_preview_label = ttk.Label(
            self.main, textvariable=self.command_preview, wraplength=1100, justify=tk.LEFT
        )
        self.command_preview_label.pack(fill=tk.X, pady=(0, 8))
        self.output = ScrolledText(self.main, height=18, wrap=tk.WORD)
        self.output.pack(fill=tk.BOTH, expand=True)

    def build_command(self) -> list[str]:
        raise NotImplementedError

    def update_preview(self) -> None:
        self.command_preview.set(shlex.join(self.build_command()))

    def set_run_button_text(self, text: str) -> None:
        self.run_button.configure(text=text)

    def attach_primary_action_button(self, button: ttk.Button, run_text: str = "Run", stop_text: str = "Stop") -> None:
        """Register a tab-local action button that mirrors the process state."""
        self.primary_action_button = button
        self.primary_run_text = run_text
        self.primary_stop_text = stop_text
        self.update_action_button_state()

    def hide_default_command_buttons(self) -> None:
        """Hide only the shared command buttons while keeping progress and logs visible."""
        self.buttons_frame.pack_forget()

    def hide_command_controls(self) -> None:
        self.buttons_frame.pack_forget()
        self.progress_row.pack_forget()
        self.command_preview_label.pack_forget()
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
        self.output.insert(tk.END, text)
        self.output.see(tk.END)

    def reset_progress(self) -> None:
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate", maximum=1.0, value=0.0)
        self.progress_text.set("Idle")
        self._profile_total = 0
        self._profile_index = 0
        self._profile_name = ""

    def start_indeterminate_progress(self, label: str = "Running...") -> None:
        self.progress_bar.stop()
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start(12)
        self.progress_text.set(label)

    def set_progress(self, current: float, total: float, label: str) -> None:
        total = max(float(total), 1.0)
        current = min(max(float(current), 0.0), total)
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
                self.output.after(0, self.append_output, line)
                self.output.after(0, self.handle_output_line, line)
        return_code = self.process.wait()
        self.output.after(0, self.append_output, f"\n[exit code {return_code}]\n")
        self.output.after(0, self.finish_progress, return_code)

    def finish_progress(self, return_code: int) -> None:
        self.progress_bar.stop()
        if return_code == 0:
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
        self.output_dir = LabeledEntry(form, "Output dir", "outputs/vitpose_full", browse=True, directory=True)
        self.output_dir.pack(fill=tk.X, padx=8, pady=4)
        self.biomod = LabeledEntry(form, "bioMod", "outputs/vitpose_full/vitpose_chain.bioMod", browse=True)
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
            row0, textvariable=self.pose_data_mode, values=["raw", "filtered", "cleaned"], width=12, state="readonly"
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
            "Version des detections 2D utilisee par le pipeline: brute, lissee, ou nettoyee apres rejet des points aberrants.",
        )
        attach_tooltip(
            pose_mode_box,
            "Version des detections 2D utilisee par le pipeline: brute, lissee, ou nettoyee apres rejet des points aberrants.",
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
            values=["epipolar", "triangulation_once", "triangulation_greedy", "triangulation_exhaustive"],
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
        self.no_unwrap_var = tk.BooleanVar(value=False)
        self.model_only_var = tk.BooleanVar(value=False)
        self.triang_only_var = tk.BooleanVar(value=False)
        self.reuse_triang_var = tk.BooleanVar(value=False)
        self.skip_low_coh_var = tk.BooleanVar(value=False)
        self.lock_dof_var = tk.BooleanVar(value=False)
        self.animate_var = tk.BooleanVar(value=False)
        check_tooltips = {
            "compare-biorbd-kalman": "Calcule aussi la comparaison directe avec l'EKF 3D biorbd.",
            "run-ekf-2d-flip-acc": "Lance la variante EKF 2D avec correction gauche/droite et predicteur acceleration.",
            "no-root-unwrap": "Desactive l'unwrap temporel des angles de racine.",
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
            ("no-root-unwrap", self.no_unwrap_var),
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
        if self.no_unwrap_var.get():
            cmd.append("--no-root-unwrap")
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
        super().__init__(master, "3D animation")
        self.state = state
        self.bundle: dict[str, object] | None = None
        self._view_state: tuple[float, float, float] | None = None
        self._dragging_frame_scale = False
        self.uses_shared_reconstruction_panel = True
        self.shared_reconstruction_selectmode = "extended"
        form = ttk.LabelFrame(self.main, text="animation/animate_dual_stick_comparison.py")
        form.pack(fill=tk.X, pady=(0, 8), before=self.output)

        self.output_gif = LabeledEntry(
            form, "Output GIF", display_path(current_figures_dir(state) / "dual_animation.gif")
        )
        self.output_gif.pack(fill=tk.X, padx=8, pady=4)

        row = ttk.Frame(form)
        row.pack(fill=tk.X, padx=8, pady=4)
        self.fps = LabeledEntry(row, "GIF fps", "12", label_width=7, entry_width=5)
        self.fps.pack(side=tk.LEFT, padx=(0, 6))
        self.stride = LabeledEntry(row, "Stride", "5", label_width=6, entry_width=4)
        self.stride.pack(side=tk.LEFT, padx=(0, 6))
        self.show_trunk_frames_var = tk.BooleanVar(value=False)
        show_trunk_check = ttk.Checkbutton(
            row, text="Show trunk local frames", variable=self.show_trunk_frames_var, command=self.refresh_preview
        )
        show_trunk_check.pack(side=tk.LEFT, padx=(0, 8))
        framing_label = ttk.Label(row, text="Framing", width=8)
        framing_label.pack(side=tk.LEFT)
        self.framing_var = tk.StringVar(value="full")
        framing_box = ttk.Combobox(
            row, textvariable=self.framing_var, values=["tight", "full"], width=8, state="readonly"
        )
        framing_box.pack(side=tk.LEFT, padx=(0, 6))
        self.marker_size = LabeledEntry(row, "Marker size", "8", label_width=10, entry_width=4)
        self.marker_size.pack(side=tk.LEFT, padx=(0, 6))
        self.generate_button = ttk.Button(row, text="GENERATE GIF", command=self.toggle_run_command)
        self.generate_button.pack(side=tk.RIGHT)
        self.attach_primary_action_button(self.generate_button, run_text="GENERATE GIF", stop_text="STOP")
        self.hide_default_command_buttons()

        preview_box = ttk.LabelFrame(self.main, text="Preview 3D")
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
        self.output_gif.set_tooltip("Chemin du GIF 3D a exporter.")
        self.fps.set_tooltip("Frequence d'images du GIF exporte.")
        self.stride.set_tooltip("Une frame sur N est exportee dans le GIF.")
        self.marker_size.set_tooltip("Taille visuelle des marqueurs du squelette 3D.")
        attach_tooltip(show_trunk_check, "Affiche le repere local du tronc pour chaque reconstruction selectionnee.")
        attach_tooltip(
            framing_label, "Tight adapte le cadrage a chaque frame; full garde le meme volume pour toute la sequence."
        )
        attach_tooltip(
            framing_box, "Tight adapte le cadrage a chaque frame; full garde le meme volume pour toute la sequence."
        )
        attach_tooltip(self.generate_button, "Genere le GIF 3D, puis devient STOP tant que l'export est en cours.")
        attach_tooltip(self.frame_scale, "Slider de navigation temporelle du preview 3D.")
        attach_tooltip(self.frame_label, "Index de frame actuellement affiche dans le preview 3D.")
        self.extra.set_tooltip(
            "Options CLI supplémentaires pour animation/animate_dual_stick_comparison.py, par exemple: --align-root"
        )
        self.framing_var.trace_add("write", lambda *_args: self.refresh_preview())
        self.state.keypoints_var.trace_add("write", lambda *_args: self.sync_dataset_defaults())
        self.state.output_root_var.trace_add("write", lambda *_args: self.sync_dataset_defaults())
        self.state.register_reconstruction_listener(self.refresh_available_reconstructions)
        self.refresh_available_reconstructions()

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
        self.output_gif.var.set(display_path(current_figures_dir(self.state) / "dual_animation.gif"))
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
            self.output_gif.get(),
            "--fps",
            self.fps.get(),
            "--stride",
            self.stride.get(),
            "--marker-size",
            self.marker_size.get(),
            "--framing",
            self.framing_var.get(),
        ]
        selected = [name for name in self.selected_reconstruction_names() if name in available]
        if selected:
            cmd.extend(["--show", *selected])
        if self.show_trunk_frames_var.get():
            cmd.append("--show-trunk-frames")
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
        for name in show_names:
            frame_points = available[name][frame_idx]
            draw_skeleton_3d(ax, frame_points, reconstruction_color(name), reconstruction_label(name))
            if self.show_trunk_frames_var.get():
                origin, rotation = compute_root_frame_from_points(frame_points)
                if origin is not None and rotation is not None:
                    draw_coordinate_system(
                        ax,
                        origin,
                        rotation,
                        scale=0.18,
                        alpha=0.95,
                        prefix=f"{reconstruction_label(name)[:2]}_",
                        show_labels=False,
                        line_width=2.2,
                    )
        set_equal_3d_limits(ax, points_dict, None if self.framing_var.get() == "full" else frame_idx)
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
        super().__init__(master, "2D multiview")
        self.state = state
        self.pose_data = None
        self.calibrations = None
        self.preview_bundle = None
        self.projected_layers: dict[str, np.ndarray] = {}
        self.crop_limits_cache: dict[str, np.ndarray] = {}
        self.crop_limits_key: tuple[object, ...] | None = None
        self._dragging_frame_scale = False
        self.uses_shared_reconstruction_panel = True
        self.shared_reconstruction_selectmode = "extended"
        form = ttk.LabelFrame(self.main, text="animation/animate_multiview_2d_comparison.py")
        form.pack(fill=tk.X, pady=(0, 8), before=self.output)

        defaults = [("Output GIF", display_path(current_figures_dir(state) / "multiview_2d_comparison.gif"))]
        self.entries: dict[str, LabeledEntry] = {}
        for label, default in defaults:
            entry = LabeledEntry(form, label, default, browse=False, directory=False, readonly=False)
            entry.pack(fill=tk.X, padx=8, pady=4)
            self.entries[label] = entry

        row = ttk.Frame(form)
        row.pack(fill=tk.X, padx=8, pady=4)
        self.gif_fps = LabeledEntry(row, "GIF fps", "10", label_width=7, entry_width=6)
        self.gif_fps.pack(side=tk.LEFT, padx=(0, 6))
        self.stride = LabeledEntry(row, "Stride", "5", label_width=6, entry_width=4)
        self.stride.pack(side=tk.LEFT, padx=(0, 6))
        crop_mode_label = ttk.Label(row, text="Crop mode", width=10)
        crop_mode_label.pack(side=tk.LEFT)
        self.crop_mode = tk.StringVar(value="pose")
        crop_mode_box = ttk.Combobox(
            row, textvariable=self.crop_mode, values=["full", "pose"], width=10, state="readonly"
        )
        crop_mode_box.pack(side=tk.LEFT, padx=(0, 6))
        self.marker_size = LabeledEntry(row, "Marker size", "18", label_width=10, entry_width=5)
        self.marker_size.pack(side=tk.LEFT, padx=(0, 6))
        self.crop_margin = LabeledEntry(row, "Crop margin", "0.1", label_width=10, entry_width=5)
        self.crop_margin.pack(side=tk.LEFT, padx=(0, 6))
        self.generate_button = ttk.Button(row, text="GENERATE GIF", command=self.toggle_run_command)
        self.generate_button.pack(side=tk.RIGHT)
        self.attach_primary_action_button(self.generate_button, run_text="GENERATE GIF", stop_text="STOP")
        self.hide_default_command_buttons()

        preview_box = ttk.LabelFrame(self.main, text="Preview 2D multivues")
        preview_box.pack(fill=tk.BOTH, expand=True, pady=(0, 8), before=self.output)
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
        self.entries["Output GIF"].set_tooltip("Chemin du GIF multivues 2D a exporter.")
        self.gif_fps.set_tooltip("Frequence d'images du GIF 2D exporte.")
        self.stride.set_tooltip("Une frame sur N est exportee dans le GIF 2D.")
        attach_tooltip(crop_mode_label, "Choisit entre le champ complet de l'image ou un crop autour de la pose.")
        attach_tooltip(crop_mode_box, "Choisit entre le champ complet de l'image ou un crop autour de la pose.")
        self.marker_size.set_tooltip("Taille visuelle des marqueurs 2D dans le preview et le GIF.")
        self.crop_margin.set_tooltip("Marge ajoutee autour de la pose quand le crop est actif.")
        attach_tooltip(
            self.generate_button, "Genere le GIF 2D multivues, puis devient STOP tant que l'export est en cours."
        )
        attach_tooltip(self.frame_scale, "Slider de navigation temporelle du preview 2D.")
        attach_tooltip(self.frame_label, "Index de frame actuellement affiche dans le preview 2D.")
        self.extra.set_tooltip(
            "Options CLI supplémentaires pour animation/animate_multiview_2d_comparison.py, par exemple: --crop-mode full"
        )
        self.crop_mode.trace_add("write", lambda *_args: self.refresh_preview())
        self.crop_margin.var.trace_add("write", lambda *_args: self.refresh_preview())
        self.state.keypoints_var.trace_add("write", lambda *_args: self.sync_dataset_defaults())
        self.state.output_root_var.trace_add("write", lambda *_args: self.sync_dataset_defaults())
        self.state.register_reconstruction_listener(self.refresh_available_reconstructions)
        self.refresh_available_reconstructions()

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
        self.entries["Output GIF"].var.set(
            display_path(current_figures_dir(self.state) / "multiview_2d_comparison.gif")
        )
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
            self.entries["Output GIF"].get(),
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
            self.crop_mode.get(),
            "--crop-margin",
            self.crop_margin.get(),
        ]
        selected = [name for name in self.selected_reconstruction_names() if name in available]
        if selected:
            cmd.extend(["--show", *selected])
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
            self._publish_reconstruction_rows(preview_load.preview_state.rows, preview_load.preview_state.defaults)
            camera_names = list(self.pose_data.camera_names)
            self.projected_layers = {}
            for name, points_3d in self.preview_bundle["recon_3d"].items():
                self.projected_layers[name] = project_points_all_cameras(points_3d, self.calibrations, camera_names)
            self.crop_limits_cache = {}
            self.crop_limits_key = None
            bundle_frames = (
                preview_load.preview_state.max_frame + 1
                if self.preview_bundle is not None
                else len(self.pose_data.frames)
            )
            n_frames = min(len(self.pose_data.frames), bundle_frames)
            self.frame_scale.configure(to=max(n_frames - 1, 0))
            self.frame_var.set(0)
            self.preview_canvas_widget.focus_set()
            self.refresh_preview()
        except Exception as exc:
            messagebox.showerror("2D multiview", str(exc))

    def refresh_preview(self) -> None:
        if self.pose_data is None or self.calibrations is None:
            return
        n_frames = len(self.pose_data.frames)
        if self.preview_bundle is not None and len(self.preview_bundle["frames"]):
            n_frames = min(n_frames, len(self.preview_bundle["frames"]))
        if n_frames == 0:
            return
        frame_idx = clamp_frame_index(int(round(self.frame_var.get())), n_frames - 1)
        self.frame_var.set(frame_idx)
        self.frame_label.configure(text=f"frame {frame_idx}")

        self.preview_figure.clear()
        camera_names = list(self.pose_data.camera_names)
        nrows, ncols = camera_layout(len(camera_names))
        axes = self.preview_figure.subplots(nrows, ncols)
        axes = np.atleast_1d(axes).ravel()
        raw_points = np.asarray(
            self.pose_data.raw_keypoints if self.pose_data.raw_keypoints is not None else self.pose_data.keypoints,
            dtype=float,
        )
        crop_points = np.asarray(self.pose_data.keypoints, dtype=float)
        crop_mode = self.crop_mode.get()
        try:
            crop_margin = float(self.crop_margin.get())
        except ValueError:
            crop_margin = 0.1
        crop_limits = self._ensure_crop_limits(crop_points, camera_names, crop_margin) if crop_mode == "pose" else {}

        for ax_idx, ax in enumerate(axes):
            if ax_idx >= len(camera_names):
                ax.axis("off")
                continue
            cam_name = camera_names[ax_idx]
            width, height = self.calibrations[cam_name].image_size
            apply_2d_axis_limits(
                ax,
                crop_mode=crop_mode,
                crop_limits=crop_limits,
                cam_name=cam_name,
                frame_idx=frame_idx,
                width=width,
                height=height,
            )
            ax.set_title(cam_name.replace("Camera", ""))
            ax.grid(alpha=0.15)
            selected = self.selected_reconstruction_names()
            if "raw" in selected:
                draw_skeleton_2d(
                    ax, raw_points[ax_idx, frame_idx], "#444444", "Raw", marker_size=float(self.marker_size.get())
                )
            for raw_name in selected:
                mapped = "ekf_2d_acc" if raw_name == "ekf_2d" else raw_name
                mapped = "triangulation_adaptive" if raw_name == "triangulation" else mapped
                if mapped == "biorbd_kalman":
                    mapped = "ekf_3d"
                if mapped == "raw" or mapped not in self.projected_layers:
                    continue
                points_2d = self.projected_layers[mapped][ax_idx, frame_idx]
                draw_skeleton_2d(
                    ax,
                    points_2d,
                    reconstruction_color(mapped),
                    reconstruction_label(mapped),
                    marker_size=float(self.marker_size.get()),
                )
            if ax_idx >= (nrows - 1) * ncols:
                ax.set_xlabel("x (px)")
            ax.set_ylabel("y (px)")

        handles, labels = axes[0].get_legend_handles_labels() if axes.size else ([], [])
        if handles:
            uniq = {}
            for handle, label in zip(handles, labels):
                uniq[label] = handle
            self.preview_figure.legend(
                list(uniq.values()), list(uniq.keys()), loc="upper center", ncol=min(5, len(uniq)), fontsize=8
            )
        self.preview_figure.tight_layout()
        self.preview_canvas.draw_idle()

    def _ensure_crop_limits(
        self, crop_points: np.ndarray, camera_names: list[str], crop_margin: float
    ) -> dict[str, np.ndarray]:
        cache_key = (
            id(self.pose_data),
            id(self.calibrations),
            tuple(camera_names),
            tuple(crop_points.shape),
            float(crop_margin),
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


class FiguresTab(CommandTab):
    def __init__(self, master):
        super().__init__(master, "Figures")
        form = ttk.LabelFrame(self.main, text="analysis/plot_kinematic_comparison.py")
        form.pack(fill=tk.X, pady=(0, 8), before=self.output)

        self.input_dir = LabeledEntry(form, "Input dir", "outputs/vitpose_full", browse=True, directory=True)
        self.input_dir.pack(fill=tk.X, padx=8, pady=4)
        self.output_dir = LabeledEntry(form, "Output dir", "outputs/vitpose_full/figures", browse=True, directory=True)
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
            self.entry_a.var.set("outputs/vitpose_full/ekf_states.npz")
            self.entry_b.var.set("outputs/vitpose_full/jump_segmentation.png")
            self.entry_c.var.set("outputs/vitpose_full/jump_rotations.png")
            self.opt_1.var.set("120")
            self.opt_2.var.set("0.20")
            self.opt_3.var.set("0.15")
        elif script == "analysis/plot_triangulation_view_usage.py":
            self.entry_a.var.set("outputs/vitpose_full/triangulation_pose2sim_like.npz")
            self.entry_b.var.set("")
            self.entry_c.var.set("outputs/vitpose_full/triangulation_view_usage.png")
            self.opt_1.var.set("120")
            self.opt_2.var.set("used")
            self.opt_3.var.set("")
        elif script == "analysis/plot_triangulated_marker_trajectories.py":
            self.entry_a.var.set("outputs/vitpose_full/triangulation_pose2sim_like.npz")
            self.entry_b.var.set("outputs/vitpose_full/summary.json")
            self.entry_c.var.set("outputs/vitpose_full/triangulated_marker_trajectories.png")
            self.opt_1.var.set("120")
            self.opt_2.var.set("1.5")
            self.opt_3.var.set("2")
        else:
            self.entry_a.var.set("outputs/vitpose_full/triangulation_pose2sim_like.npz")
            self.entry_b.var.set("inputs/calibration/Calib.toml")
            self.entry_c.var.set("outputs/vitpose_full/posture_snapshots_3d.png")
            self.opt_1.var.set("120")
            self.opt_2.var.set("7")
            self.opt_3.var.set("outputs/vitpose_full/first_frame_root_coordinate_system.png")

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
            "Pose2Sim TRC",
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
        self.output_root = LabeledEntry(
            row_shared, "Output root", browse=True, directory=True, label_width=10, entry_width=24
        )
        self.output_root.var = state.output_root_var
        self.output_root.entry_widget.configure(textvariable=self.output_root.var)
        self.output_root.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
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
            values=["none", "flip_epipolar", "flip_epipolar_fast", "flip_triangulation"],
            width=18,
            state="readonly",
        )
        self.calibration_correction_box.pack(side=tk.LEFT, padx=(0, 6))
        self.selected_cameras_label_var = tk.StringVar(value="Cameras: all")
        ttk.Label(row_shared, textvariable=self.selected_cameras_label_var, foreground="#4f5b66").pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(row_shared, text="Clean trial outputs", command=self.clean_trial_outputs).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Label(controls, textvariable=self.dataset_summary_var, foreground="#4f5b66", justify=tk.LEFT).pack(
            fill=tk.X, padx=8, pady=(0, 4)
        )

        row_display = ttk.Frame(controls)
        row_display.pack(fill=tk.X, padx=8, pady=4)
        component_label = ttk.Label(row_display, text="Composante", width=10)
        component_label.pack(side=tk.LEFT)
        self.component = tk.StringVar(value="y")
        component_box = ttk.Combobox(
            row_display, textvariable=self.component, values=["x", "y"], width=6, state="readonly"
        )
        component_box.pack(side=tk.LEFT, padx=(0, 8))
        view_mode_label = ttk.Label(row_display, text="Traitement", width=10)
        view_mode_label.pack(side=tk.LEFT)
        self.view_mode = state.pose_data_mode_var
        self.view_mode_menu = ttk.Combobox(
            row_display,
            textvariable=self.view_mode,
            values=["raw", "filtered", "cleaned"],
            width=10,
            state="readonly",
        )
        self.view_mode_menu.pack(side=tk.LEFT, padx=(0, 8))
        flip_mode_label = ttk.Label(row_display, text="Correction L/R", width=12)
        flip_mode_label.pack(side=tk.LEFT)
        self.flip_mode = tk.StringVar(value="none")
        self.flip_mode_menu = ttk.Combobox(
            row_display,
            textvariable=self.flip_mode,
            values=["none", "epipolar", "epipolar_fast", "triangulation"],
            width=12,
            state="readonly",
        )
        self.flip_mode_menu.pack(side=tk.LEFT, padx=(0, 8))
        self.pose_filter_window = LabeledEntry(row_display, "Filter window", "9", label_width=10, entry_width=4)
        self.pose_filter_window.var = state.pose_filter_window_var
        self.pose_filter_window.entry_widget.configure(textvariable=self.pose_filter_window.var)
        self.pose_filter_window.pack(side=tk.LEFT, padx=(0, 6))
        self.pose_outlier_ratio = LabeledEntry(row_display, "Outlier ratio", "0.10", label_width=10, entry_width=5)
        self.pose_outlier_ratio.var = state.pose_outlier_ratio_var
        self.pose_outlier_ratio.entry_widget.configure(textvariable=self.pose_outlier_ratio.var)
        self.pose_outlier_ratio.pack(side=tk.LEFT)

        row_clean = ttk.Frame(controls)
        row_clean.pack(fill=tk.X, padx=8, pady=4)
        self.pose_p_low = LabeledEntry(row_clean, "P low", "5", label_width=6, entry_width=4)
        self.pose_p_low.var = state.pose_p_low_var
        self.pose_p_low.entry_widget.configure(textvariable=self.pose_p_low.var)
        self.pose_p_low.pack(side=tk.LEFT, padx=(0, 6))
        self.pose_p_high = LabeledEntry(row_clean, "P high", "95", label_width=6, entry_width=4)
        self.pose_p_high.var = state.pose_p_high_var
        self.pose_p_high.entry_widget.configure(textvariable=self.pose_p_high.var)
        self.pose_p_high.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(row_clean, text="Load 2D data", command=self.load_data).pack(side=tk.LEFT)
        ttk.Button(row_clean, text="Refresh", command=self.reload_data).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(row_clean, text="Choix des keypoints:").pack(side=tk.LEFT, padx=(14, 6))

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
        self.flip_min_cameras = LabeledEntry(
            row_flip, "Min cams", str(DEFAULT_FLIP_MIN_OTHER_CAMERAS), label_width=8, entry_width=4
        )
        self.flip_min_cameras.var = state.flip_min_other_cameras_var
        self.flip_min_cameras.entry_widget.configure(textvariable=self.flip_min_cameras.var)
        self.flip_min_cameras.pack(side=tk.LEFT, padx=(0, 10))
        triang_flip_label = ttk.Label(row_flip, text="Triang flip", width=9)
        triang_flip_label.pack(side=tk.LEFT)
        self.triang_flip_method = tk.StringVar(value="raw")
        self.triang_flip_method_box = ttk.Combobox(
            row_flip,
            textvariable=self.triang_flip_method,
            values=["once", "greedy", "exhaustive"],
            width=10,
            state="readonly",
        )
        self.triang_flip_method_box.pack(side=tk.LEFT, padx=(0, 10))
        self.flip_temporal_weight = LabeledEntry(
            row_flip, "Temp w", str(DEFAULT_FLIP_TEMPORAL_WEIGHT), label_width=7, entry_width=4
        )
        self.flip_temporal_weight.var = state.flip_temporal_weight_var
        self.flip_temporal_weight.entry_widget.configure(textvariable=self.flip_temporal_weight.var)
        self.flip_temporal_weight.pack(side=tk.LEFT, padx=(0, 6))
        self.flip_temporal_tau = LabeledEntry(
            row_flip, "Temp tau", str(DEFAULT_FLIP_TEMPORAL_TAU_PX), label_width=8, entry_width=4
        )
        self.flip_temporal_tau.var = state.flip_temporal_tau_px_var
        self.flip_temporal_tau.entry_widget.configure(textvariable=self.flip_temporal_tau.var)
        self.flip_temporal_tau.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(row_flip, textvariable=self.flip_status_var, foreground="#4f5b66", justify=tk.LEFT).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

        self.keypoint_list = tk.Listbox(controls, selectmode=tk.MULTIPLE, exportselection=False, height=5)
        for name in COCO17:
            self.keypoint_list.insert(tk.END, name)
        for idx in range(len(COCO17)):
            self.keypoint_list.selection_set(idx)
        self.keypoint_list.pack(fill=tk.X, padx=8, pady=4)

        figure_box = ttk.LabelFrame(self, text="Courbes temporelles par camera")
        figure_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, master=figure_box)
        self.toolbar = NavigationToolbar2Tk(self.canvas, figure_box, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill=tk.X, padx=8, pady=(4, 0))
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.calib.set_tooltip("Fichier de calibration partagé par tous les onglets.")
        self.keypoints.set_tooltip("Fichier JSON des détections 2D. Son nom détermine aussi le nom du dataset.")
        self.pose2sim_trc.set_tooltip(
            "TRC utilisé pour Pose2Sim. Il est recherché automatiquement à partir du JSON 2D."
        )
        self.output_root.set_tooltip(
            "Dossier racine partagé des sorties. Le dataset sera créé automatiquement dessous."
        )
        self.fps.set_tooltip("FPS partage pour les derives temporelles et les animations.")
        self.workers.set_tooltip("Nombre de workers partage pour les rendus et les calculs paralleles.")
        attach_tooltip(
            self.root_rotfix_check,
            "Si coché, la racine est réalignée en lacet autour de Z à partir de l'axe médio-latéral du tronc à t0. L'angle est arrondi au multiple de pi/2 le plus proche, puis partagé par le modèle, les reconstructions et les analyses.",
        )
        attach_tooltip(
            calib_correction_label,
            "Choisit quelle variante 2D sera utilisée par les outils de calibration/caméras: aucune correction, flip détecté par épipolaire Sampson, flip épipolaire rapide par distance symétrique, ou flip détecté par triangulation.",
        )
        attach_tooltip(
            self.calibration_correction_box,
            "Choisit quelle variante 2D sera utilisée par les outils de calibration/caméras: aucune correction, flip détecté par épipolaire Sampson, flip épipolaire rapide par distance symétrique, ou flip détecté par triangulation.",
        )
        attach_tooltip(component_label, "Composante 2D affichée sur les courbes temporelles.")
        attach_tooltip(component_box, "Composante 2D affichée sur les courbes temporelles.")
        attach_tooltip(view_mode_label, "Traitement affiché: brut, filtré, ou nettoyé après rejet des outliers.")
        attach_tooltip(self.view_mode_menu, "Traitement affiché: brut, filtré, ou nettoyé après rejet des outliers.")
        attach_tooltip(
            flip_mode_label,
            "Applique visuellement une correction gauche/droite basée sur le diagnostic choisi. Les lignes verticales montrent les suspicions épipolaires Sampson (rouge), épipolaires rapides distance symétrique (orange) et triangulation (bleu).",
        )
        attach_tooltip(
            self.flip_mode_menu,
            "Applique visuellement une correction gauche/droite basée sur le diagnostic choisi. Les lignes verticales montrent les suspicions épipolaires Sampson (rouge), épipolaires rapides distance symétrique (orange) et triangulation (bleu).",
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
            self.trc_status_var.set(f"Pose2Sim TRC auto-détecté: {rel}")
        else:
            if self.state.pose2sim_trc_var.get():
                self.state.pose2sim_trc_var.set("")
            self.trc_status_var.set(
                f"Aucun fichier TRC correspondant n'a été trouvé pour {keypoints_path.name}. "
                "Les reconstructions Pose2Sim resteront indisponibles tant qu'un TRC ne sera pas fourni."
            )
        self.update_dataset_summary()
        self.update_flip_status_text()

    def on_flip_settings_changed(self) -> None:
        self.flip_masks = {key: value for key, value in self.flip_masks.items() if key in {"epipolar", "epipolar_fast"}}
        self.flip_diagnostics = {
            key: value for key, value in self.flip_diagnostics.items() if key in {"epipolar", "epipolar_fast"}
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
        if not dataset_dir.exists():
            messagebox.showinfo(
                "Clean trial outputs", f"No outputs found for this dataset:\n{display_path(dataset_dir)}"
            )
            return
        confirmed = messagebox.askyesno(
            "Clean trial outputs",
            "Delete all generated outputs for the current dataset?\n\n"
            f"{display_path(dataset_dir)}\n\n"
            "This will remove models, reconstructions, figures, caches, and generated files for this trial.",
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
            messagebox.showinfo("Clean trial outputs", f"Deleted outputs for:\n{display_path(dataset_dir)}")
        except Exception as exc:
            messagebox.showerror("Clean trial outputs", str(exc))

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
        for method in ("epipolar", "epipolar_fast", selected_triangulation_method):
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
                        if method in {"epipolar", "epipolar_fast"}
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
        if correction_mode in {"epipolar", "epipolar_fast"} and correction_mode in self.flip_masks:
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
            f"triang mode {self.selected_triangulation_flip_method()} | flips: epi rouge --, epi fast orange -., triang bleu :",
            y=0.98,
        )
        self.figure.tight_layout()
        self.canvas.draw_idle()


class ModelTab(CommandTab):
    def __init__(self, master, state: SharedAppState):
        super().__init__(master, "Model")
        self.state = state
        self.preview_points = None
        self.preview_support_points = None
        self.preview_q0: np.ndarray | None = None
        self.preview_q_t0: np.ndarray | None = None
        self.preview_q_current: np.ndarray | None = None
        self.preview_q_names: list[str] = []
        self.preview_model = None
        self.preview_marker_names: list[str] = []
        self.preview_segment_frames: list[tuple[str, np.ndarray, np.ndarray]] = []
        self.preview_metadata: dict[str, object] = {}
        self._updating_dof_controls = False
        self.set_run_button_text("Generate model")

        form = ttk.LabelFrame(self.main, text="Construction du modèle")
        form.pack(fill=tk.X, pady=(0, 8), before=self.output)

        row = ttk.Frame(form)
        row.pack(fill=tk.X, padx=8, pady=4)
        self.subject_mass = LabeledEntry(row, "Subject mass", "55", label_width=10, entry_width=6)
        self.subject_mass.pack(side=tk.LEFT, padx=(0, 8))
        self.triang_method = tk.StringVar(value="exhaustive")
        triang_label = ttk.Label(row, text="Triangulation", width=12)
        triang_label.pack(side=tk.LEFT)
        triang_box = ttk.Combobox(
            row, textvariable=self.triang_method, values=["once", "greedy", "exhaustive"], width=12, state="readonly"
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
        if default_model_pose_mode not in ("raw", "filtered", "cleaned"):
            default_model_pose_mode = "cleaned"
        self.pose_data_mode = tk.StringVar(value=default_model_pose_mode)
        pose_mode_label = ttk.Label(row2b, text="2D source", width=10)
        pose_mode_label.pack(side=tk.LEFT)
        pose_mode_box = ttk.Combobox(
            row2b, textvariable=self.pose_data_mode, values=["raw", "filtered", "cleaned"], width=10, state="readonly"
        )
        pose_mode_box.pack(side=tk.LEFT, padx=(0, 8))
        default_pose_correction_mode = current_calibration_correction_mode(state)
        self.pose_correction_mode = tk.StringVar(value=default_pose_correction_mode)
        pose_correction_label = ttk.Label(row2b, text="L/R corr", width=8)
        pose_correction_label.pack(side=tk.LEFT)
        pose_correction_box = ttk.Combobox(
            row2b,
            textvariable=self.pose_correction_mode,
            values=["none", "flip_epipolar", "flip_epipolar_fast", "flip_triangulation"],
            width=16,
            state="readonly",
        )
        pose_correction_box.pack(side=tk.LEFT, padx=(0, 8))
        self.model_info_var = tk.StringVar(value="")
        ttk.Label(row2b, textvariable=self.model_info_var, foreground="#4f5b66").pack(side=tk.LEFT, padx=(6, 0))

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
            triang_label, "Méthode de triangulation utilisée pour construire le modèle: once, greedy, ou exhaustive."
        )
        attach_tooltip(
            triang_box,
            "once: une seule triangulation pondérée. greedy: suppression gloutonne des pires vues. exhaustive: la plus robuste mais plus coûteuse.",
        )
        attach_tooltip(
            pose_mode_label,
            "Choix de la version de base des 2D utilisées pour construire le modèle: raw, filtered ou cleaned.",
        )
        attach_tooltip(
            pose_mode_box,
            "Choix de la version de base des 2D utilisées pour construire le modèle. `cleaned` applique le rejet des points aberrants.",
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
        self.state.output_root_var.trace_add("write", lambda *_args: self.sync_paths_from_state())
        self.state.register_reconstruction_listener(self.refresh_existing_models)
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
        self.output.configure(height=8)

        # Layout: commandes/logs et liste des modeles a gauche, preview a droite.
        self.buttons_frame.pack_forget()
        self.progress_row.pack_forget()
        self.command_preview_label.pack_forget()
        self.output.pack_forget()

        form.pack_forget()
        form.grid(row=0, column=0, sticky="ew", padx=(0, 10), pady=(0, 8))
        self.buttons_frame.grid(row=1, column=0, sticky="ew", padx=(0, 10), pady=(0, 8))
        self.progress_row.grid(row=2, column=0, sticky="ew", padx=(0, 10), pady=(0, 8))
        self.command_preview_label.grid(row=3, column=0, sticky="ew", padx=(0, 10), pady=(0, 8))
        self.output.grid(row=4, column=0, sticky="nsew", padx=(0, 10), pady=(0, 8))

        existing_box = ttk.LabelFrame(self.main, text="Existing models")
        existing_box.grid(row=5, column=0, sticky="nsew", padx=(0, 10), pady=(0, 8))
        existing_controls = ttk.Frame(existing_box)
        existing_controls.pack(fill=tk.X, padx=8, pady=(8, 0))
        ttk.Button(existing_controls, text="Refresh models", command=self.refresh_existing_models).pack(side=tk.LEFT)
        ttk.Button(existing_controls, text="Clear models", command=self.clear_models).pack(side=tk.LEFT, padx=(8, 0))
        self.model_tree = ttk.Treeview(existing_box, columns=("name", "path"), show="headings", height=12)
        self.model_tree.heading("name", text="Model")
        self.model_tree.heading("path", text="bioMod path")
        self.model_tree.column("name", width=160, anchor="w")
        self.model_tree.column("path", width=360, anchor="w")
        self.model_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.model_tree.bind("<<TreeviewSelect>>", lambda _event: self.load_preview(use_selected_model=True))
        attach_tooltip(
            self.model_tree, "Liste des modeles compatibles avec les options 2D et de construction courantes."
        )

        preview_box = ttk.LabelFrame(self.main, text="Première frame triangulée / modèle")
        preview_box.grid(row=0, column=1, rowspan=6, sticky="nsew", pady=(0, 8))
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
        attach_tooltip(self.preview_dof_box, "Choisit le DoF du modele a modifier manuellement dans le preview.")
        attach_tooltip(self.preview_dof_slider, "Fait varier le DoF selectionne entre -2pi et 2pi.")
        self.preview_figure = Figure(figsize=(8, 6))
        self.preview_canvas = FigureCanvasTkAgg(self.preview_figure, master=preview_box)
        self.preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.extra = LabeledEntry(form, "Extra args", "")
        self.extra.pack(fill=tk.X, padx=8, pady=4)

        self.main.grid_columnconfigure(0, weight=1, uniform="modeltab")
        self.main.grid_columnconfigure(1, weight=2, uniform="modeltab")
        self.main.grid_rowconfigure(4, weight=1)
        self.main.grid_rowconfigure(5, weight=2)

        self.update_details()
        self.sync_paths_from_state()
        self.refresh_existing_models()

    def current_pose_correction_mode(self) -> str:
        return normalize_pose_correction_mode(self.pose_correction_mode.get())

    def current_pose_source_label(self) -> str:
        correction_mode = self.current_pose_correction_mode()
        if correction_mode == "none":
            return self.pose_data_mode.get()
        return f"{self.pose_data_mode.get()} + {correction_mode}"

    def update_details(self) -> None:
        max_frames = self.max_frames.get() or "all"
        frame_start = self.frame_start.get() or "-"
        frame_end = self.frame_end.get() or "-"
        self.details_var.set(
            f"Model creation will use: {self.current_pose_source_label()} 2D data, "
            f"frames {frame_start} -> {frame_end}, nb {max_frames}, "
            f"triangulation {self.triang_method.get()}, "
            f"root rot-fix {'on' if self.initial_rot_var.get() else 'off'}, "
            f"subject mass {self.subject_mass.get()} kg."
        )
        self.sync_paths_from_state()

    def sync_paths_from_state(self) -> None:
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
        output_root = ROOT / self.state.output_root_var.get()
        model_dir = model_output_dir(
            output_root,
            dataset_name,
            pose_data_mode=self.pose_data_mode.get(),
            triangulation_method=self.triang_method.get(),
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

    def refresh_existing_models(self) -> None:
        for item in self.model_tree.get_children():
            self.model_tree.delete(item)
        dataset_dir = current_dataset_dir(self.state)
        biomod_paths: list[Path] = []
        for model_dir in scan_model_dirs(dataset_dir):
            biomod_paths.extend(sorted(model_dir.glob("*.bioMod")))
        expected_mode = self.pose_data_mode.get()
        expected_correction = self.current_pose_correction_mode()
        expected_window = int(self.state.pose_filter_window_var.get())
        expected_ratio = float(self.state.pose_outlier_ratio_var.get())
        expected_p_low = float(self.state.pose_p_low_var.get())
        expected_p_high = float(self.state.pose_p_high_var.get())
        seen: set[Path] = set()
        matched = 0
        for biomod_path in biomod_paths:
            if biomod_path in seen:
                continue
            seen.add(biomod_path)
            if not self._model_matches_selected_2d_data(
                biomod_path.parent,
                expected_mode,
                expected_correction,
                expected_window,
                expected_ratio,
                expected_p_low,
                expected_p_high,
            ):
                continue
            parent_name = biomod_path.parent.name
            biomod_display_path = display_path(biomod_path)
            self.model_tree.insert("", "end", values=(parent_name, biomod_display_path))
            matched += 1
        if matched == 0:
            self.model_tree.insert(
                "",
                "end",
                values=(
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
                if len(values) >= 2 and values[1] and not str(values[1]).startswith("No existing"):
                    paths.append(parse_displayed_path(str(values[1])))
        else:
            for item in self.model_tree.get_children():
                values = self.model_tree.item(item, "values")
                if len(values) >= 2 and values[1] and not str(values[1]).startswith("No existing"):
                    paths.append(parse_displayed_path(str(values[1])))

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
        expected_window: int,
        expected_ratio: float,
        expected_p_low: float,
        expected_p_high: float,
    ) -> bool:
        model_stage_path = model_dir / "model_stage.npz"
        stage_metadata = self._cache_metadata(model_stage_path)
        reconstruction_cache_path = (
            Path(stage_metadata.get("reconstruction_cache_path", ""))
            if stage_metadata.get("reconstruction_cache_path")
            else None
        )
        if reconstruction_cache_path is None or not reconstruction_cache_path.exists():
            return False
        reconstruction_metadata = self._cache_metadata(reconstruction_cache_path)
        if not reconstruction_metadata:
            return False
        return (
            reconstruction_metadata.get("pose_data_mode") == expected_mode
            and str(reconstruction_metadata.get("pose_correction_mode", "none")) == expected_correction
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
            "--output-dir",
            display_path(self.derived_model_dir()),
            "--biomod",
            self.derived_biomod_path(),
        ]
        selected_cameras = current_selected_camera_names(self.state)
        if selected_cameras:
            cmd.extend(["--camera-names", ",".join(selected_cameras)])
        if self.frame_start.get():
            cmd.extend(["--frame-start", self.frame_start.get()])
        if self.frame_end.get():
            cmd.extend(["--frame-end", self.frame_end.get()])
        if self.max_frames.get():
            cmd.extend(["--max-frames", self.max_frames.get()])
        if self.initial_rot_var.get():
            cmd.append("--initial-rotation-correction")
        cmd.extend(self.parse_extra_args(self.extra.get()))
        return cmd

    def derived_model_dir(self) -> Path:
        dataset_name = current_dataset_name(self.state)
        output_root = ROOT / self.state.output_root_var.get()
        return model_output_dir(
            output_root,
            dataset_name,
            pose_data_mode=self.pose_data_mode.get(),
            triangulation_method=self.triang_method.get(),
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
        output_root = ROOT / self.state.output_root_var.get()
        return display_path(
            model_biomod_path(
                output_root,
                dataset_name,
                pose_data_mode=self.pose_data_mode.get(),
                triangulation_method=self.triang_method.get(),
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
        if len(values) < 2 or not values[1] or str(values[1]).startswith("No existing"):
            return None
        path = Path(str(values[1]))
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
            min_cameras_for_triangulation=DEFAULT_MIN_CAMERAS_FOR_TRIANGULATION,
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
                    min_cameras_for_triangulation=DEFAULT_MIN_CAMERAS_FOR_TRIANGULATION,
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

        form = ttk.LabelFrame(self.main, text="Profils de reconstruction")
        form.pack(fill=tk.X, pady=(0, 8), before=self.output)

        self.config_path = LabeledEntry(form, "Config JSON", browse=True)
        self.config_path.var = state.profiles_config_var
        self.config_path.entry_widget.configure(textvariable=self.config_path.var)
        self.config_path.pack(fill=tk.X, padx=8, pady=4)
        self.config_path.set_tooltip("Fichier JSON dans lequel charger ou sauvegarder les profils.")

        info = ttk.Label(
            form,
            text="Les chemins source, le dossier de sortie, le FPS et les workers sont repris depuis le 1er onglet.",
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
        cameras_label = ttk.Label(cameras_header, text="Cameras", width=10)
        cameras_label.pack(side=tk.LEFT)
        ttk.Button(cameras_header, text="Use current selection", command=self.use_current_camera_selection).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(cameras_header, text="All cameras", command=self.clear_profile_camera_selection).pack(side=tk.LEFT)
        self.profile_cameras_summary = tk.StringVar(value="all cameras")
        ttk.Label(cameras_header, textvariable=self.profile_cameras_summary, foreground="#4f5b66").pack(
            side=tk.LEFT, padx=(8, 0)
        )
        cameras_body = ttk.Frame(self.cameras_frame)
        cameras_body.pack(fill=tk.X, pady=(4, 0))
        self.profile_cameras_list = tk.Listbox(cameras_body, selectmode="extended", exportselection=False, height=4)
        self.profile_cameras_list.pack(fill=tk.X, expand=True)

        self.pose_mode_frame = ttk.Frame(form)
        mode_label = ttk.Label(self.pose_mode_frame, text="2D mode", width=10)
        mode_label.pack(side=tk.LEFT)
        self.pose_data_mode = tk.StringVar(value="cleaned")
        pose_mode_box = ttk.Combobox(
            self.pose_mode_frame,
            textvariable=self.pose_data_mode,
            values=["raw", "filtered", "cleaned"],
            width=12,
            state="readonly",
        )
        pose_mode_box.pack(side=tk.LEFT, padx=(0, 8))
        stride_label = ttk.Label(self.pose_mode_frame, text="1/N", width=4)
        stride_label.pack(side=tk.LEFT)
        self.frame_stride = tk.StringVar(value="1")
        stride_box = ttk.Combobox(
            self.pose_mode_frame, textvariable=self.frame_stride, values=["1", "2", "3", "4"], width=4, state="readonly"
        )
        stride_box.pack(side=tk.LEFT, padx=(0, 8))
        self.common_frame = ttk.Frame(form)
        self.common_frame.pack(fill=tk.X, padx=8, pady=4)
        self.initial_rot_var = state.initial_rotation_correction_var
        self.unwrap_var = tk.BooleanVar(value=False)
        initial_rot_check = ttk.Checkbutton(
            self.common_frame, text="initial-rotation-correction", variable=self.initial_rot_var
        )
        initial_rot_check.pack(side=tk.LEFT, padx=(0, 12))
        unwrap_check = ttk.Checkbutton(self.common_frame, text="no-root-unwrap", variable=self.unwrap_var)
        unwrap_check.pack(side=tk.LEFT)

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
        self.flip_var = tk.BooleanVar(value=False)
        flip_check = ttk.Checkbutton(self.flip_frame, text="flip left/right", variable=self.flip_var)
        flip_check.pack(side=tk.LEFT, padx=(0, 12))

        self.ekf2d_frame = ttk.Frame(form)
        predictor_label = ttk.Label(self.ekf2d_frame, text="Predictor", width=10)
        predictor_label.pack(side=tk.LEFT)
        self.predictor = tk.StringVar(value="acc")
        predictor_box = ttk.Combobox(
            self.ekf2d_frame, textvariable=self.predictor, values=["acc", "dyn"], width=8, state="readonly"
        )
        predictor_box.pack(side=tk.LEFT, padx=(0, 8))
        ekf2d_source_label = ttk.Label(self.ekf2d_frame, text="3D source", width=10)
        ekf2d_source_label.pack(side=tk.LEFT)
        self.ekf2d_3d_source = tk.StringVar(value="full_triangulation")
        ekf2d_source_box = ttk.Combobox(
            self.ekf2d_frame,
            textvariable=self.ekf2d_3d_source,
            values=["full_triangulation", "first_frame_only"],
            width=18,
            state="readonly",
        )
        ekf2d_source_box.pack(side=tk.LEFT, padx=(0, 8))
        coherence_label = ttk.Label(self.ekf2d_frame, text="Coherence", width=10)
        coherence_label.pack(side=tk.LEFT)
        self.coherence_method = tk.StringVar(value="epipolar")
        coherence_box = ttk.Combobox(
            self.ekf2d_frame,
            textvariable=self.coherence_method,
            values=["epipolar", "triangulation_once", "triangulation_greedy", "triangulation_exhaustive"],
            width=24,
            state="readonly",
        )
        coherence_box.pack(side=tk.LEFT, padx=(0, 8))
        self.lock_var = tk.BooleanVar(value=False)
        lock_check = ttk.Checkbutton(self.ekf2d_frame, text="dof_locking", variable=self.lock_var)
        lock_check.pack(side=tk.LEFT)

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
        self.ekf2d_bootstrap_passes = LabeledEntry(self.ekf2d_params_frame, "Boot passes", "5")
        self.ekf2d_bootstrap_passes.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.measurement_noise = LabeledEntry(self.ekf2d_params_frame, "EKF2D meas", "1.5")
        self.measurement_noise.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.process_noise = LabeledEntry(self.ekf2d_params_frame, "EKF2D proc", "1.0")
        self.process_noise.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.coherence_floor = LabeledEntry(self.ekf2d_params_frame, "Conf floor", "0.35")
        self.coherence_floor.pack(side=tk.LEFT, fill=tk.X, expand=True)

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
        self.biorbd_noise = LabeledEntry(self.ekf3d_frame, "EKF3D noise", "1e-8")
        self.biorbd_noise.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.biorbd_error = LabeledEntry(self.ekf3d_frame, "EKF3D error", "1e-4")
        self.biorbd_error.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.clean_frame = ttk.Frame(form)
        self.pose_filter_window = LabeledEntry(self.clean_frame, "Filter window", "9")
        self.pose_filter_window.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.pose_outlier_ratio = LabeledEntry(self.clean_frame, "Outlier ratio", "0.10")
        self.pose_outlier_ratio.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.pose_p_low = LabeledEntry(self.clean_frame, "P low", "5")
        self.pose_p_low.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.pose_p_high = LabeledEntry(self.clean_frame, "P high", "95")
        self.pose_p_high.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.config_path.set_tooltip("Fichier JSON contenant les profils de reconstruction sauvegardes.")
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
        attach_tooltip(unwrap_check, "Desactive l'unwrap temporel de la racine dans les reconstructions geometriques.")
        attach_tooltip(triang_label, "Choisit la variante de triangulation 3D: once, greedy, ou exhaustive.")
        attach_tooltip(
            triang_box,
            "once: une seule DLT pondérée. greedy: suppression gloutonne. exhaustive: la plus robuste mais plus coûteuse.",
        )
        attach_tooltip(
            flip_check, "Autorise une correction globale gauche/droite des keypoints 2D avant reconstruction."
        )
        attach_tooltip(predictor_label, "Choisit le predicteur dynamique de l'EKF 2D.")
        attach_tooltip(predictor_box, "Choisit le predicteur dynamique de l'EKF 2D.")
        attach_tooltip(ekf2d_source_label, "Choisit comment obtenir l'information 3D de support pour EKF 2D.")
        attach_tooltip(
            ekf2d_source_box,
            "Source 3D utilisee par EKF 2D pour le modele et q0: triangulation complete ou bootstrap sur la premiere frame seulement.",
        )
        attach_tooltip(
            coherence_label,
            "Pondération multivue de l'EKF 2D: épipolaire, ou basée sur une triangulation once, greedy, ou exhaustive.",
        )
        attach_tooltip(
            coherence_box,
            "Chaque variante triangulation_* choisit explicitement la cohérence issue de cette variante de triangulation.",
        )
        attach_tooltip(lock_check, "Verrouille certains DoF pour stabiliser l'EKF 2D.")
        attach_tooltip(
            q0_method_label,
            "Methode pour trouver q0: IK 3D sur la triangulation, bootstrap EKF classique, ou bootstrap depuis une pose racine geometrique extraite des hanches/epaules.",
        )
        attach_tooltip(
            q0_method_box,
            "Methode pour trouver q0: IK 3D sur la triangulation ou corrections EKF 2D repetees en remettant qdot/qddot a zero.",
        )
        self.ekf2d_bootstrap_passes.set_tooltip(
            "Nombre de passes EKF 2D utilisees pour affiner q0 quand le bootstrap est actif."
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
        self.pose_filter_window.set_tooltip("Fenêtre du lissage utilisé pour la référence filtrée 2D.")
        self.pose_outlier_ratio.set_tooltip("Ratio de rejet des points 2D trop éloignés de la référence filtrée.")
        self.pose_p_low.set_tooltip("Percentile bas pour l'amplitude robuste du mouvement 2D.")
        self.pose_p_high.set_tooltip("Percentile haut pour l'amplitude robuste du mouvement 2D.")

        actions = ttk.Frame(form)
        actions.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(actions, text="Add / replace current profile", command=self.add_current_profile).pack(side=tk.LEFT)
        ttk.Button(actions, text="-", width=3, command=self.remove_selected_profiles).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(actions, text="Generate examples", command=self.generate_examples).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(actions, text="Generate all supported", command=self.generate_all_supported).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(actions, text="Load JSON", command=self.load_profiles_from_json).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(actions, text="Save JSON", command=self.save_profiles_to_json).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(actions, text="Refresh variants scan", command=self.refresh_variant_tree).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        cols = ("enabled", "name", "family", "mode", "triang", "flags")
        self.profile_tree = ttk.Treeview(form, columns=cols, show="headings", height=8, selectmode="extended")
        headings = {
            "enabled": "Use",
            "name": "Name",
            "family": "Family",
            "mode": "2D mode",
            "triang": "Triang",
            "flags": "Flags",
        }
        widths = {"enabled": 50, "name": 240, "family": 90, "mode": 90, "triang": 100, "flags": 320}
        for col in cols:
            self.profile_tree.heading(col, text=headings[col])
            self.profile_tree.column(col, width=widths[col], anchor="w")
        self.profile_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.profile_tree.bind("<Delete>", lambda _event: self.remove_selected_profiles())
        self.profile_tree.bind("<BackSpace>", lambda _event: self.remove_selected_profiles())

        variants_box = ttk.LabelFrame(self.main, text="Reconstructions détectées pour le dataset courant")
        variants_box.pack(fill=tk.BOTH, expand=True, pady=(0, 8), before=self.output)
        self.variant_tree = ttk.Treeview(
            variants_box, columns=("name", "family", "latest", "path"), show="headings", height=5
        )
        for col, label, width in [
            ("name", "Variant", 220),
            ("family", "Family", 100),
            ("latest", "A jour", 80),
            ("path", "Path", 620),
        ]:
            self.variant_tree.heading(col, text=label)
            self.variant_tree.column(col, width=width, anchor="w")
        self.variant_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.family.trace_add("write", lambda *_args: self.update_family_controls())
        self.family.trace_add("write", lambda *_args: self.sync_profile_name())
        self.pose_data_mode.trace_add("write", lambda *_args: self.sync_profile_name())
        self.frame_stride.trace_add("write", lambda *_args: self.sync_profile_name())
        self.triang_method.trace_add("write", lambda *_args: self.sync_profile_name())
        self.predictor.trace_add("write", lambda *_args: self.sync_profile_name())
        self.ekf2d_3d_source.trace_add("write", lambda *_args: self.sync_profile_name())
        self.ekf2d_initial_state_method.trace_add("write", lambda *_args: self.sync_profile_name())
        self.biorbd_kalman_init_method.trace_add("write", lambda *_args: self.sync_profile_name())
        self.ekf2d_bootstrap_passes.var.trace_add("write", lambda *_args: self.sync_profile_name())
        self.flip_var.trace_add("write", lambda *_args: self.sync_profile_name())
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
        self.state.selected_camera_names_var.trace_add("write", lambda *_args: self.update_profile_camera_summary())
        self.profile_cameras_list.bind("<<ListboxSelect>>", lambda _event: self.on_profile_camera_selection_changed())
        self.state.register_profile_listener(self.refresh_profile_tree)
        self.state.register_reconstruction_listener(self.refresh_variant_tree)
        self.refresh_profile_camera_choices()
        self.update_family_controls()
        self.sync_profile_name()
        self.refresh_profile_tree()
        self.refresh_variant_tree()
        self.hide_command_controls()

    def update_family_controls(self) -> None:
        for frame in [
            self.cameras_frame,
            self.pose_mode_frame,
            self.triang_frame,
            self.flip_frame,
            self.ekf2d_frame,
            self.ekf2d_params_frame,
            self.ekf3d_frame,
            self.clean_frame,
        ]:
            frame.pack_forget()
        family = self.family.get()
        self.cameras_frame.pack(fill=tk.X, padx=8, pady=4)
        if family in ("triangulation", "ekf_3d", "ekf_2d"):
            self.pose_mode_frame.pack(fill=tk.X, padx=8, pady=4)
            self.clean_frame.pack(fill=tk.X, padx=8, pady=4)
        if family in ("triangulation", "ekf_3d", "ekf_2d"):
            self.triang_frame.pack(fill=tk.X, padx=8, pady=4)
        if family in ("triangulation", "ekf_3d", "ekf_2d"):
            self.flip_frame.pack(fill=tk.X, padx=8, pady=4)
        if family == "ekf_2d":
            self.ekf2d_frame.pack(fill=tk.X, padx=8, pady=4)
            self.ekf2d_params_frame.pack(fill=tk.X, padx=8, pady=4)
        if family == "ekf_3d":
            self.ekf3d_frame.pack(fill=tk.X, padx=8, pady=4)

    def selected_profile_camera_names(self) -> list[str] | None:
        indices = [int(index) for index in self.profile_cameras_list.curselection()]
        if not indices:
            return None
        camera_names = [str(self.profile_cameras_list.get(index)) for index in indices]
        return camera_names or None

    def _set_profile_camera_selection(self, camera_names: list[str] | None) -> None:
        requested = set(camera_names or [])
        self.profile_cameras_list.selection_clear(0, tk.END)
        for index in range(self.profile_cameras_list.size()):
            if str(self.profile_cameras_list.get(index)) in requested:
                self.profile_cameras_list.selection_set(index)
        self.update_profile_camera_summary()

    def refresh_profile_camera_choices(self) -> None:
        selected_before = self.selected_profile_camera_names() or current_selected_camera_names(self.state)
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
        self._set_profile_camera_selection(selected_before if camera_names else None)
        self.sync_profile_name()

    def update_profile_camera_summary(self) -> None:
        selected = self.selected_profile_camera_names()
        self.profile_cameras_summary.set(
            "all cameras" if not selected else f"profile cameras: {format_camera_names(selected)}"
        )

    def use_current_camera_selection(self) -> None:
        self._set_profile_camera_selection(current_selected_camera_names(self.state))
        self.sync_profile_name()

    def clear_profile_camera_selection(self) -> None:
        self._set_profile_camera_selection(None)
        self.sync_profile_name()

    def on_profile_camera_selection_changed(self) -> None:
        self.update_profile_camera_summary()
        self.sync_profile_name()

    def current_profile(self, *, include_name: bool = True) -> ReconstructionProfile:
        family = self.family.get()
        profile = ReconstructionProfile(
            name=self.profile_name.get() if include_name else "",
            family=family,
            camera_names=self.selected_profile_camera_names(),
            predictor=self.predictor.get() if family == "ekf_2d" else None,
            ekf2d_3d_source=self.ekf2d_3d_source.get() if family == "ekf_2d" else "full_triangulation",
            ekf2d_initial_state_method=self.ekf2d_initial_state_method.get() if family == "ekf_2d" else "ekf_bootstrap",
            ekf2d_bootstrap_passes=int(self.ekf2d_bootstrap_passes.get()) if family == "ekf_2d" else 5,
            flip=self.flip_var.get() if family in ("triangulation", "ekf_2d", "ekf_3d") else False,
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
            coherence_method=self.coherence_method.get() if family == "ekf_2d" else "epipolar",
            no_root_unwrap=self.unwrap_var.get(),
            biorbd_kalman_noise_factor=float(self.biorbd_noise.get()),
            biorbd_kalman_error_factor=float(self.biorbd_error.get()),
            biorbd_kalman_init_method=(
                self.biorbd_kalman_init_method.get() if family == "ekf_3d" else "triangulation_ik_root_translation"
            ),
            measurement_noise_scale=float(self.measurement_noise.get()),
            process_noise_scale=float(self.process_noise.get()),
            coherence_confidence_floor=float(self.coherence_floor.get()),
            pose_filter_window=int(self.pose_filter_window.get()),
            pose_outlier_threshold_ratio=float(self.pose_outlier_ratio.get()),
            pose_amplitude_lower_percentile=float(self.pose_p_low.get()),
            pose_amplitude_upper_percentile=float(self.pose_p_high.get()),
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
            if profile.family == "ekf_3d":
                init_method = getattr(profile, "biorbd_kalman_init_method", "triangulation_ik_root_translation")
                if init_method == "triangulation_ik":
                    flags.append("ikq0")
                elif init_method == "root_translation_zero_rest":
                    flags.append("roottransq0")
                elif init_method == "root_pose_zero_rest":
                    flags.append("rootq0")
            if profile.initial_rotation_correction:
                flags.append("rotfix")
            if profile.flip:
                flags.append("flip")
            if profile.dof_locking:
                flags.append("lock")
            if profile.no_root_unwrap:
                flags.append("no_unwrap")
            if getattr(profile, "camera_names", None):
                flags.append(f"cams[{format_camera_names(profile.camera_names)}]")
            if int(getattr(profile, "frame_stride", 1)) != 1:
                flags.append(f"1/{int(getattr(profile, 'frame_stride', 1))}")
            if profile.family in ("triangulation", "ekf_3d", "ekf_2d") and profile.pose_data_mode != "cleaned":
                flags.append(profile.pose_data_mode)
            mode_value = profile.pose_data_mode if profile.family in ("triangulation", "ekf_3d", "ekf_2d") else "-"
            triang_value = (
                profile.triangulation_method if profile.family in ("triangulation", "ekf_3d", "ekf_2d") else "-"
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
                    ",".join(flags),
                ),
            )

    def refresh_variant_tree(self) -> None:
        for item in self.variant_tree.get_children():
            self.variant_tree.delete(item)
        dataset_dir = current_dataset_dir(self.state)
        valid_parents = {dataset_dir, current_reconstructions_dir(self.state)}
        for path in scan_variant_output_dirs(dataset_dir.parent):
            if path.parent not in valid_parents:
                continue
            summary = load_bundle_summary(path)
            family = summary.get("family", "-")
            latest = summary.get("is_latest_family_version")
            self.variant_tree.insert(
                "",
                "end",
                values=(path.name, family, "-" if latest is None else ("oui" if latest else "non"), str(path)),
            )

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
        self.refresh_variant_tree()

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
        self.refresh_variant_tree()

    def generate_all_supported(self) -> None:
        self.state.set_profiles(generate_supported_profiles())
        synchronize_profiles_initial_rotation_correction(self.state)

    def load_profiles_from_json(self) -> None:
        try:
            self.state.set_profiles(load_profiles_json(ROOT / self.config_path.get()))
            synchronize_profiles_initial_rotation_correction(self.state)
            self.refresh_variant_tree()
        except Exception as exc:
            messagebox.showerror("Profiles", str(exc))

    def save_profiles_to_json(self) -> None:
        try:
            synchronize_profiles_initial_rotation_correction(self.state)
            save_profiles_json(ROOT / self.config_path.get(), self.state.profiles)
        except Exception as exc:
            messagebox.showerror("Profiles", str(exc))

    def selected_profiles(self) -> list[ReconstructionProfile]:
        selected = self.profile_tree.selection()
        if not selected:
            return [profile for profile in self.state.profiles if profile.enabled]
        return [self.state.profiles[int(item)] for item in selected]

    def build_command(self) -> list[str]:
        self.save_profiles_to_json()
        cmd = [
            sys.executable,
            "run_reconstruction_profiles.py",
            "--config",
            self.config_path.get(),
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
        selected_cameras = current_selected_camera_names(self.state)
        if selected_cameras:
            cmd.extend(["--camera-names", ",".join(selected_cameras)])
        if self.state.pose2sim_trc_var.get().strip():
            cmd.extend(["--pose2sim-trc", self.state.pose2sim_trc_var.get()])
        for profile in self.selected_profiles():
            cmd.extend(["--profile", profile.name])
        return cmd

    def on_command_success(self) -> None:
        self.refresh_variant_tree()
        self.state.notify_reconstructions_updated()


class ReconstructionsTab(CommandTab):
    def __init__(self, master, state: SharedAppState):
        super().__init__(master, "Reconstructions")
        self.state = state
        self.status_summaries: dict[str, dict[str, object]] = {}

        form = ttk.LabelFrame(self.main, text="Lancer les reconstructions depuis les profils")
        form.pack(fill=tk.X, pady=(0, 8), before=self.output)

        info = ttk.Label(
            form,
            text="Les options détaillées se règlent dans l'onglet Profiles. Ici on choisit simplement quoi lancer et on inspecte les caches du dataset courant.",
        )
        info.pack(fill=tk.X, padx=8, pady=(4, 6))

        self.config_path = LabeledEntry(form, "Config JSON", browse=True)
        self.config_path.var = state.profiles_config_var
        self.config_path.entry_widget.configure(textvariable=self.config_path.var)
        self.config_path.pack(fill=tk.X, padx=8, pady=4)

        controls = ttk.Frame(form)
        controls.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(controls, text="Refresh profiles", command=self.refresh_profile_tree).pack(side=tk.LEFT)
        ttk.Button(controls, text="Refresh caches", command=self.refresh_status_tree).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(controls, text="Clear reconstructions", command=self.clear_reconstructions).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        self.export_trc_button = ttk.Button(controls, text="Export TRC from q", command=self.export_selected_trc_from_q)
        self.export_trc_button.pack(side=tk.LEFT, padx=(8, 0))

        self.profile_tree = ttk.Treeview(
            form, columns=("name", "family", "mode", "flags"), show="headings", height=6, selectmode="extended"
        )
        for col, label, width in [
            ("name", "Name", 260),
            ("family", "Family", 90),
            ("mode", "2D mode", 90),
            ("flags", "Flags", 420),
        ]:
            self.profile_tree.heading(col, text=label)
            self.profile_tree.column(col, width=width, anchor="w")
        self.profile_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        attach_tooltip(self.profile_tree, "Selectionnez les profils a executer pour le dataset courant.")

        status_box = ttk.LabelFrame(self.main, text="Reconstructions disponibles pour le dataset courant")
        status_box.pack(fill=tk.BOTH, expand=True, pady=(0, 8), before=self.output)
        cols = ("name", "family", "latest", "frames", "compute_s", "reproj_mean", "reproj_std", "path")
        self.status_tree = ttk.Treeview(status_box, columns=cols, show="headings", height=8)
        headings = {
            "name": "Reconstruction",
            "family": "Family",
            "latest": "A jour",
            "frames": "Frames",
            "compute_s": "Compute (s)",
            "reproj_mean": "Reproj mean (px)",
            "reproj_std": "Reproj std (px)",
            "path": "Path",
        }
        widths = {
            "name": 220,
            "family": 90,
            "latest": 70,
            "frames": 70,
            "compute_s": 95,
            "reproj_mean": 110,
            "reproj_std": 110,
            "path": 360,
        }
        for col in cols:
            self.status_tree.heading(col, text=headings[col])
            self.status_tree.column(col, width=widths[col], anchor="w")
        self.status_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        attach_tooltip(
            self.status_tree,
            "Tableau des reconstructions du dataset courant, avec temps de calcul total et erreurs de reprojection.",
        )
        attach_tooltip(
            self.export_trc_button,
            "Reconstruit les marqueurs du modèle depuis q pour la reconstruction sélectionnée et écrit un fichier TRC dans son dossier.",
        )

        timing_box = ttk.LabelFrame(self.main, text="Durées détaillées de la reconstruction sélectionnée")
        timing_box.pack(fill=tk.BOTH, expand=False, pady=(0, 8), before=self.output)
        self.timing_details = ScrolledText(timing_box, height=10, wrap=tk.WORD)
        self.timing_details.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.timing_details.insert("1.0", "Select a reconstruction to inspect timing details.")
        self.timing_details.configure(state=tk.DISABLED)
        self.status_tree.bind("<<TreeviewSelect>>", lambda _event: self.refresh_timing_details())

        self.state.register_profile_listener(self.refresh_profile_tree)
        self.state.register_reconstruction_listener(self.refresh_status_tree)
        self.state.calib_var.trace_add("write", lambda *_args: self.refresh_status_tree())
        self.state.keypoints_var.trace_add("write", lambda *_args: self.refresh_status_tree())
        self.state.output_root_var.trace_add("write", lambda *_args: self.refresh_status_tree())
        self.refresh_profile_tree()
        self.refresh_status_tree()

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
            self.profile_tree.insert(
                "", "end", iid=str(idx), values=(profile.name, profile.family, profile.pose_data_mode, ",".join(flags))
            )

    def refresh_status_tree(self) -> None:
        selected = self.status_tree.selection()
        selected_iid = selected[0] if selected else None
        for item in self.status_tree.get_children():
            self.status_tree.delete(item)
        self.status_summaries = {}
        dataset_dir = current_dataset_dir(self.state)
        for recon_dir in reconstruction_dirs_for_path(dataset_dir):
            summary = load_bundle_summary(recon_dir)
            if not summary:
                continue
            reproj = summary.get("reprojection_px", {})
            latest = summary.get("is_latest_family_version")
            compute_s = objective_total_seconds(summary)
            iid = str(recon_dir)
            self.status_summaries[iid] = summary
            self.status_tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    summary.get("name", recon_dir.name),
                    summary.get("family", "-"),
                    "-" if latest is None else ("oui" if latest else "non"),
                    summary.get("n_frames", "-"),
                    "-" if compute_s is None else f"{compute_s:.2f}",
                    "-" if reproj.get("mean") is None else f"{float(reproj.get('mean')):.2f}",
                    "-" if reproj.get("std") is None else f"{float(reproj.get('std')):.2f}",
                    str(recon_dir),
                ),
            )
        if selected_iid and self.status_tree.exists(selected_iid):
            self.status_tree.selection_set(selected_iid)
            self.status_tree.focus(selected_iid)
        elif self.status_tree.get_children():
            first = self.status_tree.get_children()[0]
            self.status_tree.selection_set(first)
            self.status_tree.focus(first)
        self.refresh_timing_details()

    def export_selected_trc_from_q(self) -> None:
        """Export the selected q-based reconstruction as a TRC marker trajectory."""

        selected = self.status_tree.selection()
        if not selected:
            messagebox.showinfo("Export TRC from q", "Select one reconstruction first.")
            return

        recon_dir = Path(selected[0])
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
        frames = np.asarray(data["frames"], dtype=int) if "frames" in data else np.arange(q.shape[0], dtype=int)
        time_s = (
            np.asarray(data["time_s"], dtype=float)
            if "time_s" in data
            else np.arange(q.shape[0], dtype=float) / max(float(self.state.fps_var.get()), 1.0)
        )
        biomod_path = resolve_reconstruction_biomod(
            current_dataset_dir(self.state), self.status_summaries.get(str(recon_dir), {}).get("name", recon_dir.name)
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
        messagebox.showinfo("Export TRC from q", f"TRC written to:\n{output_path}")

    def refresh_timing_details(self) -> None:
        selected = self.status_tree.selection()
        text = "Select a reconstruction to inspect timing details."
        if selected:
            summary = self.status_summaries.get(selected[0], {})
            if summary:
                text = format_reconstruction_timing_details(summary)
        self.timing_details.configure(state=tk.NORMAL)
        self.timing_details.delete("1.0", tk.END)
        self.timing_details.insert("1.0", text)
        self.timing_details.configure(state=tk.DISABLED)

    def clear_reconstructions(self) -> None:
        dataset_dir = current_dataset_dir(self.state)
        recon_dirs = reconstruction_dirs_for_path(dataset_dir)
        if not recon_dirs:
            messagebox.showinfo("Reconstructions", f"Aucune reconstruction à supprimer dans {dataset_dir}.")
            return
        confirmed = messagebox.askyesno(
            "Clear reconstructions",
            f"Supprimer {len(recon_dirs)} reconstruction(s) du dataset courant ?\n\n{dataset_dir}",
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
        self.refresh_status_tree()
        self.state.notify_reconstructions_updated()
        if errors:
            messagebox.showerror("Reconstructions", "Certaines suppressions ont échoué:\n\n" + "\n".join(errors))
        else:
            messagebox.showinfo("Reconstructions", f"{len(recon_dirs)} reconstruction(s) supprimée(s).")

    def selected_profiles(self) -> list[ReconstructionProfile]:
        selected = self.profile_tree.selection()
        if not selected:
            return [profile for profile in self.state.profiles if profile.enabled]
        return [self.state.profiles[int(item)] for item in selected]

    def build_command(self) -> list[str]:
        save_profiles_json(ROOT / self.config_path.get(), self.state.profiles)
        cmd = [
            sys.executable,
            "run_reconstruction_profiles.py",
            "--config",
            self.config_path.get(),
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
            cmd.extend(["--pose2sim-trc", self.state.pose2sim_trc_var.get()])
        for profile in self.selected_profiles():
            cmd.extend(["--profile", profile.name])
        return cmd

    def on_command_success(self) -> None:
        self.refresh_status_tree()
        self.state.notify_reconstructions_updated()


class RootKinematicsTab(ttk.Frame):
    def __init__(self, master, state: SharedAppState):
        super().__init__(master)
        self.state = state
        self.bundle = None
        self.uses_shared_reconstruction_panel = True
        self.shared_reconstruction_selectmode = "extended"

        controls = ttk.LabelFrame(self, text="Cinématiques de la racine")
        controls.pack(fill=tk.X, padx=10, pady=10)
        self.output_dir = LabeledEntry(controls, "Dataset", display_path(current_dataset_dir(state)), readonly=True)
        self.output_dir.pack(fill=tk.X, padx=8, pady=4)

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
        rotation_unit_box.pack(side=tk.LEFT)

        row2 = ttk.Frame(controls)
        row2.pack(fill=tk.X, padx=8, pady=4)
        self.unwrap_var = tk.BooleanVar(value=True)
        self.reextract_var = tk.BooleanVar(value=True)
        self.fd_qdot_var = tk.BooleanVar(value=True)
        self.matrix_ignore_alpha_var = tk.BooleanVar(value=False)
        unwrap_check = ttk.Checkbutton(row2, text="unwrap rotations", variable=self.unwrap_var)
        unwrap_check.pack(side=tk.LEFT)
        reextract_check = ttk.Checkbutton(
            row2, text="recalcul matrice + re-extraction Euler", variable=self.reextract_var
        )
        reextract_check.pack(side=tk.LEFT, padx=(12, 0))
        fd_qdot_check = ttk.Checkbutton(row2, text="qdot par différence finie", variable=self.fd_qdot_var)
        fd_qdot_check.pack(side=tk.LEFT, padx=(12, 0))
        matrix_ignore_alpha_check = ttk.Checkbutton(
            row2, text="matrix without alpha correction", variable=self.matrix_ignore_alpha_var
        )
        matrix_ignore_alpha_check.pack(side=tk.LEFT, padx=(12, 0))
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
        attach_tooltip(unwrap_check, "Applique un unwrap temporel aux angles de racine pour eviter les sauts a +/-pi.")
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
        self.unwrap_var.trace_add("write", lambda *_args: self.refresh_plot())
        self.reextract_var.trace_add("write", lambda *_args: self.refresh_plot())
        self.fd_qdot_var.trace_add("write", lambda *_args: self.refresh_plot())
        self.matrix_ignore_alpha_var.trace_add("write", lambda *_args: self.refresh_plot())
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
        self.output_dir.var.set(display_path(current_dataset_dir(self.state)))
        self.refresh_available_reconstructions()

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
                self.state, ROOT / self.output_dir.get(), None, None, align_root=False
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
            family_is_translation = self.family.get() == "translations"
            matrix_mode = (not family_is_translation) and self.rotation_plot_mode.get() == "matrix"
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
                summary = recon_summary.get(name, {}) if isinstance(recon_summary, dict) else {}
                summary_family = str(summary.get("family", ""))
                geometric_family = summary_family in {"pose2sim", "triangulation"}
                geometric_rotfix_mismatch = False
                if name in recon_3d:
                    applied = summary.get("initial_rotation_correction_applied")
                    if applied is not None:
                        geometric_rotfix_mismatch = bool(applied) != bool(
                            self.state.initial_rotation_correction_var.get()
                        )
                if geometric_family and name in recon_3d:
                    series = root_series_from_points(
                        np.asarray(recon_3d[name], dtype=float),
                        quantity=effective_quantity,
                        dt=dt,
                        initial_rotation_correction=bool(self.state.initial_rotation_correction_var.get()),
                        unwrap_rotations=bool(self.unwrap_var.get()),
                    )
                elif name in recon_q:
                    series = root_series_from_q(
                        q_names,
                        recon_q[name],
                        quantity=effective_quantity,
                        dt=dt,
                        qdot=recon_qdot.get(name),
                        fd_qdot=bool(self.fd_qdot_var.get()),
                        unwrap_rotations=bool(self.unwrap_var.get()),
                        renormalize_rotations=bool(self.reextract_var.get()),
                    )
                elif name in recon_3d:
                    series = root_series_from_points(
                        np.asarray(recon_3d[name], dtype=float),
                        quantity=effective_quantity,
                        dt=dt,
                        initial_rotation_correction=bool(self.state.initial_rotation_correction_var.get()),
                        unwrap_rotations=bool(self.unwrap_var.get()),
                    )
                elif name in recon_q_root and not geometric_rotfix_mismatch:
                    series = root_series_from_precomputed(
                        np.asarray(recon_q_root[name], dtype=float),
                        quantity=effective_quantity,
                        dt=dt,
                        qdot_root=recon_qdot_root.get(name),
                        fd_qdot=bool(self.fd_qdot_var.get()),
                    )
                if series is None:
                    continue
                frame_slice = analysis_frame_slice(np.asarray(series).shape[0])
                if frame_slice.start >= np.asarray(series).shape[0]:
                    continue
                frame_indices = np.arange(np.asarray(series).shape[0], dtype=float)[frame_slice]
                t = frame_indices * dt
                if matrix_mode:
                    if geometric_family and name in recon_3d:
                        matrices = root_rotation_matrices_from_points(
                            np.asarray(recon_3d[name], dtype=float),
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
                            np.asarray(series, dtype=float),
                            initial_rotation_correction_angle_rad=initial_rotation_correction_angle_rad,
                        )
                    matrices = matrices[frame_slice]
                    for row_idx in range(3):
                        for col_idx in range(3):
                            ax = axes[row_idx, col_idx]
                            ax.plot(
                                t,
                                matrices[:, row_idx, col_idx],
                                color=reconstruction_color(name),
                                linewidth=1.5,
                                label=reconstruction_label(name),
                            )
                            ax.set_title(component_labels[row_idx][col_idx], fontsize=9)
                            ax.grid(alpha=0.25)
                else:
                    series_to_plot = scale_root_series_rotations(
                        np.asarray(series, dtype=float), family_is_translation, rotation_unit
                    )[frame_slice]
                    for axis_idx, ax in enumerate(axes):
                        ax.plot(
                            t,
                            series_to_plot[:, family_slice.start + axis_idx],
                            color=reconstruction_color(name),
                            linewidth=1.7,
                            label=reconstruction_label(name),
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
        self.output_dir = LabeledEntry(controls, "Dataset", display_path(current_dataset_dir(state)), readonly=True)
        self.output_dir.pack(fill=tk.X, padx=8, pady=4)

        row = ttk.Frame(controls)
        row.pack(fill=tk.X, padx=8, pady=4)
        quantity_label = ttk.Label(row, text="Quantité", width=10)
        quantity_label.pack(side=tk.LEFT)
        self.quantity = tk.StringVar(value="q")
        quantity_box = ttk.Combobox(row, textvariable=self.quantity, values=["q", "qdot"], width=10, state="readonly")
        quantity_box.pack(side=tk.LEFT, padx=(0, 6))
        self.fd_qdot_var = tk.BooleanVar(value=False)
        fd_qdot_check = ttk.Checkbutton(row, text="qdot par différence finie", variable=self.fd_qdot_var)
        fd_qdot_check.pack(side=tk.LEFT)
        ttk.Button(row, text="Load / refresh", command=self.refresh_plot).pack(side=tk.LEFT, padx=(12, 0))

        self.pair_list = tk.Listbox(controls, selectmode=tk.MULTIPLE, exportselection=False, height=6)
        self.pair_list.pack(fill=tk.X, padx=8, pady=4)
        attach_tooltip(quantity_label, "Choisit entre positions q et vitesses qdot pour les autres DoF.")
        attach_tooltip(quantity_box, "Choisit entre positions q et vitesses qdot pour les autres DoF.")
        attach_tooltip(fd_qdot_check, "Recalcule qdot par difference finie sur q.")
        attach_tooltip(self.pair_list, "Choisissez les paires gauche/droite de DoF a comparer sur les graphes.")

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

    def _show_empty_plot(self, message: str) -> None:
        """Render a placeholder when no joint-kinematics plot can be produced."""

        show_placeholder_figure(self.figure, self.canvas, message)

    def sync_dataset_dir(self) -> None:
        self.output_dir.var.set(display_path(current_dataset_dir(self.state)))
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
                self.state, ROOT / self.output_dir.get(), None, None, align_root=False
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
                    right_idx = name_to_index[right_name]
                    color = RECONSTRUCTION_COLORS.get(recon_name, "#333333")
                    ax.plot(
                        time_s,
                        series[:, left_idx],
                        color=color,
                        linewidth=1.7,
                        label=f"{RECONSTRUCTION_LABELS.get(recon_name, recon_name)} | L",
                    )
                    ax.plot(
                        time_s,
                        series[:, right_idx],
                        color=color,
                        linewidth=1.7,
                        linestyle="--",
                        label=f"{RECONSTRUCTION_LABELS.get(recon_name, recon_name)} | R",
                    )
                    plotted_series = True
                ax.set_title(pair_label)
                ax.grid(alpha=0.25)
                ax.set_ylabel(self.quantity.get())
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
        self.output_dir = LabeledEntry(controls, "Dataset", display_path(current_dataset_dir(state)), readonly=True)
        self.output_dir.pack(fill=tk.X, padx=8, pady=4)

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
        self.output_dir.var.set(display_path(current_dataset_dir(self.state)))
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
            dataset_dir = ROOT / self.output_dir.get()
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
        self.output_dir = LabeledEntry(controls, "Dataset", display_path(current_dataset_dir(state)), readonly=True)
        self.output_dir.pack(fill=tk.X, padx=8, pady=4)

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
        self.output_dir.var.set(display_path(current_dataset_dir(self.state)))
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
            dataset_dir = ROOT / self.output_dir.get()
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
                color = reconstruction_color(recon_name)
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
                box_ax.plot([], [], color=color, linewidth=6, alpha=0.35, label=reconstruction_label(recon_name))
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
                color = reconstruction_color(recon_name)
                label = reconstruction_label(recon_name)
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
        self._suspend_selection_callbacks = False
        self.uses_shared_reconstruction_panel = True
        self.shared_reconstruction_selectmode = "browse"

        controls = ttk.LabelFrame(self, text="Analyse d'execution")
        controls.pack(fill=tk.X, padx=10, pady=10)
        self.output_dir = LabeledEntry(controls, "Dataset", display_path(current_dataset_dir(state)), readonly=True)
        self.output_dir.pack(fill=tk.X, padx=8, pady=4)

        row = ttk.Frame(controls)
        row.pack(fill=tk.X, padx=8, pady=4)
        camera_label = ttk.Label(row, text="Camera", width=10)
        camera_label.pack(side=tk.LEFT)
        self.camera_name = tk.StringVar(value="")
        self.camera_box = ttk.Combobox(row, textvariable=self.camera_name, values=[], width=18, state="readonly")
        self.camera_box.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(row, text="Analyze / refresh", command=self.refresh_analysis).pack(side=tk.LEFT)
        ttk.Label(
            row,
            text=(
                "First pass: deductions are localized on one representative frame per error and "
                "highlighted on 3D markers plus one selected camera view."
            ),
            foreground="#4f5b66",
        ).pack(side=tk.LEFT, padx=(12, 0))
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
        self.output_dir.var.set(display_path(current_dataset_dir(self.state)))
        self.refresh_available_reconstructions()

    def refresh_available_reconstructions(self) -> None:
        try:
            dataset_dir = ROOT / self.output_dir.get()
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
            dataset_dir = ROOT / self.output_dir.get()
            biomod_path = resolve_preview_biomod(dataset_dir)
            pose2sim_trc = (
                ROOT / self.state.pose2sim_trc_var.get() if self.state.pose2sim_trc_var.get().strip() else None
            )
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
                self._show_empty_plot("Execution analysis requires q and 3D markers for the selected reconstruction.")
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
            self.dd_analysis = analyze_dd_session(
                np.asarray(root_q, dtype=float),
                fps,
                height_values=np.asarray(root_q[:, 2], dtype=float),
                angle_mode="euler",
                analysis_start_frame=ANALYSIS_START_FRAME,
                require_complete_jumps=True,
            )
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

        projected_points = np.full((len(COCO17), 2), np.nan, dtype=float)
        raw_points = np.full((len(COCO17), 2), np.nan, dtype=float)
        camera_name = self.camera_name.get().strip()
        if camera_name and isinstance(self.calibrations, dict) and camera_name in self.calibrations:
            calibration = self.calibrations[camera_name]
            for point_idx, point in enumerate(frame_points_3d):
                if np.all(np.isfinite(point)):
                    projected_points[point_idx] = calibration.project_point(point)
            if self.pose_data is not None and getattr(self.pose_data, "camera_names", None) is not None:
                camera_names = [str(name) for name in self.pose_data.camera_names]
                if camera_name in camera_names:
                    cam_idx = camera_names.index(camera_name)
                    pose_frames = np.asarray(self.pose_data.frames, dtype=int)
                    bundle_frames = np.asarray(
                        self.bundle.get("frames", np.arange(points_3d.shape[0], dtype=int)), dtype=int
                    )
                    frame_number = int(bundle_frames[min(focus_frame_idx, len(bundle_frames) - 1)])
                    pose_idx = np.where(pose_frames == frame_number)[0]
                    if pose_idx.size:
                        raw_points = np.asarray(self.pose_data.keypoints[cam_idx, int(pose_idx[0])], dtype=float)
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
                reconstruction_color(self.current_reconstruction_name),
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
        view_2d_ax.set_title(f"2D localization | {camera_name or 'no camera'}")
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

        controls = ttk.LabelFrame(self, text="Sélection de caméras + inspection flip L/R")
        controls.pack(fill=tk.X, padx=10, pady=10)
        self.dataset_dir = LabeledEntry(controls, "Dataset", display_path(current_dataset_dir(state)), readonly=True)
        self.dataset_dir.pack(fill=tk.X, padx=8, pady=4)

        row = ttk.Frame(controls)
        row.pack(fill=tk.X, padx=8, pady=4)
        reference_label = ttk.Label(row, text="Reference", width=10)
        reference_label.pack(side=tk.LEFT)
        self.reference_name_var = tk.StringVar(value="")
        self.reference_box = ttk.Combobox(row, textvariable=self.reference_name_var, width=28, state="readonly")
        self.reference_box.pack(side=tk.LEFT, padx=(0, 8))
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
                "reprojection quality, triangulation usage, and flip rates.\n"
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
            values=["epipolar", "epipolar_fast", "triangulation"],
            width=16,
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

        self.dataset_dir.set_tooltip(
            "Dataset courant utilise pour calculer les scores caméra et relire les caches de flip."
        )
        attach_tooltip(
            reference_label,
            "Reconstruction de référence utilisée pour relire epipolar coherence, reprojection per view et usage dans la triangulation.",
        )
        attach_tooltip(
            self.reference_box,
            "Reconstruction de référence utilisée pour relire epipolar coherence, reprojection per view et usage dans la triangulation.",
        )
        attach_tooltip(best_n_label, "Nombre de caméras à présélectionner automatiquement selon le classement courant.")
        attach_tooltip(best_n_entry, "Nombre de caméras à présélectionner automatiquement selon le classement courant.")
        attach_tooltip(
            self.metrics_tree,
            "Scores comparatifs pour choisir un sous-ensemble de caméras plus stable pour la reconstruction.",
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
        attach_tooltip(self.flip_frame_list, "Frames suspectes ou candidates pour la caméra et la méthode choisies.")
        attach_tooltip(
            self.flip_details, "Détails des coûts géométriques, temporels et combinés pour la frame sélectionnée."
        )

        self.reference_name_var.trace_add("write", lambda *_args: self.on_reference_changed())
        self.reference_box.bind("<<ComboboxSelected>>", lambda _event: self.on_reference_changed())
        self.flip_method_var.trace_add("write", lambda *_args: self.refresh_flip_frame_list())
        self.flip_camera_var.trace_add("write", lambda *_args: self.refresh_flip_frame_list())
        self.flip_frame_list.bind("<<ListboxSelect>>", lambda _event: self.render_flip_preview())
        for widget in (self.flip_frame_list, self.flip_canvas_widget, self.flip_method_box, self.flip_camera_box):
            widget.bind("<KeyPress-f>", self.toggle_flip_current_frame)
            widget.bind("<Enter>", lambda _event, w=widget: w.focus_set())
        self.state.keypoints_var.trace_add("write", lambda *_args: self.sync_dataset_dir())
        self.state.output_root_var.trace_add("write", lambda *_args: self.sync_dataset_dir())
        self.state.pose_data_mode_var.trace_add("write", lambda *_args: self.load_resources())
        self.state.calibration_correction_var.trace_add("write", lambda *_args: self.load_resources())
        self.state.register_reconstruction_listener(self.refresh_reference_choices)
        self.state.selected_camera_names_var.trace_add("write", lambda *_args: self.update_camera_filter_status())
        self.sync_dataset_dir()

    def toggle_flip_current_frame(self, _event=None) -> str:
        self.flip_applied_var.set(not self.flip_applied_var.get())
        self.render_flip_preview()
        return "break"

    def sync_dataset_dir(self) -> None:
        self.dataset_dir.var.set(display_path(current_dataset_dir(self.state)))
        self.refresh_reference_choices()
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

    def refresh_reference_choices(self) -> None:
        dataset_dir = current_dataset_dir(self.state)
        catalog = discover_reconstruction_catalog(
            dataset_dir, optional_root_relative_path(self.state.pose2sim_trc_var.get())
        )
        names = [str(row.get("name")) for row in catalog]
        self.reference_box.configure(values=[""] + names)
        current = self.reference_name_var.get()
        if current in names:
            return
        for preferred in ("triangulation_exhaustive", "triangulation_greedy", "pose2sim", "ekf_3d"):
            if preferred in names:
                self.reference_name_var.set(preferred)
                return
        self.reference_name_var.set("")

    def on_reference_changed(self) -> None:
        if self.pose_data is None:
            return
        self.refresh_metrics()
        self.refresh_flip_frame_list()

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
        for method in ("epipolar", "epipolar_fast", "triangulation"):
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
                        if method in {"epipolar", "epipolar_fast"}
                        else DEFAULT_REPROJECTION_THRESHOLD_PX
                    ),
                    temporal_weight=float(self.state.flip_temporal_weight_var.get()),
                    temporal_tau_px=float(self.state.flip_temporal_tau_px_var.get()),
                )
            )
            self.flip_masks[method] = suspect_mask
            self.flip_diagnostics[method] = diagnostics
            self.flip_detail_arrays[method] = load_flip_detail_arrays(cache_path)

    def _reference_payload(self) -> dict[str, np.ndarray]:
        reference_name = self.reference_name_var.get().strip()
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
            label = f"{'flip' if suspect_mask[cam_idx, local_idx] else 'candidate'} | frame {frame_number} | {self._fmt_float(nominal)} -> {self._fmt_float(swapped)}"
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

    def _reference_projection(self, camera_name: str, frame_local_idx: int) -> tuple[np.ndarray | None, str, str]:
        reference_name = self.reference_name_var.get().strip()
        if not reference_name:
            return None, "none", "#444444"
        recon_dir = reconstruction_dir_by_name(current_dataset_dir(self.state), reference_name)
        if recon_dir is None:
            return None, reference_name, reconstruction_color(reference_name)
        payload = load_bundle_payload(recon_dir)
        points_3d = np.asarray(payload.get("points_3d"), dtype=float) if "points_3d" in payload else None
        if points_3d is None or points_3d.ndim != 3 or frame_local_idx >= points_3d.shape[0]:
            return None, reconstruction_label(reference_name), reconstruction_color(reference_name)
        projected = np.full((points_3d.shape[1], 2), np.nan, dtype=float)
        calibration = self.calibrations[camera_name]
        for kp_idx, point_3d in enumerate(points_3d[frame_local_idx]):
            if np.all(np.isfinite(point_3d)):
                projected[kp_idx] = calibration.project_point(point_3d)
        return projected, reconstruction_label(reference_name), reconstruction_color(reference_name)

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
        if finite.size:
            xmin, ymin = np.min(finite, axis=0)
            xmax, ymax = np.max(finite, axis=0)
            margin_x = max(20.0, 0.15 * float(xmax - xmin))
            margin_y = max(20.0, 0.15 * float(ymax - ymin))
            x_limits = (max(0.0, float(xmin - margin_x)), min(float(width), float(xmax + margin_x)))
            y_limits = (min(float(height), float(ymax + margin_y)), max(0.0, float(ymin - margin_y)))
        else:
            x_limits = (0.0, float(width))
            y_limits = (float(height), 0.0)

        ax = self.flip_figure.subplots(1, 1)
        draw_skeleton_2d(
            ax,
            display_raw_points,
            "#000000",
            "Raw 2D",
            marker_size=28.0,
            marker_fill=False,
            marker_edge_width=1.9,
            line_alpha=0.55,
            line_style=(0, (2.0, 2.2)),
            line_width_scale=0.6,
        )
        if projected_points is not None:
            draw_skeleton_2d(ax, projected_points, projected_color, projected_label, marker_size=20.0)
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Raw {'(swapped)' if self.flip_applied_var.get() else ''} + reprojection")
        ax.grid(alpha=0.18)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            uniq = {}
            for handle, label in zip(handles, labels):
                uniq[label] = handle
            ax.legend(list(uniq.values()), list(uniq.keys()), loc="best", fontsize=8)
        detail_arrays = self.flip_detail_arrays.get(method, {})
        suspect = bool(self.flip_masks.get(method, np.zeros((0, 0), dtype=bool))[cam_idx, frame_local_idx])
        frame_number = int(self.base_pose_data.frames[frame_local_idx])
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
                    f"suspect={'yes' if suspect else 'no'}",
                    f"candidate={'yes' if self._flip_flag(detail_arrays, 'candidate_mask', cam_idx, frame_local_idx) else 'no'}",
                    f"temporal_support={'yes' if self._flip_flag(detail_arrays, 'temporal_support_mask', cam_idx, frame_local_idx) else 'no'}",
                    f"nominal geometric={self._fmt_float(self._flip_cost(detail_arrays, 'nominal_geometric_costs', cam_idx, frame_local_idx))}",
                    f"swapped geometric={self._fmt_float(self._flip_cost(detail_arrays, 'swapped_geometric_costs', cam_idx, frame_local_idx))}",
                    f"nominal temporal={self._fmt_float(self._flip_cost(detail_arrays, 'nominal_temporal_costs', cam_idx, frame_local_idx))}",
                    f"swapped temporal={self._fmt_float(self._flip_cost(detail_arrays, 'swapped_temporal_costs', cam_idx, frame_local_idx))}",
                    f"nominal combined={self._fmt_float(self._flip_cost(detail_arrays, 'nominal_combined_costs', cam_idx, frame_local_idx))}",
                    f"swapped combined={self._fmt_float(self._flip_cost(detail_arrays, 'swapped_combined_costs', cam_idx, frame_local_idx))}",
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
        self.output_dir = LabeledEntry(controls, "Dataset", display_path(current_dataset_dir(state)), readonly=True)
        self.output_dir.pack(fill=tk.X, padx=8, pady=4)
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
        self.output_dir.var.set(display_path(current_dataset_dir(self.state)))
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
            self.bundle = get_cached_preview_bundle(
                self.state, ROOT / self.output_dir.get(), None, None, align_root=False
            )
            self.current_reconstruction_name = self._selected_reconstruction()
            if self.current_reconstruction_name is None:
                self.analysis = None
                self.contacts = []
                self.render_summary()
                self.refresh_plot()
                return
            root_q, _full_q, _q_names = preview_root_series_for_reconstruction(
                bundle=self.bundle,
                name=self.current_reconstruction_name,
                initial_rotation_correction=bool(self.state.initial_rotation_correction_var.get()),
            )
            if root_q is None:
                raise ValueError(f"Aucune cinématique racine disponible pour {self.current_reconstruction_name}.")
            fps = float(self.state.fps_var.get())
            self.analysis = analyze_dd_session(
                np.asarray(root_q, dtype=float),
                fps,
                height_values=np.asarray(root_q[:, 2], dtype=float),
                angle_mode="euler",
                analysis_start_frame=ANALYSIS_START_FRAME,
                require_complete_jumps=True,
            )
            recon_3d = self.bundle.get("recon_3d", {}) if isinstance(self.bundle, dict) else {}
            if self.current_reconstruction_name in recon_3d:
                contact_series = np.asarray(recon_3d[self.current_reconstruction_name], dtype=float)
            else:
                contact_series = np.asarray(root_q[:, :2], dtype=float)
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
        self.output_dir = LabeledEntry(controls, "Dataset", display_path(current_dataset_dir(state)), readonly=True)
        self.output_dir.pack(fill=tk.X, padx=8, pady=4)
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
        attach_tooltip(angle_mode_label, "Choisit la methode utilisee pour calculer salto, vrille et tilt.")
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
        self.comparison_tree.column("codes", width=230, anchor="w")
        self.comparison_tree.tag_configure("ok", foreground="#1f7a1f")
        self.comparison_tree.tag_configure("partial", foreground="#b36b00")
        self.comparison_tree.tag_configure("bad", foreground="#b22222")
        self.comparison_tree.tag_configure("neutral", foreground="#666666")
        self.comparison_tree.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.comparison_tree.bind("<<TreeviewSelect>>", self._on_comparison_selected)
        attach_tooltip(
            self.comparison_tree,
            "Compares each reconstruction against the DD reference JSON. Green means full match, orange partial match, red mismatch.",
        )

        jumps_box = ttk.LabelFrame(left, text="Sauts détectés")
        jumps_box.pack(fill=tk.BOTH, expand=False, pady=(0, 8))
        self.jump_list = tk.Listbox(jumps_box, exportselection=False, height=8)
        self.jump_list.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        summary_box = ttk.LabelFrame(left, text="Résumé DD")
        summary_box.pack(fill=tk.BOTH, expand=True)
        self.summary = ScrolledText(summary_box, height=18, wrap=tk.WORD)
        self.summary.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

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
        self.output_dir.var.set(display_path(current_dataset_dir(self.state)))
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
            gui_debug(f"DD refresh_available_reconstructions start dataset={ROOT / self.output_dir.get()}")
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
            gui_debug(f"DD refresh_analysis start dataset={ROOT / self.output_dir.get()}")
            self.bundle = get_cached_preview_bundle(
                self.state, ROOT / self.output_dir.get(), None, None, align_root=False
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
                    self.analysis_by_name[name] = analyze_dd_session(
                        np.asarray(name_root_q, dtype=float),
                        fps,
                        height_values=self._height_series(name_root_q, name_full_q, name_q_name_list),
                        height_threshold=None if not height_threshold_abs else float(height_threshold_abs),
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
                detected_codes = " | ".join(comparison.detected_codes) if comparison.detected_codes else "-"
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

        color = reconstruction_color(self.current_reconstruction_name)
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
                np.rad2deg(selected_jump.tilt_curve_rad),
                color="#55a868",
                linewidth=1.8,
                label="tilt",
            )
            axes[2].set_title(f"S{jump_idx + 1} tilt ({selected_jump.angle_mode})")
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


class LauncherApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VitPose / EKF launcher")
        self.geometry("1450x950")

        state = SharedAppState(
            calib_var=tk.StringVar(value="inputs/calibration/Calib.toml"),
            keypoints_var=tk.StringVar(value="inputs/keypoints/1_partie_0429_keypoints.json"),
            pose2sim_trc_var=tk.StringVar(value="inputs/trc/1_partie_0429.trc"),
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
            output_root_var=tk.StringVar(value="outputs"),
            profiles_config_var=tk.StringVar(value="reconstruction_profiles.json"),
        )
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

        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True)

        self.shared_reconstruction_panel = SharedReconstructionPanel(container, state)
        self.shared_reconstruction_panel.pack(fill=tk.X, padx=10, pady=(10, 0))
        self.state.shared_reconstruction_panel = self.shared_reconstruction_panel

        self.notebook = ttk.Notebook(container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.notebook.add(DataExplorer2DTab(self.notebook, state), text="2D explorer")
        self.notebook.add(CameraToolsTab(self.notebook, state), text="Caméras")
        self.notebook.add(ModelTab(self.notebook, state), text="Modèle")
        self.notebook.add(ProfilesTab(self.notebook, state), text="Profiles")
        self.notebook.add(ReconstructionsTab(self.notebook, state), text="Reconstructions")
        self.notebook.add(DualAnimationTab(self.notebook, state), text="3D animation")
        self.notebook.add(MultiViewTab(self.notebook, state), text="2D multiview")
        self.notebook.add(ExecutionTab(self.notebook, state), text="Execution")
        self.notebook.add(DDTab(self.notebook, state), text="DD")
        self.notebook.add(TrampolineTab(self.notebook, state), text="Toile")
        self.notebook.add(RootKinematicsTab(self.notebook, state), text="Racine")
        self.notebook.add(JointKinematicsTab(self.notebook, state), text="Autres DoF")
        self.notebook.add(Analysis3DTab(self.notebook, state), text="3D analysis")
        self.notebook.add(ObservabilityTab(self.notebook, state), text="Observabilité")
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self.after_idle(self._refresh_active_reconstruction_panel)

    def _refresh_active_reconstruction_panel(self) -> None:
        """Bind the top reconstruction selector to the active tab when supported."""

        current_tab_id = self.notebook.select()
        current_tab = self.nametowidget(current_tab_id) if current_tab_id else None
        self.state.active_reconstruction_consumer = current_tab
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
