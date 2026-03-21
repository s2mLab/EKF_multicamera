#!/usr/bin/env python3
"""Animation 2D multi-vues des marqueurs bruts et reprojectes.

Le script affiche, pour toutes les cameras dans une meme fenetre:
- les detections 2D brutes,
- les marqueurs reprojetes a partir de la triangulation locale,
- les marqueurs reprojetes a partir de Pose2Sim,
- les marqueurs reprojetes a partir de l'EKF 3D,
- les marqueurs reprojetes a partir de l'EKF 2D.

Chaque vue est affichee dans un subplot distinct et l'ensemble est exporte en GIF.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOCAL_MPLCONFIG = Path("/Users/mickaelbegon/Documents/Playground/.cache/matplotlib")
LOCAL_MPLCONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_MPLCONFIG))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.animation import FuncAnimation, PillowWriter

from animation.animate_dual_stick_comparison import (
    KP_INDEX,
    LEFT_KEYPOINTS,
    LOWER_LIMB_EDGES,
    RIGHT_KEYPOINTS,
    SKELETON_EDGES,
    biorbd_markers_from_q,
    load_dataset_reconstructions,
    parse_trc,
    resample_points,
)
from reconstruction_dataset import dataset_source_paths, reconstruction_color, reconstruction_label, resolve_requested_names
from vitpose_ekf_pipeline import COCO17, fundamental_matrix, load_calibrations, load_pose_data, sampson_error_pixels


DEFAULT_CAMERA_FPS = 120.0
DEFAULT_CALIB = Path("inputs/Calib.toml")
DEFAULT_KEYPOINTS = Path("inputs/1_partie_0429_keypoints.json")
LEFT_RIGHT_SWAP_PAIRS = [
    ("left_eye", "right_eye"),
    ("left_ear", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animation 2D multi-vues des donnees brutes et des reprojections 3D.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Dossier dataset contenant les bundles de reconstruction.")
    parser.add_argument("--calib", type=Path, default=None, help="Fichier de calibration TOML")
    parser.add_argument("--keypoints", type=Path, default=None, help="JSON des detections 2D brutes")
    parser.add_argument(
        "--triangulation",
        type=Path,
        default=Path("outputs") / "vitpose_full" / "triangulation_pose2sim_like.npz",
        help="NPZ triangulation locale",
    )
    parser.add_argument("--pose2sim-trc", type=Path, default=None)
    parser.add_argument("--ekf-states", type=Path, default=Path("outputs") / "vitpose_full" / "ekf_states.npz")
    parser.add_argument("--kalman-comparison", type=Path, default=Path("outputs") / "vitpose_full" / "kalman_comparison.npz")
    parser.add_argument("--biomod", type=Path, default=Path("outputs") / "vitpose_full" / "vitpose_chain.bioMod")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "vitpose_full" / "multiview_2d_comparison.gif",
        help="GIF de sortie",
    )
    parser.add_argument("--data-fps", type=float, default=DEFAULT_CAMERA_FPS, help="Frequence des donnees source")
    parser.add_argument("--gif-fps", type=float, default=10.0, help="Frequence d'affichage du GIF")
    parser.add_argument("--stride", type=int, default=5, help="Sous-echantillonnage temporel pour le GIF")
    parser.add_argument("--marker-size", type=float, default=18.0, help="Taille des points")
    parser.add_argument("--max-frames", type=int, default=None, help="Limite optionnelle de frames")
    parser.add_argument("--workers", type=int, default=1, help="Nombre de workers pour rendre les frames en parallele.")
    parser.add_argument("--show", nargs="+", default=None, help="Liste des couches a afficher dans l'animation 2D multi-vues.")
    parser.add_argument(
        "--crop-mode",
        choices=("full", "pose"),
        default="full",
        help="`full` utilise toute l'image, `pose` recadre chaque frame autour des keypoints 2D valides.",
    )
    parser.add_argument("--crop-margin", type=float, default=0.10, help="Marge relative appliquee au cadrage `pose`.")
    return parser.parse_args()


def load_triangulation(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    points = np.asarray(data["points_3d"], dtype=float)
    frames = np.asarray(data["frames"], dtype=int) if "frames" in data else np.arange(points.shape[0], dtype=int)
    return points, frames


def load_flight_parameters(triangulation_path: Path) -> tuple[float, int]:
    """Lit les parametres air/toile depuis `summary.json` si present."""
    summary_path = triangulation_path.parent / "summary.json"
    if not summary_path.exists():
        return 0.0, 1
    with summary_path.open("r") as f:
        summary = json.load(f)
    return float(summary.get("flight_height_threshold_m", 0.0)), int(summary.get("flight_min_consecutive_frames", 1))


def compute_airborne_mask(points_3d: np.ndarray, threshold_m: float, min_consecutive_frames: int) -> np.ndarray:
    """Reconstruit le masque air/toile a partir des marqueurs triangules."""
    above = np.all(points_3d[:, :, 2] > threshold_m, axis=1)
    above &= np.all(np.isfinite(points_3d[:, :, 2]), axis=1)
    mask = np.zeros(points_3d.shape[0], dtype=bool)
    consec = 0
    for i, flag in enumerate(above):
        consec = consec + 1 if flag else 0
        if consec >= max(1, min_consecutive_frames):
            mask[i] = True
    return mask


def load_q_reconstructions(
    ekf_states_path: Path, kalman_comparison_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    ekf = np.load(ekf_states_path, allow_pickle=True)
    comparison = np.load(kalman_comparison_path, allow_pickle=True)
    q_ekf_3d = comparison["q_ekf_3d"] if "q_ekf_3d" in comparison else comparison["q_biorbd_kalman"]
    q_ekf_2d_acc = np.asarray(ekf["q_ekf_2d_acc"], dtype=float) if "q_ekf_2d_acc" in ekf else np.asarray(ekf["q"], dtype=float)
    q_ekf_2d_dyn = np.asarray(ekf["q_ekf_2d_dyn"], dtype=float) if "q_ekf_2d_dyn" in ekf else None
    q_ekf_2d_flip_acc = np.asarray(ekf["q_ekf_2d_flip_acc"], dtype=float) if "q_ekf_2d_flip_acc" in ekf else None
    q_ekf_2d_flip_dyn = np.asarray(ekf["q_ekf_2d_flip_dyn"], dtype=float) if "q_ekf_2d_flip_dyn" in ekf else None
    return q_ekf_2d_acc, np.asarray(q_ekf_3d, dtype=float), q_ekf_2d_dyn, q_ekf_2d_flip_acc, q_ekf_2d_flip_dyn


def project_points(points_3d: np.ndarray, calibrations: dict, camera_names: list[str]) -> np.ndarray:
    """Projette un tenseur 3D `(n_frames, 17, 3)` dans toutes les vues."""
    n_frames, n_points, _ = points_3d.shape
    projections = np.full((len(camera_names), n_frames, n_points, 2), np.nan, dtype=float)
    for cam_idx, cam_name in enumerate(camera_names):
        calibration = calibrations[cam_name]
        for frame_idx in range(n_frames):
            for marker_idx in range(n_points):
                point = points_3d[frame_idx, marker_idx]
                if np.all(np.isfinite(point)):
                    projections[cam_idx, frame_idx, marker_idx] = calibration.project_point(point)
    return projections


def subsample_all(arrays: list[np.ndarray], stride: int, max_frames: int | None, frame_axis: int = 1) -> list[np.ndarray]:
    out = []
    for array in arrays:
        if array is None:
            out.append(None)
            continue
        current_frame_axis = min(frame_axis, array.ndim - 1)
        selector = [slice(None)] * array.ndim
        selector[current_frame_axis] = slice(None, None, max(stride, 1))
        subset = array[tuple(selector)]
        if max_frames is not None:
            selector = [slice(None)] * subset.ndim
            selector[current_frame_axis] = slice(0, max_frames)
            subset = subset[tuple(selector)]
        out.append(subset)
    return out


def subsample_layer_dict(layers: dict[str, np.ndarray], stride: int, max_frames: int | None, frame_axis: int = 1) -> dict[str, np.ndarray]:
    keys = list(layers.keys())
    values = [layers[key] for key in keys]
    subsampled = subsample_all(values, stride=stride, max_frames=max_frames, frame_axis=frame_axis)
    return {key: value for key, value in zip(keys, subsampled)}


def camera_layout(n_cameras: int) -> tuple[int, int]:
    ncols = math.ceil(math.sqrt(n_cameras))
    nrows = math.ceil(n_cameras / ncols)
    return nrows, ncols


def layer_style(name: str, marker_size: float) -> dict[str, object]:
    if name == "raw":
        return {
            "color": reconstruction_color(name),
            "label": reconstruction_label(name),
            "linestyle": "-",
            "linewidth": 1.0,
            "alpha": 0.7,
            "marker_size": marker_size,
        }
    return {
        "color": reconstruction_color(name),
        "label": reconstruction_label(name),
        "linestyle": "-",
        "linewidth": 1.3,
        "alpha": 0.9 if name == "pose2sim" else 0.85,
        "marker_size": marker_size,
    }


def grouped_points_2d(points: np.ndarray) -> dict[str, np.ndarray]:
    groups = {
        "left": [KP_INDEX[name] for name in COCO17 if name in LEFT_KEYPOINTS],
        "right": [KP_INDEX[name] for name in COCO17 if name in RIGHT_KEYPOINTS],
        "center": [KP_INDEX[name] for name in COCO17 if name not in LEFT_KEYPOINTS and name not in RIGHT_KEYPOINTS],
    }
    out: dict[str, np.ndarray] = {}
    for side, indices in groups.items():
        side_points = points[indices]
        valid = np.all(np.isfinite(side_points), axis=1)
        out[side] = side_points[valid]
    return out


def edge_linewidth(name_a: str, name_b: str, base: float) -> float:
    return base * 3.0 if (name_a, name_b) in LOWER_LIMB_EDGES else base


def scatter_markers(name: str) -> dict[str, str]:
    center = "x" if name == "raw" else "o"
    return {"center": center, "left": "^", "right": "s"}


def compute_pose_crop_limits(raw_2d: np.ndarray, calibrations: dict, camera_names: list[str], margin: float) -> dict[str, np.ndarray]:
    """Calcule des bornes de crop par frame et par camera a partir des keypoints 2D."""
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
    """Applique un cadrage 2D fixe et empeche l'autoscale de l'annuler."""
    if crop_mode == "pose":
        x0, x1, y1, y0 = crop_limits[cam_name][frame_idx]
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
    else:
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
    ax.set_autoscale_on(False)


def swap_left_right_keypoints(points_2d: np.ndarray) -> np.ndarray:
    """Construit l'hypothese alternative face/dos via un swap gauche/droite global."""
    swapped = np.array(points_2d, copy=True)
    for left_name, right_name in LEFT_RIGHT_SWAP_PAIRS:
        left_idx = KP_INDEX[left_name]
        right_idx = KP_INDEX[right_name]
        swapped[left_idx], swapped[right_idx] = np.array(points_2d[right_idx], copy=True), np.array(points_2d[left_idx], copy=True)
    return swapped


def compute_camera_epipolar_cost(
    camera_idx: int,
    candidate_points: np.ndarray,
    raw_2d_frame: np.ndarray,
    raw_scores_frame: np.ndarray,
    fundamental_matrices: dict[tuple[int, int], np.ndarray],
) -> float:
    """Mesure la coherence epipolaire d'une camera candidate avec les autres vues."""
    errors = []
    valid_candidate = np.all(np.isfinite(candidate_points), axis=1) & (raw_scores_frame[camera_idx] > 0)
    for kp_idx in range(candidate_points.shape[0]):
        if not valid_candidate[kp_idx]:
            continue
        point_i = candidate_points[kp_idx]
        for other_idx in range(raw_2d_frame.shape[0]):
            if other_idx == camera_idx:
                continue
            if raw_scores_frame[other_idx, kp_idx] <= 0:
                continue
            point_j = raw_2d_frame[other_idx, kp_idx]
            if not np.all(np.isfinite(point_j)):
                continue
            err = sampson_error_pixels(point_i, point_j, fundamental_matrices[(camera_idx, other_idx)])
            if np.isfinite(err):
                errors.append(err)
    if not errors:
        return np.nan
    return float(np.median(np.asarray(errors, dtype=float)))


def detect_face_back_confusions(
    raw_2d: np.ndarray,
    raw_scores: np.ndarray,
    calibrations: dict,
    camera_names: list[str],
    improvement_ratio: float = 0.7,
    min_gain_px: float = 3.0,
) -> np.ndarray:
    """Diagnostique une possible confusion face/dos via un swap gauche/droite.

    Une camera est marquee comme suspecte si, pour une frame donnee, le cout
    epipolaire median chute nettement quand on echange globalement les points
    gauche/droite de cette vue.
    """
    n_cams, n_frames, _, _ = raw_2d.shape
    fundamental_matrices = {
        (i_cam, j_cam): fundamental_matrix(calibrations[camera_names[i_cam]], calibrations[camera_names[j_cam]])
        for i_cam in range(n_cams)
        for j_cam in range(n_cams)
        if i_cam != j_cam
    }
    suspect_mask = np.zeros((n_cams, n_frames), dtype=bool)

    for frame_idx in range(n_frames):
        raw_points_frame = raw_2d[:, frame_idx]
        raw_scores_frame = raw_scores[:, frame_idx]
        for cam_idx in range(n_cams):
            nominal_cost = compute_camera_epipolar_cost(cam_idx, raw_points_frame[cam_idx], raw_points_frame, raw_scores_frame, fundamental_matrices)
            swapped_cost = compute_camera_epipolar_cost(cam_idx, swap_left_right_keypoints(raw_points_frame[cam_idx]), raw_points_frame, raw_scores_frame, fundamental_matrices)
            if not (np.isfinite(nominal_cost) and np.isfinite(swapped_cost)):
                continue
            if nominal_cost > 0 and swapped_cost < improvement_ratio * nominal_cost and (nominal_cost - swapped_cost) >= min_gain_px:
                suspect_mask[cam_idx, frame_idx] = True

    return suspect_mask


def create_animation(
    raw_2d: np.ndarray,
    layer_2d: dict[str, np.ndarray],
    calibrations: dict,
    camera_names: list[str],
    output_path: Path,
    gif_fps: float,
    marker_size: float,
    crop_mode: str,
    crop_margin: float,
    show: tuple[str, ...],
    airborne_mask: np.ndarray,
    confusion_mask: np.ndarray | None,
) -> None:
    """Genere le GIF multi-vues."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_cameras = len(camera_names)
    n_frames = raw_2d.shape[1]
    nrows, ncols = camera_layout(n_cameras)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows))
    axes = np.atleast_1d(axes).ravel()

    display_names = ([] if "raw" not in show else ["raw"]) + [name for name in show if name != "raw" and name in layer_2d]
    styles = {name: layer_style(name, marker_size) for name in display_names}

    artists: dict[str, list[dict[str, object]]] = {key: [] for key in display_names}
    line_artists: dict[str, list[list]] = {key: [] for key in display_names}
    title = fig.suptitle("", fontsize=14)
    crop_limits = compute_pose_crop_limits(raw_2d, calibrations, camera_names, crop_margin) if crop_mode == "pose" else {}

    for ax_idx, ax in enumerate(axes):
        if ax_idx >= n_cameras:
            ax.axis("off")
            continue
        cam_name = camera_names[ax_idx]
        width, height = calibrations[cam_name].image_size
        apply_2d_axis_limits(
            ax,
            crop_mode=crop_mode,
            crop_limits=crop_limits,
            cam_name=cam_name,
            frame_idx=0,
            width=width,
            height=height,
        )
        ax.set_aspect("equal")
        ax.set_title(cam_name)
        ax.grid(alpha=0.15)
        for key in display_names:
            style = styles[key]
            markers = scatter_markers(key)
            artist = {
                "center": ax.scatter([], [], s=marker_size, c=style["color"], marker=markers["center"], label=style["label"], alpha=style["alpha"]),
                "left": ax.scatter([], [], s=marker_size * 1.3, c=style["color"], marker=markers["left"], alpha=style["alpha"]),
                "right": ax.scatter([], [], s=marker_size * 1.3, c=style["color"], marker=markers["right"], alpha=style["alpha"]),
            }
            artists[key].append(artist)
            lines = []
            for name_a, name_b in SKELETON_EDGES:
                (line,) = ax.plot(
                    [],
                    [],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=edge_linewidth(name_a, name_b, style["linewidth"]),
                    alpha=style["alpha"],
                )
                lines.append(line)
            line_artists[key].append(lines)

    legend_handles = [artists[key][0]["center"] for key in display_names if artists[key] and artists[key][0] is not None]
    fig.legend(
        legend_handles,
        [styles[key]["label"] for key in display_names if artists[key] and artists[key][0] is not None],
        loc="lower right",
        bbox_to_anchor=(0.995, 0.02),
        ncol=1,
        frameon=True,
    )

    def set_offsets(scatter_group, points: np.ndarray) -> None:
        grouped = grouped_points_2d(points)
        for side, scatter in scatter_group.items():
            scatter.set_offsets(grouped[side] if grouped[side].size else np.empty((0, 2)))

    def set_lines(lines: list, points: np.ndarray) -> None:
        for line, (name_a, name_b) in zip(lines, SKELETON_EDGES):
            point_a = points[KP_INDEX[name_a]]
            point_b = points[KP_INDEX[name_b]]
            if np.all(np.isfinite(point_a)) and np.all(np.isfinite(point_b)):
                line.set_data([point_a[0], point_b[0]], [point_a[1], point_b[1]])
            else:
                line.set_data([], [])

    def update(frame_idx: int):
        suspect_labels = []
        for cam_idx in range(n_cameras):
            label = camera_names[cam_idx]
            if confusion_mask is not None and confusion_mask[cam_idx, frame_idx]:
                axes[cam_idx].set_title(f"{label} | face/dos ?", color="#c44e52")
                suspect_labels.append(label)
            else:
                axes[cam_idx].set_title(label, color="black")
            width, height = calibrations[label].image_size
            apply_2d_axis_limits(
                axes[cam_idx],
                crop_mode=crop_mode,
                crop_limits=crop_limits,
                cam_name=label,
                frame_idx=frame_idx,
                width=width,
                height=height,
            )
            if "raw" in artists:
                set_offsets(artists["raw"][cam_idx], raw_2d[cam_idx, frame_idx])
                set_lines(line_artists["raw"][cam_idx], raw_2d[cam_idx, frame_idx])
            for name in display_names:
                if name == "raw":
                    continue
                set_offsets(artists[name][cam_idx], layer_2d[name][cam_idx, frame_idx])
                set_lines(line_artists[name][cam_idx], layer_2d[name][cam_idx, frame_idx])
        phase = "AIR" if airborne_mask[frame_idx] else "TOILE"
        warning = "" if not suspect_labels else f" | swap suspect: {', '.join(suspect_labels)}"
        title.set_text(f"Frame {frame_idx} | {phase}{warning}")
        return [scatter for groups in artists.values() for group in groups for scatter in group.values() if scatter is not None] + [line for groups in line_artists.values() for group in groups for line in group] + [title]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / gif_fps, blit=False)
    anim.save(output_path, writer=PillowWriter(fps=gif_fps))
    plt.close(fig)


def draw_points_and_lines(ax, points: np.ndarray, style: dict) -> None:
    """Dessine une couche 2D complete sur un axe."""
    grouped = grouped_points_2d(points)
    markers = {"center": "o", "left": "^", "right": "s"}
    if grouped["center"].size:
        ax.scatter(grouped["center"][:, 0], grouped["center"][:, 1], s=style["marker_size"], c=style["color"], marker=markers["center"], alpha=style["alpha"])
    if grouped["left"].size:
        ax.scatter(grouped["left"][:, 0], grouped["left"][:, 1], s=style["marker_size"] * 1.3, c=style["color"], marker=markers["left"], alpha=style["alpha"])
    if grouped["right"].size:
        ax.scatter(grouped["right"][:, 0], grouped["right"][:, 1], s=style["marker_size"] * 1.3, c=style["color"], marker=markers["right"], alpha=style["alpha"])
    for name_a, name_b in SKELETON_EDGES:
        point_a = points[KP_INDEX[name_a]]
        point_b = points[KP_INDEX[name_b]]
        if np.all(np.isfinite(point_a)) and np.all(np.isfinite(point_b)):
            ax.plot(
                [point_a[0], point_b[0]],
                [point_a[1], point_b[1]],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=edge_linewidth(name_a, name_b, style["linewidth"]),
                alpha=style["alpha"],
            )


def render_frame(
    frame_idx: int,
    raw_2d: np.ndarray,
    layer_2d: dict[str, np.ndarray],
    calibrations: dict,
    camera_names: list[str],
    crop_mode: str,
    crop_margin: float,
    marker_size: float,
    show: tuple[str, ...],
    airborne_mask: np.ndarray,
    confusion_mask: np.ndarray | None,
    output_path: Path,
) -> Path:
    """Rend une frame PNG de l'animation 2D."""
    n_cameras = len(camera_names)
    nrows, ncols = camera_layout(n_cameras)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows))
    axes = np.atleast_1d(axes).ravel()
    crop_limits = compute_pose_crop_limits(raw_2d, calibrations, camera_names, crop_margin) if crop_mode == "pose" else {}
    display_names = ([] if "raw" not in show else ["raw"]) + [name for name in show if name != "raw" and name in layer_2d]
    styles = {name: layer_style(name, marker_size) for name in display_names}

    for ax_idx, ax in enumerate(axes):
        if ax_idx >= n_cameras:
            ax.axis("off")
            continue
        cam_name = camera_names[ax_idx]
        width, height = calibrations[cam_name].image_size
        apply_2d_axis_limits(
            ax,
            crop_mode=crop_mode,
            crop_limits=crop_limits,
            cam_name=cam_name,
            frame_idx=frame_idx,
            width=width,
            height=height,
        )
        ax.set_aspect("equal")
        if confusion_mask is not None and confusion_mask[ax_idx, frame_idx]:
            ax.set_title(f"{cam_name} | face/dos ?", color="#c44e52")
        else:
            ax.set_title(cam_name)
        ax.grid(alpha=0.15)

        if "raw" in styles:
            draw_points_and_lines(ax, raw_2d[ax_idx, frame_idx], styles["raw"])
        for name in display_names:
            if name == "raw":
                continue
            draw_points_and_lines(ax, layer_2d[name][ax_idx, frame_idx], styles[name])

    handles = [
        plt.Line2D(
            [],
            [],
            color=styles[key]["color"],
            marker=scatter_markers(key)["center"],
            linestyle=styles[key]["linestyle"],
            linewidth=styles[key]["linewidth"],
            alpha=styles[key]["alpha"],
            label=styles[key]["label"],
        )
        for key in display_names
    ]
    fig.legend(handles, [styles[key]["label"] for key in display_names], loc="lower right", bbox_to_anchor=(0.995, 0.02), ncol=1, frameon=True)
    phase = "AIR" if airborne_mask[frame_idx] else "TOILE"
    suspect_labels = [camera_names[i] for i in range(n_cameras) if confusion_mask is not None and confusion_mask[i, frame_idx]]
    warning = "" if not suspect_labels else f" | swap suspect: {', '.join(suspect_labels)}"
    fig.suptitle(f"Frame {frame_idx} | {phase}{warning}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_animation_parallel(
    raw_2d: np.ndarray,
    layer_2d: dict[str, np.ndarray],
    calibrations: dict,
    camera_names: list[str],
    output_path: Path,
    gif_fps: float,
    marker_size: float,
    crop_mode: str,
    crop_margin: float,
    workers: int,
    show: tuple[str, ...],
    airborne_mask: np.ndarray,
    confusion_mask: np.ndarray | None,
) -> None:
    """Rend les frames en parallele puis assemble le GIF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = raw_2d.shape[1]
    with tempfile.TemporaryDirectory(prefix="multiview2d_", dir=str(output_path.parent)) as tmpdir:
        tmpdir_path = Path(tmpdir)
        frame_paths = [tmpdir_path / f"frame_{frame_idx:05d}.png" for frame_idx in range(n_frames)]

        if workers <= 1:
            for frame_idx in range(n_frames):
                render_frame(
                    frame_idx,
                    raw_2d,
                    layer_2d,
                    calibrations,
                    camera_names,
                    crop_mode,
                    crop_margin,
                    marker_size,
                    show,
                    airborne_mask,
                    confusion_mask,
                    frame_paths[frame_idx],
                )
        else:
            with cf.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        render_frame,
                        frame_idx,
                        raw_2d,
                        layer_2d,
                        calibrations,
                        camera_names,
                        crop_mode,
                        crop_margin,
                        marker_size,
                        show,
                        airborne_mask,
                        confusion_mask,
                        frame_paths[frame_idx],
                    )
                    for frame_idx in range(n_frames)
                ]
                for future in futures:
                    future.result()

        images = [Image.open(frame_path) for frame_path in frame_paths]
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=int(round(1000 / gif_fps)),
            loop=0,
        )
        for image in images:
            image.close()


def main() -> None:
    args = parse_args()
    if args.dataset_dir is not None:
        sources = dataset_source_paths(args.dataset_dir, calib=args.calib, keypoints=args.keypoints, pose2sim_trc=args.pose2sim_trc)
        calib_path = Path(sources["calib"])
        keypoints_path = Path(sources["keypoints"])
        calibrations = load_calibrations(calib_path)
        pose_data = load_pose_data(keypoints_path, calibrations, max_frames=None)
        camera_names = pose_data.camera_names
        recon_3d, frames, local_time, airborne_mask = load_dataset_reconstructions(args.dataset_dir)
    else:
        calib_path = args.calib if args.calib is not None else DEFAULT_CALIB
        keypoints_path = args.keypoints if args.keypoints is not None else DEFAULT_KEYPOINTS
        pose2sim_trc = args.pose2sim_trc if args.pose2sim_trc is not None else Path("inputs/1_partie_0429.trc")
        calibrations = load_calibrations(calib_path)
        pose_data = load_pose_data(keypoints_path, calibrations, max_frames=None)
        camera_names = pose_data.camera_names

        triangulation_3d, frames = load_triangulation(args.triangulation)
        flight_threshold, flight_min_consecutive = load_flight_parameters(args.triangulation)
        airborne_mask = compute_airborne_mask(triangulation_3d, flight_threshold, flight_min_consecutive)
        local_time = frames / args.data_fps
        recon_3d = {"triangulation": triangulation_3d}

        ekf_2d_acc_q, ekf_3d_q, ekf_2d_dyn_q, ekf_2d_flip_acc_q, ekf_2d_flip_dyn_q = load_q_reconstructions(
            args.ekf_states, args.kalman_comparison
        )
        recon_3d["ekf_2d_acc"] = biorbd_markers_from_q(args.biomod, ekf_2d_acc_q[: triangulation_3d.shape[0]])
        recon_3d["ekf_3d"] = biorbd_markers_from_q(args.biomod, ekf_3d_q[: triangulation_3d.shape[0]])
        if ekf_2d_dyn_q is not None:
            recon_3d["ekf_2d_dyn"] = biorbd_markers_from_q(args.biomod, ekf_2d_dyn_q[: triangulation_3d.shape[0]])
        if ekf_2d_flip_acc_q is not None:
            recon_3d["ekf_2d_flip_acc"] = biorbd_markers_from_q(args.biomod, ekf_2d_flip_acc_q[: triangulation_3d.shape[0]])
        if ekf_2d_flip_dyn_q is not None:
            recon_3d["ekf_2d_flip_dyn"] = biorbd_markers_from_q(args.biomod, ekf_2d_flip_dyn_q[: triangulation_3d.shape[0]])

        pose2sim_3d, pose2sim_time, _ = parse_trc(pose2sim_trc)
        recon_3d["pose2sim"] = resample_points(pose2sim_3d, pose2sim_time, local_time)

    frame_to_idx = {int(frame): idx for idx, frame in enumerate(np.asarray(pose_data.frames, dtype=int))}
    if not np.all(np.isin(frames, pose_data.frames)):
        missing = [int(frame) for frame in frames if int(frame) not in frame_to_idx]
        raise ValueError(
            "The reconstruction frames and the 2D detections do not share the same frame ids. "
            f"Missing raw frames for ids: {missing[:10]}"
        )
    selected_raw_idx = np.array([frame_to_idx[int(frame)] for frame in frames], dtype=int)
    raw_2d = np.asarray(pose_data.keypoints[:, selected_raw_idx], dtype=float)
    raw_scores = np.asarray(pose_data.scores[:, selected_raw_idx], dtype=float)

    projected_layers = {name: project_points(points_3d, calibrations, camera_names) for name, points_3d in recon_3d.items()}

    raw_2d, airborne_mask, raw_scores = subsample_all(
        [raw_2d, airborne_mask, raw_scores],
        stride=args.stride,
        max_frames=args.max_frames,
    )
    projected_layers = subsample_layer_dict(projected_layers, stride=args.stride, max_frames=args.max_frames, frame_axis=1)
    confusion_mask = detect_face_back_confusions(raw_2d, raw_scores, calibrations, camera_names)

    include_raw = args.show is None or "raw" in args.show
    requested_reconstructions = None if args.show is None else [name for name in args.show if name != "raw"]
    resolved_show = resolve_requested_names(requested_reconstructions, list(projected_layers.keys()))
    show = (["raw"] if include_raw else []) + resolved_show
    if not show:
        raise ValueError("Aucune couche disponible ne correspond aux options --show demandees.")

    if args.workers <= 1:
        create_animation(
            raw_2d=raw_2d,
            layer_2d=projected_layers,
            calibrations=calibrations,
            camera_names=camera_names,
            output_path=args.output,
            gif_fps=args.gif_fps,
            marker_size=args.marker_size,
            crop_mode=args.crop_mode,
            crop_margin=args.crop_margin,
            show=tuple(show),
            airborne_mask=airborne_mask,
            confusion_mask=confusion_mask,
        )
    else:
        create_animation_parallel(
            raw_2d=raw_2d,
            layer_2d=projected_layers,
            calibrations=calibrations,
            camera_names=camera_names,
            output_path=args.output,
            gif_fps=args.gif_fps,
            marker_size=args.marker_size,
            crop_mode=args.crop_mode,
            crop_margin=args.crop_margin,
            workers=args.workers,
            show=tuple(show),
            airborne_mask=airborne_mask,
            confusion_mask=confusion_mask,
        )
    print(f"Animation 2D multi-vues exportee dans: {args.output}")


if __name__ == "__main__":
    main()
