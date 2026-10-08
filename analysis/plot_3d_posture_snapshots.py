#!/usr/bin/env python3
"""Trace un ensemble de postures 3D representatives avec les cameras.

Le script selectionne quelques frames (5 a 10 en general) en combinant:
- une repartition sur l'axe vertical du mouvement;
- une recherche de diversite posturale.

Les postures sont ensuite affichees dans le repere monde avec les cameras du
fichier de calibration.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOCAL_MPLCONFIG = ROOT / ".cache" / "matplotlib"
LOCAL_MPLCONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_MPLCONFIG))

import matplotlib.pyplot as plt
import numpy as np

from analysis.plot_kinematic_comparison import compute_trunk_dofs_from_triangulation
from animation.animate_dual_stick_comparison import KP_INDEX, SKELETON_EDGES
from vitpose_ekf_pipeline import load_calibrations

DEFAULT_TRIANGULATION = Path("output/vitpose_full/triangulation_pose2sim_like.npz")
DEFAULT_CALIB = Path("inputs/calibration/Calib.toml")
DEFAULT_OUTPUT = Path("output/vitpose_full/posture_snapshots_3d.png")
DEFAULT_FIRST_FRAME_OUTPUT = Path("output/vitpose_full/first_frame_root_coordinate_system.png")
DEFAULT_FPS = 120.0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the 3D posture snapshot figure generator."""

    parser = argparse.ArgumentParser(description="Affiche quelques postures 3D representatives avec les cameras.")
    parser.add_argument("--triangulation", type=Path, default=DEFAULT_TRIANGULATION, help="NPZ de triangulation.")
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB, help="Calibration des cameras.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Figure de sortie.")
    parser.add_argument(
        "--first-frame-output",
        type=Path,
        default=DEFAULT_FIRST_FRAME_OUTPUT,
        help="Figure dediee a la premiere frame avec repere global et repere racine.",
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Frequence pour annoter le temps.")
    parser.add_argument("--n-postures", type=int, default=7, help="Nombre de postures a extraire.")
    parser.add_argument("--camera-scale", type=float, default=0.25, help="Longueur du vecteur de visee des cameras.")
    return parser.parse_args()


def pelvis_center(points_3d: np.ndarray) -> np.ndarray:
    """Return the pelvis center estimated as the midpoint between both hips."""

    left_hip = points_3d[:, KP_INDEX["left_hip"], :]
    right_hip = points_3d[:, KP_INDEX["right_hip"], :]
    return 0.5 * (left_hip + right_hip)


def posture_descriptor(points_frame: np.ndarray) -> np.ndarray | None:
    """Construit un descripteur de posture independent de la translation globale."""
    if not np.all(np.isfinite(points_frame[[KP_INDEX["left_hip"], KP_INDEX["right_hip"]]])):
        return None
    root = 0.5 * (points_frame[KP_INDEX["left_hip"]] + points_frame[KP_INDEX["right_hip"]])
    centered = points_frame - root
    valid = np.all(np.isfinite(centered), axis=1)
    if np.count_nonzero(valid) < 8:
        return None
    scale = np.nanmedian(np.linalg.norm(centered[valid], axis=1))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    descriptor = np.full(points_frame.shape[0] * 3, np.nan, dtype=float)
    descriptor.reshape(points_frame.shape)[valid] = centered[valid] / scale
    return descriptor


def descriptor_distance(desc_a: np.ndarray | None, desc_b: np.ndarray | None) -> float:
    """Return a finite-only distance between two posture descriptors."""

    if desc_a is None or desc_b is None:
        return -np.inf
    valid = np.isfinite(desc_a) & np.isfinite(desc_b)
    if np.count_nonzero(valid) < 12:
        return -np.inf
    return float(np.linalg.norm(desc_a[valid] - desc_b[valid]))


def select_posture_frames(points_3d: np.ndarray, n_postures: int) -> list[int]:
    """Selectionne des frames reparties en hauteur avec diversite posturale."""
    n_postures = max(1, int(n_postures))
    root = pelvis_center(points_3d)
    root_z = root[:, 2]
    valid_frames = np.where(np.isfinite(root_z))[0]
    if valid_frames.size == 0:
        raise ValueError("Aucune frame valide pour la selection des postures.")

    descriptors = [posture_descriptor(points_3d[i]) for i in range(points_3d.shape[0])]
    height_values = root_z[valid_frames]
    quantiles = np.linspace(0.05, 0.95, min(n_postures, valid_frames.size))
    target_heights = np.quantile(height_values, quantiles)
    selected: list[int] = []

    for target_height in target_heights:
        candidates = valid_frames[np.argsort(np.abs(root_z[valid_frames] - target_height))]
        best_idx = None
        best_score = -np.inf
        for candidate in candidates[: min(80, len(candidates))]:
            desc = descriptors[candidate]
            if desc is None:
                continue
            if candidate in selected:
                continue
            novelty = 1.0
            if selected:
                novelty = min(descriptor_distance(desc, descriptors[idx]) for idx in selected)
            height_match = -abs(root_z[candidate] - target_height)
            score = 4.0 * novelty + height_match
            if score > best_score:
                best_score = score
                best_idx = int(candidate)
        if best_idx is not None:
            selected.append(best_idx)

    if not selected:
        selected = [int(valid_frames[np.nanargmax(root_z[valid_frames])])]

    # Complete au besoin avec des poses les plus differentes possible.
    while len(selected) < min(n_postures, valid_frames.size):
        best_idx = None
        best_score = -np.inf
        for candidate in valid_frames:
            if int(candidate) in selected:
                continue
            desc = descriptors[int(candidate)]
            if desc is None:
                continue
            novelty = min(descriptor_distance(desc, descriptors[idx]) for idx in selected)
            if novelty > best_score:
                best_score = novelty
                best_idx = int(candidate)
        if best_idx is None:
            break
        selected.append(best_idx)

    selected = sorted(set(selected), key=lambda idx: root_z[idx])
    return selected[:n_postures]


def camera_center_and_direction(calibration) -> tuple[np.ndarray, np.ndarray]:
    """Retourne le centre optique et l'axe de visee d'une camera."""
    center = -calibration.R.T @ calibration.tvec.reshape(3)
    direction = calibration.R.T @ np.array([0.0, 0.0, 1.0])
    norm = np.linalg.norm(direction)
    if norm > 1e-12:
        direction = direction / norm
    return center, direction


def finite_bounds(points_3d: np.ndarray, camera_centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return finite lower and upper bounds covering poses and camera centers."""

    xyz = points_3d[np.all(np.isfinite(points_3d), axis=2)]
    if camera_centers.size:
        xyz = np.vstack((xyz, camera_centers))
    mins = np.nanmin(xyz, axis=0)
    maxs = np.nanmax(xyz, axis=0)
    pad = 0.12 * np.maximum(maxs - mins, 0.5)
    return mins - pad, maxs + pad


def set_equal_3d_axes(ax, mins: np.ndarray, maxs: np.ndarray) -> None:
    """Apply equal-aspect limits to one 3D Matplotlib axis."""

    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def draw_world_coordinate_system(ax, mins: np.ndarray, maxs: np.ndarray) -> None:
    """Dessine le repere monde pour expliciter l'origine et l'orientation des axes."""
    origin = np.zeros(3, dtype=float)
    axis_length = 0.18 * np.max(maxs - mins)
    axis_length = max(axis_length, 0.25)
    axes_spec = (
        ("X", np.array([1.0, 0.0, 0.0]), "#d62728"),
        ("Y", np.array([0.0, 1.0, 0.0]), "#2ca02c"),
        ("Z", np.array([0.0, 0.0, 1.0]), "#1f77b4"),
    )
    ax.scatter(origin[0], origin[1], origin[2], color="black", s=36, marker="o")
    ax.text(origin[0], origin[1], origin[2], "O", color="black", fontsize=10)
    for axis_name, axis_dir, color in axes_spec:
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            axis_dir[0],
            axis_dir[1],
            axis_dir[2],
            length=axis_length,
            color=color,
            linewidth=2.0,
            arrow_length_ratio=0.15,
        )
        end = origin + axis_length * axis_dir
        ax.text(end[0], end[1], end[2], axis_name, color=color, fontsize=10)


def draw_coordinate_system(
    ax, origin: np.ndarray, rotation_matrix: np.ndarray, axis_length: float, label_prefix: str
) -> None:
    """Dessine un repere local 3D a partir d'une origine et d'une matrice de rotation."""
    axes_spec = (
        (f"{label_prefix}X", rotation_matrix[:, 0], "#d62728"),
        (f"{label_prefix}Y", rotation_matrix[:, 1], "#2ca02c"),
        (f"{label_prefix}Z", rotation_matrix[:, 2], "#1f77b4"),
    )
    ax.scatter(origin[0], origin[1], origin[2], color="black", s=36, marker="o")
    ax.text(origin[0], origin[1], origin[2], label_prefix.rstrip("_") or "R", color="black", fontsize=10)
    for axis_name, axis_dir, color in axes_spec:
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            axis_dir[0],
            axis_dir[1],
            axis_dir[2],
            length=axis_length,
            color=color,
            linewidth=2.0,
            arrow_length_ratio=0.15,
        )
        end = origin + axis_length * axis_dir
        ax.text(end[0], end[1], end[2], axis_name, color=color, fontsize=10)


def draw_translated_global_coordinate_system(
    ax, origin: np.ndarray, axis_length: float, label_prefix: str = "G_"
) -> None:
    """Dessine un repere global translate a une origine donnee avec des couleurs plus claires."""
    axes_spec = (
        (f"{label_prefix}X", np.array([1.0, 0.0, 0.0]), "#f4a6a6"),
        (f"{label_prefix}Y", np.array([0.0, 1.0, 0.0]), "#a6dba0"),
        (f"{label_prefix}Z", np.array([0.0, 0.0, 1.0]), "#9ecae1"),
    )
    ax.scatter(origin[0], origin[1], origin[2], color="black", s=36, marker="o")
    ax.text(origin[0], origin[1], origin[2], label_prefix.rstrip("_") or "G", color="black", fontsize=10)
    for axis_name, axis_dir, color in axes_spec:
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            axis_dir[0],
            axis_dir[1],
            axis_dir[2],
            length=axis_length,
            color=color,
            linewidth=1.8,
            arrow_length_ratio=0.15,
        )
        end = origin + axis_length * axis_dir
        ax.text(end[0], end[1], end[2], axis_name, color=color, fontsize=10)


def draw_frame_skeleton(ax, points_frame: np.ndarray, color: str = "#444444") -> None:
    """Dessine un stick figure 3D pour une frame donnee."""
    valid = np.all(np.isfinite(points_frame), axis=1)
    if np.any(valid):
        ax.scatter(
            points_frame[valid, 0], points_frame[valid, 1], points_frame[valid, 2], s=22, color=color, alpha=0.95
        )
    for name_a, name_b in SKELETON_EDGES:
        point_a = points_frame[KP_INDEX[name_a]]
        point_b = points_frame[KP_INDEX[name_b]]
        if np.all(np.isfinite(point_a)) and np.all(np.isfinite(point_b)):
            ax.plot(
                [point_a[0], point_b[0]],
                [point_a[1], point_b[1]],
                [point_a[2], point_b[2]],
                color=color,
                linewidth=2.4,
                alpha=0.95,
            )


def export_first_frame_with_root_coordinate_system(
    points_3d: np.ndarray, frames: np.ndarray, fps: float, output_path: Path
) -> None:
    """Exporte la premiere frame avec le repere monde et le repere racine du tronc."""
    from scipy.spatial.transform import Rotation

    points_frame = points_3d[0]
    translations, rotations_xyz = compute_trunk_dofs_from_triangulation(points_3d[:1])
    root_translation = translations[0]
    root_rotation = rotations_xyz[0]

    finite_points = points_frame[np.all(np.isfinite(points_frame), axis=1)]
    if finite_points.size == 0:
        raise ValueError("La premiere frame ne contient aucun point 3D fini.")

    mins = np.nanmin(finite_points, axis=0)
    maxs = np.nanmax(finite_points, axis=0)
    pad = 0.2 * np.maximum(maxs - mins, 0.4)
    mins = mins - pad
    maxs = maxs + pad

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    draw_frame_skeleton(ax, points_frame)
    draw_world_coordinate_system(ax, mins, maxs)

    if np.all(np.isfinite(root_translation)) and np.all(np.isfinite(root_rotation)):
        axis_length = max(0.18 * np.max(maxs - mins), 0.2)
        draw_translated_global_coordinate_system(ax, root_translation, axis_length, "G_")
        root_rotation_matrix = Rotation.from_euler("xyz", root_rotation, degrees=False).as_matrix()
        draw_coordinate_system(ax, root_translation, root_rotation_matrix, axis_length, "R_")

    set_equal_3d_axes(ax, mins, maxs)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Premiere frame avec repere global et repere racine: f{int(frames[0])} / {frames[0] / fps:.3f}s")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Build the 3D posture snapshot figure from one reconstruction bundle."""

    args = parse_args()
    data = np.load(args.triangulation, allow_pickle=True)
    points_3d = np.asarray(data["points_3d"], dtype=float)
    frames = np.asarray(data["frames"], dtype=int) if "frames" in data else np.arange(points_3d.shape[0], dtype=int)
    calibrations = load_calibrations(args.calib)

    selected_frames = select_posture_frames(points_3d, args.n_postures)
    camera_names = list(calibrations.keys())
    camera_centers = []
    camera_directions = []
    for cam_name in camera_names:
        center, direction = camera_center_and_direction(calibrations[cam_name])
        camera_centers.append(center)
        camera_directions.append(direction)
    camera_centers_arr = np.asarray(camera_centers, dtype=float)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.cm.plasma(np.linspace(0.05, 0.9, len(selected_frames)))
    for color, frame_idx in zip(colors, selected_frames):
        pts = points_3d[frame_idx]
        valid = np.all(np.isfinite(pts), axis=1)
        if np.any(valid):
            ax.scatter(pts[valid, 0], pts[valid, 1], pts[valid, 2], s=16, color=color, alpha=0.9)
        for name_a, name_b in SKELETON_EDGES:
            point_a = pts[KP_INDEX[name_a]]
            point_b = pts[KP_INDEX[name_b]]
            if np.all(np.isfinite(point_a)) and np.all(np.isfinite(point_b)):
                ax.plot(
                    [point_a[0], point_b[0]],
                    [point_a[1], point_b[1]],
                    [point_a[2], point_b[2]],
                    color=color,
                    linewidth=2.0,
                    alpha=0.9,
                )
        root = pelvis_center(points_3d[frame_idx : frame_idx + 1])[0]
        if np.all(np.isfinite(root)):
            time_s = frames[frame_idx] / args.fps
            ax.text(root[0], root[1], root[2], f"f{frames[frame_idx]} / {time_s:.2f}s", color=color, fontsize=8)

    for cam_name, center, direction in zip(camera_names, camera_centers, camera_directions):
        ax.scatter(center[0], center[1], center[2], color="black", s=28, marker="^")
        ax.quiver(
            center[0],
            center[1],
            center[2],
            direction[0],
            direction[1],
            direction[2],
            length=args.camera_scale,
            color="black",
            linewidth=1.3,
            arrow_length_ratio=0.2,
        )
        ax.text(center[0], center[1], center[2], cam_name.replace("Camera", ""), fontsize=8, color="black")

    mins, maxs = finite_bounds(points_3d[selected_frames], camera_centers_arr)
    draw_world_coordinate_system(ax, mins, maxs)
    set_equal_3d_axes(ax, mins, maxs)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Postures 3D representatives et position des cameras")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    export_first_frame_with_root_coordinate_system(points_3d, frames, args.fps, args.first_frame_output)
    print(f"Figure exportee dans: {args.output}")
    print(f"Figure premiere frame exportee dans: {args.first_frame_output}")
    print("Frames selectionnees:", [int(frames[idx]) for idx in selected_frames])


if __name__ == "__main__":
    main()
