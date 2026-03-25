#!/usr/bin/env python3
"""Animation 3D d'une stick figure a partir des points 3D triangules.

Le script lit `triangulation_pose2sim_like.npz` produit par le pipeline et
construit une animation 3D simple en reliant les keypoints COCO17 pertinents.
L'objectif est de verifier visuellement la reconstruction 3D brute avant
l'etape de filtrage cinematique.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOCAL_MPLCONFIG = Path("/Users/mickaelbegon/Documents/Playground/.cache/matplotlib")
LOCAL_MPLCONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_MPLCONFIG))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

DEFAULT_CAMERA_FPS = 120.0


COCO17 = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
KP_INDEX = {name: i for i, name in enumerate(COCO17)}

# Stick figure simple pour COCO17.
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

LEFT_COLOR = "#4c72b0"
RIGHT_COLOR = "#dd8452"
MID_COLOR = "#55a868"


def parse_args() -> argparse.Namespace:
    """Construit l'interface CLI du script d'animation."""
    parser = argparse.ArgumentParser(description="Anime les points 3D triangules sous forme de stick figure.")
    parser.add_argument(
        "--triangulation",
        type=Path,
        default=Path("output") / "vitpose_biobuddy_check" / "triangulation_pose2sim_like.npz",
        help="Fichier NPZ contenant `points_3d`",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output") / "vitpose_biobuddy_check" / "triangulated_stick_figure.gif",
        help="Chemin du GIF de sortie",
    )
    parser.add_argument("--fps", type=float, default=20.0, help="Frequence d'affichage du GIF")
    parser.add_argument("--stride", type=int, default=1, help="Sous-echantillonnage temporel pour accelerer l'export")
    parser.add_argument(
        "--triangulation-fps",
        type=float,
        default=DEFAULT_CAMERA_FPS,
        help="Frequence d'acquisition de la reconstruction triangulee en Hz",
    )
    return parser.parse_args()


def load_points(triangulation_path: Path, stride: int, triangulation_fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Charge et sous-echantillonne les points 3D triangules.

    Si le fichier ne contient que des indices de frame, ils sont interpretes
    a la frequence d'acquisition fournie.
    """
    data = np.load(triangulation_path, allow_pickle=True)
    points = data["points_3d"][::stride]
    frames = data["frames"][::stride] if "frames" in data else np.arange(points.shape[0])
    time_s = np.asarray(frames, dtype=float) / triangulation_fps
    return points, time_s


def valid_segment(point_a: np.ndarray, point_b: np.ndarray) -> bool:
    """Retourne vrai si les deux extremites du segment sont disponibles."""
    return np.all(np.isfinite(point_a)) and np.all(np.isfinite(point_b))


def edge_color(name_a: str, name_b: str) -> str:
    """Attribue une couleur gauche/droite/axiale a chaque segment."""
    joined = f"{name_a} {name_b}"
    if "left" in joined:
        return LEFT_COLOR
    if "right" in joined:
        return RIGHT_COLOR
    return MID_COLOR


def compute_axis_limits(points: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Calcule des bornes 3D fixes pour eviter une camera qui bouge dans l'animation."""
    flat = points.reshape(-1, 3)
    valid = np.all(np.isfinite(flat), axis=1)
    flat = flat[valid]
    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.55 * np.max(maxs - mins)
    radius = max(radius, 0.25)
    return (
        (center[0] - radius, center[0] + radius),
        (center[1] - radius, center[1] + radius),
        (center[2] - radius, center[2] + radius),
    )


def create_animation(points: np.ndarray, time_s: np.ndarray, output_path: Path, fps: float) -> None:
    """Cree et enregistre un GIF anime de la reconstruction 3D brute."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    xlim, ylim, zlim = compute_axis_limits(points)

    scatter = ax.scatter([], [], [], s=35, c=MID_COLOR, depthshade=False)
    line_artists = []
    for name_a, name_b in SKELETON_EDGES:
        (line,) = ax.plot([], [], [], linewidth=2.2, color=edge_color(name_a, name_b))
        line_artists.append((line, name_a, name_b))

    label = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    def init():
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Stick figure 3D issue de la triangulation")
        ax.view_init(elev=18, azim=-65)
        return [scatter, label] + [line for line, _, _ in line_artists]

    def update(frame_idx: int):
        frame_points = points[frame_idx]
        valid = np.all(np.isfinite(frame_points), axis=1)
        xyz = frame_points[valid]
        if xyz.size:
            scatter._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
        else:
            scatter._offsets3d = ([], [], [])

        for line, name_a, name_b in line_artists:
            point_a = frame_points[KP_INDEX[name_a]]
            point_b = frame_points[KP_INDEX[name_b]]
            if valid_segment(point_a, point_b):
                line.set_data([point_a[0], point_b[0]], [point_a[1], point_b[1]])
                line.set_3d_properties([point_a[2], point_b[2]])
            else:
                line.set_data([], [])
                line.set_3d_properties([])

        label.set_text(f"t = {time_s[frame_idx]:.2f} s")
        return [scatter, label] + [line for line, _, _ in line_artists]

    anim = FuncAnimation(fig, update, init_func=init, frames=points.shape[0], interval=1000 / fps, blit=False)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main() -> None:
    """Point d'entree CLI."""
    args = parse_args()
    points, time_s = load_points(
        args.triangulation, stride=max(args.stride, 1), triangulation_fps=args.triangulation_fps
    )
    create_animation(points, time_s, args.output, fps=args.fps)
    print(f"Animation exportee dans: {args.output}")


if __name__ == "__main__":
    main()
