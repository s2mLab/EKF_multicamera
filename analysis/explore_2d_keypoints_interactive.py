#!/usr/bin/env python3
"""Exploration interactive temporelle des keypoints 2D sur les 8 cameras.

Le script ouvre une fenetre matplotlib avec:
- 8 subplots (une camera par panneau),
- tous les keypoints traces temporellement,
- des cases a cocher pour masquer/afficher un keypoint globalement,
- un selecteur pour changer la composante affichee (`x`, `y`, `score`).

Il ouvre egalement une seconde figure qui montre, pour chaque keypoint,
le nombre de cameras qui le voient au cours du temps.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOCAL_MPLCONFIG = Path("/Users/mickaelbegon/Documents/Playground/.cache/matplotlib")
LOCAL_MPLCONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_MPLCONFIG))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons, RadioButtons

from vitpose_ekf_pipeline import COCO17, load_calibrations, load_pose_data

DEFAULT_CALIB = Path("inputs/calibration/Calib.toml")
DEFAULT_KEYPOINTS = Path("inputs/keypoints/1_partie_0429_keypoints.json")
DEFAULT_FPS = 120.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exploration interactive des keypoints 2D sur les 8 cameras.")
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB, help="Calibration TOML.")
    parser.add_argument("--keypoints", type=Path, default=DEFAULT_KEYPOINTS, help="JSON des detections 2D.")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Frequence pour l'axe temporel.")
    parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=0.2,
        help="Seuil de score pour considerer qu'un keypoint est reellement vu par une camera.",
    )
    parser.add_argument(
        "--component",
        choices=("x", "y", "score"),
        default="y",
        help="Composante affichee au lancement.",
    )
    return parser.parse_args()


def camera_layout(n_cameras: int) -> tuple[int, int]:
    ncols = 4
    nrows = int(np.ceil(n_cameras / ncols))
    return nrows, ncols


def extract_component(points: np.ndarray, scores: np.ndarray, component: str) -> np.ndarray:
    """Retourne la serie temporelle voulue pour tous les keypoints."""
    valid = np.all(np.isfinite(points), axis=2) & (scores > 0)
    if component == "x":
        data = np.where(valid, points[:, :, 0], np.nan)
    elif component == "y":
        data = np.where(valid, points[:, :, 1], np.nan)
    else:
        data = np.where(valid, scores, np.nan)
    return data


def configure_axis_limits(ax, component: str, image_size: tuple[int, int]) -> None:
    width, height = image_size
    if component == "x":
        ax.set_ylim(0, width)
        ax.set_ylabel("x (px)")
    elif component == "y":
        ax.set_ylim(height, 0)
        ax.set_ylabel("y (px)")
    else:
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("score")


def compute_camera_counts(points: np.ndarray, scores: np.ndarray, visibility_threshold: float) -> np.ndarray:
    """Compte le nombre de cameras qui voient reellement chaque keypoint.

    Un keypoint est considere visible si ses coordonnees sont finies et si son
    score depasse un seuil de confiance. Avec les sorties VITPose, `score > 0`
    est trop permissif et conduit souvent a un comptage artificiellement binaire
    (presque toujours 8).
    """
    valid = np.all(np.isfinite(points), axis=3) & (scores >= visibility_threshold)
    return valid.sum(axis=0)


def main() -> None:
    args = parse_args()
    calibrations = load_calibrations(args.calib)
    pose_data = load_pose_data(args.keypoints, calibrations)
    t = pose_data.frames / args.fps

    n_cams = len(pose_data.camera_names)
    nrows, ncols = camera_layout(n_cams)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 9), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    fig.subplots_adjust(left=0.05, right=0.78, top=0.92, bottom=0.07, wspace=0.22, hspace=0.32)
    fig_counts, ax_counts = plt.subplots(figsize=(16, 6))
    fig_counts.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.10)

    colors = plt.cm.tab20(np.linspace(0, 1, len(COCO17)))
    visible = {kp_name: True for kp_name in COCO17}
    line_handles: dict[str, list] = {kp_name: [] for kp_name in COCO17}
    count_handles: dict[str, object] = {}
    current_component = {"value": args.component}
    camera_counts = compute_camera_counts(
        np.asarray(pose_data.keypoints, dtype=float),
        np.asarray(pose_data.scores, dtype=float),
        args.visibility_threshold,
    )

    def redraw(component: str) -> None:
        for ax_idx, ax in enumerate(axes):
            ax.clear()
            if ax_idx >= n_cams:
                ax.axis("off")
                continue
            cam_name = pose_data.camera_names[ax_idx]
            points = np.asarray(pose_data.keypoints[ax_idx], dtype=float)
            scores = np.asarray(pose_data.scores[ax_idx], dtype=float)
            data = extract_component(points, scores, component)

            line_handles_per_cam = []
            for kp_idx, (kp_name, color) in enumerate(zip(COCO17, colors)):
                valid = np.isfinite(data[:, kp_idx])
                (line,) = ax.plot(
                    t[valid],
                    data[valid, kp_idx],
                    color=color,
                    linestyle="None",
                    marker="o",
                    markersize=2.2,
                    alpha=0.9 if visible[kp_name] else 0.08,
                    visible=visible[kp_name],
                    label=kp_name,
                )
                line_handles[kp_name].append(line)
                line_handles_per_cam.append(line)

            ax.set_title(cam_name.replace("Camera", ""))
            configure_axis_limits(ax, component, calibrations[cam_name].image_size)
            ax.grid(alpha=0.2)
            if ax_idx >= (nrows - 1) * ncols:
                ax.set_xlabel("Temps (s)")

        fig.suptitle(f"Exploration interactive des keypoints 2D | composante = {component}", y=0.98)
        fig.canvas.draw_idle()

    def redraw_counts() -> None:
        ax_counts.clear()
        for kp_idx, (kp_name, color) in enumerate(zip(COCO17, colors)):
            valid = camera_counts[:, kp_idx] > 0
            (line,) = ax_counts.plot(
                t[valid],
                camera_counts[valid, kp_idx],
                color=color,
                linestyle="None",
                marker="o",
                markersize=2.4,
                alpha=0.9 if visible[kp_name] else 0.08,
                visible=visible[kp_name],
                label=kp_name,
            )
            count_handles[kp_name] = line

        ax_counts.set_title(
            f"Nombre de cameras qui voient chaque keypoint | seuil score >= {args.visibility_threshold:.2f}"
        )
        ax_counts.set_xlabel("Temps (s)")
        ax_counts.set_ylabel("Nombre de cameras")
        ax_counts.set_ylim(-0.2, n_cams + 0.2)
        ax_counts.set_yticks(np.arange(0, n_cams + 1, 1))
        ax_counts.grid(alpha=0.25)
        fig_counts.canvas.draw_idle()

    redraw(current_component["value"])
    redraw_counts()

    check_ax = fig.add_axes([0.81, 0.16, 0.17, 0.70])
    checks = CheckButtons(check_ax, COCO17, [visible[name] for name in COCO17])
    check_ax.set_title("Keypoints")

    radio_ax = fig.add_axes([0.81, 0.05, 0.17, 0.08])
    radio = RadioButtons(radio_ax, ("x", "y", "score"), active=("x", "y", "score").index(args.component))
    radio_ax.set_title("Composante")

    def on_toggle(label: str) -> None:
        visible[label] = not visible[label]
        for line in line_handles[label]:
            line.set_visible(visible[label])
        if label in count_handles:
            count_handles[label].set_visible(visible[label])
        fig.canvas.draw_idle()
        fig_counts.canvas.draw_idle()

    def on_component(label: str) -> None:
        for kp_name in COCO17:
            line_handles[kp_name].clear()
        current_component["value"] = label
        redraw(label)

    checks.on_clicked(on_toggle)
    radio.on_clicked(on_component)
    plt.show()


if __name__ == "__main__":
    main()
