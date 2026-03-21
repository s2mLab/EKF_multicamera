#!/usr/bin/env python3
"""Explore temporellement les detections 2D des cameras.

La figure generee aide a diagnostiquer:
- trous temporels par camera,
- sauts de trajectoire,
- baisse de confiance de l'estimateur,
- decalages entre vues.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

LOCAL_MPLCONFIG = Path("/Users/mickaelbegon/Documents/Playground/.cache/matplotlib")
LOCAL_MPLCONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_MPLCONFIG))

import matplotlib.pyplot as plt
import numpy as np

from vitpose_ekf_pipeline import load_calibrations, load_pose_data


DEFAULT_CALIB = Path("inputs/Calib.toml")
DEFAULT_KEYPOINTS = Path("inputs/1_partie_0429_keypoints.json")
DEFAULT_OUTPUT = Path("outputs/vitpose_full/camera_2d_temporal_exploration.png")
DEFAULT_FPS = 120.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Figure temporelle d'exploration des donnees 2D par camera.")
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB, help="Calibration TOML.")
    parser.add_argument("--keypoints", type=Path, default=DEFAULT_KEYPOINTS, help="JSON des detections 2D.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="PNG de sortie.")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Frequence pour l'axe temporel.")
    return parser.parse_args()


def robust_center(points_2d: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcule un centre 2D robuste par frame a partir des keypoints visibles."""
    valid = np.all(np.isfinite(points_2d), axis=2) & (scores > 0)
    x = np.where(valid, points_2d[:, :, 0], np.nan)
    y = np.where(valid, points_2d[:, :, 1], np.nan)
    return np.nanmedian(x, axis=1), np.nanmedian(y, axis=1)


def save_figure(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    calibrations = load_calibrations(args.calib)
    pose_data = load_pose_data(args.keypoints, calibrations)

    t = pose_data.frames / args.fps
    n_cams = len(pose_data.camera_names)
    fig, axes = plt.subplots(n_cams, 4, figsize=(18, 2.5 * n_cams), sharex=True)
    axes = np.atleast_2d(axes)

    for cam_idx, cam_name in enumerate(pose_data.camera_names):
        points = np.asarray(pose_data.keypoints[cam_idx], dtype=float)
        scores = np.asarray(pose_data.scores[cam_idx], dtype=float)
        valid = np.all(np.isfinite(points), axis=2) & (scores > 0)
        x_center, y_center = robust_center(points, scores)
        mean_conf = np.nanmean(np.where(valid, scores, np.nan), axis=1)
        valid_ratio = np.mean(valid, axis=1)

        width, height = calibrations[cam_name].image_size

        ax = axes[cam_idx, 0]
        ax.plot(t, x_center, color="#1f77b4", linewidth=1.3)
        ax.set_ylim(0, width)
        ax.set_ylabel(cam_name.replace("Camera", ""))
        ax.set_title("Centre X (px)")
        ax.grid(alpha=0.2)

        ax = axes[cam_idx, 1]
        ax.plot(t, y_center, color="#ff7f0e", linewidth=1.3)
        ax.set_ylim(height, 0)
        ax.set_title("Centre Y (px)")
        ax.grid(alpha=0.2)

        ax = axes[cam_idx, 2]
        ax.plot(t, mean_conf, color="#2ca02c", linewidth=1.3)
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Confiance moyenne")
        ax.grid(alpha=0.2)

        ax = axes[cam_idx, 3]
        ax.plot(t, valid_ratio, color="#d62728", linewidth=1.3)
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Taux de keypoints valides")
        ax.grid(alpha=0.2)

    for ax in axes[-1]:
        ax.set_xlabel("Temps (s)")

    fig.suptitle("Exploration temporelle des donnees 2D par camera", y=1.01)
    save_figure(fig, args.output)
    print(f"Figure exportee dans: {args.output}")


if __name__ == "__main__":
    main()
