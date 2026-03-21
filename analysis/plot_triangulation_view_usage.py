#!/usr/bin/env python3
"""Visualise quelles vues sont retenues pendant la triangulation.

Le script lit le cache NPZ produit par la triangulation robuste et construit
une figure montrant, frame par frame, quelles cameras ont effectivement
contribue a la reconstruction:

1. une heatmap agregee camera x frame avec le nombre de keypoints triangules
   en utilisant chaque camera ;
2. une heatmap detaillee (camera, keypoint) x frame pour voir plus finement
   quelles detections sont exclues.
"""
from __future__ import annotations

import argparse
import json
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

DEFAULT_CAMERA_FPS = 120.0


def parse_args() -> argparse.Namespace:
    """Construit l'interface CLI du script."""
    parser = argparse.ArgumentParser(description="Genere une figure montrant les vues utilisees pour la triangulation a chaque image.")
    parser.add_argument(
        "--triangulation",
        type=Path,
        default=Path("outputs/vitpose_full/triangulation_pose2sim_like.npz"),
        help="Cache NPZ de triangulation robuste.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/vitpose_full/triangulation_view_usage.png"),
        help="Figure de sortie.",
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_CAMERA_FPS, help="Frequence pour l'axe temporel.")
    parser.add_argument(
        "--detail-mode",
        choices=("used", "coherence", "reprojection"),
        default="used",
        help="Information affichee dans la heatmap detaillee.",
    )
    return parser.parse_args()


def load_metadata(raw_metadata: np.ndarray | str | bytes) -> dict[str, object]:
    """Decode le champ metadata, stocke comme JSON texte dans le cache NPZ."""
    if isinstance(raw_metadata, np.ndarray):
        raw_metadata = raw_metadata.item()
    if isinstance(raw_metadata, bytes):
        raw_metadata = raw_metadata.decode("utf-8")
    if isinstance(raw_metadata, str):
        return json.loads(raw_metadata)
    return {}


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Ecrit la figure sur disque en creant le dossier si besoin."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_detail_matrix(
    detail_mode: str,
    used_views: np.ndarray,
    multiview_coherence: np.ndarray,
    reprojection_error_per_view: np.ndarray,
) -> tuple[np.ndarray, str, str]:
    """Construit la matrice detaillee a afficher et ses labels."""
    n_frames, n_keypoints, n_cams = used_views.shape

    if detail_mode == "used":
        detail = used_views.astype(float).transpose(2, 1, 0).reshape(n_cams * n_keypoints, n_frames)
        return detail, "Vue utilisee", "binary"

    if detail_mode == "coherence":
        detail = multiview_coherence.transpose(2, 1, 0).reshape(n_cams * n_keypoints, n_frames)
        return detail, "Coherence multivue", "viridis"

    detail = reprojection_error_per_view.transpose(2, 1, 0).reshape(n_cams * n_keypoints, n_frames)
    return detail, "Erreur de reprojection (px)", "magma_r"


def main() -> None:
    """Charge les donnees de triangulation et genere la figure."""
    args = parse_args()
    data = np.load(args.triangulation, allow_pickle=True)

    excluded_views = np.asarray(data["excluded_views"], dtype=bool)
    used_views = ~excluded_views
    frames = np.asarray(data["frames"])
    multiview_coherence = np.asarray(data["multiview_coherence"], dtype=float)
    reprojection_error_per_view = np.asarray(data["reprojection_error_per_view"], dtype=float)
    keypoint_names = np.asarray(data["keypoint_names"])
    metadata = load_metadata(data["metadata"])
    camera_names = metadata.get("camera_names", [f"cam_{idx}" for idx in range(used_views.shape[2])])

    # Nombre de keypoints pour lesquels chaque camera a ete retenue a chaque frame.
    used_count_per_camera = used_views.sum(axis=1).T  # (n_cam, n_frames)

    detail_matrix, detail_label, detail_cmap = build_detail_matrix(
        args.detail_mode, used_views, multiview_coherence, reprojection_error_per_view
    )

    time_s = frames / args.fps
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.8, 0.5])

    ax_top = fig.add_subplot(gs[0, 0])
    im_top = ax_top.imshow(
        used_count_per_camera,
        aspect="auto",
        interpolation="nearest",
        cmap="Greys",
        vmin=0,
        vmax=used_views.shape[1],
        extent=[time_s[0], time_s[-1], used_views.shape[2] - 0.5, -0.5],
    )
    ax_top.set_yticks(np.arange(len(camera_names)))
    ax_top.set_yticklabels(camera_names)
    ax_top.set_ylabel("Camera")
    ax_top.set_title("Nombre de keypoints triangules avec chaque vue")
    cbar_top = fig.colorbar(im_top, ax=ax_top, pad=0.01)
    cbar_top.set_label("Keypoints utilises")

    ax_mid = fig.add_subplot(gs[1, 0], sharex=ax_top)
    if args.detail_mode == "used":
        vmin, vmax = 0.0, 1.0
    elif args.detail_mode == "coherence":
        vmin, vmax = 0.0, 1.0
    else:
        finite = detail_matrix[np.isfinite(detail_matrix)]
        vmax = np.percentile(finite, 95) if finite.size else 1.0
        vmin = 0.0
    im_mid = ax_mid.imshow(
        detail_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=detail_cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[time_s[0], time_s[-1], detail_matrix.shape[0] - 0.5, -0.5],
    )
    row_labels = [f"{cam}:{kp}" for cam in camera_names for kp in keypoint_names]
    step = max(1, len(row_labels) // 24)
    ax_mid.set_yticks(np.arange(0, len(row_labels), step))
    ax_mid.set_yticklabels(row_labels[::step], fontsize=7)
    ax_mid.set_ylabel("Camera : keypoint")
    ax_mid.set_xlabel("Temps (s)")
    ax_mid.set_title(f"Detail frame par frame ({args.detail_mode})")
    cbar_mid = fig.colorbar(im_mid, ax=ax_mid, pad=0.01)
    cbar_mid.set_label(detail_label)

    ax_bottom = fig.add_subplot(gs[2, 0], sharex=ax_top)
    n_used_per_frame = np.sum(used_views, axis=(1, 2))
    max_possible = used_views.shape[1] * used_views.shape[2]
    ax_bottom.plot(time_s, n_used_per_frame, color="black", linewidth=1.6)
    ax_bottom.set_ylabel("Nb obs.")
    ax_bottom.set_xlabel("Temps (s)")
    ax_bottom.set_title("Nombre total d'observations 2D retenues pour la triangulation")
    ax_bottom.set_ylim(0, max_possible * 1.05)
    ax_bottom.grid(alpha=0.3)

    fig.suptitle("Utilisation des vues pendant la triangulation robuste", y=0.995)
    save_figure(fig, args.output)
    print(f"Figure ecrite dans {args.output}")


if __name__ == "__main__":
    main()
