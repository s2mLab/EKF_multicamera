#!/usr/bin/env python3
"""Generation de figures de comparaison cinematique.

Ce script lit les sorties du pipeline d'estimation (`ekf_states.npz`,
`kalman_comparison.npz`, `summary.json`) et exporte plusieurs familles de
figures pour comparer le nouvel EKF multi-vues avec le Kalman classique de
`biorbd`.
"""
from __future__ import annotations

import argparse
import json
import os
from math import ceil
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

from kinematics.root_kinematics import (
    TRUNK_ROTATION_NAMES,
    centered_finite_difference,
    compute_trunk_dofs_from_points,
    extract_root_from_q,
)
from vitpose_ekf_pipeline import (
    COCO17,
    DEFAULT_COHERENCE_CONFIDENCE_FLOOR,
    load_calibrations,
    load_pose_data,
)

DEFAULT_CAMERA_FPS = 120.0
DEFAULT_CALIB = Path("inputs/Calib.toml")
DEFAULT_KEYPOINTS = Path("inputs/1_partie_0429_keypoints.json")
TRIANGULATION_ROOT_LABELS = np.asarray(
    [
        "Triangulation:TRUNK:TransX",
        "Triangulation:TRUNK:TransY",
        "Triangulation:TRUNK:TransZ",
        "Triangulation:TRUNK:RotY",
        "Triangulation:TRUNK:RotX",
        "Triangulation:TRUNK:RotZ",
    ],
    dtype=object,
)
TRIANGULATION_ROOT_VELOCITY_LABELS = np.asarray(
    [
        "Triangulation:TRUNK:VelX",
        "Triangulation:TRUNK:VelY",
        "Triangulation:TRUNK:VelZ",
        "Triangulation:TRUNK:VelRotY",
        "Triangulation:TRUNK:VelRotX",
        "Triangulation:TRUNK:VelRotZ",
    ],
    dtype=object,
)


def parse_args() -> argparse.Namespace:
    """Construit l'interface CLI du script de visualisation."""
    parser = argparse.ArgumentParser(description="Genere des figures de comparaison cinematique a partir des sorties du pipeline.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs") / "vitpose_biobuddy_check",
        help="Dossier contenant ekf_states.npz, kalman_comparison.npz et summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Dossier de sortie pour les figures. Par defaut: <input-dir>/figures",
    )
    parser.add_argument(
        "--ekf-2d-variant",
        choices=("acc", "dyn", "flip_acc", "flip_dyn", "both"),
        default="acc",
        help="Version EKF 2D a comparer a EKF 3D.",
    )
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB, help="Fichier de calibration pour reconstruire les poids camera.")
    parser.add_argument("--keypoints", type=Path, default=DEFAULT_KEYPOINTS, help="JSON des detections 2D pour reconstruire les poids camera.")
    parser.add_argument("--fps", type=float, default=DEFAULT_CAMERA_FPS, help="Frequence d'echantillonnage pour l'axe temporel.")
    parser.add_argument("--top-dofs", type=int, default=8, help="Nombre de DoF les plus divergents a mettre en avant.")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    """Cree le dossier cible si necessaire et retourne le chemin."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_data(input_dir: Path) -> dict[str, object]:
    """Charge les fichiers numeriques produits par le pipeline."""
    ekf = np.load(input_dir / "ekf_states.npz", allow_pickle=True)
    comparison = np.load(input_dir / "kalman_comparison.npz", allow_pickle=True)
    triangulation = np.load(input_dir / "triangulation_pose2sim_like.npz", allow_pickle=True)
    summary_path = input_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        with summary_path.open("r") as f:
            summary = json.load(f)
    return {"ekf": ekf, "comparison": comparison, "triangulation": triangulation, "summary": summary}


def load_optional_npz(path: Path) -> np.lib.npyio.NpzFile | None:
    """Charge un NPZ si present."""
    return np.load(path, allow_pickle=True) if path.exists() else None


def select_ekf_2d_variant(input_dir: Path, variant: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Retourne la trajectoire 2D choisie et le fichier de comparaison associe."""
    ekf = np.load(input_dir / "ekf_states.npz", allow_pickle=True)
    comparison_acc = load_optional_npz(input_dir / "kalman_comparison_acc.npz")
    comparison_dyn = load_optional_npz(input_dir / "kalman_comparison_dyn.npz")
    comparison_default = np.load(input_dir / "kalman_comparison.npz", allow_pickle=True)

    if variant == "dyn":
        if "q_ekf_2d_dyn" not in ekf or comparison_dyn is None:
            raise FileNotFoundError("La variante EKF 2D DYN n'est pas disponible dans ce dossier.")
        return np.asarray(ekf["q_ekf_2d_dyn"], dtype=float), comparison_dyn["rmse_per_dof"], "DYN"

    if variant == "flip_acc":
        comparison_flip_acc = load_optional_npz(input_dir / "kalman_comparison_flip_acc.npz")
        if "q_ekf_2d_flip_acc" not in ekf or comparison_flip_acc is None:
            raise FileNotFoundError("La variante EKF 2D FLIP ACC n'est pas disponible dans ce dossier.")
        return np.asarray(ekf["q_ekf_2d_flip_acc"], dtype=float), comparison_flip_acc["rmse_per_dof"], "FLIP_ACC"

    if variant == "flip_dyn":
        comparison_flip_dyn = load_optional_npz(input_dir / "kalman_comparison_flip_dyn.npz")
        if "q_ekf_2d_flip_dyn" not in ekf or comparison_flip_dyn is None:
            raise FileNotFoundError("La variante EKF 2D FLIP DYN n'est pas disponible dans ce dossier.")
        return np.asarray(ekf["q_ekf_2d_flip_dyn"], dtype=float), comparison_flip_dyn["rmse_per_dof"], "FLIP_DYN"

    if variant == "both":
        if "q_ekf_2d_dyn" not in ekf:
            raise FileNotFoundError("La variante EKF 2D DYN n'est pas disponible dans ce dossier.")
        return np.asarray(ekf["q_ekf_2d_dyn"], dtype=float), comparison_dyn["rmse_per_dof"] if comparison_dyn is not None else comparison_default["rmse_per_dof"], "DYN"

    q_acc = np.asarray(ekf["q_ekf_2d_acc"], dtype=float) if "q_ekf_2d_acc" in ekf else np.asarray(ekf["q"], dtype=float)
    rmse_acc = comparison_acc["rmse_per_dof"] if comparison_acc is not None else comparison_default["rmse_per_dof"]
    return q_acc, rmse_acc, "ACC"


def comparison_q_ekf_3d(comparison: np.lib.npyio.NpzFile) -> np.ndarray:
    """Retourne la trajectoire du filtre classique, en priorite sous le nom `ekf_3d`."""
    return comparison["q_ekf_3d"] if "q_ekf_3d" in comparison else comparison["q_biorbd_kalman"]


def comparison_qdot_ekf_3d(comparison: np.lib.npyio.NpzFile) -> np.ndarray | None:
    """Retourne les vitesses du filtre classique, avec fallback ancien nom."""
    if "qdot_ekf_3d" in comparison:
        return comparison["qdot_ekf_3d"]
    if "qdot_biorbd_kalman" in comparison:
        return comparison["qdot_biorbd_kalman"]
    return None


def comparison_qddot_ekf_3d(comparison: np.lib.npyio.NpzFile) -> np.ndarray | None:
    """Retourne les accelerations du filtre classique, avec fallback ancien nom."""
    if "qddot_ekf_3d" in comparison:
        return comparison["qddot_ekf_3d"]
    if "qddot_biorbd_kalman" in comparison:
        return comparison["qddot_biorbd_kalman"]
    return None


def time_vector(n_frames: int, fps: float) -> np.ndarray:
    """Construit l'axe temporel a partir du nombre de frames et de la frequence."""
    return np.arange(n_frames) / fps


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


def add_phase_background(ax, t: np.ndarray, airborne_mask: np.ndarray) -> None:
    """Ajoute un fond colore pour distinguer toile et phase aerienne."""
    n = min(len(t), len(airborne_mask))
    t = t[:n]
    airborne_mask = airborne_mask[:n]
    if n == 0:
        return
    starts = [0]
    for i in range(1, n):
        if airborne_mask[i] != airborne_mask[i - 1]:
            starts.append(i)
    starts.append(n)
    for start, stop in zip(starts[:-1], starts[1:]):
        color = "#f6d55c" if airborne_mask[start] else "#d9edf7"
        alpha = 0.12 if airborne_mask[start] else 0.08
        ax.axvspan(t[start], t[stop - 1] if stop - 1 < n else t[-1], color=color, alpha=alpha, zorder=0)


def compute_trunk_dofs_from_triangulation(points_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruit les 6 DoF du tronc a partir des keypoints triangules."""
    root_q = compute_trunk_dofs_from_points(np.asarray(points_3d, dtype=float), unwrap_rotations=True)
    return root_q[:, :3], root_q[:, 3:6]


def extract_trunk_root_dofs(q_names: np.ndarray, q_trajectory: np.ndarray) -> np.ndarray:
    """Extrait les 6 DoF du tronc depuis une trajectoire de coordonnees generalisees.

    Pour les rotations de la racine, on ne compare pas directement les angles
    extraits du filtre. On repasse d'abord par une matrice de rotation en
    utilisant la sequence du modele (`yxz` pour le tronc), puis on re-decompose
    cette matrice avec la meme sequence avant d'appliquer l'unwrap temporel.

    Cela limite les ecarts artificiels lies a une representation Euler brute
    qui peut rester equivalente geometriquement tout en donnant des angles
    differents.

    Point important pour l'interpretation:
    - `np.unwrap` n'essaie pas de "corriger" la biomécanique,
    - il choisit seulement la branche angulaire la plus continue dans le temps,
    - donc un angle comme `RotY` (tilt) peut devenir grand en valeur absolue si
      la decomposition Euler traverse plusieurs fois la coupure `[-pi, pi]`.

    Autrement dit, un tilt eleve apres unwrap ne signifie pas automatiquement
    une grande inclinaison physique. Cela peut aussi refleter:
    - l'accumulation de tours mathematiquement equivalents,
    - ou la sensibilite de la decomposition Euler `yxz` pres de certaines
      configurations de la racine.
    """
    return extract_root_from_q(
        np.asarray(q_names, dtype=object),
        np.asarray(q_trajectory, dtype=float),
        unwrap_rotations=True,
        renormalize_rotations=True,
    )


def extract_trunk_root_dofs_no_unwrap(q_names: np.ndarray, q_trajectory: np.ndarray) -> np.ndarray:
    """Extrait les 6 DoF du tronc en repassant par la matrice de rotation, sans unwrap.

    Ce helper est utile pour diagnostiquer si de grandes variations sur les
    angles Euler, en particulier le tilt (`RotY`), viennent de l'etape
    d'unwrap ou existent deja dans la decomposition Euler elle-meme.
    """
    return extract_root_from_q(
        np.asarray(q_names, dtype=object),
        np.asarray(q_trajectory, dtype=float),
        unwrap_rotations=False,
        renormalize_rotations=True,
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Applique un layout raisonnable puis ecrit la figure sur disque."""
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_summary_metrics(summary: dict, output_dir: Path) -> None:
    """Genere une vignette de synthese des metriques globales."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")

    lines = [
        f"Frames: {summary.get('n_frames', 'n/a')}",
        f"Erreur moyenne de reprojection: {summary.get('mean_reprojection_error_px', float('nan')):.2f} px"
        if "mean_reprojection_error_px" in summary
        else "Erreur moyenne de reprojection: n/a",
    ]
    kalman = summary.get("kalman_comparison", {})
    if kalman:
        lines.append(f"RMSE moyenne q: {kalman.get('mean_rmse_rad_or_m', float('nan')):.4f}")
        lines.append(f"MAE moyenne q: {kalman.get('mean_mae_rad_or_m', float('nan')):.4f}")
        ekf_2d_reproj = kalman.get("ekf_2d_reprojection_px")
        if ekf_2d_reproj:
            lines.append(f"Reproj EKF 2D: {ekf_2d_reproj.get('mean', float('nan')):.2f} +/- {ekf_2d_reproj.get('std', float('nan')):.2f} px")
        ekf_3d_reproj = kalman.get("ekf_3d_reprojection_px")
        if ekf_3d_reproj:
            lines.append(f"Reproj EKF 3D: {ekf_3d_reproj.get('mean', float('nan')):.2f} +/- {ekf_3d_reproj.get('std', float('nan')):.2f} px")

    ax.text(0.02, 0.95, "\n".join(lines), va="top", ha="left", fontsize=12, family="monospace")
    ax.set_title("Resume des metriques")
    save_figure(fig, output_dir / "00_summary_metrics.png")


def plot_error_bars(q_names: np.ndarray, rmse: np.ndarray, mae: np.ndarray, output_dir: Path) -> None:
    """Genere des barplots pour trier rapidement les DoF les plus divergents."""
    order = np.argsort(rmse)[::-1]
    labels = q_names[order]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    axes[0].bar(np.arange(len(labels)), rmse[order], color="#c44e52")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("RMSE par DoF")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(np.arange(len(labels)), mae[order], color="#4c72b0")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("MAE par DoF")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_xticks(np.arange(len(labels)))
    axes[1].set_xticklabels(labels, rotation=80, ha="right", fontsize=8)
    save_figure(fig, output_dir / "01_rmse_mae_bars.png")


def plot_error_heatmap(q_names: np.ndarray, q_ekf: np.ndarray, q_ekf_3d: np.ndarray, output_dir: Path) -> None:
    """Visualise l'erreur absolue par DoF et par frame sous forme de heatmap."""
    diff = np.abs(q_ekf - q_ekf_3d).T
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(diff, aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_yticks(np.arange(len(q_names)))
    ax.set_yticklabels(q_names, fontsize=8)
    ax.set_xlabel("Frame")
    ax.set_title("|q_EKF 2D - q_EKF 3D| par DoF et par frame")
    fig.colorbar(im, ax=ax, label="Erreur absolue")
    save_figure(fig, output_dir / "02_error_heatmap.png")


def plot_global_trace_pages(
    q_names: np.ndarray,
    q_ekf: np.ndarray,
    q_ekf_3d: np.ndarray,
    t: np.ndarray,
    output_dir: Path,
    airborne_mask: np.ndarray,
    page_size: int = 6,
) -> None:
    """Produit des pages multi-panneaux couvrant l'ensemble des DoF.

    Chaque page empile plusieurs signaux temporels pour rester lisible meme
    lorsque le modele contient beaucoup de DoF.
    """
    n_pages = ceil(len(q_names) / page_size)
    for page_idx in range(n_pages):
        start = page_idx * page_size
        stop = min(len(q_names), (page_idx + 1) * page_size)
        subset = range(start, stop)
        fig, axes = plt.subplots(len(list(subset)), 1, figsize=(13, 2.6 * len(list(subset))), sharex=True)
        if stop - start == 1:
            axes = [axes]
        for ax, dof_idx in zip(axes, subset):
            add_phase_background(ax, t, airborne_mask)
            ax.plot(t, q_ekf[:, dof_idx], label="EKF multi-vues", color="#dd8452", linewidth=2)
            ax.plot(t, q_ekf_3d[:, dof_idx], label="EKF 3D", color="#4c72b0", linewidth=1.8, alpha=0.9)
            ax.set_ylabel(q_names[dof_idx], fontsize=9)
            ax.grid(alpha=0.3)
        axes[0].legend(loc="upper right")
        axes[-1].set_xlabel("Temps (s)")
        fig.suptitle(f"Comparaison des cinematiques (page {page_idx + 1}/{n_pages})", y=1.02)
        save_figure(fig, output_dir / f"03_traces_page_{page_idx + 1:02d}.png")


def plot_top_dof_focus(
    q_names: np.ndarray,
    q_ekf: np.ndarray,
    q_ekf_3d: np.ndarray,
    rmse: np.ndarray,
    t: np.ndarray,
    output_dir: Path,
    top_dofs: int,
    airborne_mask: np.ndarray,
) -> None:
    """Met en avant les DoF dont le RMSE est le plus eleve."""
    top_indices = np.argsort(rmse)[::-1][:top_dofs]
    ncols = 2
    nrows = ceil(len(top_indices) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, dof_idx in zip(axes, top_indices):
        add_phase_background(ax, t, airborne_mask)
        ax.plot(t, q_ekf[:, dof_idx], label="EKF multi-vues", color="#dd8452", linewidth=2)
        ax.plot(t, q_ekf_3d[:, dof_idx], label="EKF 3D", color="#4c72b0", linewidth=1.6)
        ax.set_title(f"{q_names[dof_idx]} | RMSE={rmse[dof_idx]:.3f}")
        ax.grid(alpha=0.3)
    for ax in axes[len(top_indices) :]:
        ax.axis("off")
    axes[0].legend(loc="best")
    for ax in axes[-ncols:]:
        ax.set_xlabel("Temps (s)")
    save_figure(fig, output_dir / "04_top_dofs_focus.png")


def plot_velocity_acceleration_summary(
    ekf_states: np.lib.npyio.NpzFile,
    comparison: np.lib.npyio.NpzFile,
    t: np.ndarray,
    output_dir: Path,
    top_dofs: int,
    airborne_mask: np.ndarray,
) -> None:
    """Compare aussi les vitesses et accelerations quand elles sont disponibles."""
    qdot_ekf_3d = comparison_qdot_ekf_3d(comparison)
    qddot_ekf_3d = comparison_qddot_ekf_3d(comparison)
    if "qdot" not in ekf_states or qdot_ekf_3d is None or "qddot" not in ekf_states or qddot_ekf_3d is None:
        return

    q_names = ekf_states["q_names"]
    qdot_ekf = ekf_states["qdot"]
    qddot_ekf = ekf_states["qddot"]
    qdot_ekf_3d = qdot_ekf_3d
    qddot_ekf_3d = qddot_ekf_3d

    vel_rmse = np.sqrt(np.mean((qdot_ekf - qdot_ekf_3d) ** 2, axis=0))
    acc_rmse = np.sqrt(np.mean((qddot_ekf - qddot_ekf_3d) ** 2, axis=0))
    top_indices = np.argsort(vel_rmse + acc_rmse)[::-1][:top_dofs]

    fig, axes = plt.subplots(len(top_indices), 2, figsize=(14, 3 * len(top_indices)), sharex=True)
    if len(top_indices) == 1:
        axes = np.array([axes])
    for row, dof_idx in enumerate(top_indices):
        add_phase_background(axes[row, 0], t, airborne_mask)
        axes[row, 0].plot(t, qdot_ekf[:, dof_idx], color="#55a868", linewidth=2)
        axes[row, 0].plot(t, qdot_ekf_3d[:, dof_idx], color="#4c72b0", linewidth=1.6)
        axes[row, 0].set_title(f"Vitesse - {q_names[dof_idx]}")
        axes[row, 0].grid(alpha=0.3)

        add_phase_background(axes[row, 1], t, airborne_mask)
        axes[row, 1].plot(t, qddot_ekf[:, dof_idx], color="#c44e52", linewidth=2)
        axes[row, 1].plot(t, qddot_ekf_3d[:, dof_idx], color="#8172b3", linewidth=1.6)
        axes[row, 1].set_title(f"Acceleration - {q_names[dof_idx]}")
        axes[row, 1].grid(alpha=0.3)

    axes[0, 0].legend(["EKF 2D", "EKF 3D"], loc="best")
    axes[-1, 0].set_xlabel("Temps (s)")
    axes[-1, 1].set_xlabel("Temps (s)")
    save_figure(fig, output_dir / "05_velocity_acceleration_focus.png")


def plot_scatter_identity(q_names: np.ndarray, q_ekf: np.ndarray, q_ekf_3d: np.ndarray, rmse: np.ndarray, output_dir: Path, top_dofs: int) -> None:
    """Trace des scatter plots EKF 2D vs EKF 3D avec la diagonale d'identite."""
    top_indices = np.argsort(rmse)[::-1][:top_dofs]
    ncols = 2
    nrows = ceil(len(top_indices) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, dof_idx in zip(axes, top_indices):
        x = q_ekf_3d[:, dof_idx]
        y = q_ekf[:, dof_idx]
        ax.scatter(x, y, color="#4c72b0", s=30, alpha=0.85)
        lo = min(np.min(x), np.min(y))
        hi = max(np.max(x), np.max(y))
        ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1)
        ax.set_title(q_names[dof_idx])
        ax.set_xlabel("EKF 3D")
        ax.set_ylabel("EKF 2D")
        ax.grid(alpha=0.3)
    for ax in axes[len(top_indices) :]:
        ax.axis("off")
    save_figure(fig, output_dir / "06_identity_scatter_top_dofs.png")


def plot_trunk_root_comparison(
    triangulation_points: np.ndarray,
    q_names: np.ndarray,
    q_ekf_2d: np.ndarray,
    q_ekf_3d: np.ndarray,
    fps: float,
    output_dir: Path,
    airborne_mask: np.ndarray,
) -> None:
    """Compare les 3 translations du tronc avec q a gauche et qdot a droite."""
    triangulation_trans, triangulation_rot = compute_trunk_dofs_from_triangulation(triangulation_points)
    triangulation_root = np.hstack((triangulation_trans, triangulation_rot))
    triangulation_root_vel = centered_finite_difference(triangulation_root, 1.0 / fps)

    ekf_2d_root = extract_trunk_root_dofs(q_names, q_ekf_2d)
    ekf_3d_root = extract_trunk_root_dofs(q_names, q_ekf_3d)
    ekf_2d_root_vel = centered_finite_difference(ekf_2d_root, 1.0 / fps)
    ekf_3d_root_vel = centered_finite_difference(ekf_3d_root, 1.0 / fps)

    n_frames = min(
        triangulation_root.shape[0],
        triangulation_root_vel.shape[0],
        ekf_2d_root.shape[0],
        ekf_2d_root_vel.shape[0],
        ekf_3d_root.shape[0],
        ekf_3d_root_vel.shape[0],
    )
    t = time_vector(n_frames, fps)
    labels = ["TransX", "TransY", "TransZ"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for axis_idx, label in enumerate(labels):
        q_ax = axes[axis_idx, 0]
        qdot_ax = axes[axis_idx, 1]
        add_phase_background(q_ax, t, airborne_mask)
        add_phase_background(qdot_ax, t, airborne_mask)

        q_ax.plot(t, triangulation_root[:n_frames, axis_idx], color="#dd8452", linewidth=2.0, label="Triangulation")
        q_ax.plot(t, ekf_2d_root[:n_frames, axis_idx], color="#c44e52", linewidth=1.7, label="EKF 2D")
        q_ax.plot(t, ekf_3d_root[:n_frames, axis_idx], color="#55a868", linewidth=1.7, label="EKF 3D")
        q_ax.set_title(f"{label} - q")
        q_ax.set_ylabel("m")
        q_ax.grid(alpha=0.3)

        qdot_ax.plot(t, triangulation_root_vel[:n_frames, axis_idx], color="#dd8452", linewidth=2.0, label="Triangulation")
        qdot_ax.plot(t, ekf_2d_root_vel[:n_frames, axis_idx], color="#c44e52", linewidth=1.7, label="EKF 2D")
        qdot_ax.plot(t, ekf_3d_root_vel[:n_frames, axis_idx], color="#55a868", linewidth=1.7, label="EKF 3D")
        qdot_ax.set_title(f"{label} - qdot")
        qdot_ax.set_ylabel("m/s")
        qdot_ax.grid(alpha=0.3)

    axes[0, 0].legend(loc="best")
    axes[-1, 0].set_xlabel("Temps (s)")
    axes[-1, 1].set_xlabel("Temps (s)")
    fig.suptitle("Translations de la racine: q a gauche, qdot a droite", y=1.02)
    save_figure(fig, output_dir / "07_root_translations_q_qdot_comparison.png")

    np.savez(
        output_dir / "07_root_translations_q_qdot_comparison.npz",
        labels=np.asarray(labels, dtype=object),
        triangulation_q=triangulation_root[:n_frames, :3],
        ekf_2d_q=ekf_2d_root[:n_frames, :3],
        ekf_3d_q=ekf_3d_root[:n_frames, :3],
        triangulation_qdot=triangulation_root_vel[:n_frames, :3],
        ekf_2d_qdot=ekf_2d_root_vel[:n_frames, :3],
        ekf_3d_qdot=ekf_3d_root_vel[:n_frames, :3],
        time=t,
    )


def plot_trunk_root_velocity_comparison(
    triangulation_points: np.ndarray,
    q_names: np.ndarray,
    q_ekf_2d: np.ndarray,
    q_ekf_3d: np.ndarray,
    fps: float,
    output_dir: Path,
    airborne_mask: np.ndarray,
) -> None:
    """Compare les 3 rotations de la racine avec q a gauche et qdot a droite."""
    triangulation_trans, triangulation_rot = compute_trunk_dofs_from_triangulation(triangulation_points)
    triangulation_root = np.hstack((triangulation_trans, triangulation_rot))
    triangulation_root_vel = centered_finite_difference(triangulation_root, 1.0 / fps)

    ekf_2d_root = extract_trunk_root_dofs(q_names, q_ekf_2d)
    ekf_3d_root = extract_trunk_root_dofs(q_names, q_ekf_3d)
    ekf_2d_root_vel = centered_finite_difference(ekf_2d_root, 1.0 / fps)
    ekf_3d_root_vel = centered_finite_difference(ekf_3d_root, 1.0 / fps)

    n_frames = min(triangulation_root_vel.shape[0], ekf_2d_root_vel.shape[0], ekf_3d_root_vel.shape[0])
    t = time_vector(n_frames, fps)
    labels = ["RotY (salto)", "RotX (tilt)", "RotZ (twist)"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for row, label in enumerate(labels):
        q_idx = 3 + row
        q_ax = axes[row, 0]
        qdot_ax = axes[row, 1]
        add_phase_background(q_ax, t, airborne_mask)
        add_phase_background(qdot_ax, t, airborne_mask)

        q_ax.plot(t, triangulation_root[:n_frames, q_idx], color="#dd8452", linewidth=2.0, label="Triangulation")
        q_ax.plot(t, ekf_2d_root[:n_frames, q_idx], color="#c44e52", linewidth=1.7, label="EKF 2D")
        q_ax.plot(t, ekf_3d_root[:n_frames, q_idx], color="#55a868", linewidth=1.7, label="EKF 3D")
        q_ax.set_title(f"{label} - q")
        q_ax.set_ylabel("rad")
        q_ax.grid(alpha=0.3)

        qdot_ax.plot(t, triangulation_root_vel[:n_frames, q_idx], color="#dd8452", linewidth=2.0, label="Triangulation")
        qdot_ax.plot(t, ekf_2d_root_vel[:n_frames, q_idx], color="#c44e52", linewidth=1.7, label="EKF 2D")
        qdot_ax.plot(t, ekf_3d_root_vel[:n_frames, q_idx], color="#55a868", linewidth=1.7, label="EKF 3D")
        qdot_ax.set_title(f"{label} - qdot")
        qdot_ax.set_ylabel("rad/s")
        qdot_ax.grid(alpha=0.3)

    axes[0, 0].legend(loc="best")
    axes[-1, 0].set_xlabel("Temps (s)")
    axes[-1, 1].set_xlabel("Temps (s)")
    fig.suptitle("Rotations de la racine: q a gauche, qdot a droite", y=1.02)
    save_figure(fig, output_dir / "07b_root_rotations_q_qdot_comparison.png")

    np.savez(
        output_dir / "07b_root_rotations_q_qdot_comparison.npz",
        labels=np.asarray(labels, dtype=object),
        triangulation_q=triangulation_root[:n_frames, 3:6],
        ekf_2d_q=ekf_2d_root[:n_frames, 3:6],
        ekf_3d_q=ekf_3d_root[:n_frames, 3:6],
        triangulation_qdot=triangulation_root_vel[:n_frames, 3:6],
        ekf_2d_qdot=ekf_2d_root_vel[:n_frames, 3:6],
        ekf_3d_qdot=ekf_3d_root_vel[:n_frames, 3:6],
        time=t,
    )


def plot_acc_vs_dyn(q_names: np.ndarray, q_acc: np.ndarray, q_dyn: np.ndarray, t: np.ndarray, output_dir: Path, top_dofs: int) -> None:
    """Compare directement les variantes EKF 2D ACC et DYN."""
    diff = np.sqrt(np.mean((q_acc - q_dyn) ** 2, axis=0))
    top_indices = np.argsort(diff)[::-1][:top_dofs]
    ncols = 2
    nrows = ceil(len(top_indices) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, dof_idx in zip(axes, top_indices):
        ax.plot(t, q_acc[:, dof_idx], color="#c44e52", linewidth=1.8, label="EKF 2D ACC")
        ax.plot(t, q_dyn[:, dof_idx], color="#8172b3", linewidth=1.8, label="EKF 2D DYN")
        ax.set_title(f"{q_names[dof_idx]} | RMS diff={diff[dof_idx]:.4f}")
        ax.grid(alpha=0.3)
    for ax in axes[len(top_indices) :]:
        ax.axis("off")
    axes[0].legend(loc="best")
    save_figure(fig, output_dir / "08_ekf2d_acc_vs_dyn.png")


def plot_qdot_filter_vs_finite_difference(
    q_names: np.ndarray,
    q_ekf_2d: np.ndarray,
    qdot_ekf_2d: np.ndarray,
    q_ekf_3d: np.ndarray,
    qdot_ekf_3d: np.ndarray,
    fps: float,
    output_dir: Path,
    top_dofs: int,
    airborne_mask: np.ndarray,
) -> None:
    """Compare les qdot bruts des filtres a des qdot recalcules sur les 3 rotations de la racine."""
    dt = 1.0 / fps
    qdot_ekf_2d_fd = centered_finite_difference(q_ekf_2d, dt)
    qdot_ekf_3d_fd = centered_finite_difference(q_ekf_3d, dt)

    n_frames = min(
        q_ekf_2d.shape[0],
        qdot_ekf_2d.shape[0],
        qdot_ekf_2d_fd.shape[0],
        q_ekf_3d.shape[0],
        qdot_ekf_3d.shape[0],
        qdot_ekf_3d_fd.shape[0],
    )
    t = time_vector(n_frames, fps)

    diff_ekf_2d = np.sqrt(np.nanmean((qdot_ekf_2d[:n_frames] - qdot_ekf_2d_fd[:n_frames]) ** 2, axis=0))
    diff_ekf_3d = np.sqrt(np.nanmean((qdot_ekf_3d[:n_frames] - qdot_ekf_3d_fd[:n_frames]) ** 2, axis=0))
    name_to_index = {str(name): idx for idx, name in enumerate(q_names)}
    root_rotation_indices = [name_to_index[name] for name in TRUNK_ROTATION_NAMES if name in name_to_index]
    root_rotation_labels = [name for name in TRUNK_ROTATION_NAMES if name in name_to_index]
    if not root_rotation_indices:
        return

    fig, axes = plt.subplots(len(root_rotation_indices), 2, figsize=(15, 3.4 * len(root_rotation_indices)), sharex=True)
    if len(root_rotation_indices) == 1:
        axes = np.array([axes])

    for row, (dof_idx, dof_label) in enumerate(zip(root_rotation_indices, root_rotation_labels)):
        add_phase_background(axes[row, 0], t, airborne_mask)
        axes[row, 0].plot(t, qdot_ekf_2d[:n_frames, dof_idx], color="#c44e52", linewidth=1.8, label="EKF 2D qdot brut")
        axes[row, 0].plot(
            t,
            qdot_ekf_2d_fd[:n_frames, dof_idx],
            color="#dd8452",
            linewidth=1.5,
            label="EKF 2D qdot diff finie sur q",
        )
        axes[row, 0].set_title(f"EKF 2D | {dof_label} | RMS diff={diff_ekf_2d[dof_idx]:.2f}")
        axes[row, 0].grid(alpha=0.3)
        axes[row, 0].set_ylabel("rad/s")

        add_phase_background(axes[row, 1], t, airborne_mask)
        axes[row, 1].plot(t, qdot_ekf_3d[:n_frames, dof_idx], color="#55a868", linewidth=1.8, label="EKF 3D qdot brut")
        axes[row, 1].plot(
            t,
            qdot_ekf_3d_fd[:n_frames, dof_idx],
            color="#4c72b0",
            linewidth=1.5,
            label="EKF 3D qdot diff finie sur q",
        )
        axes[row, 1].set_title(f"EKF 3D | {dof_label} | RMS diff={diff_ekf_3d[dof_idx]:.2f}")
        axes[row, 1].grid(alpha=0.3)

    axes[0, 0].legend(loc="best")
    axes[0, 1].legend(loc="best")
    axes[-1, 0].set_xlabel("Temps (s)")
    axes[-1, 1].set_xlabel("Temps (s)")
    fig.suptitle("qdot racine: sortie brute EKF vs difference finie sur q", y=1.01)
    save_figure(fig, output_dir / "05b_qdot_filter_vs_finite_difference.png")

    np.savez(
        output_dir / "05b_qdot_filter_vs_finite_difference.npz",
        q_names=np.asarray(root_rotation_labels, dtype=object),
        time=t,
        qdot_ekf_2d_raw=qdot_ekf_2d[:n_frames, root_rotation_indices],
        qdot_ekf_2d_fd=qdot_ekf_2d_fd[:n_frames, root_rotation_indices],
        qdot_ekf_3d_raw=qdot_ekf_3d[:n_frames, root_rotation_indices],
        qdot_ekf_3d_fd=qdot_ekf_3d_fd[:n_frames, root_rotation_indices],
        rms_diff_ekf_2d=diff_ekf_2d[root_rotation_indices],
        rms_diff_ekf_3d=diff_ekf_3d[root_rotation_indices],
    )


def plot_root_rotations_no_unwrap(
    q_names: np.ndarray,
    q_ekf_2d: np.ndarray,
    q_ekf_3d: np.ndarray,
    fps: float,
    output_dir: Path,
    airborne_mask: np.ndarray,
) -> None:
    """Compare les rotations de la racine apres re-extraction Euler sans unwrap."""
    ekf_2d_root = extract_trunk_root_dofs_no_unwrap(q_names, q_ekf_2d)
    ekf_3d_root = extract_trunk_root_dofs_no_unwrap(q_names, q_ekf_3d)
    n_frames = min(ekf_2d_root.shape[0], ekf_3d_root.shape[0])
    t = time_vector(n_frames, fps)
    labels = ["RotY (salto)", "RotX (tilt)", "RotZ (twist)"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for row, label in enumerate(labels):
        ax = axes[row]
        add_phase_background(ax, t, airborne_mask)
        q_idx = 3 + row
        ax.plot(t, ekf_2d_root[:n_frames, q_idx], color="#c44e52", linewidth=1.8, label="EKF 2D")
        ax.plot(t, ekf_3d_root[:n_frames, q_idx], color="#55a868", linewidth=1.8, label="EKF 3D")
        ax.set_title(f"{label} sans unwrap")
        ax.set_ylabel("rad")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Temps (s)")
    fig.suptitle("Rotations de la racine via matrice puis Euler, sans unwrap", y=1.01)
    save_figure(fig, output_dir / "07c_root_rotations_no_unwrap.png")

    np.savez(
        output_dir / "07c_root_rotations_no_unwrap.npz",
        labels=np.asarray(labels, dtype=object),
        time=t,
        ekf_2d_root_no_unwrap=ekf_2d_root[:n_frames, 3:6],
        ekf_3d_root_no_unwrap=ekf_3d_root[:n_frames, 3:6],
    )


def plot_airborne_timeline(t: np.ndarray, airborne_mask: np.ndarray, output_dir: Path) -> None:
    """Exporte une figure simple indiquant toile vs air."""
    fig, ax = plt.subplots(figsize=(14, 1.8))
    y = airborne_mask[: len(t)].astype(float)
    ax.step(t, y, where="post", color="black", linewidth=1.6)
    add_phase_background(ax, t, airborne_mask)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Toile", "Air"])
    ax.set_xlabel("Temps (s)")
    ax.set_title("Detection phase aerienne / toile")
    ax.set_ylim(-0.2, 1.2)
    ax.grid(alpha=0.2, axis="x")
    save_figure(fig, output_dir / "09_airborne_timeline.png")


def compute_ekf2d_camera_weights(
    input_dir: Path,
    calib_path: Path,
    keypoints_path: Path,
    summary: dict,
    fps: float,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, list[str]]:
    """Reconstruit les poids de mesure utilises par l'EKF 2D pour chaque camera.

    Le poids scalaire associe a une observation 2D suit l'inverse de la variance
    de mesure:
        w = 1 / sigma^2
          = effective_confidence^2 / (16 * measurement_noise_scale)
    avec:
        effective_confidence = score_pose * blended_coherence
        blended_coherence = floor + (1 - floor) * coherence
    """
    triangulation = np.load(input_dir / "triangulation_pose2sim_like.npz", allow_pickle=True)
    calib = load_calibrations(calib_path)
    pose_data = load_pose_data(keypoints_path, calib)

    reconstruction_frames = np.asarray(triangulation["frames"], dtype=int)
    frame_to_idx = {int(frame): idx for idx, frame in enumerate(np.asarray(pose_data.frames, dtype=int))}
    selected_raw_idx = np.array([frame_to_idx[int(frame)] for frame in reconstruction_frames if int(frame) in frame_to_idx], dtype=int)
    if selected_raw_idx.size != reconstruction_frames.size:
        raise ValueError("Les frames du cache de triangulation et des detections 2D ne correspondent pas.")

    scores = np.asarray(pose_data.scores[:, selected_raw_idx], dtype=float)
    coherence = np.asarray(triangulation["multiview_coherence"], dtype=float).transpose(2, 0, 1)
    floor = float(summary.get("coherence_confidence_floor", DEFAULT_COHERENCE_CONFIDENCE_FLOOR))
    measurement_noise_scale = float(summary.get("measurement_noise_scale", 1.0))
    blended_coherence = floor + (1.0 - floor) * coherence
    effective_confidence = scores * blended_coherence
    valid = effective_confidence > 1e-3
    weights = np.zeros_like(effective_confidence)
    weights[valid] = (effective_confidence[valid] ** 2) / (16.0 * measurement_noise_scale)

    weight_sum_per_camera = np.sum(weights, axis=2)
    total_per_frame = np.sum(weight_sum_per_camera, axis=0, keepdims=True)
    weight_fraction_per_camera = np.divide(
        weight_sum_per_camera,
        total_per_frame,
        out=np.zeros_like(weight_sum_per_camera),
        where=total_per_frame > 0,
    )
    row_labels = [f"{cam}:{kp}" for cam in pose_data.camera_names for kp in COCO17]
    weight_by_camera_keypoint = weights.reshape(weights.shape[0] * weights.shape[2], weights.shape[1])
    return (
        reconstruction_frames / float(fps),
        weight_sum_per_camera,
        list(pose_data.camera_names),
        weight_fraction_per_camera,
        weight_by_camera_keypoint,
        row_labels,
    )


def plot_ekf2d_camera_weights(
    t: np.ndarray,
    weight_sum_per_camera: np.ndarray,
    weight_fraction_per_camera: np.ndarray,
    camera_names: list[str],
    airborne_mask: np.ndarray,
    output_dir: Path,
) -> None:
    """Affiche les poids absolus et relatifs des cameras dans l'EKF 2D."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(camera_names)))

    for ax in axes:
        add_phase_background(ax, t, airborne_mask)

    for cam_idx, cam_name in enumerate(camera_names):
        axes[0].plot(t, weight_sum_per_camera[cam_idx], linewidth=1.6, color=colors[cam_idx], label=cam_name)
    axes[0].set_ylabel("Somme des poids EKF")
    axes[0].set_title("Poids de mesure utilises par camera dans l'EKF 2D")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True, fontsize=8)

    axes[1].stackplot(t, weight_fraction_per_camera, labels=camera_names, colors=colors, alpha=0.85)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Fraction relative")
    axes[1].set_xlabel("Temps (s)")
    axes[1].set_title("Contribution relative de chaque camera a la correction EKF 2D")
    axes[1].grid(alpha=0.2)

    save_figure(fig, output_dir / "10_ekf2d_camera_weights.png")


def plot_ekf2d_camera_keypoint_weight_heatmap(
    t: np.ndarray,
    weight_by_camera_keypoint: np.ndarray,
    row_labels: list[str],
    output_dir: Path,
) -> None:
    """Affiche une heatmap frame par frame des poids EKF par camera:keypoint."""
    fig, ax = plt.subplots(figsize=(22, 8))
    vmax = np.nanpercentile(weight_by_camera_keypoint, 99.0)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    image = ax.imshow(
        weight_by_camera_keypoint,
        aspect="auto",
        interpolation="nearest",
        cmap="gray_r",
        extent=(t[0], t[-1] if len(t) else 0.0, weight_by_camera_keypoint.shape[0], 0),
        vmin=0.0,
        vmax=vmax,
    )
    if row_labels:
        step = max(1, len(row_labels) // 30)
        yticks = np.arange(0, len(row_labels), step)
        ax.set_yticks(yticks + 0.5)
        ax.set_yticklabels([row_labels[idx] for idx in yticks], fontsize=8)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Camera : keypoint")
    ax.set_title("Poids EKF 2D frame par frame (camera:keypoint)")
    cbar = fig.colorbar(image, ax=ax, pad=0.01)
    cbar.set_label("Poids utilise")
    save_figure(fig, output_dir / "11_ekf2d_camera_keypoint_weights_heatmap.png")


def main() -> None:
    """Point d'entree CLI du script de figures."""
    args = parse_args()
    output_dir = ensure_dir(args.output_dir or (args.input_dir / "figures"))
    data = load_data(args.input_dir)
    ekf = data["ekf"]
    comparison = data["comparison"]
    triangulation = data["triangulation"]
    summary = data["summary"]

    q_names = comparison["q_names"]
    q_ekf = comparison["q_ekf"]
    q_ekf_3d = comparison_q_ekf_3d(comparison)
    rmse = comparison["rmse_per_dof"]
    mae = comparison["mae_per_dof"]
    t = time_vector(q_ekf.shape[0], args.fps)
    flight_threshold = float(summary.get("flight_height_threshold_m", 0.0))
    flight_min_consecutive = int(summary.get("flight_min_consecutive_frames", 1))
    airborne_mask = compute_airborne_mask(np.asarray(triangulation["points_3d"], dtype=float), flight_threshold, flight_min_consecutive)

    if args.ekf_2d_variant == "dyn":
        comparison_dyn = load_optional_npz(args.input_dir / "kalman_comparison_dyn.npz")
        if comparison_dyn is None:
            raise FileNotFoundError("kalman_comparison_dyn.npz introuvable.")
        q_ekf = np.asarray(ekf["q_ekf_2d_dyn"], dtype=float)
        comparison = comparison_dyn
        q_names = comparison["q_names"]
        q_ekf_3d = comparison_q_ekf_3d(comparison)
        rmse = comparison["rmse_per_dof"]
        mae = comparison["mae_per_dof"]
        t = time_vector(q_ekf.shape[0], args.fps)

    plot_summary_metrics(summary, output_dir)
    plot_airborne_timeline(t, airborne_mask, output_dir)
    plot_error_bars(q_names, rmse, mae, output_dir)
    plot_error_heatmap(q_names, q_ekf, q_ekf_3d, output_dir)
    plot_global_trace_pages(q_names, q_ekf, q_ekf_3d, t, output_dir, airborne_mask)
    plot_top_dof_focus(q_names, q_ekf, q_ekf_3d, rmse, t, output_dir, args.top_dofs, airborne_mask)
    plot_velocity_acceleration_summary(ekf, comparison, t, output_dir, args.top_dofs, airborne_mask)
    qdot_ekf_3d = comparison_qdot_ekf_3d(comparison)
    if "qdot" in ekf and qdot_ekf_3d is not None:
        plot_qdot_filter_vs_finite_difference(
            q_names,
            np.asarray(q_ekf, dtype=float),
            np.asarray(ekf["qdot"], dtype=float),
            np.asarray(q_ekf_3d, dtype=float),
            np.asarray(qdot_ekf_3d, dtype=float),
            args.fps,
            output_dir,
            args.top_dofs,
            airborne_mask,
        )
    plot_scatter_identity(q_names, q_ekf, q_ekf_3d, rmse, output_dir, args.top_dofs)
    plot_trunk_root_comparison(np.asarray(triangulation["points_3d"], dtype=float), q_names, q_ekf, q_ekf_3d, args.fps, output_dir, airborne_mask)
    plot_trunk_root_velocity_comparison(
        np.asarray(triangulation["points_3d"], dtype=float),
        q_names,
        q_ekf,
        q_ekf_3d,
        args.fps,
        output_dir,
        airborne_mask,
    )
    plot_root_rotations_no_unwrap(
        q_names,
        q_ekf,
        q_ekf_3d,
        args.fps,
        output_dir,
        airborne_mask,
    )
    try:
        (
            t_weights,
            weight_sum_per_camera,
            camera_names,
            weight_fraction_per_camera,
            weight_by_camera_keypoint,
            row_labels,
        ) = compute_ekf2d_camera_weights(
            args.input_dir,
            args.calib,
            args.keypoints,
            summary,
            args.fps,
        )
        plot_ekf2d_camera_weights(
            t_weights,
            weight_sum_per_camera,
            weight_fraction_per_camera,
            camera_names,
            airborne_mask,
            output_dir,
        )
        plot_ekf2d_camera_keypoint_weight_heatmap(
            t_weights,
            weight_by_camera_keypoint,
            row_labels,
            output_dir,
        )
    except Exception as exc:
        print(f"Figure des poids camera EKF 2D non generee: {exc}")
    if args.ekf_2d_variant == "both" and "q_ekf_2d_acc" in ekf and "q_ekf_2d_dyn" in ekf:
        q_acc = np.asarray(ekf["q_ekf_2d_acc"], dtype=float)
        q_dyn = np.asarray(ekf["q_ekf_2d_dyn"], dtype=float)
        n = min(q_acc.shape[0], q_dyn.shape[0])
        plot_acc_vs_dyn(q_names, q_acc[:n], q_dyn[:n], time_vector(n, args.fps), output_dir, args.top_dofs)

    print(f"Figures exportees dans: {output_dir}")


if __name__ == "__main__":
    main()
