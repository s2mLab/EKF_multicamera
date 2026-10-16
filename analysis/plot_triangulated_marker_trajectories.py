#!/usr/bin/env python3
"""Trace les trajectoires 3D triangulees de tous les marqueurs.

La figure comporte trois sous-graphiques:
- X(t)
- Y(t)
- Z(t)

Chaque courbe correspond a un marqueur COCO17 triangule. C'est utile pour
diagnostiquer un probleme de repere, de hauteur absolue ou de detection de
phase aerienne.

Le script supporte aussi un mode interactif Matplotlib: on peut cliquer sur
la legende pour masquer/afficher un marqueur sur les trois subplots en meme
temps. Cela evite de multiplier les exports statiques quand on veut inspecter
un petit sous-ensemble de courbes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOCAL_MPLCONFIG = ROOT / ".cache" / "matplotlib"
LOCAL_MPLCONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_MPLCONFIG))

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_TRIANGULATION = Path("output/vitpose_full/triangulation_pose2sim_like.npz")
DEFAULT_OUTPUT = Path("output/vitpose_full/triangulated_marker_trajectories.png")
DEFAULT_FPS = 120.0
DEFAULT_SUMMARY = Path("output/vitpose_full/summary.json")


def parse_args() -> argparse.Namespace:
    """Construit l'interface CLI du script."""
    parser = argparse.ArgumentParser(description="Trace les trajectoires 3D triangulees de tous les marqueurs.")
    parser.add_argument("--triangulation", type=Path, default=DEFAULT_TRIANGULATION, help="NPZ de triangulation.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Figure PNG de sortie.")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Frequence pour l'axe temporel.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Summary JSON pour recuperer les parametres de phase aerienne.",
    )
    parser.add_argument(
        "--flight-height-threshold-m",
        type=float,
        default=None,
        help="Seuil vertical pour le critere AIR/TOILE. Si absent, on lit summary.json puis fallback a 0.",
    )
    parser.add_argument(
        "--flight-min-consecutive-frames",
        type=int,
        default=None,
        help="Nombre minimal de frames consecutives au-dessus du seuil. Si absent, on lit summary.json puis fallback a 1.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Ouvre une fenetre interactive avec legende cliquable au lieu de seulement sauver un PNG.",
    )
    return parser.parse_args()


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Sauvegarde la figure sur disque."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")


def load_flight_parameters(summary_path: Path) -> tuple[float, int]:
    """Charge les parametres AIR/TOILE depuis le summary du pipeline si disponible."""
    if not summary_path.exists():
        return 0.0, 1
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    return float(summary.get("flight_height_threshold_m", 0.0)), int(summary.get("flight_min_consecutive_frames", 1))


def compute_airborne_mask(
    points_3d: np.ndarray, threshold_m: float, min_consecutive_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruit le masque AIR/TOILE et la trajectoire min_z associee.

    Le critere est le meme que dans le pipeline:
    - tous les marqueurs triangules valides doivent verifier `z > threshold_m`
    - cette condition doit etre vraie sur `min_consecutive_frames` frames consecutives
    """
    z_values = points_3d[:, :, 2]
    valid = np.isfinite(z_values)
    min_z = np.full(points_3d.shape[0], np.nan, dtype=float)
    for frame_idx in range(points_3d.shape[0]):
        if np.any(valid[frame_idx]):
            min_z[frame_idx] = np.min(z_values[frame_idx, valid[frame_idx]])

    above = np.all(z_values > threshold_m, axis=1)
    above &= np.all(valid, axis=1)

    mask = np.zeros(points_3d.shape[0], dtype=bool)
    consec = 0
    for frame_idx, flag in enumerate(above):
        consec = consec + 1 if flag else 0
        if consec >= max(1, min_consecutive_frames):
            mask[frame_idx] = True
    return mask, min_z


def add_phase_background(ax: plt.Axes, t: np.ndarray, airborne_mask: np.ndarray) -> None:
    """Ajoute un fond colore pour visualiser les phases toile/air."""
    n = min(len(t), len(airborne_mask))
    if n == 0:
        return
    starts = [0]
    for idx in range(1, n):
        if airborne_mask[idx] != airborne_mask[idx - 1]:
            starts.append(idx)
    starts.append(n)
    for start, stop in zip(starts[:-1], starts[1:]):
        is_air = airborne_mask[start]
        color = "#f6d55c" if is_air else "#d9edf7"
        alpha = 0.12 if is_air else 0.08
        ax.axvspan(t[start], t[stop - 1], color=color, alpha=alpha, zorder=0)


def register_clickable_legend(
    fig: plt.Figure,
    legend_ax: plt.Axes,
    marker_lines: dict[str, list[Any]],
    extra_lines: dict[str, list[Any]],
) -> None:
    """Rend la legende cliquable pour afficher/masquer les courbes.

    Un nom de marqueur pilote les trois courbes X/Y/Z correspondantes.
    Les lignes auxiliaires (`min_z`, `seuil`) sont aussi pilotables.
    """
    handles: list[Any] = []
    labels: list[str] = []
    grouped_lines: dict[str, list[Any]] = {}

    for name, lines in marker_lines.items():
        if not lines:
            continue
        handles.append(lines[0])
        labels.append(name)
        grouped_lines[name] = lines

    for name, lines in extra_lines.items():
        if not lines:
            continue
        handles.append(lines[0])
        labels.append(name)
        grouped_lines[name] = lines

    legend = legend_ax.legend(
        handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True, fontsize=8, ncol=1
    )
    legend_ax._clickable_legend = legend  # type: ignore[attr-defined]

    proxy_to_label: dict[Any, str] = {}
    for proxy, label in zip(legend.legend_handles, labels):
        proxy.set_picker(True)
        proxy.set_pickradius(8)
        proxy_to_label[proxy] = label

    def on_pick(event) -> None:
        proxy = event.artist
        label = proxy_to_label.get(proxy)
        if label is None:
            return
        lines = grouped_lines[label]
        currently_visible = any(line.get_visible() for line in lines)
        new_visible = not currently_visible
        for line in lines:
            line.set_visible(new_visible)
        proxy.set_alpha(1.0 if new_visible else 0.2)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)


def main() -> None:
    """Charge la triangulation et exporte la figure des trajectoires 3D."""
    args = parse_args()
    data = np.load(args.triangulation, allow_pickle=True)
    points_3d = np.asarray(data["points_3d"], dtype=float)
    frames = np.asarray(data["frames"], dtype=int) if "frames" in data else np.arange(points_3d.shape[0], dtype=int)
    keypoint_names = np.asarray(data["keypoint_names"], dtype=object)
    t = frames / args.fps
    default_threshold, default_min_consecutive = load_flight_parameters(args.summary)
    flight_threshold = (
        default_threshold if args.flight_height_threshold_m is None else float(args.flight_height_threshold_m)
    )
    flight_min_consecutive = (
        default_min_consecutive
        if args.flight_min_consecutive_frames is None
        else int(args.flight_min_consecutive_frames)
    )
    airborne_mask, min_z = compute_airborne_mask(points_3d, flight_threshold, flight_min_consecutive)

    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
    axis_labels = ["X (m)", "Y (m)", "Z (m)"]
    axis_titles = [
        "Trajectoires 3D triangulees - axe X",
        "Trajectoires 3D triangulees - axe Y",
        "Trajectoires 3D triangulees - axe Z",
    ]

    colors = plt.cm.tab20(np.linspace(0, 1, len(keypoint_names)))
    for ax in axes:
        add_phase_background(ax, t, airborne_mask)
    marker_lines: dict[str, list[Any]] = {}
    for kp_idx, (kp_name, color) in enumerate(zip(keypoint_names, colors)):
        kp_lines: list[Any] = []
        for axis_idx, ax in enumerate(axes):
            (line,) = ax.plot(
                t, points_3d[:, kp_idx, axis_idx], label=str(kp_name), linewidth=1.3, color=color, alpha=0.9
            )
            kp_lines.append(line)
            ax.set_ylabel(axis_labels[axis_idx])
            ax.set_title(axis_titles[axis_idx])
            ax.grid(alpha=0.25)
        marker_lines[str(kp_name)] = kp_lines

    # Sur l'axe Z, on montre explicitement le critere utilise par le pipeline:
    # le minimum vertical parmi tous les marqueurs, compare au seuil.
    (min_z_line,) = axes[2].plot(t, min_z, color="black", linewidth=2.2, label="min_z marqueurs")
    threshold_line = axes[2].axhline(
        flight_threshold,
        color="black",
        linewidth=1.2,
        linestyle="--",
        alpha=0.8,
        label=f"seuil AIR/TOILE = {flight_threshold:.3f} m",
    )
    axes[2].text(
        0.995,
        0.02,
        f"AIR si tous les marqueurs ont z > {flight_threshold:.3f} m pendant {flight_min_consecutive} frame(s)",
        transform=axes[2].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

    axes[-1].set_xlabel("Temps (s)")
    register_clickable_legend(
        fig,
        axes[0],
        marker_lines,
        {
            "min_z marqueurs": [min_z_line],
            f"seuil AIR/TOILE = {flight_threshold:.3f} m": [threshold_line],
        },
    )
    fig.suptitle("Trajectoires 3D de tous les marqueurs triangules avec critere AIR/TOILE", y=1.01)
    save_figure(fig, args.output)
    if args.interactive:
        plt.show()
    plt.close(fig)
    print(f"Figure exportee dans: {args.output}")


if __name__ == "__main__":
    main()
