#!/usr/bin/env python3
"""Animation 3D comparative entre plusieurs reconstructions stick figure.

Le script compare:
- une reconstruction 3D issue de `triangulation_pose2sim_like.npz`
- une reconstruction 3D Pose2Sim au format `.trc`
- la cinematique du Kalman marqueurs classique de `biorbd`
- la cinematique de l'EKF multi-vues utilisant les observations 2D

Les deux stick figures sont affichees simultanement avec des couleurs
distinctes pour mettre en evidence les ecarts entre les reconstructions.
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
from matplotlib.animation import FuncAnimation, PillowWriter
from judging.trampoline_displacement import BED_X_MAX, BED_Y_MAX, TRAMPOLINE_GEOMETRY, X_MAX, Y_MAX

from reconstruction.reconstruction_dataset import (
    align_array_to_frames,
    load_bundle_entries,
    preferred_master_name,
    preferred_triangulation_name,
    reconstruction_color,
    reconstruction_label,
    resolve_requested_names,
)

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
TRC_TO_COCO = {
    "Nose": "nose",
    "L Eye": "left_eye",
    "R Eye": "right_eye",
    "L Ear": "left_ear",
    "R Ear": "right_ear",
    "L Shoulder": "left_shoulder",
    "R Shoulder": "right_shoulder",
    "L Elbow": "left_elbow",
    "R Elbow": "right_elbow",
    "L Wrist": "left_wrist",
    "R Wrist": "right_wrist",
    "L Hip": "left_hip",
    "R Hip": "right_hip",
    "L Knee": "left_knee",
    "R Knee": "right_knee",
    "L Ankle": "left_ankle",
    "R Ankle": "right_ankle",
}

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


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < 1e-12:
        return np.full(3, np.nan)
    return vector / norm


def parse_args() -> argparse.Namespace:
    """Construit l'interface CLI."""
    parser = argparse.ArgumentParser(description="Animation 3D comparative de deux stick figures.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Dossier dataset contenant des sous-dossiers de bundles de reconstruction.",
    )
    parser.add_argument(
        "--triangulation",
        type=Path,
        default=Path("outputs") / "vitpose_full" / "triangulation_pose2sim_like.npz",
        help="NPZ de reconstruction 3D issue du pipeline local",
    )
    parser.add_argument(
        "--pose2sim-trc",
        type=Path,
        default=Path("inputs/trc/1_partie_0429.trc"),
        help="TRC produit par Pose2Sim",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "vitpose_full" / "multi_stick_comparison.gif",
        help="GIF de sortie",
    )
    parser.add_argument(
        "--ekf-states",
        type=Path,
        default=Path("outputs") / "vitpose_full" / "ekf_states.npz",
        help="Etats du nouvel EKF multi-vues",
    )
    parser.add_argument(
        "--kalman-comparison",
        type=Path,
        default=Path("outputs") / "vitpose_full" / "kalman_comparison.npz",
        help="Fichier de comparaison contenant `q_ekf_3d`",
    )
    parser.add_argument(
        "--biomod",
        type=Path,
        default=Path("outputs") / "vitpose_full" / "vitpose_chain.bioMod",
        help="Modele biorbd utilise pour convertir les `q` en positions de keypoints",
    )
    parser.add_argument(
        "--triangulation-fps", type=float, default=DEFAULT_CAMERA_FPS, help="Frequence supposee pour le NPZ triangule"
    )
    parser.add_argument("--fps", type=float, default=15.0, help="Frequence d'affichage du GIF")
    parser.add_argument("--stride", type=int, default=3, help="Sous-echantillonnage temporel")
    parser.add_argument("--marker-size", type=float, default=12.0, help="Taille des marqueurs affiches dans le GIF")
    parser.add_argument(
        "--framing",
        choices=("tight", "full"),
        default="full",
        help="Cadrage de la vue 3D: adapte a chaque frame ou fixe sur toute la sequence.",
    )
    parser.add_argument(
        "--show", nargs="+", default=None, help="Liste des reconstructions a afficher dans l'animation."
    )
    parser.add_argument(
        "--show-trunk-frames",
        action="store_true",
        help="Affiche les reperes locaux du tronc pour chaque reconstruction.",
    )
    parser.add_argument(
        "--show-trampoline",
        action="store_true",
        help="Affiche un contour simplifié du trampoline dans la vue 3D.",
    )
    parser.add_argument(
        "--align-root",
        action="store_true",
        help="Aligne la reconstruction Pose2Sim sur la triangulation via le centre du bassin au premier frame commun",
    )
    return parser.parse_args()


def load_triangulation(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Charge la reconstruction 3D locale et retourne points + temps."""
    data = np.load(npz_path, allow_pickle=True)
    points = data["points_3d"]
    frames = data["frames"] if "frames" in data else np.arange(points.shape[0])
    return points, frames


def load_flight_parameters(triangulation_path: Path) -> tuple[float, int]:
    """Lit les parametres de detection air/toile depuis `summary.json` si present."""
    summary_path = triangulation_path.parent / "summary.json"
    if not summary_path.exists():
        return 0.0, 1
    with summary_path.open("r") as f:
        summary = json.load(f)
    return float(summary.get("flight_height_threshold_m", 0.0)), int(summary.get("flight_min_consecutive_frames", 1))


def load_flight_parameters_from_summary(summary: dict[str, object]) -> tuple[float, int]:
    """Lit les parametres air/toile depuis le resume d'un bundle."""
    return float(summary.get("flight_height_threshold_m", 0.0)), int(summary.get("flight_min_consecutive_frames", 1))


def compute_airborne_mask(points_3d: np.ndarray, threshold_m: float, min_consecutive_frames: int) -> np.ndarray:
    """Reconstruit le masque air/toile a partir des points triangules."""
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
    """Charge les trajectoires articulaires des deux filtres cinematiques."""
    ekf = np.load(ekf_states_path, allow_pickle=True)
    comparison = np.load(kalman_comparison_path, allow_pickle=True)
    q_ekf_3d = comparison["q_ekf_3d"] if "q_ekf_3d" in comparison else comparison["q_biorbd_kalman"]
    q_ekf_2d_acc = ekf["q_ekf_2d_acc"] if "q_ekf_2d_acc" in ekf else ekf["q"]
    q_ekf_2d_dyn = ekf["q_ekf_2d_dyn"] if "q_ekf_2d_dyn" in ekf else None
    q_ekf_2d_flip_acc = ekf["q_ekf_2d_flip_acc"] if "q_ekf_2d_flip_acc" in ekf else None
    q_ekf_2d_flip_dyn = ekf["q_ekf_2d_flip_dyn"] if "q_ekf_2d_flip_dyn" in ekf else None
    return q_ekf_2d_acc, q_ekf_3d, q_ekf_2d_dyn, q_ekf_2d_flip_acc, q_ekf_2d_flip_dyn


def parse_trc(trc_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Parse un fichier TRC Pose2Sim vers un tableau `(n_frames, 17, 3)`."""
    with trc_path.open("r") as f:
        lines = [line.rstrip("\n") for line in f]

    metadata_keys = lines[1].split("\t")
    metadata_values = lines[2].split("\t")
    metadata = {k: v for k, v in zip(metadata_keys, metadata_values)}
    data_rate = float(metadata["DataRate"])

    marker_labels = [label for label in lines[3].split("\t")[2:] if label]
    raw = []
    for line in lines[5:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        parts = parts[: 2 + 3 * len(marker_labels)]
        raw.append(parts)

    frames = np.asarray([int(row[0]) for row in raw], dtype=int)
    time = np.asarray([float(row[1]) for row in raw], dtype=float)
    xyz_values = np.asarray(
        [[float(value) if value != "" else np.nan for value in row[2:]] for row in raw],
        dtype=float,
    )

    points = np.full((len(frames), len(COCO17), 3), np.nan)
    for marker_idx, marker_name in enumerate(marker_labels):
        if marker_name not in TRC_TO_COCO:
            continue
        coco_name = TRC_TO_COCO[marker_name]
        coco_idx = KP_INDEX[coco_name]
        points[:, coco_idx, :] = xyz_values[:, 3 * marker_idx : 3 * marker_idx + 3]
    return points, time, data_rate


def resample_points(points: np.ndarray, source_time: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    """Resample une trajectoire 3D marker-par-marker sur une nouvelle grille temporelle."""
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
    """Applique la cinematique directe du modele `biorbd` pour reconstruire les keypoints.

    Les marqueurs du modele portent les memes noms que les keypoints COCO17,
    ce qui permet de reconstruire directement un tenseur `(n_frames, 17, 3)`.
    """
    import biorbd

    model = biorbd.Model(str(biomod_path))
    marker_names = [name.to_string() for name in model.markerNames()]
    points = np.full((q_series.shape[0], len(COCO17), 3), np.nan)

    for frame_idx, q in enumerate(q_series):
        for marker_name, marker in zip(marker_names, model.markers(q)):
            if marker_name in KP_INDEX:
                points[frame_idx, KP_INDEX[marker_name], :] = marker.to_array()
    return points


def resolve_dataset_biomod(dataset_dir: Path, biomod_path: Path | None) -> Path | None:
    """Trouve un biomod utilisable pour les bundles bases sur `q`."""
    if biomod_path is not None and biomod_path.exists():
        return biomod_path
    candidates = sorted(dataset_dir.glob("*/vitpose_chain.bioMod"))
    return candidates[0] if candidates else None


def load_dataset_reconstructions(
    dataset_dir: Path, biomod_path: Path | None = None
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Charge toutes les reconstructions 3D disponibles depuis un dossier dataset."""
    entries = load_bundle_entries(dataset_dir)
    point_entries = [
        entry
        for entry in entries
        if np.asarray(entry["points_3d"]).ndim == 3 and np.asarray(entry["points_3d"]).shape[1] == len(COCO17)
    ]
    if not point_entries:
        raise ValueError(f"Aucune reconstruction bundle 3D disponible dans {dataset_dir}")

    available_names = [str(entry["name"]) for entry in point_entries]
    master_name = preferred_master_name(available_names)
    if master_name is None:
        raise ValueError(f"Aucune reconstruction exploitable trouvee dans {dataset_dir}")
    master_entry = next(entry for entry in point_entries if entry["name"] == master_name)
    master_frames = np.asarray(master_entry["frames"], dtype=int)
    master_time = np.asarray(master_entry["time_s"], dtype=float)
    resolved_biomod = resolve_dataset_biomod(dataset_dir, biomod_path)

    recon_points: dict[str, np.ndarray] = {}
    summary_by_name: dict[str, dict[str, object]] = {}
    for entry in point_entries:
        name = str(entry["name"])
        frames = np.asarray(entry["frames"], dtype=int)
        q = np.asarray(entry["q"], dtype=float)
        points = np.asarray(entry["points_3d"], dtype=float)
        summary = dict(entry["summary"])
        if (
            resolved_biomod is not None
            and q.ndim == 2
            and q.shape[1] > 6
            and (summary.get("points_3d_source") != "model_forward_kinematics" or not np.any(np.isfinite(points)))
        ):
            points = biorbd_markers_from_q(resolved_biomod, q)
        if not np.array_equal(frames, master_frames):
            points = align_array_to_frames(points, frames, master_frames)
        recon_points[name] = points
        summary_by_name[name] = summary

    flight_source_name = preferred_triangulation_name(list(recon_points.keys())) or master_name
    flight_summary = summary_by_name.get(flight_source_name, {})
    flight_threshold, flight_min_consecutive = load_flight_parameters_from_summary(flight_summary)
    airborne_mask = compute_airborne_mask(recon_points[flight_source_name], flight_threshold, flight_min_consecutive)
    return recon_points, master_frames, master_time, airborne_mask


def root_center(points: np.ndarray) -> np.ndarray:
    """Centre du bassin a partir des hanches gauche et droite."""
    left = points[:, KP_INDEX["left_hip"], :]
    right = points[:, KP_INDEX["right_hip"], :]
    return 0.5 * (left + right)


def align_pose2sim_to_triangulation(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """Aligne rigidement la seconde reconstruction par translation du bassin au premier frame valide."""
    ref_root = root_center(reference)
    mov_root = root_center(moving)
    valid = np.where(np.all(np.isfinite(ref_root), axis=1) & np.all(np.isfinite(mov_root), axis=1))[0]
    if valid.size == 0:
        return moving
    delta = ref_root[valid[0]] - mov_root[valid[0]]
    return moving + delta[np.newaxis, np.newaxis, :]


def compute_axis_limits(
    *point_sets: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Bornes 3D fixes calculees sur l'ensemble des reconstructions comparees."""
    point_sets = [pts for pts in point_sets if pts is not None]
    flat = np.concatenate([pts.reshape(-1, 3) for pts in point_sets], axis=0)
    flat = flat[np.all(np.isfinite(flat), axis=1)]
    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(0.25, 0.55 * np.max(maxs - mins))
    return (
        (center[0] - radius, center[0] + radius),
        (center[1] - radius, center[1] + radius),
        (center[2] - radius, center[2] + radius),
    )


def compute_frame_axis_limits(
    recon_points: dict[str, np.ndarray], show_names: list[str], frame_idx: int
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Compute 3D limits from the currently displayed frame only."""
    frame_sets = [
        recon_points[name][frame_idx : frame_idx + 1] for name in show_names if frame_idx < recon_points[name].shape[0]
    ]
    return compute_axis_limits(*frame_sets)


def apply_axis_limits(ax, limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]) -> None:
    """Apply precomputed axis limits to a 3D axes."""
    xlim, ylim, zlim = limits
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)


def draw_trampoline_bed(ax, z_level: float) -> None:
    """Draw a simple trampoline reference bed below the athlete trajectories."""

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
    cross = TRAMPOLINE_GEOMETRY.cross
    ax.plot(outer[:, 0], outer[:, 1], outer[:, 2], color="#2b6cb0", linewidth=1.8, alpha=0.45)
    ax.plot(big_rect[:, 0], big_rect[:, 1], big_rect[:, 2], color="#b56576", linewidth=1.2, alpha=0.35)
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


def grouped_marker_points(frame_points: np.ndarray) -> dict[str, np.ndarray]:
    groups = {
        "left": [KP_INDEX[name] for name in COCO17 if name in LEFT_KEYPOINTS],
        "right": [KP_INDEX[name] for name in COCO17 if name in RIGHT_KEYPOINTS],
        "center": [KP_INDEX[name] for name in COCO17 if name not in LEFT_KEYPOINTS and name not in RIGHT_KEYPOINTS],
    }
    out: dict[str, np.ndarray] = {}
    for side, indices in groups.items():
        points = frame_points[indices]
        valid = np.all(np.isfinite(points), axis=1)
        out[side] = points[valid]
    return out


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
    ax, origin: np.ndarray, rotation: np.ndarray, scale: float = 0.18, alpha: float = 1.0, line_width: float = 2.0
):
    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    artists = []
    for axis_idx in range(3):
        direction = rotation[:, axis_idx]
        endpoint = origin + scale * direction
        (line,) = ax.plot(
            [origin[0], endpoint[0]],
            [origin[1], endpoint[1]],
            [origin[2], endpoint[2]],
            color=colors[axis_idx],
            linewidth=line_width,
            alpha=alpha,
        )
        artists.append(line)
    return artists


def edge_linewidth(name_a: str, name_b: str, base: float = 2.0) -> float:
    return base * 3.0 if (name_a, name_b) in LOWER_LIMB_EDGES else base


def init_artists(ax, color: str, label: str, marker_size: float, line_style: str = "-"):
    """Cree les artistes matplotlib pour une stick figure."""
    scatter = {
        "center": ax.scatter([], [], [], s=marker_size, c=color, marker="o", depthshade=False, label=label),
        "left": ax.scatter([], [], [], s=marker_size * 1.35, c=color, marker="^", depthshade=False),
        "right": ax.scatter([], [], [], s=marker_size * 1.35, c=color, marker="s", depthshade=False),
    }
    lines = []
    for name_a, name_b in SKELETON_EDGES:
        (line,) = ax.plot(
            [], [], [], linewidth=edge_linewidth(name_a, name_b), color=color, alpha=0.9, linestyle=line_style
        )
        lines.append(line)
    return scatter, lines


def build_artist_group(ax, enabled: bool, color: str, label: str, marker_size: float, line_style: str = "-"):
    """Construit un groupe d'artistes actif ou inactif selon la reconstruction choisie."""
    return init_artists(ax, color, label, marker_size, line_style) if enabled else (None, [])


def update_artists(scatter, lines, frame_points: np.ndarray):
    """Met a jour les points et segments d'une stick figure."""
    if scatter is None:
        return
    grouped = grouped_marker_points(frame_points)
    for side, artist in scatter.items():
        xyz = grouped[side]
        if xyz.size:
            artist._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
        else:
            artist._offsets3d = ([], [], [])

    for line, (name_a, name_b) in zip(lines, SKELETON_EDGES):
        point_a = frame_points[KP_INDEX[name_a]]
        point_b = frame_points[KP_INDEX[name_b]]
        if np.all(np.isfinite(point_a)) and np.all(np.isfinite(point_b)):
            line.set_data([point_a[0], point_b[0]], [point_a[1], point_b[1]])
            line.set_3d_properties([point_a[2], point_b[2]])
        else:
            line.set_data([], [])
            line.set_3d_properties([])


def create_animation(
    recon_points: dict[str, np.ndarray],
    time_s: np.ndarray,
    output_path: Path,
    fps: float,
    marker_size: float,
    show: tuple[str, ...],
    airborne_mask: np.ndarray,
    show_trunk_frames: bool,
    framing: str,
    show_trampoline: bool,
) -> None:
    """Exporte un GIF comparatif pour l'ensemble des reconstructions demandees."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    show_names = [name for name in show if name in recon_points]
    if not show_names:
        raise ValueError("Aucune reconstruction selectionnee n'est disponible pour l'animation 3D.")
    full_limits = compute_axis_limits(*[recon_points[name] for name in show_names])
    stacked_points = np.concatenate([recon_points[name].reshape(-1, 3) for name in show_names], axis=0)
    valid_points = stacked_points[np.all(np.isfinite(stacked_points), axis=1)]
    trampoline_z = float(np.nanpercentile(valid_points[:, 2], 5)) if valid_points.size else 0.0

    artists: dict[str, tuple[object | None, list]] = {}
    for name in show_names:
        artists[name] = build_artist_group(
            ax, True, reconstruction_color(name), reconstruction_label(name), marker_size, "-"
        )
    trunk_frame_artists: dict[str, list] = {name: [] for name in show_names}
    label = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    def init():
        apply_axis_limits(ax, full_limits)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Comparaison 3D des reconstructions selectionnees")
        ax.view_init(elev=18, azim=-65)
        ax.legend(loc="upper right")
        if show_trampoline:
            draw_trampoline_bed(ax, trampoline_z)
        init_artists = [label]
        for scatter, lines in artists.values():
            if scatter is not None:
                init_artists.extend(scatter.values())
            init_artists.extend(lines)
        return init_artists

    def update(frame_idx: int):
        updated = [label]
        if framing == "tight":
            apply_axis_limits(ax, compute_frame_axis_limits(recon_points, show_names, frame_idx))
        else:
            apply_axis_limits(ax, full_limits)
        for name in show_names:
            for artist in trunk_frame_artists[name]:
                try:
                    artist.remove()
                except Exception:
                    pass
            trunk_frame_artists[name] = []
        for name in show_names:
            scatter, lines = artists[name]
            frame_points = recon_points[name][frame_idx]
            update_artists(scatter, lines, frame_points)
            if scatter is not None:
                updated.extend(scatter.values())
            updated.extend(lines)
            if show_trunk_frames:
                origin, rotation = compute_root_frame_from_points(frame_points)
                if origin is not None and rotation is not None:
                    trunk_frame_artists[name] = draw_coordinate_system(
                        ax, origin, rotation, scale=0.18, alpha=0.95, line_width=2.2
                    )
                    updated.extend(trunk_frame_artists[name])
        phase = "AIR" if airborne_mask[frame_idx] else "TOILE"
        label.set_text(f"t = {time_s[frame_idx]:.2f} s | {phase}")
        return updated

    first_name = show_names[0]
    anim = FuncAnimation(
        fig, update, init_func=init, frames=recon_points[first_name].shape[0], interval=1000 / fps, blit=False
    )
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main() -> None:
    """Point d'entree CLI."""
    args = parse_args()

    if args.dataset_dir is not None:
        recon_points, frames, time_s, airborne_mask = load_dataset_reconstructions(args.dataset_dir, args.biomod)
    else:
        local_points, local_frames = load_triangulation(args.triangulation)
        local_time = local_frames / args.triangulation_fps
        flight_threshold, flight_min_consecutive = load_flight_parameters(args.triangulation)
        airborne_mask = compute_airborne_mask(local_points, flight_threshold, flight_min_consecutive)
        ekf_2d_acc_q, ekf_3d_q, ekf_2d_dyn_q, ekf_2d_flip_acc_q, ekf_2d_flip_dyn_q = load_q_reconstructions(
            args.ekf_states, args.kalman_comparison
        )
        ekf_2d_acc_points = biorbd_markers_from_q(args.biomod, ekf_2d_acc_q)
        ekf_3d_points = biorbd_markers_from_q(args.biomod, ekf_3d_q)
        ekf_2d_dyn_points = None if ekf_2d_dyn_q is None else biorbd_markers_from_q(args.biomod, ekf_2d_dyn_q)
        ekf_2d_flip_acc_points = (
            None if ekf_2d_flip_acc_q is None else biorbd_markers_from_q(args.biomod, ekf_2d_flip_acc_q)
        )
        ekf_2d_flip_dyn_points = (
            None if ekf_2d_flip_dyn_q is None else biorbd_markers_from_q(args.biomod, ekf_2d_flip_dyn_q)
        )

        pose2sim_points, pose2sim_time, _ = parse_trc(args.pose2sim_trc)

        # La comparaison se fait sur l'intervalle temporel commun, puis la
        # reconstruction Pose2Sim est re-echantillonnee sur la grille du NPZ local.
        t_max = min(local_time[-1], pose2sim_time[-1])
        keep = local_time <= t_max
        local_points = local_points[keep]
        local_time = local_time[keep]
        airborne_mask = airborne_mask[keep]
        ekf_2d_acc_points = ekf_2d_acc_points[keep]
        ekf_3d_points = ekf_3d_points[keep]
        if ekf_2d_dyn_points is not None:
            ekf_2d_dyn_points = ekf_2d_dyn_points[keep]
        if ekf_2d_flip_acc_points is not None:
            ekf_2d_flip_acc_points = ekf_2d_flip_acc_points[keep]
        if ekf_2d_flip_dyn_points is not None:
            ekf_2d_flip_dyn_points = ekf_2d_flip_dyn_points[keep]
        pose2sim_points = resample_points(pose2sim_points, pose2sim_time, local_time)

        recon_points = {
            "triangulation": local_points,
            "pose2sim": pose2sim_points,
            "ekf_3d": ekf_3d_points,
            "ekf_2d_acc": ekf_2d_acc_points,
        }
        if ekf_2d_dyn_points is not None:
            recon_points["ekf_2d_dyn"] = ekf_2d_dyn_points
        if ekf_2d_flip_acc_points is not None:
            recon_points["ekf_2d_flip_acc"] = ekf_2d_flip_acc_points
        if ekf_2d_flip_dyn_points is not None:
            recon_points["ekf_2d_flip_dyn"] = ekf_2d_flip_dyn_points
        frames = local_frames[keep]
        time_s = local_time

    show_names = resolve_requested_names(args.show, list(recon_points.keys()))
    if not show_names:
        raise ValueError("Aucune reconstruction disponible ne correspond aux options --show demandees.")

    stride = max(args.stride, 1)
    recon_points = {name: points[::stride] for name, points in recon_points.items()}
    time_s = time_s[::stride]
    airborne_mask = airborne_mask[::stride]

    if args.align_root:
        reference_name = preferred_triangulation_name(show_names) or show_names[0]
        reference_points = recon_points[reference_name]
        for name in list(recon_points.keys()):
            if name == reference_name:
                continue
            recon_points[name] = align_pose2sim_to_triangulation(reference_points, recon_points[name])

    create_animation(
        recon_points,
        time_s,
        args.output,
        fps=args.fps,
        marker_size=args.marker_size,
        show=tuple(show_names),
        airborne_mask=airborne_mask,
        show_trunk_frames=bool(args.show_trunk_frames),
        framing=str(args.framing),
        show_trampoline=bool(args.show_trampoline),
    )
    print(f"Animation comparative exportee dans: {args.output}")


if __name__ == "__main__":
    main()
