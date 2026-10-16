#!/usr/bin/env python3
"""Analyse des sauts de trampoline a partir des coordonnees generalisees.

Le script:
1. segmente les sauts a partir de la hauteur de la racine,
2. estime le nombre de saltos et de vrilles par salto,
3. detecte une forme corps groupee / carpee / tendue,
4. affiche un code simplifie inspire du code de pointage.

Hypotheses par defaut pour le modele courant:
- hauteur racine: `TRUNK:TransZ`,
- axe de salto: `TRUNK:RotY`,
- axe de vrille: `TRUNK:RotZ`,
- flexion hanches: `LEFT_THIGH:RotY` et `RIGHT_THIGH:RotY`,
- flexion genoux: `LEFT_SHANK:RotY` et `RIGHT_SHANK:RotY`.

Le code produit suit une convention simplifiee:
- premier chiffre: `8` pour un salto arriere, `4` pour un salto avant,
- deuxieme chiffre: nombre total de saltos arrondi au 1/4,
- chiffres suivants: vrilles de chaque salto complet en demi-vrilles,
- suffixe: `o` groupe, `<` carpe, `/` tendu.
Exemple: `822<` = double salto arriere avec 1 vrille dans chaque salto, en carpe.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

LOCAL_MPLCONFIG = ROOT / ".cache" / "matplotlib"
LOCAL_MPLCONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_MPLCONFIG))

import matplotlib.pyplot as plt

DEFAULT_FPS = 120.0
DEFAULT_STATES = Path("output") / "vitpose_full" / "ekf_states.npz"


@dataclass
class JumpSegment:
    """Continuous airborne segment delimited by contact minima and one peak."""

    start: int
    end: int
    peak_index: int


@dataclass
class JumpAnalysis:
    """Aggregated salto, twist, and body-shape analysis for one jump."""

    segment: JumpSegment
    total_saltos: float
    twists_per_salto: list[float]
    body_shape: str
    code: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the trampoline-jump analysis script."""

    parser = argparse.ArgumentParser(
        description="Analyse des sauts de trampoline a partir des coordonnees generalisees."
    )
    parser.add_argument("--states", type=Path, default=DEFAULT_STATES, help="Fichier ekf_states.npz")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Frequence d'echantillonnage")
    parser.add_argument("--figure", type=Path, default=None, help="PNG optionnel montrant le decoupage des sauts")
    parser.add_argument(
        "--rotation-figure",
        type=Path,
        default=None,
        help="PNG optionnel montrant saltos et vrilles en tours avec la separation des sauts",
    )
    parser.add_argument(
        "--height-threshold",
        type=float,
        default=None,
        help="Seuil absolu de hauteur racine. Si absent, utilise un seuil relatif.",
    )
    parser.add_argument(
        "--height-threshold-range-ratio",
        type=float,
        default=0.20,
        help="Seuil relatif exprime en fraction du range de hauteur.",
    )
    parser.add_argument(
        "--smoothing-window-s", type=float, default=0.15, help="Fenetre de lissage de la hauteur pour la segmentation."
    )
    parser.add_argument(
        "--min-airtime-s", type=float, default=0.25, help="Duree minimale au-dessus du seuil pour conserver un saut"
    )
    parser.add_argument(
        "--min-gap-s", type=float, default=0.08, help="Ecart minimal entre deux phases aeriennes avant fusion"
    )
    parser.add_argument(
        "--min-peak-prominence-m",
        type=float,
        default=0.35,
        help="Proeminence minimale du pic de hauteur par rapport aux points bas",
    )
    parser.add_argument(
        "--contact-window-s", type=float, default=0.35, help="Fenetre de recherche locale pour les minima de contact"
    )
    parser.add_argument("--height-dof", type=str, default="TRUNK:TransZ", help="DoF de hauteur de la racine")
    parser.add_argument(
        "--salto-dof", type=str, default="TRUNK:RotY", help="DoF racine utilise pour compter les saltos"
    )
    parser.add_argument(
        "--twist-dof", type=str, default="TRUNK:RotZ", help="DoF racine utilise pour compter les vrilles"
    )
    parser.add_argument(
        "--hip-dofs",
        nargs="+",
        default=("LEFT_THIGH:RotY", "RIGHT_THIGH:RotY"),
        help="DoF de flexion des hanches",
    )
    parser.add_argument(
        "--knee-dofs",
        nargs="+",
        default=("LEFT_SHANK:RotY", "RIGHT_SHANK:RotY"),
        help="DoF de flexion des genoux",
    )
    parser.add_argument(
        "--hip-threshold-deg",
        type=float,
        default=70.0,
        help="Seuil de flexion hanche pour detecter une forme groupee ou carpee",
    )
    parser.add_argument(
        "--knee-tuck-threshold-deg",
        type=float,
        default=70.0,
        help="Seuil de flexion genou pour classer en groupe",
    )
    parser.add_argument(
        "--knee-pike-threshold-deg",
        type=float,
        default=20.0,
        help="Seuil max de flexion genou pour classer en carpe",
    )
    return parser.parse_args()


def load_states(states_path: Path) -> tuple[np.ndarray, list[str]]:
    """Load generalized coordinates and DoF names from one NPZ file."""

    data = np.load(states_path, allow_pickle=True)
    q = np.asarray(data["q"], dtype=float)
    q_names = [str(name) for name in data["q_names"]]
    return q, q_names


def find_dof_indices(q_names: list[str], names: list[str]) -> list[int]:
    """Resolve requested DoF names to their indices in the q-name list."""

    index_map = {name: idx for idx, name in enumerate(q_names)}
    missing = [name for name in names if name not in index_map]
    if missing:
        raise KeyError(f"DoF absents des etats: {missing}")
    return [index_map[name] for name in names]


def contiguous_true_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive index ranges for all contiguous ``True`` regions."""

    regions: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            regions.append((start, idx - 1))
            start = None
    if start is not None:
        regions.append((start, len(mask) - 1))
    return regions


def merge_close_regions(regions: list[tuple[int, int]], max_gap_frames: int) -> list[tuple[int, int]]:
    """Fusionne des phases aeriennes tres proches qui appartiennent au meme saut."""
    if not regions:
        return []
    merged = [list(regions[0])]
    for start, end in regions[1:]:
        if start - merged[-1][1] - 1 <= max_gap_frames:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def local_minimum_index(height: np.ndarray, center: int, left_limit: int, right_limit: int, window_frames: int) -> int:
    """Cherche un minimum local de contact autour d'un point de transition."""
    start = max(left_limit, center - window_frames)
    end = min(right_limit, center + window_frames)
    if end <= start:
        return center
    return start + int(np.argmin(height[start : end + 1]))


def refine_jump_boundaries(
    height: np.ndarray, airborne_regions: list[tuple[int, int]], contact_window_frames: int
) -> list[JumpSegment]:
    """Convert airborne ranges into jump segments snapped to nearby contacts."""

    segments: list[JumpSegment] = []
    if not airborne_regions:
        return segments

    previous_end = 0
    for region_idx, (start, end) in enumerate(airborne_regions):
        next_region_start = (
            airborne_regions[region_idx + 1][0] if region_idx + 1 < len(airborne_regions) else len(height) - 1
        )
        jump_start = local_minimum_index(
            height,
            center=start,
            left_limit=previous_end,
            right_limit=start,
            window_frames=contact_window_frames,
        )
        jump_end = local_minimum_index(
            height,
            center=end,
            left_limit=end,
            right_limit=next_region_start,
            window_frames=contact_window_frames,
        )
        peak_local = jump_start + int(np.argmax(height[jump_start : jump_end + 1]))
        segments.append(JumpSegment(start=jump_start, end=jump_end, peak_index=peak_local))
        previous_end = jump_end

    return segments


def filter_jump_segments(
    height: np.ndarray,
    segments: list[JumpSegment],
    height_threshold: float,
    min_airtime_frames: int,
    min_peak_prominence_m: float,
) -> list[JumpSegment]:
    """Retire les micro-sauts et les traversées de seuil peu pertinentes."""
    kept: list[JumpSegment] = []
    for segment in segments:
        segment_height = height[segment.start : segment.end + 1]
        airborne_frames = int(np.count_nonzero(segment_height > height_threshold))
        if airborne_frames < min_airtime_frames:
            continue
        start_h = float(height[segment.start])
        end_h = float(height[segment.end])
        peak_h = float(height[segment.peak_index])
        prominence = peak_h - max(start_h, end_h)
        if prominence < min_peak_prominence_m:
            continue
        kept.append(segment)
    return kept


def smooth_signal(signal: np.ndarray, window_frames: int) -> np.ndarray:
    """Lisse un signal 1D par moyenne glissante centree."""
    window_frames = max(1, int(window_frames))
    if window_frames <= 1:
        return signal.copy()
    if window_frames % 2 == 0:
        window_frames += 1
    kernel = np.ones(window_frames, dtype=float) / window_frames
    padded = np.pad(signal, (window_frames // 2, window_frames // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def relative_height_threshold(height: np.ndarray, ratio: float) -> float:
    """Construit un seuil a partir du minimum et du range de hauteur."""
    min_h = float(np.nanmin(height))
    max_h = float(np.nanmax(height))
    return min_h + ratio * (max_h - min_h)


def plot_jump_segmentation(
    height: np.ndarray,
    smoothed_height: np.ndarray,
    fps: float,
    height_threshold: float,
    airborne_regions: list[tuple[int, int]],
    jump_segments: list[JumpSegment],
    output_path: Path,
) -> None:
    """Exporte une figure de segmentation des sauts sur la hauteur de racine."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(height.shape[0]) / fps

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, height, color="#1f4e79", linewidth=1.0, alpha=0.45, label="Hauteur racine brute")
    ax.plot(t, smoothed_height, color="#1f4e79", linewidth=2.0, label="Hauteur racine lisse")
    ax.axhline(height_threshold, color="#c44e52", linestyle="--", linewidth=1.5, label="Seuil")

    for idx, (start, end) in enumerate(airborne_regions):
        ax.axvspan(t[start], t[end], color="#dd8452", alpha=0.12, label="Phase aerienne" if idx == 0 else None)

    for idx, segment in enumerate(jump_segments):
        ax.axvline(
            t[segment.start],
            color="#55a868",
            linestyle="-",
            linewidth=1.8,
            label="Debut/fin saut" if idx == 0 else None,
        )
        ax.axvline(t[segment.end], color="#55a868", linestyle="-", linewidth=1.8)
        ax.scatter(
            t[segment.peak_index],
            height[segment.peak_index],
            color="#8172b3",
            s=35,
            zorder=3,
            label="Pic saut" if idx == 0 else None,
        )
        ax.text(
            0.5 * (t[segment.start] + t[segment.end]),
            smoothed_height[segment.peak_index] + 0.05,
            f"S{idx + 1}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Hauteur racine (m)")
    ax.set_title("Decoupage des sauts a partir de la hauteur de racine")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_jump_rotations(
    q: np.ndarray,
    fps: float,
    salto_idx: int,
    twist_idx: int,
    jump_segments: list[JumpSegment],
    output_path: Path,
) -> None:
    """Exporte une figure des rotations de salto et vrille en tours."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(q.shape[0]) / fps
    salto_turns = np.full(q.shape[0], np.nan)
    twist_turns = np.full(q.shape[0], np.nan)

    for segment in jump_segments:
        q_jump = q[segment.start : segment.end + 1]
        salto_local = np.unwrap(q_jump[:, salto_idx])
        twist_local = np.unwrap(q_jump[:, twist_idx])
        salto_turns[segment.start : segment.end + 1] = (salto_local - salto_local[0]) / (2 * np.pi)
        twist_turns[segment.start : segment.end + 1] = (twist_local - twist_local[0]) / (2 * np.pi)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(t, salto_turns, color="#4c72b0", linewidth=1.8, label="Salto (tours)")
    axes[0].set_ylabel("Tours")
    axes[0].set_title("Rotation de salto")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, twist_turns, color="#c44e52", linewidth=1.8, label="Vrille (tours)")
    axes[1].set_ylabel("Tours")
    axes[1].set_xlabel("Temps (s)")
    axes[1].set_title("Rotation de vrille")
    axes[1].grid(alpha=0.3)

    for idx, segment in enumerate(jump_segments):
        for ax in axes:
            ax.axvline(
                t[segment.start],
                color="#55a868",
                linestyle="-",
                linewidth=1.6,
                label="Debut/fin saut" if idx == 0 else None,
            )
            ax.axvline(t[segment.end], color="#55a868", linestyle="-", linewidth=1.6)
            ax.axvspan(
                t[segment.start],
                t[segment.end],
                color="#55a868",
                alpha=0.08,
                label="Intervalle saut" if idx == 0 else None,
            )

    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def round_to_nearest(value: float, step: float) -> float:
    """Round one value to the nearest multiple of ``step``."""

    return round(value / step) * step


def crossing_index(cumulative_turns: np.ndarray, target: float) -> int | None:
    """Return the first index where the cumulative turns reach ``target``."""

    hits = np.where(cumulative_turns >= target)[0]
    return int(hits[0]) if hits.size else None


def detect_body_shape(
    q_segment: np.ndarray,
    hip_indices: list[int],
    knee_indices: list[int],
    hip_threshold_deg: float,
    knee_tuck_threshold_deg: float,
    knee_pike_threshold_deg: float,
) -> str:
    """Classify one jump as tucked, piked, or straight from flexion angles."""

    hip_peak_deg = np.rad2deg(np.nanmax(np.abs(q_segment[:, hip_indices]), axis=0))
    knee_peak_deg = np.rad2deg(np.nanmax(np.abs(q_segment[:, knee_indices]), axis=0))
    hip_value = float(np.nanmax(hip_peak_deg))
    knee_value = float(np.nanmax(knee_peak_deg))

    if hip_value >= hip_threshold_deg and knee_value >= knee_tuck_threshold_deg:
        return "grouped"
    if hip_value >= hip_threshold_deg and knee_value <= knee_pike_threshold_deg:
        return "piked"
    return "straight"


def body_shape_suffix(body_shape: str) -> str:
    """Return the abbreviated judging suffix associated with one body shape."""

    return {"grouped": "o", "piked": "<", "straight": "/"}.get(body_shape, "?")


def salto_count_token(total_saltos: float) -> str:
    """Encode the salto count using the simplified DD token convention."""

    integer_part = int(total_saltos)
    quarter_map = {0.0: "", 0.25: ".", 0.5: "+", 0.75: "-"}
    fraction = round(total_saltos - integer_part, 2)
    return f"{integer_part}{quarter_map.get(fraction, '')}"


def analyze_jump(
    q: np.ndarray,
    segment: JumpSegment,
    salto_idx: int,
    twist_idx: int,
    hip_indices: list[int],
    knee_indices: list[int],
    hip_threshold_deg: float,
    knee_tuck_threshold_deg: float,
    knee_pike_threshold_deg: float,
) -> JumpAnalysis:
    """Analyze one jump segment and derive the simplified DD-style summary."""

    q_segment = q[segment.start : segment.end + 1]
    salto = np.unwrap(q_segment[:, salto_idx])
    twist = np.unwrap(q_segment[:, twist_idx])

    delta_salto_turns = (salto[-1] - salto[0]) / (2 * np.pi)
    delta_twist_turns = (twist[-1] - twist[0]) / (2 * np.pi)
    total_saltos = abs(round_to_nearest(delta_salto_turns, 0.25))
    direction_prefix = "8" if delta_salto_turns >= 0 else "4"

    cumulative_salto_turns = np.abs((salto - salto[0]) / (2 * np.pi))
    cumulative_twist_turns = (twist - twist[0]) / (2 * np.pi)
    twists_per_salto: list[float] = []

    completed_saltos = int(np.floor(total_saltos + 1e-9))
    segment_start = 0
    for salto_number in range(1, completed_saltos + 1):
        end_idx = crossing_index(cumulative_salto_turns, float(salto_number))
        if end_idx is None or end_idx <= segment_start:
            break
        twist_delta = cumulative_twist_turns[end_idx] - cumulative_twist_turns[segment_start]
        twists_per_salto.append(abs(round_to_nearest(twist_delta, 0.5)))
        segment_start = end_idx

    body_shape = detect_body_shape(
        q_segment,
        hip_indices,
        knee_indices,
        hip_threshold_deg=hip_threshold_deg,
        knee_tuck_threshold_deg=knee_tuck_threshold_deg,
        knee_pike_threshold_deg=knee_pike_threshold_deg,
    )
    twist_tokens = "".join(str(int(round(twists * 2))) for twists in twists_per_salto) if twists_per_salto else "0"
    code = f"{direction_prefix}{salto_count_token(total_saltos)}{twist_tokens}{body_shape_suffix(body_shape)}"

    return JumpAnalysis(
        segment=segment,
        total_saltos=total_saltos,
        twists_per_salto=twists_per_salto,
        body_shape=body_shape,
        code=code,
    )


def main() -> None:
    """Run trampoline-jump analysis from the command line."""

    args = parse_args()
    q, q_names = load_states(args.states)

    height_idx = find_dof_indices(q_names, [args.height_dof])[0]
    salto_idx = find_dof_indices(q_names, [args.salto_dof])[0]
    twist_idx = find_dof_indices(q_names, [args.twist_dof])[0]
    hip_indices = find_dof_indices(q_names, list(args.hip_dofs))
    knee_indices = find_dof_indices(q_names, list(args.knee_dofs))

    height = q[:, height_idx]
    smoothed_height = smooth_signal(height, window_frames=max(1, int(round(args.smoothing_window_s * args.fps))))
    height_threshold = (
        args.height_threshold
        if args.height_threshold is not None
        else relative_height_threshold(
            smoothed_height,
            ratio=args.height_threshold_range_ratio,
        )
    )
    airborne_mask = smoothed_height > height_threshold
    airborne_regions = contiguous_true_regions(airborne_mask)
    airborne_regions = merge_close_regions(
        airborne_regions, max_gap_frames=max(0, int(round(args.min_gap_s * args.fps)))
    )
    jump_segments = refine_jump_boundaries(
        smoothed_height,
        airborne_regions,
        contact_window_frames=max(1, int(round(args.contact_window_s * args.fps))),
    )
    jump_segments = filter_jump_segments(
        smoothed_height,
        jump_segments,
        height_threshold=height_threshold,
        min_airtime_frames=max(1, int(round(args.min_airtime_s * args.fps))),
        min_peak_prominence_m=args.min_peak_prominence_m,
    )

    if not jump_segments:
        print("Aucun saut detecte avec le seuil choisi.")
        return

    figure_path = args.figure or (args.states.parent / "jump_segmentation.png")
    rotation_figure_path = args.rotation_figure or (args.states.parent / "jump_rotations.png")
    plot_jump_segmentation(
        height=height,
        smoothed_height=smoothed_height,
        fps=args.fps,
        height_threshold=height_threshold,
        airborne_regions=airborne_regions,
        jump_segments=jump_segments,
        output_path=figure_path,
    )
    plot_jump_rotations(
        q=q,
        fps=args.fps,
        salto_idx=salto_idx,
        twist_idx=twist_idx,
        jump_segments=jump_segments,
        output_path=rotation_figure_path,
    )

    print(f"Seuil de segmentation utilise: {height_threshold:.3f} m")
    print(f"Sauts detectes: {len(jump_segments)}")
    for jump_idx, segment in enumerate(jump_segments, start=1):
        analysis = analyze_jump(
            q,
            segment,
            salto_idx,
            twist_idx,
            hip_indices,
            knee_indices,
            hip_threshold_deg=args.hip_threshold_deg,
            knee_tuck_threshold_deg=args.knee_tuck_threshold_deg,
            knee_pike_threshold_deg=args.knee_pike_threshold_deg,
        )
        start_t = analysis.segment.start / args.fps
        end_t = analysis.segment.end / args.fps
        peak_h = float(np.nanmax(height[analysis.segment.start : analysis.segment.end + 1]))
        twists_str = (
            ", ".join(f"{value:.1f}" for value in analysis.twists_per_salto) if analysis.twists_per_salto else "none"
        )

        print(
            f"Saut {jump_idx}: frames {analysis.segment.start}-{analysis.segment.end} "
            f"({start_t:.2f}-{end_t:.2f} s), pic hauteur {peak_h:.2f} m"
        )
        print(
            f"  saltos={analysis.total_saltos:.2f}, vrilles/salto=[{twists_str}], "
            f"forme={analysis.body_shape}, code={analysis.code}"
        )
    print(f"Figure de decoupage exportee dans: {figure_path}")
    print(f"Figure rotations exportee dans: {rotation_figure_path}")


if __name__ == "__main__":
    main()
