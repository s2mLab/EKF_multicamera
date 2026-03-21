#!/usr/bin/env python3
"""Teste si un remapping camera <-> calibration pourrait mieux expliquer les vues.

L'idee est de comparer la coherence epipolaire entre paires de flux 2D pour
plusieurs hypothese de mapping:
- mapping courant (identite),
- tous les swaps simples,
- quelques permutations ciblees sur les cameras peu utilisees.

Un score plus faible indique qu'un mapping est plus coherent avec la geometrie
des calibrations.
"""
from __future__ import annotations

import argparse
import itertools as it
import json
from pathlib import Path

import numpy as np

import vitpose_ekf_pipeline as vp


def parse_args() -> argparse.Namespace:
    """Construit l'interface CLI."""
    parser = argparse.ArgumentParser(description="Analyse d'eventuelles inversions de cameras.")
    parser.add_argument("--calib", type=Path, default=vp.DEFAULT_CALIB, help="Fichier Calib.toml")
    parser.add_argument("--keypoints", type=Path, default=vp.DEFAULT_KEYPOINTS, help="JSON des keypoints 2D")
    parser.add_argument(
        "--triangulation",
        type=Path,
        default=Path("outputs/vitpose_full/triangulation_pose2sim_like.npz"),
        help="NPZ de triangulation pour lire les stats d'utilisation des vues.",
    )
    parser.add_argument("--frame-stride", type=int, default=10, help="Sous-echantillonnage temporel pour accelerer l'analyse.")
    parser.add_argument("--score-threshold", type=float, default=0.2, help="Score 2D minimal pour garder un point.")
    parser.add_argument("--top-swaps", type=int, default=10, help="Nombre de swaps a afficher.")
    return parser.parse_args()


def median_sampson_for_assignment(
    pose: vp.PoseData,
    calibrations: dict[str, vp.CameraCalibration],
    assigned_cameras: tuple[int, ...],
    frame_stride: int,
    score_threshold: float,
) -> tuple[float, int]:
    """Calcule un cout median pairwise de coherence epipolaire pour un mapping donne."""
    camera_names = pose.camera_names
    frame_idx = np.arange(0, pose.keypoints.shape[1], frame_stride)
    total = 0.0
    n_pairs = 0

    for i_stream, j_stream in it.combinations(range(len(camera_names)), 2):
        cam_i = camera_names[assigned_cameras[i_stream]]
        cam_j = camera_names[assigned_cameras[j_stream]]
        F_ij = vp.fundamental_matrix(calibrations[cam_i], calibrations[cam_j])

        pts_i = pose.keypoints[i_stream, frame_idx].reshape(-1, 2)
        pts_j = pose.keypoints[j_stream, frame_idx].reshape(-1, 2)
        score_i = pose.scores[i_stream, frame_idx].reshape(-1)
        score_j = pose.scores[j_stream, frame_idx].reshape(-1)

        valid = (
            (score_i > score_threshold)
            & (score_j > score_threshold)
            & np.isfinite(pts_i).all(axis=1)
            & np.isfinite(pts_j).all(axis=1)
        )
        if valid.sum() < 20:
            continue

        errors = np.array([vp.sampson_error_pixels(a, b, F_ij) for a, b in zip(pts_i[valid], pts_j[valid])], dtype=float)
        finite = errors[np.isfinite(errors)]
        if finite.size == 0:
            continue

        total += float(np.median(finite))
        n_pairs += 1

    return total / max(n_pairs, 1), n_pairs


def load_usage_counts(triangulation_path: Path) -> tuple[list[str], np.ndarray]:
    """Retourne le nombre total d'utilisations par camera depuis le cache NPZ."""
    data = np.load(triangulation_path, allow_pickle=True)
    metadata = json.loads(data["metadata"].item())
    camera_names = metadata["camera_names"]
    used = ~np.asarray(data["excluded_views"], dtype=bool)
    return camera_names, used.sum(axis=(0, 1))


def main() -> None:
    """Lance l'analyse et imprime les hypotheses les plus plausibles."""
    args = parse_args()
    calibrations = vp.load_calibrations(args.calib)
    pose = vp.load_pose_data(args.keypoints, calibrations)
    camera_names, usage_counts = load_usage_counts(args.triangulation)

    print("Ordre des flux 2D / calibrations charges :")
    for idx, name in enumerate(pose.camera_names):
        print(f"  {idx}: {name}")

    print("\nUtilisation actuelle dans la triangulation :")
    for name, count in zip(camera_names, usage_counts):
        print(f"  {name}: {int(count)} observations retenues")

    identity = tuple(range(len(pose.camera_names)))
    baseline_cost, baseline_pairs = median_sampson_for_assignment(
        pose, calibrations, identity, args.frame_stride, args.score_threshold
    )
    print(f"\nCout median epipolaire baseline: {baseline_cost:.3f} px sur {baseline_pairs} paires de flux")

    swap_results: list[tuple[float, tuple[int, ...], str]] = []
    for i, j in it.combinations(range(len(pose.camera_names)), 2):
        perm = list(identity)
        perm[i], perm[j] = perm[j], perm[i]
        cost, n_pairs = median_sampson_for_assignment(
            pose, calibrations, tuple(perm), args.frame_stride, args.score_threshold
        )
        swap_results.append((cost - baseline_cost, tuple(perm), f"{pose.camera_names[i]} <-> {pose.camera_names[j]} ({n_pairs} paires)"))

    low_usage_idx = list(np.argsort(usage_counts)[:3])
    targeted = []
    for perm_low in it.permutations(low_usage_idx):
        perm = list(identity)
        for src, dst in zip(low_usage_idx, perm_low):
            perm[src] = dst
        if tuple(perm) == identity:
            continue
        cost, n_pairs = median_sampson_for_assignment(
            pose, calibrations, tuple(perm), args.frame_stride, args.score_threshold
        )
        targeted.append((cost - baseline_cost, tuple(perm), f"perm faibles vues {perm_low} ({n_pairs} paires)"))

    swap_results.sort(key=lambda item: item[0])
    targeted.sort(key=lambda item: item[0])

    print(f"\nMeilleurs swaps simples (delta negatif = meilleur que le mapping courant):")
    for delta, _, label in swap_results[: args.top_swaps]:
        print(f"  {label}: delta={delta:.3f} px")

    if targeted:
        print("\nPermutations ciblees sur les 3 cameras les moins utilisees :")
        for delta, perm, label in targeted[: min(args.top_swaps, len(targeted))]:
            remap = ", ".join(f"{pose.camera_names[i]}->{pose.camera_names[perm[i]]}" for i in low_usage_idx)
            print(f"  {label}: delta={delta:.3f} px | {remap}")


if __name__ == "__main__":
    main()
