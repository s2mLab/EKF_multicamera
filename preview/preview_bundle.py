#!/usr/bin/env python3
"""Dataset-first preview bundle helpers shared by GUI tabs."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from reconstruction.reconstruction_dataset import align_array_to_frames, load_bundle_entries
from kinematics.root_kinematics import TRUNK_ROTATION_NAMES, TRUNK_TRANSLATION_NAMES
from reconstruction.reconstruction_dataset import preferred_master_name


PreviewBundle = dict[str, object]


def empty_preview_bundle() -> PreviewBundle:
    """Return an empty preview-bundle payload with the expected top-level keys."""

    return {
        "frames": np.array([], dtype=int),
        "time_s": np.array([], dtype=float),
        "q_names": np.array([], dtype=object),
        "recon_3d": {},
        "recon_q": {},
        "recon_qdot": {},
        "recon_q_root": {},
        "recon_qdot_root": {},
        "recon_summary": {},
    }


def root_center(points: np.ndarray, *, left_hip_idx: int = 11, right_hip_idx: int = 12) -> np.ndarray:
    """Compute the per-frame hip midpoint used to align preview reconstructions."""

    left = points[:, left_hip_idx, :]
    right = points[:, right_hip_idx, :]
    return 0.5 * (left + right)


def align_to_reference(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """Align a moving 3D reconstruction to a reference using the first valid root center."""

    ref_root = root_center(reference)
    mov_root = root_center(moving)
    valid = np.where(np.all(np.isfinite(ref_root), axis=1) & np.all(np.isfinite(mov_root), axis=1))[0]
    if valid.size == 0:
        return np.asarray(moving, dtype=float)
    delta = ref_root[valid[0]] - mov_root[valid[0]]
    return np.asarray(moving, dtype=float) + delta[np.newaxis, np.newaxis, :]


def project_points_all_cameras(points_3d: np.ndarray, calibrations: dict, camera_names: list[str]) -> np.ndarray:
    """Project a full 3D point trajectory into every camera used by the preview."""

    n_frames, n_points, _ = points_3d.shape
    projections = np.full((len(camera_names), n_frames, n_points, 2), np.nan, dtype=float)
    for cam_idx, cam_name in enumerate(camera_names):
        calibration = calibrations[cam_name]
        for frame_idx in range(n_frames):
            for point_idx in range(n_points):
                point = points_3d[frame_idx, point_idx]
                if np.all(np.isfinite(point)):
                    projections[cam_idx, frame_idx, point_idx] = calibration.project_point(point)
    return projections


def _master_entry(entries: list[dict[str, object]]) -> dict[str, object] | None:
    """Choose the preview entry that defines the master frame/time grid."""

    if not entries:
        return None
    preferred = preferred_master_name([str(entry["name"]) for entry in entries])
    if preferred is None:
        return entries[0]
    for entry in entries:
        if str(entry["name"]) == preferred:
            return entry
    return entries[0]


def assemble_dataset_preview_bundle(
    entries: list[dict[str, object]],
    biomod_path: Path | None,
    marker_builder: Callable[[Path, np.ndarray], np.ndarray],
) -> PreviewBundle:
    """Assemble a common preview bundle from heterogeneous reconstruction entries."""

    bundle = empty_preview_bundle()
    master_entry = _master_entry(entries)
    if master_entry is None:
        return bundle

    master_frames = np.asarray(master_entry["frames"], dtype=int)
    master_time = np.asarray(master_entry["time_s"], dtype=float)
    bundle["frames"] = master_frames
    bundle["time_s"] = master_time

    for entry in entries:
        name = str(entry["name"])
        frames = np.asarray(entry["frames"], dtype=int)
        points_3d = np.asarray(entry["points_3d"], dtype=float)
        q = np.asarray(entry["q"], dtype=float)
        qdot = np.asarray(entry["qdot"], dtype=float)
        q_root = np.asarray(entry.get("q_root", np.empty((len(frames), 0), dtype=float)), dtype=float)
        qdot_root = np.asarray(entry.get("qdot_root", np.empty((len(frames), 0), dtype=float)), dtype=float)
        q_names = np.asarray(entry["q_names"], dtype=object)
        summary = dict(entry.get("summary", {}))
        points_3d_source = str(entry.get("points_3d_source", ""))

        if (
            biomod_path is not None
            and q.ndim == 2
            and q.shape[1] > len(TRUNK_TRANSLATION_NAMES) + len(TRUNK_ROTATION_NAMES)
            and (points_3d_source != "model_forward_kinematics" or not np.any(np.isfinite(points_3d)))
        ):
            # Rebuild marker trajectories when a model-based entry only stores generalized coordinates.
            points_3d = marker_builder(biomod_path, q)

        bundle["recon_summary"][name] = summary
        bundle["recon_3d"][name] = align_array_to_frames(points_3d, frames, master_frames)
        if q.size:
            bundle["recon_q"][name] = align_array_to_frames(q, frames, master_frames)
            if qdot.size:
                bundle["recon_qdot"][name] = align_array_to_frames(qdot, frames, master_frames)
            if bundle["q_names"].size == 0 and q_names.size:
                bundle["q_names"] = q_names
        if q_root.size:
            bundle["recon_q_root"][name] = align_array_to_frames(q_root, frames, master_frames)
        if qdot_root.size:
            bundle["recon_qdot_root"][name] = align_array_to_frames(qdot_root, frames, master_frames)

    return bundle


def load_dataset_preview_bundle(
    output_dir: Path,
    biomod_path: Path | None,
    marker_builder: Callable[[Path, np.ndarray], np.ndarray],
) -> PreviewBundle:
    """Load and assemble the dataset preview bundle from reconstruction outputs on disk."""

    entries = load_bundle_entries(output_dir)
    return assemble_dataset_preview_bundle(entries, biomod_path, marker_builder)
