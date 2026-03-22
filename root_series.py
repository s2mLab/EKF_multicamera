#!/usr/bin/env python3
"""Helpers to build comparable root kinematic series from multiple sources."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from reconstruction_bundle import extract_root_from_points
from root_kinematics import (
    ROOT_Q_NAMES,
    ROOT_ROTATION_SLICE,
    TRUNK_ROTATION_NAMES,
    TRUNK_ROOT_ROTATION_SEQUENCE,
    TRUNK_TRANSLATION_NAMES,
    centered_finite_difference,
    extract_root_from_q,
    rotation_unit_label,
    rotation_unit_scale,
)


def quantity_unit_label(quantity: str, family_is_translation: bool, rotation_unit: str) -> str:
    if family_is_translation:
        return "m" if quantity == "q" else "m/s"
    return rotation_unit_label(rotation_unit, quantity)


def scale_root_series_rotations(series: np.ndarray, family_is_translation: bool, rotation_unit: str) -> np.ndarray:
    output = np.asarray(series, dtype=float)
    if family_is_translation:
        return output
    scaled = np.array(output, copy=True)
    scaled[:, ROOT_ROTATION_SLICE] *= rotation_unit_scale(rotation_unit)
    return scaled


def extract_root_from_qdot(q_names: np.ndarray, qdot_trajectory: np.ndarray) -> np.ndarray:
    return extract_root_from_q(
        q_names,
        qdot_trajectory,
        unwrap_rotations=False,
        renormalize_rotations=False,
    )


def root_series_from_points(
    points_3d: np.ndarray,
    *,
    quantity: str,
    dt: float,
    initial_rotation_correction: bool,
    unwrap_rotations: bool,
) -> np.ndarray:
    root_q, _ = extract_root_from_points(
        np.asarray(points_3d, dtype=float),
        bool(initial_rotation_correction),
        bool(unwrap_rotations),
    )
    if quantity == "q":
        return root_q
    return centered_finite_difference(root_q, dt)


def root_series_from_q(
    q_names: np.ndarray,
    q: np.ndarray,
    *,
    quantity: str,
    dt: float,
    qdot: np.ndarray | None = None,
    fd_qdot: bool = False,
    unwrap_rotations: bool = True,
    renormalize_rotations: bool = True,
) -> np.ndarray:
    root_q = extract_root_from_q(
        q_names,
        np.asarray(q, dtype=float),
        unwrap_rotations=bool(unwrap_rotations),
        renormalize_rotations=bool(renormalize_rotations),
    )
    if quantity == "q":
        return root_q
    if fd_qdot or qdot is None:
        return centered_finite_difference(root_q, dt)
    return extract_root_from_qdot(q_names, np.asarray(qdot, dtype=float))


def root_series_from_precomputed(
    root_q: np.ndarray,
    *,
    quantity: str,
    dt: float,
    qdot_root: np.ndarray | None = None,
    fd_qdot: bool = False,
) -> np.ndarray:
    root_q = np.asarray(root_q, dtype=float)
    if quantity == "q":
        return root_q
    if fd_qdot or qdot_root is None:
        return centered_finite_difference(root_q, dt)
    return np.asarray(qdot_root, dtype=float)


def root_rotation_matrices_from_series(root_q: np.ndarray) -> np.ndarray:
    root_q = np.asarray(root_q, dtype=float)
    matrices = np.full((root_q.shape[0], 3, 3), np.nan, dtype=float)
    if root_q.shape[1] < 6:
        return matrices
    for frame_idx, angles in enumerate(root_q[:, ROOT_ROTATION_SLICE]):
        if not np.all(np.isfinite(angles)):
            continue
        matrices[frame_idx] = Rotation.from_euler(TRUNK_ROOT_ROTATION_SEQUENCE, angles, degrees=False).as_matrix()
    return matrices


def root_axis_labels(family: str) -> list[str]:
    return TRUNK_TRANSLATION_NAMES if family == "translations" else TRUNK_ROTATION_NAMES


def root_axis_display_labels(family: str) -> list[str]:
    if family == "translations":
        return ["TransX", "TransY", "TransZ"]
    return ["Somersault", "Tilt", "Twist"]


def root_ordered_names() -> np.ndarray:
    return np.asarray(ROOT_Q_NAMES, dtype=object)
