#!/usr/bin/env python3
"""Helpers to build comparable root kinematic series from multiple sources."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from kinematics.root_kinematics import (
    ROOT_Q_NAMES,
    ROOT_ROTATION_SLICE,
    TRUNK_ROOT_ROTATION_SCIPY_SEQUENCE,
    TRUNK_ROOT_ROTATION_SEQUENCE,
    TRUNK_ROTATION_NAMES,
    TRUNK_TRANSLATION_NAMES,
    build_root_rotation_matrices,
    centered_finite_difference,
    extract_root_from_q,
    root_z_correction_angle_from_points,
    rotation_unit_label,
    rotation_unit_scale,
)
from reconstruction.reconstruction_bundle import extract_root_from_points


def quantity_unit_label(quantity: str, family_is_translation: bool, rotation_unit: str) -> str:
    """Return the display unit for one root-series plot family/quantity pair."""

    if family_is_translation:
        return "m" if quantity == "q" else "m/s"
    return rotation_unit_label(rotation_unit, quantity)


def scale_root_series_rotations(series: np.ndarray, family_is_translation: bool, rotation_unit: str) -> np.ndarray:
    """Scale only the rotational root components to the requested display unit."""

    output = np.asarray(series, dtype=float)
    if family_is_translation:
        return output
    scaled = np.array(output, copy=True)
    scaled[:, ROOT_ROTATION_SLICE] *= rotation_unit_scale(rotation_unit)
    return scaled


def extract_root_from_qdot(q_names: np.ndarray, qdot_trajectory: np.ndarray) -> np.ndarray:
    """Extract root generalized velocities without re-wrapping rotation components."""

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
    """Build a root q or qdot series directly from geometric 3D points."""

    root_q, _ = extract_root_from_points(
        np.asarray(points_3d, dtype=float),
        bool(initial_rotation_correction),
        bool(unwrap_rotations),
    )
    if quantity == "q":
        return root_q
    return centered_finite_difference(root_q, dt)


def root_series_from_model_markers(
    q_series: np.ndarray,
    *,
    biomod_path: Path,
    marker_builder: Callable[[Path, np.ndarray], np.ndarray],
    marker_points: np.ndarray | None = None,
    quantity: str,
    dt: float,
    initial_rotation_correction: bool,
    unwrap_rotations: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one root series from model markers reconstructed from ``q``.

    This helper is used when we want to compare q-based reconstructions against
    triangulation/Pose2Sim using the same geometric trunk-frame extraction.
    """

    if marker_points is None:
        marker_points = marker_builder(Path(biomod_path), np.asarray(q_series, dtype=float))
    marker_points = np.asarray(marker_points, dtype=float)
    series = root_series_from_points(
        marker_points,
        quantity=quantity,
        dt=dt,
        initial_rotation_correction=initial_rotation_correction,
        unwrap_rotations=unwrap_rotations,
    )
    return series, marker_points


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
    """Build a root q or qdot series from generalized coordinates."""

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
    """Use precomputed root series, optionally recomputing qdot by finite difference."""

    root_q = np.asarray(root_q, dtype=float)
    if quantity == "q":
        return root_q
    if fd_qdot or qdot_root is None:
        return centered_finite_difference(root_q, dt)
    return np.asarray(qdot_root, dtype=float)


def root_rotation_matrices_from_series(
    root_q: np.ndarray,
    *,
    initial_rotation_correction_angle_rad: float = 0.0,
) -> np.ndarray:
    """Convert ordered root Euler angles into per-frame rotation matrices.

    When ``initial_rotation_correction_angle_rad`` is non-zero, the fixed trunk
    RT rotation is pre-multiplied so the output matches the raw body
    orientation ``R_body = Rz(alpha) * R(q)`` used by the model.
    """

    root_q = np.asarray(root_q, dtype=float)
    matrices = np.full((root_q.shape[0], 3, 3), np.nan, dtype=float)
    if root_q.shape[1] < 6:
        return matrices
    initial_rotation_matrix = None
    if abs(float(initial_rotation_correction_angle_rad)) > 1e-8:
        initial_rotation_matrix = Rotation.from_euler(
            "z",
            float(initial_rotation_correction_angle_rad),
            degrees=False,
        ).as_matrix()
    for frame_idx, angles in enumerate(root_q[:, ROOT_ROTATION_SLICE]):
        if not np.all(np.isfinite(angles)):
            continue
        matrix = Rotation.from_euler(TRUNK_ROOT_ROTATION_SCIPY_SEQUENCE, angles, degrees=False).as_matrix()
        if initial_rotation_matrix is not None:
            matrix = initial_rotation_matrix @ matrix
        matrices[frame_idx] = matrix
    return matrices


def root_rotation_matrices_from_points(
    points_3d: np.ndarray,
    *,
    initial_rotation_correction: bool,
) -> np.ndarray:
    """Build raw or alpha-corrected trunk rotation matrices from marker data."""

    points_3d = np.asarray(points_3d, dtype=float)
    _, matrices = build_root_rotation_matrices(points_3d)
    if not initial_rotation_correction:
        return matrices
    correction_angle = root_z_correction_angle_from_points(points_3d)
    if abs(correction_angle) <= 1e-8:
        return matrices
    corrected = np.array(matrices, copy=True)
    correction_matrix = Rotation.from_euler("z", -correction_angle, degrees=False).as_matrix()
    valid_mask = np.all(np.isfinite(corrected), axis=(1, 2))
    corrected[valid_mask] = correction_matrix @ corrected[valid_mask]
    return corrected


def root_axis_labels(family: str) -> list[str]:
    """Return the technical root DoF names for one plot family."""

    return TRUNK_TRANSLATION_NAMES if family == "translations" else TRUNK_ROTATION_NAMES


def root_axis_display_labels(family: str) -> list[str]:
    """Return user-facing labels for the displayed root components."""

    if family == "translations":
        return ["TransX", "TransY", "TransZ"]
    return ["Somersault", "Tilt", "Twist"]


def root_ordered_names() -> np.ndarray:
    """Return the canonical ordered list of root generalized-coordinate names."""

    return np.asarray(ROOT_Q_NAMES, dtype=object)
