#!/usr/bin/env python3
"""Shared helpers for root kinematics extraction and conversion."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

TRUNK_TRANSLATION_NAMES = ["TRUNK:TransX", "TRUNK:TransY", "TRUNK:TransZ"]
TRUNK_ROTATION_NAMES = ["TRUNK:RotY", "TRUNK:RotX", "TRUNK:RotZ"]
TRUNK_ROOT_ROTATION_SEQUENCE = "yxz"
TRUNK_ROOT_ROTATION_SCIPY_SEQUENCE = TRUNK_ROOT_ROTATION_SEQUENCE.upper()
SUPPORTED_ROOT_UNWRAP_MODES = ("off", "single", "double")
ROOT_Q_NAMES = np.asarray(TRUNK_TRANSLATION_NAMES + TRUNK_ROTATION_NAMES, dtype=object)
# Shared slices avoid hard-coded root DoF indexing across the codebase.
ROOT_TRANSLATION_SLICE = slice(0, len(TRUNK_TRANSLATION_NAMES))
ROOT_ROTATION_SLICE = slice(len(TRUNK_TRANSLATION_NAMES), len(TRUNK_TRANSLATION_NAMES) + len(TRUNK_ROTATION_NAMES))
RIGHT_ANGLE_RAD = 0.5 * math.pi


def normalize(vector: np.ndarray) -> np.ndarray:
    """Return a unit vector, or NaNs when the input norm is degenerate."""

    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < 1e-12:
        return np.full(3, np.nan)
    return np.asarray(vector, dtype=float) / norm


def unwrap_with_gaps(values: np.ndarray) -> np.ndarray:
    """Unwrap angle trajectories while treating NaN gaps as segment boundaries."""

    array = np.asarray(values, dtype=float)
    squeeze = array.ndim == 1
    if squeeze:
        array = array[:, np.newaxis]
    unwrapped = np.array(array, copy=True)
    for col_idx in range(unwrapped.shape[1]):
        column = unwrapped[:, col_idx]
        valid_idx = np.flatnonzero(np.isfinite(column))
        if valid_idx.size == 0:
            continue
        split_points = np.where(np.diff(valid_idx) > 1)[0] + 1
        for segment in np.split(valid_idx, split_points):
            if segment.size:
                unwrapped[segment, col_idx] = np.unwrap(column[segment])
    return unwrapped[:, 0] if squeeze else unwrapped


def scipy_intrinsic_sequence(sequence: str) -> str:
    """Map one bioMod Euler sequence to SciPy's intrinsic-sequence notation."""

    return str(sequence).upper()


def reextract_euler_with_gaps(rotations: np.ndarray, sequence: str = TRUNK_ROOT_ROTATION_SEQUENCE) -> np.ndarray:
    """Re-extract Euler angles frame by frame without bridging missing segments."""

    array = np.asarray(rotations, dtype=float)
    scipy_sequence = scipy_intrinsic_sequence(sequence)
    reextracted = np.full_like(array, np.nan, dtype=float)
    valid_idx = np.flatnonzero(np.all(np.isfinite(array), axis=1))
    if valid_idx.size == 0:
        return reextracted
    split_points = np.where(np.diff(valid_idx) > 1)[0] + 1
    for segment in np.split(valid_idx, split_points):
        for frame_idx in segment:
            rotation_matrix = Rotation.from_euler(scipy_sequence, array[frame_idx], degrees=False).as_matrix()
            reextracted[frame_idx] = Rotation.from_matrix(rotation_matrix).as_euler(scipy_sequence, degrees=False)
    return reextracted


def normalize_root_unwrap_mode(mode: str | None, *, legacy_unwrap: bool | None = None) -> str:
    """Normalize one root-angle stabilization mode with backward compatibility."""

    candidate = "" if mode is None else str(mode).strip().lower()
    if not candidate:
        if legacy_unwrap is None:
            candidate = "single"
        else:
            candidate = "single" if bool(legacy_unwrap) else "off"
    if candidate not in SUPPORTED_ROOT_UNWRAP_MODES:
        raise ValueError(f"Unsupported root unwrap mode: {mode}")
    return candidate


def stabilize_root_rotations(
    rotations: np.ndarray, mode: str | None, *, legacy_unwrap: bool | None = None
) -> np.ndarray:
    """Apply root Euler reextraction / unwrap stabilization according to one explicit mode."""

    normalized_mode = normalize_root_unwrap_mode(mode, legacy_unwrap=legacy_unwrap)
    stabilized = np.array(rotations, copy=True)
    if normalized_mode == "off":
        return reextract_euler_with_gaps(stabilized, TRUNK_ROOT_ROTATION_SEQUENCE)
    passes = 1 if normalized_mode == "single" else 2
    for _ in range(passes):
        stabilized = reextract_euler_with_gaps(stabilized, TRUNK_ROOT_ROTATION_SEQUENCE)
        stabilized = unwrap_with_gaps(stabilized)
    return stabilized


def build_root_rotation_matrices(
    points_3d: np.ndarray,
    left_hip_idx: int = 11,
    right_hip_idx: int = 12,
    left_shoulder_idx: int = 5,
    right_shoulder_idx: int = 6,
    translation_origin: str = "pelvis",
) -> tuple[np.ndarray, np.ndarray]:
    """Build trunk translations and body-to-world rotation matrices from markers."""

    translations = np.full((points_3d.shape[0], 3), np.nan, dtype=float)
    rotation_matrices = np.full((points_3d.shape[0], 3, 3), np.nan, dtype=float)

    left_hip = points_3d[:, left_hip_idx]
    right_hip = points_3d[:, right_hip_idx]
    left_shoulder = points_3d[:, left_shoulder_idx]
    right_shoulder = points_3d[:, right_shoulder_idx]

    for frame_idx in range(points_3d.shape[0]):
        if not (
            np.all(np.isfinite(left_hip[frame_idx]))
            and np.all(np.isfinite(right_hip[frame_idx]))
            and np.all(np.isfinite(left_shoulder[frame_idx]))
            and np.all(np.isfinite(right_shoulder[frame_idx]))
        ):
            continue
        hip_center = 0.5 * (left_hip[frame_idx] + right_hip[frame_idx])
        shoulder_center = 0.5 * (left_shoulder[frame_idx] + right_shoulder[frame_idx])
        # The trunk frame uses z = longitudinal, y = medio-lateral, x = antero-posterior.
        z_axis = normalize(shoulder_center - hip_center)
        y_seed = 0.5 * (
            (left_hip[frame_idx] - right_hip[frame_idx]) + (left_shoulder[frame_idx] - right_shoulder[frame_idx])
        )
        y_axis_seed = normalize(y_seed)
        x_axis = normalize(np.cross(y_axis_seed, z_axis))
        y_axis = normalize(np.cross(z_axis, x_axis))
        if not (np.all(np.isfinite(x_axis)) and np.all(np.isfinite(y_axis)) and np.all(np.isfinite(z_axis))):
            continue
        translations[frame_idx] = shoulder_center if str(translation_origin) == "upper_trunk" else hip_center
        rotation_matrices[frame_idx] = np.column_stack((x_axis, y_axis, z_axis))
    return translations, rotation_matrices


def wrap_angle_pi(angle_rad: float) -> float:
    """Wrap an angle to ]-pi, pi] with a stable representation for +pi."""

    wrapped = (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi
    if abs(wrapped + math.pi) < 1e-12:
        return math.pi
    return wrapped


def root_z_correction_angle_from_rotation_matrices(rotation_matrices: np.ndarray) -> float:
    """Estimate the snapped initial yaw correction from raw trunk matrices."""

    for matrix in np.asarray(rotation_matrices, dtype=float):
        if not np.all(np.isfinite(matrix)):
            continue
        y_axis = matrix[:, 1]
        horizontal_y = np.asarray([y_axis[0], y_axis[1]], dtype=float)
        norm = np.linalg.norm(horizontal_y)
        if not np.isfinite(norm) or norm < 1e-12:
            continue
        heading = math.atan2(float(horizontal_y[0]), float(horizontal_y[1]))
        snapped = round(heading / RIGHT_ANGLE_RAD) * RIGHT_ANGLE_RAD
        return wrap_angle_pi(snapped)
    return 0.0


def root_z_correction_angle_from_points(
    points_3d: np.ndarray,
    left_hip_idx: int = 11,
    right_hip_idx: int = 12,
    left_shoulder_idx: int = 5,
    right_shoulder_idx: int = 6,
) -> float:
    """Estimate the initial yaw correction directly from 3D trunk markers."""

    _, rotation_matrices = build_root_rotation_matrices(
        points_3d,
        left_hip_idx=left_hip_idx,
        right_hip_idx=right_hip_idx,
        left_shoulder_idx=left_shoulder_idx,
        right_shoulder_idx=right_shoulder_idx,
    )
    return root_z_correction_angle_from_rotation_matrices(rotation_matrices)


def compute_trunk_dofs_from_points(
    points_3d: np.ndarray,
    unwrap_rotations: bool = True,
    translation_origin: str = "pelvis",
) -> np.ndarray:
    """Convert 3D trunk markers into the ordered root DoFs [T, R]."""

    translations, rotation_matrices = build_root_rotation_matrices(points_3d, translation_origin=translation_origin)
    rotations_xyz = np.full((points_3d.shape[0], 3), np.nan, dtype=float)
    for frame_idx in range(points_3d.shape[0]):
        matrix = rotation_matrices[frame_idx]
        if np.all(np.isfinite(matrix)):
            rotations_xyz[frame_idx] = Rotation.from_matrix(matrix).as_euler(
                TRUNK_ROOT_ROTATION_SCIPY_SEQUENCE,
                degrees=False,
            )
    if unwrap_rotations:
        rotations_xyz = unwrap_with_gaps(rotations_xyz)
    return np.hstack((translations, rotations_xyz))


def extract_root_from_q(
    q_names: np.ndarray,
    q_trajectory: np.ndarray,
    unwrap_rotations: bool = True,
    renormalize_rotations: bool = True,
    unwrap_mode: str | None = None,
) -> np.ndarray:
    """Extract ordered root DoFs from a full generalized-coordinate trajectory."""

    name_to_index = {str(name): idx for idx, name in enumerate(np.asarray(q_names, dtype=object))}
    root_q = np.full((q_trajectory.shape[0], len(ROOT_Q_NAMES)), np.nan, dtype=float)
    for out_idx, dof_name in enumerate(ROOT_Q_NAMES):
        if str(dof_name) in name_to_index:
            root_q[:, out_idx] = q_trajectory[:, name_to_index[str(dof_name)]]
    rotations = np.array(root_q[:, ROOT_ROTATION_SLICE], copy=True)
    if renormalize_rotations or unwrap_rotations:
        if unwrap_rotations:
            mode = normalize_root_unwrap_mode(unwrap_mode, legacy_unwrap=True)
            rotations = stabilize_root_rotations(rotations, mode)
        else:
            rotations = reextract_euler_with_gaps(rotations, TRUNK_ROOT_ROTATION_SEQUENCE)
        root_q[:, ROOT_ROTATION_SLICE] = rotations
    return root_q


def rotation_unit_scale(unit: str) -> float:
    """Return the multiplicative factor that converts radians to the requested unit."""

    if unit == "deg":
        return 180.0 / math.pi
    if unit == "turns":
        return 1.0 / (2.0 * math.pi)
    return 1.0


def rotation_unit_label(unit: str, quantity: str) -> str:
    """Return the display label for root rotation units."""

    if unit == "deg":
        return "deg" if quantity == "q" else "deg/s"
    if unit == "turns":
        return "turn" if quantity == "q" else "turn/s"
    return "rad" if quantity == "q" else "rad/s"


def centered_finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    """Compute a centered finite-difference derivative while preserving NaN gaps."""

    derivative = np.full_like(values, np.nan, dtype=float)
    if values.shape[0] < 2:
        return derivative
    for col_idx in range(values.shape[1]):
        column = values[:, col_idx]
        if np.isfinite(column[0]) and np.isfinite(column[1]):
            derivative[0, col_idx] = (column[1] - column[0]) / dt
        if np.isfinite(column[-1]) and np.isfinite(column[-2]):
            derivative[-1, col_idx] = (column[-1] - column[-2]) / dt
        for frame_idx in range(1, values.shape[0] - 1):
            if np.isfinite(column[frame_idx - 1]) and np.isfinite(column[frame_idx + 1]):
                derivative[frame_idx, col_idx] = (column[frame_idx + 1] - column[frame_idx - 1]) / (2.0 * dt)
    return derivative
