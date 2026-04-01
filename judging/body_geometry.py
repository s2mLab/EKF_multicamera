#!/usr/bin/env python3
"""Shared geometric body-angle helpers used by DD and execution analysis."""

from __future__ import annotations

import numpy as np

from vitpose_ekf_pipeline import KP_INDEX


def vector_angle(vectors_a: np.ndarray, vectors_b: np.ndarray) -> np.ndarray:
    """Return the frame-wise angle between two vector series."""

    vectors_a = np.asarray(vectors_a, dtype=float)
    vectors_b = np.asarray(vectors_b, dtype=float)
    if vectors_a.shape != vectors_b.shape:
        raise ValueError("Both vector series must share the same shape.")
    norms_a = np.linalg.norm(vectors_a, axis=-1)
    norms_b = np.linalg.norm(vectors_b, axis=-1)
    denom = norms_a * norms_b
    cosine = np.full(vectors_a.shape[:-1], np.nan, dtype=float)
    valid = (
        np.all(np.isfinite(vectors_a), axis=-1)
        & np.all(np.isfinite(vectors_b), axis=-1)
        & np.isfinite(denom)
        & (denom > 1e-12)
    )
    if np.any(valid):
        cosine[valid] = np.sum(vectors_a[valid] * vectors_b[valid], axis=-1) / denom[valid]
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def midpoint(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    """Return the midpoint between two point trajectories."""

    return 0.5 * (np.asarray(points_a, dtype=float) + np.asarray(points_b, dtype=float))


def joint_angle_series(
    points_proximal: np.ndarray,
    points_joint: np.ndarray,
    points_distal: np.ndarray,
) -> np.ndarray:
    """Return the internal joint angle defined by three point series."""

    return vector_angle(
        np.asarray(points_proximal, dtype=float) - np.asarray(points_joint, dtype=float),
        np.asarray(points_distal, dtype=float) - np.asarray(points_joint, dtype=float),
    )


def knee_angle_series(points_3d: np.ndarray) -> np.ndarray:
    """Estimate the average internal knee angle from left and right leg markers."""

    points_3d = np.asarray(points_3d, dtype=float)
    left = joint_angle_series(
        points_3d[:, KP_INDEX["left_hip"], :],
        points_3d[:, KP_INDEX["left_knee"], :],
        points_3d[:, KP_INDEX["left_ankle"], :],
    )
    right = joint_angle_series(
        points_3d[:, KP_INDEX["right_hip"], :],
        points_3d[:, KP_INDEX["right_knee"], :],
        points_3d[:, KP_INDEX["right_ankle"], :],
    )
    return np.nanmean(np.column_stack((left, right)), axis=1)


def hip_angle_series(points_3d: np.ndarray) -> np.ndarray:
    """Estimate the average internal hip angle from trunk and thigh directions."""

    points_3d = np.asarray(points_3d, dtype=float)
    shoulder_center = midpoint(points_3d[:, KP_INDEX["left_shoulder"], :], points_3d[:, KP_INDEX["right_shoulder"], :])
    left = joint_angle_series(
        shoulder_center,
        points_3d[:, KP_INDEX["left_hip"], :],
        points_3d[:, KP_INDEX["left_knee"], :],
    )
    right = joint_angle_series(
        shoulder_center,
        points_3d[:, KP_INDEX["right_hip"], :],
        points_3d[:, KP_INDEX["right_knee"], :],
    )
    return np.nanmean(np.column_stack((left, right)), axis=1)


def arm_raise_series(points_3d: np.ndarray) -> np.ndarray:
    """Estimate how far the upper arms move away from the trunk."""

    points_3d = np.asarray(points_3d, dtype=float)
    hip_center = midpoint(points_3d[:, KP_INDEX["left_hip"], :], points_3d[:, KP_INDEX["right_hip"], :])
    left = vector_angle(
        points_3d[:, KP_INDEX["left_elbow"], :] - points_3d[:, KP_INDEX["left_shoulder"], :],
        hip_center - points_3d[:, KP_INDEX["left_shoulder"], :],
    )
    right = vector_angle(
        points_3d[:, KP_INDEX["right_elbow"], :] - points_3d[:, KP_INDEX["right_shoulder"], :],
        hip_center - points_3d[:, KP_INDEX["right_shoulder"], :],
    )
    return np.nanmean(np.column_stack((left, right)), axis=1)
