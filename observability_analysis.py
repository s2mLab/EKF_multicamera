#!/usr/bin/env python3
"""Helpers to inspect marker and observation Jacobian rank over time."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _to_array(value) -> np.ndarray:
    """Convert biorbd-style containers and numpy-like values into float arrays."""

    if hasattr(value, "to_array"):
        return np.asarray(value.to_array(), dtype=float)
    return np.asarray(value, dtype=float)


def stacked_marker_jacobian(model, q_values: np.ndarray) -> np.ndarray:
    """Stack per-marker 3D Jacobians into one dense matrix."""

    marker_jacobians = model.markersJacobian(np.asarray(q_values, dtype=float))
    blocks = [_to_array(jacobian) for jacobian in marker_jacobians]
    if not blocks:
        return np.zeros((0, int(model.nbQ())), dtype=float)
    return np.vstack(blocks)


def stacked_observation_jacobian(
    model,
    q_values: np.ndarray,
    camera_calibrations: list[object],
) -> np.ndarray:
    """Stack projected 2D observation Jacobians over all visible cameras and markers."""

    q_values = np.asarray(q_values, dtype=float)
    marker_positions = [_to_array(marker) for marker in model.markers(q_values)]
    marker_jacobians = [_to_array(jacobian) for jacobian in model.markersJacobian(q_values)]
    if not marker_positions or not camera_calibrations:
        return np.zeros((0, int(model.nbQ())), dtype=float)

    marker_points_array = np.asarray(marker_positions, dtype=float)
    marker_jacobians_array = np.asarray(marker_jacobians, dtype=float)
    finite_markers = np.all(np.isfinite(marker_points_array), axis=1)
    blocks: list[np.ndarray] = []
    for calibration in camera_calibrations:
        projected_uv, projected_jac = calibration.project_points_and_jacobians(marker_points_array)
        H_q_blocks = np.einsum("mab,mbq->maq", projected_jac, marker_jacobians_array, optimize=True)
        # Keep only rows where the marker and its projected Jacobian are finite.
        valid = (
            finite_markers
            & np.all(np.isfinite(projected_uv), axis=1)
            & np.all(np.isfinite(H_q_blocks.reshape(H_q_blocks.shape[0], -1)), axis=1)
        )
        if np.any(valid):
            blocks.append(H_q_blocks[valid].reshape(-1, marker_jacobians_array.shape[-1]))
    if not blocks:
        return np.zeros((0, int(model.nbQ())), dtype=float)
    return np.vstack(blocks)


def matrix_rank_with_full_rank(matrix: np.ndarray) -> tuple[int, int]:
    """Return the finite-row matrix rank and its maximum achievable rank."""

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D matrix.")
    if matrix.size == 0:
        return 0, 0
    finite_rows = np.all(np.isfinite(matrix), axis=1)
    finite_matrix = matrix[finite_rows]
    if finite_matrix.size == 0:
        return 0, min(matrix.shape)
    return int(np.linalg.matrix_rank(finite_matrix)), int(min(finite_matrix.shape))


@dataclass
class ObservabilityRankSeries:
    """Frame-wise marker and observation rank trajectories."""

    marker_rank: np.ndarray
    marker_full_rank: int
    observation_rank: np.ndarray
    observation_full_rank: int


def compute_observability_rank_series(
    model,
    q_series: np.ndarray,
    camera_calibrations: list[object],
) -> ObservabilityRankSeries:
    """Compute marker and observation Jacobian rank at every frame."""

    q_series = np.asarray(q_series, dtype=float)
    if q_series.ndim != 2:
        raise ValueError("Expected q_series with shape (n_frames, n_q).")

    marker_ranks = np.zeros(q_series.shape[0], dtype=int)
    observation_ranks = np.zeros(q_series.shape[0], dtype=int)
    marker_full_rank = 0
    observation_full_rank = 0

    for frame_idx, q_values in enumerate(q_series):
        marker_jacobian = stacked_marker_jacobian(model, q_values)
        marker_rank, marker_full_rank = matrix_rank_with_full_rank(marker_jacobian)
        marker_ranks[frame_idx] = marker_rank

        observation_jacobian = stacked_observation_jacobian(model, q_values, camera_calibrations)
        observation_rank, observation_full_rank = matrix_rank_with_full_rank(observation_jacobian)
        observation_ranks[frame_idx] = observation_rank

    return ObservabilityRankSeries(
        marker_rank=marker_ranks,
        marker_full_rank=int(marker_full_rank),
        observation_rank=observation_ranks,
        observation_full_rank=int(observation_full_rank),
    )


def summarize_rank_series(ranks: np.ndarray, full_rank: int) -> dict[str, float]:
    """Summarize a rank trajectory for display in the GUI."""

    ranks = np.asarray(ranks, dtype=float)
    if ranks.size == 0:
        return {"min": 0.0, "median": 0.0, "max": 0.0, "full_rank_ratio": 0.0}
    return {
        "min": float(np.min(ranks)),
        "median": float(np.median(ranks)),
        "max": float(np.max(ranks)),
        "full_rank_ratio": float(np.mean(ranks >= float(full_rank))) if full_rank > 0 else 0.0,
    }
