#!/usr/bin/env python3
"""Shared helpers for 3D reconstruction analysis views."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kinematics.root_kinematics import centered_finite_difference
from vitpose_ekf_pipeline import KP_INDEX

SEGMENT_LENGTH_DEFINITIONS = [
    ("Trunk", "left_shoulder", "left_hip", "right_shoulder", "right_hip"),
    ("Shoulders", "left_shoulder", "right_shoulder"),
    ("Hips", "left_hip", "right_hip"),
    ("L upper arm", "left_shoulder", "left_elbow"),
    ("R upper arm", "right_shoulder", "right_elbow"),
    ("L forearm", "left_elbow", "left_wrist"),
    ("R forearm", "right_elbow", "right_wrist"),
    ("L thigh", "left_hip", "left_knee"),
    ("R thigh", "right_hip", "right_knee"),
    ("L shank", "left_knee", "left_ankle"),
    ("R shank", "right_knee", "right_ankle"),
]


def _segment_length(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    """Return the frame-wise Euclidean distance between two 3D point series."""

    points_a = np.asarray(points_a, dtype=float)
    points_b = np.asarray(points_b, dtype=float)
    if points_a.shape != points_b.shape:
        raise ValueError("Both point series must share the same shape.")
    length = np.linalg.norm(points_a - points_b, axis=-1)
    invalid = ~np.all(np.isfinite(points_a), axis=-1) | ~np.all(np.isfinite(points_b), axis=-1)
    length[invalid] = np.nan
    return length


def segment_length_series(points_3d: np.ndarray) -> dict[str, np.ndarray]:
    """Compute frame-wise segment lengths from COCO17-like 3D keypoints."""

    points_3d = np.asarray(points_3d, dtype=float)
    if points_3d.ndim != 3 or points_3d.shape[2] != 3:
        raise ValueError("points_3d must have shape (n_frames, n_markers, 3).")
    series: dict[str, np.ndarray] = {}
    for definition in SEGMENT_LENGTH_DEFINITIONS:
        label = str(definition[0])
        if len(definition) == 3:
            point_a = points_3d[:, KP_INDEX[str(definition[1])], :]
            point_b = points_3d[:, KP_INDEX[str(definition[2])], :]
        else:
            point_a = 0.5 * (
                points_3d[:, KP_INDEX[str(definition[1])], :] + points_3d[:, KP_INDEX[str(definition[3])], :]
            )
            point_b = 0.5 * (
                points_3d[:, KP_INDEX[str(definition[2])], :] + points_3d[:, KP_INDEX[str(definition[4])], :]
            )
        series[label] = _segment_length(point_a, point_b)
    return series


def valid_segment_length_samples(series: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Drop NaNs from a segment-length dictionary before boxplotting."""

    samples: dict[str, np.ndarray] = {}
    for label, values in series.items():
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            samples[str(label)] = finite
    return samples


def _to_array(value) -> np.ndarray:
    """Convert biorbd-style vectors and numpy-like values into a float array."""

    if hasattr(value, "to_array"):
        return np.asarray(value.to_array(), dtype=float)
    return np.asarray(value, dtype=float)


def angular_momentum_series(model, q_series: np.ndarray, qdot_series: np.ndarray) -> np.ndarray:
    """Compute frame-wise 3D angular momentum from one model trajectory."""

    q_series = np.asarray(q_series, dtype=float)
    qdot_series = np.asarray(qdot_series, dtype=float)
    if q_series.ndim != 2 or qdot_series.ndim != 2:
        raise ValueError("q_series and qdot_series must both be 2D arrays.")
    if q_series.shape != qdot_series.shape:
        raise ValueError("q_series and qdot_series must share the same shape.")

    momentum = np.full((q_series.shape[0], 3), np.nan, dtype=float)
    for frame_idx, (q_values, qdot_values) in enumerate(zip(q_series, qdot_series)):
        if not (np.all(np.isfinite(q_values)) and np.all(np.isfinite(qdot_values))):
            continue
        try:
            value = model.angularMomentum(q_values, qdot_values)
        except TypeError:
            value = model.CalcAngularMomentum(q_values, qdot_values)
        momentum[frame_idx] = _to_array(value).reshape(-1)[:3]
    return momentum


@dataclass
class AngularMomentumPlotData:
    """Prepared angular-momentum series ready to plot."""

    time_s: np.ndarray
    components: np.ndarray
    norm: np.ndarray


def angular_momentum_plot_data(
    model,
    q_series: np.ndarray,
    qdot_series: np.ndarray | None,
    time_s: np.ndarray,
) -> AngularMomentumPlotData:
    """Build one angular-momentum plot payload, deriving qdot when needed."""

    q_series = np.asarray(q_series, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    if q_series.ndim != 2:
        raise ValueError("q_series must have shape (n_frames, n_q).")
    if time_s.shape[0] != q_series.shape[0]:
        raise ValueError("time_s length must match q_series.")
    if qdot_series is None:
        positive_dt = np.diff(time_s)
        positive_dt = positive_dt[np.isfinite(positive_dt) & (positive_dt > 0.0)]
        dt = float(np.median(positive_dt)) if positive_dt.size else 1.0
        qdot_series = centered_finite_difference(q_series, dt)
    components = angular_momentum_series(model, q_series, np.asarray(qdot_series, dtype=float))
    norm = np.linalg.norm(components, axis=1)
    norm[~np.all(np.isfinite(components), axis=1)] = np.nan
    return AngularMomentumPlotData(
        time_s=time_s,
        components=components,
        norm=norm,
    )
