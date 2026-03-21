#!/usr/bin/env python3
"""Camera-quality metrics to guide camera subset selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vitpose_ekf_pipeline import PoseData


@dataclass
class CameraMetricRow:
    camera_name: str
    valid_ratio: float
    mean_score: float | None
    mean_epipolar_coherence: float | None
    weighted_confidence: float | None
    reprojection_mean_px: float | None
    reprojection_good_frame_ratio: float | None
    triangulation_usage_ratio: float | None
    flip_rate_epipolar: float | None
    flip_rate_triangulation: float | None


def camera_metric_sort_key(row: CameraMetricRow) -> tuple[float, ...]:
    """Return a descending quality key used to suggest stable camera subsets."""
    return (
        _sort_value(row.weighted_confidence),
        _sort_value(row.mean_epipolar_coherence),
        _sort_value(row.triangulation_usage_ratio),
        _sort_value(row.reprojection_good_frame_ratio),
        -_sort_value(row.reprojection_mean_px),
        -_sort_value(row.flip_rate_epipolar),
        -_sort_value(row.flip_rate_triangulation),
        _sort_value(row.valid_ratio),
        _sort_value(row.mean_score),
    )


def suggest_best_camera_names(rows: list[CameraMetricRow], count: int) -> list[str]:
    ranked = sorted(rows, key=camera_metric_sort_key, reverse=True)
    return [row.camera_name for row in ranked[: max(0, int(count))]]


def compute_camera_metric_rows(
    pose_data: "PoseData",
    *,
    epipolar_coherence: np.ndarray | None = None,
    reprojection_error_per_view: np.ndarray | None = None,
    excluded_views: np.ndarray | None = None,
    flip_masks: dict[str, np.ndarray] | None = None,
    good_reprojection_threshold_px: float = 5.0,
) -> list[CameraMetricRow]:
    n_cams, n_frames, n_keypoints = pose_data.scores.shape
    flip_masks = flip_masks or {}
    valid_measurements = (pose_data.scores > 0) & np.all(np.isfinite(pose_data.keypoints), axis=-1)
    rows: list[CameraMetricRow] = []
    for cam_idx, camera_name in enumerate(pose_data.camera_names):
        valid_mask = valid_measurements[cam_idx]
        valid_count = int(np.count_nonzero(valid_mask))
        total_count = int(valid_mask.size)
        valid_ratio = float(valid_count / total_count) if total_count else 0.0
        mean_score = _finite_mean(np.where(valid_mask, pose_data.scores[cam_idx], np.nan))

        mean_epipolar = None
        weighted_confidence = None
        if epipolar_coherence is not None and epipolar_coherence.shape[:3] == (n_frames, n_keypoints, n_cams):
            coherence_cam = np.asarray(epipolar_coherence[:, :, cam_idx], dtype=float)
            coherence_valid = np.where(valid_mask, coherence_cam, np.nan)
            mean_epipolar = _finite_mean(coherence_valid)
            weighted_confidence = _finite_mean(np.where(valid_mask, coherence_cam * pose_data.scores[cam_idx], np.nan))

        reproj_mean = None
        reproj_good_ratio = None
        if reprojection_error_per_view is not None and reprojection_error_per_view.shape[:3] == (n_frames, n_keypoints, n_cams):
            reproj_cam = np.asarray(reprojection_error_per_view[:, :, cam_idx], dtype=float)
            reproj_mean = _finite_mean(np.where(valid_mask, reproj_cam, np.nan))
            masked_reproj = np.where(valid_mask, reproj_cam, np.nan)
            valid_per_frame = np.count_nonzero(np.isfinite(masked_reproj), axis=1)
            per_frame_sum = np.nansum(masked_reproj, axis=1)
            per_frame_mean = np.full(n_frames, np.nan, dtype=float)
            valid_frames = valid_per_frame > 0
            per_frame_mean[valid_frames] = per_frame_sum[valid_frames] / valid_per_frame[valid_frames]
            finite_frame_mean = per_frame_mean[np.isfinite(per_frame_mean)]
            if finite_frame_mean.size:
                reproj_good_ratio = float(np.mean(finite_frame_mean <= float(good_reprojection_threshold_px)))

        usage_ratio = None
        if excluded_views is not None and excluded_views.shape[:3] == (n_frames, n_keypoints, n_cams):
            used = valid_mask & ~np.asarray(excluded_views[:, :, cam_idx], dtype=bool)
            usage_ratio = float(np.count_nonzero(used) / valid_count) if valid_count else None

        flip_rate_epipolar = _flip_rate(flip_masks.get("epipolar"), cam_idx, n_frames)
        flip_rate_triangulation = _flip_rate(flip_masks.get("triangulation"), cam_idx, n_frames)
        rows.append(
            CameraMetricRow(
                camera_name=str(camera_name),
                valid_ratio=valid_ratio,
                mean_score=mean_score,
                mean_epipolar_coherence=mean_epipolar,
                weighted_confidence=weighted_confidence,
                reprojection_mean_px=reproj_mean,
                reprojection_good_frame_ratio=reproj_good_ratio,
                triangulation_usage_ratio=usage_ratio,
                flip_rate_epipolar=flip_rate_epipolar,
                flip_rate_triangulation=flip_rate_triangulation,
            )
        )
    return rows


def _finite_mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _flip_rate(mask: np.ndarray | None, cam_idx: int, n_frames: int) -> float | None:
    if mask is None or mask.ndim != 2 or cam_idx >= mask.shape[0]:
        return None
    if n_frames <= 0:
        return None
    return float(np.count_nonzero(mask[cam_idx]) / n_frames)


def _sort_value(value: float | None) -> float:
    return float(value) if value is not None and np.isfinite(value) else float("-inf")
