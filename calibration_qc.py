#!/usr/bin/env python3
"""Calibration quality analysis helpers for 2D and 3D diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reconstruction.reconstruction_bundle import summarize_reprojection_errors, summarize_view_usage
from vitpose_ekf_pipeline import COCO17, CameraCalibration, PoseData, sampson_error_pixels_vectorized


@dataclass
class Calibration2DQC:
    trim_fraction: float
    trim_threshold_px: float | None
    kept_ratio: float
    pairwise_median_px: np.ndarray
    pairwise_mean_px: np.ndarray
    pairwise_sample_count: np.ndarray
    per_camera_median_px: np.ndarray
    per_camera_mean_px: np.ndarray
    per_frame_mean_px: np.ndarray
    per_keypoint_mean_px: np.ndarray


@dataclass
class Calibration3DQC:
    reprojection_summary: dict[str, object]
    view_usage_summary: dict[str, object]
    point_positions: np.ndarray
    point_errors_px: np.ndarray
    per_frame_mean_px: np.ndarray
    spatial_bin_mean_px: np.ndarray
    spatial_bin_count: np.ndarray
    spatial_xz_mean_px: np.ndarray
    spatial_xz_count: np.ndarray
    spatial_uniformity_cv: float | None
    spatial_uniformity_range_px: float | None
    spatial_axis_means_px: dict[str, np.ndarray]


@dataclass
class CalibrationQC:
    two_d: Calibration2DQC
    three_d: Calibration3DQC | None


def compute_calibration_qc(
    pose_data: PoseData,
    calibrations: dict[str, CameraCalibration],
    *,
    reconstruction_payload: dict[str, np.ndarray] | None = None,
    trim_fraction: float = 0.15,
    spatial_bins: int = 3,
) -> CalibrationQC:
    """Compute a compact 2D/3D calibration-quality summary."""

    two_d = compute_2d_calibration_qc(pose_data, calibrations, trim_fraction=trim_fraction)
    three_d = None
    if reconstruction_payload:
        three_d = compute_3d_calibration_qc(
            reconstruction_payload,
            camera_names=list(pose_data.camera_names),
            spatial_bins=spatial_bins,
        )
    return CalibrationQC(two_d=two_d, three_d=three_d)


def compute_2d_calibration_qc(
    pose_data: PoseData,
    calibrations: dict[str, CameraCalibration],
    *,
    trim_fraction: float = 0.15,
) -> Calibration2DQC:
    """Summarize pairwise epipolar consistency after trimming the worst 2D samples."""

    camera_names = list(pose_data.camera_names)
    n_cams = len(camera_names)
    n_frames = int(pose_data.frames.shape[0])
    n_keypoints = len(COCO17)

    pairwise_median = np.full((n_cams, n_cams), np.nan, dtype=float)
    pairwise_mean = np.full((n_cams, n_cams), np.nan, dtype=float)
    pairwise_count = np.zeros((n_cams, n_cams), dtype=int)

    valid_mask = (pose_data.scores > 0) & np.all(np.isfinite(pose_data.keypoints), axis=-1)
    sample_errors: list[np.ndarray] = []
    sample_camera_i: list[np.ndarray] = []
    sample_camera_j: list[np.ndarray] = []
    sample_frames: list[np.ndarray] = []
    sample_keypoints: list[np.ndarray] = []

    for cam_i in range(n_cams):
        calib_i = calibrations[camera_names[cam_i]]
        for cam_j in range(cam_i + 1, n_cams):
            calib_j = calibrations[camera_names[cam_j]]
            mask_ij = valid_mask[cam_i] & valid_mask[cam_j]
            if not np.any(mask_ij):
                continue
            points_i = np.asarray(pose_data.keypoints[cam_i][mask_ij], dtype=float)
            points_j = np.asarray(pose_data.keypoints[cam_j][mask_ij], dtype=float)
            if points_i.size == 0:
                continue
            errors = sampson_error_pixels_vectorized(
                points_i,
                points_j[np.newaxis, :, :],
                np.stack((fundamental_matrix(calib_i, calib_j),), axis=0),
            )[0]
            finite = np.isfinite(errors)
            if not np.any(finite):
                continue
            frame_idx, kp_idx = np.where(mask_ij)
            sample_errors.append(errors[finite])
            sample_camera_i.append(np.full(int(np.count_nonzero(finite)), cam_i, dtype=int))
            sample_camera_j.append(np.full(int(np.count_nonzero(finite)), cam_j, dtype=int))
            sample_frames.append(frame_idx[finite].astype(int, copy=False))
            sample_keypoints.append(kp_idx[finite].astype(int, copy=False))

    if sample_errors:
        all_errors = np.concatenate(sample_errors)
        all_camera_i = np.concatenate(sample_camera_i)
        all_camera_j = np.concatenate(sample_camera_j)
        all_frames = np.concatenate(sample_frames)
        all_keypoints = np.concatenate(sample_keypoints)
        kept_mask = _trim_upper_mask(all_errors, trim_fraction)
        trimmed_values = all_errors[np.isfinite(all_errors) & ~kept_mask]
        trim_threshold = float(np.min(trimmed_values)) if trimmed_values.size else None
        kept_ratio = float(np.mean(kept_mask)) if kept_mask.size else 0.0

        for cam_i in range(n_cams):
            for cam_j in range(cam_i + 1, n_cams):
                pair_mask = kept_mask & (all_camera_i == cam_i) & (all_camera_j == cam_j)
                if not np.any(pair_mask):
                    continue
                values = all_errors[pair_mask]
                pairwise_median[cam_i, cam_j] = pairwise_median[cam_j, cam_i] = float(np.median(values))
                pairwise_mean[cam_i, cam_j] = pairwise_mean[cam_j, cam_i] = float(np.mean(values))
                pairwise_count[cam_i, cam_j] = pairwise_count[cam_j, cam_i] = int(values.size)

        per_camera_median = np.full(n_cams, np.nan, dtype=float)
        per_camera_mean = np.full(n_cams, np.nan, dtype=float)
        for cam_idx in range(n_cams):
            cam_mask = kept_mask & ((all_camera_i == cam_idx) | (all_camera_j == cam_idx))
            if np.any(cam_mask):
                per_camera_median[cam_idx] = float(np.median(all_errors[cam_mask]))
                per_camera_mean[cam_idx] = float(np.mean(all_errors[cam_mask]))

        per_frame_sum = np.zeros(n_frames, dtype=float)
        per_frame_count = np.zeros(n_frames, dtype=int)
        np.add.at(per_frame_sum, all_frames[kept_mask], all_errors[kept_mask])
        np.add.at(per_frame_count, all_frames[kept_mask], 1)
        per_frame_mean = np.full(n_frames, np.nan, dtype=float)
        valid_frames = per_frame_count > 0
        per_frame_mean[valid_frames] = per_frame_sum[valid_frames] / per_frame_count[valid_frames]

        per_key_sum = np.zeros(n_keypoints, dtype=float)
        per_key_count = np.zeros(n_keypoints, dtype=int)
        np.add.at(per_key_sum, all_keypoints[kept_mask], all_errors[kept_mask])
        np.add.at(per_key_count, all_keypoints[kept_mask], 1)
        per_key_mean = np.full(n_keypoints, np.nan, dtype=float)
        valid_keys = per_key_count > 0
        per_key_mean[valid_keys] = per_key_sum[valid_keys] / per_key_count[valid_keys]
    else:
        trim_threshold = None
        kept_ratio = 0.0
        per_camera_median = np.full(n_cams, np.nan, dtype=float)
        per_camera_mean = np.full(n_cams, np.nan, dtype=float)
        per_frame_mean = np.full(n_frames, np.nan, dtype=float)
        per_key_mean = np.full(n_keypoints, np.nan, dtype=float)

    np.fill_diagonal(pairwise_median, 0.0)
    np.fill_diagonal(pairwise_mean, 0.0)
    np.fill_diagonal(pairwise_count, 0)
    return Calibration2DQC(
        trim_fraction=float(np.clip(trim_fraction, 0.0, 1.0)),
        trim_threshold_px=trim_threshold,
        kept_ratio=kept_ratio,
        pairwise_median_px=pairwise_median,
        pairwise_mean_px=pairwise_mean,
        pairwise_sample_count=pairwise_count,
        per_camera_median_px=per_camera_median,
        per_camera_mean_px=per_camera_mean,
        per_frame_mean_px=per_frame_mean,
        per_keypoint_mean_px=per_key_mean,
    )


def frame_camera_epipolar_errors(
    pose_data: PoseData,
    calibrations: dict[str, CameraCalibration],
    *,
    frame_idx: int,
    camera_idx: int,
) -> np.ndarray:
    """Return one epipolar-error value per keypoint for one camera/frame."""

    camera_names = list(pose_data.camera_names)
    n_cams = len(camera_names)
    result = np.full(len(COCO17), np.nan, dtype=float)
    if camera_idx < 0 or camera_idx >= n_cams or frame_idx < 0 or frame_idx >= pose_data.frames.shape[0]:
        return result
    candidate_points = np.asarray(pose_data.keypoints[camera_idx, frame_idx], dtype=float)
    candidate_scores = np.asarray(pose_data.scores[camera_idx, frame_idx], dtype=float)
    other_indices = [idx for idx in range(n_cams) if idx != camera_idx]
    if not other_indices:
        return result
    other_points = np.asarray(pose_data.keypoints[other_indices, frame_idx], dtype=float)
    other_scores = np.asarray(pose_data.scores[other_indices, frame_idx], dtype=float)
    valid_candidate = np.all(np.isfinite(candidate_points), axis=1) & (candidate_scores > 0)
    if not np.any(valid_candidate):
        return result
    valid_other = np.all(np.isfinite(other_points), axis=2) & (other_scores > 0)
    fundamental_blocks = np.stack(
        [
            fundamental_matrix(calibrations[camera_names[camera_idx]], calibrations[camera_names[other_idx]])
            for other_idx in other_indices
        ],
        axis=0,
    )
    errors = sampson_error_pixels_vectorized(candidate_points, other_points, fundamental_blocks)
    valid = valid_other & valid_candidate[np.newaxis, :] & np.isfinite(errors)
    if not np.any(valid):
        return result
    counts = np.count_nonzero(valid, axis=0)
    sums = np.nansum(np.where(valid, errors, np.nan), axis=0)
    finite = counts > 0
    result[finite] = sums[finite] / counts[finite]
    return result


def compute_3d_calibration_qc(
    reconstruction_payload: dict[str, np.ndarray],
    *,
    camera_names: list[str],
    spatial_bins: int = 3,
) -> Calibration3DQC | None:
    """Summarize reprojection quality and its spatial uniformity in 3D."""

    points_3d = reconstruction_payload.get("points_3d")
    reprojection_error_per_view = reconstruction_payload.get("reprojection_error_per_view")
    excluded_views = reconstruction_payload.get("excluded_views")
    if points_3d is None or reprojection_error_per_view is None:
        return None

    points_3d = np.asarray(points_3d, dtype=float)
    reprojection_error_per_view = np.asarray(reprojection_error_per_view, dtype=float)
    excluded_views = None if excluded_views is None else np.asarray(excluded_views, dtype=bool)
    if points_3d.ndim != 3 or reprojection_error_per_view.ndim != 3:
        return None

    reprojection_summary = summarize_reprojection_errors(reprojection_error_per_view, camera_names)
    view_usage_summary = summarize_view_usage(excluded_views, camera_names)

    point_error = np.nanmean(reprojection_error_per_view, axis=2)
    valid = np.all(np.isfinite(points_3d), axis=2) & np.isfinite(point_error)
    if not np.any(valid):
        empty_bins = np.full((1, 1, 1), np.nan, dtype=float)
        return Calibration3DQC(
            reprojection_summary=reprojection_summary,
            view_usage_summary=view_usage_summary,
            point_positions=np.empty((0, 3), dtype=float),
            point_errors_px=np.empty((0,), dtype=float),
            per_frame_mean_px=np.full(points_3d.shape[0], np.nan, dtype=float),
            spatial_bin_mean_px=empty_bins,
            spatial_bin_count=np.zeros((1, 1, 1), dtype=int),
            spatial_xz_mean_px=np.full((1, 1), np.nan, dtype=float),
            spatial_xz_count=np.zeros((1, 1), dtype=int),
            spatial_uniformity_cv=None,
            spatial_uniformity_range_px=None,
            spatial_axis_means_px={"x": np.full(1, np.nan), "y": np.full(1, np.nan), "z": np.full(1, np.nan)},
        )

    positions = points_3d[valid]
    errors = point_error[valid]
    per_frame_mean = np.nanmean(point_error, axis=1)
    x_idx, n_x = _quantile_bin_indices(positions[:, 0], spatial_bins)
    y_idx, n_y = _quantile_bin_indices(positions[:, 1], spatial_bins)
    z_idx, n_z = _quantile_bin_indices(positions[:, 2], spatial_bins)
    bin_sum = np.zeros((n_x, n_y, n_z), dtype=float)
    bin_count = np.zeros((n_x, n_y, n_z), dtype=int)
    np.add.at(bin_sum, (x_idx, y_idx, z_idx), errors)
    np.add.at(bin_count, (x_idx, y_idx, z_idx), 1)
    bin_mean = np.full_like(bin_sum, np.nan, dtype=float)
    occupied = bin_count > 0
    bin_mean[occupied] = bin_sum[occupied] / bin_count[occupied]
    xz_sum = np.sum(bin_sum, axis=1)
    xz_count = np.sum(bin_count, axis=1)
    xz_mean = np.full_like(xz_sum, np.nan, dtype=float)
    xz_occupied = xz_count > 0
    xz_mean[xz_occupied] = xz_sum[xz_occupied] / xz_count[xz_occupied]
    occupied_means = bin_mean[occupied]
    if occupied_means.size >= 2:
        overall_mean = float(np.mean(occupied_means))
        uniformity_cv = float(np.std(occupied_means) / overall_mean) if overall_mean > 1e-12 else None
        uniformity_range = float(np.max(occupied_means) - np.min(occupied_means))
    elif occupied_means.size == 1:
        uniformity_cv = 0.0
        uniformity_range = 0.0
    else:
        uniformity_cv = None
        uniformity_range = None

    spatial_axis_means = {
        "x": _axis_bin_means(x_idx, n_x, errors),
        "y": _axis_bin_means(y_idx, n_y, errors),
        "z": _axis_bin_means(z_idx, n_z, errors),
    }
    return Calibration3DQC(
        reprojection_summary=reprojection_summary,
        view_usage_summary=view_usage_summary,
        point_positions=positions,
        point_errors_px=errors,
        per_frame_mean_px=per_frame_mean,
        spatial_bin_mean_px=bin_mean,
        spatial_bin_count=bin_count,
        spatial_xz_mean_px=xz_mean,
        spatial_xz_count=xz_count,
        spatial_uniformity_cv=uniformity_cv,
        spatial_uniformity_range_px=uniformity_range,
        spatial_axis_means_px=spatial_axis_means,
    )


def _axis_bin_means(indices: np.ndarray, n_bins: int, errors: np.ndarray) -> np.ndarray:
    sums = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=int)
    np.add.at(sums, indices, errors)
    np.add.at(counts, indices, 1)
    means = np.full(n_bins, np.nan, dtype=float)
    valid = counts > 0
    means[valid] = sums[valid] / counts[valid]
    return means


def _quantile_bin_indices(values: np.ndarray, requested_bins: int) -> tuple[np.ndarray, int]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.zeros(0, dtype=int), 1
    n_bins = max(1, int(requested_bins))
    if n_bins == 1:
        return np.zeros(values.shape[0], dtype=int), 1
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges)
    if edges.size <= 1:
        return np.zeros(values.shape[0], dtype=int), 1
    bins = np.searchsorted(edges[1:-1], values, side="right")
    return np.asarray(bins, dtype=int), int(edges.size - 1)


def _trim_upper_mask(values: np.ndarray, trim_fraction: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(values)
    finite_idx = np.flatnonzero(keep)
    if finite_idx.size == 0:
        return keep
    fraction = float(np.clip(trim_fraction, 0.0, 1.0))
    n_remove = int(np.floor(finite_idx.size * fraction))
    if n_remove <= 0:
        return keep
    order = finite_idx[np.argsort(values[finite_idx])]
    keep[order[-n_remove:]] = False
    return keep


def fundamental_matrix(calib_a: CameraCalibration, calib_b: CameraCalibration) -> np.ndarray:
    """Small local wrapper to avoid importing the whole GUI stack."""

    from vitpose_ekf_pipeline import fundamental_matrix as _fundamental_matrix

    return _fundamental_matrix(calib_a, calib_b)
