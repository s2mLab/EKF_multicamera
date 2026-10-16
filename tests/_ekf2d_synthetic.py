"""Deterministic synthetic multi-camera scene for EKF2D tests.

The scene uses the repository's own ``build_biomod`` model (COCO17 markers),
pinhole cameras looking at the trampoline volume, and keypoints obtained by
projecting the model markers of a known ``q`` trajectory. Units follow the
pipeline contracts: metres/radians in the world frame (``z`` up), pixels in the
images.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vitpose_ekf_pipeline import (
    COCO17,
    KP_INDEX,
    CameraCalibration,
    PoseData,
    ReconstructionResult,
    SegmentLengths,
    build_biomod,
    ensure_local_imports,
    marker_name_list,
)


def synthetic_lengths() -> SegmentLengths:
    return SegmentLengths(
        trunk_height=0.6,
        head_length=0.2,
        shoulder_half_width=0.18,
        hip_half_width=0.12,
        upper_arm_length=0.3,
        forearm_length=0.25,
        thigh_length=0.45,
        shank_length=0.4,
        eye_offset_x=0.03,
        eye_offset_y=0.025,
        ear_offset_y=0.06,
    )


def load_synthetic_model(tmp_path: Path, model_variant: str = "single_trunk"):
    """Build and load the repository bioMod, skipping when biorbd/biobuddy are unavailable."""

    biorbd = pytest.importorskip("biorbd")
    ensure_local_imports()
    pytest.importorskip("biobuddy")
    biomod_path = tmp_path / f"synthetic_{model_variant}.bioMod"
    build_biomod(synthetic_lengths(), biomod_path, model_variant=model_variant)
    return biorbd.Model(str(biomod_path))


def look_at_camera(
    name: str,
    center: np.ndarray,
    target: np.ndarray,
    focal_px: float = 1200.0,
    dist: np.ndarray | None = None,
) -> CameraCalibration:
    """Pinhole camera at ``center`` whose optical axis points to ``target`` (world ``z`` up)."""

    center = np.asarray(center, dtype=float)
    forward = np.asarray(target, dtype=float) - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.vstack((right, down, forward))
    tvec = (-R @ center).reshape(3, 1)
    K = np.array([[focal_px, 0.0, 960.0], [0.0, focal_px, 540.0], [0.0, 0.0, 1.0]])
    from scipy.spatial.transform import Rotation

    rvec = Rotation.from_matrix(R).as_rotvec().reshape(3, 1)
    return CameraCalibration(
        name=name,
        image_size=(1920, 1080),
        K=K,
        dist=np.zeros(5) if dist is None else np.asarray(dist, dtype=float).reshape(-1),
        rvec=rvec,
        tvec=tvec,
        R=R,
        P=K @ np.hstack((R, tvec)),
    )


def synthetic_cameras(n_cameras: int = 5, dist: np.ndarray | None = None) -> dict[str, CameraCalibration]:
    target = np.array([0.0, 0.0, 1.2])
    cameras = {}
    for i_cam in range(n_cameras):
        angle = 2.0 * np.pi * i_cam / n_cameras + 0.3
        center = np.array([6.0 * np.cos(angle), 6.0 * np.sin(angle), 1.6 + 0.2 * i_cam])
        cameras[f"cam{i_cam}"] = look_at_camera(f"cam{i_cam}", center, target, dist=dist)
    return cameras


def model_keypoints_3d(model, q_trajectory: np.ndarray) -> np.ndarray:
    """Return COCO17 3D points ``(n_frames, 17, 3)`` of the model markers along ``q``."""

    marker_names = marker_name_list(model)
    points = np.full((q_trajectory.shape[0], len(COCO17), 3), np.nan)
    for frame_idx, q in enumerate(q_trajectory):
        markers = model.markers(np.asarray(q, dtype=float))
        for marker_idx, marker_name in enumerate(marker_names):
            if marker_name in KP_INDEX:
                points[frame_idx, KP_INDEX[marker_name]] = markers[marker_idx].to_array()
    return points


def synthetic_q_trajectory(model, n_frames: int, fps: float = 120.0, seed: int = 0) -> np.ndarray:
    """Smooth joint motion around a standing pose with the root around ``z = 1 m``."""

    rng = np.random.default_rng(seed)
    nq = model.nbQ()
    t = np.arange(n_frames) / fps
    amplitude = 0.15 * rng.random(nq)
    phase = 2.0 * np.pi * rng.random(nq)
    q = amplitude[np.newaxis, :] * np.sin(2.0 * np.pi * 1.5 * t[:, np.newaxis] + phase[np.newaxis, :])
    q[:, 2] += 1.0
    return q


def project_keypoints(points_3d: np.ndarray, calibrations: dict[str, CameraCalibration]) -> np.ndarray:
    """Project ``(n_frames, 17, 3)`` world points into ``(n_cam, n_frames, 17, 2)`` pixels (pinhole)."""

    names = list(calibrations)
    keypoints = np.full((len(names), points_3d.shape[0], points_3d.shape[1], 2), np.nan)
    for i_cam, name in enumerate(names):
        calibration = calibrations[name]
        for frame_idx in range(points_3d.shape[0]):
            for kp_idx in range(points_3d.shape[1]):
                point = points_3d[frame_idx, kp_idx]
                if np.all(np.isfinite(point)):
                    keypoints[i_cam, frame_idx, kp_idx] = calibration.project_point(point)
    return keypoints


def make_pose_and_reconstruction(
    keypoints: np.ndarray,
    points_3d: np.ndarray,
    calibrations: dict[str, CameraCalibration],
    scores: np.ndarray | None = None,
) -> tuple[PoseData, ReconstructionResult]:
    n_cam, n_frames, n_kp, _ = keypoints.shape
    if scores is None:
        scores = np.where(np.all(np.isfinite(keypoints), axis=-1), 0.9, 0.0)
    frames = np.arange(n_frames, dtype=int)
    pose_data = PoseData(camera_names=list(calibrations), frames=frames, keypoints=keypoints, scores=scores)
    coherence = np.ones((n_frames, n_kp, n_cam))
    reconstruction = ReconstructionResult(
        frames=frames,
        points_3d=np.asarray(points_3d, dtype=float),
        mean_confidence=np.full((n_frames, n_kp), 0.9),
        reprojection_error=np.zeros((n_frames, n_kp)),
        reprojection_error_per_view=np.zeros((n_frames, n_kp, n_cam)),
        multiview_coherence=coherence,
        epipolar_coherence=coherence,
        triangulation_coherence=coherence,
        excluded_views=np.zeros((n_frames, n_kp, n_cam), dtype=bool),
        coherence_method="epipolar",
    )
    return pose_data, reconstruction
