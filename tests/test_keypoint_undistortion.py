"""Opt-in undistortion of 2D keypoints at load time (``undistort_keypoints``)."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from _ekf2d_synthetic import synthetic_cameras

from reconstruction.reconstruction_bundle import (
    epipolar_cache_metadata,
    keypoints_undistorted,
    load_or_compute_epipolar_cache,
)
from reconstruction.reconstruction_profiles import ReconstructionProfile, build_pipeline_command, validate_profile
from vitpose_ekf_pipeline import (
    COCO17,
    calibration_signature,
    calibrations_with_undistorted_keypoints,
    distort_normalized_points,
    load_pose_data,
    metadata_cache_matches,
    triangulate_pose2sim_like,
    undistort_pixel_points,
)

# Coefficients of camera M11139 in inputs/calibration/Calib.toml (k1, k2, p1, p2).
REAL_DIST = np.array([-0.1291770657092432, 0.14377615647497777, 0.0014812860769815334, 0.000248635362314446])
STRONG_DIST = np.array([-0.28, 0.09, 0.002, -0.001, 0.01])
K_REAL = np.array([[1237.12, 0.0, 955.02], [0.0, 1237.37, 540.54], [0.0, 0.0, 1.0]])


def _distort_pixels(points_px, K, dist):
    y = (points_px[..., 1] - K[1, 2]) / K[1, 1]
    x = (points_px[..., 0] - K[0, 2] - K[0, 1] * y) / K[0, 0]
    distorted = distort_normalized_points(np.stack((x, y), axis=-1), dist)
    return np.stack(
        (K[0, 0] * distorted[..., 0] + K[0, 1] * distorted[..., 1] + K[0, 2], K[1, 1] * distorted[..., 1] + K[1, 2]),
        axis=-1,
    )


def _image_grid():
    u, v = np.meshgrid(np.linspace(0.0, 1919.0, 25), np.linspace(0.0, 1079.0, 15))
    return np.stack((u.ravel(), v.ravel()), axis=-1)


@pytest.mark.parametrize("dist", [REAL_DIST, STRONG_DIST])
def test_distortion_round_trip_is_exact_over_the_image(dist):
    ideal = _image_grid()
    distorted = _distort_pixels(ideal, K_REAL, dist)

    recovered = undistort_pixel_points(distorted, K_REAL, dist)

    np.testing.assert_allclose(recovered, ideal, atol=1e-6, rtol=0.0)
    assert np.max(np.linalg.norm(distorted - ideal, axis=1)) > 5.0  # the distortion is not negligible


def test_undistortion_matches_opencv_when_available():
    cv2 = pytest.importorskip("cv2")
    ideal = _image_grid()
    object_points = np.column_stack(
        (
            (ideal[:, 0] - K_REAL[0, 2]) / K_REAL[0, 0],
            (ideal[:, 1] - K_REAL[1, 2]) / K_REAL[1, 1],
            np.ones(ideal.shape[0]),
        )
    )
    projected, _ = cv2.projectPoints(object_points, np.zeros(3), np.zeros(3), K_REAL, REAL_DIST)
    np.testing.assert_allclose(_distort_pixels(ideal, K_REAL, REAL_DIST), projected.reshape(-1, 2), atol=1e-8)

    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-14)
    if hasattr(cv2, "undistortPointsIter"):  # OpenCV 4.x
        reference = cv2.undistortPointsIter(projected, K_REAL, REAL_DIST, None, K_REAL, criteria)
    else:  # OpenCV 5.x exposes the criteria on undistortPoints
        reference = cv2.undistortPoints(projected, K_REAL, REAL_DIST, None, None, K_REAL, criteria)
    reference = reference.reshape(-1, 2)
    np.testing.assert_allclose(
        undistort_pixel_points(projected.reshape(-1, 2), K_REAL, REAL_DIST), reference, atol=1e-5
    )


def test_undistortion_is_identity_without_distortion_and_preserves_nan():
    points = np.array([[10.0, 20.0], [np.nan, np.nan], [1900.0, 1000.0]])

    np.testing.assert_array_equal(undistort_pixel_points(points, K_REAL, np.zeros(5)), points)
    undistorted = undistort_pixel_points(points, K_REAL, REAL_DIST)
    assert np.all(np.isnan(undistorted[1]))
    assert np.all(np.isfinite(undistorted[[0, 2]]))
    with pytest.raises(ValueError):
        undistort_pixel_points(points, K_REAL, np.r_[REAL_DIST, 0.0, 0.0, 0.0, 0.0, 0.1])


def _write_keypoints_json(path: Path, keypoints: np.ndarray, camera_names: list[str]) -> None:
    payload = {}
    n_frames = keypoints.shape[1]
    for cam_idx, name in enumerate(camera_names):
        payload[f"video_{name}"] = {
            "frames": list(range(n_frames)),
            "keypoints": np.nan_to_num(keypoints[cam_idx], nan=np.nan).tolist(),
            "scores": np.where(np.all(np.isfinite(keypoints[cam_idx]), axis=-1), 0.9, 0.0).tolist(),
        }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _synthetic_points(n_frames: int = 6) -> np.ndarray:
    rng = np.random.default_rng(4)
    # Spread over the trampoline volume so that points reach the image periphery.
    base = np.array([0.0, 0.0, 1.2]) + rng.uniform(-2.0, 2.0, size=(len(COCO17), 3)) * np.array([1.0, 1.0, 0.6])
    return base[np.newaxis] + 0.01 * np.arange(n_frames)[:, np.newaxis, np.newaxis]


def _distorted_scene(tmp_path: Path, dist):
    calibrations = synthetic_cameras(4, dist=dist)
    names = list(calibrations)
    points_3d = _synthetic_points()
    keypoints = np.zeros((len(names), points_3d.shape[0], len(COCO17), 2))
    for cam_idx, name in enumerate(names):
        calibration = calibrations[name]
        ideal = np.array([[calibration.project_point(p) for p in frame] for frame in points_3d])
        keypoints[cam_idx] = _distort_pixels(ideal, calibration.K, calibration.dist)
    keypoints[0, 2, 3] = np.nan
    keypoints_path = tmp_path / "synthetic_keypoints.json"
    _write_keypoints_json(keypoints_path, keypoints, names)
    return calibrations, keypoints_path, points_3d


def test_load_pose_data_undistort_is_invariant_without_distortion(tmp_path):
    calibrations, keypoints_path, _points = _distorted_scene(tmp_path, dist=np.zeros(4))

    reference = load_pose_data(keypoints_path, calibrations, data_mode="raw")
    undistorted = load_pose_data(keypoints_path, calibrations, data_mode="raw", undistort_keypoints=True)

    np.testing.assert_array_equal(undistorted.keypoints, reference.keypoints)
    np.testing.assert_array_equal(undistorted.scores, reference.scores)


def test_undistortion_improves_triangulation_reprojection_on_distorted_scene(tmp_path):
    calibrations, keypoints_path, points_3d = _distorted_scene(tmp_path, dist=STRONG_DIST)

    pose_raw = load_pose_data(keypoints_path, calibrations, data_mode="raw")
    flagged = calibrations_with_undistorted_keypoints(calibrations)
    pose_undistorted = load_pose_data(keypoints_path, flagged, data_mode="raw", undistort_keypoints=True)
    assert np.isnan(pose_undistorted.keypoints[0, 2, 3, 0]) and pose_undistorted.scores[0, 2, 3] == 0.0

    kwargs = dict(error_threshold_px=None, min_cameras_for_triangulation=2, triangulation_method="once", n_workers=1)
    rec_raw = triangulate_pose2sim_like(pose_raw, calibrations, **kwargs)
    rec_undistorted = triangulate_pose2sim_like(pose_undistorted, flagged, **kwargs)

    raw_reprojection = float(np.nanmean(rec_raw.reprojection_error))
    undistorted_reprojection = float(np.nanmean(rec_undistorted.reprojection_error))
    assert raw_reprojection > 0.5
    assert undistorted_reprojection < 1e-4
    raw_3d_error = np.nanmean(np.linalg.norm(rec_raw.points_3d - points_3d, axis=-1))
    undistorted_3d_error = np.nanmean(np.linalg.norm(rec_undistorted.points_3d - points_3d, axis=-1))
    assert undistorted_3d_error < 1e-6 < raw_3d_error


def test_calibration_signature_tracks_undistort_option_and_coefficients():
    calibrations = synthetic_cameras(3, dist=REAL_DIST)
    names = list(calibrations)
    base = calibration_signature(calibrations, names)
    explicit_off = {name: replace(cal, keypoints_undistorted=False) for name, cal in calibrations.items()}
    flagged = calibrations_with_undistorted_keypoints(calibrations)
    other_dist = {name: replace(cal, dist=cal.dist * 1.01) for name, cal in flagged.items()}

    assert calibration_signature(explicit_off, names) == base
    assert calibration_signature(flagged, names) != base
    assert calibration_signature(other_dist, names) != calibration_signature(flagged, names)
    assert keypoints_undistorted(flagged) and not keypoints_undistorted(calibrations)


def test_epipolar_cache_is_not_reused_across_undistort_modes(tmp_path):
    calibrations, keypoints_path, _points = _distorted_scene(tmp_path, dist=REAL_DIST)
    pose_raw = load_pose_data(keypoints_path, calibrations, data_mode="raw")
    kwargs = dict(
        output_dir=tmp_path / "out",
        coherence_method="epipolar",
        epipolar_threshold_px=15.0,
        pose_data_mode="raw",
        pose_filter_window=9,
        pose_outlier_threshold_ratio=0.1,
        pose_amplitude_lower_percentile=5.0,
        pose_amplitude_upper_percentile=95.0,
    )
    _c, _t, raw_path, source = load_or_compute_epipolar_cache(pose_data=pose_raw, calibrations=calibrations, **kwargs)
    assert source == "computed_now"
    _c, _t, _p, source = load_or_compute_epipolar_cache(pose_data=pose_raw, calibrations=calibrations, **kwargs)
    assert source == "cache"

    # Same 2D content but calibrations flagged: the cache entry must not be reused.
    flagged = calibrations_with_undistorted_keypoints(calibrations)
    flagged_metadata = epipolar_cache_metadata(pose_raw, 15.0, "sampson", "raw", 9, 0.1, 5.0, 95.0, flagged)
    assert not metadata_cache_matches(raw_path, flagged_metadata)
    pose_undistorted = load_pose_data(keypoints_path, flagged, data_mode="raw", undistort_keypoints=True)
    _c, _t, undistorted_path, source = load_or_compute_epipolar_cache(
        pose_data=pose_undistorted, calibrations=flagged, **kwargs
    )
    assert source == "computed_now"
    assert undistorted_path != raw_path


def test_profile_undistort_option_is_named_and_forwarded(tmp_path):
    profile = validate_profile(ReconstructionProfile(name="", family="ekf_2d", undistort_keypoints=True))
    pose2sim = validate_profile(ReconstructionProfile(name="", family="pose2sim", undistort_keypoints=True))

    assert profile.name.endswith("undist")
    assert "--undistort-keypoints" in build_pipeline_command(profile, tmp_path, Path("c.toml"), Path("k.json"))
    assert pose2sim.undistort_keypoints is False
    default = validate_profile(ReconstructionProfile(name="", family="ekf_2d"))
    assert "--undistort-keypoints" not in build_pipeline_command(default, tmp_path, Path("c.toml"), Path("k.json"))
