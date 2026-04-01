import numpy as np

import calibration_qc
from vitpose_ekf_pipeline import CameraCalibration, PoseData


def _make_camera(name: str, tx: float) -> CameraCalibration:
    K = np.eye(3, dtype=float)
    R = np.eye(3, dtype=float)
    tvec = np.array([[tx], [0.0], [0.0]], dtype=float)
    P = np.hstack((R, tvec))
    return CameraCalibration(
        name=name,
        image_size=(1920, 1080),
        K=K,
        dist=np.zeros(5, dtype=float),
        rvec=np.zeros(3, dtype=float),
        tvec=tvec,
        R=R,
        P=P,
    )


def test_compute_2d_calibration_qc_trims_worst_fraction(monkeypatch):
    keypoints = np.full((2, 10, 17, 2), np.nan, dtype=float)
    scores = np.zeros((2, 10, 17), dtype=float)
    keypoints[0, :, 0] = np.column_stack((np.arange(10, dtype=float), np.zeros(10, dtype=float)))
    keypoints[1, :, 0] = np.column_stack((np.arange(10, dtype=float), np.ones(10, dtype=float)))
    scores[:, :, 0] = 1.0
    pose_data = PoseData(camera_names=["cam0", "cam1"], frames=np.arange(10), keypoints=keypoints, scores=scores)
    calibrations = {"cam0": _make_camera("cam0", 0.0), "cam1": _make_camera("cam1", 1.0)}

    monkeypatch.setattr(
        calibration_qc,
        "sampson_error_pixels_vectorized",
        lambda _pts_a, _pts_b, _f: np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]], dtype=float),
    )

    qc = calibration_qc.compute_2d_calibration_qc(pose_data, calibrations, trim_fraction=0.15)

    assert qc.trim_threshold_px is not None
    assert qc.kept_ratio == 0.9
    assert qc.pairwise_sample_count[0, 1] == 9
    assert qc.pairwise_mean_px[0, 1] == 5.0
    assert np.isnan(qc.per_frame_mean_px[-1])


def test_compute_3d_calibration_qc_reports_spatial_non_uniformity():
    points_3d = np.full((6, 17, 3), np.nan, dtype=float)
    reproj = np.full((6, 17, 2), np.nan, dtype=float)
    excluded = np.zeros((6, 17, 2), dtype=bool)
    low_points = np.array([[-1.0, 0.0, 1.0], [-0.8, 0.0, 1.1], [-0.6, 0.0, 1.2]], dtype=float)
    high_points = np.array([[0.6, 0.0, 1.3], [0.8, 0.0, 1.4], [1.0, 0.0, 1.5]], dtype=float)
    for kp_idx in range(17):
        points_3d[:3, kp_idx] = low_points
        points_3d[3:, kp_idx] = high_points
        reproj[:3, kp_idx, :] = 1.0
        reproj[3:, kp_idx, :] = 10.0

    qc = calibration_qc.compute_3d_calibration_qc(
        {
            "points_3d": points_3d,
            "reprojection_error_per_view": reproj,
            "excluded_views": excluded,
        },
        camera_names=["cam0", "cam1"],
        spatial_bins=2,
    )

    assert qc is not None
    assert qc.spatial_uniformity_cv is not None
    assert qc.spatial_uniformity_cv > 0.0
    assert qc.spatial_uniformity_range_px == 9.0
    np.testing.assert_allclose(qc.spatial_axis_means_px["x"], np.array([1.0, 10.0]))
    finite_spatial = np.sort(qc.spatial_xz_mean_px[np.isfinite(qc.spatial_xz_mean_px)])
    np.testing.assert_allclose(finite_spatial, np.array([1.0, 10.0]))
    np.testing.assert_array_equal(np.sort(qc.spatial_xz_count[qc.spatial_xz_count > 0]), np.array([51, 51]))
    assert qc.view_usage_summary["per_camera"]["cam0"]["excluded_ratio"] == 0.0


def test_frame_camera_epipolar_errors_averages_other_views(monkeypatch):
    keypoints = np.full((3, 1, 17, 2), np.nan, dtype=float)
    scores = np.zeros((3, 1, 17), dtype=float)
    for cam_idx in range(3):
        keypoints[cam_idx, 0, 0] = [10.0 + cam_idx, 20.0]
        scores[cam_idx, 0, 0] = 1.0
    pose_data = PoseData(
        camera_names=["cam0", "cam1", "cam2"], frames=np.array([0]), keypoints=keypoints, scores=scores
    )
    calibrations = {
        "cam0": _make_camera("cam0", 0.0),
        "cam1": _make_camera("cam1", 1.0),
        "cam2": _make_camera("cam2", 2.0),
    }
    monkeypatch.setattr(
        calibration_qc,
        "sampson_error_pixels_vectorized",
        lambda _pts_a, _pts_b, _f: np.array([[2.0] + [np.nan] * 16, [4.0] + [np.nan] * 16], dtype=float),
    )

    values = calibration_qc.frame_camera_epipolar_errors(pose_data, calibrations, frame_idx=0, camera_idx=0)

    assert values[0] == 3.0
    assert np.isnan(values[1])
