import numpy as np

from vitpose_ekf_pipeline import (
    CameraCalibration,
    compute_camera_triangulation_cost,
    project_point_with_projection_matrices,
    triangulation_reference_from_other_views,
    weighted_triangulation,
)


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


def test_weighted_triangulation_recovers_known_point():
    point = np.array([0.2, -0.1, 2.0], dtype=float)
    cameras = [_make_camera("cam0", 0.0), _make_camera("cam1", 1.0)]
    observations = np.array([camera.project_point(point) for camera in cameras], dtype=float)
    reconstructed = weighted_triangulation([camera.P for camera in cameras], observations, np.array([1.0, 0.8]))
    np.testing.assert_allclose(reconstructed, point, atol=1e-8)


def test_triangulation_reference_matches_perfect_reprojection():
    point = np.array([0.15, 0.05, 2.5], dtype=float)
    cameras = [_make_camera("cam0", 0.0), _make_camera("cam1", 1.0), _make_camera("cam2", -0.8)]
    raw_2d_frame = np.stack([[camera.project_point(point)] for camera in cameras], axis=0)
    raw_scores_frame = np.ones((3, 1), dtype=float)

    references = triangulation_reference_from_other_views(
        raw_2d_frame,
        raw_scores_frame,
        cameras,
        min_other_cameras=2,
    )
    for cam_idx, camera in enumerate(cameras):
        np.testing.assert_allclose(references[cam_idx, 0], camera.project_point(point), atol=1e-8)


def test_project_point_with_projection_matrices_matches_camera_projection():
    point = np.array([0.15, 0.05, 2.5], dtype=float)
    cameras = [_make_camera("cam0", 0.0), _make_camera("cam1", 1.0), _make_camera("cam2", -0.8)]
    projected = project_point_with_projection_matrices(np.asarray([camera.P for camera in cameras], dtype=float), point)
    expected = np.asarray([camera.project_point(point) for camera in cameras], dtype=float)
    np.testing.assert_allclose(projected, expected, atol=1e-8)


def test_precomputed_triangulation_cost_matches_direct_computation():
    point = np.array([0.1, -0.2, 3.0], dtype=float)
    cameras = [_make_camera("cam0", 0.0), _make_camera("cam1", 0.8), _make_camera("cam2", -0.9)]
    raw_2d_frame = np.stack([[camera.project_point(point)] for camera in cameras], axis=0)
    raw_scores_frame = np.ones((3, 1), dtype=float)
    references = triangulation_reference_from_other_views(raw_2d_frame, raw_scores_frame, cameras, min_other_cameras=2)

    direct = compute_camera_triangulation_cost(
        0,
        raw_2d_frame[0],
        raw_2d_frame,
        raw_scores_frame,
        cameras,
        min_other_cameras=2,
    )
    cached = compute_camera_triangulation_cost(
        0,
        raw_2d_frame[0],
        raw_2d_frame,
        raw_scores_frame,
        cameras,
        min_other_cameras=2,
        precomputed_reprojected_points=references,
    )
    assert abs(direct - cached) < 1e-12
