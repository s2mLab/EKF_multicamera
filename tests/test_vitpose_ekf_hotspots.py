import numpy as np

from vitpose_ekf_pipeline import (
    CameraCalibration,
    apply_measurement_update_batch,
    apply_measurement_update_sequential,
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


def test_sequential_measurement_update_matches_batch_update():
    rng = np.random.default_rng(42)
    nq = 5
    nx = 3 * nq
    identity_x = np.eye(nx, dtype=float)
    predicted_state = rng.normal(size=nx)
    base = rng.normal(size=(nx, nx))
    predicted_covariance = base @ base.T + 1e-3 * np.eye(nx)

    H1 = rng.normal(size=(6, nq))
    H2 = rng.normal(size=(4, nq))
    z1 = rng.normal(size=6)
    z2 = rng.normal(size=4)
    h1 = rng.normal(size=6)
    h2 = rng.normal(size=4)
    R1 = 0.1 + rng.random(6)
    R2 = 0.1 + rng.random(4)

    batch = apply_measurement_update_batch(
        predicted_state=predicted_state,
        predicted_covariance=predicted_covariance,
        z=np.concatenate((z1, z2)),
        h=np.concatenate((h1, h2)),
        H_q=np.vstack((H1, H2)),
        R_diag_array=np.concatenate((R1, R2)),
        nq=nq,
        identity_x=identity_x,
    )
    sequential = apply_measurement_update_sequential(
        predicted_state=predicted_state,
        predicted_covariance=predicted_covariance,
        measurement_blocks=[(z1, h1, H1, R1), (z2, h2, H2, R2)],
        nq=nq,
        identity_x=identity_x,
    )

    assert batch is not None
    assert sequential is not None
    batch_state, batch_covariance = batch
    sequential_state, sequential_covariance = sequential
    np.testing.assert_allclose(sequential_state, batch_state, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(sequential_covariance, batch_covariance, atol=1e-8, rtol=1e-8)
