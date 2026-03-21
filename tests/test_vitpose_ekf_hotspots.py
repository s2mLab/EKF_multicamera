import numpy as np
import vitpose_ekf_pipeline

from vitpose_ekf_pipeline import (
    CameraCalibration,
    apply_measurement_update_batch,
    apply_measurement_update_sequential,
    compute_biorbd_kalman_initial_state,
    apply_root_pose_guess_to_state,
    build_fundamental_matrix_array,
    build_flip_epipolar_pair_weights,
    build_flip_epipolar_pair_weight_array,
    compute_camera_epipolar_costs_vectorized,
    compute_camera_triangulation_cost,
    compute_camera_epipolar_cost,
    compute_camera_epipolar_cost_legacy,
    project_point_with_projection_matrices,
    q_names_from_model,
    sampson_error_pixels_vectorized,
    sample_frames_uniformly,
    smooth_camera_time_series,
    triangulation_reference_from_other_views,
    weighted_triangulation,
    weighted_median,
    KP_INDEX,
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


def test_project_points_and_jacobians_matches_scalar_projection():
    camera = _make_camera("cam0", 0.2)
    points = np.array([[0.15, 0.05, 2.5], [0.1, -0.2, 3.0]], dtype=float)
    projected_batch, jac_batch = camera.project_points_and_jacobians(points)
    projected_scalar = []
    jac_scalar = []
    for point in points:
        uv, jac = camera.project_point_and_jacobian(point)
        projected_scalar.append(uv)
        jac_scalar.append(jac)
    np.testing.assert_allclose(projected_batch, np.asarray(projected_scalar), atol=1e-8)
    np.testing.assert_allclose(jac_batch, np.asarray(jac_scalar), atol=1e-8)


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


def test_weighted_median_prefers_heavily_weighted_values():
    values = np.array([1.0, 10.0, 100.0], dtype=float)
    weights = np.array([5.0, 1.0, 1.0], dtype=float)
    assert weighted_median(values, weights) == 1.0


def test_build_flip_epipolar_pair_weights_reflects_baseline_strength():
    cameras = [_make_camera("cam0", 0.0), _make_camera("cam1", 0.5), _make_camera("cam2", 2.0)]
    weights = build_flip_epipolar_pair_weights(cameras)
    assert weights[(0, 2)] > weights[(0, 1)]
    assert weights[(2, 0)] > weights[(1, 0)]


def test_compute_camera_epipolar_cost_uses_pair_and_keypoint_weights(monkeypatch):
    candidate_points = np.full((17, 2), np.nan, dtype=float)
    raw_2d_frame = np.full((3, 17, 2), np.nan, dtype=float)
    raw_scores_frame = np.ones((3, 17), dtype=float)
    shoulder_idx = KP_INDEX["left_shoulder"]
    wrist_idx = KP_INDEX["left_wrist"]

    candidate_points[shoulder_idx] = [0.0, 0.0]
    candidate_points[wrist_idx] = [0.0, 0.0]
    raw_2d_frame[0] = candidate_points
    raw_2d_frame[1, shoulder_idx] = [1.0, 0.0]
    raw_2d_frame[2, shoulder_idx] = [10.0, 0.0]
    raw_2d_frame[1, wrist_idx] = [100.0, 0.0]
    raw_2d_frame[2, wrist_idx] = [100.0, 0.0]

    def fake_sampson(_point_i, point_j, _f_matrix):
        return float(point_j[0])

    monkeypatch.setattr("vitpose_ekf_pipeline.sampson_error_pixels", fake_sampson)
    cost = compute_camera_epipolar_cost(
        0,
        candidate_points,
        raw_2d_frame,
        raw_scores_frame,
        {(0, 1): np.eye(3), (0, 2): np.eye(3)},
        pair_weights={(0, 1): 10.0, (0, 2): 1.0},
        keypoint_weights=np.array([0.0] * shoulder_idx + [2.0] + [0.0] * (wrist_idx - shoulder_idx - 1) + [0.5] + [0.0] * (16 - wrist_idx), dtype=float),
        min_other_cameras=2,
    )
    assert abs(cost - 1.0) < 1e-12


def test_sampson_error_pixels_vectorized_matches_scalar():
    point_a = np.array([[0.2, -0.1], [0.5, 0.3]], dtype=float)
    other_points = np.array([[[0.1, 0.4], [0.6, -0.2]]], dtype=float)
    F = np.array([[[0.0, -1.0, 0.2], [1.0, 0.0, -0.1], [-0.2, 0.1, 0.0]]], dtype=float)

    vectorized = sampson_error_pixels_vectorized(point_a, other_points, F)
    scalar = np.array(
        [
            [
                vitpose_ekf_pipeline.sampson_error_pixels(point_a[0], other_points[0, 0], F[0]),
                vitpose_ekf_pipeline.sampson_error_pixels(point_a[1], other_points[0, 1], F[0]),
            ]
        ],
        dtype=float,
    )
    np.testing.assert_allclose(vectorized, scalar, atol=1e-12)


def test_vectorized_epipolar_cost_matches_scalar_versions():
    cameras = [_make_camera("cam0", 0.0), _make_camera("cam1", 0.5), _make_camera("cam2", 2.0)]
    point = np.array([0.2, -0.1, 2.5], dtype=float)
    raw_2d_frame = np.stack([[camera.project_point(point) for _ in range(17)] for camera in cameras], axis=0)
    raw_scores_frame = np.ones((3, 17), dtype=float)
    candidate_points = raw_2d_frame[0]
    fundamental_matrices = {
        (i_cam, j_cam): vitpose_ekf_pipeline.fundamental_matrix(cameras[i_cam], cameras[j_cam])
        for i_cam in range(3)
        for j_cam in range(3)
        if i_cam != j_cam
    }
    weighted_scalar = compute_camera_epipolar_cost(
        0,
        candidate_points,
        raw_2d_frame,
        raw_scores_frame,
        fundamental_matrices,
        pair_weights=build_flip_epipolar_pair_weights(cameras),
        keypoint_weights=np.ones(17, dtype=float),
        min_other_cameras=2,
    )
    legacy_scalar = compute_camera_epipolar_cost_legacy(
        0,
        candidate_points,
        raw_2d_frame,
        raw_scores_frame,
        fundamental_matrices,
    )
    weighted_vectorized, legacy_vectorized = compute_camera_epipolar_costs_vectorized(
        0,
        candidate_points,
        raw_2d_frame,
        raw_scores_frame,
        build_fundamental_matrix_array(cameras),
        pair_weights_array=build_flip_epipolar_pair_weight_array(cameras),
        keypoint_weights=np.ones(17, dtype=float),
        min_other_cameras=2,
    )
    np.testing.assert_allclose(weighted_vectorized, weighted_scalar, atol=1e-12)
    np.testing.assert_allclose(legacy_vectorized, legacy_scalar, atol=1e-12)


def test_smooth_camera_time_series_reduces_isolated_spike():
    series = np.array([[0.0, 9.0, 0.0, 0.0]], dtype=float)
    smoothed = smooth_camera_time_series(series, window=3)
    assert smoothed[0, 1] < series[0, 1]
    assert abs(smoothed[0, 1] - 3.0) < 1e-12


class _FakeName:
    def __init__(self, value: str):
        self._value = value

    def to_string(self) -> str:
        return self._value


class _FakeSegment:
    def __init__(self, name: str, dof_names: list[str]):
        self._name = name
        self._dof_names = list(dof_names)

    def name(self):
        return _FakeName(self._name)

    def nbDof(self) -> int:
        return len(self._dof_names)

    def nameDof(self, idx: int):
        return _FakeName(self._dof_names[idx])


class _FakeModel:
    def __init__(self):
        self._segments = [
            _FakeSegment("TRUNK", ["TransX", "TransY", "TransZ", "RotY", "RotX", "RotZ"]),
            _FakeSegment("LEFT_THIGH", ["RotY"]),
        ]

    def nbSegment(self) -> int:
        return len(self._segments)

    def segment(self, idx: int):
        return self._segments[idx]

    def nbQ(self) -> int:
        return sum(segment.nbDof() for segment in self._segments)


def test_apply_root_pose_guess_to_state_sets_root_dofs_only():
    model = _FakeModel()
    state = np.zeros(3 * model.nbQ(), dtype=float)
    root_pose = np.array([1.0, 2.0, 3.0, 0.4, -0.2, 1.1], dtype=float)

    updated = apply_root_pose_guess_to_state(model, state, root_pose)
    q_names = q_names_from_model(model)

    assert updated[q_names.index("TRUNK:TransX")] == 1.0
    assert updated[q_names.index("TRUNK:TransY")] == 2.0
    assert updated[q_names.index("TRUNK:TransZ")] == 3.0
    assert updated[q_names.index("TRUNK:RotY")] == 0.4
    assert updated[q_names.index("TRUNK:RotX")] == -0.2
    assert updated[q_names.index("TRUNK:RotZ")] == 1.1
    assert updated[q_names.index("LEFT_THIGH:RotY")] == 0.0


def test_compute_biorbd_kalman_initial_state_root_pose_zero_rest(monkeypatch):
    model = _FakeModel()
    reconstruction = object()
    triangulation_state = np.full(3 * model.nbQ(), 7.0, dtype=float)
    root_pose = np.array([1.0, 2.0, 3.0, 0.4, -0.2, 1.1], dtype=float)

    monkeypatch.setattr(vitpose_ekf_pipeline, "initial_state_from_triangulation", lambda _model, _reconstruction: triangulation_state)
    monkeypatch.setattr(vitpose_ekf_pipeline, "first_valid_root_pose_from_triangulation", lambda _reconstruction: (12, root_pose))

    state, diagnostics = compute_biorbd_kalman_initial_state(
        model,
        reconstruction,
        method="root_pose_zero_rest",
    )
    q_names = q_names_from_model(model)

    assert diagnostics["method"] == "root_pose_zero_rest"
    assert diagnostics["used_triangulation_ik"] is False
    assert diagnostics["used_root_pose_guess"] is True
    assert diagnostics["bootstrap_frame_idx"] == 12
    assert state[q_names.index("TRUNK:TransX")] == 1.0
    assert state[q_names.index("TRUNK:TransY")] == 2.0
    assert state[q_names.index("TRUNK:TransZ")] == 3.0
    assert state[q_names.index("TRUNK:RotY")] == 0.4
    assert state[q_names.index("TRUNK:RotX")] == -0.2
    assert state[q_names.index("TRUNK:RotZ")] == 1.1
    assert state[q_names.index("LEFT_THIGH:RotY")] == 0.0


def test_sample_frames_uniformly_spreads_indices_over_range():
    frames = np.arange(100, 110, dtype=int)
    sampled = sample_frames_uniformly(frames, 4)
    np.testing.assert_array_equal(sampled, np.array([100, 103, 106, 109], dtype=int))
