import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

import vitpose_ekf_pipeline
from vitpose_ekf_pipeline import (
    KP_INDEX,
    CameraCalibration,
    PoseData,
    align_root_translation_guess_to_frame_zero,
    apply_measurement_update_batch,
    apply_measurement_update_sequential,
    apply_root_pose_guess_to_state,
    build_flip_epipolar_pair_weight_array,
    build_flip_epipolar_pair_weights,
    build_fundamental_matrix_array,
    canonical_coherence_method,
    canonical_triangulation_method,
    canonicalize_model_q_rotation_branches,
    canonicalize_state_q_rotation_branches,
    choose_ekf_prediction_gate_measurements,
    compute_biorbd_kalman_initial_state,
    compute_camera_epipolar_cost,
    compute_camera_epipolar_cost_legacy,
    compute_camera_epipolar_costs_vectorized,
    compute_camera_triangulation_cost,
    compute_epipolar_coherence,
    compute_epipolar_fast_frame_coherence,
    compute_epipolar_frame_coherence,
    compute_framewise_epipolar_measurement_weights,
    load_pose_data,
    once_triangulation_from_best_cameras,
    project_point_with_projection_matrices,
    q_names_from_model,
    sample_frames_uniformly,
    sampson_error_pixels_vectorized,
    smooth_camera_time_series,
    stack_measurement_blocks,
    support_coherence_method_for_runtime,
    symmetric_epipolar_distance_vectorized,
    triangulation_method_from_coherence_method,
    triangulation_reference_from_other_views,
    viterbi_flip_state_path,
    weighted_median,
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


def test_once_triangulation_ignores_nan_observations_and_requires_enough_valid_views():
    point = np.array([0.2, -0.1, 2.0], dtype=float)
    cameras = [_make_camera("cam0", 0.0), _make_camera("cam1", 1.0), _make_camera("cam2", -0.8)]
    observations = np.asarray([camera.project_point(point) for camera in cameras], dtype=float)
    observations[2] = np.array([np.nan, np.nan], dtype=float)
    confidences = np.array([1.0, 0.8, 0.9], dtype=float)

    triangulated, mean_error, per_view_error, coherence_per_view, excluded_views = once_triangulation_from_best_cameras(
        [camera.P for camera in cameras],
        observations,
        confidences,
        cameras,
        error_threshold_px=5.0,
        min_cameras_for_triangulation=2,
    )

    np.testing.assert_allclose(triangulated, point, atol=1e-8)
    assert np.isfinite(mean_error)
    assert np.isnan(per_view_error[2])
    assert coherence_per_view[2] == 0.0
    assert bool(excluded_views[2]) is True

    triangulated, mean_error, per_view_error, coherence_per_view, excluded_views = once_triangulation_from_best_cameras(
        [camera.P for camera in cameras],
        observations,
        confidences,
        cameras,
        error_threshold_px=5.0,
        min_cameras_for_triangulation=3,
    )

    assert np.all(np.isnan(triangulated))
    assert np.isnan(mean_error)
    assert np.all(np.isnan(per_view_error))
    assert np.all(coherence_per_view == 0.0)
    assert np.array_equal(excluded_views, np.array([False, False, True]))


def test_once_triangulation_none_threshold_keeps_high_reprojection_solution():
    point = np.array([0.2, -0.1, 2.0], dtype=float)
    cameras = [_make_camera("cam0", 0.0), _make_camera("cam1", 1.0), _make_camera("cam2", -0.8)]
    observations = np.asarray([camera.project_point(point) for camera in cameras], dtype=float)
    observations[2] += np.array([80.0, -60.0], dtype=float)
    confidences = np.array([1.0, 0.9, 0.8], dtype=float)

    triangulated_limited, mean_error_limited, *_ = once_triangulation_from_best_cameras(
        [camera.P for camera in cameras],
        observations,
        confidences,
        cameras,
        error_threshold_px=15.0,
        min_cameras_for_triangulation=2,
    )
    assert np.all(np.isnan(triangulated_limited))
    assert mean_error_limited > 15.0

    triangulated_unbounded, mean_error_unbounded, per_view_error, coherence_per_view, excluded_views = (
        once_triangulation_from_best_cameras(
            [camera.P for camera in cameras],
            observations,
            confidences,
            cameras,
            error_threshold_px=None,
            min_cameras_for_triangulation=2,
        )
    )
    assert np.all(np.isfinite(triangulated_unbounded))
    assert mean_error_unbounded > 15.0
    assert np.count_nonzero(np.isfinite(per_view_error)) == 3
    assert np.all(coherence_per_view >= 0.0)
    assert np.array_equal(excluded_views, np.array([False, False, False]))


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


def test_canonical_triangulation_and_coherence_methods_support_once():
    assert canonical_triangulation_method("raw") == "once"
    assert canonical_triangulation_method("once") == "once"
    assert canonical_coherence_method("triangulation", "greedy") == "triangulation_greedy"
    assert canonical_coherence_method("epipolar_framewise") == "epipolar_framewise"
    assert support_coherence_method_for_runtime("epipolar_fast_framewise") == "epipolar_fast"
    assert triangulation_method_from_coherence_method("triangulation_once", "exhaustive") == "once"
    assert triangulation_method_from_coherence_method("epipolar_framewise", "exhaustive") == "exhaustive"


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


def test_stack_measurement_blocks_concatenates_valid_blocks():
    z1 = np.array([1.0, 2.0], dtype=float)
    h1 = np.array([0.1, 0.2], dtype=float)
    H1 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    R1 = np.array([0.5, 0.6], dtype=float)
    z2 = np.array([3.0], dtype=float)
    h2 = np.array([0.3], dtype=float)
    H2 = np.array([[0.4, 0.5]], dtype=float)
    R2 = np.array([0.7], dtype=float)

    stacked = stack_measurement_blocks([(z1, h1, H1, R1), (z2, h2, H2, R2)], nq=2)

    assert stacked is not None
    z, h, H_q, R_diag_array = stacked
    np.testing.assert_allclose(z, np.array([1.0, 2.0, 3.0], dtype=float))
    np.testing.assert_allclose(h, np.array([0.1, 0.2, 0.3], dtype=float))
    np.testing.assert_allclose(H_q, np.array([[1.0, 0.0], [0.0, 1.0], [0.4, 0.5]], dtype=float))
    np.testing.assert_allclose(R_diag_array, np.array([0.5, 0.6, 0.7], dtype=float))


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
        keypoint_weights=np.array(
            [0.0] * shoulder_idx + [2.0] + [0.0] * (wrist_idx - shoulder_idx - 1) + [0.5] + [0.0] * (16 - wrist_idx),
            dtype=float,
        ),
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


def test_symmetric_epipolar_distance_vectorized_is_zero_on_perfect_matches():
    points = np.array([[0.1, 0.0], [0.2, 0.1]], dtype=float)
    other_points = np.array([[[0.1, 0.0], [0.2, 0.1]]], dtype=float)
    f_matrix = np.array([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=float)

    distances = symmetric_epipolar_distance_vectorized(points, other_points, f_matrix)
    np.testing.assert_allclose(distances, np.zeros_like(distances), atol=1e-12)


def test_compute_epipolar_coherence_accepts_symmetric_distance_mode():
    point = np.array([0.15, 0.05, 2.5], dtype=float)
    cameras = [_make_camera("cam0", 0.0), _make_camera("cam1", 1.0)]
    keypoints = np.stack([[camera.project_point(point)] for camera in cameras], axis=0)[:, np.newaxis, :, :]
    scores = np.ones((2, 1, 1), dtype=float)
    pose_data = PoseData(
        camera_names=[camera.name for camera in cameras],
        frames=np.array([0], dtype=int),
        keypoints=keypoints,
        scores=scores,
    )
    fundamental_matrices = {
        (i_cam, j_cam): vitpose_ekf_pipeline.fundamental_matrix(cameras[i_cam], cameras[j_cam])
        for i_cam in range(len(cameras))
        for j_cam in range(len(cameras))
        if i_cam != j_cam
    }

    coherence = compute_epipolar_coherence(
        pose_data,
        fundamental_matrices,
        threshold_px=10.0,
        distance_mode="symmetric",
    )

    assert coherence.shape == (1, 1, 2)
    np.testing.assert_allclose(coherence, 1.0, atol=1e-12)


def test_compute_epipolar_frame_coherence_matches_one_frame_sequence_result():
    point = np.array([0.15, 0.05, 2.5], dtype=float)
    cameras = [_make_camera("cam0", 0.0), _make_camera("cam1", 1.0)]
    frame_keypoints = np.stack([[camera.project_point(point)] for camera in cameras], axis=0)
    frame_scores = np.ones((2, 1), dtype=float)
    pose_data = PoseData(
        camera_names=[camera.name for camera in cameras],
        frames=np.array([0], dtype=int),
        keypoints=frame_keypoints[:, np.newaxis, :, :],
        scores=frame_scores[:, np.newaxis, :],
    )
    fundamental_matrices = {
        (i_cam, j_cam): vitpose_ekf_pipeline.fundamental_matrix(cameras[i_cam], cameras[j_cam])
        for i_cam in range(len(cameras))
        for j_cam in range(len(cameras))
        if i_cam != j_cam
    }

    sequence_coherence = compute_epipolar_coherence(
        pose_data,
        fundamental_matrices,
        threshold_px=10.0,
        distance_mode="sampson",
    )[0]
    frame_coherence = compute_epipolar_frame_coherence(
        frame_keypoints,
        frame_scores,
        fundamental_matrices,
        threshold_px=10.0,
    )

    np.testing.assert_allclose(frame_coherence, sequence_coherence, atol=1e-12)


def test_compute_epipolar_fast_frame_coherence_matches_one_frame_sequence_result():
    point = np.array([0.15, 0.05, 2.5], dtype=float)
    cameras = [_make_camera("cam0", 0.0), _make_camera("cam1", 1.0)]
    frame_keypoints = np.stack([[camera.project_point(point)] for camera in cameras], axis=0)
    frame_scores = np.ones((2, 1), dtype=float)
    pose_data = PoseData(
        camera_names=[camera.name for camera in cameras],
        frames=np.array([0], dtype=int),
        keypoints=frame_keypoints[:, np.newaxis, :, :],
        scores=frame_scores[:, np.newaxis, :],
    )
    fundamental_matrices = {
        (i_cam, j_cam): vitpose_ekf_pipeline.fundamental_matrix(cameras[i_cam], cameras[j_cam])
        for i_cam in range(len(cameras))
        for j_cam in range(len(cameras))
        if i_cam != j_cam
    }

    sequence_coherence = compute_epipolar_coherence(
        pose_data,
        fundamental_matrices,
        threshold_px=10.0,
        distance_mode="symmetric",
    )[0]
    frame_coherence = compute_epipolar_fast_frame_coherence(
        frame_keypoints,
        frame_scores,
        fundamental_matrices,
        threshold_px=10.0,
    )

    np.testing.assert_allclose(frame_coherence, sequence_coherence, atol=1e-12)


def test_compute_framewise_epipolar_measurement_weights_matches_expected_variances():
    point = np.array([0.15, 0.05, 2.5], dtype=float)
    cameras = [_make_camera("cam0", 0.0), _make_camera("cam1", 1.0)]
    frame_keypoints = np.stack([[camera.project_point(point)] for camera in cameras], axis=0)
    frame_scores = np.array([[0.9], [0.8]], dtype=float)
    fundamental_matrices = {
        (i_cam, j_cam): vitpose_ekf_pipeline.fundamental_matrix(cameras[i_cam], cameras[j_cam])
        for i_cam in range(len(cameras))
        for j_cam in range(len(cameras))
        if i_cam != j_cam
    }

    frame_coherence, effective_confidences, measurement_variances = compute_framewise_epipolar_measurement_weights(
        frame_keypoints,
        frame_scores,
        fundamental_matrices,
        threshold_px=10.0,
        distance_mode="sampson",
        coherence_confidence_floor=0.35,
        measurement_noise_scale=1.5,
    )

    np.testing.assert_allclose(frame_coherence, np.ones_like(frame_coherence), atol=1e-12)
    np.testing.assert_allclose(effective_confidences, frame_scores, atol=1e-12)
    expected_variances = 1.5 * (4.0 / frame_scores) ** 2
    np.testing.assert_allclose(measurement_variances, expected_variances, atol=1e-12)


def test_viterbi_flip_state_path_rejects_isolated_positive_frame():
    nominal = np.array([12.0, 12.0, 12.0, 12.0, 12.0], dtype=float)
    swapped = np.array([13.0, 13.0, 2.0, 13.0, 13.0], dtype=float)
    candidate_mask = np.ones(5, dtype=bool)

    decoded = viterbi_flip_state_path(nominal, swapped, candidate_mask, transition_cost=6.0)

    np.testing.assert_array_equal(decoded, np.zeros(5, dtype=bool))


def test_viterbi_flip_state_path_keeps_sustained_positive_run():
    nominal = np.array([12.0, 12.0, 12.0, 12.0, 12.0], dtype=float)
    swapped = np.array([13.0, 2.0, 2.0, 2.0, 13.0], dtype=float)
    candidate_mask = np.ones(5, dtype=bool)

    decoded = viterbi_flip_state_path(nominal, swapped, candidate_mask, transition_cost=6.0)

    np.testing.assert_array_equal(decoded, np.array([False, True, True, True, False], dtype=bool))


def test_epipolar_flip_diagnostics_keep_local_decision_by_default(monkeypatch):
    pose_data = PoseData(
        camera_names=["cam0"],
        frames=np.arange(5, dtype=int),
        keypoints=np.zeros((1, 5, 17, 2), dtype=float),
        scores=np.ones((1, 5, 17), dtype=float),
    )

    cost_sequence = iter(
        [
            (12.0, 12.0),
            (12.0, 12.0),
            (12.0, 12.0),
            (12.0, 12.0),
            (12.0, 12.0),
            (13.0, 13.0),
            (13.0, 13.0),
            (2.0, 13.0),
            (13.0, 13.0),
            (13.0, 13.0),
        ]
    )

    monkeypatch.setattr(vitpose_ekf_pipeline, "build_fundamental_matrix_array", lambda _calibs: np.zeros((1, 1, 3, 3)))
    monkeypatch.setattr(vitpose_ekf_pipeline, "build_flip_epipolar_pair_weight_array", lambda _calibs: np.ones((1, 1)))
    monkeypatch.setattr(
        vitpose_ekf_pipeline,
        "compute_camera_epipolar_costs_vectorized",
        lambda *_args, **_kwargs: next(cost_sequence),
    )
    monkeypatch.setattr(
        vitpose_ekf_pipeline,
        "build_temporal_reference_points",
        lambda _pose_data: (np.zeros((1, 5, 17, 2), dtype=float), np.zeros((1, 5), dtype=int)),
    )
    monkeypatch.setattr(vitpose_ekf_pipeline, "compute_camera_temporal_cost", lambda *_args, **_kwargs: np.nan)

    suspect_mask, diagnostics, _details = vitpose_ekf_pipeline.detect_left_right_flip_diagnostics(
        pose_data=pose_data,
        calibrations={"cam0": object()},
        method="epipolar",
        improvement_ratio=0.7,
        min_gain_px=3.0,
        temporal_weight=0.0,
    )

    np.testing.assert_array_equal(suspect_mask[0], np.array([False, False, True, False, False], dtype=bool))
    assert diagnostics["temporal_decision_method"] == "local_threshold"


def test_epipolar_viterbi_flip_diagnostics_remain_explicit(monkeypatch):
    pose_data = PoseData(
        camera_names=["cam0"],
        frames=np.arange(5, dtype=int),
        keypoints=np.zeros((1, 5, 17, 2), dtype=float),
        scores=np.ones((1, 5, 17), dtype=float),
    )

    cost_sequence = iter(
        [
            (12.0, 12.0),
            (12.0, 12.0),
            (12.0, 12.0),
            (12.0, 12.0),
            (12.0, 12.0),
            (13.0, 13.0),
            (13.0, 13.0),
            (2.0, 13.0),
            (13.0, 13.0),
            (13.0, 13.0),
        ]
    )

    monkeypatch.setattr(vitpose_ekf_pipeline, "build_fundamental_matrix_array", lambda _calibs: np.zeros((1, 1, 3, 3)))
    monkeypatch.setattr(vitpose_ekf_pipeline, "build_flip_epipolar_pair_weight_array", lambda _calibs: np.ones((1, 1)))
    monkeypatch.setattr(
        vitpose_ekf_pipeline,
        "compute_camera_epipolar_costs_vectorized",
        lambda *_args, **_kwargs: next(cost_sequence),
    )
    monkeypatch.setattr(
        vitpose_ekf_pipeline,
        "build_temporal_reference_points",
        lambda _pose_data: (np.zeros((1, 5, 17, 2), dtype=float), np.zeros((1, 5), dtype=int)),
    )
    monkeypatch.setattr(vitpose_ekf_pipeline, "compute_camera_temporal_cost", lambda *_args, **_kwargs: np.nan)
    monkeypatch.setattr(
        vitpose_ekf_pipeline,
        "viterbi_flip_state_path",
        lambda *_args, **_kwargs: np.zeros(5, dtype=bool),
    )

    suspect_mask, diagnostics, _details = vitpose_ekf_pipeline.detect_left_right_flip_diagnostics(
        pose_data=pose_data,
        calibrations={"cam0": object()},
        method="epipolar_viterbi",
        improvement_ratio=0.7,
        min_gain_px=3.0,
        temporal_weight=0.0,
    )

    np.testing.assert_array_equal(suspect_mask[0], np.zeros(5, dtype=bool))
    assert diagnostics["temporal_decision_method"] == "viterbi_two_state"


def test_load_pose_data_applies_frame_stride(tmp_path: Path):
    keypoints_path = tmp_path / "keypoints.json"
    payload = {
        "cam_M11139": {
            "frames": [0, 1, 2, 3, 4, 5],
            "keypoints": np.zeros((6, 17, 2), dtype=float).tolist(),
            "scores": np.ones((6, 17), dtype=float).tolist(),
        }
    }
    keypoints_path.write_text(json.dumps(payload), encoding="utf-8")
    calibrations = {"M11139": _make_camera("M11139", 0.0)}

    pose_data = load_pose_data(keypoints_path, calibrations, frame_stride=3, data_mode="raw")

    np.testing.assert_array_equal(pose_data.frames, np.array([0, 3], dtype=int))
    assert pose_data.frame_stride == 3


def test_load_pose_data_ignores_unselected_json_cameras(tmp_path: Path):
    keypoints_path = tmp_path / "keypoints.json"
    payload = {
        "Camera1_M11139": {
            "frames": [0, 1],
            "keypoints": np.zeros((2, 17, 2), dtype=float).tolist(),
            "scores": np.ones((2, 17), dtype=float).tolist(),
        },
        "Camera2_M11140": {
            "frames": [0, 1],
            "keypoints": np.ones((2, 17, 2), dtype=float).tolist(),
            "scores": np.ones((2, 17), dtype=float).tolist(),
        },
        "Camera5_M11459": {
            "frames": [0, 1],
            "keypoints": (2.0 * np.ones((2, 17, 2), dtype=float)).tolist(),
            "scores": np.ones((2, 17), dtype=float).tolist(),
        },
    }
    keypoints_path.write_text(json.dumps(payload), encoding="utf-8")
    calibrations = {
        "M11139": _make_camera("M11139", 0.0),
        "M11140": _make_camera("M11140", 1.0),
    }

    pose_data = load_pose_data(keypoints_path, calibrations, data_mode="raw")

    assert pose_data.camera_names == ["M11139", "M11140"]
    assert pose_data.keypoints.shape[0] == 2


def test_load_pose_data_annotated_overlays_sparse_annotations(tmp_path: Path):
    keypoints_path = tmp_path / "keypoints.json"
    annotations_path = tmp_path / "annotations.json"
    payload = {
        "Camera1_M11139": {
            "frames": [0, 1],
            "keypoints": np.zeros((2, 17, 2), dtype=float).tolist(),
            "scores": np.ones((2, 17), dtype=float).tolist(),
        }
    }
    keypoints_path.write_text(json.dumps(payload), encoding="utf-8")
    annotations_path.write_text(
        json.dumps(
            {
                "annotations": {
                    "M11139": {
                        "1": {
                            "left_wrist": {
                                "xy": [123.0, 456.0],
                                "score": 0.75,
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    calibrations = {"M11139": _make_camera("M11139", 0.0)}

    pose_data = load_pose_data(
        keypoints_path,
        calibrations,
        data_mode="annotated",
        annotations_path=annotations_path,
    )

    left_wrist_idx = KP_INDEX["left_wrist"]
    np.testing.assert_allclose(pose_data.keypoints[0, 1, left_wrist_idx], np.array([123.0, 456.0], dtype=float))
    assert pose_data.scores[0, 1, left_wrist_idx] == 0.75
    np.testing.assert_allclose(pose_data.raw_keypoints[0, 1, left_wrist_idx], np.array([0.0, 0.0], dtype=float))
    np.testing.assert_allclose(
        pose_data.annotated_keypoints[0, 1, left_wrist_idx], np.array([123.0, 456.0], dtype=float)
    )


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


def test_vectorized_epipolar_symmetric_cost_returns_finite_values():
    cameras = [_make_camera("cam0", 0.0), _make_camera("cam1", 0.5), _make_camera("cam2", 2.0)]
    point = np.array([0.15, -0.05, 2.5], dtype=float)
    raw_2d_frame = np.stack([[camera.project_point(point) for _ in range(17)] for camera in cameras], axis=0)
    raw_scores_frame = np.ones((3, 17), dtype=float)

    weighted_cost, legacy_cost = compute_camera_epipolar_costs_vectorized(
        0,
        raw_2d_frame[0],
        raw_2d_frame,
        raw_scores_frame,
        build_fundamental_matrix_array(cameras),
        pair_weights_array=build_flip_epipolar_pair_weight_array(cameras),
        keypoint_weights=np.ones(17, dtype=float),
        min_other_cameras=2,
        distance_mode="symmetric",
    )

    assert np.isfinite(weighted_cost)
    assert np.isfinite(legacy_cost)


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
            _FakeSegment("LEFT_UPPER_ARM", ["RotY", "RotX"]),
            _FakeSegment("LEFT_THIGH", ["RotY"]),
        ]

    def nbSegment(self) -> int:
        return len(self._segments)

    def segment(self, idx: int):
        return self._segments[idx]

    def nbQ(self) -> int:
        return sum(segment.nbDof() for segment in self._segments)


class _FakeReconstruction:
    def __init__(self, points_3d: np.ndarray):
        self.points_3d = np.asarray(points_3d, dtype=float)


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
    assert updated[q_names.index("LEFT_UPPER_ARM:RotY")] == 0.0
    assert updated[q_names.index("LEFT_THIGH:RotY")] == 0.0


def test_compute_biorbd_kalman_initial_state_root_pose_zero_rest(monkeypatch):
    model = _FakeModel()
    reconstruction = object()
    triangulation_state = np.full(3 * model.nbQ(), 7.0, dtype=float)
    root_pose = np.array([1.0, 2.0, 3.0, 0.4, -0.2, 1.1], dtype=float)

    monkeypatch.setattr(
        vitpose_ekf_pipeline, "initial_state_from_triangulation", lambda _model, _reconstruction: triangulation_state
    )
    monkeypatch.setattr(
        vitpose_ekf_pipeline,
        "first_valid_root_pose_from_triangulation",
        lambda _reconstruction, **_kwargs: (12, root_pose),
    )

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
    assert math.isclose(state[q_names.index("TRUNK:RotY")], 0.4)
    assert math.isclose(state[q_names.index("TRUNK:RotX")], -0.2)
    assert math.isclose(state[q_names.index("TRUNK:RotZ")], 1.1)
    assert state[q_names.index("LEFT_UPPER_ARM:RotY")] == 0.0
    assert state[q_names.index("LEFT_THIGH:RotY")] == 0.0


def test_align_root_translation_guess_to_frame_zero_reanchors_horizontal_root_position():
    model = _FakeModel()
    q_names = q_names_from_model(model)
    state = np.zeros(3 * model.nbQ(), dtype=float)
    state[q_names.index("TRUNK:TransX")] = 1.2
    state[q_names.index("TRUNK:TransY")] = -0.4
    state[q_names.index("TRUNK:TransZ")] = 0.9

    points_3d = np.full((3, 17, 3), np.nan, dtype=float)
    points_3d[0, KP_INDEX["left_hip"]] = np.array([0.1, 0.2, 1.0], dtype=float)
    points_3d[0, KP_INDEX["right_hip"]] = np.array([0.3, -0.2, 1.0], dtype=float)
    points_3d[2, KP_INDEX["left_hip"]] = np.array([1.0, 0.2, 1.1], dtype=float)
    points_3d[2, KP_INDEX["right_hip"]] = np.array([1.2, -0.2, 1.1], dtype=float)
    reconstruction = _FakeReconstruction(points_3d)

    aligned = align_root_translation_guess_to_frame_zero(
        model,
        state,
        reconstruction,
        source_frame_idx=2,
    )

    assert aligned[q_names.index("TRUNK:TransX")] == 0.2
    assert aligned[q_names.index("TRUNK:TransY")] == 0.0
    assert aligned[q_names.index("TRUNK:TransZ")] == 1.0


def test_sample_frames_uniformly_spreads_indices_over_range():
    frames = np.arange(100, 110, dtype=int)
    sampled = sample_frames_uniformly(frames, 4)
    np.testing.assert_array_equal(sampled, np.array([100, 103, 106, 109], dtype=int))


def test_canonicalize_model_q_rotation_branches_reextracts_multi_axis_blocks():
    model = _FakeModel()
    q_names = q_names_from_model(model)
    q = np.zeros(model.nbQ(), dtype=float)
    root_input = np.array([math.pi + 0.4, -0.2, 2.0 * math.pi + 0.3], dtype=float)
    upper_arm_input = np.array([2.0 * math.pi + 0.2, -2.0 * math.pi - 0.3], dtype=float)
    thigh_input = 2.0 * math.pi + 0.1
    q[q_names.index("TRUNK:RotY")] = root_input[0]
    q[q_names.index("TRUNK:RotX")] = root_input[1]
    q[q_names.index("TRUNK:RotZ")] = root_input[2]
    q[q_names.index("LEFT_UPPER_ARM:RotY")] = upper_arm_input[0]
    q[q_names.index("LEFT_UPPER_ARM:RotX")] = upper_arm_input[1]
    q[q_names.index("LEFT_THIGH:RotY")] = thigh_input

    canonical_q = canonicalize_model_q_rotation_branches(model, q)

    np.testing.assert_allclose(
        Rotation.from_euler("YXZ", canonical_q[[3, 4, 5]], degrees=False).as_matrix(),
        Rotation.from_euler("YXZ", root_input, degrees=False).as_matrix(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        Rotation.from_euler("YX", canonical_q[[6, 7]], degrees=False).as_matrix(),
        Rotation.from_euler("YX", upper_arm_input, degrees=False).as_matrix(),
        atol=1e-12,
    )
    assert abs(canonical_q[q_names.index("LEFT_THIGH:RotY")] - 0.1) < 1e-12
    assert abs(canonical_q[q_names.index("TRUNK:RotZ")]) < abs(root_input[2])


def test_canonicalize_state_q_rotation_branches_only_changes_q_block():
    model = _FakeModel()
    q_names = q_names_from_model(model)
    nq = model.nbQ()
    state = np.zeros(3 * nq, dtype=float)
    state[q_names.index("TRUNK:RotY")] = math.pi + 0.4
    state[q_names.index("TRUNK:RotX")] = -0.2
    state[q_names.index("TRUNK:RotZ")] = 2.0 * math.pi + 0.3
    state[nq:] = 7.0

    canonical_state = canonicalize_state_q_rotation_branches(model, state)

    np.testing.assert_allclose(canonical_state[nq:], state[nq:], atol=1e-12)
    np.testing.assert_allclose(
        Rotation.from_euler("YXZ", canonical_state[[3, 4, 5]], degrees=False).as_matrix(),
        Rotation.from_euler("YXZ", state[[3, 4, 5]], degrees=False).as_matrix(),
        atol=1e-12,
    )
    assert abs(canonical_state[q_names.index("TRUNK:RotZ")]) < abs(state[q_names.index("TRUNK:RotZ")])


def test_initial_state_from_ekf_bootstrap_canonicalizes_rotation_branches(monkeypatch):
    model = _FakeModel()
    q_names = q_names_from_model(model)
    nq = model.nbQ()

    class _FakeBootstrapEkf:
        def __init__(self, *args, **kwargs):
            self.nq = nq
            self.nx = 3 * nq
            self.skip_correction_countdown = 0

        def predict(self, state, covariance, frame_idx):
            return np.array(state, copy=True), np.array(covariance, copy=True)

        def update(self, predicted_state, predicted_covariance, frame_idx):
            corrected_state = np.array(predicted_state, copy=True)
            corrected_state[q_names.index("TRUNK:RotY")] = math.pi + 0.25
            corrected_state[q_names.index("TRUNK:RotX")] = -0.15
            corrected_state[q_names.index("TRUNK:RotZ")] = 2.0 * math.pi + 0.35
            corrected_state[q_names.index("LEFT_UPPER_ARM:RotY")] = 2.0 * math.pi + 0.1
            corrected_state[q_names.index("LEFT_UPPER_ARM:RotX")] = -2.0 * math.pi - 0.2
            corrected_state[q_names.index("LEFT_THIGH:RotY")] = 2.0 * math.pi + 0.4
            return corrected_state, np.array(predicted_covariance, copy=True), "corrected"

    monkeypatch.setattr(
        vitpose_ekf_pipeline, "initial_state_from_triangulation", lambda *_args, **_kwargs: np.zeros(3 * nq)
    )
    monkeypatch.setattr(
        vitpose_ekf_pipeline, "first_valid_marker_tensor_from_reconstruction", lambda *_args, **_kwargs: (0, object())
    )
    monkeypatch.setattr(vitpose_ekf_pipeline, "MultiViewKinematicEKF", _FakeBootstrapEkf)

    state, diagnostics = vitpose_ekf_pipeline.initial_state_from_ekf_bootstrap(
        model=model,
        calibrations={},
        pose_data=object(),
        reconstruction=object(),
        fps=120.0,
        passes=1,
    )

    assert diagnostics["used_fallback"] is False
    assert abs(state[q_names.index("LEFT_THIGH:RotY")] - 0.4) < 1e-12
    assert abs(state[q_names.index("LEFT_UPPER_ARM:RotY")] - 0.1) < 1e-12
    assert abs(state[q_names.index("LEFT_UPPER_ARM:RotX")] + 0.2) < 1e-12
    np.testing.assert_allclose(
        Rotation.from_euler(
            "YXZ",
            state[[q_names.index("TRUNK:RotY"), q_names.index("TRUNK:RotX"), q_names.index("TRUNK:RotZ")]],
            degrees=False,
        ).as_matrix(),
        Rotation.from_euler("YXZ", [math.pi + 0.25, -0.15, 2.0 * math.pi + 0.35], degrees=False).as_matrix(),
        atol=1e-12,
    )


def test_compute_biorbd_kalman_initial_state_canonicalizes_triangulation_ik(monkeypatch):
    model = _FakeModel()
    q_names = q_names_from_model(model)
    nq = model.nbQ()
    state = np.zeros(3 * nq, dtype=float)
    state[q_names.index("TRUNK:RotY")] = math.pi + 0.35
    state[q_names.index("TRUNK:RotX")] = -0.15
    state[q_names.index("TRUNK:RotZ")] = 2.0 * math.pi + 0.25

    monkeypatch.setattr(vitpose_ekf_pipeline, "initial_state_from_triangulation", lambda *_args, **_kwargs: state)

    corrected_state, diagnostics = compute_biorbd_kalman_initial_state(model, object(), method="triangulation_ik")

    assert diagnostics["method"] == "triangulation_ik"
    np.testing.assert_allclose(
        Rotation.from_euler("YXZ", corrected_state[[3, 4, 5]], degrees=False).as_matrix(),
        Rotation.from_euler("YXZ", state[[3, 4, 5]], degrees=False).as_matrix(),
        atol=1e-12,
    )
    assert abs(corrected_state[q_names.index("TRUNK:RotZ")]) < abs(state[q_names.index("TRUNK:RotZ")])


def test_choose_ekf_prediction_gate_measurements_prefers_swapped_when_prediction_matches():
    frame_keypoints = np.full((17, 2), np.nan, dtype=float)
    frame_variances = np.full(17, np.inf, dtype=float)
    left_idx = KP_INDEX["left_shoulder"]
    right_idx = KP_INDEX["right_shoulder"]
    frame_keypoints[left_idx] = [20.0, 0.0]
    frame_keypoints[right_idx] = [10.0, 0.0]
    frame_variances[left_idx] = 1.0
    frame_variances[right_idx] = 1.0
    predicted_uv = np.array([[10.0, 0.0], [20.0, 0.0]], dtype=float)

    selected_mask, selected_points, selected_variances, diagnostics = choose_ekf_prediction_gate_measurements(
        frame_keypoints,
        frame_variances,
        predicted_uv,
        np.array([left_idx, right_idx], dtype=int),
        improvement_ratio=0.7,
        min_gain_px=3.0,
        min_valid_keypoints=2,
        activation_error_threshold_px=8.0,
        activation_error_delta_threshold_px=3.0,
        previous_nominal_rms_px=5.0,
    )

    np.testing.assert_array_equal(selected_mask, np.array([True, True], dtype=bool))
    np.testing.assert_allclose(selected_points, predicted_uv)
    np.testing.assert_allclose(selected_variances, np.array([1.0, 1.0], dtype=float))
    assert diagnostics["used_swapped"] is True
    assert diagnostics["decision"] == "swapped"


def test_choose_ekf_prediction_gate_measurements_skips_when_error_is_not_high_enough():
    frame_keypoints = np.full((17, 2), np.nan, dtype=float)
    frame_variances = np.full(17, np.inf, dtype=float)
    left_idx = KP_INDEX["left_shoulder"]
    right_idx = KP_INDEX["right_shoulder"]
    frame_keypoints[left_idx] = [13.0, 0.0]
    frame_keypoints[right_idx] = [11.0, 0.0]
    frame_variances[left_idx] = 1.0
    frame_variances[right_idx] = 1.0
    predicted_uv = np.array([[10.0, 0.0], [12.0, 0.0]], dtype=float)

    _selected_mask, selected_points, _selected_variances, diagnostics = choose_ekf_prediction_gate_measurements(
        frame_keypoints,
        frame_variances,
        predicted_uv,
        np.array([left_idx, right_idx], dtype=int),
        improvement_ratio=0.7,
        min_gain_px=3.0,
        min_valid_keypoints=2,
        activation_error_threshold_px=6.0,
        activation_error_delta_threshold_px=2.0,
        previous_nominal_rms_px=3.5,
    )

    np.testing.assert_allclose(selected_points, frame_keypoints[[left_idx, right_idx]])
    assert diagnostics["used_swapped"] is False
    assert diagnostics["decision"] == "raw_below_error_threshold"


def test_choose_ekf_prediction_gate_measurements_skips_when_error_delta_is_too_small():
    frame_keypoints = np.full((17, 2), np.nan, dtype=float)
    frame_variances = np.full(17, np.inf, dtype=float)
    left_idx = KP_INDEX["left_shoulder"]
    right_idx = KP_INDEX["right_shoulder"]
    frame_keypoints[left_idx] = [20.0, 0.0]
    frame_keypoints[right_idx] = [10.0, 0.0]
    frame_variances[left_idx] = 1.0
    frame_variances[right_idx] = 1.0
    predicted_uv = np.array([[10.0, 0.0], [20.0, 0.0]], dtype=float)

    _selected_mask, selected_points, _selected_variances, diagnostics = choose_ekf_prediction_gate_measurements(
        frame_keypoints,
        frame_variances,
        predicted_uv,
        np.array([left_idx, right_idx], dtype=int),
        improvement_ratio=0.7,
        min_gain_px=3.0,
        min_valid_keypoints=2,
        activation_error_threshold_px=7.0,
        activation_error_delta_threshold_px=3.0,
        previous_nominal_rms_px=7.5,
    )

    np.testing.assert_allclose(selected_points, frame_keypoints[[left_idx, right_idx]])
    assert diagnostics["used_swapped"] is False
    assert diagnostics["decision"] == "raw_below_error_delta"


def test_upper_back_pseudo_measurement_block_tracks_mean_hip_flexion():
    ekf = vitpose_ekf_pipeline.MultiViewKinematicEKF.__new__(vitpose_ekf_pipeline.MultiViewKinematicEKF)
    ekf.nq = 6
    ekf.upper_back_sagittal_idx = 1
    ekf.hip_flexion_indices = (3, 4)
    ekf.upper_back_sagittal_gain = 0.2
    ekf.upper_back_pseudo_std_rad = np.deg2rad(10.0)

    reference_q = np.array([0.0, 0.05, 0.0, 1.0, 0.5, 0.0], dtype=float)

    pseudo_block = ekf._upper_back_pseudo_measurement_block(reference_q)

    assert pseudo_block is not None
    z, h, h_q, variance = pseudo_block
    np.testing.assert_allclose(z, np.array([0.15], dtype=float))
    np.testing.assert_allclose(h, np.array([0.05], dtype=float))
    np.testing.assert_allclose(h_q, np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=float))
    np.testing.assert_allclose(variance, np.array([np.deg2rad(10.0) ** 2], dtype=float))


def test_upper_back_zero_prior_blocks_pull_lateral_and_axial_dofs_to_zero():
    ekf = vitpose_ekf_pipeline.MultiViewKinematicEKF.__new__(vitpose_ekf_pipeline.MultiViewKinematicEKF)
    ekf.nq = 6
    ekf.upper_back_zero_prior_indices = (2, 5)
    ekf.upper_back_pseudo_std_rad = np.deg2rad(8.0)

    reference_q = np.array([0.0, 0.0, 0.2, 0.0, 0.0, -0.15], dtype=float)

    blocks = ekf._upper_back_zero_prior_blocks(reference_q)

    assert len(blocks) == 2
    first_z, first_h, first_h_q, first_variance = blocks[0]
    second_z, second_h, second_h_q, second_variance = blocks[1]
    np.testing.assert_allclose(first_z, np.array([0.0], dtype=float))
    np.testing.assert_allclose(first_h, np.array([0.2], dtype=float))
    np.testing.assert_allclose(first_h_q, np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=float))
    np.testing.assert_allclose(first_variance, np.array([np.deg2rad(8.0) ** 2], dtype=float))
    np.testing.assert_allclose(second_z, np.array([0.0], dtype=float))
    np.testing.assert_allclose(second_h, np.array([-0.15], dtype=float))
    np.testing.assert_allclose(second_h_q, np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=float))
    np.testing.assert_allclose(second_variance, np.array([np.deg2rad(8.0) ** 2], dtype=float))


def test_ankle_bed_pseudo_measurement_blocks_keep_xz_targets_when_not_airborne():
    ekf = vitpose_ekf_pipeline.MultiViewKinematicEKF.__new__(vitpose_ekf_pipeline.MultiViewKinematicEKF)
    ekf.ankle_bed_pseudo_obs = True
    ekf.ankle_bed_pseudo_std_m = 0.02
    ekf.ankle_bed_pair_indices = ((0, vitpose_ekf_pipeline.KP_INDEX["left_ankle"]),)
    ekf.flight_height_threshold_m = 1.5
    ekf.reconstruction = SimpleNamespace(points_3d=np.full((1, 17, 3), np.nan, dtype=float))
    ekf.reconstruction.points_3d[0, vitpose_ekf_pipeline.KP_INDEX["left_ankle"]] = np.array([0.4, -0.1, 1.2])

    marker_points_array = np.array([[0.3, 0.0, 1.25]], dtype=float)
    marker_jacobians_array = np.array(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ],
        dtype=float,
    )

    blocks = ekf._ankle_bed_pseudo_measurement_blocks(0, marker_points_array, marker_jacobians_array)

    assert len(blocks) == 1
    z, h, h_q, variance = blocks[0]
    np.testing.assert_allclose(z, np.array([0.4, 1.2], dtype=float))
    np.testing.assert_allclose(h, np.array([0.3, 1.25], dtype=float))
    np.testing.assert_allclose(h_q, np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float))
    np.testing.assert_allclose(variance, np.array([0.0004, 0.0004], dtype=float))


def test_ankle_bed_pseudo_measurement_blocks_are_disabled_in_airborne_frames():
    ekf = vitpose_ekf_pipeline.MultiViewKinematicEKF.__new__(vitpose_ekf_pipeline.MultiViewKinematicEKF)
    ekf.ankle_bed_pseudo_obs = True
    ekf.ankle_bed_pseudo_std_m = 0.02
    ekf.ankle_bed_pair_indices = ((0, vitpose_ekf_pipeline.KP_INDEX["left_ankle"]),)
    ekf.flight_height_threshold_m = 1.5
    ekf.reconstruction = SimpleNamespace(points_3d=np.full((1, 17, 3), np.nan, dtype=float))
    ekf.reconstruction.points_3d[0, :, 2] = 1.7

    blocks = ekf._ankle_bed_pseudo_measurement_blocks(
        0,
        np.array([[0.3, 0.0, 1.25]], dtype=float),
        np.array([np.eye(3, dtype=float)], dtype=float),
    )

    assert blocks == []


def test_apply_left_right_flip_corrections_halves_scores_for_swapped_views():
    keypoints = np.zeros((1, 2, 17, 2), dtype=float)
    scores = np.ones((1, 2, 17), dtype=float)
    pose_data = vitpose_ekf_pipeline.PoseData(
        camera_names=["cam0"],
        frames=np.array([0, 1], dtype=int),
        keypoints=keypoints,
        scores=scores,
    )
    suspect_mask = np.array([[False, True]], dtype=bool)

    corrected = vitpose_ekf_pipeline.apply_left_right_flip_corrections(pose_data, suspect_mask)

    np.testing.assert_allclose(corrected.scores[0, 0], np.ones(17, dtype=float))
    np.testing.assert_allclose(corrected.scores[0, 1], 0.5 * np.ones(17, dtype=float))


def test_history3_prediction_updates_joint_dofs_from_last_three_states():
    class _Model:
        def nbSegment(self):
            return 0

    ekf = vitpose_ekf_pipeline.MultiViewKinematicEKF.__new__(vitpose_ekf_pipeline.MultiViewKinematicEKF)
    ekf.model = _Model()
    ekf.nq = 4
    ekf.dt = 0.1
    ekf.joint_indices = np.array([2, 3], dtype=int)
    ekf.corrected_q_history = [
        np.array([0.0, 0.0, 1.0, 2.0], dtype=float),
        np.array([0.0, 0.0, 2.0, 4.0], dtype=float),
        np.array([0.0, 0.0, 4.0, 8.0], dtype=float),
    ]
    predicted_state = np.zeros(12, dtype=float)
    predicted_state[:4] = np.array([10.0, 20.0, -1.0, -1.0], dtype=float)

    updated = ekf._apply_history3_prediction(predicted_state)

    np.testing.assert_allclose(updated[:4], np.array([10.0, 20.0, 7.0, 14.0], dtype=float))
    np.testing.assert_allclose(updated[4:8], np.array([0.0, 0.0, 30.0, 60.0], dtype=float))
    np.testing.assert_allclose(updated[8:12], np.array([0.0, 0.0, 100.0, 200.0], dtype=float))


def test_back_pseudo_segment_name_for_q_names_prefers_lower_trunk_when_present():
    q_names = np.asarray(["TRUNK:RotY", "LOWER_TRUNK:RotY", "LEFT_THIGH:RotY", "RIGHT_THIGH:RotY"], dtype=object)

    assert vitpose_ekf_pipeline.back_pseudo_segment_name_for_q_names(q_names) == "LOWER_TRUNK"
