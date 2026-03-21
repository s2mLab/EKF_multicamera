import math

import numpy as np
from scipy.spatial.transform import Rotation

from root_kinematics import (
    TRUNK_ROOT_ROTATION_SEQUENCE,
    build_root_rotation_matrices,
    centered_finite_difference,
    compute_trunk_dofs_from_points,
    extract_root_from_q,
    reextract_euler_with_gaps,
    root_z_correction_angle_from_points,
    rotation_unit_label,
    rotation_unit_scale,
    unwrap_with_gaps,
)


def test_unwrap_with_gaps_keeps_gaps_independent():
    values = np.array([3.0, -3.1, np.nan, -3.0, 3.05], dtype=float)
    result = unwrap_with_gaps(values)
    expected = np.array([3.0, -3.1 + 2.0 * math.pi, np.nan, -3.0, 3.05 - 2.0 * math.pi], dtype=float)
    np.testing.assert_allclose(result[[0, 1, 3, 4]], expected[[0, 1, 3, 4]], atol=1e-12)
    assert math.isnan(result[2])


def test_reextract_euler_with_gaps_preserves_rotation_and_nans():
    rotations = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.4, -0.2, 0.1],
            [np.nan, np.nan, np.nan],
            [0.2, 0.3, -0.5],
        ],
        dtype=float,
    )
    result = reextract_euler_with_gaps(rotations, TRUNK_ROOT_ROTATION_SEQUENCE)
    assert np.all(np.isnan(result[2]))
    valid = np.where(np.all(np.isfinite(rotations), axis=1))[0]
    for frame_idx in valid:
        original_matrix = Rotation.from_euler(TRUNK_ROOT_ROTATION_SEQUENCE, rotations[frame_idx]).as_matrix()
        result_matrix = Rotation.from_euler(TRUNK_ROOT_ROTATION_SEQUENCE, result[frame_idx]).as_matrix()
        np.testing.assert_allclose(result_matrix, original_matrix, atol=1e-12)


def test_extract_root_from_q_respects_name_mapping_and_unwrap():
    q_names = np.asarray(
        [
            "LEFT_ARM:RotX",
            "TRUNK:RotY",
            "TRUNK:TransZ",
            "TRUNK:RotX",
            "TRUNK:TransY",
            "TRUNK:RotZ",
            "TRUNK:TransX",
        ],
        dtype=object,
    )
    q = np.array(
        [
            [9.0, 0.0, 3.0, 0.1, 2.0, 3.0, 1.0],
            [8.0, 0.1, 3.1, 0.2, 2.1, -3.1, 1.1],
        ],
        dtype=float,
    )
    root_q = extract_root_from_q(q_names, q, unwrap_rotations=True, renormalize_rotations=False)
    np.testing.assert_allclose(root_q[:, :3], np.array([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]]), atol=1e-12)
    np.testing.assert_allclose(root_q[:, 3], np.array([0.0, 0.1]), atol=1e-12)
    np.testing.assert_allclose(root_q[:, 4], np.array([0.1, 0.2]), atol=1e-12)
    assert root_q[1, 5] > root_q[0, 5]


def test_extract_root_from_q_can_skip_unwrap_but_keep_rotation_reextraction():
    q_names = np.asarray(
        [
            "TRUNK:TransX",
            "TRUNK:TransY",
            "TRUNK:TransZ",
            "TRUNK:RotY",
            "TRUNK:RotX",
            "TRUNK:RotZ",
        ],
        dtype=object,
    )
    q = np.array(
        [
            [0.0, 0.0, 0.0, 0.2, -0.1, 3.10],
            [0.0, 0.0, 0.0, 0.2, -0.1, -3.12],
        ],
        dtype=float,
    )
    root_q = extract_root_from_q(q_names, q, unwrap_rotations=False, renormalize_rotations=True)
    np.testing.assert_allclose(root_q[:, :5], q[:, :5], atol=1e-12)
    assert root_q[0, 5] > 3.0
    assert root_q[1, 5] < -3.0


def test_centered_finite_difference_handles_edges_and_nans():
    values = np.array(
        [
            [0.0, 0.0],
            [1.0, np.nan],
            [3.0, 4.0],
            [6.0, 9.0],
        ],
        dtype=float,
    )
    derivative = centered_finite_difference(values, dt=1.0)
    np.testing.assert_allclose(derivative[:, 0], np.array([1.0, 1.5, 2.5, 3.0]), atol=1e-12)
    assert math.isnan(derivative[0, 1])
    np.testing.assert_allclose(derivative[2:, 1], np.array([np.nan, 5.0]), equal_nan=True, atol=1e-12)


def test_compute_trunk_dofs_from_points_identity_pose():
    points = np.full((2, 17, 3), np.nan, dtype=float)
    points[0, 11] = [0.0, 1.0, 0.0]
    points[0, 12] = [0.0, -1.0, 0.0]
    points[0, 5] = [0.0, 1.0, 1.0]
    points[0, 6] = [0.0, -1.0, 1.0]
    root_q = compute_trunk_dofs_from_points(points, unwrap_rotations=True)
    np.testing.assert_allclose(root_q[0], np.zeros(6), atol=1e-12)
    assert np.all(np.isnan(root_q[1]))


def test_compute_trunk_dofs_from_points_uses_yxz_sequence():
    base_points = np.full((1, 17, 3), np.nan, dtype=float)
    base_points[0, 11] = [0.0, 1.0, 0.0]
    base_points[0, 12] = [0.0, -1.0, 0.0]
    base_points[0, 5] = [0.0, 1.0, 1.0]
    base_points[0, 6] = [0.0, -1.0, 1.0]
    matrix = Rotation.from_euler(TRUNK_ROOT_ROTATION_SEQUENCE, [0.25, 0.0, 0.0]).as_matrix()
    rotated_points = np.array(base_points, copy=True)
    for kp_idx in (11, 12, 5, 6):
        rotated_points[0, kp_idx] = matrix @ base_points[0, kp_idx]
    root_q = compute_trunk_dofs_from_points(rotated_points, unwrap_rotations=False)
    np.testing.assert_allclose(root_q[0, :3], np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(root_q[0, 3:], np.array([0.25, 0.0, 0.0]), atol=1e-12)


def test_root_z_correction_angle_from_points_snaps_to_nearest_right_angle():
    base_points = np.full((1, 17, 3), np.nan, dtype=float)
    base_points[0, 11] = [0.0, 1.0, 0.0]
    base_points[0, 12] = [0.0, -1.0, 0.0]
    base_points[0, 5] = [0.0, 1.0, 1.0]
    base_points[0, 6] = [0.0, -1.0, 1.0]
    yawed = np.array(base_points, copy=True)
    yaw_matrix = Rotation.from_euler("z", 0.49 * math.pi, degrees=False).as_matrix()
    for kp_idx in (11, 12, 5, 6):
        yawed[0, kp_idx] = yaw_matrix @ base_points[0, kp_idx]
    angle = root_z_correction_angle_from_points(yawed)
    _, rotation_matrices = build_root_rotation_matrices(yawed)
    corrected = rotation_matrices[0] @ Rotation.from_euler("z", angle, degrees=False).as_matrix()
    np.testing.assert_allclose(corrected[:, 1], np.array([0.0, 1.0, 0.0]), atol=5e-2)


def test_rotation_unit_helpers():
    assert rotation_unit_scale("rad") == 1.0
    assert rotation_unit_scale("deg") == 180.0 / math.pi
    assert rotation_unit_label("turns", "q") == "turn"
    assert rotation_unit_label("turns", "qdot") == "turn/s"
