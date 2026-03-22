import numpy as np

from observability.observability_analysis import (
    compute_observability_rank_series,
    matrix_rank_with_full_rank,
    stacked_marker_jacobian,
    stacked_observation_jacobian,
    summarize_rank_series,
)


class _FakeArray:
    def __init__(self, array):
        self._array = np.asarray(array, dtype=float)

    def to_array(self):
        return np.asarray(self._array, dtype=float)


class _FakeModel:
    def nbQ(self):
        return 3

    def markers(self, q_values):
        q_values = np.asarray(q_values, dtype=float)
        return [
            _FakeArray([q_values[0], q_values[1], 1.0]),
            _FakeArray([q_values[1], q_values[2], 2.0]),
        ]

    def markersJacobian(self, q_values):
        q_values = np.asarray(q_values, dtype=float)
        if q_values[0] < 0.5:
            return [
                _FakeArray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
                _FakeArray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
            ]
        return [
            _FakeArray([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            _FakeArray([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        ]


class _FakeCalibration:
    def __init__(self, jacobian_blocks):
        self._jacobian_blocks = np.asarray(jacobian_blocks, dtype=float)

    def project_points_and_jacobians(self, points_world):
        points_world = np.asarray(points_world, dtype=float)
        projected = points_world[:, :2]
        return projected, np.asarray(self._jacobian_blocks, dtype=float)


def test_stacked_marker_jacobian_stacks_all_markers():
    model = _FakeModel()
    matrix = stacked_marker_jacobian(model, np.array([0.0, 0.0, 0.0]))
    assert matrix.shape == (6, 3)
    rank, full_rank = matrix_rank_with_full_rank(matrix)
    assert rank == 3
    assert full_rank == 3


def test_stacked_observation_jacobian_uses_camera_projections():
    model = _FakeModel()
    calibration = _FakeCalibration(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ]
    )
    matrix = stacked_observation_jacobian(model, np.array([0.0, 0.0, 0.0]), [calibration])
    assert matrix.shape == (4, 3)
    rank, full_rank = matrix_rank_with_full_rank(matrix)
    assert rank == 3
    assert full_rank == 3


def test_compute_observability_rank_series_tracks_rank_drop_over_time():
    model = _FakeModel()
    calibration = _FakeCalibration(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ]
    )
    q_series = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    series = compute_observability_rank_series(model, q_series, [calibration])
    np.testing.assert_array_equal(series.marker_rank, np.array([3, 1]))
    np.testing.assert_array_equal(series.observation_rank, np.array([3, 1]))
    assert series.marker_full_rank == 3
    assert series.observation_full_rank == 3


def test_summarize_rank_series_reports_full_rank_ratio():
    summary = summarize_rank_series(np.array([3, 3, 2, 1]), 3)
    assert summary["min"] == 1.0
    assert summary["median"] == 2.5
    assert summary["max"] == 3.0
    assert summary["full_rank_ratio"] == 0.5
