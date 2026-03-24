import numpy as np

from kinematics.analysis_3d import (
    angular_momentum_plot_data,
    angular_momentum_series,
    segment_length_series,
    valid_segment_length_samples,
)


class _FakeArray:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def to_array(self):
        return np.asarray(self._values, dtype=float)


class _FakeModel:
    def angularMomentum(self, q_values, qdot_values):
        q_values = np.asarray(q_values, dtype=float)
        qdot_values = np.asarray(qdot_values, dtype=float)
        return _FakeArray(
            [
                q_values[0] + qdot_values[0],
                q_values[1] + qdot_values[1],
                q_values[2] + qdot_values[2],
            ]
        )


def test_segment_length_series_computes_expected_distances():
    points = np.full((2, 17, 3), np.nan, dtype=float)
    points[:, 5] = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    points[:, 11] = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    points[:, 6] = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    points[:, 12] = np.array([[1.0, 0.0, 1.0], [2.0, 0.0, 2.0]])
    points[:, 7] = np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
    points[:, 8] = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]])
    points[:, 9] = np.array([[0.0, 2.0, 0.0], [0.0, 4.0, 0.0]])
    points[:, 10] = np.array([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0]])
    points[:, 13] = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]])
    points[:, 14] = np.array([[1.0, 0.0, 2.0], [2.0, 0.0, 4.0]])
    points[:, 15] = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 6.0]])
    points[:, 16] = np.array([[1.0, 0.0, 3.0], [2.0, 0.0, 6.0]])

    series = segment_length_series(points)

    np.testing.assert_allclose(series["Trunk"], np.array([1.0, 2.0]))
    np.testing.assert_allclose(series["Shoulders"], np.array([1.0, 2.0]))
    np.testing.assert_allclose(series["L upper arm"], np.array([1.0, 2.0]))
    np.testing.assert_allclose(series["L shank"], np.array([1.0, 2.0]))


def test_valid_segment_length_samples_removes_nans():
    samples = valid_segment_length_samples(
        {
            "A": np.array([1.0, np.nan, 2.0]),
            "B": np.array([np.nan, np.nan]),
        }
    )

    assert set(samples.keys()) == {"A"}
    np.testing.assert_allclose(samples["A"], np.array([1.0, 2.0]))


def test_angular_momentum_series_uses_model_output():
    model = _FakeModel()
    q = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    qdot = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    momentum = angular_momentum_series(model, q, qdot)

    np.testing.assert_allclose(momentum, np.array([[1.1, 2.2, 3.3], [0.9, 2.0, 3.1]]))


def test_angular_momentum_plot_data_derives_norm():
    model = _FakeModel()
    q = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    qdot = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    time_s = np.array([0.0, 0.1])

    plot_data = angular_momentum_plot_data(model, q, qdot, time_s)

    np.testing.assert_allclose(plot_data.components, np.array([[1.1, 2.2, 3.3], [0.9, 2.0, 3.1]]))
    np.testing.assert_allclose(plot_data.norm, np.linalg.norm(plot_data.components, axis=1))
