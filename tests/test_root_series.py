import numpy as np

from root_series import (
    quantity_unit_label,
    root_axis_labels,
    root_series_from_precomputed,
    root_series_from_q,
    scale_root_series_rotations,
)


def test_root_series_from_q_uses_precomputed_qdot_when_requested():
    q_names = np.asarray(
        [
            "TRUNK:TransX",
            "TRUNK:TransY",
            "TRUNK:TransZ",
            "TRUNK:RotY",
            "TRUNK:RotX",
            "TRUNK:RotZ",
            "OTHER",
        ],
        dtype=object,
    )
    q = np.array([[1, 2, 3, 0.1, 0.2, 0.3, 9], [4, 5, 6, 0.4, 0.5, 0.6, 8]], dtype=float)
    qdot = np.array([[10, 20, 30, 1.1, 1.2, 1.3, 7], [40, 50, 60, 1.4, 1.5, 1.6, 6]], dtype=float)
    series = root_series_from_q(
        q_names,
        q,
        quantity="qdot",
        dt=0.01,
        qdot=qdot,
        fd_qdot=False,
        unwrap_rotations=False,
        renormalize_rotations=False,
    )
    np.testing.assert_allclose(series, np.array([[10, 20, 30, 1.1, 1.2, 1.3], [40, 50, 60, 1.4, 1.5, 1.6]]), atol=1e-12)


def test_root_series_from_precomputed_uses_fd_when_requested():
    root_q = np.array([[0, 0, 0, 0, 0, 0], [1, 2, 3, 0.1, 0.2, 0.3], [3, 6, 9, 0.3, 0.4, 0.5]], dtype=float)
    qdot_root = np.full_like(root_q, -999.0)
    series = root_series_from_precomputed(root_q, quantity="qdot", dt=1.0, qdot_root=qdot_root, fd_qdot=True)
    np.testing.assert_allclose(series[:, 0], np.array([1.0, 1.5, 2.0]), atol=1e-12)
    assert not np.any(series == -999.0)


def test_scale_root_series_rotations_only_scales_rotations():
    series = np.array([[1.0, 2.0, 3.0, 0.5, 1.0, 1.5]], dtype=float)
    scaled = scale_root_series_rotations(series, family_is_translation=False, rotation_unit="deg")
    np.testing.assert_allclose(scaled[:, :3], series[:, :3], atol=1e-12)
    np.testing.assert_allclose(scaled[:, 3:], series[:, 3:] * (180.0 / np.pi), atol=1e-12)


def test_root_axis_labels_and_units_follow_family():
    assert root_axis_labels("translations") == ["TRUNK:TransX", "TRUNK:TransY", "TRUNK:TransZ"]
    assert root_axis_labels("rotations") == ["TRUNK:RotY", "TRUNK:RotX", "TRUNK:RotZ"]
    assert quantity_unit_label("q", True, "deg") == "m"
    assert quantity_unit_label("qdot", False, "turns") == "turn/s"
