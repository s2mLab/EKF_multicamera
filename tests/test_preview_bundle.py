from pathlib import Path

import numpy as np

from preview.preview_bundle import align_to_reference, assemble_dataset_preview_bundle, root_center


def test_root_center_uses_midpoint_of_hips():
    points = np.full((2, 17, 3), np.nan, dtype=float)
    points[0, 11] = [0.0, 2.0, 1.0]
    points[0, 12] = [2.0, 0.0, 3.0]
    points[1, 11] = [1.0, 1.0, 1.0]
    points[1, 12] = [3.0, 5.0, 7.0]
    centers = root_center(points)
    np.testing.assert_allclose(centers, np.array([[1.0, 1.0, 2.0], [2.0, 3.0, 4.0]]), atol=1e-12)


def test_align_to_reference_matches_first_valid_root_position():
    reference = np.full((2, 17, 3), np.nan, dtype=float)
    moving = np.full((2, 17, 3), np.nan, dtype=float)
    reference[0, 11] = [0.0, 1.0, 0.0]
    reference[0, 12] = [0.0, -1.0, 0.0]
    moving[0, 11] = [3.0, 1.0, 0.0]
    moving[0, 12] = [3.0, -1.0, 0.0]
    aligned = align_to_reference(reference, moving)
    np.testing.assert_allclose(root_center(aligned)[0], root_center(reference)[0], atol=1e-12)
    np.testing.assert_allclose(aligned[0, 11], np.array([0.0, 1.0, 0.0]), atol=1e-12)


def test_assemble_dataset_preview_bundle_aligns_frames_and_keeps_q_root():
    points_a = np.full((2, 17, 3), np.nan, dtype=float)
    points_a[:, 11] = [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    points_a[:, 12] = [[0.0, -1.0, 0.0], [0.0, -1.0, 0.0]]
    points_b = np.full((1, 17, 3), np.nan, dtype=float)
    points_b[:, 11] = [[1.0, 1.0, 0.0]]
    points_b[:, 12] = [[1.0, -1.0, 0.0]]

    entries = [
        {
            "name": "pose2sim",
            "frames": np.array([10, 11], dtype=int),
            "time_s": np.array([1.0, 1.1], dtype=float),
            "points_3d": points_a,
            "points_3d_source": "",
            "summary": {"family": "pose2sim"},
            "q": np.empty((2, 0), dtype=float),
            "qdot": np.empty((2, 0), dtype=float),
            "q_root": np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=float),
            "qdot_root": np.empty((2, 0), dtype=float),
            "q_names": np.array([], dtype=object),
        },
        {
            "name": "ekf_3d",
            "frames": np.array([11], dtype=int),
            "time_s": np.array([1.1], dtype=float),
            "points_3d": points_b,
            "points_3d_source": "model_forward_kinematics",
            "summary": {"family": "ekf_3d"},
            "q": np.zeros((1, 7), dtype=float),
            "qdot": np.zeros((1, 7), dtype=float),
            "q_root": np.array([[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]], dtype=float),
            "qdot_root": np.array([[1.5, 1.6, 1.7, 1.8, 1.9, 2.0]], dtype=float),
            "q_names": np.array(["a", "b"], dtype=object),
        },
    ]
    bundle = assemble_dataset_preview_bundle(entries, None, lambda _path, _q: np.empty((0, 17, 3)))
    np.testing.assert_array_equal(bundle["frames"], np.array([10, 11], dtype=int))
    assert "pose2sim" in bundle["recon_3d"]
    assert "ekf_3d" in bundle["recon_q_root"]
    assert np.all(np.isnan(bundle["recon_3d"]["ekf_3d"][0]))
    np.testing.assert_allclose(bundle["recon_q_root"]["ekf_3d"][1], np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]), atol=1e-12)


def test_assemble_dataset_preview_bundle_rebuilds_markers_when_needed():
    fake_points = np.full((1, 17, 3), np.nan, dtype=float)
    entries = [
        {
            "name": "ekf_2d_acc",
            "frames": np.array([0], dtype=int),
            "time_s": np.array([0.0], dtype=float),
            "points_3d": fake_points,
            "points_3d_source": "",
            "summary": {"family": "ekf_2d"},
            "q": np.zeros((1, 8), dtype=float),
            "qdot": np.zeros((1, 8), dtype=float),
            "q_root": np.empty((1, 0), dtype=float),
            "qdot_root": np.empty((1, 0), dtype=float),
            "q_names": np.array(["q0"], dtype=object),
        }
    ]

    def marker_builder(path: Path, q: np.ndarray) -> np.ndarray:
        assert path == Path("/tmp/model.bioMod")
        assert q.shape == (1, 8)
        points = np.full((1, 17, 3), np.nan, dtype=float)
        points[0, 0] = [1.0, 2.0, 3.0]
        return points

    bundle = assemble_dataset_preview_bundle(entries, Path("/tmp/model.bioMod"), marker_builder)
    np.testing.assert_allclose(bundle["recon_3d"]["ekf_2d_acc"][0, 0], np.array([1.0, 2.0, 3.0]), atol=1e-12)
