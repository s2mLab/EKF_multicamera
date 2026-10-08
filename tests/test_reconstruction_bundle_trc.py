from pathlib import Path
from types import SimpleNamespace

import numpy as np

from reconstruction.reconstruction_bundle import build_pose2sim_bundle, parse_trc_points, root_kinematics_from_trc
from reconstruction.reconstruction_dataset import write_trc_root_kinematics_sidecar


def _write_trc(path: Path, marker_row: str, data_row: str) -> Path:
    """Write a minimal TRC fixture to disk."""

    content = "\n".join(
        [
            f"PathFileType\t4\t(X/Y/Z)\t{path.name}",
            "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
            "120\t120\t1\t2\tm\t120\t0\t1",
            marker_row,
            "\t\tX1\tY1\tZ1\tX2\tY2\tZ2",
            data_row,
        ]
    )
    path.write_text(content, encoding="utf-8")
    return path


def test_parse_trc_points_accepts_pose2sim_header_names(tmp_path: Path):
    trc_path = _write_trc(
        tmp_path / "pose2sim_style.trc",
        "Frame#\tTime\tNose\tL Hip",
        "0\t0.0\t0.1\t0.2\t0.3\t1.0\t1.1\t1.2",
    )

    frames, time_s, points_3d, data_rate = parse_trc_points(trc_path)

    np.testing.assert_array_equal(frames, np.array([0], dtype=int))
    np.testing.assert_allclose(time_s, np.array([0.0], dtype=float))
    assert data_rate == 120.0
    np.testing.assert_allclose(points_3d[0, 0], np.array([0.1, 0.2, 0.3], dtype=float))
    np.testing.assert_allclose(points_3d[0, 11], np.array([1.0, 1.1, 1.2], dtype=float))


def test_parse_trc_points_accepts_exported_marker_names_with_underscores(tmp_path: Path):
    trc_path = _write_trc(
        tmp_path / "export_style.trc",
        "Frame#\tTime\tleft_hip\tright_shoulder",
        "0\t0.0\t0.4\t0.5\t0.6\t1.3\t1.4\t1.5",
    )

    frames, time_s, points_3d, data_rate = parse_trc_points(trc_path)

    np.testing.assert_array_equal(frames, np.array([0], dtype=int))
    np.testing.assert_allclose(time_s, np.array([0.0], dtype=float))
    assert data_rate == 120.0
    np.testing.assert_allclose(points_3d[0, 11], np.array([0.4, 0.5, 0.6], dtype=float))
    np.testing.assert_allclose(points_3d[0, 6], np.array([1.3, 1.4, 1.5], dtype=float))


def test_root_kinematics_from_trc_prefers_exported_sidecar(tmp_path: Path):
    trc_path = _write_trc(
        tmp_path / "export_style.trc",
        "Frame#\tTime\tleft_hip\tright_hip\tleft_shoulder\tright_shoulder",
        "0\t0.0\t0.4\t0.5\t0.6\t0.7\t0.8\t0.9\t1.0\t1.1\t1.2\t1.3\t1.4\t1.5",
    )
    frames, time_s, points_3d, _ = parse_trc_points(trc_path)
    q_root = np.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]], dtype=float)
    qdot_root = np.array([[4.0, 5.0, 6.0, 0.4, 0.5, 0.6]], dtype=float)
    write_trc_root_kinematics_sidecar(trc_path, q_root, qdot_root, frames, time_s)

    resolved_q_root, resolved_qdot_root, correction_applied, source = root_kinematics_from_trc(
        trc_path,
        frames=frames,
        time_s=time_s,
        points_3d=points_3d,
        fps=120.0,
        initial_rotation_correction=True,
        unwrap_root=False,
    )

    np.testing.assert_allclose(resolved_q_root, q_root)
    np.testing.assert_allclose(resolved_qdot_root, qdot_root)
    assert correction_applied is True
    assert source == "trc_root_kinematics_sidecar"


def test_build_pose2sim_bundle_initializes_excluded_views_without_reconstruction_object(tmp_path, monkeypatch):
    trc_path = tmp_path / "demo.trc"
    trc_path.write_text("demo", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(
        "reconstruction.reconstruction_bundle.parse_trc_points",
        lambda _path: (
            np.array([0, 1], dtype=int),
            np.array([0.0, 1.0 / 120.0], dtype=float),
            np.zeros((2, 17, 3), dtype=float),
            120.0,
        ),
    )
    monkeypatch.setattr(
        "reconstruction.reconstruction_bundle.root_kinematics_from_trc",
        lambda *_args, **_kwargs: (
            np.zeros((2, 6), dtype=float),
            np.zeros((2, 6), dtype=float),
            False,
            "trc_points",
        ),
    )
    monkeypatch.setattr(
        "reconstruction.reconstruction_bundle.compute_points_reprojection_error_per_view",
        lambda *_args, **_kwargs: np.zeros((2, 17, 2), dtype=float),
    )
    monkeypatch.setattr(
        "reconstruction.reconstruction_bundle.summarize_reprojection_errors",
        lambda _errors, camera_names: {
            "mean_px": 0.0,
            "std_px": 0.0,
            "per_keypoint": {},
            "per_camera": {str(name): {"mean_px": 0.0, "std_px": 0.0} for name in camera_names},
        },
    )
    monkeypatch.setattr(
        "reconstruction.reconstruction_bundle.root_z_correction_angle_from_points",
        lambda *_args, **_kwargs: 0.0,
    )

    def fake_build_bundle_payload(**kwargs):
        captured.update(kwargs)
        return {"excluded_views": kwargs["excluded_views"]}

    monkeypatch.setattr("reconstruction.reconstruction_bundle.build_bundle_payload", fake_build_bundle_payload)
    monkeypatch.setattr("reconstruction.reconstruction_bundle.write_bundle", lambda *_args, **_kwargs: None)

    pose_data = SimpleNamespace(camera_names=["cam0", "cam1"])

    result = build_pose2sim_bundle(
        name="pose2sim_demo",
        output_dir=tmp_path / "output",
        pose2sim_trc=trc_path,
        calibrations={"cam0": object(), "cam1": object()},
        pose_data=pose_data,
        fps=120.0,
        initial_rotation_correction=False,
        unwrap_root=False,
    )

    np.testing.assert_array_equal(result.payload["excluded_views"], np.zeros((2, 17, 2), dtype=bool))
    np.testing.assert_array_equal(captured["excluded_views"], np.zeros((2, 17, 2), dtype=bool))
