import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import export_reconstruction_bundle
from kinematics.root_kinematics import TRUNK_ROOT_ROTATION_SCIPY_SEQUENCE
from reconstruction.reconstruction_bundle import build_pose2sim_bundle, parse_trc_points, root_kinematics_from_trc
from reconstruction.reconstruction_dataset import write_trc_root_kinematics_sidecar
from vitpose_ekf_pipeline import COCO17, KP_INDEX, CameraCalibration, PoseData

REPO_ROOT = Path(__file__).resolve().parents[1]
TRC_FPS = 120.0
PELVIS_VELOCITY_M_S = np.array([1.2, -0.3, 0.6], dtype=float)
YAW_RATE_RAD_S = 2.4
# Pose2Sim-style TRC labels, in COCO17 order.
POSE2SIM_TRC_LABELS = [
    "Nose",
    "L Eye",
    "R Eye",
    "L Ear",
    "R Ear",
    "L Shoulder",
    "R Shoulder",
    "L Elbow",
    "R Elbow",
    "L Wrist",
    "R Wrist",
    "L Hip",
    "R Hip",
    "L Knee",
    "R Knee",
    "L Ankle",
    "R Ankle",
]
# Standing COCO17 skeleton in metres (x forward, y left, z up), pelvis at the origin.
STANDING_COCO17_M = np.array(
    [
        [0.10, 0.00, 0.62],
        [0.08, 0.04, 0.65],
        [0.08, -0.04, 0.65],
        [0.02, 0.08, 0.63],
        [0.02, -0.08, 0.63],
        [0.00, 0.20, 0.45],
        [0.00, -0.20, 0.45],
        [0.02, 0.26, 0.17],
        [0.02, -0.26, 0.17],
        [0.08, 0.28, -0.08],
        [0.08, -0.28, -0.08],
        [0.00, 0.12, 0.00],
        [0.00, -0.12, 0.00],
        [0.03, 0.12, -0.45],
        [0.03, -0.12, -0.45],
        [0.00, 0.12, -0.88],
        [0.00, -0.12, -0.88],
    ],
    dtype=float,
)


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


def _synthetic_trc_trajectory(n_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frames, time (s) and COCO17 points (m) of a skeleton translating and yawing at constant rates."""

    frames = np.arange(n_frames, dtype=int)
    time_s = frames / TRC_FPS
    pelvis_start_m = np.array([0.4, 0.1, 1.0], dtype=float)
    points_3d = np.stack(
        [
            Rotation.from_euler("z", YAW_RATE_RAD_S * t).apply(STANDING_COCO17_M)
            + pelvis_start_m
            + PELVIS_VELOCITY_M_S * t
            for t in time_s
        ]
    )
    return frames, time_s, points_3d


def _write_coco17_trc(path: Path, frames: np.ndarray, time_s: np.ndarray, points_3d: np.ndarray) -> Path:
    """Write a Pose2Sim-like TRC file (17 COCO markers, metres) to disk."""

    n_markers = len(POSE2SIM_TRC_LABELS)
    marker_row = "Frame#\tTime\t" + "\t\t\t".join(POSE2SIM_TRC_LABELS) + "\t\t\t"
    axis_row = "\t\t" + "\t".join(f"{axis}{idx + 1}" for idx in range(n_markers) for axis in "XYZ") + "\t"
    data_rows = [
        f"{frame}\t{t:.12f}\t" + "\t".join(f"{value:.12f}" for value in frame_points.reshape(-1))
        for frame, t, frame_points in zip(frames, time_s, points_3d)
    ]
    lines = [
        f"PathFileType\t4\t(X/Y/Z)\t{path.name}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{TRC_FPS:g}\t{TRC_FPS:g}\t{len(frames)}\t{n_markers}\tm\t{TRC_FPS:g}\t0\t{len(frames)}",
        marker_row,
        axis_row,
        *data_rows,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _synthetic_camera(name: str, tx_m: float) -> CameraCalibration:
    """Pinhole camera looking along +Z, placed so that the synthetic skeleton is in front of it."""

    intrinsics = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 480.0], [0.0, 0.0, 1.0]], dtype=float)
    rotation = np.eye(3, dtype=float)
    translation = np.array([[tx_m], [0.0], [4.0]], dtype=float)
    return CameraCalibration(
        name=name,
        image_size=(1280, 960),
        K=intrinsics,
        dist=np.zeros(5, dtype=float),
        rvec=np.zeros(3, dtype=float),
        tvec=translation,
        R=rotation,
        P=intrinsics @ np.hstack((rotation, translation)),
    )


def _pose_data_from_points(
    calibrations: dict[str, CameraCalibration], frames: np.ndarray, points_3d: np.ndarray
) -> PoseData:
    """Project 3D points into each camera to build perfectly consistent 2D observations."""

    keypoints = np.stack(
        [
            np.stack([[calibration.project_point(point) for point in frame_points] for frame_points in points_3d])
            for calibration in calibrations.values()
        ]
    )
    return PoseData(
        camera_names=list(calibrations),
        frames=np.asarray(frames, dtype=int),
        keypoints=keypoints,
        scores=np.ones(keypoints.shape[:3], dtype=float),
    )


@pytest.mark.parametrize(
    ("unwrap_root", "root_unwrap_mode"),
    [(False, "off"), (True, "single"), (True, "double"), (False, "")],
)
def test_build_pose2sim_bundle_imports_generic_trc_without_sidecar(tmp_path: Path, unwrap_root, root_unwrap_mode):
    """Non-regression: a generic TRC (no root sidecar) goes through geometric root extraction end to end."""

    n_frames = 8
    frames, time_s, points_3d = _synthetic_trc_trajectory(n_frames)
    trc_path = _write_coco17_trc(tmp_path / "generic.trc", frames, time_s, points_3d)
    calibrations = {"cam0": _synthetic_camera("cam0", 0.0), "cam1": _synthetic_camera("cam1", -1.0)}
    pose_data = _pose_data_from_points(calibrations, frames, points_3d)
    output_dir = tmp_path / "bundle"

    result = build_pose2sim_bundle(
        name="trc_generic",
        output_dir=output_dir,
        pose2sim_trc=trc_path,
        calibrations=calibrations,
        pose_data=pose_data,
        fps=TRC_FPS,
        initial_rotation_correction=False,
        unwrap_root=unwrap_root,
        root_unwrap_mode=root_unwrap_mode,
    )

    for file_name in ("reconstruction_bundle.npz", "bundle_summary.json", "summary.json"):
        assert (output_dir / file_name).is_file()
    summary = json.loads((output_dir / "bundle_summary.json").read_text(encoding="utf-8"))
    assert summary["family"] == "pose2sim"
    assert summary["root_kinematics_source"] == "geometric_trc_markers"
    assert summary["n_frames"] == n_frames
    assert summary["trc_rate_hz"] == TRC_FPS
    assert summary["reprojection_px"]["mean"] == pytest.approx(0.0, abs=1e-6)

    with np.load(output_dir / "reconstruction_bundle.npz", allow_pickle=True) as bundle:
        np.testing.assert_array_equal(bundle["frames"], frames)
        np.testing.assert_allclose(bundle["time_s"], time_s, atol=1e-12)
        assert bundle["points_3d"].shape == (n_frames, len(COCO17), 3)
        np.testing.assert_allclose(bundle["points_3d"], points_3d, atol=1e-9)
        q_root = np.asarray(bundle["q_root"], dtype=float)
        qdot_root = np.asarray(bundle["qdot_root"], dtype=float)
    np.testing.assert_allclose(result.payload["q_root"], q_root)

    # Root translation is the hip centre in metres, its derivative the imposed pelvis velocity in m/s.
    assert q_root.shape == (n_frames, 6) and qdot_root.shape == (n_frames, 6)
    assert np.all(np.isfinite(q_root)) and np.all(np.isfinite(qdot_root))
    hip_center = 0.5 * (points_3d[:, KP_INDEX["left_hip"]] + points_3d[:, KP_INDEX["right_hip"]])
    np.testing.assert_allclose(q_root[:, :3], hip_center, atol=1e-9)
    np.testing.assert_allclose(qdot_root[:, :3], np.tile(PELVIS_VELOCITY_M_S, (n_frames, 1)), atol=1e-6)

    # Root orientation follows the imposed yaw: R_t = Rz(yaw_rate * t) @ R_0, whatever the Euler unwrap mode.
    root_matrices = Rotation.from_euler(TRUNK_ROOT_ROTATION_SCIPY_SEQUENCE, q_root[:, 3:6]).as_matrix()
    # One angle per frame as a column: recent SciPy requires the last axis to match the single "z" axis.
    yaw_angles = (YAW_RATE_RAD_S * time_s)[:, np.newaxis]
    expected_matrices = Rotation.from_euler("z", yaw_angles).as_matrix() @ root_matrices[0]
    np.testing.assert_allclose(root_matrices, expected_matrices, atol=1e-9)


def test_export_reconstruction_bundle_cli_runs_pose2sim_family(tmp_path: Path, monkeypatch):
    """Non-regression: the CLI entry point used by the GUI imports a generic TRC file without error."""

    n_frames = 5
    frames, time_s, points_3d = _synthetic_trc_trajectory(n_frames)
    trc_path = _write_coco17_trc(tmp_path / "generic.trc", frames, time_s, points_3d)
    output_dir = tmp_path / "cli_bundle"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_reconstruction_bundle.py",
            "--name",
            "trc_cli",
            "--family",
            "pose2sim",
            "--output-dir",
            str(output_dir),
            "--calib",
            str(REPO_ROOT / "inputs" / "calibration" / "Calib.toml"),
            "--keypoints",
            str(REPO_ROOT / "inputs" / "keypoints" / "1_partie_0429_15f_keypoints.json"),
            "--trc-file",
            str(trc_path),
            "--fps",
            f"{TRC_FPS:g}",
            "--max-frames",
            str(n_frames),
        ],
    )

    export_reconstruction_bundle.main()

    summary = json.loads((output_dir / "bundle_summary.json").read_text(encoding="utf-8"))
    assert summary["family"] == "pose2sim"
    assert summary["root_kinematics_source"] == "geometric_trc_markers"
    assert summary["n_frames"] == n_frames
    with np.load(output_dir / "reconstruction_bundle.npz", allow_pickle=True) as bundle:
        np.testing.assert_allclose(bundle["points_3d"], points_3d, atol=1e-9)
        assert bundle["q_root"].shape == (n_frames, 6)
        assert np.all(np.isfinite(bundle["q_root"]))
