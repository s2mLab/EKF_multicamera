import numpy as np

from judging.dd_analysis import DDSessionAnalysis, JumpSegment
from judging.execution import (
    analyze_execution_session,
    build_execution_overlay_frame,
    compute_time_of_flight_robust,
    detect_contacts_velocity,
    execution_focus_frame,
    infer_execution_images_root,
    resolve_execution_image_path,
)


def _base_points(n_frames: int) -> np.ndarray:
    """Build a simple upright COCO17-like skeleton trajectory."""

    points = np.full((n_frames, 17, 3), np.nan, dtype=float)
    for frame_idx in range(n_frames):
        points[frame_idx, 5] = (-0.1, 0.2, 1.0)  # left_shoulder
        points[frame_idx, 6] = (-0.1, -0.2, 1.0)  # right_shoulder
        points[frame_idx, 11] = (0.0, 0.15, 0.0)  # left_hip
        points[frame_idx, 12] = (0.0, -0.15, 0.0)  # right_hip
        points[frame_idx, 13] = (0.0, 0.15, -1.0)  # left_knee
        points[frame_idx, 14] = (0.0, -0.15, -1.0)  # right_knee
        points[frame_idx, 15] = (0.0, 0.15, -2.0)  # left_ankle
        points[frame_idx, 16] = (0.0, -0.15, -2.0)  # right_ankle
        points[frame_idx, 7] = (-0.1, 0.2, 0.5)  # left_elbow
        points[frame_idx, 8] = (-0.1, -0.2, 0.5)  # right_elbow
        points[frame_idx, 9] = (-0.1, 0.2, 0.0)  # left_wrist
        points[frame_idx, 10] = (-0.1, -0.2, 0.0)  # right_wrist
    return points


def _session_with_one_jump(n_frames: int) -> DDSessionAnalysis:
    """Build the minimum DD session structure required by execution analysis."""

    segment = JumpSegment(start=0, end=n_frames - 1, peak_index=n_frames // 2)
    return DDSessionAnalysis(
        root_q=np.zeros((n_frames, 6), dtype=float),
        height=np.zeros(n_frames, dtype=float),
        smoothed_height=np.zeros(n_frames, dtype=float),
        height_threshold=0.0,
        airborne_regions=[(1, n_frames - 2)],
        jump_segments=[segment],
        jumps=[],
        analysis_start_frame=0,
    )


def test_execution_analysis_detects_arm_and_landing_deductions():
    n_frames = 10
    points = _base_points(n_frames)
    points[3:7, 7] = (-0.1, 1.0, 1.8)
    points[3:7, 8] = (-0.1, -1.0, 1.8)

    q_names = np.asarray(
        [
            "TRUNK:RotX",
            "TRUNK:TransX",
            "TRUNK:TransY",
            "TRUNK:TransZ",
        ],
        dtype=object,
    )
    q = np.zeros((n_frames, len(q_names)), dtype=float)
    qdot = np.zeros_like(q)
    qdot[-1, 0:4] = np.array([0.0, 1.5, 1.5, 1.5])

    session = analyze_execution_session(_session_with_one_jump(n_frames), q, qdot, q_names, points, fs=120.0)

    assert len(session.jumps) == 1
    jump = session.jumps[0]
    codes = {event.code for event in jump.deduction_events}
    assert "arms" in codes
    assert "landing" in codes
    assert 0.4 <= jump.capped_deduction <= 0.5
    assert execution_focus_frame(jump) == jump.event_frame_idx


def test_execution_analysis_caps_one_jump_at_half_point():
    n_frames = 12
    points = _base_points(n_frames)
    points[3:9, 13, 2] = -0.4
    points[3:9, 14, 2] = -0.4
    points[5, 7] = (-0.1, 1.3, 1.0)
    points[5, 8] = (-0.1, -1.3, 1.0)

    q_names = np.asarray(
        [
            "TRUNK:RotX",
            "TRUNK:TransX",
            "TRUNK:TransY",
            "TRUNK:TransZ",
        ],
        dtype=object,
    )
    q = np.zeros((n_frames, len(q_names)), dtype=float)
    q[:, 0] = np.linspace(-0.5, 0.5, n_frames)
    qdot = np.zeros_like(q)
    qdot[-1, 1:4] = np.array([3.0, 0.0, 0.0])

    session = analyze_execution_session(_session_with_one_jump(n_frames), q, qdot, q_names, points, fs=120.0)

    jump = session.jumps[0]
    assert jump.total_deduction > 0.5
    assert jump.capped_deduction == 0.5
    assert session.execution_score == 19.5


def test_execution_analysis_can_reach_stronger_body_straightness_deduction():
    n_frames = 8
    points = _base_points(n_frames)
    points[:, 5] = (-0.6, 0.2, 1.0)
    points[:, 6] = (-0.6, -0.2, 1.0)

    q_names = np.asarray(["TRUNK:TransZ"], dtype=object)
    q = np.zeros((n_frames, len(q_names)), dtype=float)

    session = analyze_execution_session(_session_with_one_jump(n_frames), q, None, q_names, points, fs=120.0)

    jump = session.jumps[0]
    straightness_events = [event for event in jump.deduction_events if event.code == "form_hips"]
    assert straightness_events
    assert straightness_events[0].deduction == 0.2


def test_compute_time_of_flight_robust_ignores_micro_bounces():
    time = np.array([0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.6, 0.8], dtype=float)
    tz = np.array([0.0, 1.0, 0.0, 1.2, 0.95, 1.1, 0.0, 1.0], dtype=float)

    contacts = detect_contacts_velocity(tz, time)

    assert contacts == [0, 2, 4, 6]
    assert np.isclose(compute_time_of_flight_robust(tz, time), 0.25)


def test_execution_analysis_reports_session_time_of_flight():
    n_frames = 9
    q_names = np.asarray(["TRUNK:TransZ"], dtype=object)
    q = np.array([[0.0], [1.0], [0.5], [0.0], [0.5], [0.0], [0.5], [1.0], [0.0]], dtype=float)
    qdot = np.zeros_like(q)
    points = _base_points(n_frames)
    session = analyze_execution_session(_session_with_one_jump(n_frames), q, qdot, q_names, points, fs=10.0)

    assert np.isclose(session.time_of_flight_s, 0.6)


def test_infer_execution_images_root_finds_dataset_images_folder(tmp_path):
    keypoints_path = tmp_path / "inputs" / "keypoints" / "trial_keypoints.json"
    images_root = tmp_path / "inputs" / "images" / "trial"
    images_root.mkdir(parents=True)
    keypoints_path.parent.mkdir(parents=True)
    keypoints_path.write_text("{}")

    inferred = infer_execution_images_root(keypoints_path)

    assert inferred == images_root


def test_resolve_execution_image_path_matches_camera_folder_and_frame(tmp_path):
    images_root = tmp_path / "images"
    camera_dir = images_root / "camA"
    camera_dir.mkdir(parents=True)
    image_path = camera_dir / "frame_000123.png"
    image_path.write_bytes(b"fake")

    resolved = resolve_execution_image_path(images_root, "camA", 123)

    assert resolved == image_path


def test_build_execution_overlay_frame_collects_raw_projected_points_and_image(tmp_path):
    class _Calibration:
        def project_point(self, point):
            return np.asarray(point[:2], dtype=float)

    pose_data = type(
        "PoseDataStub",
        (),
        {
            "camera_names": ["camA"],
            "frames": np.array([10], dtype=int),
            "keypoints": np.zeros((1, 1, 17, 2), dtype=float),
        },
    )()
    pose_data.keypoints[0, 0, 11] = np.array([12.0, 34.0], dtype=float)
    images_root = tmp_path / "images"
    camera_dir = images_root / "camA"
    camera_dir.mkdir(parents=True)
    image_path = camera_dir / "000010.png"
    image_path.write_bytes(b"fake")
    frame_points_3d = np.zeros((17, 3), dtype=float)
    frame_points_3d[11] = np.array([1.0, 2.0, 3.0], dtype=float)

    overlay = build_execution_overlay_frame(
        camera_name="camA",
        frame_idx=0,
        frame_number=10,
        frame_points_3d=frame_points_3d,
        calibrations={"camA": _Calibration()},
        pose_data=pose_data,
        keypoint_names=("left_hip",),
        images_root=images_root,
    )

    assert overlay.image_path == image_path
    np.testing.assert_allclose(overlay.raw_points_2d[11], np.array([12.0, 34.0], dtype=float))
    np.testing.assert_allclose(overlay.projected_points_2d[11], np.array([1.0, 2.0], dtype=float))
