import numpy as np

from judging.dd_analysis import DDSessionAnalysis, JumpSegment
from judging.execution import (
    analyze_execution_session,
    compute_time_of_flight_robust,
    detect_contacts_velocity,
    execution_focus_frame,
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
