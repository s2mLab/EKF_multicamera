import numpy as np

from judging.dd_analysis import (
    JumpSegment,
    analyze_single_jump,
    default_body_shape_indices,
    signed_threshold_crossing_indices,
)


def test_default_body_shape_indices_use_rot_y_flexion_axes():
    q_names = [
        "LEFT_THIGH:RotY",
        "RIGHT_THIGH:RotY",
        "LEFT_SHANK:RotY",
        "RIGHT_SHANK:RotY",
        "TRUNK:TransZ",
    ]
    body_indices = default_body_shape_indices(q_names)
    assert body_indices == ([0, 1], [2, 3])


def test_signed_threshold_crossing_indices_tracks_quarters_and_halves():
    cumulative = np.array([0.0, 0.10, 0.27, 0.51, 0.74, 1.02], dtype=float)
    quarter_indices = signed_threshold_crossing_indices(cumulative, 0.25)
    half_indices = signed_threshold_crossing_indices(cumulative, 0.5)
    assert quarter_indices == [2, 3, 5]
    assert half_indices == [3, 5]


def test_analyze_single_jump_code_uses_half_twists_per_completed_salto(monkeypatch):
    som_curve = np.array([0.0, -0.60, -1.05, -1.70, -2.02], dtype=float) * (2.0 * np.pi)
    tw_curve = np.array([0.0, 0.60, 1.52, 1.80, 2.01, 2.05], dtype=float) * (2.0 * np.pi)
    tw_curve = tw_curve[: som_curve.shape[0]]
    tilt_curve = np.zeros_like(som_curve)

    def fake_compute_angles_over_jump(_root_q, _start, _end, rotation_sequence="yxz", angle_mode="euler"):
        return som_curve, tw_curve, tilt_curve

    monkeypatch.setattr("judging.dd_analysis.compute_angles_over_jump", fake_compute_angles_over_jump)
    jump = analyze_single_jump(np.zeros((6, 6)), JumpSegment(start=0, end=5, peak_index=3))
    assert jump.twists_per_salto == [1.5, 0.5]
    assert jump.code == "831"
    assert jump.full_salto_event_indices == [2, 4]
    assert jump.quarter_salto_event_indices == [1, 2, 3, 4]
    assert jump.half_twist_event_indices == [1, 2, 4]


def test_analyze_single_jump_infers_last_salto_twist_from_end_of_jump(monkeypatch):
    som_curve = np.array([0.0, 0.55, 1.02, 1.55, 1.94], dtype=float) * (2.0 * np.pi)
    tw_curve = np.array([0.0, 0.45, 1.02, 1.32, 1.53], dtype=float) * (2.0 * np.pi)
    tilt_curve = np.zeros_like(som_curve)

    def fake_compute_angles_over_jump(_root_q, _start, _end, rotation_sequence="yxz", angle_mode="euler"):
        return som_curve, tw_curve, tilt_curve

    monkeypatch.setattr("judging.dd_analysis.compute_angles_over_jump", fake_compute_angles_over_jump)
    jump = analyze_single_jump(np.zeros((5, 6)), JumpSegment(start=0, end=4, peak_index=2))
    assert jump.twists_per_salto == [1.0, 0.5]
    assert jump.code == "821"
