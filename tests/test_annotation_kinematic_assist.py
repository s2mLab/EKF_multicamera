import numpy as np

import annotation.kinematic_assist as kinematic_assist
from annotation.kinematic_assist import (
    annotation_relevant_q_mask,
    constrain_annotation_state_to_mask,
    refine_annotation_window_states,
    resolve_annotation_kinematic_state_info,
    store_annotation_kinematic_state,
)
from vitpose_ekf_pipeline import PoseData


class _FakeModel:
    def __init__(self, nq: int):
        self._nq = int(nq)

    def nbQ(self):
        return self._nq


def test_resolve_annotation_kinematic_state_info_prefers_exact_match():
    model = _FakeModel(2)
    frame_states = {
        ("demo", 10): np.array([1.0, 2.0], dtype=float),
        ("demo", 12): np.array([9.0, 9.0], dtype=float),
    }

    info = resolve_annotation_kinematic_state_info(
        frame_states,
        model_label="demo",
        frame_number=10,
        model=model,
    )

    assert info.is_exact is True
    assert info.source_frame == 10
    np.testing.assert_allclose(info.state, np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0]))


def test_resolve_annotation_kinematic_state_info_returns_nearest_valid_state():
    model = _FakeModel(2)
    frame_states = {
        ("demo", 4): np.array([4.0, 5.0], dtype=float),
        ("demo", 14): np.array([14.0, 15.0], dtype=float),
        ("other", 9): np.array([99.0, 99.0], dtype=float),
        ("demo", 11): np.array([1.0, 2.0, 3.0], dtype=float),
    }

    info = resolve_annotation_kinematic_state_info(
        frame_states,
        model_label="demo",
        frame_number=12,
        model=model,
    )

    assert info.is_exact is False
    assert info.source_frame == 14
    np.testing.assert_allclose(info.state, np.array([14.0, 15.0, 0.0, 0.0, 0.0, 0.0]))


def test_store_annotation_kinematic_state_normalizes_before_storing():
    model = _FakeModel(2)
    frame_states = {}

    stored = store_annotation_kinematic_state(
        frame_states,
        model_label="demo",
        frame_number=8,
        model=model,
        state=np.array([1.0, 2.0], dtype=float),
    )

    np.testing.assert_allclose(stored, np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(frame_states[("demo", 8)], stored)


def test_annotation_relevant_q_mask_targets_left_arm_and_trunk():
    mask = annotation_relevant_q_mask(
        [
            "TRUNK:RotX",
            "LEFT_UPPER_ARM:RotY",
            "LEFT_LOWER_ARM:RotZ",
            "RIGHT_THIGH:RotX",
        ],
        "left_wrist",
    )

    np.testing.assert_array_equal(mask, np.array([True, True, True, False]))


def test_constrain_annotation_state_to_mask_freezes_inactive_dofs():
    state = np.array([10.0, 20.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    reference = np.array([5.0, 6.0, 0.1, 0.2, 0.3, 0.4], dtype=float)

    constrained = constrain_annotation_state_to_mask(state, reference, np.array([True, False]))

    np.testing.assert_allclose(constrained, np.array([10.0, 6.0, 1.0, 0.2, 3.0, 0.4]))


def test_refine_annotation_window_states_propagates_and_refines_neighbor_frames(monkeypatch):
    model = _FakeModel(2)
    seed_state = np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    pose_data = PoseData(
        camera_names=["cam0"],
        frames=np.array([0], dtype=int),
        keypoints=np.zeros((1, 1, 17, 2), dtype=float),
        scores=np.ones((1, 1, 17), dtype=float),
    )
    propagated_deltas = []
    refined_frames = []

    def _fake_propagate(_model, state, *, dt, frame_delta):
        propagated_deltas.append((dt, frame_delta))
        propagated = np.array(state, copy=True)
        propagated[:2] += float(frame_delta)
        return propagated

    def _fake_refine(**kwargs):
        refined_frames.append(int(kwargs["frame_number"]))
        refined = np.array(kwargs["seed_state"], copy=True)
        refined[:2] += 10.0
        return refined, {"used_fallback": False}

    monkeypatch.setattr(kinematic_assist, "propagate_annotation_kinematic_state", _fake_propagate)
    monkeypatch.setattr(kinematic_assist, "refine_annotation_q_with_local_ekf", _fake_refine)

    refined_states, diagnostics = refine_annotation_window_states(
        model=model,
        calibrations={"cam0": object()},
        pose_data_by_frame={9: pose_data, 10: pose_data, 11: pose_data},
        center_frame_number=10,
        seed_state=seed_state,
        fps=120.0,
        passes=2,
        epipolar_threshold_px=15.0,
        q_names=["TRUNK:RotX", "TRUNK:RotY"],
    )

    assert sorted(refined_states) == [9, 10, 11]
    np.testing.assert_allclose(refined_states[10], seed_state)
    assert refined_frames == [11, 9]
    assert propagated_deltas == [(1.0 / 120.0, 1), (1.0 / 120.0, -1)]
    assert diagnostics["completed_frames"] == 2
    assert diagnostics["frame_statuses"] == {11: "corrected", 9: "corrected"}
