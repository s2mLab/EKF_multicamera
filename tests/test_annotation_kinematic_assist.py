import numpy as np

from annotation.kinematic_assist import (
    annotation_relevant_q_mask,
    constrain_annotation_state_to_mask,
    resolve_annotation_kinematic_state_info,
    store_annotation_kinematic_state,
)


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
