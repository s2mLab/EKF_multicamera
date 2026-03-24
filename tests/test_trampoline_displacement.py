import numpy as np

from judging.dd_analysis import DDSessionAnalysis, JumpSegment
from judging.trampoline_displacement import (
    TRAMPOLINE_GEOMETRY,
    X_INNER,
    X_MAX,
    Y_INNER,
    Y_MAX,
    analyze_trampoline_contacts,
    contact_segments_between_jumps,
    total_trampoline_penalty,
    trampoline_geometry_from_reference,
    trampoline_penalty_refined,
)


def _fake_session() -> DDSessionAnalysis:
    return DDSessionAnalysis(
        root_q=np.zeros((10, 6)),
        height=np.zeros(10),
        smoothed_height=np.zeros(10),
        height_threshold=0.0,
        airborne_regions=[],
        jump_segments=[
            JumpSegment(start=1, end=3, peak_index=2),
            JumpSegment(start=5, end=7, peak_index=6),
            JumpSegment(start=8, end=9, peak_index=8),
        ],
        jumps=[],
    )


def test_contact_segments_between_jumps_uses_intervals_between_jump_boundaries():
    assert contact_segments_between_jumps(_fake_session()) == [(3, 5), (7, 8)]


def test_trampoline_penalty_refined_matches_expected_zones():
    assert trampoline_penalty_refined(0.0, 0.0) == 0.0
    assert trampoline_penalty_refined(0.5 * (X_INNER + X_MAX), 0.0) == 0.1
    assert trampoline_penalty_refined(0.0, 0.5 * (Y_INNER + Y_MAX)) == 0.1
    assert trampoline_penalty_refined(0.5 * (X_INNER + X_MAX), 0.5 * (Y_INNER + Y_MAX)) == 0.3


def test_trampoline_geometry_is_centered_and_contains_expected_markers():
    geometry = trampoline_geometry_from_reference()
    np.testing.assert_allclose(geometry.center, [0.0, 0.0])
    assert set(geometry.cross) == {"left", "right", "bottom", "top"}
    assert set(geometry.small_square) == {"bottom_left", "top_left", "bottom_right", "top_right"}
    assert set(geometry.big_rectangle) == {"bottom_left", "top_left", "bottom_right", "top_right"}
    assert X_INNER < X_MAX
    assert Y_INNER < Y_MAX
    np.testing.assert_allclose(geometry.cross["left"], TRAMPOLINE_GEOMETRY.cross["left"])


def test_analyze_trampoline_contacts_uses_contact_medians_and_sums_penalties():
    session = _fake_session()
    xy = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.2, 0.1],
            [0.2, 0.1],
            [0.2, 0.1],
            [0.0, 0.0],
            [0.5 * (X_INNER + X_MAX), 0.0],
            [0.5 * (X_INNER + X_MAX), 0.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )
    contacts = analyze_trampoline_contacts(session, xy)
    assert len(contacts) == 2
    assert contacts[0].penalty == 0.0
    assert contacts[1].penalty == 0.1
    assert abs(total_trampoline_penalty(contacts) - 0.1) < 1e-12


def test_analyze_trampoline_contacts_from_feet_uses_worst_foot_penalty():
    session = _fake_session()
    points = np.full((10, 17, 3), np.nan, dtype=float)
    points[:, 15, :2] = [0.0, 0.0]
    points[:, 16, :2] = [0.0, 0.0]
    points[3:6, 15, :2] = [0.0, 0.0]
    points[3:6, 16, :2] = [0.5 * (X_INNER + X_MAX), 0.0]
    points[7:9, 15, :2] = [0.5 * (X_INNER + X_MAX), 0.5 * (Y_INNER + Y_MAX)]
    points[7:9, 16, :2] = [0.0, 0.0]

    contacts = analyze_trampoline_contacts(session, points)
    assert len(contacts) == 2
    assert contacts[0].penalty == 0.1
    assert abs(contacts[0].x - 0.5 * (X_INNER + X_MAX)) < 1e-12
    assert contacts[1].penalty == 0.3
    assert abs(total_trampoline_penalty(contacts) - 0.4) < 1e-12
