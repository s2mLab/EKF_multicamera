import numpy as np

from analysis.plot_kinematic_comparison import (
    compute_trunk_dofs_from_triangulation,
    extract_trunk_root_dofs,
    extract_trunk_root_dofs_no_unwrap,
)


def test_compute_trunk_dofs_from_triangulation_matches_shared_root_layout():
    points = np.full((1, 17, 3), np.nan, dtype=float)
    points[0, 11] = [0.0, 1.0, 0.0]
    points[0, 12] = [0.0, -1.0, 0.0]
    points[0, 5] = [0.0, 1.0, 1.0]
    points[0, 6] = [0.0, -1.0, 1.0]
    trans, rot = compute_trunk_dofs_from_triangulation(points)
    np.testing.assert_allclose(trans[0], np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(rot[0], np.zeros(3), atol=1e-12)


def test_extract_trunk_root_dofs_no_unwrap_preserves_wrapped_branch():
    q_names = np.asarray(
        [
            "TRUNK:TransX",
            "TRUNK:TransY",
            "TRUNK:TransZ",
            "TRUNK:RotY",
            "TRUNK:RotX",
            "TRUNK:RotZ",
        ],
        dtype=object,
    )
    q = np.array(
        [
            [0.0, 0.0, 0.0, 0.2, -0.1, 3.10],
            [0.0, 0.0, 0.0, 0.2, -0.1, -3.12],
        ],
        dtype=float,
    )
    unwrapped = extract_trunk_root_dofs(q_names, q)
    no_unwrap = extract_trunk_root_dofs_no_unwrap(q_names, q)
    assert unwrapped[1, 5] > 3.0
    assert no_unwrap[1, 5] < -3.0
