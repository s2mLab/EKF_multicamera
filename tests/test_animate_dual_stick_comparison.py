import numpy as np

from animation.animate_dual_stick_comparison import compute_axis_limits, compute_frame_axis_limits


def test_compute_frame_axis_limits_uses_only_current_frame():
    recon_points = {
        "a": np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                [[10.0, 10.0, 10.0], [11.0, 11.0, 11.0]],
            ],
            dtype=float,
        ),
        "b": np.array(
            [
                [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
                [[20.0, 20.0, 20.0], [21.0, 21.0, 21.0]],
            ],
            dtype=float,
        ),
    }

    full_limits = compute_axis_limits(*recon_points.values())
    frame_limits = compute_frame_axis_limits(recon_points, ["a", "b"], frame_idx=0)

    assert full_limits != frame_limits
    assert frame_limits == compute_axis_limits(recon_points["a"][0:1], recon_points["b"][0:1])
