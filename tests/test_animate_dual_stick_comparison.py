import numpy as np

from animation.animate_dual_stick_comparison import (
    KP_INDEX,
    compute_axis_limits,
    compute_frame_axis_limits,
    init_artists,
    trampoline_contact_zone_xy,
)
from judging.trampoline_displacement import X_INNER, X_MAX, Y_INNER


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


def test_trampoline_contact_zone_xy_returns_the_strongest_judged_zone():
    frame_points = np.full((17, 3), np.nan, dtype=float)
    frame_points[KP_INDEX["left_ankle"]] = [0.0, 0.0, 1.2]
    frame_points[KP_INDEX["right_ankle"]] = [0.5 * (X_INNER + X_MAX), 0.0, 1.2]

    polygon_xy = trampoline_contact_zone_xy([frame_points])

    assert polygon_xy is not None
    assert polygon_xy.shape == (4, 2)
    np.testing.assert_allclose(
        polygon_xy,
        [[X_INNER, -Y_INNER], [X_MAX, -Y_INNER], [X_MAX, Y_INNER], [X_INNER, Y_INNER]],
    )


class _FakeLine:
    def __init__(self, kwargs):
        self.kwargs = kwargs


class _FakeAxes3D:
    def __init__(self):
        self.scatter_calls = []
        self.plot_calls = []

    def scatter(self, *args, **kwargs):
        self.scatter_calls.append((args, kwargs))
        return object()

    def plot(self, *args, **kwargs):
        self.plot_calls.append((args, kwargs))
        return (_FakeLine(kwargs),)


def test_init_artists_skips_markers_when_marker_size_is_zero():
    ax = _FakeAxes3D()

    scatter, lines = init_artists(ax, color="#4c72b0", label="1", marker_size=0.0)

    assert ax.scatter_calls == []
    assert scatter == {"center": None, "left": None, "right": None}
    assert len(lines) > 0
    assert ax.plot_calls[0][1]["label"] == "1"
    assert all(call[1].get("label") is None for call in ax.plot_calls[1:])
