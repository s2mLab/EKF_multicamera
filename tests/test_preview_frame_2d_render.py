import numpy as np

from preview.frame_2d_render import SkeletonLayer2D, render_camera_frame_2d


class _FakeAxis:
    def __init__(self) -> None:
        self.title = None
        self.aspect = None
        self.xlim = None
        self.ylim = None
        self.autoscale = None
        self.xlabel = None
        self.ylabel = None
        self.grid_alpha = None

    def set_xlim(self, *args):
        self.xlim = args

    def set_ylim(self, *args):
        self.ylim = args

    def set_autoscale_on(self, value):
        self.autoscale = value

    def set_aspect(self, aspect, adjustable=None):
        self.aspect = (aspect, adjustable)

    def set_title(self, title):
        self.title = title

    def grid(self, alpha=None):
        self.grid_alpha = alpha

    def set_xlabel(self, label):
        self.xlabel = label

    def set_ylabel(self, label):
        self.ylabel = label


def test_render_camera_frame_2d_draws_background_layers_and_explicit_limits():
    ax = _FakeAxis()
    calls = {"background": 0, "layers": 0, "hidden": 0}

    has_background = render_camera_frame_2d(
        ax,
        width=640,
        height=480,
        title="cam0",
        layers=[SkeletonLayer2D(points=np.zeros((17, 2), dtype=float), color="#123456", label="Raw")],
        draw_skeleton_fn=lambda *_args, **_kwargs: calls.__setitem__("layers", calls["layers"] + 1),
        background_image=np.zeros((10, 10, 3), dtype=float),
        draw_background_fn=lambda *_args, **_kwargs: calls.__setitem__("background", calls["background"] + 1),
        x_limits=(10.0, 110.0),
        y_limits=(210.0, 10.0),
        hide_axes=True,
        hide_axes_fn=lambda _ax: calls.__setitem__("hidden", calls["hidden"] + 1),
    )

    assert has_background is True
    assert ax.title == "cam0"
    assert ax.xlim == (10.0, 110.0)
    assert ax.ylim == (210.0, 10.0)
    assert ax.aspect == ("equal", "box")
    assert calls == {"background": 1, "layers": 1, "hidden": 1}
