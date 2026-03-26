from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tkinter")

import pipeline_gui


class _FakeAxis:
    def __init__(self):
        self.plots = []
        self.scatters = []
        self.ylabel = None

    def plot(self, *args, **kwargs):
        self.plots.append({"args": args, "kwargs": kwargs})
        return None

    def scatter(self, *args, **kwargs):
        self.scatters.append({"args": args, "kwargs": kwargs})
        return None

    def set_title(self, *_args, **_kwargs):
        return None

    def grid(self, *_args, **_kwargs):
        return None

    def set_ylabel(self, value, **_kwargs):
        self.ylabel = value
        return None

    def set_xlabel(self, *_args, **_kwargs):
        return None

    def get_legend_handles_labels(self):
        return [], []

    def legend(self, *_args, **_kwargs):
        return None


class _FakeFigure:
    def clear(self):
        return None

    def subplots(self, nrows, _ncols=None, **_kwargs):
        if int(nrows) == 1:
            return _FakeAxis()
        return np.asarray([_FakeAxis() for _ in range(int(nrows))], dtype=object)

    def tight_layout(self):
        return None


class _FakeCanvas:
    def draw_idle(self):
        return None


class _FakeListbox:
    def __init__(self, items: list[str], selected: list[int]):
        self.items = list(items)
        self.selected = list(selected)

    def size(self):
        return len(self.items)

    def get(self, idx: int):
        return self.items[idx]

    def curselection(self):
        return tuple(self.selected)

    def delete(self, _start, _end=None):
        self.items.clear()
        self.selected.clear()

    def insert(self, _where, value: str):
        self.items.append(value)

    def selection_set(self, idx: int):
        if idx not in self.selected:
            self.selected.append(idx)


def test_joint_kinematics_refresh_plot_does_not_repopulate_rows(monkeypatch):
    bundle = {
        "q_names": np.asarray(["LEFT_KNEE:RotY", "RIGHT_KNEE:RotY"], dtype=object),
        "recon_q": {"demo": np.zeros((20, 2), dtype=float)},
        "recon_qdot": {"demo": np.zeros((20, 2), dtype=float)},
    }

    tab = pipeline_gui.JointKinematicsTab.__new__(pipeline_gui.JointKinematicsTab)
    tab.state = SimpleNamespace(
        fps_var=SimpleNamespace(get=lambda: "120"),
        shared_reconstruction_selection=["demo"],
    )
    tab.pair_list = _FakeListbox(["Knee"], [0])
    tab.figure = _FakeFigure()
    tab.canvas = _FakeCanvas()
    tab.quantity = SimpleNamespace(get=lambda: "q")
    tab.rotation_unit = SimpleNamespace(get=lambda: "rad")
    tab.fd_qdot_var = SimpleNamespace(get=lambda: False)
    tab.bundle = None
    tab.q_names = np.array([], dtype=object)
    tab._show_empty_plot = lambda _message: (_ for _ in ()).throw(AssertionError("unexpected empty plot"))
    tab._set_reconstruction_rows = lambda _rows, _defaults: (_ for _ in ()).throw(
        AssertionError("refresh_plot should not repopulate rows")
    )

    monkeypatch.setattr(pipeline_gui, "get_cached_preview_bundle", lambda *_args, **_kwargs: bundle)
    monkeypatch.setattr(pipeline_gui, "current_dataset_dir", lambda _state: "output/1_partie_0429")
    monkeypatch.setattr(pipeline_gui, "bundle_available_reconstruction_names", lambda *_args, **_kwargs: ["demo"])
    monkeypatch.setattr(
        pipeline_gui, "pair_dof_names", lambda _q_names: [("Knee", "LEFT_KNEE:RotY", "RIGHT_KNEE:RotY")]
    )
    monkeypatch.setattr(
        pipeline_gui.messagebox,
        "showerror",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected GUI error")),
    )

    pipeline_gui.JointKinematicsTab.refresh_plot(tab)


def test_joint_kinematics_sync_dataset_dir_only_refreshes(monkeypatch):
    tab = pipeline_gui.JointKinematicsTab.__new__(pipeline_gui.JointKinematicsTab)
    tab.state = SimpleNamespace()
    calls = []
    tab.refresh_available_reconstructions = lambda: calls.append("refresh")

    pipeline_gui.JointKinematicsTab.sync_dataset_dir(tab)

    assert calls == ["refresh"]


def test_joint_kinematics_pair_selection_triggers_refresh():
    tab = pipeline_gui.JointKinematicsTab.__new__(pipeline_gui.JointKinematicsTab)
    calls = []
    tab.refresh_plot = lambda: calls.append("refresh")

    pipeline_gui.JointKinematicsTab._on_pair_selection_changed(tab)

    assert calls == ["refresh"]


def test_joint_kinematics_uses_reconstruction_colors_and_side_styles(monkeypatch):
    axis = _FakeAxis()
    bundle = {
        "q_names": np.asarray(["LEFT_KNEE:RotY", "RIGHT_KNEE:RotY"], dtype=object),
        "recon_q": {"demo": np.column_stack((np.linspace(0.0, 1.0, 20), np.linspace(1.0, 0.0, 20)))},
        "recon_qdot": {"demo": np.zeros((20, 2), dtype=float)},
    }

    tab = pipeline_gui.JointKinematicsTab.__new__(pipeline_gui.JointKinematicsTab)
    tab.state = SimpleNamespace(
        fps_var=SimpleNamespace(get=lambda: "120"),
        shared_reconstruction_selection=["demo"],
    )
    tab.pair_list = _FakeListbox(["Knee"], [0])
    tab.figure = _FakeFigure()
    tab.figure.subplots = lambda *_args, **_kwargs: axis
    tab.canvas = _FakeCanvas()
    tab.quantity = SimpleNamespace(get=lambda: "q")
    tab.rotation_unit = SimpleNamespace(get=lambda: "rad")
    tab.fd_qdot_var = SimpleNamespace(get=lambda: False)
    tab.bundle = None
    tab.q_names = np.array([], dtype=object)
    tab._show_empty_plot = lambda _message: (_ for _ in ()).throw(AssertionError("unexpected empty plot"))

    monkeypatch.setattr(pipeline_gui, "get_cached_preview_bundle", lambda *_args, **_kwargs: bundle)
    monkeypatch.setattr(pipeline_gui, "current_dataset_dir", lambda _state: "output/1_partie_0429")
    monkeypatch.setattr(pipeline_gui, "bundle_available_reconstruction_names", lambda *_args, **_kwargs: ["demo"])
    monkeypatch.setattr(
        pipeline_gui, "pair_dof_names", lambda _q_names: [("Knee", "LEFT_KNEE:RotY", "RIGHT_KNEE:RotY")]
    )
    monkeypatch.setattr(pipeline_gui, "reconstruction_display_color", lambda _state, _name: "#123456")
    monkeypatch.setattr(pipeline_gui, "reconstruction_legend_label", lambda _state, _name: "7")
    monkeypatch.setattr(
        pipeline_gui.messagebox,
        "showerror",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected GUI error")),
    )

    pipeline_gui.JointKinematicsTab.refresh_plot(tab)

    assert len(axis.plots) == 2
    assert axis.plots[0]["kwargs"]["color"] == "#123456"
    assert axis.plots[0]["kwargs"]["linestyle"] == "-"
    assert axis.plots[0]["kwargs"]["marker"] == "o"
    assert axis.plots[0]["kwargs"]["label"] == "7 | L"
    assert axis.plots[1]["kwargs"]["color"] == "#123456"
    assert axis.plots[1]["kwargs"]["linestyle"] == "--"
    assert axis.plots[1]["kwargs"]["marker"] == "s"
    assert axis.plots[1]["kwargs"]["label"] == "7 | R"


def test_joint_kinematics_rotation_unit_popup_converts_rotational_dofs(monkeypatch):
    axis = _FakeAxis()
    bundle = {
        "q_names": np.asarray(["LEFT_KNEE:RotY", "RIGHT_KNEE:RotY"], dtype=object),
        "recon_q": {"demo": np.column_stack((np.linspace(0.0, np.pi, 20), np.linspace(np.pi, 0.0, 20)))},
        "recon_qdot": {"demo": np.zeros((20, 2), dtype=float)},
    }

    tab = pipeline_gui.JointKinematicsTab.__new__(pipeline_gui.JointKinematicsTab)
    tab.state = SimpleNamespace(
        fps_var=SimpleNamespace(get=lambda: "120"),
        shared_reconstruction_selection=["demo"],
    )
    tab.pair_list = _FakeListbox(["Knee"], [0])
    tab.figure = _FakeFigure()
    tab.figure.subplots = lambda *_args, **_kwargs: axis
    tab.canvas = _FakeCanvas()
    tab.quantity = SimpleNamespace(get=lambda: "q")
    tab.rotation_unit = SimpleNamespace(get=lambda: "deg")
    tab.fd_qdot_var = SimpleNamespace(get=lambda: False)
    tab.bundle = None
    tab.q_names = np.array([], dtype=object)
    tab._show_empty_plot = lambda _message: (_ for _ in ()).throw(AssertionError("unexpected empty plot"))

    monkeypatch.setattr(pipeline_gui, "get_cached_preview_bundle", lambda *_args, **_kwargs: bundle)
    monkeypatch.setattr(pipeline_gui, "current_dataset_dir", lambda _state: "output/1_partie_0429")
    monkeypatch.setattr(pipeline_gui, "bundle_available_reconstruction_names", lambda *_args, **_kwargs: ["demo"])
    monkeypatch.setattr(
        pipeline_gui, "pair_dof_names", lambda _q_names: [("Knee", "LEFT_KNEE:RotY", "RIGHT_KNEE:RotY")]
    )
    monkeypatch.setattr(
        pipeline_gui.messagebox,
        "showerror",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected GUI error")),
    )

    pipeline_gui.JointKinematicsTab.refresh_plot(tab)

    expected_left = np.linspace(0.0, np.pi, 20)[10:] * 180.0 / np.pi
    expected_right = np.linspace(np.pi, 0.0, 20)[10:] * 180.0 / np.pi
    np.testing.assert_allclose(axis.plots[0]["args"][1], expected_left)
    np.testing.assert_allclose(axis.plots[1]["args"][1], expected_right)
    assert axis.ylabel == "deg"


def test_pair_dof_names_includes_upper_back_singletons():
    q_names = np.asarray(
        ["LEFT_KNEE:RotY", "RIGHT_KNEE:RotY", "UPPER_BACK:RotX", "UPPER_BACK:RotY", "UPPER_BACK:RotZ"],
        dtype=object,
    )

    pairs = pipeline_gui.pair_dof_names(q_names)

    assert ("KNEE:RotY", "LEFT_KNEE:RotY", "RIGHT_KNEE:RotY") in pairs
    assert ("UPPER_BACK:RotX", "UPPER_BACK:RotX", None) in pairs
    assert ("UPPER_BACK:RotY", "UPPER_BACK:RotY", None) in pairs
    assert ("UPPER_BACK:RotZ", "UPPER_BACK:RotZ", None) in pairs


def test_upper_back_target_series_uses_hips_for_roty_and_zero_for_other_axes():
    series = np.array(
        [
            [0.0, 1.0, 0.5, -0.2, 0.3],
            [0.0, 0.5, 1.5, 0.1, -0.1],
        ],
        dtype=float,
    )
    name_to_index = {
        "UPPER_BACK:RotY": 0,
        "LEFT_THIGH:RotY": 1,
        "RIGHT_THIGH:RotY": 2,
        "UPPER_BACK:RotX": 3,
        "UPPER_BACK:RotZ": 4,
    }
    roty_target = pipeline_gui.JointKinematicsTab._upper_back_target_series(series, name_to_index, "UPPER_BACK:RotY")
    rotx_target = pipeline_gui.JointKinematicsTab._upper_back_target_series(series, name_to_index, "UPPER_BACK:RotX")
    rotz_target = pipeline_gui.JointKinematicsTab._upper_back_target_series(series, name_to_index, "UPPER_BACK:RotZ")

    np.testing.assert_allclose(roty_target, np.array([0.15, 0.2], dtype=float))
    np.testing.assert_allclose(rotx_target, np.zeros(2))
    np.testing.assert_allclose(rotz_target, np.zeros(2))


def test_draw_upper_back_preview_draws_back_centerline():
    axis = _FakeAxis()
    frame_points = np.full((len(pipeline_gui.COCO17), 3), np.nan, dtype=float)
    frame_points[pipeline_gui.KP_INDEX["left_hip"]] = np.array([0.0, 0.2, 0.0])
    frame_points[pipeline_gui.KP_INDEX["right_hip"]] = np.array([0.0, -0.2, 0.0])
    frame_points[pipeline_gui.KP_INDEX["left_shoulder"]] = np.array([0.0, 0.3, 1.0])
    frame_points[pipeline_gui.KP_INDEX["right_shoulder"]] = np.array([0.0, -0.3, 1.0])
    segment_frames = [
        ("TRUNK", np.array([0.0, 0.0, 0.0]), np.eye(3)),
        ("UPPER_BACK", np.array([0.0, 0.0, 0.5]), np.eye(3)),
    ]

    pipeline_gui.draw_upper_back_preview(axis, frame_points, segment_frames)

    assert len(axis.plots) >= 2
    assert len(axis.scatters) == 1
