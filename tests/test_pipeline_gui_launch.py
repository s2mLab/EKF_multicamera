import json
import os

import pytest

import pipeline_gui


def _find_tab_by_text(app, text: str):
    for tab_id in app.notebook.tabs():
        if app.notebook.tab(tab_id, "text") == text:
            return app.nametowidget(tab_id)
    raise AssertionError(f"Notebook tab {text!r} not found")


def _find_descendant_by_text(widget, *, cls, text: str):
    for child in widget.winfo_children():
        if isinstance(child, cls) and str(child.cget("text")) == text:
            return child
        nested = _find_descendant_by_text(child, cls=cls, text=text)
        if nested is not None:
            return nested
    return None


def _assert_widget_inside_parent(widget, *, tolerance: int = 4):
    widget.update_idletasks()
    parent = widget.nametowidget(widget.winfo_parent())
    parent.update_idletasks()
    assert widget.winfo_x() >= -tolerance
    assert widget.winfo_y() >= -tolerance
    assert widget.winfo_width() > 0
    assert widget.winfo_height() > 0
    assert widget.winfo_x() + widget.winfo_width() <= parent.winfo_width() + tolerance
    assert widget.winfo_y() + widget.winfo_height() <= parent.winfo_height() + tolerance


def test_small_keypoint_fixture_keeps_15_frames_per_camera():
    fixture_path = pipeline_gui.ROOT / "inputs/keypoints/1_partie_0429_15f_keypoints.json"
    with fixture_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    assert data
    assert all(len(camera_data["frames"]) == 15 for camera_data in data.values())
    assert all(len(camera_data["keypoints"]) == 15 for camera_data in data.values())
    assert all(len(camera_data["scores"]) == 15 for camera_data in data.values())


@pytest.mark.skipif(
    os.environ.get("RUN_PIPELINE_GUI_SMOKE") != "1",
    reason="Set RUN_PIPELINE_GUI_SMOKE=1 to enable the real Tk smoke test.",
)
def test_launcher_app_starts_with_small_keypoint_fixture(monkeypatch):
    pytest.importorskip("tkinter")

    monkeypatch.setattr(
        pipeline_gui,
        "DEFAULT_GUI_KEYPOINTS_PATH",
        "inputs/keypoints/1_partie_0429_15f_keypoints.json",
    )
    monkeypatch.setattr(pipeline_gui, "DEFAULT_GUI_CALIB_PATH", "inputs/calibration/Calib.toml")
    monkeypatch.setattr(pipeline_gui, "DEFAULT_GUI_TRC_PATH", "inputs/trc/1_partie_0429.trc")

    app = pipeline_gui.LauncherApp()
    try:
        app.withdraw()
        app.update_idletasks()
        app.update()

        assert app.notebook.index("end") >= 1
        assert app.state.keypoints_var.get() == "inputs/keypoints/1_partie_0429_15f_keypoints.json"
    finally:
        app.destroy()


@pytest.mark.skipif(
    os.environ.get("RUN_PIPELINE_GUI_SMOKE") != "1",
    reason="Set RUN_PIPELINE_GUI_SMOKE=1 to enable the real Tk smoke test.",
)
def test_launcher_layout_keeps_annotation_camera_and_model_widgets_inside_window(monkeypatch):
    pytest.importorskip("tkinter")

    monkeypatch.setattr(
        pipeline_gui,
        "DEFAULT_GUI_KEYPOINTS_PATH",
        "inputs/keypoints/1_partie_0429_15f_keypoints.json",
    )
    monkeypatch.setattr(pipeline_gui, "DEFAULT_GUI_CALIB_PATH", "inputs/calibration/Calib.toml")
    monkeypatch.setattr(pipeline_gui, "DEFAULT_GUI_TRC_PATH", "inputs/trc/1_partie_0429.trc")

    app = pipeline_gui.LauncherApp()
    try:
        app.update_idletasks()
        app.update()

        annotation_tab = _find_tab_by_text(app, "Annotation")
        app.notebook.select(annotation_tab)
        app.update_idletasks()
        app.update()
        _assert_widget_inside_parent(annotation_tab.annotation_cameras_list)
        _assert_widget_inside_parent(annotation_tab.annotation_keypoints_list)
        _assert_widget_inside_parent(annotation_tab.preview_canvas_widget)
        _assert_widget_inside_parent(annotation_tab.frame_scale)

        cameras_tab = _find_tab_by_text(app, "Cameras")
        app.notebook.select(cameras_tab)
        app.update_idletasks()
        app.update()
        _assert_widget_inside_parent(cameras_tab.metrics_tree)
        _assert_widget_inside_parent(cameras_tab.flip_canvas_widget)

        models_tab = _find_tab_by_text(app, "Models")
        app.notebook.select(models_tab)
        app.update_idletasks()
        app.update()
        symmetrize_check = _find_descendant_by_text(
            models_tab,
            cls=pipeline_gui.ttk.Checkbutton,
            text="Symmetrize limbs",
        )
        assert symmetrize_check is not None
        _assert_widget_inside_parent(symmetrize_check)
    finally:
        app.destroy()
