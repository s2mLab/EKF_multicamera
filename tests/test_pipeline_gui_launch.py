import json
import os

import pytest

import pipeline_gui


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
