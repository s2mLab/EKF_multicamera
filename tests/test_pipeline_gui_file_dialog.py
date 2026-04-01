import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.figure import Figure

import pipeline_gui
from annotation import frame_navigation
from preview import two_d_view
from preview.dataset_preview_state import DatasetPreviewState
from preview.shared_reconstruction_panel import SharedReconstructionPanel
from vitpose_ekf_pipeline import CameraCalibration


class _FakeTree:
    def __init__(self):
        self.rows = {}
        self._selection = ()
        self._focus = None
        self._identified_row = None
        self._seen = None

    def get_children(self, _root=""):
        return tuple(self.rows.keys())

    def delete(self, item):
        self.rows.pop(item, None)

    def insert(self, _parent, _where, iid, values):
        self.rows[iid] = tuple(values)

    def item(self, item, option=None):
        values = self.rows[item]
        if option == "values":
            return values
        return {"values": values}

    def selection_set(self, selection):
        self._selection = tuple(selection)

    def selection_add(self, selection):
        if isinstance(selection, str):
            values = [selection]
        else:
            values = list(selection)
        updated = list(self._selection)
        for value in values:
            if value not in updated:
                updated.append(value)
        self._selection = tuple(updated)

    def selection(self):
        return self._selection

    def exists(self, name):
        return name in self.rows

    def identify_row(self, _y):
        return self._identified_row or ""

    def focus(self, item=None):
        if item is not None:
            self._focus = item
        return self._focus

    def see(self, item):
        self._seen = item


def test_normalize_pose_correction_mode_accepts_epipolar_fast():
    assert pipeline_gui.normalize_pose_correction_mode("flip_epipolar_fast") == "flip_epipolar_fast"


def test_normalize_pose_correction_mode_accepts_explicit_viterbi_modes():
    assert pipeline_gui.normalize_pose_correction_mode("flip_epipolar_viterbi") == "flip_epipolar_viterbi"
    assert pipeline_gui.normalize_pose_correction_mode("flip_epipolar_fast_viterbi") == "flip_epipolar_fast_viterbi"


def test_normalize_pose_correction_mode_falls_back_to_none():
    assert pipeline_gui.normalize_pose_correction_mode("unexpected_mode") == "none"


def test_schedule_after_idle_once_coalesces_multiple_requests():
    calls = []

    class _FakeWidget:
        def __init__(self):
            self._queued = []
            self._token = 0

        def after_idle(self, callback):
            self._token += 1
            self._queued.append((self._token, callback))
            return self._token

        def after_cancel(self, _token):
            return

    widget = _FakeWidget()

    pipeline_gui.schedule_after_idle_once(widget, "_scheduled_demo", lambda: calls.append("run"))
    pipeline_gui.schedule_after_idle_once(widget, "_scheduled_demo", lambda: calls.append("run"))

    assert len(widget._queued) == 1
    _token, callback = widget._queued.pop()
    callback()
    assert calls == ["run"]


def test_annotation_path_change_is_ignored_while_syncing_defaults():
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab._syncing_annotation_defaults = True
    tab.request_load_resources = lambda: (_ for _ in ()).throw(
        AssertionError("request_load_resources should not be called")
    )

    pipeline_gui.AnnotationTab._on_annotation_path_changed(tab)


def test_annotation_sync_dataset_defaults_requests_one_load(monkeypatch, tmp_path):
    keypoints_path = tmp_path / "inputs" / "keypoints" / "trial_keypoints.json"
    keypoints_path.parent.mkdir(parents=True)
    keypoints_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pipeline_gui, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline_gui, "default_annotation_path", lambda _path: tmp_path / "inputs" / "annotations.json")
    monkeypatch.setattr(pipeline_gui, "infer_execution_images_root", lambda _path: tmp_path / "inputs" / "images")

    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.state = SimpleNamespace(
        keypoints_var=SimpleNamespace(get=lambda: "inputs/keypoints/trial_keypoints.json"),
        annotation_path_var=SimpleNamespace(
            get=lambda: "", set=lambda value: setattr(tab, "_annotation_path_set", value)
        ),
    )
    tab.images_root_entry = SimpleNamespace(
        var=SimpleNamespace(get=lambda: "", set=lambda value: setattr(tab, "_images_root_set", value))
    )
    tab.refresh_kinematic_model_choices = lambda: setattr(tab, "_models_refreshed", True)
    calls = []
    tab.request_load_resources = lambda: calls.append("load")

    pipeline_gui.AnnotationTab.sync_dataset_defaults(tab)

    assert calls == ["load"]
    assert tab._annotation_path_set.endswith("inputs/annotations.json")
    assert tab._images_root_set.endswith("inputs/images")


def test_append_default_pose2sim_profile_adds_pose2sim_when_trc_exists():
    triangulation = pipeline_gui.ReconstructionProfile(name="tri", family="triangulation")
    pose2sim = pipeline_gui.ReconstructionProfile(name="p2s", family="pose2sim")

    selected = pipeline_gui.append_default_pose2sim_profile([triangulation], [triangulation, pose2sim], "trial.trc")

    assert [profile.name for profile in selected] == ["tri", "p2s"]


def test_append_default_pose2sim_profile_keeps_explicit_selection_without_trc():
    triangulation = pipeline_gui.ReconstructionProfile(name="tri", family="triangulation")
    pose2sim = pipeline_gui.ReconstructionProfile(name="p2s", family="pose2sim")

    selected = pipeline_gui.append_default_pose2sim_profile([triangulation], [triangulation, pose2sim], "")

    assert [profile.name for profile in selected] == ["tri"]


def test_keypoint_preset_names_body_only_removes_face_side_keypoints():
    body_only = pipeline_gui.keypoint_preset_names("body_only")

    assert "left_eye" not in body_only
    assert "right_eye" not in body_only
    assert "left_ear" not in body_only
    assert "right_ear" not in body_only
    assert "left_ankle" in body_only
    assert "right_wrist" in body_only


def test_annotation_keypoint_order_groups_left_then_right_then_head():
    assert list(pipeline_gui.ANNOTATION_KEYPOINT_ORDER[:6]) == [
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "left_hip",
        "left_knee",
        "left_ankle",
    ]
    assert list(pipeline_gui.ANNOTATION_KEYPOINT_ORDER[6:12]) == [
        "right_shoulder",
        "right_elbow",
        "right_wrist",
        "right_hip",
        "right_knee",
        "right_ankle",
    ]
    assert list(pipeline_gui.ANNOTATION_KEYPOINT_ORDER[12:]) == [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
    ]


def test_annotation_keypoint_names_for_biomod_adds_mid_back_only_for_segmented_back(monkeypatch):
    monkeypatch.setattr(pipeline_gui, "biomod_supports_upper_back_options", lambda path: bool(path))

    with_mid_back = pipeline_gui.annotation_keypoint_names_for_biomod("demo.bioMod")
    without_mid_back = pipeline_gui.annotation_keypoint_names_for_biomod(None)

    assert "mid_back" in with_mid_back
    assert with_mid_back.index("mid_back") == 12
    assert "mid_back" not in without_mid_back


def test_refresh_annotation_keypoint_choices_preserves_selection_when_possible(monkeypatch):
    monkeypatch.setattr(
        pipeline_gui,
        "annotation_keypoint_names_for_biomod",
        lambda _path: ("left_shoulder", "mid_back", "nose"),
    )

    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.annotation_keypoints_list = _FakeListbox()
    tab.current_marker_var = SimpleNamespace(set=lambda value: setattr(tab, "_marker_text", value))
    tab.kinematic_model_choices = {"demo": Path("demo.bioMod")}
    tab.kinematic_model_var = SimpleNamespace(get=lambda: "demo")
    for keypoint_name in ("left_shoulder", "nose"):
        tab.annotation_keypoints_list.insert("end", keypoint_name)
    tab.annotation_keypoints_list.selection_set(1)

    pipeline_gui.AnnotationTab.refresh_annotation_keypoint_choices(tab)

    assert tab.annotation_keypoints_list.items == ["left_shoulder", "mid_back", "nose"]
    assert tab.annotation_keypoints_list.curselection() == (2,)
    assert tab._marker_text == "Current marker: nose"


def test_load_shared_reconstruction_preview_state_returns_bundle_and_preview_state(monkeypatch):
    state = SimpleNamespace()
    bundle = {"recon_q": {"demo": object()}}
    preview_state = DatasetPreviewState(
        rows=[{"name": "demo"}], defaults=["demo"], available_names=["demo"], max_frame=9
    )

    monkeypatch.setattr(pipeline_gui, "current_dataset_dir", lambda _state: Path("outputs/demo"))
    monkeypatch.setattr(pipeline_gui, "get_cached_preview_bundle", lambda *_args, **_kwargs: bundle)
    monkeypatch.setattr(
        pipeline_gui,
        "current_dataset_preview_state",
        lambda _state, **_kwargs: (Path("outputs/demo"), preview_state),
    )

    output_dir, loaded_bundle, loaded_preview_state = pipeline_gui.load_shared_reconstruction_preview_state(
        state,
        preferred_names=["demo"],
        fallback_count=1,
    )

    assert output_dir == Path("outputs/demo")
    assert loaded_bundle is bundle
    assert loaded_preview_state is preview_state


def test_calibration_tab_refresh_analysis_uses_selected_reconstruction_payload(monkeypatch):
    tab = pipeline_gui.CalibrationTab.__new__(pipeline_gui.CalibrationTab)
    tab.pose_data = SimpleNamespace(frames=np.array([0], dtype=int))
    tab.calibrations = {"cam0": object()}
    tab.state = SimpleNamespace(
        shared_reconstruction_selection=("triangulation_exhaustive",),
        output_root_var=SimpleNamespace(get=lambda: "output"),
        keypoints_var=SimpleNamespace(get=lambda: "inputs/keypoints/demo_keypoints.json"),
    )
    tab.trim_fraction_var = SimpleNamespace(get=lambda: "15")
    tab.pose_data_mode = SimpleNamespace(get=lambda: "cleaned")
    tab.status_var = SimpleNamespace(set=lambda value: setattr(tab, "_status_text", value))
    tab.worst_frame_list = _FakeListbox()
    tab.render_summary = lambda: setattr(tab, "_summary_rendered", True)
    tab.refresh_plot = lambda: setattr(tab, "_plot_rendered", True)

    captured = {}

    monkeypatch.setattr(pipeline_gui, "current_dataset_dir", lambda _state: Path("output/demo"))
    monkeypatch.setattr(
        pipeline_gui,
        "reconstruction_dir_by_name",
        lambda _dataset_dir, _name: Path("output/demo/reconstructions/triangulation_exhaustive"),
    )
    monkeypatch.setattr(
        pipeline_gui,
        "load_bundle_payload",
        lambda _path: {"points_3d": np.zeros((1, 17, 3)), "reprojection_error_per_view": np.zeros((1, 17, 1))},
    )
    monkeypatch.setattr(pipeline_gui, "load_bundle_summary", lambda _path: {"pose_data_mode": "cleaned"})

    def _fake_qc(pose_data, calibrations, *, reconstruction_payload, trim_fraction, spatial_bins):
        captured["pose_data"] = pose_data
        captured["calibrations"] = calibrations
        captured["payload"] = reconstruction_payload
        captured["trim_fraction"] = trim_fraction
        captured["spatial_bins"] = spatial_bins
        return SimpleNamespace(
            two_d=SimpleNamespace(trim_fraction=trim_fraction, per_frame_mean_px=np.array([1.0], dtype=float)),
            three_d=None,
        )

    monkeypatch.setattr(pipeline_gui, "compute_calibration_qc", _fake_qc)

    pipeline_gui.CalibrationTab.refresh_analysis(tab)

    assert captured["pose_data"] is tab.pose_data
    assert captured["calibrations"] is tab.calibrations
    assert "points_3d" in captured["payload"]
    assert captured["trim_fraction"] == 0.15
    assert captured["spatial_bins"] == 3
    assert tab._summary_rendered is True
    assert tab._plot_rendered is True


def test_annotation_jump_context_uses_shared_jump_analysis(monkeypatch):
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = SimpleNamespace(frames=np.array([100], dtype=int))
    tab.state = SimpleNamespace(shared_reconstruction_selection=("demo_recon",))
    tab.annotation_jump_analysis = None
    tab.current_frame_number = lambda: 120

    analysis = SimpleNamespace(
        jumps=[
            SimpleNamespace(
                classification="straight",
                segment=SimpleNamespace(start=110, end=130),
            )
        ]
    )
    monkeypatch.setattr(pipeline_gui, "shared_jump_analysis_for_reconstruction", lambda _state, _name: analysis)

    text = pipeline_gui.AnnotationTab._annotation_jump_context(tab)

    assert text == "Jump context: S1 | straight | frames 110-130"
    assert tab.annotation_jump_analysis is analysis


def test_gui_busy_popup_does_not_show_before_delay(monkeypatch):
    created = []

    class _FakeBusyWindow:
        def __init__(self, _parent, title, message):
            created.append((title, message))

        def set_status(self, _message):
            return

        def close(self):
            return

        def update(self):
            return

    times = iter([0.0, 0.2])
    monkeypatch.setattr(pipeline_gui, "BusyStatusWindow", _FakeBusyWindow)
    monkeypatch.setattr(pipeline_gui.time, "monotonic", lambda: next(times))

    with pipeline_gui.gui_busy_popup(
        SimpleNamespace(update_idletasks=lambda: None), title="Test", message="Short"
    ) as popup:
        popup.set_status("Still short")

    assert created == []


def test_gui_busy_popup_shows_after_delay(monkeypatch):
    created = []

    class _FakeBusyWindow:
        def __init__(self, _parent, title, message):
            self.title = title
            self.message = message
            created.append((title, message))

        def set_status(self, message):
            self.message = message

        def close(self):
            return

        def update(self):
            return

    times = iter([0.0, 0.8])
    monkeypatch.setattr(pipeline_gui, "BusyStatusWindow", _FakeBusyWindow)
    monkeypatch.setattr(pipeline_gui.time, "monotonic", lambda: next(times))

    with pipeline_gui.gui_busy_popup(
        SimpleNamespace(update_idletasks=lambda: None), title="DD", message="Analyse..."
    ) as popup:
        popup.set_status("Long")

    assert created == [("DD", "Long")]


def test_annotation_only_pose_data_keeps_only_manual_annotations(tmp_path):
    keypoints_path = tmp_path / "inputs" / "keypoints" / "trial_keypoints.json"
    annotations_path = tmp_path / "inputs" / "annotations" / "trial_annotations.json"
    keypoints_path.parent.mkdir(parents=True)
    annotations_path.parent.mkdir(parents=True)
    keypoints_path.write_text("{}", encoding="utf-8")
    annotations_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "annotations": {"cam0": {"0": {"nose": {"xy": [11.0, 22.0], "score": 1.0}}}},
            }
        ),
        encoding="utf-8",
    )
    pose_data = pipeline_gui.PoseData(
        camera_names=["cam0", "cam1"],
        frames=np.array([0], dtype=int),
        keypoints=np.full((2, 1, len(pipeline_gui.COCO17), 2), 99.0, dtype=float),
        scores=np.ones((2, 1, len(pipeline_gui.COCO17)), dtype=float),
    )

    sparse_pose = pipeline_gui.annotation_only_pose_data(
        pose_data,
        keypoints_path=keypoints_path,
        annotations_path=annotations_path,
    )

    np.testing.assert_allclose(sparse_pose.keypoints[0, 0, 0], np.array([11.0, 22.0]))
    assert sparse_pose.scores[0, 0, 0] == 1.0
    assert np.isnan(sparse_pose.keypoints[1]).all()
    assert np.all(sparse_pose.scores[1] == 0.0)


def test_calibration_tab_refresh_analysis_hides_3d_when_reconstruction_pose_mode_mismatches(monkeypatch):
    tab = pipeline_gui.CalibrationTab.__new__(pipeline_gui.CalibrationTab)
    tab.pose_data = SimpleNamespace(frames=np.array([0], dtype=int))
    tab.calibrations = {"cam0": object()}
    tab.state = SimpleNamespace(shared_reconstruction_selection=("demo",))
    tab.trim_fraction_var = SimpleNamespace(get=lambda: "15")
    tab.pose_data_mode = SimpleNamespace(get=lambda: "annotated")
    tab.status_var = SimpleNamespace(set=lambda value: setattr(tab, "_status_text", value))
    tab.worst_frame_list = _FakeListbox()
    tab.render_summary = lambda: setattr(tab, "_summary_rendered", True)
    tab.refresh_plot = lambda: setattr(tab, "_plot_rendered", True)
    tab.refresh_worst_frame_list = lambda: setattr(tab, "_worst_rendered", True)

    captured = {}

    monkeypatch.setattr(pipeline_gui, "current_dataset_dir", lambda _state: Path("output/demo"))
    monkeypatch.setattr(
        pipeline_gui,
        "reconstruction_dir_by_name",
        lambda _dataset_dir, _name: Path("output/demo/reconstructions/demo"),
    )
    monkeypatch.setattr(
        pipeline_gui,
        "load_bundle_payload",
        lambda _path: {"points_3d": np.zeros((1, 17, 3)), "reprojection_error_per_view": np.zeros((1, 17, 1))},
    )
    monkeypatch.setattr(pipeline_gui, "load_bundle_summary", lambda _path: {"pose_data_mode": "cleaned"})

    def _fake_qc(pose_data, calibrations, *, reconstruction_payload, trim_fraction, spatial_bins):
        captured["reconstruction_payload"] = reconstruction_payload
        return SimpleNamespace(
            two_d=SimpleNamespace(trim_fraction=trim_fraction, per_frame_mean_px=np.array([1.0], dtype=float)),
            three_d=None,
        )

    monkeypatch.setattr(pipeline_gui, "compute_calibration_qc", _fake_qc)

    pipeline_gui.CalibrationTab.refresh_analysis(tab)

    assert captured["reconstruction_payload"] is None
    assert "3D hidden" in tab._status_text
    assert tab._summary_rendered is True
    assert tab._plot_rendered is True


def test_calibration_tab_refresh_pose_mode_choices_falls_back_when_annotated_missing(monkeypatch, tmp_path):
    root = tmp_path

    class _Var:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):
            self.value = value

    tab = pipeline_gui.CalibrationTab.__new__(pipeline_gui.CalibrationTab)
    tab.state = SimpleNamespace(
        keypoints_var=SimpleNamespace(get=lambda: "inputs/keypoints/trial_keypoints.json"),
        annotation_path_var=SimpleNamespace(get=lambda: "inputs/annotations/trial_annotations.json"),
    )
    tab.pose_mode_box = _FakeCombobox()
    tab.pose_data_mode = _Var("annotated")

    monkeypatch.setattr(pipeline_gui, "ROOT", root)

    pipeline_gui.CalibrationTab.refresh_pose_mode_choices(tab)

    assert tab.pose_mode_box.values == ["raw", "cleaned"]
    assert tab.pose_data_mode.get() == "cleaned"


def test_calibration_tab_load_resources_uses_local_pose_mode(monkeypatch):
    tab = pipeline_gui.CalibrationTab.__new__(pipeline_gui.CalibrationTab)
    tab.state = SimpleNamespace(
        keypoints_var=SimpleNamespace(get=lambda: "inputs/keypoints/trial_keypoints.json"),
        calib_var=SimpleNamespace(get=lambda: "inputs/calibration/Calib.toml"),
        pose_filter_window_var=SimpleNamespace(get=lambda: "9"),
        pose_outlier_ratio_var=SimpleNamespace(get=lambda: "0.1"),
        pose_p_low_var=SimpleNamespace(get=lambda: "5"),
        pose_p_high_var=SimpleNamespace(get=lambda: "95"),
        pose_data_mode_var=SimpleNamespace(get=lambda: "cleaned"),
        annotation_path_var=SimpleNamespace(get=lambda: "inputs/annotations/trial_annotations.json"),
    )
    tab.pose_mode_box = _FakeCombobox()
    tab.pose_data_mode = SimpleNamespace(get=lambda: "raw", set=lambda _value: None)
    tab.refresh_analysis = lambda: setattr(tab, "_analysis_refreshed", True)
    tab.refresh_pose_mode_choices = lambda: None

    captured = {}

    def _fake_get_cached_pose_data(_state, **kwargs):
        captured.update(kwargs)
        return {"cam0": object()}, object()

    monkeypatch.setattr(pipeline_gui, "get_cached_pose_data", _fake_get_cached_pose_data)

    pipeline_gui.CalibrationTab.load_resources(tab)

    assert captured["data_mode"] == "raw"
    assert tab._analysis_refreshed is True


def test_calibration_tab_refresh_worst_frame_list_populates_both_2d_and_3d():
    tab = pipeline_gui.CalibrationTab.__new__(pipeline_gui.CalibrationTab)
    tab.pose_data = SimpleNamespace(frames=np.array([10, 11, 12, 13], dtype=int))
    tab.qc = SimpleNamespace(
        two_d=SimpleNamespace(per_frame_mean_px=np.array([1.0, 4.0, np.nan, 3.0], dtype=float)),
        three_d=SimpleNamespace(per_frame_mean_px=np.array([2.0, np.nan, 7.0, 5.0], dtype=float)),
    )
    tab.worst_frame_list = _FakeListbox()

    pipeline_gui.CalibrationTab.refresh_worst_frame_list(tab)

    assert tab.worst_frame_list.items[0].startswith("2D | frame 11")
    assert any(item.startswith("3D | frame 12") for item in tab.worst_frame_list.items)


def test_camera_tools_tab_qa_overlay_data_reads_reprojection_and_excluded_payload(monkeypatch):
    tab = pipeline_gui.CameraToolsTab.__new__(pipeline_gui.CameraToolsTab)
    tab.pose_data = SimpleNamespace(camera_names=["cam0", "cam1"])
    tab.calibrations = {"cam0": object(), "cam1": object()}
    tab.qa_overlay_var = SimpleNamespace(get=lambda: "3D reproj")
    tab._reference_payload = lambda: {
        "reprojection_error_per_view": np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=float),
        "excluded_views": np.array([[[False, True], [True, False]]], dtype=bool),
    }

    label, values, mask, cmap = pipeline_gui.CameraToolsTab._qa_overlay_data(tab, "cam1", 0)

    assert label == "3D reproj"
    np.testing.assert_array_equal(values, np.array([2.0, 4.0]))
    assert mask is None
    assert cmap == "turbo"

    tab.qa_overlay_var = SimpleNamespace(get=lambda: "3D excluded")
    label, values, mask, cmap = pipeline_gui.CameraToolsTab._qa_overlay_data(tab, "cam0", 0)
    assert label == "3D excluded"
    assert values is None
    np.testing.assert_array_equal(mask, np.array([False, True]))
    assert cmap is None


def test_multiview_tab_qa_overlay_data_reads_reprojection_and_excluded_payload(monkeypatch):
    tab = pipeline_gui.MultiViewTab.__new__(pipeline_gui.MultiViewTab)
    tab.pose_data = SimpleNamespace(camera_names=["cam0", "cam1"])
    tab.calibrations = {"cam0": object(), "cam1": object()}
    tab.qa_overlay_var = SimpleNamespace(get=lambda: "3D reproj")
    tab._selected_reconstruction = lambda: "demo"
    tab.reconstruction_payloads = {
        "demo": {
            "reprojection_error_per_view": np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=float),
            "excluded_views": np.array([[[False, True], [True, False]]], dtype=bool),
        }
    }

    label, values, mask, cmap = pipeline_gui.MultiViewTab._qa_overlay_data(tab, "cam1", 0)

    assert label == "3D reproj"
    np.testing.assert_array_equal(values, np.array([2.0, 4.0]))
    assert mask is None
    assert cmap == "turbo"

    tab.qa_overlay_var = SimpleNamespace(get=lambda: "3D excluded")
    label, values, mask, cmap = pipeline_gui.MultiViewTab._qa_overlay_data(tab, "cam0", 0)
    assert label == "3D excluded"
    assert values is None
    np.testing.assert_array_equal(mask, np.array([False, True]))
    assert cmap is None


def test_get_cached_calibrations_reports_startup_status(monkeypatch, tmp_path):
    messages = []
    state = SimpleNamespace(calibration_cache={}, startup_status_callback=messages.append)
    calib_path = tmp_path / "Calib.toml"
    calib_path.write_text("")

    monkeypatch.setattr(pipeline_gui, "load_calibrations", lambda _path: {"cam0": object()})

    calibrations = pipeline_gui.get_cached_calibrations(state, calib_path)

    assert "Loading calibrations: Calib.toml" in messages
    assert calibrations.keys() == {"cam0"}


def test_get_cached_pose_data_reports_cached_status(monkeypatch, tmp_path):
    messages = []
    keypoints_path = tmp_path / "trial_keypoints.json"
    keypoints_path.write_text("{}")
    calib_path = tmp_path / "Calib.toml"
    calib_path.write_text("")
    cache_key = pipeline_gui.pose_data_cache_key(
        keypoints_path=keypoints_path,
        calib_path=calib_path,
        max_frames=None,
        frame_start=None,
        frame_end=None,
        data_mode="cleaned",
        smoothing_window=9,
        outlier_threshold_ratio=0.1,
        lower_percentile=5.0,
        upper_percentile=95.0,
    )
    cached_pose_data = object()
    state = SimpleNamespace(
        calibration_cache={pipeline_gui.calibration_cache_key(calib_path): {"cam0": object()}},
        pose_data_cache={cache_key: cached_pose_data},
        startup_status_callback=messages.append,
    )

    calibrations, pose_data = pipeline_gui.get_cached_pose_data(
        state,
        keypoints_path=keypoints_path,
        calib_path=calib_path,
    )

    assert "Using cached 2D poses: trial_keypoints.json" in messages
    assert "Using cached calibrations: Calib.toml" in messages
    assert calibrations.keys() == {"cam0"}
    assert pose_data is cached_pose_data


def test_reconstruction_legend_label_uses_shared_panel_order():
    tree = SimpleNamespace(get_children=lambda _root="": ("recon_b", "recon_a", "__placeholder__"))
    panel = SimpleNamespace(tree=tree)
    state = SimpleNamespace(shared_reconstruction_panel=panel, shared_reconstruction_selection=[])

    assert pipeline_gui.reconstruction_legend_label(state, "recon_b") == "1"
    assert pipeline_gui.reconstruction_legend_label(state, "recon_a") == "2"


def test_reconstruction_display_color_uses_shared_panel_order():
    tree = SimpleNamespace(get_children=lambda _root="": ("recon_b", "recon_a"))
    panel = SimpleNamespace(tree=tree)
    state = SimpleNamespace(shared_reconstruction_panel=panel, shared_reconstruction_selection=[])

    assert pipeline_gui.reconstruction_display_color(state, "recon_b") == "#4c72b0"
    assert pipeline_gui.reconstruction_display_color(state, "recon_a") == "#dd8452"


def test_reconstruction_legend_label_falls_back_to_reconstruction_name_without_panel_order():
    state = SimpleNamespace(shared_reconstruction_panel=None, shared_reconstruction_selection=[])

    assert pipeline_gui.reconstruction_legend_label(state, "pose2sim") == "Pose2Sim"


def test_reconstruction_display_color_falls_back_without_panel_order():
    state = SimpleNamespace(shared_reconstruction_panel=None, shared_reconstruction_selection=[])

    assert pipeline_gui.reconstruction_display_color(state, "pose2sim") == pipeline_gui.reconstruction_color("pose2sim")


def test_shared_reconstruction_panel_prepends_numeric_index_column():
    panel = SharedReconstructionPanel.__new__(SharedReconstructionPanel)
    panel.tree = _FakeTree()
    panel._default_names = []
    panel._selection_callback = None
    panel._refresh_callback = None
    panel._suspend_selection_callback = False
    panel.state = SimpleNamespace(set_shared_reconstruction_selection=lambda _names: None)

    SharedReconstructionPanel.set_rows(
        panel,
        rows=[
            {"name": "recon_b", "label": "B", "family": "ekf_2d", "frames": 12, "reproj_mean": 1.2, "path": "/b"},
            {"name": "recon_a", "label": "A", "family": "pose2sim", "frames": 8, "reproj_mean": 2.3, "path": "/a"},
        ],
        default_names=["recon_b"],
    )

    assert panel.tree.rows["recon_b"][0] == "1"
    assert panel.tree.rows["recon_a"][0] == "2"


def test_shared_reconstruction_panel_does_not_number_raw_2d_rows():
    panel = SharedReconstructionPanel.__new__(SharedReconstructionPanel)
    panel.tree = _FakeTree()
    panel._default_names = []
    panel._selection_callback = None
    panel._refresh_callback = None
    panel._suspend_selection_callback = False
    panel.state = SimpleNamespace(set_shared_reconstruction_selection=lambda _names: None)

    SharedReconstructionPanel.set_rows(
        panel,
        rows=[
            {"name": "raw", "label": "Raw 2D", "family": "2d", "frames": "-", "reproj_mean": None, "path": "-"},
            {
                "name": "pose2sim",
                "label": "TRC file",
                "family": "pose2sim",
                "frames": 8,
                "reproj_mean": 2.3,
                "path": "/a",
            },
        ],
        default_names=["raw"],
    )

    assert panel.tree.rows["raw"][0] == ""
    assert panel.tree.rows["pose2sim"][0] == "2"


def test_extend_listbox_selection_adds_next_item():
    listbox = _FakeListbox()
    for value in ("a", "b", "c"):
        listbox.insert("end", value)
    listbox.selection_set(0)

    result = pipeline_gui.extend_listbox_selection(listbox, 1)

    assert result == "break"
    assert listbox.curselection() == (0, 1)
    assert listbox._active == 1
    assert listbox._seen == 1


def test_select_all_listbox_selects_every_item():
    listbox = _FakeListbox()
    for value in ("a", "b", "c"):
        listbox.insert("end", value)

    result = pipeline_gui.select_all_listbox(listbox)

    assert result == "break"
    assert listbox.curselection() == (0, 1, 2)


def test_shared_reconstruction_tree_shortcuts_extend_and_select_all():
    panel = SharedReconstructionPanel.__new__(SharedReconstructionPanel)
    panel.tree = _FakeTree()
    panel._default_names = []
    panel._selection_callback = None
    panel._refresh_callback = None
    panel._suspend_selection_callback = False
    panel._allow_empty_selection = False
    panel._explicitly_cleared = False
    panel.state = SimpleNamespace(set_shared_reconstruction_selection=lambda _names: None)

    SharedReconstructionPanel.set_rows(
        panel,
        rows=[
            {"name": "recon_a", "label": "A", "family": "ekf_2d", "frames": 12, "reproj_mean": 1.2, "path": "/a"},
            {"name": "recon_b", "label": "B", "family": "ekf_2d", "frames": 10, "reproj_mean": 1.5, "path": "/b"},
            {"name": "recon_c", "label": "C", "family": "ekf_2d", "frames": 8, "reproj_mean": 2.0, "path": "/c"},
        ],
        default_names=["recon_a"],
    )
    panel.tree.selection_set(("recon_a",))
    panel.tree.focus("recon_a")

    from preview.shared_reconstruction_panel import _extend_treeview_selection, _select_all_treeview

    assert _extend_treeview_selection(panel.tree, 1) == "break"
    assert panel.tree.selection() == ("recon_a", "recon_b")
    assert panel.tree.focus() == "recon_b"
    assert _select_all_treeview(panel.tree) == "break"
    assert panel.tree.selection() == ("recon_a", "recon_b", "recon_c")


def test_shared_reconstruction_panel_can_clear_browse_selection_on_same_row_click():
    published = []
    callbacks = []
    panel = SharedReconstructionPanel.__new__(SharedReconstructionPanel)
    panel.tree = _FakeTree()
    panel._default_names = ["recon_a"]
    panel._selection_callback = lambda: callbacks.append("called")
    panel._refresh_callback = None
    panel._suspend_selection_callback = False
    panel._allow_empty_selection = True
    panel._selectmode = "browse"
    panel._explicitly_cleared = False
    panel.state = SimpleNamespace(set_shared_reconstruction_selection=lambda names: published.append(list(names)))

    SharedReconstructionPanel.set_rows(
        panel,
        rows=[
            {"name": "recon_a", "label": "A", "family": "ekf_2d", "frames": 12, "reproj_mean": 1.2, "path": "/a"},
            {"name": "recon_b", "label": "B", "family": "ekf_2d", "frames": 10, "reproj_mean": 1.5, "path": "/b"},
        ],
        default_names=["recon_a"],
    )
    panel.tree.selection_set(("recon_a",))
    panel.tree._identified_row = "recon_a"

    result = SharedReconstructionPanel._on_button_press(panel, SimpleNamespace(y=12))

    assert result == "break"
    assert panel.tree.selection() == ()
    assert panel.selected_names() == []
    assert published[-1] == []
    assert callbacks == ["called"]


def test_shared_reconstruction_panel_can_clear_browse_selection_on_blank_click():
    published = []
    callbacks = []
    panel = SharedReconstructionPanel.__new__(SharedReconstructionPanel)
    panel.tree = _FakeTree()
    panel._default_names = ["recon_a"]
    panel._selection_callback = lambda: callbacks.append("called")
    panel._refresh_callback = None
    panel._suspend_selection_callback = False
    panel._allow_empty_selection = True
    panel._selectmode = "browse"
    panel._explicitly_cleared = False
    panel.state = SimpleNamespace(set_shared_reconstruction_selection=lambda names: published.append(list(names)))

    SharedReconstructionPanel.set_rows(
        panel,
        rows=[
            {"name": "recon_a", "label": "A", "family": "ekf_2d", "frames": 12, "reproj_mean": 1.2, "path": "/a"},
            {"name": "recon_b", "label": "B", "family": "ekf_2d", "frames": 10, "reproj_mean": 1.5, "path": "/b"},
        ],
        default_names=["recon_a"],
    )
    panel.tree.selection_set(("recon_a",))
    panel.tree._identified_row = ""

    result = SharedReconstructionPanel._on_button_press(panel, SimpleNamespace(y=999))

    assert result == "break"
    assert panel.tree.selection() == ()
    assert panel.selected_names() == []
    assert published[-1] == []
    assert callbacks == ["called"]


def test_compose_multiview_crop_points_includes_selected_reprojections():
    base = np.zeros((1, 2, 3, 2), dtype=float)
    reproj = np.ones((1, 2, 3, 2), dtype=float)

    crop_points = pipeline_gui.compose_multiview_crop_points(
        base,
        {"ekf_demo": reproj},
        ["raw", "ekf_demo"],
    )

    assert crop_points.shape == (1, 2, 6, 2)
    np.testing.assert_allclose(crop_points[:, :, :3], base)
    np.testing.assert_allclose(crop_points[:, :, 3:], reproj)


def test_square_crop_bounds_returns_square_window():
    x0, x1, y1, y0 = pipeline_gui.square_crop_bounds(
        xmin=1200.0,
        xmax=1380.0,
        ymin=550.0,
        ymax=800.0,
        width=1920.0,
        height=1080.0,
        margin=0.1,
    )

    assert np.isclose(x1 - x0, y1 - y0)


def test_draw_2d_background_image_uses_pixel_extent():
    ax = _FakeAxis()
    image = np.zeros((1080, 1920, 3), dtype=float)

    pipeline_gui.draw_2d_background_image(ax, image, 1920.0, 1080.0)

    assert len(ax.images) == 1
    _image, kwargs = ax.images[0]
    assert kwargs["extent"] == (0.0, 1920.0, 1080.0, 0.0)
    assert kwargs["origin"] == "upper"


def test_hide_2d_axes_removes_pixel_ticks():
    ax = _FakeAxis()

    pipeline_gui.hide_2d_axes(ax)

    assert ax.xticks == []
    assert ax.yticks == []


def test_camera_layout_is_adaptive_for_small_camera_counts():
    assert pipeline_gui.camera_layout(1) == (1, 1)
    assert pipeline_gui.camera_layout(2) == (1, 2)
    assert pipeline_gui.camera_layout(3) == (2, 2)
    assert pipeline_gui.camera_layout(4) == (2, 2)
    assert pipeline_gui.camera_layout(5) == (2, 3)


def test_annotation_motion_prior_center_extrapolates_velocity():
    center = pipeline_gui.annotation_motion_prior_center(
        np.array([100.0, 80.0], dtype=float),
        np.array([92.0, 74.0], dtype=float),
    )

    np.testing.assert_allclose(center, np.array([108.0, 86.0], dtype=float))


def test_interpolate_short_nan_runs_fills_only_short_interior_gaps():
    values = np.array(
        [
            [0.0, 10.0],
            [np.nan, np.nan],
            [2.0, 14.0],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [5.0, 20.0],
        ],
        dtype=float,
    )

    interpolated = pipeline_gui.interpolate_short_nan_runs(values, max_gap_frames=1)

    np.testing.assert_allclose(interpolated[1], np.array([1.0, 12.0]))
    assert np.all(np.isnan(interpolated[3:5]))


def test_interpolate_short_nan_runs_keeps_edge_nans():
    values = np.array([np.nan, 1.0, np.nan, 3.0, np.nan], dtype=float)

    interpolated = pipeline_gui.interpolate_short_nan_runs(values, max_gap_frames=2)

    assert np.isnan(interpolated[0])
    np.testing.assert_allclose(interpolated[2], 2.0)
    assert np.isnan(interpolated[4])


def test_fill_short_edge_nan_runs_fills_only_short_edges():
    values = np.array([np.nan, np.nan, 2.0, 4.0, np.nan, np.nan, np.nan], dtype=float)

    filled = pipeline_gui.fill_short_edge_nan_runs(values, max_gap_frames=2)

    np.testing.assert_allclose(filled[:2], np.array([2.0, 2.0]))
    np.testing.assert_allclose(filled[2:4], np.array([2.0, 4.0]))
    assert np.all(np.isnan(filled[4:]))


def test_annotation_adjust_image_changes_brightness_and_contrast():
    image = np.array([[[0.25, 0.5, 0.75], [0.4, 0.5, 0.6]]], dtype=float)

    adjusted = pipeline_gui.annotation_adjust_image(image, brightness=1.2, contrast=0.5)

    assert adjusted.shape == image.shape
    assert np.all(np.isfinite(adjusted))
    assert not np.allclose(adjusted, image)


def test_find_annotation_frame_with_images_skips_missing_frames(tmp_path):
    images_root = tmp_path / "images"
    images_root.mkdir(parents=True)
    target_image = images_root / "Camera1_M11139_frame_00005.jpg"
    target_image.write_bytes(b"test")

    frame_idx = pipeline_gui.find_annotation_frame_with_images(
        frames=np.array([3, 4, 5, 6], dtype=int),
        current_index=0,
        direction=1,
        camera_names=["M11139"],
        images_root=images_root,
    )

    assert frame_idx == 2


def test_annotation_tab_set_frame_index_saves_before_frame_change():
    saved = []
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = SimpleNamespace(frames=np.array([10, 11, 12], dtype=int))
    tab._current_frame_idx = 0
    tab.frame_var = SimpleNamespace(set=lambda value: setattr(tab, "_frame_value", value))
    tab.save_annotations = lambda: saved.append("saved")

    pipeline_gui.AnnotationTab._set_frame_index(tab, 2)

    assert saved == ["saved"]
    assert tab._current_frame_idx == 2
    assert tab._frame_value == 2


def test_annotation_click_advances_marker_without_monoview():
    saved = []
    refreshed = []
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = SimpleNamespace(frames=np.array([10, 11, 12], dtype=int))
    tab.calibrations = {"cam0": object()}
    tab.annotation_payload = {}
    tab._axis_to_camera = {"axis0": "cam0"}
    tab.advance_marker_var = SimpleNamespace(get=lambda: True)
    tab._pending_reprojection_points = {}
    tab.kinematic_assist_var = SimpleNamespace(get=lambda: False)
    tab.save_annotations = lambda: saved.append("saved")
    tab.refresh_preview = lambda: refreshed.append("refreshed")
    tab._advance_to_next_keypoint = lambda: setattr(tab, "_advanced", True)
    tab.selected_keypoint_name = lambda: "left_shoulder"
    tab.current_frame_number = lambda: 10
    tab._clear_kinematic_assist_preview = lambda: None
    tab._snap_annotation_xy = lambda **kwargs: np.array([kwargs["pointer_xy"][0], kwargs["pointer_xy"][1]], dtype=float)

    pipeline_gui.AnnotationTab.on_preview_click(
        tab,
        SimpleNamespace(inaxes="axis0", xdata=120.0, ydata=240.0, button=1),
    )
    pipeline_gui.AnnotationTab.on_preview_release(
        tab,
        SimpleNamespace(inaxes="axis0", xdata=120.0, ydata=240.0, button=1),
    )

    assert getattr(tab, "_advanced", False) is True
    assert saved == ["saved"]
    assert refreshed == ["refreshed"]


def test_annotation_refresh_preview_preserves_saved_view_limits(monkeypatch):
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = SimpleNamespace(
        camera_names=["cam0"],
        frames=np.array([10], dtype=int),
    )
    tab.calibrations = {"cam0": SimpleNamespace(image_size=(640, 480))}
    tab.annotation_payload = {}
    tab.preview_figure = Figure(figsize=(4, 4))
    tab.preview_canvas = SimpleNamespace(draw_idle=lambda: None)
    tab.frame_var = SimpleNamespace(get=lambda: 0, set=lambda _value: None)
    tab.frame_label = SimpleNamespace(configure=lambda **_kwargs: None)
    tab.crop_var = SimpleNamespace(get=lambda: False)
    tab.show_images_var = SimpleNamespace(get=lambda: False)
    tab.image_brightness_var = SimpleNamespace(get=lambda: 1.0)
    tab.image_contrast_var = SimpleNamespace(get=lambda: 1.0)
    tab.show_reference_reprojection_var = SimpleNamespace(get=lambda: False)
    tab.show_motion_prior_var = SimpleNamespace(get=lambda: False)
    tab.show_triangulated_hint_var = SimpleNamespace(get=lambda: False)
    tab.motion_prior_diameter = SimpleNamespace(get=lambda: "15")
    tab.show_epipolar_var = SimpleNamespace(get=lambda: False)
    tab.kinematic_assist_var = SimpleNamespace(get=lambda: False)
    tab.kinematic_projected_points = None
    tab.kinematic_segmented_back_projected = {}
    tab._annotation_hover_entries = {}
    tab._cursor_artists = {}
    tab._axis_to_camera = {}
    tab._annotation_view_limits = {"cam0": ((10.0, 20.0), (40.0, 30.0))}
    tab._pending_reprojection_points = {}
    tab.current_frame_number = lambda: 10
    tab.selected_annotation_camera_names = lambda: ["cam0"]
    tab._current_images_root = lambda: None
    tab._frame_filter_mode = lambda: "all"
    tab._filtered_annotation_frame_local_indices = lambda: [0]
    tab.selected_keypoint_name = lambda: "nose"
    tab._reference_projected_points = lambda *_args, **_kwargs: (None, None, "#6c5ce7")
    tab._annotation_xy = lambda *_args, **_kwargs: None

    monkeypatch.setattr(
        pipeline_gui,
        "render_annotation_camera_view",
        lambda ax, **_kwargs: (ax.scatter([], []), [])[1],
    )

    pipeline_gui.AnnotationTab.refresh_preview(tab)

    ax = tab.preview_figure.axes[0]
    np.testing.assert_allclose(ax.get_xlim(), np.array([10.0, 20.0]))
    np.testing.assert_allclose(ax.get_ylim(), np.array([40.0, 30.0]))


def test_annotation_set_frame_index_resets_user_view_limits_on_frame_change():
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = SimpleNamespace(frames=np.array([10, 11], dtype=int))
    tab._current_frame_idx = 0
    tab._annotation_view_limits = {"cam0": ((10.0, 20.0), (40.0, 30.0))}
    tab._reset_annotation_view_limits_on_next_refresh = False
    tab.save_annotations = lambda: setattr(tab, "_saved", True)
    tab._clear_pending_reprojection = lambda: setattr(tab, "_cleared_pending", True)
    tab._clear_kinematic_assist_preview = lambda: setattr(tab, "_cleared_kinematic", True)
    tab.frame_var = SimpleNamespace(set=lambda value: setattr(tab, "_frame_var_value", value))

    pipeline_gui.AnnotationTab._set_frame_index(tab, 1)

    assert tab._annotation_view_limits == {}
    assert tab._reset_annotation_view_limits_on_next_refresh is True
    assert tab._frame_var_value == 1
    assert tab._saved is True
    assert tab._cleared_pending is True
    assert tab._cleared_kinematic is True


def test_annotation_set_frame_index_keeps_user_view_limits_when_frame_unchanged():
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = SimpleNamespace(frames=np.array([10, 11], dtype=int))
    tab._current_frame_idx = 1
    original_limits = {"cam0": ((10.0, 20.0), (40.0, 30.0))}
    tab._annotation_view_limits = dict(original_limits)
    tab._reset_annotation_view_limits_on_next_refresh = False
    tab.save_annotations = lambda: (_ for _ in ()).throw(AssertionError("save_annotations should not be called"))
    tab._clear_pending_reprojection = lambda: (_ for _ in ()).throw(
        AssertionError("_clear_pending_reprojection should not be called")
    )
    tab._clear_kinematic_assist_preview = lambda: (_ for _ in ()).throw(
        AssertionError("_clear_kinematic_assist_preview should not be called")
    )
    tab.frame_var = SimpleNamespace(set=lambda value: setattr(tab, "_frame_var_value", value))

    pipeline_gui.AnnotationTab._set_frame_index(tab, 1)

    assert tab._annotation_view_limits == original_limits
    assert tab._reset_annotation_view_limits_on_next_refresh is False
    assert tab._frame_var_value == 1


def test_annotation_refresh_preview_skips_storing_old_view_limits_after_frame_change(monkeypatch):
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = SimpleNamespace(
        camera_names=["cam0"],
        frames=np.array([10, 11], dtype=int),
    )
    tab.calibrations = {"cam0": SimpleNamespace(image_size=(640, 480))}
    tab.annotation_payload = {}
    tab.preview_figure = Figure(figsize=(4, 4))
    tab.preview_canvas = SimpleNamespace(draw_idle=lambda: None)
    tab.frame_var = SimpleNamespace(get=lambda: 1, set=lambda _value: None)
    tab.frame_label = SimpleNamespace(configure=lambda **_kwargs: None)
    tab.crop_var = SimpleNamespace(get=lambda: False)
    tab.show_images_var = SimpleNamespace(get=lambda: False)
    tab.image_brightness_var = SimpleNamespace(get=lambda: 1.0)
    tab.image_contrast_var = SimpleNamespace(get=lambda: 1.0)
    tab.show_reference_reprojection_var = SimpleNamespace(get=lambda: False)
    tab.show_motion_prior_var = SimpleNamespace(get=lambda: False)
    tab.show_triangulated_hint_var = SimpleNamespace(get=lambda: False)
    tab.motion_prior_diameter = SimpleNamespace(get=lambda: "15")
    tab.show_epipolar_var = SimpleNamespace(get=lambda: False)
    tab.kinematic_assist_var = SimpleNamespace(get=lambda: False)
    tab.kinematic_projected_points = None
    tab.kinematic_segmented_back_projected = {}
    tab._annotation_hover_entries = {}
    tab._cursor_artists = {}
    tab._axis_to_camera = {}
    tab._annotation_view_limits = {"cam0": ((10.0, 20.0), (40.0, 30.0))}
    tab._reset_annotation_view_limits_on_next_refresh = True
    tab._pending_reprojection_points = {}
    tab.current_frame_number = lambda: 11
    tab.selected_annotation_camera_names = lambda: ["cam0"]
    tab._current_images_root = lambda: None
    tab._frame_filter_mode = lambda: "all"
    tab._filtered_annotation_frame_local_indices = lambda: [0, 1]
    tab.selected_keypoint_name = lambda: "nose"
    tab._reference_projected_points = lambda *_args, **_kwargs: (None, None, "#6c5ce7")
    tab._annotation_xy = lambda *_args, **_kwargs: None

    monkeypatch.setattr(
        pipeline_gui,
        "render_annotation_camera_view",
        lambda ax, **_kwargs: (ax.scatter([], []), [])[1],
    )
    monkeypatch.setattr(
        pipeline_gui.AnnotationTab,
        "_store_current_annotation_view_limits",
        lambda _self: (_ for _ in ()).throw(AssertionError("old view limits should not be stored")),
    )

    pipeline_gui.AnnotationTab.refresh_preview(tab)

    assert tab._annotation_view_limits == {}
    assert tab._reset_annotation_view_limits_on_next_refresh is False


def test_annotation_delete_nearest_annotation_removes_closest_marker():
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.annotation_payload = pipeline_gui.empty_annotation_payload()
    pipeline_gui.set_annotation_point(
        tab.annotation_payload,
        camera_name="cam0",
        frame_number=10,
        keypoint_name="left_shoulder",
        xy=[100.0, 200.0],
    )
    pipeline_gui.set_annotation_point(
        tab.annotation_payload,
        camera_name="cam0",
        frame_number=10,
        keypoint_name="right_shoulder",
        xy=[300.0, 400.0],
    )
    tab._annotation_xy = lambda camera_name, frame_number, keypoint_name: (
        np.asarray(
            pipeline_gui.get_annotation_point(
                tab.annotation_payload,
                camera_name=camera_name,
                frame_number=frame_number,
                keypoint_name=keypoint_name,
            )[0]
        )
        if pipeline_gui.get_annotation_point(
            tab.annotation_payload,
            camera_name=camera_name,
            frame_number=frame_number,
            keypoint_name=keypoint_name,
        )[0]
        is not None
        else None
    )

    deleted = pipeline_gui.AnnotationTab._delete_nearest_annotation(
        tab, "cam0", 10, np.array([108.0, 206.0], dtype=float)
    )

    assert deleted is True
    assert (
        pipeline_gui.get_annotation_point(
            tab.annotation_payload,
            camera_name="cam0",
            frame_number=10,
            keypoint_name="left_shoulder",
        )[0]
        is None
    )
    assert (
        pipeline_gui.get_annotation_point(
            tab.annotation_payload,
            camera_name="cam0",
            frame_number=10,
            keypoint_name="right_shoulder",
        )[0]
        is not None
    )


def test_annotation_ctrl_click_deletes_nearest_marker():
    saved = []
    refreshed = []
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = SimpleNamespace(frames=np.array([10, 11, 12], dtype=int))
    tab.calibrations = {"cam0": object()}
    tab.annotation_payload = {}
    tab._axis_to_camera = {"axis0": "cam0"}
    tab.advance_marker_var = SimpleNamespace(get=lambda: True)
    tab._pending_reprojection_points = {}
    tab.kinematic_assist_var = SimpleNamespace(get=lambda: False)
    tab.save_annotations = lambda: saved.append("saved")
    tab.refresh_preview = lambda: refreshed.append("refreshed")
    tab.selected_keypoint_name = lambda: "left_shoulder"
    tab.current_frame_number = lambda: 10
    tab._delete_nearest_annotation = (
        lambda camera_name, frame_number, xy: setattr(tab, "_deleted", (camera_name, frame_number, tuple(xy))) or True
    )

    pipeline_gui.AnnotationTab.on_preview_click(
        tab,
        SimpleNamespace(inaxes="axis0", xdata=120.0, ydata=240.0, button=1, key="control"),
    )

    assert tab._deleted == ("cam0", 10, (120.0, 240.0))
    assert saved == ["saved"]
    assert refreshed == ["refreshed"]


def test_annotation_tab_frame_scale_click_uses_slider_position(monkeypatch):
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.frame_scale = _FakeScale(from_value=0, to_value=20, width=300)
    tab.frame_var = SimpleNamespace(set=lambda value: setattr(tab, "_frame_value", value))
    tab._current_frame_idx = 0
    tab._dragging_frame_scale = False
    tab._set_frame_index = lambda value: setattr(tab, "_frame_index", value)
    tab.refresh_preview = lambda: setattr(tab, "_refreshed", True)

    monkeypatch.setattr(pipeline_gui, "frame_from_slider_click", lambda **_kwargs: 7)

    result = pipeline_gui.AnnotationTab._on_frame_scale_click(tab, SimpleNamespace(x=40))

    assert result == "break"
    assert tab._dragging_frame_scale is True
    assert tab.frame_scale.focused is True
    assert tab._frame_index == 7
    assert tab._refreshed is True


def test_annotation_marker_shape_uses_side_specific_symbols():
    assert pipeline_gui.annotation_marker_shape("left_wrist") == "+"
    assert pipeline_gui.annotation_marker_shape("right_wrist") == "x"
    assert pipeline_gui.annotation_marker_shape("nose") == "+"


def test_annotation_blend_q_by_relevance_updates_only_selected_subtree():
    q_names = ["TRUNK:RotY", "LEFT_UPPER_ARM:RotY", "RIGHT_UPPER_ARM:RotY", "LEFT_THIGH:RotY"]
    previous_q = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    estimated_q = np.array([10.0, 11.0, 12.0, 13.0], dtype=float)

    blended = pipeline_gui.annotation_blend_q_by_relevance(q_names, previous_q, estimated_q, "left_wrist")

    np.testing.assert_allclose(blended, np.array([10.0, 11.0, 2.0, 3.0], dtype=float))


def test_annotation_tab_frame_scale_release_stops_dragging(monkeypatch):
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.frame_scale = _FakeScale(from_value=0, to_value=20, width=300)
    tab._dragging_frame_scale = True
    tab._set_frame_index = lambda value: setattr(tab, "_frame_index", value)
    tab.refresh_preview = lambda: setattr(tab, "_refreshed", True)

    monkeypatch.setattr(pipeline_gui, "frame_from_slider_click", lambda **_kwargs: 9)

    result = pipeline_gui.AnnotationTab._on_frame_scale_release(tab, SimpleNamespace(x=75))

    assert result == "break"
    assert tab._dragging_frame_scale is False
    assert tab._frame_index == 9
    assert tab._refreshed is True


def test_annotation_step_frame_uses_filtered_candidates():
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = SimpleNamespace(frames=np.array([10, 11, 12, 13, 14], dtype=int))
    tab.frame_var = SimpleNamespace(get=lambda: 1)
    tab._set_frame_index = lambda value: setattr(tab, "_frame_index", value)
    tab.refresh_preview = lambda: setattr(tab, "_refreshed", True)
    tab._navigable_annotation_frame_local_indices = lambda: [1, 4]

    pipeline_gui.AnnotationTab.step_frame(tab, 1)

    assert tab._frame_index == 4
    assert tab._refreshed is True


def test_annotation_step_frame_wraps_within_filtered_candidates():
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = SimpleNamespace(frames=np.array([10, 11, 12, 13, 14], dtype=int))
    tab.frame_var = SimpleNamespace(get=lambda: 4)
    tab._set_frame_index = lambda value: setattr(tab, "_frame_index", value)
    tab.refresh_preview = lambda: setattr(tab, "_refreshed", True)
    tab._navigable_annotation_frame_local_indices = lambda: [1, 4]

    pipeline_gui.AnnotationTab.step_frame(tab, 1)

    assert tab._frame_index == 1
    assert tab._refreshed is True


def test_annotation_step_frame_wraps_within_worst_reprojection_set():
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = SimpleNamespace(frames=np.arange(20, dtype=int))
    tab.frame_var = SimpleNamespace(get=lambda: 12)
    tab._set_frame_index = lambda value: setattr(tab, "_frame_index", value)
    tab.refresh_preview = lambda: setattr(tab, "_refreshed", True)
    tab._navigable_annotation_frame_local_indices = lambda: [3, 7, 12]

    pipeline_gui.AnnotationTab.step_frame(tab, 1)

    assert tab._frame_index == 3
    assert tab._refreshed is True


def test_annotation_flip_frame_filter_uses_selected_cameras(monkeypatch, tmp_path):
    pose_data = SimpleNamespace(frames=np.array([10, 11, 12], dtype=int), camera_names=["cam0", "cam1"])
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = pose_data
    tab.calibrations = {"cam0": object(), "cam1": object()}
    tab.state = SimpleNamespace(
        keypoints_var=SimpleNamespace(get=lambda: "inputs/keypoints/trial_keypoints.json"),
        calib_var=SimpleNamespace(get=lambda: "inputs/calibration/Calib.toml"),
        pose_data_mode_var=SimpleNamespace(get=lambda: "cleaned"),
        pose_filter_window_var=SimpleNamespace(get=lambda: "9"),
        pose_outlier_ratio_var=SimpleNamespace(get=lambda: "0.1"),
        pose_p_low_var=SimpleNamespace(get=lambda: "5"),
        pose_p_high_var=SimpleNamespace(get=lambda: "95"),
        calibration_correction_var=SimpleNamespace(get=lambda: "flip_epipolar"),
        flip_improvement_ratio_var=SimpleNamespace(get=lambda: "0.7"),
        flip_min_gain_px_var=SimpleNamespace(get=lambda: "3.0"),
        flip_min_other_cameras_var=SimpleNamespace(get=lambda: "2"),
        flip_restrict_to_outliers_var=SimpleNamespace(get=lambda: True),
        flip_outlier_percentile_var=SimpleNamespace(get=lambda: "85"),
        flip_outlier_floor_px_var=SimpleNamespace(get=lambda: "5"),
        flip_temporal_weight_var=SimpleNamespace(get=lambda: "0.35"),
        flip_temporal_tau_px_var=SimpleNamespace(get=lambda: "20"),
    )
    tab.selected_annotation_camera_names = lambda: ["cam1"]

    monkeypatch.setattr(pipeline_gui, "ROOT", tmp_path)
    monkeypatch.setattr(
        pipeline_gui,
        "get_cached_pose_data",
        lambda *_args, **_kwargs: (tab.calibrations, pose_data),
    )
    monkeypatch.setattr(pipeline_gui, "current_dataset_dir", lambda _state: tmp_path / "output" / "trial")
    monkeypatch.setattr(
        pipeline_gui,
        "load_or_compute_left_right_flip_cache",
        lambda **_kwargs: (
            np.array([[True, False, False], [False, False, True]], dtype=bool),
            {},
            0.0,
            tmp_path / "flip_cache.npz",
            "cache",
        ),
    )

    local_indices = pipeline_gui.AnnotationTab._annotation_flip_frame_local_indices(tab)

    assert local_indices == [2]


def test_annotation_worst_reprojection_filter_uses_selected_reconstruction_and_cameras(monkeypatch, tmp_path):
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = SimpleNamespace(frames=np.arange(20, dtype=int), camera_names=["cam0", "cam1"])
    tab.state = SimpleNamespace(shared_reconstruction_selection=["demo"])
    tab.selected_annotation_camera_names = lambda: ["cam1"]

    errors = np.ones((20, 17, 2), dtype=float)
    errors[3, :, 0] = 50.0
    errors[7, :, 1] = 90.0
    payload = {
        "reprojection_error_per_view": errors,
        "frames": np.arange(20, dtype=int),
        "camera_names": np.array(["cam0", "cam1"], dtype=object),
    }

    monkeypatch.setattr(pipeline_gui, "current_dataset_dir", lambda _state: tmp_path / "output" / "trial")
    monkeypatch.setattr(pipeline_gui, "reconstruction_dir_by_name", lambda *_args, **_kwargs: tmp_path / "recon")
    monkeypatch.setattr(pipeline_gui, "load_bundle_payload", lambda _path: payload)

    local_indices = pipeline_gui.AnnotationTab._annotation_worst_reprojection_frame_local_indices(tab)

    assert local_indices == [7]


def test_annotation_configure_shared_reconstruction_panel_uses_shared_selection():
    configured = {}
    published = {}

    class _FakePanel:
        def configure_for_consumer(self, **kwargs):
            configured.update(kwargs)

        def set_rows(self, rows, defaults):
            published["rows"] = rows
            published["defaults"] = defaults

    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.uses_shared_reconstruction_panel = True
    tab.shared_reconstruction_selectmode = "browse"
    tab.state = SimpleNamespace(
        shared_reconstruction_selection=["demo"],
        shared_reconstruction_panel=_FakePanel(),
        active_reconstruction_consumer=tab,
    )
    tab.on_frame_filter_changed = lambda: published.setdefault("selection_changed", True)
    tab.refresh_available_reconstructions = lambda: published.setdefault("refreshed", True)

    pipeline_gui.AnnotationTab.configure_shared_reconstruction_panel(tab, tab.state.shared_reconstruction_panel)
    configured["selection_callback"]()

    assert configured["title"] == "Reconstructions | Annotation"
    assert configured["selectmode"] == "browse"
    assert published["refreshed"] is True
    assert published["selection_changed"] is True
    assert tab.uses_shared_reconstruction_panel is True


def test_annotation_on_frame_filter_changed_does_not_scan_images():
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = SimpleNamespace(frames=np.array([10, 11, 12], dtype=int))
    tab.frame_var = SimpleNamespace(get=lambda: 2)
    tab._set_frame_index = lambda value: setattr(tab, "_frame_index", value)
    tab.refresh_preview = lambda: setattr(tab, "_refreshed", True)
    tab._filtered_annotation_frame_local_indices = lambda: [0, 1]
    tab._navigable_annotation_frame_local_indices = lambda: (_ for _ in ()).throw(
        AssertionError("image scan should not run during filter change")
    )

    pipeline_gui.AnnotationTab.on_frame_filter_changed(tab)

    assert tab._frame_index == 0
    assert tab._refreshed is True


def test_annotation_navigable_frames_use_indexed_image_availability(monkeypatch, tmp_path):
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = SimpleNamespace(frames=np.array([10, 11, 12, 13], dtype=int))
    tab._filtered_annotation_frame_local_indices = lambda: [0, 1, 2, 3]
    tab.selected_annotation_camera_names = lambda: ["cam0", "cam1"]
    tab._current_images_root = lambda: tmp_path
    tab._navigable_frame_cache = {}

    calls = []
    monkeypatch.setattr(
        frame_navigation,
        "available_execution_image_frames",
        lambda root, camera_names: calls.append((root, tuple(camera_names))) or {"cam0": {11}, "cam1": {13}},
    )
    monkeypatch.setattr(
        pipeline_gui,
        "resolve_execution_image_path",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not resolve image paths here")),
    )

    local_indices = pipeline_gui.AnnotationTab._navigable_annotation_frame_local_indices(tab)

    assert local_indices == [1, 3]
    assert calls == [(tmp_path, ("cam0", "cam1"))]


def test_annotation_crop_limits_use_selected_camera_indices(monkeypatch):
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = SimpleNamespace(
        camera_names=["cam0", "cam1", "cam2"],
        raw_keypoints=np.arange(3 * 2 * 17 * 2, dtype=float).reshape(3, 2, 17, 2),
        keypoints=np.zeros((3, 2, 17, 2), dtype=float),
    )
    tab.calibrations = {
        "cam0": SimpleNamespace(image_size=(100, 100)),
        "cam1": SimpleNamespace(image_size=(100, 100)),
        "cam2": SimpleNamespace(image_size=(100, 100)),
    }
    tab.crop_limits_cache = {}
    tab.crop_limits_key = None

    captured = {}

    def fake_compute_pose_crop_limits_2d(raw_2d, calibrations, camera_names, margin):
        captured["raw_2d"] = np.asarray(raw_2d)
        captured["camera_names"] = list(camera_names)
        captured["margin"] = margin
        return {name: np.zeros((2, 4), dtype=float) for name in camera_names}

    monkeypatch.setattr(pipeline_gui, "compute_pose_crop_limits_2d", fake_compute_pose_crop_limits_2d)

    result = pipeline_gui.AnnotationTab._ensure_crop_limits(tab, ["cam2"])

    assert list(result.keys()) == ["cam2"]
    assert captured["camera_names"] == ["cam2"]
    assert captured["margin"] == 0.2
    np.testing.assert_array_equal(captured["raw_2d"], tab.pose_data.raw_keypoints[[2]])


def test_annotation_triangulated_reprojection_projects_hint_into_target_view():
    calibrations = {
        "cam0": CameraCalibration(
            name="cam0",
            image_size=(1920, 1080),
            K=np.eye(3, dtype=float),
            dist=np.zeros(5, dtype=float),
            rvec=np.zeros(3, dtype=float),
            tvec=np.array([[0.0], [0.0], [0.0]], dtype=float),
            R=np.eye(3, dtype=float),
            P=np.hstack((np.eye(3, dtype=float), np.array([[0.0], [0.0], [0.0]], dtype=float))),
        ),
        "cam1": CameraCalibration(
            name="cam1",
            image_size=(1920, 1080),
            K=np.eye(3, dtype=float),
            dist=np.zeros(5, dtype=float),
            rvec=np.zeros(3, dtype=float),
            tvec=np.array([[1.0], [0.0], [0.0]], dtype=float),
            R=np.eye(3, dtype=float),
            P=np.hstack((np.eye(3, dtype=float), np.array([[1.0], [0.0], [0.0]], dtype=float))),
        ),
        "cam2": CameraCalibration(
            name="cam2",
            image_size=(1920, 1080),
            K=np.eye(3, dtype=float),
            dist=np.zeros(5, dtype=float),
            rvec=np.zeros(3, dtype=float),
            tvec=np.array([[-0.5], [0.0], [0.0]], dtype=float),
            R=np.eye(3, dtype=float),
            P=np.hstack((np.eye(3, dtype=float), np.array([[-0.5], [0.0], [0.0]], dtype=float))),
        ),
    }
    point_3d = np.array([0.2, -0.1, 2.0], dtype=float)
    source_points = [calibrations["cam0"].project_point(point_3d), calibrations["cam1"].project_point(point_3d)]

    hint = pipeline_gui.annotation_triangulated_reprojection(
        calibrations,
        target_camera_name="cam2",
        source_camera_names=["cam0", "cam1"],
        source_points_2d=source_points,
    )

    np.testing.assert_allclose(hint, calibrations["cam2"].project_point(point_3d), atol=1e-8)


def test_annotation_snap_annotation_xy_prefers_nearest_reprojection_candidate():
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.snap_reprojection_var = SimpleNamespace(get=lambda: True)
    tab.snap_epipolar_var = SimpleNamespace(get=lambda: False)
    tab._triangulated_hint_for_keypoint = lambda *_args, **_kwargs: np.array([10.0, 10.0], dtype=float)
    tab._reference_projected_keypoint = lambda *_args, **_kwargs: np.array([30.0, 30.0], dtype=float)
    tab._epipolar_lines_for_keypoint = lambda *_args, **_kwargs: []

    snapped = pipeline_gui.AnnotationTab._snap_annotation_xy(
        tab,
        camera_name="cam0",
        frame_number=10,
        keypoint_name="left_wrist",
        pointer_xy=np.array([12.0, 11.0], dtype=float),
    )

    np.testing.assert_allclose(snapped, np.array([10.0, 10.0], dtype=float))


def test_annotation_snap_annotation_xy_projects_to_epipolar_line():
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.snap_reprojection_var = SimpleNamespace(get=lambda: False)
    tab.snap_epipolar_var = SimpleNamespace(get=lambda: True)
    tab._triangulated_hint_for_keypoint = lambda *_args, **_kwargs: None
    tab._reference_projected_keypoint = lambda *_args, **_kwargs: None
    tab._epipolar_lines_for_keypoint = lambda *_args, **_kwargs: [np.array([1.0, 0.0, -20.0], dtype=float)]

    snapped = pipeline_gui.AnnotationTab._snap_annotation_xy(
        tab,
        camera_name="cam0",
        frame_number=10,
        keypoint_name="left_wrist",
        pointer_xy=np.array([12.0, 15.0], dtype=float),
    )

    np.testing.assert_allclose(snapped, np.array([20.0, 15.0], dtype=float))


def test_triangulate_annotation_frame_points_uses_existing_annotations_only():
    calibrations = {
        "cam0": CameraCalibration(
            name="cam0",
            image_size=(1920, 1080),
            K=np.eye(3, dtype=float),
            dist=np.zeros(5, dtype=float),
            rvec=np.zeros(3, dtype=float),
            tvec=np.array([[0.0], [0.0], [0.0]], dtype=float),
            R=np.eye(3, dtype=float),
            P=np.hstack((np.eye(3, dtype=float), np.array([[0.0], [0.0], [0.0]], dtype=float))),
        ),
        "cam1": CameraCalibration(
            name="cam1",
            image_size=(1920, 1080),
            K=np.eye(3, dtype=float),
            dist=np.zeros(5, dtype=float),
            rvec=np.zeros(3, dtype=float),
            tvec=np.array([[1.0], [0.0], [0.0]], dtype=float),
            R=np.eye(3, dtype=float),
            P=np.hstack((np.eye(3, dtype=float), np.array([[1.0], [0.0], [0.0]], dtype=float))),
        ),
    }
    payload = pipeline_gui.empty_annotation_payload()
    point_3d = np.array([0.2, -0.1, 2.0], dtype=float)
    for camera_name in ("cam0", "cam1"):
        pipeline_gui.set_annotation_point(
            payload,
            camera_name=camera_name,
            frame_number=10,
            keypoint_name="left_shoulder",
            xy=calibrations[camera_name].project_point(point_3d),
        )

    points_3d = pipeline_gui.triangulate_annotation_frame_points(
        calibrations,
        camera_names=["cam0", "cam1"],
        frame_number=10,
        annotation_payload=payload,
    )

    np.testing.assert_allclose(points_3d[pipeline_gui.KP_INDEX["left_shoulder"]], point_3d, atol=1e-8)
    assert np.all(np.isnan(points_3d[pipeline_gui.KP_INDEX["nose"]]))


def test_annotation_step_keypoint_down_updates_selection():
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.annotation_keypoints_list = _FakeListbox()
    for keypoint_name in pipeline_gui.ANNOTATION_KEYPOINT_ORDER[:3]:
        tab.annotation_keypoints_list.insert("end", keypoint_name)
    tab.annotation_keypoints_list.selection_set(0)
    tab.on_keypoint_selection_changed = lambda: setattr(tab, "_selection_changed", True)

    result = pipeline_gui.AnnotationTab._step_annotation_keypoint(tab, 1)

    assert result == "break"
    assert tab.annotation_keypoints_list.curselection() == (1,)
    assert tab.annotation_keypoints_list._active == 1
    assert tab._selection_changed is True


def test_annotation_step_keypoint_up_clamps_at_first_item():
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.annotation_keypoints_list = _FakeListbox()
    for keypoint_name in pipeline_gui.ANNOTATION_KEYPOINT_ORDER[:3]:
        tab.annotation_keypoints_list.insert("end", keypoint_name)
    tab.annotation_keypoints_list.selection_set(0)
    tab.on_keypoint_selection_changed = lambda: setattr(tab, "_selection_changed", True)

    result = pipeline_gui.AnnotationTab._step_annotation_keypoint(tab, -1)

    assert result == "break"
    assert tab.annotation_keypoints_list.curselection() == (2,)
    assert tab.annotation_keypoints_list._active == 2
    assert tab._selection_changed is True


def test_annotation_advance_to_next_keypoint_wraps_to_start():
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.annotation_keypoints_list = _FakeListbox()
    for keypoint_name in pipeline_gui.ANNOTATION_KEYPOINT_ORDER[:3]:
        tab.annotation_keypoints_list.insert("end", keypoint_name)
    tab.annotation_keypoints_list.selection_set(2)
    tab.advance_marker_var = SimpleNamespace(get=lambda: True)
    tab.on_keypoint_selection_changed = lambda: setattr(tab, "_selection_changed", True)

    pipeline_gui.AnnotationTab._advance_to_next_keypoint(tab)

    assert tab.annotation_keypoints_list.curselection() == (0,)
    assert tab.annotation_keypoints_list._active == 0
    assert tab._selection_changed is True


def test_annotation_estimate_kinematic_assist_state_sets_projected_overlay(monkeypatch, tmp_path):
    biomod_path = tmp_path / "demo.bioMod"
    biomod_path.write_text("demo", encoding="utf-8")

    class _FakeBiorbdModel:
        def __init__(self, _path):
            pass

        def nbQ(self):
            return 2

    import sys

    monkeypatch.setitem(sys.modules, "biorbd", SimpleNamespace(Model=_FakeBiorbdModel))

    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = SimpleNamespace(frames=np.array([10], dtype=int))
    tab.calibrations = {"cam0": object(), "cam1": object()}
    tab.annotation_payload = {}
    tab.kinematic_model_choices = {"demo": biomod_path}
    tab.kinematic_model_var = SimpleNamespace(get=lambda: "demo")
    tab.kinematic_frame_states = {}
    tab.selected_annotation_camera_names = lambda: ["cam0", "cam1"]
    tab.current_frame_number = lambda: 10
    tab.kinematic_status_var = SimpleNamespace(set=lambda value: setattr(tab, "_kin_status", value))
    tab.refresh_preview = lambda: setattr(tab, "_refreshed", True)

    monkeypatch.setattr(
        pipeline_gui,
        "triangulate_annotation_frame_points",
        lambda *_args, **_kwargs: np.pad(np.ones((2, 3), dtype=float), ((0, 15), (0, 0)), constant_values=np.nan),
    )
    monkeypatch.setattr(
        pipeline_gui,
        "initial_state_from_triangulation",
        lambda model, reconstruction: np.array([0.1, -0.2, 0.0, 0.0, 0.0, 0.0], dtype=float),
    )
    monkeypatch.setattr(pipeline_gui, "canonicalize_model_q_rotation_branches", lambda _model, q: np.asarray(q))
    monkeypatch.setattr(pipeline_gui, "biorbd_q_names", lambda _model: ["q0", "q1"])
    monkeypatch.setattr(
        pipeline_gui,
        "biorbd_markers_from_q",
        lambda _biomod_path, q_series: np.zeros((q_series.shape[0], len(pipeline_gui.COCO17), 3), dtype=float),
    )
    monkeypatch.setattr(
        pipeline_gui,
        "project_points_all_cameras",
        lambda points_3d, calibrations, camera_names: np.zeros(
            (len(camera_names), points_3d.shape[0], len(pipeline_gui.COCO17), 2), dtype=float
        ),
    )
    monkeypatch.setattr(pipeline_gui, "segmented_back_overlay_from_q", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        pipeline_gui.messagebox,
        "showerror",
        lambda *_args, **_kwargs: pytest.fail("estimate_kinematic_assist_state unexpectedly called showerror"),
    )

    pipeline_gui.AnnotationTab.estimate_kinematic_assist_state(tab)

    np.testing.assert_allclose(tab.kinematic_q_current, np.array([0.1, -0.2], dtype=float))
    assert tab.kinematic_projected_points.shape == (2, 1, len(pipeline_gui.COCO17), 2)
    assert tab._kin_status == "Estimated q from triangulation (2 markers) | local ekf fallback."
    assert tab._refreshed is True


def test_annotation_pose_data_for_frame_uses_only_existing_annotations():
    base_pose_data = pipeline_gui.PoseData(
        camera_names=["cam0", "cam1"],
        frames=np.array([10, 11], dtype=int),
        keypoints=np.full((2, 2, len(pipeline_gui.COCO17), 2), np.nan, dtype=float),
        scores=np.zeros((2, 2, len(pipeline_gui.COCO17)), dtype=float),
    )
    payload = pipeline_gui.empty_annotation_payload()
    pipeline_gui.set_annotation_point(
        payload, camera_name="cam0", frame_number=11, keypoint_name="nose", xy=[10.0, 20.0]
    )
    pipeline_gui.set_annotation_point(
        payload, camera_name="cam1", frame_number=11, keypoint_name="left_shoulder", xy=[30.0, 40.0]
    )

    pose_data = pipeline_gui.annotation_pose_data_for_frame(
        base_pose_data,
        camera_names=["cam0", "cam1"],
        frame_number=11,
        annotation_payload=payload,
    )

    assert pose_data.frames.tolist() == [11]
    np.testing.assert_allclose(pose_data.keypoints[0, 0, pipeline_gui.KP_INDEX["nose"]], np.array([10.0, 20.0]))
    np.testing.assert_allclose(
        pose_data.keypoints[1, 0, pipeline_gui.KP_INDEX["left_shoulder"]], np.array([30.0, 40.0])
    )
    assert pose_data.scores[0, 0, pipeline_gui.KP_INDEX["nose"]] == 1.0
    assert np.isnan(pose_data.keypoints[0, 0, pipeline_gui.KP_INDEX["left_shoulder"]]).all()


def test_annotation_reconstruction_from_points_uses_framewise_coherence():
    points_3d = np.full((len(pipeline_gui.COCO17), 3), np.nan, dtype=float)
    points_3d[pipeline_gui.KP_INDEX["nose"]] = np.array([1.0, 2.0, 3.0], dtype=float)

    reconstruction = pipeline_gui.annotation_reconstruction_from_points(points_3d, frame_number=10, n_cameras=2)

    assert reconstruction.coherence_method == "epipolar_fast_framewise"
    assert reconstruction.frames.tolist() == [10]
    assert reconstruction.points_3d.shape == (1, len(pipeline_gui.COCO17), 3)


def test_annotation_hover_label_includes_source_and_current_marker():
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.selected_keypoint_name = lambda: "left_wrist"

    label = pipeline_gui.AnnotationTab._annotation_hover_label(
        tab,
        {"keypoint_name": "left_wrist", "source": "annotated"},
    )

    assert label == "left_wrist | annotated | current"


def test_annotation_nearest_hover_entry_returns_closest_point():
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    ax = object()
    tab._annotation_hover_entries = {
        ax: [
            {"xy": np.array([10.0, 20.0]), "keypoint_name": "nose", "source": "annotated"},
            {"xy": np.array([30.0, 40.0]), "keypoint_name": "left_wrist", "source": "model reproj"},
        ]
    }

    nearest = pipeline_gui.AnnotationTab._nearest_annotation_hover_entry(tab, ax, 31.0, 39.0)

    assert nearest is not None
    assert nearest["keypoint_name"] == "left_wrist"


def test_annotation_nearest_hover_entry_uses_small_hover_radius():
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    ax = object()
    tab._annotation_hover_entries = {
        ax: [
            {"xy": np.array([10.0, 20.0]), "keypoint_name": "nose", "source": "annotated"},
        ]
    }

    nearest = pipeline_gui.AnnotationTab._nearest_annotation_hover_entry(tab, ax, 25.0, 20.0)

    assert nearest is None


def test_annotation_estimate_kinematic_q_with_keypoint_reports_local_update(monkeypatch, tmp_path):
    biomod_path = tmp_path / "demo.bioMod"
    biomod_path.write_text("demo", encoding="utf-8")

    class _FakeBiorbdModel:
        def __init__(self, _path):
            pass

        def nbQ(self):
            return 2

    import sys

    monkeypatch.setitem(sys.modules, "biorbd", SimpleNamespace(Model=_FakeBiorbdModel))

    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = SimpleNamespace(frames=np.array([10], dtype=int))
    tab.calibrations = {"cam0": object(), "cam1": object()}
    tab.annotation_payload = {}
    tab.kinematic_model_choices = {"demo": biomod_path}
    tab.kinematic_model_var = SimpleNamespace(get=lambda: "demo")
    tab.kinematic_frame_states = {("demo", 10): np.array([1.0, 2.0], dtype=float)}
    tab.selected_annotation_camera_names = lambda: ["cam0", "cam1"]
    tab.current_frame_number = lambda: 10
    tab.kinematic_status_var = SimpleNamespace(set=lambda value: setattr(tab, "_kin_status", value))

    monkeypatch.setattr(
        pipeline_gui,
        "triangulate_annotation_frame_points",
        lambda *_args, **_kwargs: pytest.fail("triangulation should not be used when a current state exists"),
    )
    monkeypatch.setattr(pipeline_gui, "canonicalize_model_q_rotation_branches", lambda _model, q: np.asarray(q))
    monkeypatch.setattr(pipeline_gui, "biorbd_q_names", lambda _model: ["TRUNK:RotY", "LEFT_LOWER_ARM:RotY"])
    monkeypatch.setattr(
        pipeline_gui,
        "biorbd_markers_from_q",
        lambda _biomod_path, q_series: np.zeros((q_series.shape[0], len(pipeline_gui.COCO17), 3), dtype=float),
    )
    monkeypatch.setattr(
        pipeline_gui,
        "project_points_all_cameras",
        lambda points_3d, calibrations, camera_names: np.zeros(
            (len(camera_names), points_3d.shape[0], len(pipeline_gui.COCO17), 2), dtype=float
        ),
    )
    monkeypatch.setattr(pipeline_gui, "segmented_back_overlay_from_q", lambda *_args, **_kwargs: None)

    estimated_q = pipeline_gui.AnnotationTab._estimate_kinematic_q(tab, keypoint_name="left_wrist")

    np.testing.assert_allclose(estimated_q, np.array([1.0, 2.0], dtype=float))
    assert (
        tab._kin_status == "Updated model after left_wrist using current state from current frame | local ekf fallback."
    )


def test_annotation_estimate_kinematic_q_runs_bootstrap_when_measurements_are_available(monkeypatch, tmp_path):
    biomod_path = tmp_path / "demo.bioMod"
    biomod_path.write_text("demo", encoding="utf-8")

    class _FakeBiorbdModel:
        def __init__(self, _path):
            pass

        def nbQ(self):
            return 2

    import sys

    monkeypatch.setitem(sys.modules, "biorbd", SimpleNamespace(Model=_FakeBiorbdModel))

    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = pipeline_gui.PoseData(
        camera_names=["cam0", "cam1"],
        frames=np.array([10], dtype=int),
        keypoints=np.full((2, 1, len(pipeline_gui.COCO17), 2), np.nan, dtype=float),
        scores=np.zeros((2, 1, len(pipeline_gui.COCO17)), dtype=float),
    )
    tab.calibrations = {"cam0": object(), "cam1": object()}
    tab.annotation_payload = pipeline_gui.empty_annotation_payload()
    pipeline_gui.set_annotation_point(
        tab.annotation_payload, camera_name="cam0", frame_number=10, keypoint_name="nose", xy=[1.0, 2.0]
    )
    pipeline_gui.set_annotation_point(
        tab.annotation_payload, camera_name="cam1", frame_number=10, keypoint_name="nose", xy=[2.0, 3.0]
    )
    pipeline_gui.set_annotation_point(
        tab.annotation_payload, camera_name="cam0", frame_number=10, keypoint_name="left_shoulder", xy=[4.0, 5.0]
    )
    pipeline_gui.set_annotation_point(
        tab.annotation_payload, camera_name="cam1", frame_number=10, keypoint_name="left_shoulder", xy=[5.0, 6.0]
    )
    tab.kinematic_model_choices = {"demo": biomod_path}
    tab.kinematic_model_var = SimpleNamespace(get=lambda: "demo")
    tab.kinematic_frame_states = {}
    tab.selected_annotation_camera_names = lambda: ["cam0", "cam1"]
    tab.current_frame_number = lambda: 10
    tab.kinematic_status_var = SimpleNamespace(set=lambda value: setattr(tab, "_kin_status", value))
    tab.state = SimpleNamespace(fps_var=SimpleNamespace(get=lambda: "120"))

    monkeypatch.setattr(
        pipeline_gui,
        "triangulate_annotation_frame_points",
        lambda *_args, **_kwargs: np.pad(np.ones((2, 3), dtype=float), ((0, 15), (0, 0)), constant_values=np.nan),
    )
    monkeypatch.setattr(
        pipeline_gui,
        "initial_state_from_triangulation",
        lambda model, reconstruction: np.array([0.1, -0.2, 0.0, 0.0, 0.0, 0.0], dtype=float),
    )
    monkeypatch.setattr(pipeline_gui, "canonicalize_model_q_rotation_branches", lambda _model, q: np.asarray(q))
    monkeypatch.setattr(pipeline_gui, "biorbd_q_names", lambda _model: ["q0", "q1"])
    monkeypatch.setattr(
        pipeline_gui,
        "biorbd_markers_from_q",
        lambda _biomod_path, q_series: np.zeros((q_series.shape[0], len(pipeline_gui.COCO17), 3), dtype=float),
    )
    monkeypatch.setattr(
        pipeline_gui,
        "project_points_all_cameras",
        lambda points_3d, calibrations, camera_names: np.zeros(
            (len(camera_names), points_3d.shape[0], len(pipeline_gui.COCO17), 2), dtype=float
        ),
    )
    monkeypatch.setattr(pipeline_gui, "segmented_back_overlay_from_q", lambda *_args, **_kwargs: None)
    ekf_calls = {}

    def fake_refine_local_ekf(**kwargs):
        ekf_calls.update(kwargs)
        return np.array([0.4, -0.5], dtype=float), {"completed_passes": 3, "used_fallback": False}

    monkeypatch.setattr(pipeline_gui, "refine_annotation_q_with_local_ekf", fake_refine_local_ekf)

    estimated_q = pipeline_gui.AnnotationTab._estimate_kinematic_q(tab)

    np.testing.assert_allclose(estimated_q, np.array([0.4, -0.5], dtype=float))
    assert ekf_calls["passes"] == (
        pipeline_gui.ANNOTATION_KINEMATIC_BOOTSTRAP_PASSES
        * pipeline_gui.ANNOTATION_KINEMATIC_INITIAL_BOOTSTRAP_MULTIPLIER
    )
    assert ekf_calls["pose_data"].frames.tolist() == [10]
    assert "local ekf 3/" in tab._kin_status


def test_annotation_estimate_kinematic_q_click_zeroes_derivatives_and_uses_short_passes(monkeypatch, tmp_path):
    biomod_path = tmp_path / "demo.bioMod"
    biomod_path.write_text("demo", encoding="utf-8")

    class _FakeBiorbdModel:
        def __init__(self, _path):
            pass

        def nbQ(self):
            return 2

    import sys

    monkeypatch.setitem(sys.modules, "biorbd", SimpleNamespace(Model=_FakeBiorbdModel))

    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = pipeline_gui.PoseData(
        camera_names=["cam0", "cam1"],
        frames=np.array([10], dtype=int),
        keypoints=np.full((2, 1, len(pipeline_gui.COCO17), 2), np.nan, dtype=float),
        scores=np.zeros((2, 1, len(pipeline_gui.COCO17)), dtype=float),
    )
    tab.calibrations = {"cam0": object(), "cam1": object()}
    tab.annotation_payload = pipeline_gui.empty_annotation_payload()
    pipeline_gui.set_annotation_point(
        tab.annotation_payload, camera_name="cam0", frame_number=10, keypoint_name="left_wrist", xy=[1.0, 2.0]
    )
    pipeline_gui.set_annotation_point(
        tab.annotation_payload, camera_name="cam1", frame_number=10, keypoint_name="left_wrist", xy=[2.0, 3.0]
    )
    tab.kinematic_model_choices = {"demo": biomod_path}
    tab.kinematic_model_var = SimpleNamespace(get=lambda: "demo")
    tab.kinematic_frame_states = {("demo", 10): np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)}
    tab.selected_annotation_camera_names = lambda: ["cam0", "cam1"]
    tab.current_frame_number = lambda: 10
    tab.kinematic_status_var = SimpleNamespace(set=lambda value: setattr(tab, "_kin_status", value))
    tab.state = SimpleNamespace(fps_var=SimpleNamespace(get=lambda: "120"))

    monkeypatch.setattr(pipeline_gui, "canonicalize_model_q_rotation_branches", lambda _model, q: np.asarray(q))
    monkeypatch.setattr(pipeline_gui, "biorbd_q_names", lambda _model: ["TRUNK:RotY", "LEFT_LOWER_ARM:RotY"])
    monkeypatch.setattr(
        pipeline_gui,
        "biorbd_markers_from_q",
        lambda _biomod_path, q_series: np.zeros((q_series.shape[0], len(pipeline_gui.COCO17), 3), dtype=float),
    )
    monkeypatch.setattr(
        pipeline_gui,
        "project_points_all_cameras",
        lambda points_3d, calibrations, camera_names: np.zeros(
            (len(camera_names), points_3d.shape[0], len(pipeline_gui.COCO17), 2), dtype=float
        ),
    )
    monkeypatch.setattr(pipeline_gui, "segmented_back_overlay_from_q", lambda *_args, **_kwargs: None)

    ekf_calls = {}

    def fake_refine_local_ekf(**kwargs):
        ekf_calls.update(kwargs)
        return np.array([7.0, 8.0, 0.1, 0.2, 0.3, 0.4], dtype=float), {"completed_passes": 2, "used_fallback": False}

    monkeypatch.setattr(pipeline_gui, "refine_annotation_q_with_local_ekf", fake_refine_local_ekf)

    pipeline_gui.AnnotationTab._estimate_kinematic_q(tab, keypoint_name="left_wrist")

    np.testing.assert_allclose(ekf_calls["seed_state"][2:], np.zeros(4, dtype=float))
    assert ekf_calls["passes"] == pipeline_gui.ANNOTATION_KINEMATIC_CLICK_PASSES
    assert ekf_calls["keypoint_name"] is None


def test_annotation_click_release_places_selected_marker_without_drag(monkeypatch):
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = SimpleNamespace(frames=np.array([10], dtype=int))
    tab.annotation_payload = pipeline_gui.empty_annotation_payload()
    tab._pending_reprojection_points = {}
    tab._axis_to_camera = {}
    tab._drag_annotation_state = None
    tab._clear_kinematic_assist_preview = lambda: None
    tab.refresh_preview = lambda: setattr(tab, "_refreshed", True)
    tab.save_annotations = lambda: setattr(tab, "_saved", True)
    tab.current_frame_number = lambda: 10
    tab.selected_keypoint_name = lambda: "nose"
    tab.advance_marker_var = SimpleNamespace(get=lambda: False)
    tab.kinematic_assist_var = SimpleNamespace(get=lambda: False)
    tab._snap_annotation_xy = lambda **kwargs: np.array([kwargs["pointer_xy"][0], kwargs["pointer_xy"][1]], dtype=float)
    ax = object()
    tab._axis_to_camera[ax] = "cam0"
    tab._nearest_annotated_drag_entry = lambda *_args, **_kwargs: {
        "xy": np.array([100.0, 100.0]),
        "keypoint_name": "left_eye",
        "source": "annotated",
    }

    press_event = SimpleNamespace(inaxes=ax, xdata=120.0, ydata=80.0, button=1, key="")
    release_event = SimpleNamespace(inaxes=ax, xdata=120.0, ydata=80.0, button=1, key="")

    pipeline_gui.AnnotationTab.on_preview_click(tab, press_event)
    pipeline_gui.AnnotationTab.on_preview_release(tab, release_event)

    point_xy, _score = pipeline_gui.get_annotation_point(
        tab.annotation_payload,
        camera_name="cam0",
        frame_number=10,
        keypoint_name="nose",
    )
    np.testing.assert_allclose(point_xy, np.array([120.0, 80.0], dtype=float))
    left_eye_xy, _score = pipeline_gui.get_annotation_point(
        tab.annotation_payload,
        camera_name="cam0",
        frame_number=10,
        keypoint_name="left_eye",
    )
    assert left_eye_xy is None


def test_annotation_drag_moves_existing_marker_only_after_motion(monkeypatch):
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = SimpleNamespace(frames=np.array([10], dtype=int))
    tab.annotation_payload = pipeline_gui.empty_annotation_payload()
    pipeline_gui.set_annotation_point(
        tab.annotation_payload, camera_name="cam0", frame_number=10, keypoint_name="left_eye", xy=[100.0, 100.0]
    )
    tab._pending_reprojection_points = {}
    tab._axis_to_camera = {}
    tab._drag_annotation_state = None
    tab._clear_kinematic_assist_preview = lambda: None
    tab.refresh_preview = lambda: None
    tab.save_annotations = lambda: None
    tab.current_frame_number = lambda: 10
    tab.selected_keypoint_name = lambda: "nose"
    tab.advance_marker_var = SimpleNamespace(get=lambda: False)
    tab.kinematic_assist_var = SimpleNamespace(get=lambda: False)
    tab._snap_annotation_xy = lambda **kwargs: np.array([kwargs["pointer_xy"][0], kwargs["pointer_xy"][1]], dtype=float)
    tab._update_preview_cursor = lambda _event: None
    ax = object()
    tab._axis_to_camera[ax] = "cam0"
    tab._nearest_annotated_drag_entry = lambda *_args, **_kwargs: {
        "xy": np.array([100.0, 100.0]),
        "keypoint_name": "left_eye",
        "source": "annotated",
    }

    press_event = SimpleNamespace(inaxes=ax, xdata=100.0, ydata=100.0, button=1, key="")
    move_event = SimpleNamespace(inaxes=ax, xdata=108.0, ydata=103.0, button=1, key="")
    release_event = SimpleNamespace(inaxes=ax, xdata=108.0, ydata=103.0, button=1, key="")

    pipeline_gui.AnnotationTab.on_preview_click(tab, press_event)
    pipeline_gui.AnnotationTab.on_preview_motion(tab, move_event)
    pipeline_gui.AnnotationTab.on_preview_release(tab, release_event)

    left_eye_xy, _score = pipeline_gui.get_annotation_point(
        tab.annotation_payload,
        camera_name="cam0",
        frame_number=10,
        keypoint_name="left_eye",
    )
    np.testing.assert_allclose(left_eye_xy, np.array([108.0, 103.0], dtype=float))
    nose_xy, _score = pipeline_gui.get_annotation_point(
        tab.annotation_payload,
        camera_name="cam0",
        frame_number=10,
        keypoint_name="nose",
    )
    assert nose_xy is None


def test_annotation_estimate_kinematic_q_click_uses_single_measurement_when_state_exists(monkeypatch, tmp_path):
    biomod_path = tmp_path / "demo.bioMod"
    biomod_path.write_text("demo", encoding="utf-8")

    class _FakeBiorbdModel:
        def __init__(self, _path):
            pass

        def nbQ(self):
            return 2

    import sys

    monkeypatch.setitem(sys.modules, "biorbd", SimpleNamespace(Model=_FakeBiorbdModel))

    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = pipeline_gui.PoseData(
        camera_names=["cam0"],
        frames=np.array([10], dtype=int),
        keypoints=np.full((1, 1, len(pipeline_gui.COCO17), 2), np.nan, dtype=float),
        scores=np.zeros((1, 1, len(pipeline_gui.COCO17)), dtype=float),
    )
    tab.calibrations = {"cam0": object()}
    tab.annotation_payload = pipeline_gui.empty_annotation_payload()
    pipeline_gui.set_annotation_point(
        tab.annotation_payload, camera_name="cam0", frame_number=10, keypoint_name="left_wrist", xy=[1.0, 2.0]
    )
    tab.kinematic_model_choices = {"demo": biomod_path}
    tab.kinematic_model_var = SimpleNamespace(get=lambda: "demo")
    tab.kinematic_frame_states = {("demo", 10): np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0], dtype=float)}
    tab.selected_annotation_camera_names = lambda: ["cam0"]
    tab.current_frame_number = lambda: 10
    tab.kinematic_status_var = SimpleNamespace(set=lambda value: setattr(tab, "_kin_status", value))
    tab.state = SimpleNamespace(fps_var=SimpleNamespace(get=lambda: "120"))

    monkeypatch.setattr(pipeline_gui, "canonicalize_model_q_rotation_branches", lambda _model, q: np.asarray(q))
    monkeypatch.setattr(pipeline_gui, "biorbd_q_names", lambda _model: ["TRUNK:RotY", "LEFT_LOWER_ARM:RotY"])
    monkeypatch.setattr(
        pipeline_gui,
        "biorbd_markers_from_q",
        lambda _biomod_path, q_series: np.zeros((q_series.shape[0], len(pipeline_gui.COCO17), 3), dtype=float),
    )
    monkeypatch.setattr(
        pipeline_gui,
        "project_points_all_cameras",
        lambda points_3d, calibrations, camera_names: np.zeros(
            (len(camera_names), points_3d.shape[0], len(pipeline_gui.COCO17), 2), dtype=float
        ),
    )
    monkeypatch.setattr(pipeline_gui, "segmented_back_overlay_from_q", lambda *_args, **_kwargs: None)

    ekf_calls = {}

    def fake_refine_local_ekf(**kwargs):
        ekf_calls.update(kwargs)
        return np.array([1.5, 2.5, 0.1, 0.2, 0.3, 0.4], dtype=float), {"completed_passes": 1, "used_fallback": False}

    monkeypatch.setattr(pipeline_gui, "refine_annotation_q_with_local_ekf", fake_refine_local_ekf)

    estimated_q = pipeline_gui.AnnotationTab._estimate_kinematic_q(tab, keypoint_name="left_wrist")

    np.testing.assert_allclose(estimated_q, np.array([1.5, 2.5], dtype=float))
    assert ekf_calls["pose_data"].scores.sum() == 1.0
    assert "local ekf 1/" in tab._kin_status


def test_annotation_estimate_kinematic_q_click_uses_direct_fit_when_local_ekf_is_insufficient(monkeypatch, tmp_path):
    biomod_path = tmp_path / "demo.bioMod"
    biomod_path.write_text("demo", encoding="utf-8")

    class _FakeBiorbdModel:
        def __init__(self, _path):
            pass

        def nbQ(self):
            return 2

    import sys

    monkeypatch.setitem(sys.modules, "biorbd", SimpleNamespace(Model=_FakeBiorbdModel))

    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = pipeline_gui.PoseData(
        camera_names=["cam0"],
        frames=np.array([10], dtype=int),
        keypoints=np.full((1, 1, len(pipeline_gui.COCO17), 2), np.nan, dtype=float),
        scores=np.zeros((1, 1, len(pipeline_gui.COCO17)), dtype=float),
    )
    tab.calibrations = {"cam0": object()}
    tab.annotation_payload = pipeline_gui.empty_annotation_payload()
    pipeline_gui.set_annotation_point(
        tab.annotation_payload, camera_name="cam0", frame_number=10, keypoint_name="left_wrist", xy=[1.0, 2.0]
    )
    tab.kinematic_model_choices = {"demo": biomod_path}
    tab.kinematic_model_var = SimpleNamespace(get=lambda: "demo")
    tab.kinematic_frame_states = {("demo", 10): np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0], dtype=float)}
    tab.selected_annotation_camera_names = lambda: ["cam0"]
    tab.current_frame_number = lambda: 10
    tab.kinematic_status_var = SimpleNamespace(set=lambda value: setattr(tab, "_kin_status", value))
    tab.state = SimpleNamespace(fps_var=SimpleNamespace(get=lambda: "120"))

    monkeypatch.setattr(pipeline_gui, "canonicalize_model_q_rotation_branches", lambda _model, q: np.asarray(q))
    monkeypatch.setattr(pipeline_gui, "biorbd_q_names", lambda _model: ["TRUNK:RotY", "LEFT_LOWER_ARM:RotY"])
    monkeypatch.setattr(
        pipeline_gui,
        "biorbd_markers_from_q",
        lambda _biomod_path, q_series: np.zeros((q_series.shape[0], len(pipeline_gui.COCO17), 3), dtype=float),
    )
    monkeypatch.setattr(
        pipeline_gui,
        "project_points_all_cameras",
        lambda points_3d, calibrations, camera_names: np.zeros(
            (len(camera_names), points_3d.shape[0], len(pipeline_gui.COCO17), 2), dtype=float
        ),
    )
    monkeypatch.setattr(pipeline_gui, "segmented_back_overlay_from_q", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        pipeline_gui,
        "refine_annotation_q_with_local_ekf",
        lambda **kwargs: (
            np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
            {"completed_passes": 3, "used_fallback": True},
        ),
    )

    direct_calls = {}

    def fake_direct_fit(**kwargs):
        direct_calls.update(kwargs)
        return np.array([3.0, 4.0, 0.5, 0.6, 0.7, 0.8], dtype=float), {"completed_passes": 1, "used_fallback": False}

    monkeypatch.setattr(pipeline_gui, "refine_annotation_q_with_direct_measurements", fake_direct_fit)

    estimated_q = pipeline_gui.AnnotationTab._estimate_kinematic_q(tab, keypoint_name="left_wrist")

    np.testing.assert_allclose(estimated_q, np.array([3.0, 4.0], dtype=float))
    assert direct_calls["passes"] == pipeline_gui.ANNOTATION_KINEMATIC_CLICK_DIRECT_PASSES
    assert "direct fit 1/" in tab._kin_status


def test_annotation_step_frame_propagates_state_with_frame_delta(monkeypatch, tmp_path):
    biomod_path = tmp_path / "demo.bioMod"
    biomod_path.write_text("demo", encoding="utf-8")

    class _FakeBiorbdModel:
        def __init__(self, _path):
            pass

        def nbQ(self):
            return 1

    import sys

    monkeypatch.setitem(sys.modules, "biorbd", SimpleNamespace(Model=_FakeBiorbdModel))
    monkeypatch.setattr(pipeline_gui, "canonicalize_model_q_rotation_branches", lambda _model, q: np.asarray(q))

    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = SimpleNamespace(frames=np.array([10, 12, 15], dtype=int))
    tab.frame_var = SimpleNamespace(get=lambda: 0, set=lambda value: setattr(tab, "_frame_index", value))
    tab._current_frame_idx = 0
    tab.kinematic_assist_var = SimpleNamespace(get=lambda: True)
    tab.kinematic_model_choices = {"demo": biomod_path}
    tab.kinematic_model_var = SimpleNamespace(get=lambda: "demo")
    tab.kinematic_frame_states = {("demo", 10): np.array([1.0, 2.0, 3.0], dtype=float)}
    tab.state = SimpleNamespace(fps_var=SimpleNamespace(get=lambda: "10"))
    tab._selected_kinematic_biomod_path = lambda: biomod_path
    tab._navigable_annotation_frame_local_indices = lambda: [2]
    tab.selected_annotation_camera_names = lambda: []
    tab._current_images_root = lambda: None
    tab.save_annotations = lambda: None
    tab._clear_pending_reprojection = lambda: None
    tab._clear_kinematic_assist_preview = lambda: None
    tab.refresh_preview = lambda: setattr(tab, "_refreshed", True)

    pipeline_gui.AnnotationTab.step_frame(tab, 1)

    assert tab._frame_index == 2
    np.testing.assert_allclose(tab.kinematic_frame_states[("demo", 15)], np.array([2.375, 3.5, 3.0]))
    assert tab._refreshed is True


def test_annotation_step_frame_runs_local_estimate_when_target_frame_has_annotations(monkeypatch, tmp_path):
    biomod_path = tmp_path / "demo.bioMod"
    biomod_path.write_text("demo", encoding="utf-8")

    class _FakeBiorbdModel:
        def __init__(self, _path):
            pass

        def nbQ(self):
            return 1

    import sys

    monkeypatch.setitem(sys.modules, "biorbd", SimpleNamespace(Model=_FakeBiorbdModel))
    monkeypatch.setattr(pipeline_gui, "canonicalize_model_q_rotation_branches", lambda _model, q: np.asarray(q))

    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = SimpleNamespace(frames=np.array([10, 12, 15], dtype=int))
    tab.frame_var = SimpleNamespace(get=lambda: 0, set=lambda value: setattr(tab, "_frame_index", value))
    tab._current_frame_idx = 0
    tab.kinematic_assist_var = SimpleNamespace(get=lambda: True)
    tab.kinematic_model_choices = {"demo": biomod_path}
    tab.kinematic_model_var = SimpleNamespace(get=lambda: "demo")
    tab.kinematic_frame_states = {("demo", 10): np.array([1.0, 2.0, 3.0], dtype=float)}
    tab.state = SimpleNamespace(fps_var=SimpleNamespace(get=lambda: "10"))
    tab._selected_kinematic_biomod_path = lambda: biomod_path
    tab._navigable_annotation_frame_local_indices = lambda: [2]
    tab.selected_annotation_camera_names = lambda: ["cam0"]
    tab._current_images_root = lambda: None
    tab._frame_annotation_measurement_count = lambda frame_number, camera_names: 4 if int(frame_number) == 15 else 0
    tab._estimate_kinematic_q = lambda: setattr(tab, "_estimate_called", True) or np.array([0.5], dtype=float)
    tab.save_annotations = lambda: None
    tab._clear_pending_reprojection = lambda: None
    tab._clear_kinematic_assist_preview = lambda: None
    tab.refresh_preview = lambda: setattr(tab, "_refreshed", True)

    pipeline_gui.AnnotationTab.step_frame(tab, 1)

    assert tab._frame_index == 2
    assert tab._estimate_called is True
    assert tab._refreshed is True


def test_annotation_refresh_preview_restores_nearest_saved_kinematic_state(monkeypatch, tmp_path):
    biomod_path = tmp_path / "demo.bioMod"
    biomod_path.write_text("demo", encoding="utf-8")

    class _FakeBiorbdModel:
        def __init__(self, _path):
            pass

        def nbQ(self):
            return 2

    import sys

    monkeypatch.setitem(sys.modules, "biorbd", SimpleNamespace(Model=_FakeBiorbdModel))
    monkeypatch.setattr(pipeline_gui, "canonicalize_model_q_rotation_branches", lambda _model, q: np.asarray(q))

    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = SimpleNamespace(
        frames=np.array([10, 11, 12], dtype=int),
        camera_names=["cam0"],
        raw_keypoints=np.zeros((1, 3, len(pipeline_gui.COCO17), 2), dtype=float),
    )
    tab.calibrations = {"cam0": SimpleNamespace(image_size=(640, 480))}
    tab.kinematic_model_choices = {"demo": biomod_path}
    tab.kinematic_model_var = SimpleNamespace(get=lambda: "demo")
    tab.kinematic_frame_states = {("demo", 10): np.array([1.0, 2.0], dtype=float)}
    tab.kinematic_status_var = SimpleNamespace(set=lambda value: setattr(tab, "_kin_status", value))
    tab.kinematic_assist_var = SimpleNamespace(get=lambda: True)
    tab.kinematic_projected_points = None
    tab.kinematic_segmented_back_projected = {}
    tab._current_frame_idx = 2
    tab.frame_var = SimpleNamespace(get=lambda: 2, set=lambda value: setattr(tab, "_frame_index", value))
    tab.frame_label = SimpleNamespace(configure=lambda **kwargs: setattr(tab, "_frame_label_text", kwargs["text"]))
    tab.frame_filter_var = SimpleNamespace(get=lambda: pipeline_gui.ANNOTATION_FRAME_FILTER_OPTIONS["all"])
    tab.crop_var = SimpleNamespace(get=lambda: False)
    tab.selected_keypoint_name = lambda: "nose"
    tab.selected_annotation_camera_names = lambda: ["cam0"]
    tab.show_images_var = SimpleNamespace(get=lambda: False)
    tab._current_images_root = lambda: None
    tab.preview_figure = pipeline_gui.Figure(figsize=(4, 4))
    tab.preview_canvas = SimpleNamespace(draw_idle=lambda: None)
    tab.motion_prior_diameter = SimpleNamespace(get=lambda: "15")
    tab.show_motion_prior_var = SimpleNamespace(get=lambda: False)
    tab.show_epipolar_var = SimpleNamespace(get=lambda: False)
    tab.show_triangulated_hint_var = SimpleNamespace(get=lambda: False)
    tab._pending_reprojection_points = {}
    tab.annotation_payload = {}
    tab._annotation_xy = lambda *_args, **_kwargs: None
    tab._axis_to_camera = {}
    tab._cursor_artists = {}
    tab.image_brightness_var = SimpleNamespace(get=lambda: 1.0)
    tab.image_contrast_var = SimpleNamespace(get=lambda: 1.0)
    tab.preview_canvas_widget = SimpleNamespace()

    monkeypatch.setattr(pipeline_gui, "camera_layout", lambda _n: (1, 1))
    monkeypatch.setattr(pipeline_gui, "apply_2d_axis_limits", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline_gui, "hide_2d_axes", lambda ax: None)
    monkeypatch.setattr(pipeline_gui, "draw_skeleton_2d", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline_gui, "draw_upper_back_overlay_2d", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline_gui, "resolve_execution_image_path", lambda *_args, **_kwargs: None)

    def fake_set_preview(_biomod_path, _camera_names, q_values):
        tab.kinematic_q_current = np.asarray(q_values, dtype=float)
        tab.kinematic_projected_points = np.zeros((1, 1, len(pipeline_gui.COCO17), 2), dtype=float)
        tab.kinematic_segmented_back_projected = {}

    tab._set_kinematic_preview_from_q = fake_set_preview
    tab._ensure_crop_limits = lambda _camera_names: {}

    pipeline_gui.AnnotationTab.refresh_preview(tab)

    np.testing.assert_allclose(tab.kinematic_q_current, np.array([1.0, 2.0], dtype=float))
    assert tab._kin_status == "Using nearest saved q from frame 10 for frame 12."


def test_annotation_refresh_preview_draws_model_overlay_with_light_style(monkeypatch):
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = SimpleNamespace(
        frames=np.array([10], dtype=int),
        camera_names=["cam0"],
        raw_keypoints=np.zeros((1, 1, len(pipeline_gui.COCO17), 2), dtype=float),
    )
    tab.calibrations = {"cam0": SimpleNamespace(image_size=(640, 480))}
    tab.kinematic_assist_var = SimpleNamespace(get=lambda: True)
    tab.kinematic_projected_points = np.zeros((1, 1, len(pipeline_gui.COCO17), 2), dtype=float)
    tab.kinematic_segmented_back_projected = {
        "hip_triangle": np.zeros((1, 1, 4, 2), dtype=float),
        "shoulder_triangle": np.zeros((1, 1, 4, 2), dtype=float),
        "mid_back": np.zeros((1, 1, 1, 2), dtype=float),
    }
    tab.frame_var = SimpleNamespace(get=lambda: 0, set=lambda value: setattr(tab, "_frame_index", value))
    tab._current_frame_idx = 0
    tab.frame_label = SimpleNamespace(configure=lambda **kwargs: setattr(tab, "_frame_label_text", kwargs["text"]))
    tab.frame_filter_var = SimpleNamespace(get=lambda: pipeline_gui.ANNOTATION_FRAME_FILTER_OPTIONS["all"])
    tab.crop_var = SimpleNamespace(get=lambda: False)
    tab.selected_keypoint_name = lambda: "nose"
    tab.selected_annotation_camera_names = lambda: ["cam0"]
    tab.show_images_var = SimpleNamespace(get=lambda: False)
    tab._current_images_root = lambda: None
    tab.preview_figure = pipeline_gui.Figure(figsize=(4, 4))
    tab.preview_canvas = SimpleNamespace(draw_idle=lambda: None)
    tab.motion_prior_diameter = SimpleNamespace(get=lambda: "15")
    tab.show_motion_prior_var = SimpleNamespace(get=lambda: False)
    tab.show_epipolar_var = SimpleNamespace(get=lambda: False)
    tab.show_triangulated_hint_var = SimpleNamespace(get=lambda: False)
    tab._pending_reprojection_points = {}
    tab.annotation_payload = {}
    tab._annotation_xy = lambda *_args, **_kwargs: None
    tab._axis_to_camera = {}
    tab._cursor_artists = {}
    tab.image_brightness_var = SimpleNamespace(get=lambda: 1.0)
    tab.image_contrast_var = SimpleNamespace(get=lambda: 1.0)
    tab.preview_canvas_widget = SimpleNamespace()
    tab._ensure_crop_limits = lambda _camera_names: {}

    captured: dict[str, dict[str, object]] = {}

    monkeypatch.setattr(pipeline_gui, "camera_layout", lambda _n: (1, 1))
    monkeypatch.setattr(pipeline_gui, "apply_2d_axis_limits", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline_gui, "hide_2d_axes", lambda ax: None)
    monkeypatch.setattr(pipeline_gui, "resolve_execution_image_path", lambda *_args, **_kwargs: None)

    def capture_skeleton(*args, **kwargs):
        captured["skeleton"] = kwargs

    def capture_back_overlay(*args, **kwargs):
        captured["back"] = kwargs

    monkeypatch.setattr(pipeline_gui, "draw_skeleton_2d", capture_skeleton)
    monkeypatch.setattr(pipeline_gui, "draw_upper_back_overlay_2d", capture_back_overlay)

    pipeline_gui.AnnotationTab.refresh_preview(tab)

    assert captured["skeleton"]["marker_size"] == pytest.approx(3.2)
    assert captured["skeleton"]["marker_fill"] is False
    assert captured["skeleton"]["line_alpha"] == pytest.approx(0.32)
    assert captured["skeleton"]["line_style"] == "--"
    assert captured["skeleton"]["line_width_scale"] == pytest.approx(0.62)
    assert captured["back"]["line_width"] == pytest.approx(1.0)
    assert captured["back"]["line_alpha"] == pytest.approx(0.32)
    assert captured["back"]["line_style"] == "--"
    assert captured["back"]["marker_size"] == pytest.approx(18.0)
    assert captured["back"]["marker_alpha"] == pytest.approx(0.38)


def test_annotation_refresh_preview_keeps_epipolar_guides_from_all_cameras_when_single_view_selected(monkeypatch):
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = SimpleNamespace(
        frames=np.array([10], dtype=int),
        camera_names=["cam0", "cam1", "cam2"],
        raw_keypoints=np.zeros((3, 1, len(pipeline_gui.COCO17), 2), dtype=float),
    )
    tab.calibrations = {
        "cam0": SimpleNamespace(image_size=(640, 480)),
        "cam1": SimpleNamespace(image_size=(640, 480)),
        "cam2": SimpleNamespace(image_size=(640, 480)),
    }
    tab.kinematic_assist_var = SimpleNamespace(get=lambda: False)
    tab.kinematic_projected_points = None
    tab.kinematic_segmented_back_projected = {}
    tab.frame_var = SimpleNamespace(get=lambda: 0, set=lambda value: setattr(tab, "_frame_index", value))
    tab._current_frame_idx = 0
    tab.frame_label = SimpleNamespace(configure=lambda **kwargs: setattr(tab, "_frame_label_text", kwargs["text"]))
    tab.frame_filter_var = SimpleNamespace(get=lambda: pipeline_gui.ANNOTATION_FRAME_FILTER_OPTIONS["all"])
    tab.crop_var = SimpleNamespace(get=lambda: False)
    tab.selected_keypoint_name = lambda: "nose"
    tab.selected_annotation_camera_names = lambda: ["cam0"]
    tab.show_images_var = SimpleNamespace(get=lambda: False)
    tab._current_images_root = lambda: None
    tab.preview_figure = pipeline_gui.Figure(figsize=(4, 4))
    tab.preview_canvas = SimpleNamespace(draw_idle=lambda: None)
    tab.motion_prior_diameter = SimpleNamespace(get=lambda: "15")
    tab.show_motion_prior_var = SimpleNamespace(get=lambda: False)
    tab.show_epipolar_var = SimpleNamespace(get=lambda: True)
    tab.show_triangulated_hint_var = SimpleNamespace(get=lambda: False)
    tab._pending_reprojection_points = {}
    tab.annotation_payload = {}
    tab._axis_to_camera = {}
    tab._cursor_artists = {}
    tab.image_brightness_var = SimpleNamespace(get=lambda: 1.0)
    tab.image_contrast_var = SimpleNamespace(get=lambda: 1.0)
    tab.preview_canvas_widget = SimpleNamespace()
    tab._ensure_crop_limits = lambda _camera_names: {}
    tab._annotation_xy = lambda camera_name, _frame_number, _keypoint_name: (
        np.array([100.0, 200.0], dtype=float) if camera_name in {"cam1", "cam2"} else None
    )

    seen_sources = []

    monkeypatch.setattr(pipeline_gui, "camera_layout", lambda _n: (1, 1))
    monkeypatch.setattr(pipeline_gui, "apply_2d_axis_limits", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline_gui, "hide_2d_axes", lambda ax: None)
    monkeypatch.setattr(pipeline_gui, "draw_skeleton_2d", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline_gui, "draw_upper_back_overlay_2d", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline_gui, "render_annotation_camera_view", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        pipeline_gui,
        "annotation_epipolar_guides",
        lambda _calibs, source_camera_name, target_camera_name, _xy: (
            seen_sources.append((source_camera_name, target_camera_name)) or None
        ),
    )

    pipeline_gui.AnnotationTab.refresh_preview(tab)

    assert ("cam1", "cam0") in seen_sources
    assert ("cam2", "cam0") in seen_sources


def test_annotation_compute_pending_reprojection_points_updates_only_existing_annotations(monkeypatch):
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab.pose_data = SimpleNamespace(frames=np.array([10], dtype=int), camera_names=["cam0", "cam1", "cam2"])
    tab.calibrations = {"cam0": object(), "cam1": object(), "cam2": object()}
    tab.current_frame_number = lambda: 10
    tab.selected_annotation_camera_names = lambda: ["cam0", "cam1", "cam2"]
    points = {
        ("cam0", 10, "left_shoulder"): np.array([100.0, 200.0], dtype=float),
        ("cam1", 10, "left_shoulder"): np.array([105.0, 205.0], dtype=float),
        ("cam2", 10, "left_shoulder"): np.array([110.0, 210.0], dtype=float),
        ("cam0", 10, "nose"): np.array([120.0, 220.0], dtype=float),
        ("cam1", 10, "nose"): np.array([125.0, 225.0], dtype=float),
    }
    tab._annotation_xy = lambda camera_name, frame_number, keypoint_name: points.get(
        (camera_name, frame_number, keypoint_name)
    )

    def fake_reproject(_calibrations, *, target_camera_name, source_camera_names, source_points_2d):
        return np.array([len(source_camera_names), float(np.sum(np.asarray(source_points_2d)))], dtype=float)

    monkeypatch.setattr(pipeline_gui, "annotation_triangulated_reprojection", fake_reproject)

    pending = pipeline_gui.AnnotationTab._compute_pending_reprojection_points(tab)

    assert ("cam2", 10, "nose") not in pending
    assert ("cam0", 10, "left_shoulder") in pending
    assert ("cam2", 10, "left_shoulder") in pending
    assert ("cam0", 10, "nose") in pending


def test_annotation_reproject_button_toggles_to_confirm_and_applies_updates():
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab._pending_reprojection_points = {}
    tab.reproject_button_var = SimpleNamespace(set=lambda value: setattr(tab, "_button_text", value))
    tab.refresh_preview = lambda: setattr(tab, "_refreshed", getattr(tab, "_refreshed", 0) + 1)
    tab.save_annotations = lambda: setattr(tab, "_saved", getattr(tab, "_saved", 0) + 1)
    tab.annotation_payload = {}
    tab._compute_pending_reprojection_points = lambda: {
        ("cam0", 10, "left_shoulder"): np.array([111.0, 222.0], dtype=float)
    }

    pipeline_gui.AnnotationTab.on_reproject_button(tab)

    assert tab._button_text == "Confirm"
    assert len(tab._pending_reprojection_points) == 1

    pipeline_gui.AnnotationTab.on_reproject_button(tab)

    xy, _score = pipeline_gui.get_annotation_point(
        tab.annotation_payload, camera_name="cam0", frame_number=10, keypoint_name="left_shoulder"
    )
    np.testing.assert_allclose(np.asarray(xy, dtype=float), np.array([111.0, 222.0]))
    assert tab._saved == 1
    assert tab._button_text == "Reproject"
    assert tab._pending_reprojection_points == {}


def test_annotation_click_outside_cancels_pending_reprojection():
    refreshed = []
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab._pending_reprojection_points = {("cam0", 10, "left_shoulder"): np.array([1.0, 2.0], dtype=float)}
    tab.reproject_button_var = SimpleNamespace(set=lambda value: setattr(tab, "_button_text", value))
    tab.refresh_preview = lambda: refreshed.append("refreshed")
    tab.pose_data = SimpleNamespace(frames=np.array([10], dtype=int))

    pipeline_gui.AnnotationTab.on_preview_click(
        tab,
        SimpleNamespace(inaxes=None, xdata=None, ydata=None, button=1),
    )

    assert tab._pending_reprojection_points == {}
    assert tab._button_text == "Reproject"
    assert refreshed == ["refreshed"]


def test_annotation_cancel_pending_reprojection_from_tk_event_ignores_reproject_button():
    tab = pipeline_gui.AnnotationTab.__new__(pipeline_gui.AnnotationTab)
    tab._pending_reprojection_points = {("cam0", 10, "left_shoulder"): np.array([1.0, 2.0], dtype=float)}
    tab.reproject_button = object()
    tab.reproject_button_var = SimpleNamespace(set=lambda value: setattr(tab, "_button_text", value))
    tab.refresh_preview = lambda: setattr(tab, "_refreshed", True)

    pipeline_gui.AnnotationTab._cancel_pending_reprojection_from_tk_event(
        tab, SimpleNamespace(widget=tab.reproject_button)
    )

    assert tab._pending_reprojection_points
    assert not hasattr(tab, "_refreshed")


def test_preview_pose_frame_indices_aligns_sparse_bundle_frames():
    pose_frames = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=int)
    target_frames = np.array([0, 3, 6], dtype=int)

    indices = pipeline_gui.preview_pose_frame_indices(pose_frames, target_frames)

    np.testing.assert_array_equal(indices, np.array([0, 3, 6], dtype=int))


def test_infer_pose2sim_trc_from_keypoints_uses_inputs_trc_folder(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    (root / "inputs" / "keypoints").mkdir(parents=True)
    (root / "inputs" / "trc").mkdir(parents=True)
    keypoints_path = root / "inputs" / "keypoints" / "trial_keypoints.json"
    trc_path = root / "inputs" / "trc" / "trial.trc"
    keypoints_path.write_text("{}", encoding="utf-8")
    trc_path.write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(pipeline_gui, "ROOT", root)
    monkeypatch.setattr(pipeline_gui, "infer_dataset_name", lambda **_kwargs: "trial")

    assert pipeline_gui.infer_pose2sim_trc_from_keypoints(keypoints_path) == trc_path


def test_sync_dd_reference_path_ignores_empty_keypoints_path():
    tab = pipeline_gui.DDTab.__new__(pipeline_gui.DDTab)
    tab.state = SimpleNamespace(keypoints_var=SimpleNamespace(get=lambda: ""))
    tab.dd_reference_path = SimpleNamespace(var=SimpleNamespace(set=lambda value: setattr(tab, "_dd_value", value)))
    tab._dd_value = "unchanged"

    pipeline_gui.DDTab.sync_dd_reference_path(tab)

    assert tab._dd_value == ""


def test_dd_sync_dataset_dir_no_longer_depends_on_local_dataset_widget():
    tab = pipeline_gui.DDTab.__new__(pipeline_gui.DDTab)
    tab.state = SimpleNamespace()
    calls = []
    tab.sync_dd_reference_path = lambda: calls.append("dd")
    tab.refresh_available_reconstructions = lambda: calls.append("refresh")

    pipeline_gui.DDTab.sync_dataset_dir(tab)

    assert calls == ["dd", "refresh"]


def test_jump_list_label_with_reference_shows_detected_and_expected_codes():
    jump = SimpleNamespace(somersault_turns=-0.99, twist_turns=-0.02, code="40/")

    label = pipeline_gui.jump_list_label_with_reference(4, jump, "42/")

    assert label == "S4 | som -0.99 | tw -0.02 | det 40/ | exp 42/"


def test_sanitize_filetypes_keeps_regular_patterns(monkeypatch):
    filetypes = (
        ("JSON files", "*.json"),
        ("All files", "*.*"),
    )

    monkeypatch.setattr(pipeline_gui.sys, "platform", "linux")
    sanitized = pipeline_gui.LabeledEntry._sanitize_filetypes(filetypes)

    assert sanitized == filetypes


def test_sanitize_filetypes_simplifies_macos_basename_wildcards(monkeypatch):
    monkeypatch.setattr(pipeline_gui.sys, "platform", "darwin")
    filetypes = (
        ("DD JSON", "*_DD.json"),
        ("2D keypoints JSON", "*_keypoints.json"),
        ("All files", "*.*"),
    )

    sanitized = pipeline_gui.LabeledEntry._sanitize_filetypes(filetypes)

    assert sanitized == (
        ("DD JSON", "*.json"),
        ("2D keypoints JSON", "*.json"),
        ("All files", "*"),
    )


def test_labeled_entry_browse_invokes_callback_after_setting_relative_path(monkeypatch, tmp_path):
    root = tmp_path / "workspace"
    root.mkdir(parents=True)
    selected = root / "profiles.json"
    selected.write_text("[]", encoding="utf-8")
    values = {"current": "", "callback": None}

    entry = pipeline_gui.LabeledEntry.__new__(pipeline_gui.LabeledEntry)
    entry.directory = False
    entry.filetypes = None
    entry.browse_initialdir = None
    entry.on_browse_selected = lambda value: values.__setitem__("callback", value)
    entry.var = SimpleNamespace(set=lambda value: values.__setitem__("current", value))
    entry.get = lambda: values["current"]

    monkeypatch.setattr(pipeline_gui, "ROOT", root)
    monkeypatch.setattr(pipeline_gui.filedialog, "askopenfilename", lambda **_kwargs: str(selected))

    pipeline_gui.LabeledEntry._browse(entry)

    assert values["current"] == "profiles.json"
    assert values["callback"] == "profiles.json"


def test_flip_method_display_name_uses_user_friendly_labels():
    assert pipeline_gui.flip_method_display_name("none") == "None"
    assert pipeline_gui.flip_method_display_name("epipolar_fast_viterbi") == "Epipolar fast (Viterbi)"
    assert pipeline_gui.flip_method_display_name("triangulation_once") == "Triangulation once"


def test_coherence_method_display_name_uses_precomputed_labels():
    assert pipeline_gui.coherence_method_display_name("epipolar") == "Epipolar (precomputed)"
    assert pipeline_gui.coherence_method_display_name("epipolar_fast") == "Epipolar fast (precomputed)"
    assert pipeline_gui.coherence_method_display_name("epipolar_framewise") == "Epipolar (framewise)"
    assert pipeline_gui.coherence_method_display_name("epipolar_fast_framewise") == "Epipolar fast (framewise)"
    assert pipeline_gui.coherence_method_from_display_name("Epipolar fast (precomputed)") == "epipolar_fast"
    assert pipeline_gui.coherence_method_from_display_name("Epipolar fast (framewise)") == "epipolar_fast_framewise"


def test_profiles_tab_selected_profile_flip_method_disables_flip_for_none():
    tab = pipeline_gui.ProfilesTab.__new__(pipeline_gui.ProfilesTab)
    tab.flip_method = SimpleNamespace(get=lambda: "none")

    assert pipeline_gui.ProfilesTab.selected_profile_flip_method(tab) is None


def test_profiles_tab_refresh_profile_tree_shows_flip_mode_column():
    tab = pipeline_gui.ProfilesTab.__new__(pipeline_gui.ProfilesTab)
    tab.profile_tree = _FakeTree()
    tab.state = SimpleNamespace(
        profiles=[
            pipeline_gui.ReconstructionProfile(
                name="demo",
                family="ekf_2d",
                pose_data_mode="cleaned",
                triangulation_method="exhaustive",
                flip=True,
                flip_method="epipolar_fast",
            )
        ]
    )

    pipeline_gui.ProfilesTab.refresh_profile_tree(tab)

    assert tab.profile_tree.rows["0"][5] == "Epipolar fast (local)"


def test_profiles_tab_load_profile_into_form_restores_profile_options():
    tab = pipeline_gui.ProfilesTab.__new__(pipeline_gui.ProfilesTab)
    tab.state = SimpleNamespace(
        initial_rotation_correction_var=SimpleNamespace(
            get=lambda: False, set=lambda value: setattr(tab, "_rotfix", bool(value))
        ),
        pose_filter_window_var=SimpleNamespace(set=lambda value: setattr(tab, "_pose_filter_window", value)),
        pose_outlier_ratio_var=SimpleNamespace(set=lambda value: setattr(tab, "_pose_outlier_ratio", value)),
        pose_p_low_var=SimpleNamespace(set=lambda value: setattr(tab, "_pose_p_low", value)),
        pose_p_high_var=SimpleNamespace(set=lambda value: setattr(tab, "_pose_p_high", value)),
        flip_improvement_ratio_var=SimpleNamespace(set=lambda value: setattr(tab, "_flip_ratio", value)),
        flip_min_gain_px_var=SimpleNamespace(set=lambda value: setattr(tab, "_flip_gain", value)),
        flip_min_other_cameras_var=SimpleNamespace(set=lambda value: setattr(tab, "_flip_min_cams", value)),
        flip_restrict_to_outliers_var=SimpleNamespace(set=lambda value: setattr(tab, "_flip_restrict", bool(value))),
        flip_outlier_percentile_var=SimpleNamespace(set=lambda value: setattr(tab, "_flip_pct", value)),
        flip_outlier_floor_px_var=SimpleNamespace(set=lambda value: setattr(tab, "_flip_floor", value)),
        flip_temporal_weight_var=SimpleNamespace(set=lambda value: setattr(tab, "_flip_temp_w", value)),
        flip_temporal_tau_px_var=SimpleNamespace(set=lambda value: setattr(tab, "_flip_temp_tau", value)),
    )
    tab._updating_profile_name = False
    tab.profile_name = _FakeEntryField("")
    tab.family = SimpleNamespace(get=lambda: tab._family, set=lambda value: setattr(tab, "_family", value))
    tab._family = "ekf_2d"
    tab.pose_data_mode = SimpleNamespace(
        get=lambda: tab._pose_mode, set=lambda value: setattr(tab, "_pose_mode", value)
    )
    tab._pose_mode = "raw"
    tab.frame_stride = SimpleNamespace(get=lambda: tab._stride, set=lambda value: setattr(tab, "_stride", value))
    tab._stride = "1"
    tab.triang_method = SimpleNamespace(get=lambda: tab._triang, set=lambda value: setattr(tab, "_triang", value))
    tab._triang = "greedy"
    tab.reprojection_threshold_var = SimpleNamespace(
        get=lambda: tab._reproj_threshold,
        set=lambda value: setattr(tab, "_reproj_threshold", value),
    )
    tab._reproj_threshold = "15"
    tab.predictor = SimpleNamespace(get=lambda: tab._predictor, set=lambda value: setattr(tab, "_predictor", value))
    tab._predictor = "acc"
    tab.coherence_method = SimpleNamespace(
        get=lambda: tab._coherence,
        set=lambda value: setattr(tab, "_coherence", value),
    )
    tab._coherence = pipeline_gui.coherence_method_display_name("epipolar")
    tab.ekf2d_initial_state_method = SimpleNamespace(
        get=lambda: tab._q0,
        set=lambda value: setattr(tab, "_q0", value),
    )
    tab._q0 = "ekf_bootstrap"
    tab.ekf2d_bootstrap_passes = _FakeEntryField("5")
    tab.upper_back_sagittal_gain = _FakeEntryField("0.2")
    tab.upper_back_pseudo_std_deg = _FakeEntryField("10")
    tab.flip_method = SimpleNamespace(
        get=lambda: tab._flip_method, set=lambda value: setattr(tab, "_flip_method", value)
    )
    tab._flip_method = "none"
    tab.flip_method_label_var = SimpleNamespace(set=lambda value: setattr(tab, "_flip_label", value))
    tab._flip_label = ""
    tab.lock_var = SimpleNamespace(get=lambda: tab._lock, set=lambda value: setattr(tab, "_lock", bool(value)))
    tab._lock = False
    tab.initial_rot_var = SimpleNamespace(
        get=lambda: tab._rotfix, set=lambda value: setattr(tab, "_rotfix", bool(value))
    )
    tab._rotfix = False
    tab.biorbd_noise = _FakeEntryField("1e-8")
    tab.biorbd_error = _FakeEntryField("1e-4")
    tab.biorbd_kalman_init_method = SimpleNamespace(
        get=lambda: tab._ekf3d_init,
        set=lambda value: setattr(tab, "_ekf3d_init", value),
    )
    tab._ekf3d_init = "triangulation_ik_root_translation"
    tab.measurement_noise = _FakeEntryField("1.5")
    tab.process_noise = _FakeEntryField("1.0")
    tab.coherence_floor = _FakeEntryField("0.35")
    tab.profile_cameras_list = _FakeListbox()
    for name in ("cam_a", "cam_b", "cam_c"):
        tab.profile_cameras_list.insert("end", name)
    tab.profile_cameras_summary = SimpleNamespace(set=lambda value: setattr(tab, "_camera_summary", value))
    tab._camera_summary = ""
    tab.profile_models_list = _FakeListbox()
    for label in ("model_a", "model_b"):
        tab.profile_models_list.insert("end", label)
    tab.profile_models_summary = SimpleNamespace(set=lambda value: setattr(tab, "_model_summary", value))
    tab.ekf_model_info_var = SimpleNamespace(set=lambda value: setattr(tab, "_model_info", value))
    tab._profile_model_choices = {
        "model_a": "output/demo/models/model_a.bioMod",
        "model_b": "output/demo/models/model_b.bioMod",
    }
    tab._model_summary = ""
    tab._model_info = ""
    tab._pose_filter_window = ""
    tab._pose_outlier_ratio = ""
    tab._pose_p_low = ""
    tab._pose_p_high = ""
    tab._flip_ratio = ""
    tab._flip_gain = ""
    tab._flip_min_cams = ""
    tab._flip_restrict = False
    tab._flip_pct = ""
    tab._flip_floor = ""
    tab._flip_temp_w = ""
    tab._flip_temp_tau = ""
    tab.add_profile_button = _FakeButton()
    tab.refresh_profile_camera_choices = lambda: None
    tab.refresh_profile_model_choices = lambda: None
    tab.update_family_controls = lambda: None

    profile = pipeline_gui.ReconstructionProfile(
        name="ekf_profile",
        family="ekf_2d",
        camera_names=["cam_b", "cam_c"],
        ekf_model_path="output/demo/models/model_b.bioMod",
        predictor="dyn",
        ekf2d_initial_state_method="root_pose_bootstrap",
        ekf2d_bootstrap_passes=7,
        flip=True,
        flip_method="ekf_prediction_gate",
        dof_locking=True,
        initial_rotation_correction=True,
        pose_data_mode="cleaned",
        frame_stride=3,
        triangulation_method="exhaustive",
        reprojection_threshold_px=None,
        coherence_method="epipolar_fast_framewise",
        no_root_unwrap=True,
        measurement_noise_scale=2.0,
        process_noise_scale=0.5,
        coherence_confidence_floor=0.4,
        upper_back_sagittal_gain=0.35,
        upper_back_pseudo_std_deg=6.0,
        pose_filter_window=11,
        pose_outlier_threshold_ratio=0.25,
        pose_amplitude_lower_percentile=7.0,
        pose_amplitude_upper_percentile=92.0,
        flip_improvement_ratio=0.8,
        flip_min_gain_px=4.0,
        flip_min_other_cameras=3,
        flip_restrict_to_outliers=False,
        flip_outlier_percentile=90.0,
        flip_outlier_floor_px=6.0,
        flip_temporal_weight=0.5,
        flip_temporal_tau_px=12.0,
    )

    pipeline_gui.ProfilesTab.load_profile_into_form(tab, profile)

    assert tab.profile_name.get() == "ekf_profile"
    assert tab._family == "ekf_2d"
    assert tab._predictor == "dyn"
    assert tab._reproj_threshold == "none"
    assert tab._flip_method == "ekf_prediction_gate"
    assert tab._flip_label == "EKF prediction gate"
    assert tab._coherence == pipeline_gui.coherence_method_display_name("epipolar_fast_framewise")
    assert tab.ekf2d_bootstrap_passes.get() == "7"
    assert tab.upper_back_sagittal_gain.get() == "0.35"
    assert tab.upper_back_pseudo_std_deg.get() == "6"
    assert tab.profile_cameras_list.curselection() == (1, 2)
    assert tab.profile_models_list.curselection() == (1,)
    assert tab._pose_filter_window == "11"
    assert tab._pose_outlier_ratio == "0.25"
    assert tab._pose_p_low == "7"
    assert tab._pose_p_high == "92"
    assert tab._flip_ratio == "0.8"
    assert tab._flip_gain == "4"
    assert tab._flip_min_cams == "3"
    assert tab._flip_restrict is False
    assert tab._flip_pct == "90"
    assert tab._flip_floor == "6"
    assert tab._flip_temp_w == "0.5"
    assert tab._flip_temp_tau == "12"
    assert tab.add_profile_button.state == "normal"


def test_infer_model_variant_from_biomod_detects_upper_back_layouts(tmp_path):
    single_trunk = tmp_path / "single.bioMod"
    single_trunk.write_text("segment\tTRUNK\nendsegment\n", encoding="utf-8")
    one_dof = tmp_path / "back_1d.bioMod"
    one_dof.write_text("segment\tUPPER_BACK\nrotations\ty\nendsegment\n", encoding="utf-8")
    three_dof = tmp_path / "back_3dof.bioMod"
    three_dof.write_text("segment\tUPPER_BACK\nrotations\tyxz\nendsegment\n", encoding="utf-8")
    upper_root_one_dof = tmp_path / "upper_root_back_1d.bioMod"
    upper_root_one_dof.write_text("segment\tLOWER_TRUNK\nrotations\ty\nendsegment\n", encoding="utf-8")
    upper_root_three_dof = tmp_path / "upper_root_back_3dof.bioMod"
    upper_root_three_dof.write_text("segment\tLOWER_TRUNK\nrotations\tyxz\nendsegment\n", encoding="utf-8")

    assert pipeline_gui.infer_model_variant_from_biomod(single_trunk) == pipeline_gui.DEFAULT_MODEL_VARIANT
    assert pipeline_gui.infer_model_variant_from_biomod(one_dof) == "back_flexion_1d"
    assert pipeline_gui.infer_model_variant_from_biomod(three_dof) == "back_3dof"
    assert pipeline_gui.infer_model_variant_from_biomod(upper_root_one_dof) == "upper_root_back_flexion_1d"
    assert pipeline_gui.infer_model_variant_from_biomod(upper_root_three_dof) == "upper_root_back_3dof"


def test_biomod_supports_upper_back_options(tmp_path):
    one_dof = tmp_path / "back_1d.bioMod"
    one_dof.write_text("segment\tUPPER_BACK\nrotations\ty\nendsegment\n", encoding="utf-8")

    assert not pipeline_gui.biomod_supports_upper_back_options(None)
    assert pipeline_gui.biomod_supports_upper_back_options(one_dof)


def test_profiles_tab_load_selected_profile_from_tree_uses_double_clicked_row():
    tab = pipeline_gui.ProfilesTab.__new__(pipeline_gui.ProfilesTab)
    loaded = []
    tab.state = SimpleNamespace(
        profiles=[
            pipeline_gui.ReconstructionProfile(name="first", family="pose2sim"),
            pipeline_gui.ReconstructionProfile(name="second", family="pose2sim"),
        ]
    )
    tab.profile_tree = _FakeTree()
    tab.profile_tree.selection_set(("0",))
    tab.profile_tree._identified_row = "1"
    tab.load_profile_into_form = lambda profile: loaded.append(profile.name)

    result = pipeline_gui.ProfilesTab.load_selected_profile_from_tree(tab, SimpleNamespace(y=12))

    assert loaded == ["second"]
    assert result == "break"
    assert tab.profile_tree.selection() == ("1",)


def test_profiles_tab_build_command_uses_runtime_profiles_cache(monkeypatch):
    tab = pipeline_gui.ProfilesTab.__new__(pipeline_gui.ProfilesTab)
    tab.state = SimpleNamespace(
        output_root_var=SimpleNamespace(get=lambda: "output"),
        calib_var=SimpleNamespace(get=lambda: "inputs/calibration/Calib.toml"),
        keypoints_var=SimpleNamespace(get=lambda: "inputs/keypoints/trial_keypoints.json"),
        fps_var=SimpleNamespace(get=lambda: "120"),
        workers_var=SimpleNamespace(get=lambda: "6"),
        pose2sim_trc_var=SimpleNamespace(get=lambda: ""),
    )
    tab.config_path = SimpleNamespace(get=lambda: "reconstruction_profiles.json")
    tab.selected_profiles = lambda: []

    monkeypatch.setattr(
        pipeline_gui, "write_runtime_profiles_config", lambda _state: Path(".cache/runtime_profiles.json")
    )
    monkeypatch.setattr(pipeline_gui, "display_path", lambda path: str(path))
    monkeypatch.setattr(pipeline_gui, "current_dataset_name", lambda _state: "trial")
    monkeypatch.setattr(pipeline_gui, "current_selected_camera_names", lambda _state: [])

    cmd = pipeline_gui.ProfilesTab.build_command(tab)

    assert cmd[cmd.index("--config") + 1] == ".cache/runtime_profiles.json"


def test_profiles_tab_build_command_passes_annotations_path_for_annotated_profiles(monkeypatch):
    tab = pipeline_gui.ProfilesTab.__new__(pipeline_gui.ProfilesTab)
    annotated_profile = pipeline_gui.ReconstructionProfile(
        name="annotated", family="ekf_2d", pose_data_mode="annotated"
    )
    tab.state = SimpleNamespace(
        output_root_var=SimpleNamespace(get=lambda: "output"),
        calib_var=SimpleNamespace(get=lambda: "inputs/calibration/Calib.toml"),
        keypoints_var=SimpleNamespace(get=lambda: "inputs/keypoints/trial_keypoints.json"),
        annotation_path_var=SimpleNamespace(get=lambda: "inputs/annotations/trial_annotations.json"),
        fps_var=SimpleNamespace(get=lambda: "120"),
        workers_var=SimpleNamespace(get=lambda: "6"),
        pose2sim_trc_var=SimpleNamespace(get=lambda: ""),
    )
    tab.selected_profiles = lambda: [annotated_profile]

    monkeypatch.setattr(
        pipeline_gui, "write_runtime_profiles_config", lambda _state: Path(".cache/runtime_profiles.json")
    )
    monkeypatch.setattr(pipeline_gui, "display_path", lambda path: str(path))
    monkeypatch.setattr(pipeline_gui, "current_dataset_name", lambda _state: "trial")
    monkeypatch.setattr(pipeline_gui, "current_selected_camera_names", lambda _state: [])

    cmd = pipeline_gui.ProfilesTab.build_command(tab)

    assert cmd[cmd.index("--annotations-path") + 1] == "inputs/annotations/trial_annotations.json"


def test_reconstructions_tab_build_command_uses_runtime_profiles_cache(monkeypatch):
    tab = pipeline_gui.ReconstructionsTab.__new__(pipeline_gui.ReconstructionsTab)
    tab.state = SimpleNamespace(
        output_root_var=SimpleNamespace(get=lambda: "output"),
        calib_var=SimpleNamespace(get=lambda: "inputs/calibration/Calib.toml"),
        keypoints_var=SimpleNamespace(get=lambda: "inputs/keypoints/trial_keypoints.json"),
        fps_var=SimpleNamespace(get=lambda: "120"),
        workers_var=SimpleNamespace(get=lambda: "6"),
        pose2sim_trc_var=SimpleNamespace(get=lambda: ""),
    )
    tab.selected_profiles = lambda: []

    monkeypatch.setattr(
        pipeline_gui, "write_runtime_profiles_config", lambda _state: Path(".cache/runtime_profiles.json")
    )
    monkeypatch.setattr(pipeline_gui, "display_path", lambda path: str(path))
    monkeypatch.setattr(pipeline_gui, "current_dataset_name", lambda _state: "trial")
    monkeypatch.setattr(pipeline_gui, "current_selected_camera_names", lambda _state: [])

    cmd = pipeline_gui.ReconstructionsTab.build_command(tab)

    assert cmd[cmd.index("--config") + 1] == ".cache/runtime_profiles.json"


def test_batch_tab_scan_keypoints_files_populates_tree(monkeypatch, tmp_path):
    keypoints_a = tmp_path / "inputs" / "keypoints" / "trial_a_keypoints.json"
    keypoints_b = tmp_path / "inputs" / "keypoints" / "trial_b_keypoints.json"
    keypoints_a.parent.mkdir(parents=True)
    keypoints_a.write_text("{}", encoding="utf-8")
    keypoints_b.write_text("{}", encoding="utf-8")

    tab = pipeline_gui.BatchTab.__new__(pipeline_gui.BatchTab)
    tab.keypoints_glob_entry = _FakeEntryField("inputs/keypoints/*.json")
    tab.keypoints_tree = _FakeTree()

    monkeypatch.setattr(
        pipeline_gui,
        "batch_discover_keypoints_files",
        lambda patterns, root=None: [keypoints_a, keypoints_b],
    )
    monkeypatch.setattr(
        pipeline_gui,
        "batch_infer_annotations_for_keypoints",
        lambda path: path.with_name(f"{path.stem.replace('_keypoints', '')}_annotations.json"),
    )
    monkeypatch.setattr(
        pipeline_gui,
        "batch_infer_pose2sim_trc_for_keypoints",
        lambda path: path.parent.parent / "trc" / f"{path.stem.replace('_keypoints', '')}.trc",
    )
    monkeypatch.setattr(
        pipeline_gui,
        "infer_dataset_name",
        lambda **kwargs: Path(kwargs["keypoints_path"]).stem.replace("_keypoints", ""),
    )
    monkeypatch.setattr(pipeline_gui, "display_path", lambda path: str(Path(path).name))

    pipeline_gui.BatchTab.scan_keypoints_files(tab)

    assert set(tab.keypoints_tree.rows.keys()) == {str(keypoints_a), str(keypoints_b)}
    assert tab.keypoints_tree.rows[str(keypoints_a)][0] == "trial_a"
    assert tab.keypoints_tree.rows[str(keypoints_b)][2] == "trial_b_annotations.json"


def test_batch_tab_build_command_uses_selected_datasets_and_profiles():
    tab = pipeline_gui.BatchTab.__new__(pipeline_gui.BatchTab)
    tab.state = SimpleNamespace(
        output_root_var=SimpleNamespace(get=lambda: "output"),
        calib_var=SimpleNamespace(get=lambda: "inputs/calibration/Calib.toml"),
        fps_var=SimpleNamespace(get=lambda: "120"),
        workers_var=SimpleNamespace(get=lambda: "6"),
        notify_reconstructions_updated=lambda: None,
    )
    tab.config_path = _FakeEntryField("reconstruction_profiles.json")
    tab.excel_output_entry = _FakeEntryField("output/batch_summary.xlsx")
    tab.batch_name_entry = _FakeEntryField("demo_batch")
    tab.continue_on_error_var = SimpleNamespace(get=lambda: True)
    tab.export_only_var = SimpleNamespace(get=lambda: False)
    tab.keypoints_glob_entry = _FakeEntryField("inputs/keypoints/*.json")
    tab.keypoints_tree = _FakeTree()
    keypoints_a = "inputs/keypoints/trial_a_keypoints.json"
    keypoints_b = "inputs/keypoints/trial_b_keypoints.json"
    tab.keypoints_tree.insert("", "end", iid=keypoints_a, values=("trial_a", keypoints_a, "-", "-"))
    tab.keypoints_tree.insert("", "end", iid=keypoints_b, values=("trial_b", keypoints_b, "-", "-"))
    tab.keypoints_tree.selection_set((keypoints_b,))
    tab.batch_profiles = [
        pipeline_gui.ReconstructionProfile(name="tri", family="triangulation"),
        pipeline_gui.ReconstructionProfile(name="ekf", family="ekf_2d"),
    ]
    tab.batch_profile_tree = _FakeTree()
    tab.batch_profile_tree.insert("", "end", iid="0", values=("tri", "triangulation", "cleaned"))
    tab.batch_profile_tree.insert("", "end", iid="1", values=("ekf", "ekf_2d", "cleaned"))
    tab.batch_profile_tree.selection_set(("1",))

    cmd = pipeline_gui.BatchTab.build_command(tab)

    assert cmd[:2] == [pipeline_gui.sys.executable, "batch_run.py"]
    assert cmd[cmd.index("--config") + 1] == "reconstruction_profiles.json"
    assert cmd[cmd.index("--excel-output") + 1] == "output/batch_summary.xlsx"
    assert cmd[cmd.index("--batch-name") + 1] == "demo_batch"
    assert cmd[cmd.index("--keypoints-glob") + 1] == keypoints_b
    assert cmd[cmd.index("--profile") + 1] == "ekf"
    assert "--continue-on-error" in cmd


def test_reconstructions_tab_export_selected_pseudo_root_from_points_writes_npz(monkeypatch, tmp_path):
    recon_dir = tmp_path / "output" / "trial" / "reconstructions" / "demo"
    recon_dir.mkdir(parents=True)
    points_3d = np.zeros((3, 17, 3), dtype=float)
    for frame_idx in range(points_3d.shape[0]):
        z_offset = 1.0 + 0.1 * frame_idx
        points_3d[frame_idx, pipeline_gui.KP_INDEX["left_hip"]] = (-0.2, 0.0, z_offset)
        points_3d[frame_idx, pipeline_gui.KP_INDEX["right_hip"]] = (0.2, 0.0, z_offset)
        points_3d[frame_idx, pipeline_gui.KP_INDEX["left_shoulder"]] = (-0.25, 0.0, z_offset + 0.5)
        points_3d[frame_idx, pipeline_gui.KP_INDEX["right_shoulder"]] = (0.25, 0.0, z_offset + 0.5)
    bundle_path = recon_dir / "reconstruction_bundle.npz"
    np.savez(
        bundle_path,
        points_3d=points_3d,
        frames=np.array([0, 1, 2], dtype=int),
        time_s=np.array([0.0, 1.0 / 120.0, 2.0 / 120.0], dtype=float),
    )

    info_messages = []
    tab = pipeline_gui.ReconstructionsTab.__new__(pipeline_gui.ReconstructionsTab)
    tab.state = SimpleNamespace(
        fps_var=SimpleNamespace(get=lambda: "120"),
        initial_rotation_correction_var=SimpleNamespace(get=lambda: True),
    )
    tab.status_summaries = {"demo": {"initial_rotation_correction_applied": True}}
    tab._selected_reconstruction_dir = lambda: recon_dir
    monkeypatch.setattr(
        pipeline_gui.messagebox, "showinfo", lambda title, message: info_messages.append((title, message))
    )

    pipeline_gui.ReconstructionsTab.export_selected_pseudo_root_from_points(tab)

    output_path = recon_dir / "demo_pseudo_root_q.npz"
    assert output_path.exists()
    exported = np.load(output_path, allow_pickle=True)
    assert exported["q"].shape == (3, 6)
    assert exported["qdot"].shape == (3, 6)
    assert exported["qddot"].shape == (3, 6)
    assert exported["q_names"].tolist() == pipeline_gui.ROOT_Q_NAMES.tolist()
    assert info_messages


def test_reconstructions_tab_export_selected_pseudo_root_interpolates_short_nan_gap(monkeypatch, tmp_path):
    recon_dir = tmp_path / "output" / "trial" / "reconstructions" / "demo"
    recon_dir.mkdir(parents=True)
    points_3d = np.zeros((5, 17, 3), dtype=float)
    for frame_idx in range(points_3d.shape[0]):
        z_offset = 1.0 + 0.1 * frame_idx
        points_3d[frame_idx, pipeline_gui.KP_INDEX["left_hip"]] = (-0.2, 0.0, z_offset)
        points_3d[frame_idx, pipeline_gui.KP_INDEX["right_hip"]] = (0.2, 0.0, z_offset)
        points_3d[frame_idx, pipeline_gui.KP_INDEX["left_shoulder"]] = (-0.25, 0.0, z_offset + 0.5)
        points_3d[frame_idx, pipeline_gui.KP_INDEX["right_shoulder"]] = (0.25, 0.0, z_offset + 0.5)
    points_3d[2, pipeline_gui.KP_INDEX["left_shoulder"]] = np.array([np.nan, np.nan, np.nan], dtype=float)
    bundle_path = recon_dir / "reconstruction_bundle.npz"
    np.savez(
        bundle_path,
        points_3d=points_3d,
        frames=np.arange(5, dtype=int),
        time_s=np.arange(5, dtype=float) / 120.0,
    )

    info_messages = []
    tab = pipeline_gui.ReconstructionsTab.__new__(pipeline_gui.ReconstructionsTab)
    tab.state = SimpleNamespace(
        fps_var=SimpleNamespace(get=lambda: "120"),
        initial_rotation_correction_var=SimpleNamespace(get=lambda: True),
    )
    tab.status_summaries = {"demo": {"initial_rotation_correction_applied": True}}
    tab._selected_reconstruction_dir = lambda: recon_dir
    monkeypatch.setattr(
        pipeline_gui.messagebox, "showinfo", lambda title, message: info_messages.append((title, message))
    )

    pipeline_gui.ReconstructionsTab.export_selected_pseudo_root_from_points(tab)

    exported = np.load(recon_dir / "demo_pseudo_root_q.npz", allow_pickle=True)
    assert np.all(np.isfinite(exported["q"][2]))
    assert np.all(np.isfinite(exported["qdot"][2]))
    assert info_messages


def test_reconstructions_tab_export_selected_pseudo_root_fills_short_edge_gaps(monkeypatch, tmp_path):
    recon_dir = tmp_path / "output" / "trial" / "reconstructions" / "demo"
    recon_dir.mkdir(parents=True)
    points_3d = np.zeros((5, 17, 3), dtype=float)
    for frame_idx in range(points_3d.shape[0]):
        z_offset = 1.0 + 0.1 * frame_idx
        points_3d[frame_idx, pipeline_gui.KP_INDEX["left_hip"]] = (-0.2, 0.0, z_offset)
        points_3d[frame_idx, pipeline_gui.KP_INDEX["right_hip"]] = (0.2, 0.0, z_offset)
        points_3d[frame_idx, pipeline_gui.KP_INDEX["left_shoulder"]] = (-0.25, 0.0, z_offset + 0.5)
        points_3d[frame_idx, pipeline_gui.KP_INDEX["right_shoulder"]] = (0.25, 0.0, z_offset + 0.5)
    points_3d[0, pipeline_gui.KP_INDEX["left_shoulder"]] = np.array([np.nan, np.nan, np.nan], dtype=float)
    points_3d[-1, pipeline_gui.KP_INDEX["right_shoulder"]] = np.array([np.nan, np.nan, np.nan], dtype=float)
    np.savez(
        recon_dir / "reconstruction_bundle.npz",
        points_3d=points_3d,
        frames=np.arange(5, dtype=int),
        time_s=np.arange(5, dtype=float) / 120.0,
    )

    tab = pipeline_gui.ReconstructionsTab.__new__(pipeline_gui.ReconstructionsTab)
    tab.state = SimpleNamespace(
        fps_var=SimpleNamespace(get=lambda: "120"),
        initial_rotation_correction_var=SimpleNamespace(get=lambda: True),
    )
    tab.status_summaries = {"demo": {"initial_rotation_correction_applied": True}}
    tab._selected_reconstruction_dir = lambda: recon_dir
    monkeypatch.setattr(pipeline_gui.messagebox, "showinfo", lambda *_args, **_kwargs: None)

    pipeline_gui.ReconstructionsTab.export_selected_pseudo_root_from_points(tab)

    exported = np.load(recon_dir / "demo_pseudo_root_q.npz", allow_pickle=True)
    assert np.all(np.isfinite(exported["q"][0]))
    assert np.all(np.isfinite(exported["q"][-1]))
    assert np.all(np.isfinite(exported["qdot"][0]))
    assert np.all(np.isfinite(exported["qdot"][-1]))


def test_multiview_tab_build_command_uses_selected_cameras_and_images_root(monkeypatch):
    tab = pipeline_gui.MultiViewTab.__new__(pipeline_gui.MultiViewTab)
    tab.state = SimpleNamespace(
        fps_var=SimpleNamespace(get=lambda: "120"),
        workers_var=SimpleNamespace(get=lambda: "6"),
        pose2sim_trc_var=SimpleNamespace(get=lambda: ""),
    )
    tab.output_gif = _FakeEntryField("trial.gif")
    tab.gif_fps = _FakeEntryField("10")
    tab.stride = _FakeEntryField("5")
    tab.marker_size = _FakeEntryField("18")
    tab.images_root_entry = _FakeEntryField("inputs/images/trial")
    tab.crop_var = SimpleNamespace(get=lambda: True)
    tab.show_images_var = SimpleNamespace(get=lambda: True)
    tab.extra = _FakeEntryField("")
    tab.pose_data = SimpleNamespace(camera_names=["M11139", "M11140", "M11141"])
    tab.multiview_cameras_list = _FakeListbox()
    for name in tab.pose_data.camera_names:
        tab.multiview_cameras_list.insert("end", name)
    tab.multiview_cameras_list.selection_set(0)
    tab.multiview_cameras_list.selection_set(2)
    tab.selected_reconstruction_names = lambda: ["raw", "pose2sim"]
    tab.parse_extra_args = lambda raw: raw.split() if raw else []
    tab.resolved_output_gif_path = lambda: Path("output/trial/figures/trial.gif")

    monkeypatch.setattr(
        pipeline_gui,
        "discover_reconstruction_catalog",
        lambda *_args, **_kwargs: [{"name": "pose2sim", "cached": True}],
    )
    monkeypatch.setattr(pipeline_gui, "current_dataset_dir", lambda _state: Path("output/trial"))
    monkeypatch.setattr(pipeline_gui, "display_path", lambda path: str(path))

    cmd = pipeline_gui.MultiViewTab.build_command(tab)

    assert cmd[cmd.index("--camera-names") + 1] == "M11139,M11141"
    assert "--show-images" in cmd
    assert cmd[cmd.index("--images-root") + 1] == "inputs/images/trial"


class _FakeEntryField:
    def __init__(self, value: str = ""):
        self._value = str(value)
        self.var = SimpleNamespace(set=self.set)

    def get(self) -> str:
        return self._value

    def set(self, value: str) -> None:
        self._value = str(value)


class _FakeListbox:
    def __init__(self):
        self.items = []
        self._selection = []
        self._active = 0
        self._seen = None

    def delete(self, _start, _end=None):
        self.items = []
        self._selection = []

    def insert(self, _index, value):
        self.items.append(str(value))

    def selection_clear(self, _start, _end=None):
        self._selection = []

    def selection_set(self, first, last=None):
        if last in (None, ""):
            values = [int(first)]
        else:
            end = len(self.items) - 1 if last == pipeline_gui.tk.END else int(last)
            values = list(range(int(first), end + 1))
        for value in values:
            if value not in self._selection:
                self._selection.append(value)
        self._selection.sort()

    def curselection(self):
        return tuple(self._selection)

    def size(self):
        return len(self.items)

    def get(self, index):
        return self.items[int(index)]

    def see(self, _index):
        self._seen = int(_index)
        return None

    def activate(self, index):
        self._active = int(index)

    def index(self, value):
        if value == pipeline_gui.tk.ACTIVE:
            return self._active
        return int(value)


class _FakePackFrame:
    def __init__(self):
        self.pack_calls = []
        self.pack_forget_calls = 0

    def pack(self, *args, **kwargs):
        self.pack_calls.append((args, kwargs))

    def pack_forget(self):
        self.pack_forget_calls += 1


class _FakeButton:
    def __init__(self):
        self.state = None
        self.text = None

    def configure(self, **kwargs):
        if "state" in kwargs:
            self.state = kwargs["state"]
        if "text" in kwargs:
            self.text = kwargs["text"]


class _FakeCombobox:
    def __init__(self):
        self.values = None

    def configure(self, **kwargs):
        if "values" in kwargs:
            self.values = list(kwargs["values"])


class _FakeScale:
    def __init__(self, *, from_value=0, to_value=10, width=200):
        self._from = from_value
        self._to = to_value
        self._width = width
        self.focused = False

    def winfo_width(self):
        return self._width

    def cget(self, key):
        if key == "from":
            return self._from
        if key == "to":
            return self._to
        raise KeyError(key)

    def focus_set(self):
        self.focused = True


class _FakeAxis:
    def __init__(self):
        self.images = []
        self.xticks = None
        self.yticks = None
        self.tick_kwargs = None
        self.spines = {
            name: SimpleNamespace(set_visible=lambda _visible: None) for name in ("left", "right", "top", "bottom")
        }

    def imshow(self, image, **kwargs):
        self.images.append((image, kwargs))

    def set_xlim(self, *_args, **_kwargs):
        return None

    def set_ylim(self, *_args, **_kwargs):
        return None

    def set_aspect(self, *_args, **_kwargs):
        return None

    def set_title(self, *_args, **_kwargs):
        return None

    def grid(self, *_args, **_kwargs):
        return None

    def set_xlabel(self, *_args, **_kwargs):
        return None

    def set_ylabel(self, *_args, **_kwargs):
        return None

    def set_xticks(self, ticks):
        self.xticks = ticks

    def set_yticks(self, ticks):
        self.yticks = ticks

    def get_legend_handles_labels(self):
        return [], []

    def legend(self, *_args, **_kwargs):
        return None

    def text(self, *_args, **_kwargs):
        return None

    def set_axis_off(self):
        return None

    def tick_params(self, **kwargs):
        self.tick_kwargs = kwargs


class _FakeFigure:
    def __init__(self):
        self.axis = _FakeAxis()

    def clear(self):
        return None

    def subplots(self, *_args, **_kwargs):
        return self.axis

    def suptitle(self, *_args, **_kwargs):
        return None

    def tight_layout(self):
        return None


class _FakeCanvas:
    def draw_idle(self):
        return None


class _FakeText:
    def delete(self, *_args, **_kwargs):
        return None

    def insert(self, *_args, **_kwargs):
        return None


def test_model_tab_sync_frame_range_defaults_uses_available_2d_bounds(monkeypatch):
    tab = pipeline_gui.ModelTab.__new__(pipeline_gui.ModelTab)
    tab.frame_start = _FakeEntryField("")
    tab.frame_end = _FakeEntryField("")
    tab._auto_frame_range = None
    tab._syncing_frame_defaults = False

    monkeypatch.setattr(pipeline_gui.ModelTab, "_available_pose_frame_bounds", lambda _self: (12, 345))

    pipeline_gui.ModelTab._sync_frame_range_defaults(tab)

    assert tab.frame_start.get() == "12"
    assert tab.frame_end.get() == "345"
    assert tab._auto_frame_range == ("12", "345")


def test_model_tab_sync_frame_range_defaults_preserves_manual_selection(monkeypatch):
    tab = pipeline_gui.ModelTab.__new__(pipeline_gui.ModelTab)
    tab.frame_start = _FakeEntryField("50")
    tab.frame_end = _FakeEntryField("120")
    tab._auto_frame_range = ("0", "1745")
    tab._syncing_frame_defaults = False

    monkeypatch.setattr(pipeline_gui.ModelTab, "_available_pose_frame_bounds", lambda _self: (0, 1999))

    pipeline_gui.ModelTab._sync_frame_range_defaults(tab)

    assert tab.frame_start.get() == "50"
    assert tab.frame_end.get() == "120"
    assert tab._auto_frame_range == ("0", "1745")


def test_model_tab_on_command_success_refreshes_existing_models():
    tab = pipeline_gui.ModelTab.__new__(pipeline_gui.ModelTab)
    calls = []
    tab.refresh_existing_models = lambda: calls.append("refresh")

    pipeline_gui.ModelTab.on_command_success(tab)

    assert calls == ["refresh"]


def test_available_model_pose_modes_include_annotated_when_annotations_exist(monkeypatch, tmp_path):
    root = tmp_path / "workspace"
    keypoints_path = root / "inputs" / "keypoints" / "trial_keypoints.json"
    annotations_path = root / "inputs" / "annotations" / "trial_annotations.json"
    keypoints_path.parent.mkdir(parents=True)
    annotations_path.parent.mkdir(parents=True)
    keypoints_path.write_text("{}", encoding="utf-8")
    annotations_path.write_text("{}", encoding="utf-8")
    state = SimpleNamespace(
        annotation_path_var=SimpleNamespace(get=lambda: "inputs/annotations/trial_annotations.json")
    )

    monkeypatch.setattr(pipeline_gui, "ROOT", root)

    modes = pipeline_gui.available_model_pose_modes(state, keypoints_path)

    assert modes == ["raw", "annotated", "cleaned"]


def test_model_tab_sync_available_pose_modes_drops_annotated_when_missing(monkeypatch, tmp_path):
    root = tmp_path / "workspace"
    keypoints_path = root / "inputs" / "keypoints" / "trial_keypoints.json"
    keypoints_path.parent.mkdir(parents=True)
    keypoints_path.write_text("{}", encoding="utf-8")

    tab = pipeline_gui.ModelTab.__new__(pipeline_gui.ModelTab)

    class _Var:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):
            self.value = value

    tab.state = SimpleNamespace(
        keypoints_var=SimpleNamespace(get=lambda: "inputs/keypoints/trial_keypoints.json"),
        annotation_path_var=SimpleNamespace(get=lambda: "inputs/annotations/trial_annotations.json"),
    )
    tab.pose_mode_box = _FakeCombobox()
    tab.pose_data_mode = _Var("annotated")

    monkeypatch.setattr(pipeline_gui, "ROOT", root)

    pipeline_gui.ModelTab._sync_available_pose_modes(tab)

    assert tab.pose_mode_box.values == ["raw", "cleaned"]
    assert tab.pose_data_mode.get() == "cleaned"


def test_model_tab_build_command_uses_two_cameras_min_for_annotated(monkeypatch):
    tab = pipeline_gui.ModelTab.__new__(pipeline_gui.ModelTab)
    tab.state = SimpleNamespace(
        keypoints_var=SimpleNamespace(get=lambda: "inputs/keypoints/trial_keypoints.json"),
        calib_var=SimpleNamespace(get=lambda: "inputs/calibration/Calib.toml"),
        fps_var=SimpleNamespace(get=lambda: "120"),
        workers_var=SimpleNamespace(get=lambda: "6"),
        output_root_var=SimpleNamespace(get=lambda: "output"),
        pose_filter_window_var=SimpleNamespace(get=lambda: "9"),
        pose_outlier_ratio_var=SimpleNamespace(get=lambda: "0.1"),
        pose_p_low_var=SimpleNamespace(get=lambda: "5"),
        pose_p_high_var=SimpleNamespace(get=lambda: "95"),
    )
    tab.subject_mass = _FakeEntryField("55")
    tab.triang_method = SimpleNamespace(get=lambda: "exhaustive")
    tab.model_variant = SimpleNamespace(get=lambda: "single_trunk")
    tab.pose_data_mode = SimpleNamespace(get=lambda: "annotated")
    tab.pose_correction_mode = SimpleNamespace(get=lambda: "none")
    tab.symmetrize_limbs_var = SimpleNamespace(get=lambda: True)
    tab.frame_start = _FakeEntryField("")
    tab.frame_end = _FakeEntryField("")
    tab.max_frames = _FakeEntryField("")
    tab.initial_rot_var = SimpleNamespace(get=lambda: False)
    tab.extra = _FakeEntryField("")
    tab.parse_extra_args = lambda value: []
    tab.derived_model_dir = lambda: Path("output/trial/models/demo")
    tab.derived_biomod_path = lambda: "output/trial/models/demo/vitpose_chain.bioMod"
    monkeypatch.setattr(pipeline_gui, "current_selected_camera_names", lambda _state: [])

    cmd = pipeline_gui.ModelTab.build_command(tab)

    assert "--min-cameras-for-triangulation" in cmd
    idx = cmd.index("--min-cameras-for-triangulation")
    assert cmd[idx + 1] == "2"


def test_model_tab_preview_uses_two_camera_min_for_annotated(monkeypatch):
    tab = pipeline_gui.ModelTab.__new__(pipeline_gui.ModelTab)
    tab.state = SimpleNamespace(
        workers_var=SimpleNamespace(get=lambda: "6"),
        pose_filter_window_var=SimpleNamespace(get=lambda: "9"),
        pose_outlier_ratio_var=SimpleNamespace(get=lambda: "0.1"),
        pose_p_low_var=SimpleNamespace(get=lambda: "5"),
        pose_p_high_var=SimpleNamespace(get=lambda: "95"),
    )
    tab.pose_data_mode = SimpleNamespace(get=lambda: "annotated")
    tab.pose_correction_mode = SimpleNamespace(get=lambda: "none")
    tab.triang_method = SimpleNamespace(get=lambda: "exhaustive")

    pose_data = SimpleNamespace(frames=np.array([12], dtype=int))
    recorded = {}

    monkeypatch.setattr(pipeline_gui, "slice_pose_data", lambda _pose_data, _indices: pose_data)

    def fake_load_or_compute_triangulation_cache(**kwargs):
        recorded.update(kwargs)
        reconstruction = SimpleNamespace(points_3d=np.ones((1, len(pipeline_gui.COCO17), 3), dtype=float))
        return reconstruction, None, None, "computed_now"

    monkeypatch.setattr(pipeline_gui, "load_or_compute_triangulation_cache", fake_load_or_compute_triangulation_cache)

    reconstruction, preview_idx = pipeline_gui.ModelTab._first_valid_preview_reconstruction(
        tab,
        calibrations={"cam0": object(), "cam1": object()},
        pose_data=pose_data,
        model_dir=Path("output/trial/models/demo"),
    )

    assert reconstruction is not None
    assert preview_idx == 0
    assert recorded["min_cameras_for_triangulation"] == 2


def test_profiles_tab_refresh_profile_model_choices_populates_existing_models(monkeypatch):
    model_dir = Path("output/1_partie_0429/models/model_demo")
    biomod_path = model_dir / "model_demo.bioMod"
    tab = pipeline_gui.ProfilesTab.__new__(pipeline_gui.ProfilesTab)
    tab.state = SimpleNamespace()
    tab.family = SimpleNamespace(get=lambda: "ekf_2d")
    tab.profile_models_list = _FakeListbox()
    tab.profile_models_summary = SimpleNamespace(set=lambda value: setattr(tab, "_model_summary", value))
    tab.ekf_model_info_var = SimpleNamespace(set=lambda value: setattr(tab, "_model_info", value))
    tab._profile_model_choices = {"auto": None}
    tab.sync_profile_name = lambda: None
    tab.update_add_profile_button_state = lambda: None
    tab._model_info = ""
    tab._model_summary = ""

    monkeypatch.setattr(pipeline_gui, "current_dataset_dir", lambda _state: Path("output/1_partie_0429"))
    monkeypatch.setattr(pipeline_gui, "current_models_dir", lambda _state: Path("output/1_partie_0429/models"))
    monkeypatch.setattr(pipeline_gui, "scan_model_dirs", lambda _dataset_dir: [model_dir])
    monkeypatch.setattr(pipeline_gui, "display_path", lambda path: str(path))
    monkeypatch.setattr(
        Path,
        "glob",
        lambda self, pattern: (
            [biomod_path]
            if (self == Path("output/1_partie_0429/models") and pattern == "**/*.bioMod")
            or (self == model_dir and pattern == "*.bioMod")
            else []
        ),
    )

    pipeline_gui.ProfilesTab.refresh_profile_model_choices(tab)

    assert tuple(tab.profile_models_list.items) == ("model_demo",)
    assert tab._profile_model_choices["model_demo"] == str(biomod_path)
    assert tab._model_info == "reuse existing single_trunk model (faster)"
    assert tab.profile_models_list.curselection() == (0,)
    assert tab._model_summary == Path(str(biomod_path)).stem


def test_profiles_tab_refresh_profile_camera_choices_selects_all_by_default(monkeypatch):
    tab = pipeline_gui.ProfilesTab.__new__(pipeline_gui.ProfilesTab)
    tab.state = SimpleNamespace(calib_var=SimpleNamespace(get=lambda: "inputs/calibration/Calib.toml"))
    tab.profile_cameras_list = _FakeListbox()
    tab.profile_cameras_summary = SimpleNamespace(set=lambda value: setattr(tab, "_camera_summary", value))
    tab.sync_profile_name = lambda: None
    tab.update_add_profile_button_state = lambda: None
    tab._camera_summary = ""

    monkeypatch.setattr(
        pipeline_gui,
        "load_calibrations",
        lambda _path: {"cam_a": object(), "cam_b": object(), "cam_c": object()},
    )

    pipeline_gui.ProfilesTab.refresh_profile_camera_choices(tab)

    assert tuple(tab.profile_cameras_list.items) == ("cam_a", "cam_b", "cam_c")
    assert tab.profile_cameras_list.curselection() == (0, 1, 2)
    assert tab._camera_summary == "Cameras (n=3/3)"


def test_profiles_tab_current_profile_reuses_2d_explorer_clean_settings():
    tab = pipeline_gui.ProfilesTab.__new__(pipeline_gui.ProfilesTab)
    tab.state = SimpleNamespace(
        pose_filter_window_var=SimpleNamespace(get=lambda: "11"),
        pose_outlier_ratio_var=SimpleNamespace(get=lambda: "0.2"),
        pose_p_low_var=SimpleNamespace(get=lambda: "7"),
        pose_p_high_var=SimpleNamespace(get=lambda: "93"),
        flip_improvement_ratio_var=SimpleNamespace(get=lambda: "0.7"),
        flip_min_gain_px_var=SimpleNamespace(get=lambda: "3.0"),
        flip_min_other_cameras_var=SimpleNamespace(get=lambda: "2"),
        flip_restrict_to_outliers_var=SimpleNamespace(get=lambda: True),
        flip_outlier_percentile_var=SimpleNamespace(get=lambda: "85.0"),
        flip_outlier_floor_px_var=SimpleNamespace(get=lambda: "5.0"),
        flip_temporal_weight_var=SimpleNamespace(get=lambda: "0.35"),
        flip_temporal_tau_px_var=SimpleNamespace(get=lambda: "20.0"),
    )
    tab.family = SimpleNamespace(get=lambda: "ekf_2d")
    tab.profile_name = SimpleNamespace(get=lambda: "demo")
    tab.selected_profile_camera_names = lambda: ["cam_a"]
    tab.selected_profile_model_path = lambda: "output/demo/models/model.bioMod"
    tab.predictor = SimpleNamespace(get=lambda: "acc")
    tab.ekf2d_initial_state_method = SimpleNamespace(get=lambda: "ekf_bootstrap")
    tab.ekf2d_bootstrap_passes = SimpleNamespace(get=lambda: "5")
    tab.selected_profile_flip_method = lambda: None
    tab.lock_var = SimpleNamespace(get=lambda: False)
    tab.initial_rot_var = SimpleNamespace(get=lambda: False)
    tab.pose_data_mode = SimpleNamespace(get=lambda: "cleaned")
    tab.frame_stride = SimpleNamespace(get=lambda: "1")
    tab.triang_method = SimpleNamespace(get=lambda: "once")
    tab.reprojection_threshold_var = SimpleNamespace(get=lambda: "none")
    tab.coherence_method = SimpleNamespace(get=lambda: "Epipolar (precomputed)")
    tab.biorbd_noise = SimpleNamespace(get=lambda: "1e-8")
    tab.biorbd_error = SimpleNamespace(get=lambda: "1e-4")
    tab.biorbd_kalman_init_method = SimpleNamespace(get=lambda: "triangulation_ik_root_translation")
    tab.measurement_noise = SimpleNamespace(get=lambda: "1.5")
    tab.process_noise = SimpleNamespace(get=lambda: "1.0")
    tab.coherence_floor = SimpleNamespace(get=lambda: "0.35")

    profile = pipeline_gui.ProfilesTab.current_profile(tab)

    assert profile.pose_filter_window == 11
    assert profile.pose_outlier_threshold_ratio == 0.2
    assert profile.pose_amplitude_lower_percentile == 7.0
    assert profile.pose_amplitude_upper_percentile == 93.0
    assert profile.coherence_method == "epipolar"
    assert profile.reprojection_threshold_px is None
    assert profile.root_unwrap_mode == "off"
    assert profile.no_root_unwrap is True


def test_profiles_tab_update_profile_model_info_marks_existing_biomod_as_faster():
    tab = pipeline_gui.ProfilesTab.__new__(pipeline_gui.ProfilesTab)
    tab.family = SimpleNamespace(get=lambda: "ekf_2d")
    tab.ekf_model_info_var = SimpleNamespace(set=lambda value: setattr(tab, "_model_info", value))
    tab.selected_profile_model_path = lambda: "output/demo/models/model.bioMod"
    tab._model_info = ""

    pipeline_gui.ProfilesTab.update_profile_model_info(tab)

    assert tab._model_info == "reuse existing single_trunk model (faster)"


def test_profiles_tab_update_profile_model_info_requires_existing_biomod_for_ekf2d():
    tab = pipeline_gui.ProfilesTab.__new__(pipeline_gui.ProfilesTab)
    tab.family = SimpleNamespace(get=lambda: "ekf_2d")
    tab.ekf_model_info_var = SimpleNamespace(set=lambda value: setattr(tab, "_model_info", value))
    tab.selected_profile_model_path = lambda: None
    tab._model_info = ""

    pipeline_gui.ProfilesTab.update_profile_model_info(tab)

    assert tab._model_info == "auto-build single_trunk model from current 2D data (slower)"


def test_profiles_tab_update_add_profile_button_state_requires_two_cameras_and_model_for_ekf2d():
    tab = pipeline_gui.ProfilesTab.__new__(pipeline_gui.ProfilesTab)
    tab.family = SimpleNamespace(get=lambda: "ekf_2d")
    tab.add_profile_button = _FakeButton()
    tab.profile_cameras_list = _FakeListbox()
    tab.selected_profile_camera_names = lambda: ["cam_a"]
    tab.selected_profile_model_path = lambda: None

    pipeline_gui.ProfilesTab.update_add_profile_button_state(tab)
    assert tab.add_profile_button.state == "disabled"

    tab.selected_profile_camera_names = lambda: ["cam_a", "cam_b"]
    tab.selected_profile_model_path = lambda: "output/demo/models/model.bioMod"

    pipeline_gui.ProfilesTab.update_add_profile_button_state(tab)
    assert tab.add_profile_button.state == "normal"


def test_profiles_tab_update_family_controls_hides_triangulation_for_ekf2d():
    tab = pipeline_gui.ProfilesTab.__new__(pipeline_gui.ProfilesTab)
    tab.family = SimpleNamespace(get=lambda: "ekf_2d")
    tab.cameras_frame = _FakePackFrame()
    tab.pose_mode_frame = _FakePackFrame()
    tab.triang_frame = _FakePackFrame()
    tab.flip_frame = _FakePackFrame()
    tab.ekf2d_frame = _FakePackFrame()
    tab.ekf2d_observation_frame = _FakePackFrame()
    tab.ekf2d_params_frame = _FakePackFrame()
    tab.ekf3d_frame = _FakePackFrame()
    tab.clean_frame = _FakePackFrame()
    tab.ekf_model_frame = _FakePackFrame()
    tab.update_profile_model_info = lambda: None

    pipeline_gui.ProfilesTab.update_family_controls(tab)

    assert tab.triang_frame.pack_calls == []
    assert tab.ekf2d_frame.pack_calls


def test_model_tab_metadata_path_resolves_workspace_relative_paths(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir(parents=True)
    cache_path = root / "output" / "trial" / "models" / "demo" / "triangulation_pose2sim_like.npz"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(pipeline_gui, "ROOT", root)

    resolved = pipeline_gui.ModelTab._metadata_path(
        cache_path.parent, "output/trial/models/demo/triangulation_pose2sim_like.npz"
    )

    assert resolved == cache_path


def test_model_tab_refresh_existing_models_keeps_detected_models_with_match_status(monkeypatch):
    model_dir = Path("output/1_partie_0429/models/model_demo")
    biomod_path = model_dir / "model_demo.bioMod"
    tab = pipeline_gui.ModelTab.__new__(pipeline_gui.ModelTab)
    tab.model_tree = _FakeTree()
    tab.state = SimpleNamespace(
        pose_filter_window_var=SimpleNamespace(get=lambda: "9"),
        pose_outlier_ratio_var=SimpleNamespace(get=lambda: "0.1"),
        pose_p_low_var=SimpleNamespace(get=lambda: "5.0"),
        pose_p_high_var=SimpleNamespace(get=lambda: "95.0"),
    )
    tab.pose_data_mode = SimpleNamespace(get=lambda: "cleaned")
    tab.current_pose_correction_mode = lambda: "flip_epipolar_fast"
    tab.current_pose_source_label = lambda: "cleaned + flip_epipolar_fast"

    monkeypatch.setattr(pipeline_gui, "current_dataset_dir", lambda _state: Path("output/1_partie_0429"))
    monkeypatch.setattr(pipeline_gui, "scan_model_dirs", lambda _dataset_dir: [model_dir])
    monkeypatch.setattr(Path, "glob", lambda self, pattern: [biomod_path] if self == model_dir else [])
    monkeypatch.setattr(pipeline_gui, "display_path", lambda path: str(path))
    monkeypatch.setattr(pipeline_gui.ModelTab, "_model_matches_selected_2d_data", lambda *_args, **_kwargs: False)

    pipeline_gui.ModelTab.refresh_existing_models(tab)

    assert tab.model_tree.rows
    row = next(iter(tab.model_tree.rows.values()))
    assert row[0] == "model_demo"
    assert row[1] == "no"
    assert row[2] == str(biomod_path)


def test_model_tab_update_preview_viewer_controls_switches_button_state():
    tab = pipeline_gui.ModelTab.__new__(pipeline_gui.ModelTab)
    tab.preview_viewer = SimpleNamespace(get=lambda: "pyorerun")
    tab.open_preview_viewer_button = _FakeButton()

    pipeline_gui.ModelTab.update_preview_viewer_controls(tab)

    assert tab.open_preview_viewer_button.state == "normal"
    assert tab.open_preview_viewer_button.text == "Open pyorerun"


def test_model_tab_open_preview_in_viewer_launches_pyorerun(monkeypatch, tmp_path):
    biomod_path = tmp_path / "output" / "trial" / "models" / "demo" / "model_demo.bioMod"
    biomod_path.parent.mkdir(parents=True)
    biomod_path.write_text("bioMod", encoding="utf-8")
    launched = []

    tab = pipeline_gui.ModelTab.__new__(pipeline_gui.ModelTab)
    tab.preview_viewer = SimpleNamespace(get=lambda: "pyorerun")
    tab.preview_q_current = np.array([0.1, -0.2], dtype=float)
    tab.state = SimpleNamespace(fps_var=SimpleNamespace(get=lambda: "120"))
    tab._preview_model_path = lambda use_selected_model=False: biomod_path

    monkeypatch.setattr(pipeline_gui.subprocess, "Popen", lambda cmd, cwd=None: launched.append((cmd, cwd)))

    pipeline_gui.ModelTab.open_preview_in_viewer(tab)

    assert launched
    cmd, cwd = launched[0]
    assert cmd[0] == pipeline_gui.sys.executable
    assert cmd[1] == "tools/show_biomod_pyorerun.py"
    assert "--biomod" in cmd
    assert "--states" in cmd
    assert "--mode" in cmd
    assert "trajectory" in cmd
    states_path = biomod_path.parent / "preview_pyorerun_states.npz"
    assert states_path.exists()
    q = np.load(states_path)["q"]
    assert q.shape == (2, 2)
    assert np.allclose(q[0], [0.1, -0.2])
    assert cwd == pipeline_gui.ROOT


def test_clean_trial_outputs_aborts_when_confirmation_is_declined(monkeypatch, tmp_path):
    tab = pipeline_gui.DataExplorer2DTab.__new__(pipeline_gui.DataExplorer2DTab)
    tab.state = SimpleNamespace(
        pose_data_cache={("demo",): object()},
        calibration_cache={"demo": object()},
        notify_reconstructions_updated=lambda: None,
    )
    tab.update_dataset_summary = lambda: None
    dataset_dir = tmp_path / "output" / "trial"
    dataset_dir.mkdir(parents=True)
    shown_messages = []

    monkeypatch.setattr(pipeline_gui, "current_dataset_dir", lambda _state: dataset_dir)
    monkeypatch.setattr(pipeline_gui, "current_dataset_name", lambda _state: "trial")
    monkeypatch.setattr(pipeline_gui, "display_path", lambda path: str(path))
    monkeypatch.setattr(
        pipeline_gui.messagebox,
        "askyesno",
        lambda title, message, icon=None: shown_messages.append((title, message, icon)) or False,
    )

    pipeline_gui.DataExplorer2DTab.clean_trial_outputs(tab)

    assert dataset_dir.exists()
    assert shown_messages
    assert "trial 'trial'" in shown_messages[0][1]
    assert "cannot be undone" in shown_messages[0][1]


def test_clean_trial_outputs_deletes_dataset_after_confirmation(monkeypatch, tmp_path):
    notifications = []
    summaries = []
    refreshes = []
    tab = pipeline_gui.DataExplorer2DTab.__new__(pipeline_gui.DataExplorer2DTab)
    tab.state = SimpleNamespace(
        pose_data_cache={("demo",): object()},
        calibration_cache={"demo": object()},
        notify_reconstructions_updated=lambda: notifications.append("updated"),
        shared_reconstruction_panel=SimpleNamespace(_refresh_callback=lambda: refreshes.append("refresh")),
    )
    tab.update_dataset_summary = lambda: summaries.append("summary")
    tab.after_idle = lambda callback: callback()
    dataset_dir = tmp_path / "output" / "trial"
    dataset_dir.mkdir(parents=True)
    info_messages = []

    monkeypatch.setattr(pipeline_gui, "current_dataset_dir", lambda _state: dataset_dir)
    monkeypatch.setattr(pipeline_gui, "current_dataset_name", lambda _state: "trial")
    monkeypatch.setattr(pipeline_gui, "display_path", lambda path: str(path))
    monkeypatch.setattr(pipeline_gui.messagebox, "askyesno", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        pipeline_gui.messagebox,
        "showinfo",
        lambda title, message: info_messages.append((title, message)),
    )

    pipeline_gui.DataExplorer2DTab.clean_trial_outputs(tab)

    assert not dataset_dir.exists()
    assert tab.state.pose_data_cache == {}
    assert tab.state.calibration_cache == {}
    assert notifications == ["updated"]
    assert summaries == ["summary"]
    assert refreshes == ["refresh"]
    assert info_messages


def test_clean_trial_caches_deletes_only_cache_artifacts(monkeypatch, tmp_path):
    notifications = []
    summaries = []
    refreshes = []
    tab = pipeline_gui.DataExplorer2DTab.__new__(pipeline_gui.DataExplorer2DTab)
    tab.state = SimpleNamespace(
        pose_data_cache={("demo",): object()},
        calibration_cache={"demo": object()},
        notify_reconstructions_updated=lambda: notifications.append("updated"),
        shared_reconstruction_panel=SimpleNamespace(_refresh_callback=lambda: refreshes.append("refresh")),
    )
    tab.update_dataset_summary = lambda: summaries.append("summary")
    tab.after_idle = lambda callback: callback()
    dataset_dir = tmp_path / "output" / "trial"
    cache_root = dataset_dir / "_cache" / "pose2d"
    cache_root.mkdir(parents=True)
    (cache_root / "demo.npz").write_text("dummy", encoding="utf-8")
    model_dir = dataset_dir / "models" / "demo"
    model_dir.mkdir(parents=True)
    preview_cache = model_dir / "preview_q0_cache.npz"
    preview_cache.write_text("preview", encoding="utf-8")
    kept_file = model_dir / "model.bioMod"
    kept_file.write_text("bioMod", encoding="utf-8")
    info_messages = []

    monkeypatch.setattr(pipeline_gui, "current_dataset_dir", lambda _state: dataset_dir)
    monkeypatch.setattr(pipeline_gui, "current_dataset_name", lambda _state: "trial")
    monkeypatch.setattr(pipeline_gui, "display_path", lambda path: str(path))
    monkeypatch.setattr(pipeline_gui.messagebox, "askyesno", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        pipeline_gui.messagebox,
        "showinfo",
        lambda title, message: info_messages.append((title, message)),
    )

    pipeline_gui.DataExplorer2DTab.clean_trial_caches(tab)

    assert not (dataset_dir / "_cache").exists()
    assert not preview_cache.exists()
    assert kept_file.exists()
    assert dataset_dir.exists()
    assert tab.state.pose_data_cache == {}
    assert tab.state.calibration_cache == {}
    assert notifications == ["updated"]
    assert summaries == ["summary"]
    assert refreshes == ["refresh"]
    assert info_messages


def test_clear_models_deletes_only_selected_model_dirs(monkeypatch, tmp_path):
    selected_dir = tmp_path / "output" / "trial" / "models" / "selected_model"
    kept_dir = tmp_path / "output" / "trial" / "models" / "kept_model"
    selected_dir.mkdir(parents=True)
    kept_dir.mkdir(parents=True)
    (selected_dir / "selected.bioMod").write_text("bioMod", encoding="utf-8")
    (kept_dir / "kept.bioMod").write_text("bioMod", encoding="utf-8")

    tab = pipeline_gui.ModelTab.__new__(pipeline_gui.ModelTab)
    tab.model_tree = _FakeTree()
    tab.model_tree.insert(
        "", "end", iid="selected", values=("selected_model", "yes", str(selected_dir / "selected.bioMod"))
    )
    tab.model_tree.insert("", "end", iid="kept", values=("kept_model", "yes", str(kept_dir / "kept.bioMod")))
    tab.model_tree.selection_set(("selected",))
    sync_calls = []
    refresh_calls = []
    info_messages = []
    tab.sync_paths_from_state = lambda: sync_calls.append("sync")
    tab.refresh_existing_models = lambda: refresh_calls.append("refresh")

    monkeypatch.setattr(pipeline_gui.messagebox, "askyesno", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        pipeline_gui.messagebox, "showinfo", lambda title, message: info_messages.append((title, message))
    )

    pipeline_gui.ModelTab.clear_models(tab)

    assert not selected_dir.exists()
    assert kept_dir.exists()
    assert sync_calls == ["sync"]
    assert refresh_calls == ["refresh"]
    assert info_messages


def test_clear_reconstructions_deletes_only_selected_rows(monkeypatch, tmp_path):
    recon_root = tmp_path / "output" / "trial" / "reconstructions"
    kept_dir = recon_root / "kept"
    deleted_dir = recon_root / "selected"
    kept_dir.mkdir(parents=True)
    deleted_dir.mkdir(parents=True)
    notifications = []
    refreshes = []
    info_messages = []
    tab = pipeline_gui.ReconstructionsTab.__new__(pipeline_gui.ReconstructionsTab)
    tab.state = SimpleNamespace(
        shared_reconstruction_selection=["selected"],
        notify_reconstructions_updated=lambda: notifications.append("updated"),
    )
    tab.refresh_status_rows = lambda: refreshes.append("refresh")

    monkeypatch.setattr(pipeline_gui, "current_dataset_dir", lambda _state: tmp_path / "output" / "trial")
    monkeypatch.setattr(pipeline_gui, "reconstruction_dirs_for_path", lambda _dataset_dir: [deleted_dir, kept_dir])
    monkeypatch.setattr(pipeline_gui.messagebox, "askyesno", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        pipeline_gui.messagebox,
        "showinfo",
        lambda title, message: info_messages.append((title, message)),
    )

    pipeline_gui.ReconstructionsTab.clear_reconstructions(tab)

    assert not deleted_dir.exists()
    assert kept_dir.exists()
    assert tab.state.shared_reconstruction_selection == []
    assert notifications == ["updated"]
    assert refreshes == ["refresh"]
    assert info_messages


def test_load_preview_bundle_accepts_missing_pose2sim_trc(tmp_path):
    output_dir = tmp_path / "empty_dataset"
    output_dir.mkdir(parents=True)

    bundle = pipeline_gui.load_preview_bundle(output_dir, biomod_path=None, pose2sim_trc=None, align_root=False)

    np.testing.assert_array_equal(bundle["frames"], np.array([], dtype=int))
    np.testing.assert_array_equal(bundle["time_s"], np.array([], dtype=float))
    assert bundle["recon_3d"] == {}


def test_camera_tools_tab_sync_dataset_dir_infers_images_root(monkeypatch, tmp_path):
    tab = pipeline_gui.CameraToolsTab.__new__(pipeline_gui.CameraToolsTab)
    tab.state = SimpleNamespace(keypoints_var=SimpleNamespace(get=lambda: "inputs/keypoints/trial_keypoints.json"))
    tab.images_root_entry = _FakeEntryField("")
    tab.refresh_available_reconstructions = lambda: None
    tab.update_camera_filter_status = lambda: None
    tab.load_resources = lambda: None
    inferred_root = tmp_path / "inputs" / "images" / "trial"
    inferred_root.mkdir(parents=True)

    monkeypatch.setattr(pipeline_gui, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline_gui, "display_path", lambda path: str(path))
    monkeypatch.setattr(pipeline_gui, "infer_execution_images_root", lambda _path: inferred_root)

    pipeline_gui.CameraToolsTab.sync_dataset_dir(tab)

    assert tab.images_root == inferred_root
    assert tab.images_root_entry.get() == str(inferred_root)


def test_camera_tools_render_flip_preview_uses_image_frame_number_before_overlay(monkeypatch, tmp_path):
    tab = pipeline_gui.CameraToolsTab.__new__(pipeline_gui.CameraToolsTab)
    tab.base_pose_data = SimpleNamespace(
        camera_names=["M11139"],
        frames=np.array([24], dtype=int),
        keypoints=np.zeros((1, 1, 17, 2), dtype=float),
    )
    tab.base_pose_data.keypoints[0, 0, 11] = np.array([100.0, 200.0], dtype=float)
    tab.calibrations = {"M11139": SimpleNamespace(image_size=(1920, 1080))}
    tab.flip_method_var = SimpleNamespace(get=lambda: "epipolar")
    tab.flip_camera_var = SimpleNamespace(get=lambda: "M11139")
    tab.flip_applied_var = SimpleNamespace(get=lambda: False)
    tab.flip_figure = _FakeFigure()
    tab.flip_canvas = _FakeCanvas()
    tab.flip_details = _FakeText()
    tab.flip_status_var = SimpleNamespace(set=lambda _value: None)
    tab.images_root_entry = _FakeEntryField(str(tmp_path))
    tab.images_root = tmp_path
    tab.show_images_var = SimpleNamespace(get=lambda: True)
    tab.flip_masks = {"epipolar": np.zeros((1, 1), dtype=bool)}
    tab.flip_detail_arrays = {"epipolar": {}}
    tab._selected_flip_frame_local_idx = lambda: 0
    tab._reference_projection = lambda *_args, **_kwargs: (None, "none", "#444444")

    requested = []
    monkeypatch.setattr(
        two_d_view,
        "resolve_execution_image_path",
        lambda root, camera_name, frame_number: requested.append((root, camera_name, frame_number)) or None,
    )
    drawn_colors = []
    monkeypatch.setattr(
        pipeline_gui,
        "draw_skeleton_2d",
        lambda _ax, _points, color, *_args, **_kwargs: drawn_colors.append(color),
    )

    pipeline_gui.CameraToolsTab.render_flip_preview(tab)

    assert requested == [(tmp_path, "M11139", 24)]
    assert drawn_colors[0] == "#000000"
