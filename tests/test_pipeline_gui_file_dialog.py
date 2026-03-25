from pathlib import Path
from types import SimpleNamespace

import numpy as np

import pipeline_gui
from preview.dataset_preview_state import DatasetPreviewState
from preview.shared_reconstruction_panel import SharedReconstructionPanel


class _FakeTree:
    def __init__(self):
        self.rows = {}
        self._selection = ()

    def get_children(self, _root=""):
        return tuple(self.rows.keys())

    def delete(self, item):
        self.rows.pop(item, None)

    def insert(self, _parent, _where, iid, values):
        self.rows[iid] = tuple(values)

    def selection_set(self, selection):
        self._selection = tuple(selection)

    def selection(self):
        return self._selection

    def exists(self, name):
        return name in self.rows


def test_normalize_pose_correction_mode_accepts_epipolar_fast():
    assert pipeline_gui.normalize_pose_correction_mode("flip_epipolar_fast") == "flip_epipolar_fast"


def test_normalize_pose_correction_mode_accepts_explicit_viterbi_modes():
    assert pipeline_gui.normalize_pose_correction_mode("flip_epipolar_viterbi") == "flip_epipolar_viterbi"
    assert pipeline_gui.normalize_pose_correction_mode("flip_epipolar_fast_viterbi") == "flip_epipolar_fast_viterbi"


def test_normalize_pose_correction_mode_falls_back_to_none():
    assert pipeline_gui.normalize_pose_correction_mode("unexpected_mode") == "none"


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

    def delete(self, _start, _end=None):
        self.items = []
        self._selection = []

    def insert(self, _index, value):
        self.items.append(str(value))

    def selection_clear(self, _start, _end=None):
        self._selection = []

    def selection_set(self, index):
        value = int(index)
        if value not in self._selection:
            self._selection.append(value)

    def curselection(self):
        return tuple(self._selection)

    def size(self):
        return len(self.items)

    def get(self, index):
        return self.items[int(index)]

    def see(self, _index):
        return None


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

    def configure(self, **kwargs):
        if "state" in kwargs:
            self.state = kwargs["state"]


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
    assert tab._model_info == "reuse existing model (faster)"
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
    tab.triang_method = SimpleNamespace(get=lambda: "exhaustive")
    tab.coherence_method = SimpleNamespace(get=lambda: "Epipolar (precomputed)")
    tab.unwrap_var = SimpleNamespace(get=lambda: False)
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


def test_profiles_tab_update_profile_model_info_marks_existing_biomod_as_faster():
    tab = pipeline_gui.ProfilesTab.__new__(pipeline_gui.ProfilesTab)
    tab.family = SimpleNamespace(get=lambda: "ekf_2d")
    tab.ekf_model_info_var = SimpleNamespace(set=lambda value: setattr(tab, "_model_info", value))
    tab.selected_profile_model_path = lambda: "output/demo/models/model.bioMod"
    tab._model_info = ""

    pipeline_gui.ProfilesTab.update_profile_model_info(tab)

    assert tab._model_info == "reuse existing model (faster)"


def test_profiles_tab_update_profile_model_info_requires_existing_biomod_for_ekf2d():
    tab = pipeline_gui.ProfilesTab.__new__(pipeline_gui.ProfilesTab)
    tab.family = SimpleNamespace(get=lambda: "ekf_2d")
    tab.ekf_model_info_var = SimpleNamespace(set=lambda value: setattr(tab, "_model_info", value))
    tab.selected_profile_model_path = lambda: None
    tab._model_info = ""

    pipeline_gui.ProfilesTab.update_profile_model_info(tab)

    assert tab._model_info == "auto-build model from current 2D data (slower)"


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
