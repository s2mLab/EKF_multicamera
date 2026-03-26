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
    tab.unwrap_var = SimpleNamespace(get=lambda: tab._unwrap, set=lambda value: setattr(tab, "_unwrap", bool(value)))
    tab._unwrap = False
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


class _FakeAxis:
    def __init__(self):
        self.images = []

    def imshow(self, image):
        self.images.append(image)

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

    def get_legend_handles_labels(self):
        return [], []

    def legend(self, *_args, **_kwargs):
        return None

    def text(self, *_args, **_kwargs):
        return None

    def set_axis_off(self):
        return None


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
        pipeline_gui,
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
