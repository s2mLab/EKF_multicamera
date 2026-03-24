from pathlib import Path
from types import SimpleNamespace

import pipeline_gui
from preview.dataset_preview_state import DatasetPreviewState


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
    assert pipeline_gui.flip_method_display_name("epipolar_fast_viterbi") == "Epipolar fast (Viterbi)"
    assert pipeline_gui.flip_method_display_name("triangulation_once") == "Triangulation once"


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
