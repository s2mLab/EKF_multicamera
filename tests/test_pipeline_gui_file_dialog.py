from pathlib import Path
from types import SimpleNamespace

from preview.dataset_preview_state import DatasetPreviewState

import pipeline_gui


def test_normalize_pose_correction_mode_accepts_epipolar_fast():
    assert pipeline_gui.normalize_pose_correction_mode("flip_epipolar_fast") == "flip_epipolar_fast"


def test_normalize_pose_correction_mode_falls_back_to_none():
    assert pipeline_gui.normalize_pose_correction_mode("unexpected_mode") == "none"


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
