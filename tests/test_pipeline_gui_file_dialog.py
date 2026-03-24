import pipeline_gui


def test_normalize_pose_correction_mode_accepts_epipolar_fast():
    assert pipeline_gui.normalize_pose_correction_mode("flip_epipolar_fast") == "flip_epipolar_fast"


def test_normalize_pose_correction_mode_falls_back_to_none():
    assert pipeline_gui.normalize_pose_correction_mode("unexpected_mode") == "none"


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
