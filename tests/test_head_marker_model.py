"""Opt-in anthropometric head marker geometry (``head_marker_model``) of ``build_biomod``."""

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from _ekf2d_synthetic import (
    make_pose_and_reconstruction,
    model_keypoints_3d,
    project_keypoints,
    synthetic_cameras,
    synthetic_lengths,
    synthetic_q_trajectory,
)

import vitpose_ekf_pipeline as vep
from reconstruction.reconstruction_bundle import cache_key, load_or_build_model_cache
from reconstruction.reconstruction_profiles import ReconstructionProfile, build_pipeline_command, validate_profile
from vitpose_ekf_pipeline import (
    ANTHROPOMETRIC_HEAD_MARKER_RATIOS,
    KP_INDEX,
    SUPPORTED_MODEL_VARIANTS,
    build_biomod,
    ensure_local_imports,
    head_marker_local_positions,
    head_marker_model_metadata,
    metadata_cache_matches,
    model_stage_metadata,
    normalize_head_marker_model,
    run_ekf,
    save_model_stage,
)

HEAD_MARKERS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
MARKER_RE = re.compile(r"marker\t(\w+)\n\tparent\t(\w+)\n\tposition\t([^\n]+)\n")


def _lengths(head_length: float = 0.182):
    lengths = synthetic_lengths()
    lengths.head_length = head_length
    return lengths


def _markers(text: str) -> dict[str, tuple[str, np.ndarray]]:
    return {
        m.group(1): (m.group(2), np.array([float(v) for v in m.group(3).split()])) for m in MARKER_RE.finditer(text)
    }


def _require_biobuddy() -> None:
    ensure_local_imports()  # BIOBUDDY_ROOT / sibling checkout first, as build_biomod does
    pytest.importorskip("biobuddy")


def _build(tmp_path: Path, name: str, **kwargs) -> str:
    _require_biobuddy()
    path = tmp_path / f"{name}.bioMod"
    build_biomod(_lengths(), path, **kwargs)
    return path.read_text()


def _without_head(text: str) -> str:
    """Remove the HEAD mesh line and the five head marker blocks."""

    text = MARKER_RE.sub(lambda m: "" if m.group(1) in HEAD_MARKERS else m.group(0), text)
    head = text.index("segment\tHEAD")
    end = text.index("endsegment", head)
    block = "\n".join(line for line in text[head:end].split("\n") if not line.strip().startswith("mesh"))
    return text[:head] + block + text[end:]


def test_normalize_head_marker_model():
    assert normalize_head_marker_model(None) == "legacy"
    assert normalize_head_marker_model(" Anthropometric ") == "anthropometric"
    with pytest.raises(ValueError):
        normalize_head_marker_model("fitted")


def test_legacy_positions_are_the_historical_expressions():
    lengths = _lengths()
    h = lengths.head_length
    positions = head_marker_local_positions(lengths, "legacy")

    assert positions == {
        "nose": [h, 0, h],
        "left_eye": [h - lengths.eye_offset_x, lengths.eye_offset_y, h],
        "right_eye": [h - lengths.eye_offset_x, -lengths.eye_offset_y, h],
        "left_ear": [0, lengths.ear_offset_y, 0.7 * h],
        "right_ear": [0, -lengths.ear_offset_y, 0.7 * h],
    }
    assert head_marker_local_positions(lengths) == positions


@pytest.mark.parametrize("model_variant", SUPPORTED_MODEL_VARIANTS)
def test_legacy_biomod_is_byte_identical_to_the_default(tmp_path, model_variant):
    default = _build(tmp_path, "default", model_variant=model_variant)
    legacy = _build(tmp_path, "legacy", model_variant=model_variant, head_marker_model="legacy")

    assert legacy == default
    markers = _markers(legacy)
    h = 0.182
    np.testing.assert_allclose(markers["nose"][1], [h, 0.0, h])
    np.testing.assert_allclose(markers["left_ear"][1], [0.0, 0.06, 0.7 * h])
    assert "mesh\t0.182000\t0.000000\t0.182000" in legacy


@pytest.mark.parametrize("model_variant", SUPPORTED_MODEL_VARIANTS)
def test_anthropometric_changes_only_head_markers_and_head_mesh(tmp_path, model_variant):
    legacy = _build(tmp_path, "legacy", model_variant=model_variant)
    anthropometric = _build(tmp_path, "anthro", model_variant=model_variant, head_marker_model="anthropometric")
    legacy_markers, anthro_markers = _markers(legacy), _markers(anthropometric)

    # Same marker names, order and parents (COCO17 contract); every other line is unchanged.
    assert list(anthro_markers) == list(legacy_markers)
    assert all(anthro_markers[name][0] == legacy_markers[name][0] for name in legacy_markers)
    assert all(anthro_markers[name][0] == "HEAD" for name in HEAD_MARKERS)
    assert _without_head(anthropometric) == _without_head(legacy)
    for name in HEAD_MARKERS:
        np.testing.assert_allclose(
            anthro_markers[name][1], 0.182 * np.asarray(ANTHROPOMETRIC_HEAD_MARKER_RATIOS[name]), atol=1e-6
        )


def test_anthropometric_geometry_is_in_the_measured_range_and_symmetric():
    lengths = _lengths(0.182)
    p = {name: np.asarray(value) for name, value in head_marker_local_positions(lengths, "anthropometric").items()}
    h = lengths.head_length
    eye_mid = 0.5 * (p["left_eye"] + p["right_eye"])
    ear_mid = 0.5 * (p["left_ear"] + p["right_ear"])

    # Ranges (in units of h) of the per-trajectory shapes: 3 sequences x 3 detectors of pose2sim triangulations.
    assert 0.983 <= np.linalg.norm(p["nose"]) / h <= 1.011  # by definition of head_length (legacy: sqrt(2))
    assert 0.472 <= np.linalg.norm(p["nose"] - ear_mid) / h <= 0.556  # legacy: ~1.05 h (191 mm vs ~110 mm GT)
    assert 1.016 <= np.linalg.norm(eye_mid) / h <= 1.082
    assert 0.155 <= np.linalg.norm(p["nose"] - eye_mid) / h <= 0.184
    assert 0.187 <= np.linalg.norm(p["left_eye"] - p["right_eye"]) / h <= 0.262
    assert 0.612 <= np.linalg.norm(p["left_ear"] - p["right_ear"]) / h <= 0.770
    assert 0.759 <= np.linalg.norm(ear_mid) / h <= 0.850
    # Gauge: nose at 45 deg in the sagittal plane (same direction as legacy), eyes/ears in front of the ears.
    assert p["nose"][1] == 0.0 and p["nose"][0] == pytest.approx(p["nose"][2])
    assert p["nose"][0] > p["left_eye"][0] > p["left_ear"][0] > 0.0
    # Left/right mirror symmetry about the sagittal plane, left on +y.
    for left, right in (("left_eye", "right_eye"), ("left_ear", "right_ear")):
        np.testing.assert_array_equal(p[left] * np.array([1.0, -1.0, 1.0]), p[right])
        assert p[left][1] > 0.0
    # Scales with head_length.
    scaled = head_marker_local_positions(_lengths(2 * h), "anthropometric")
    np.testing.assert_allclose(scaled["left_ear"], 2.0 * p["left_ear"])


@pytest.mark.parametrize("model_variant", SUPPORTED_MODEL_VARIANTS)
def test_head_pivot_is_the_shoulder_centre_for_every_variant(tmp_path, model_variant):
    biorbd = pytest.importorskip("biorbd")
    _require_biobuddy()
    path = tmp_path / f"{model_variant}.bioMod"
    build_biomod(_lengths(), path, model_variant=model_variant, head_marker_model="anthropometric")
    model = biorbd.Model(str(path))
    names = [model.markerNames()[i].to_string() for i in range(model.nbMarkers())]
    markers = model.markers(np.zeros(model.nbQ()))
    world = {name: markers[i].to_array() for i, name in enumerate(names)}
    shoulder_centre = 0.5 * (world["left_shoulder"] + world["right_shoulder"])
    expected = head_marker_local_positions(_lengths(), "anthropometric")

    for name in HEAD_MARKERS:
        np.testing.assert_allclose(world[name] - shoulder_centre, expected[name], atol=1e-9)


def test_model_stage_metadata_is_unchanged_for_legacy_and_tracks_the_geometry(monkeypatch, tmp_path):
    reconstruction = SimpleNamespace(frames=np.arange(2), points_3d=np.zeros((2, len(KP_INDEX), 3)))
    args = (tmp_path / "triangulation.npz", reconstruction, 120.0, 55.0, False)
    default = model_stage_metadata(*args)
    legacy = model_stage_metadata(*args, head_marker_model="legacy")
    anthro = model_stage_metadata(*args, head_marker_model="anthropometric")

    historical_keys = {
        "model_stage_version",
        "reconstruction_cache_path",
        "reconstruction_n_frames",
        "reconstruction_frame_signature",
        "reconstruction_signature",
        "fps",
        "subject_mass_kg",
        "initial_rotation_correction",
        "model_variant",
        "symmetrize_limbs",
    }
    assert set(default) == historical_keys and legacy == default
    assert cache_key(legacy) == cache_key(default)
    assert head_marker_model_metadata("legacy") == {}
    assert anthro["head_marker_model"] == "anthropometric"
    assert anthro["reconstruction_signature"] == default["reconstruction_signature"]
    assert cache_key(anthro) != cache_key(default)

    biomod = tmp_path / "model.bioMod"
    biomod.write_text("version 4", encoding="utf-8")
    cache = tmp_path / "model_stage.npz"
    save_model_stage(cache, _lengths(), biomod, legacy)
    assert metadata_cache_matches(cache, legacy)
    assert not metadata_cache_matches(cache, anthro)

    # Changing the ratios changes the geometry signature, hence invalidates anthropometric caches.
    save_model_stage(cache, _lengths(), biomod, anthro)
    assert metadata_cache_matches(cache, anthro)
    changed = dict(ANTHROPOMETRIC_HEAD_MARKER_RATIOS, nose=(0.7, 0.0, 0.7))
    monkeypatch.setattr(vep, "ANTHROPOMETRIC_HEAD_MARKER_RATIOS", changed)
    changed_meta = model_stage_metadata(*args, head_marker_model="anthropometric")
    assert changed_meta["head_marker_geometry_signature"] != anthro["head_marker_geometry_signature"]
    assert not metadata_cache_matches(cache, changed_meta)


def test_model_cache_is_not_reused_across_head_marker_models(monkeypatch, tmp_path):
    reconstruction = SimpleNamespace(frames=np.arange(3), points_3d=np.zeros((3, len(KP_INDEX), 3)))
    built = []

    def fake_build(_lengths_arg, output_path, **kwargs):
        built.append(kwargs.get("head_marker_model", "legacy"))
        output_path.write_text(f"version 4\n// {kwargs.get('head_marker_model', 'legacy')}", encoding="utf-8")
        return output_path

    monkeypatch.setattr("reconstruction.reconstruction_bundle.estimate_segment_lengths", lambda *_a, **_k: _lengths())
    monkeypatch.setattr("reconstruction.reconstruction_bundle.build_biomod", fake_build)
    common = dict(
        output_dir=tmp_path / "reconstructions" / "run",
        reconstruction=reconstruction,
        reconstruction_cache_path=tmp_path / "triangulation_stage.npz",
        fps=120.0,
        subject_mass_kg=55.0,
        initial_rotation_correction=False,
        lengths_mode="full_triangulation",
    )

    _, legacy_biomod, legacy_cache, _, _, source = load_or_build_model_cache(**common)
    assert source == "computed_now"
    _, default_biomod, _, _, _, source = load_or_build_model_cache(**common, head_marker_model="legacy")
    assert source == "cache" and default_biomod == legacy_biomod
    _, anthro_biomod, anthro_cache, _, _, source = load_or_build_model_cache(
        **common, head_marker_model="anthropometric"
    )
    assert source == "computed_now"
    assert anthro_biomod.parent != legacy_biomod.parent and anthro_cache != legacy_cache
    assert "anthropometric" in anthro_biomod.read_text() and "legacy" in legacy_biomod.read_text()
    assert load_or_build_model_cache(**common, head_marker_model="anthropometric")[-1] == "cache"
    assert load_or_build_model_cache(**common)[-1] == "cache"
    assert built == ["legacy", "anthropometric"]


@pytest.mark.parametrize("family", ["ekf_2d", "ekf_3d"])
def test_profile_head_marker_model_is_named_and_forwarded(tmp_path, family):
    profile = validate_profile(ReconstructionProfile(name="", family=family, head_marker_model="Anthropometric"))
    cmd = build_pipeline_command(profile, tmp_path, Path("c.toml"), Path("k.json"))

    assert profile.head_marker_model == "anthropometric"
    assert "headanthro" in profile.name
    assert cmd[cmd.index("--head-marker-model") + 1] == "anthropometric"
    default = validate_profile(ReconstructionProfile(name="", family=family))
    assert default.head_marker_model == "legacy" and "headanthro" not in default.name
    assert "--head-marker-model" not in build_pipeline_command(default, tmp_path, Path("c.toml"), Path("k.json"))
    with pytest.raises(ValueError):
        validate_profile(ReconstructionProfile(name="", family=family, head_marker_model="fitted"))


def test_profile_head_marker_model_is_reset_without_a_model(tmp_path):
    profile = validate_profile(
        ReconstructionProfile(name="", family="triangulation", head_marker_model="anthropometric")
    )

    assert profile.head_marker_model == "legacy" and "headanthro" not in profile.name
    assert "--head-marker-model" not in build_pipeline_command(profile, tmp_path, Path("c.toml"), Path("k.json"))


@pytest.mark.parametrize(
    "module_name, argv",
    [
        ("export_reconstruction_bundle", ["prog", "--name", "x", "--family", "ekf_2d", "--output-dir", "out"]),
        ("vitpose_ekf_pipeline", ["prog"]),
    ],
)
def test_cli_head_marker_model_option(monkeypatch, module_name, argv):
    module = __import__(module_name)
    monkeypatch.setattr(sys, "argv", argv)
    assert module.parse_args().head_marker_model == "legacy"
    monkeypatch.setattr(sys, "argv", argv + ["--head-marker-model", "anthropometric"])
    assert module.parse_args().head_marker_model == "anthropometric"
    monkeypatch.setattr(sys, "argv", argv + ["--head-marker-model", "fitted"])
    with pytest.raises(SystemExit):
        module.parse_args()


def test_ekf_with_the_corrected_model_reproduces_head_markers_better(tmp_path):
    """Synthetic scene generated with the anthropometric head: the matching model tracks the face markers."""

    biorbd = pytest.importorskip("biorbd")
    _require_biobuddy()
    models = {}
    for name in ("legacy", "anthropometric"):
        path = tmp_path / f"{name}.bioMod"
        build_biomod(synthetic_lengths(), path, head_marker_model=name)
        models[name] = biorbd.Model(str(path))
    truth = models["anthropometric"]
    n_frames = 30
    q_true = synthetic_q_trajectory(truth, n_frames, seed=4)
    calibrations = synthetic_cameras(5)
    points_true = model_keypoints_3d(truth, q_true)
    keypoints = project_keypoints(points_true, calibrations)
    keypoints += np.random.default_rng(5).normal(scale=1.0, size=keypoints.shape)
    pose_data, reconstruction = make_pose_and_reconstruction(keypoints, points_true, calibrations)
    nq = truth.nbQ()
    face = [KP_INDEX[name] for name in HEAD_MARKERS]
    errors = {}
    for name, model in models.items():
        assert model.nbQ() == nq
        result, _ = run_ekf(
            None,
            calibrations,
            pose_data,
            reconstruction,
            fps=120.0,
            measurement_noise_scale=1.5,
            model=model,
            initial_state=np.concatenate((q_true[0], np.zeros(nq), np.zeros(nq))),
        )
        diff = np.linalg.norm(model_keypoints_3d(model, result["q"]) - points_true, axis=-1)[5:]
        errors[name] = (float(np.mean(diff[:, face])), float(np.mean(np.delete(diff, face, axis=1))))

    face_legacy, body_legacy = errors["legacy"]
    face_anthro, body_anthro = errors["anthropometric"]
    assert face_anthro < 0.01
    assert face_anthro < 0.25 * face_legacy
    assert body_anthro <= body_legacy + 1e-3
