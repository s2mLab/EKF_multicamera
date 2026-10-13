"""Back flexion pseudo-observation of the EKF2D back model variants (``back_prior_mode``).

Model conventions (checked here with the markers of ``build_biomod`` models, no Euler branch):
``THIGH:RotY > 0`` is a hip extension, ``UPPER_BACK:RotY > 0`` a back flexion (``back_*`` variants, root
= pelvis) and ``LOWER_TRUNK:RotY > 0`` a back extension (``upper_root_*`` variants, root = upper trunk).
"""

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
from _ekf2d_synthetic import (
    load_synthetic_model,
    make_pose_and_reconstruction,
    model_keypoints_3d,
    project_keypoints,
    synthetic_cameras,
)

import reconstruction.reconstruction_profiles as reconstruction_profiles
import vitpose_ekf_pipeline
from reconstruction.reconstruction_bundle import build_ekf_2d_bundle
from reconstruction.reconstruction_profiles import ReconstructionProfile, build_pipeline_command, validate_profile
from vitpose_ekf_pipeline import (
    DEFAULT_BACK_PRIOR_MODE,
    SUPPORTED_BACK_PRIOR_MODES,
    MultiViewKinematicEKF,
    normalize_back_prior_mode,
    q_names_from_model,
    run_ekf,
)

BACK_VARIANTS = ("back_flexion_1d", "back_3dof", "upper_root_back_flexion_1d", "upper_root_back_3dof")
GAIN = 0.2


def _legacy_reference_block(q, back_idx, hip_indices, nq, gain, std_rad):
    """Frozen copy of the historical ``_upper_back_pseudo_measurement_block`` (before ``back_prior_mode``)."""

    hip_flexion = np.asarray(q[list(hip_indices)], dtype=float)
    target_value = gain * float(np.mean(hip_flexion))
    H_q = np.zeros((1, nq), dtype=float)
    H_q[0, back_idx] = 1.0
    return (
        np.array([target_value], dtype=float),
        np.array([float(q[back_idx])], dtype=float),
        H_q,
        np.array([std_rad**2], dtype=float),
    )


def _sagittal_angle_deg(a, b):
    """Signed angle from ``a`` to ``b`` in the world x-z plane, positive when ``b`` leans toward +x."""

    return float(np.rad2deg(np.arctan2(a[2] * b[0] - a[0] * b[2], a[0] * b[0] + a[2] * b[2])))


def _geometric_back_and_hip_flexion_deg(model, q):
    """Anatomical back flexion (upper vs lower trunk) and mean hip flexion (thigh vs lower trunk), degrees."""

    names = [name.to_string() for name in model.markerNames()]
    markers = {name: model.marker(np.asarray(q, dtype=float), i).to_array() for i, name in enumerate(names)}
    hip_centre = 0.5 * (markers["left_hip"] + markers["right_hip"])
    shoulder_centre = 0.5 * (markers["left_shoulder"] + markers["right_shoulder"])
    knee_centre = 0.5 * (markers["left_knee"] + markers["right_knee"])
    lower = markers["mid_back"] - hip_centre
    back = _sagittal_angle_deg(lower, shoulder_centre - markers["mid_back"])
    hip = -_sagittal_angle_deg(-lower, knee_centre - hip_centre)
    return back, hip


def _scene(model, q_true):
    calibrations = synthetic_cameras(5)
    points_3d = model_keypoints_3d(model, q_true)
    keypoints = project_keypoints(points_3d, calibrations)
    pose_data, reconstruction = make_pose_and_reconstruction(keypoints, points_3d, calibrations)
    return calibrations, pose_data, reconstruction


def _ekf(tmp_path, model_variant, **kwargs):
    model = load_synthetic_model(tmp_path, model_variant)
    q = np.zeros((1, model.nbQ()))
    q[:, 2] = 1.0
    calibrations, pose_data, reconstruction = _scene(model, q)
    return MultiViewKinematicEKF(
        model=model,
        calibrations=calibrations,
        pose_data=pose_data,
        reconstruction=reconstruction,
        dt=1.0 / 120.0,
        **kwargs,
    )


def _random_q(ekf, seed):
    q = np.random.default_rng(seed).normal(scale=0.4, size=ekf.nq)
    q[2] += 1.0
    return q


def test_back_prior_mode_constants_and_normalization():
    assert DEFAULT_BACK_PRIOR_MODE == "legacy"
    assert SUPPORTED_BACK_PRIOR_MODES == ("legacy", "consistent")
    # The profile module duplicates the constants (no heavy import); both must stay in sync.
    assert reconstruction_profiles.DEFAULT_BACK_PRIOR_MODE == DEFAULT_BACK_PRIOR_MODE
    assert reconstruction_profiles.SUPPORTED_BACK_PRIOR_MODES == SUPPORTED_BACK_PRIOR_MODES
    assert normalize_back_prior_mode(None) == DEFAULT_BACK_PRIOR_MODE
    assert normalize_back_prior_mode(" Consistent ") == "consistent"
    with pytest.raises(ValueError):
        normalize_back_prior_mode("anatomical")


@pytest.mark.parametrize("model_variant", BACK_VARIANTS)
def test_legacy_block_is_bitwise_identical_to_historical_formula(tmp_path, model_variant):
    default = _ekf(tmp_path, model_variant)
    explicit = _ekf(tmp_path, model_variant, back_prior_mode="legacy")
    names = default.q_names
    segment = "UPPER_BACK" if model_variant.startswith("back_") else "LOWER_TRUNK"
    back_idx = names.index(f"{segment}:RotY")
    hip_indices = tuple(
        names.index(name) for name in ("LEFT_THIGH:RotY", "RIGHT_THIGH:RotY", "LEFT_THIGH:RotX", "RIGHT_THIGH:RotX")
    )
    assert default.back_prior_mode == "legacy"
    assert (default.upper_back_sagittal_idx, default.hip_flexion_indices) == (back_idx, hip_indices)
    for seed in range(5):
        q = _random_q(default, seed)
        expected = _legacy_reference_block(
            q, back_idx, hip_indices, default.nq, default.upper_back_sagittal_gain, default.upper_back_pseudo_std_rad
        )
        for ekf in (default, explicit):
            for got, ref in zip(ekf._upper_back_pseudo_measurement_block(q), expected):
                np.testing.assert_array_equal(got, ref)
    assert default.back_prior_description()["target_coefficient"] == GAIN
    assert default.back_prior_description()["full_jacobian"] is False


@pytest.mark.parametrize("model_variant", BACK_VARIANTS)
def test_consistent_target_flexes_the_back_with_the_hips(tmp_path, model_variant):
    legacy = _ekf(tmp_path, model_variant)
    consistent = _ekf(tmp_path, model_variant, back_prior_mode="consistent")
    model = consistent.model
    names = consistent.q_names
    back_idx = consistent.upper_back_sagittal_idx
    assert consistent.hip_flexion_indices == (names.index("LEFT_THIGH:RotY"), names.index("RIGHT_THIGH:RotY"))
    q = np.zeros(consistent.nq)
    q[2] = 1.0
    q[list(consistent.hip_flexion_indices)] = np.deg2rad(-60.0)  # thigh RotY < 0 = hip flexion
    _back, hip = _geometric_back_and_hip_flexion_deg(model, q)
    assert hip == pytest.approx(60.0, abs=1e-6)

    # Consistent: the back value zeroing the innovation is an anatomical flexion of g x 60 deg.
    z, h, H_q, variance = consistent._upper_back_pseudo_measurement_block(q)
    q_target = np.array(q, copy=True)
    q_target[back_idx] = q[back_idx] + float(z[0] - h[0]) / H_q[0, back_idx]
    back_target, _ = _geometric_back_and_hip_flexion_deg(model, q_target)
    assert back_target == pytest.approx(GAIN * 60.0, abs=1e-6)
    np.testing.assert_allclose(variance, [np.deg2rad(10.0) ** 2])

    # Legacy (documented defect): flexion gain halved by the RotX terms, and wrong side for the low-root variants.
    z_l, h_l, _H_l, _ = legacy._upper_back_pseudo_measurement_block(q)
    q_legacy = np.array(q, copy=True)
    q_legacy[back_idx] = q[back_idx] + float(z_l[0] - h_l[0])
    back_legacy, _ = _geometric_back_and_hip_flexion_deg(model, q_legacy)
    expected_sign = -1.0 if model_variant.startswith("back_") else 1.0
    assert back_legacy == pytest.approx(expected_sign * 0.5 * GAIN * 60.0, abs=1e-6)

    # Description exported in run_ekf / bundle summary.
    description = consistent.back_prior_description()
    assert description["mode"] == "consistent" and description["full_jacobian"] is True
    assert description["target_coefficient"] == pytest.approx(-GAIN if model_variant.startswith("back_") else GAIN)


@pytest.mark.parametrize("model_variant", ("back_3dof", "upper_root_back_3dof"))
def test_consistent_block_ignores_thigh_abduction(tmp_path, model_variant):
    ekf = _ekf(tmp_path, model_variant, back_prior_mode="consistent")
    names = ekf.q_names
    q = _random_q(ekf, 7)
    reference = ekf._upper_back_pseudo_measurement_block(q)
    for left, right in ((0.3, -0.3), (0.3, 0.3), (-0.5, 0.1)):
        moved = np.array(q, copy=True)
        moved[names.index("LEFT_THIGH:RotX")] += left
        moved[names.index("RIGHT_THIGH:RotX")] += right
        for got, ref in zip(ekf._upper_back_pseudo_measurement_block(moved), reference):
            np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("model_variant", ("back_3dof", "upper_root_back_flexion_1d"))
def test_consistent_block_is_independent_of_the_thigh_euler_branch(tmp_path, model_variant):
    """(RotY + pi, pi - RotX, RotZ + pi) is the same YXZ thigh rotation; a 2 pi turn of the back is no innovation."""

    ekf = _ekf(tmp_path, model_variant, back_prior_mode="consistent")
    names = ekf.q_names
    q = np.zeros(ekf.nq)
    q[2] = 1.0
    q[names.index("LEFT_THIGH:RotY")] = np.deg2rad(-100.0)
    q[names.index("RIGHT_THIGH:RotY")] = np.deg2rad(-80.0)
    q[names.index("LEFT_THIGH:RotX")] = np.deg2rad(10.0)
    q[ekf.upper_back_sagittal_idx] = np.deg2rad(5.0)
    other = np.array(q, copy=True)
    for side in ("LEFT", "RIGHT"):
        other[names.index(f"{side}_THIGH:RotY")] += np.pi
        other[names.index(f"{side}_THIGH:RotX")] = np.pi - other[names.index(f"{side}_THIGH:RotX")]
        other[names.index(f"{side}_THIGH:RotZ")] += np.pi
    other[ekf.upper_back_sagittal_idx] += 2.0 * np.pi
    np.testing.assert_allclose(
        model_keypoints_3d(ekf.model, other[None]), model_keypoints_3d(ekf.model, q[None]), atol=1e-9
    )
    reference = ekf._upper_back_pseudo_measurement_block(q)
    for got, ref in zip(ekf._upper_back_pseudo_measurement_block(other), reference):
        np.testing.assert_allclose(got, ref, atol=1e-12)
    # Mean hip flexion 90 deg -> target anatomical back flexion g x 90 deg.
    back_sign = 1.0 if model_variant.startswith("back_") else -1.0
    target = q[ekf.upper_back_sagittal_idx] - reference[1][0]
    assert back_sign * target == pytest.approx(np.deg2rad(GAIN * 90.0), abs=1e-12)


@pytest.mark.parametrize("model_variant", BACK_VARIANTS)
def test_consistent_jacobian_matches_finite_differences(tmp_path, model_variant):
    ekf = _ekf(tmp_path, model_variant, back_prior_mode="consistent")

    def residual(q):
        z, h, _H, _ = ekf._upper_back_pseudo_measurement_block(q)
        return float(h[0] - z[0])

    eps = 1e-6
    for seed in range(3):
        q = _random_q(ekf, seed)
        _z, _h, H_q, _ = ekf._upper_back_pseudo_measurement_block(q)
        numeric = np.array(
            [(residual(q + eps * e) - residual(q - eps * e)) / (2.0 * eps) for e in np.eye(ekf.nq)], dtype=float
        )
        np.testing.assert_allclose(H_q[0], numeric, atol=1e-8)
        assert np.count_nonzero(H_q[0]) == 3
    # Legacy target depends on the thighs but its H only holds the back DoF (incomplete linearisation).
    legacy = _ekf(tmp_path, model_variant)
    q = _random_q(legacy, 0)
    _z, _h, H_legacy, _ = legacy._upper_back_pseudo_measurement_block(q)
    assert np.count_nonzero(H_legacy[0]) == 1


def _flexion_trajectory(model, n_frames, model_variant):
    """Hips flex smoothly to 80 deg while the back follows g x hip flexion (anatomical)."""

    names = q_names_from_model(model)
    t = np.linspace(0.0, 1.0, n_frames)
    hip_flexion = np.deg2rad(80.0) * 0.5 * (1.0 - np.cos(np.pi * t))
    q = np.zeros((n_frames, model.nbQ()))
    q[:, 2] = 1.0
    q[:, names.index("LEFT_THIGH:RotY")] = -hip_flexion
    q[:, names.index("RIGHT_THIGH:RotY")] = -hip_flexion
    segment = "UPPER_BACK" if model_variant.startswith("back_") else "LOWER_TRUNK"
    back_sign = 1.0 if segment == "UPPER_BACK" else -1.0
    q[:, names.index(f"{segment}:RotY")] = back_sign * GAIN * hip_flexion
    return q


@pytest.mark.parametrize("model_variant", ("back_3dof", "upper_root_back_flexion_1d"))
def test_run_ekf_consistent_back_follows_imposed_flexion(tmp_path, model_variant):
    """Pelvis pitch vs back flexion is unobservable from the COCO17 markers: the prior sets the back posture.

    With a tight prior (2 deg) the consistent mode tracks the imposed anatomical flexion (g x hip flexion)
    while the legacy prior goes to extension (low root) or half the flexion (upper root); the marker
    errors are the same, which is the signature of the unobservable mode.
    """

    model = load_synthetic_model(tmp_path, model_variant)
    n_frames = 40
    q_true = _flexion_trajectory(model, n_frames, model_variant)
    calibrations, pose_data, reconstruction = _scene(model, q_true)
    nq = model.nbQ()
    kwargs = dict(
        biomod_path=None,
        calibrations=calibrations,
        pose_data=pose_data,
        reconstruction=reconstruction,
        fps=120.0,
        measurement_noise_scale=1.5,
        model=model,
        initial_state=np.concatenate((q_true[0], np.zeros(nq), np.zeros(nq))),
        upper_back_pseudo_std_rad=np.deg2rad(2.0),
    )
    consistent, _ = run_ekf(back_prior_mode="consistent", **kwargs)
    legacy, _ = run_ekf(**kwargs)

    assert consistent["back_prior_mode"] == "consistent" and legacy["back_prior_mode"] == "legacy"
    assert consistent["back_prior"]["hip_dofs"] == ["LEFT_THIGH:RotY", "RIGHT_THIGH:RotY"]
    assert consistent["back_prior"]["std_deg"] == pytest.approx(2.0)
    true_back = np.array([_geometric_back_and_hip_flexion_deg(model, q)[0] for q in q_true])
    est_back = np.array([_geometric_back_and_hip_flexion_deg(model, q)[0] for q in consistent["q"]])
    est_back_legacy = np.array([_geometric_back_and_hip_flexion_deg(model, q)[0] for q in legacy["q"]])
    assert true_back[-1] == pytest.approx(16.0, abs=1e-6)
    assert np.max(np.abs(est_back - true_back)) < 2.5
    if model_variant.startswith("back_"):
        assert est_back_legacy[-1] < 0.0
    else:
        assert est_back_legacy[-1] < true_back[-1] - 4.0
    markers_true = model_keypoints_3d(model, q_true)
    for result in (consistent, legacy):
        assert np.nanmean(np.linalg.norm(model_keypoints_3d(model, result["q"]) - markers_true, axis=-1)) < 0.01
    # Woodbury and innovation-space solvers stay equivalent with the full-H pseudo-observation.
    updates = []
    for update_method in ("woodbury", "legacy"):
        ekf = MultiViewKinematicEKF(
            model=model,
            calibrations=calibrations,
            pose_data=pose_data,
            reconstruction=reconstruction,
            dt=1.0 / 120.0,
            measurement_noise_scale=1.5,
            back_prior_mode="consistent",
            update_method=update_method,
        )
        state = np.concatenate((q_true[30] + 0.02, np.zeros(nq), np.zeros(nq)))
        predicted = ekf.predict(state, np.eye(ekf.nx) * 1e-2, 31)
        updates.append(ekf.update(*predicted, 31))
    assert updates[0][2] == updates[1][2] == "corrected"
    np.testing.assert_allclose(updates[0][0], updates[1][0], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(updates[0][1], updates[1][1], rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("model_variant", ("single_trunk", "back_3dof"))
def test_legacy_default_and_single_trunk_trajectories_are_unchanged(tmp_path, model_variant):
    model = load_synthetic_model(tmp_path, model_variant)
    nq = model.nbQ()
    q_true = _flexion_trajectory(model, 15, "back_3dof") if model_variant != "single_trunk" else None
    if q_true is None:
        names = q_names_from_model(model)
        q_true = np.zeros((15, nq))
        q_true[:, 2] = 1.0
        q_true[:, names.index("LEFT_THIGH:RotY")] = np.linspace(0.0, -1.0, 15)
    calibrations, pose_data, reconstruction = _scene(model, q_true)
    kwargs = dict(
        biomod_path=None,
        calibrations=calibrations,
        pose_data=pose_data,
        reconstruction=reconstruction,
        fps=120.0,
        model=model,
        initial_state=np.concatenate((q_true[0] + 0.01, np.zeros(nq), np.zeros(nq))),
    )
    default, _ = run_ekf(**kwargs)
    explicit, _ = run_ekf(back_prior_mode="legacy", **kwargs)
    for key in ("q", "qdot", "qddot"):
        np.testing.assert_array_equal(default[key], explicit[key])
    if model_variant == "single_trunk":
        consistent, _ = run_ekf(back_prior_mode="consistent", **kwargs)
        assert default["back_prior"] is None and consistent["back_prior"] is None
        for key in ("q", "qdot", "qddot"):
            np.testing.assert_array_equal(default[key], consistent[key])


def test_profile_back_prior_mode_is_named_validated_and_forwarded(tmp_path):
    profile = validate_profile(
        ReconstructionProfile(name="", family="ekf_2d", model_variant="back_3dof", back_prior_mode="Consistent")
    )
    cmd = build_pipeline_command(profile, tmp_path, Path("c.toml"), Path("k.json"))

    assert profile.back_prior_mode == "consistent"
    assert "bp_consistent" in profile.name
    assert cmd[cmd.index("--ekf2d-back-prior-mode") + 1] == "consistent"
    default = validate_profile(ReconstructionProfile(name="", family="ekf_2d", model_variant="back_3dof"))
    assert "bp_" not in default.name
    assert "--ekf2d-back-prior-mode" not in build_pipeline_command(default, tmp_path, Path("c.toml"), Path("k.json"))
    with pytest.raises(ValueError):
        validate_profile(ReconstructionProfile(name="", family="ekf_2d", back_prior_mode="anatomical"))
    other = validate_profile(ReconstructionProfile(name="", family="ekf_3d", back_prior_mode="consistent"))
    assert other.back_prior_mode == "legacy" and "bp_" not in other.name


def test_cli_and_bundle_expose_back_prior_mode(monkeypatch):
    import export_reconstruction_bundle

    argv = ["prog", "--name", "x", "--family", "ekf_2d", "--output-dir", "out"]
    monkeypatch.setattr(sys, "argv", argv)
    assert export_reconstruction_bundle.parse_args().ekf2d_back_prior_mode == "legacy"
    monkeypatch.setattr(sys, "argv", argv + ["--ekf2d-back-prior-mode", "consistent"])
    assert export_reconstruction_bundle.parse_args().ekf2d_back_prior_mode == "consistent"
    monkeypatch.setattr(sys, "argv", argv + ["--ekf2d-back-prior-mode", "anatomical"])
    with pytest.raises(SystemExit):
        export_reconstruction_bundle.parse_args()
    assert inspect.signature(build_ekf_2d_bundle).parameters["back_prior_mode"].default == DEFAULT_BACK_PRIOR_MODE
    for function in (
        vitpose_ekf_pipeline.run_ekf,
        vitpose_ekf_pipeline.compute_ekf2d_initial_state,
        vitpose_ekf_pipeline.initial_state_from_ekf_bootstrap,
        vitpose_ekf_pipeline.initial_state_from_root_pose_bootstrap,
        MultiViewKinematicEKF.__init__,
    ):
        assert inspect.signature(function).parameters["back_prior_mode"].default == DEFAULT_BACK_PRIOR_MODE
