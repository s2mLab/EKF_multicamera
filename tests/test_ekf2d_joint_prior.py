"""Opt-in elbow/knee joint prior (``joint_prior``): mirror branch, sign limits, axial prior."""

from pathlib import Path

import numpy as np
import pytest
from _ekf2d_synthetic import (
    load_synthetic_model,
    make_pose_and_reconstruction,
    model_keypoints_3d,
    project_keypoints,
    synthetic_cameras,
    synthetic_q_trajectory,
)

from reconstruction.reconstruction_profiles import ReconstructionProfile, build_pipeline_command, validate_profile
from vitpose_ekf_pipeline import (
    MultiViewKinematicEKF,
    canonicalize_joint_mirror_branches,
    joint_mirror_pair_indices,
    q_names_from_model,
    run_ekf,
    wrap_to_pi,
)


def _canonical_trajectory(model, n_frames: int) -> np.ndarray:
    q = synthetic_q_trajectory(model, n_frames, seed=3)
    for axial_idx, flex_idx, sign in joint_mirror_pair_indices(q_names_from_model(model)):
        q[:, flex_idx] = sign * (0.6 + 0.1 * np.sin(np.arange(n_frames) / 10.0))
        q[:, axial_idx] = 0.2 * np.cos(np.arange(n_frames) / 15.0)
    return q


def _mirror(q: np.ndarray, pairs) -> np.ndarray:
    mirrored = np.array(q, copy=True)
    for axial_idx, flex_idx, _sign in pairs:
        mirrored[..., axial_idx] += np.pi
        mirrored[..., flex_idx] *= -1.0
    return mirrored


def test_mirror_branch_is_an_exact_marker_symmetry_and_is_canonicalized(tmp_path):
    model = load_synthetic_model(tmp_path)
    q_names = q_names_from_model(model)
    pairs = joint_mirror_pair_indices(q_names)
    assert len(pairs) == 4
    q = _canonical_trajectory(model, 5)
    q_mirror = _mirror(q, pairs)

    np.testing.assert_allclose(model_keypoints_3d(model, q_mirror), model_keypoints_3d(model, q), atol=1e-12)

    qdot = np.ones_like(q)
    q_canonical, qdot_canonical, qddot_canonical = canonicalize_joint_mirror_branches(q_names, q_mirror, qdot, None)
    np.testing.assert_allclose(q_canonical, q, atol=1e-12)
    assert qddot_canonical is None
    for _axial_idx, flex_idx, _sign in pairs:
        np.testing.assert_array_equal(qdot_canonical[:, flex_idx], -1.0)
    # Already canonical input is unchanged (idempotent).
    np.testing.assert_allclose(canonicalize_joint_mirror_branches(q_names, q)[0], q, atol=1e-15)


def _ekf_scene(tmp_path, n_frames=20):
    model = load_synthetic_model(tmp_path)
    q_true = _canonical_trajectory(model, n_frames)
    calibrations = synthetic_cameras(5)
    points_3d = model_keypoints_3d(model, q_true)
    keypoints = project_keypoints(points_3d, calibrations)
    keypoints += np.random.default_rng(2).normal(scale=1.0, size=keypoints.shape)
    pose_data, reconstruction = make_pose_and_reconstruction(keypoints, points_3d, calibrations)
    return model, calibrations, pose_data, reconstruction, q_true


def test_joint_limit_constraints_reflect_mirror_branch_and_keep_covariance_spd(tmp_path):
    model, calibrations, pose_data, reconstruction, q_true = _ekf_scene(tmp_path, n_frames=3)
    ekf = MultiViewKinematicEKF(
        model=model,
        calibrations=calibrations,
        pose_data=pose_data,
        reconstruction=reconstruction,
        dt=1.0 / 120.0,
        joint_prior=True,
    )
    nq = ekf.nq
    rng = np.random.default_rng(0)
    base = rng.normal(size=(3 * nq, 3 * nq))
    covariance = 1e-3 * (base @ base.T) + 1e-4 * np.eye(3 * nq)
    state = np.concatenate((_mirror(q_true[0], ekf.joint_mirror_pairs), rng.normal(size=nq), rng.normal(size=nq)))

    # 1) Mirror branch: exact reflection (T P T^T), no projection needed.
    reflected_state, reflected_covariance = ekf._apply_joint_limit_constraints(state, covariance)

    assert ekf.joint_prior_counts == {"axial_prior_blocks": 0, "mirror_reflections": 4, "limit_projections": 0}
    np.testing.assert_allclose(reflected_state[:nq], q_true[0], atol=1e-12)
    signs = np.ones(3 * nq)
    for _axial_idx, flex_idx, _sign in ekf.joint_mirror_pairs:
        signs[[flex_idx, nq + flex_idx, 2 * nq + flex_idx]] = -1.0
    np.testing.assert_allclose(reflected_state[nq:], (signs * state)[nq:], atol=0.0)
    np.testing.assert_allclose(reflected_covariance, signs[:, None] * covariance * signs[None, :], atol=0.0)

    # 2) Small violation (inside the reflection margin): inequality pseudo-observation only.
    (_left_axial, left_elbow, _), *_rest = ekf.joint_mirror_pairs
    small = np.array(reflected_state, copy=True)
    small[left_elbow] = 0.005
    projected_state, projected_covariance = ekf._apply_joint_limit_constraints(small, reflected_covariance)

    assert ekf.joint_prior_counts["mirror_reflections"] == 4
    assert ekf.joint_prior_counts["limit_projections"] == 1
    assert projected_state[left_elbow] < -np.deg2rad(0.5)
    for axial_idx, flex_idx, sign in ekf.joint_mirror_pairs:
        assert sign * projected_state[flex_idx] > 0.0
        assert -np.pi <= projected_state[axial_idx] < np.pi
    np.testing.assert_allclose(projected_covariance, projected_covariance.T, atol=0.0)
    assert np.min(np.linalg.eigvalsh(projected_covariance)) > 0.0
    assert np.all(np.diag(projected_covariance) <= np.diag(reflected_covariance) + 1e-15)
    assert wrap_to_pi(0.0) == pytest.approx(0.0)


def test_run_ekf_joint_prior_recovers_canonical_branch_from_mirrored_start(tmp_path):
    model, calibrations, pose_data, reconstruction, q_true = _ekf_scene(tmp_path)
    nq = model.nbQ()
    pairs = joint_mirror_pair_indices(q_names_from_model(model))
    mirrored_start = np.concatenate((_mirror(q_true[0], pairs), np.zeros(nq), np.zeros(nq)))
    kwargs = dict(
        biomod_path=None,
        calibrations=calibrations,
        pose_data=pose_data,
        reconstruction=reconstruction,
        fps=120.0,
        measurement_noise_scale=1.5,
        model=model,
        initial_state=mirrored_start,
    )

    baseline, _ = run_ekf(**kwargs)
    prior_woodbury, _ = run_ekf(joint_prior=True, joint_prior_axial_std_deg=45.0, **kwargs)
    prior_legacy, _ = run_ekf(joint_prior=True, joint_prior_axial_std_deg=45.0, update_method="legacy", **kwargs)

    assert baseline["joint_prior_counts"] is None
    assert prior_woodbury["joint_prior_counts"]["axial_prior_blocks"] > 0
    for axial_idx, flex_idx, sign in pairs:
        # Without the option the filter stays in the mirror branch; with it, the output is canonical.
        assert np.all(sign * baseline["q"][:, flex_idx] < 0.0)
        assert np.all(sign * prior_woodbury["q"][:, flex_idx] > 0.0)
        assert np.max(np.abs(prior_woodbury["q"][:, axial_idx])) < np.pi / 2
    np.testing.assert_allclose(prior_woodbury["q"], prior_legacy["q"], atol=1e-8)
    markers_prior = model_keypoints_3d(model, prior_woodbury["q"])
    markers_true = model_keypoints_3d(model, q_true)
    assert np.nanmean(np.linalg.norm(markers_prior - markers_true, axis=-1)[5:]) < 0.03


def test_profile_joint_prior_is_named_and_forwarded(tmp_path):
    profile = validate_profile(
        ReconstructionProfile(name="", family="ekf_2d", joint_prior=True, joint_prior_axial_std_deg=45.0)
    )
    cmd = build_pipeline_command(profile, tmp_path, Path("c.toml"), Path("k.json"))

    assert "jprior_ax45" in profile.name
    assert "--ekf2d-joint-prior" in cmd
    assert cmd[cmd.index("--ekf2d-joint-prior-axial-std-deg") + 1] == "45.0"
    default = validate_profile(ReconstructionProfile(name="", family="ekf_2d"))
    assert "--ekf2d-joint-prior" not in build_pipeline_command(default, tmp_path, Path("c.toml"), Path("k.json"))
    with pytest.raises(ValueError):
        validate_profile(ReconstructionProfile(name="", family="ekf_2d", joint_prior_axial_std_deg=0.0))
