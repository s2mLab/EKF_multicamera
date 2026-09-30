"""Opt-in robust inlier/outlier mixture on EKF2D keypoint measurements (``robust_mixture``)."""

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
from vitpose_ekf_pipeline import KP_INDEX, robust_mixture_measurement_variances, run_ekf


def _unit_problem(seed=0, m=6, nq=5):
    rng = np.random.default_rng(seed)
    H = rng.normal(size=(m, 2, nq)) * 200.0
    base = rng.normal(size=(nq, nq))
    P = 1e-4 * (base @ base.T) + 1e-5 * np.eye(nq)
    variances = 10.0 + 20.0 * rng.random(m)
    return H, P, variances


def test_mixture_weights_neutralize_outliers_and_keep_inliers():
    H, P, variances = _unit_problem()
    innovations = np.zeros((H.shape[0], 2))
    innovations[:, 0] = 1.0
    innovations[2] = [400.0, -300.0]

    inflated, weights = robust_mixture_measurement_variances(innovations, H, P, variances, outlier_prob=0.03)

    inliers = np.arange(H.shape[0]) != 2
    assert np.all(weights[inliers] > 0.999)
    assert weights[2] < 1e-3
    np.testing.assert_allclose(inflated[inliers], np.repeat(variances[inliers, None], 2, axis=1), rtol=1e-3)
    assert np.all(inflated[2] > 1e3 * variances[2])
    assert np.all(inflated >= variances[:, None]) and np.all(np.isfinite(inflated))


def test_mixture_variances_are_row_wise_and_order_independent():
    H, P, variances = _unit_problem(seed=4)
    innovations = np.random.default_rng(5).normal(scale=30.0, size=(H.shape[0], 2))
    perm = np.random.default_rng(6).permutation(H.shape[0])

    inflated, weights = robust_mixture_measurement_variances(innovations, H, P, variances)
    inflated_perm, weights_perm = robust_mixture_measurement_variances(innovations[perm], H[perm], P, variances[perm])

    np.testing.assert_allclose(inflated_perm, inflated[perm], rtol=1e-14)
    np.testing.assert_allclose(weights_perm, weights[perm], rtol=1e-14)
    # Matches S/w - HPH^T on the diagonal.
    hph = np.einsum("mai,ij,mbj->mab", H, P, H)
    S = hph + variances[:, None, None] * np.eye(2)
    expected = np.stack([np.diag(S[k] / weights[k] - hph[k]) for k in range(H.shape[0])])
    np.testing.assert_allclose(inflated, np.maximum(expected, variances[:, None]), rtol=1e-10)


def _scene(tmp_path, n_frames=24, outlier=True):
    model = load_synthetic_model(tmp_path)
    q_true = synthetic_q_trajectory(model, n_frames, seed=8)
    calibrations = synthetic_cameras(5)
    points_3d = model_keypoints_3d(model, q_true)
    keypoints = project_keypoints(points_3d, calibrations)
    keypoints += np.random.default_rng(9).normal(scale=1.0, size=keypoints.shape)
    if outlier:
        # Gross detector errors: two keypoints jump by hundreds of pixels in one camera.
        keypoints[1, 8:16, KP_INDEX["left_wrist"]] += np.array([350.0, -250.0])
        keypoints[3, 10:14, KP_INDEX["right_knee"]] += np.array([-300.0, 200.0])
    pose_data, reconstruction = make_pose_and_reconstruction(keypoints, points_3d, calibrations)
    return model, calibrations, pose_data, reconstruction, q_true


def _run(model, calibrations, pose_data, reconstruction, q_true, **kwargs):
    nq = model.nbQ()
    return run_ekf(
        None,
        calibrations,
        pose_data,
        reconstruction,
        fps=120.0,
        measurement_noise_scale=1.5,
        model=model,
        initial_state=np.concatenate((q_true[0], np.zeros(nq), np.zeros(nq))),
        **kwargs,
    )[0]


def _marker_error(model, q, q_true, frames):
    diff = model_keypoints_3d(model, q[frames]) - model_keypoints_3d(model, q_true[frames])
    return float(np.nanmean(np.linalg.norm(diff, axis=-1)))


def test_robust_mixture_neutralizes_injected_outliers_on_both_solvers(tmp_path):
    scene = _scene(tmp_path)
    model, q_true = scene[0], scene[4]
    baseline = _run(*scene)
    robust = _run(*scene, robust_mixture=True)
    robust_legacy = _run(*scene, robust_mixture=True, update_method="legacy")

    frames = np.arange(8, 16)
    assert baseline["robust_mixture_stats"] is None
    assert robust["robust_mixture_stats"]["downweighted_below_0_5"] >= 12
    assert _marker_error(model, robust["q"], q_true, frames) < 0.5 * _marker_error(model, baseline["q"], q_true, frames)
    np.testing.assert_allclose(robust["q"], robust_legacy["q"], atol=1e-8)


def test_robust_mixture_is_nearly_neutral_without_outliers_and_camera_order_free(tmp_path):
    model, calibrations, pose_data, reconstruction, q_true = _scene(tmp_path, outlier=False)
    baseline = _run(model, calibrations, pose_data, reconstruction, q_true)
    robust = _run(model, calibrations, pose_data, reconstruction, q_true, robust_mixture=True)

    assert robust["robust_mixture_stats"]["downweighted_below_0_5"] == 0
    assert np.max(np.abs(robust["q"] - baseline["q"])) < 1e-4

    order = [3, 0, 4, 1, 2]
    names = [pose_data.camera_names[i] for i in order]
    permuted_pose, _ = make_pose_and_reconstruction(
        pose_data.keypoints[order], reconstruction.points_3d, {name: calibrations[name] for name in names}
    )
    permuted_reconstruction = reconstruction.__class__(
        **{
            **reconstruction.__dict__,
            "multiview_coherence": reconstruction.multiview_coherence[..., order],
        }
    )
    permuted = _run(model, calibrations, permuted_pose, permuted_reconstruction, q_true, robust_mixture=True)
    np.testing.assert_allclose(permuted["q"], robust["q"], atol=1e-9)


def test_profile_robust_mixture_is_named_and_forwarded(tmp_path):
    profile = validate_profile(
        ReconstructionProfile(name="", family="ekf_2d", robust_mixture=True, robust_mixture_outlier_prob=0.05)
    )
    cmd = build_pipeline_command(profile, tmp_path, Path("c.toml"), Path("k.json"))

    assert profile.name == "ekf_2d_acc_robust_po0_05"
    assert "--ekf2d-robust-mixture" in cmd
    assert cmd[cmd.index("--ekf2d-robust-outlier-prob") + 1] == "0.05"
    default = validate_profile(ReconstructionProfile(name="", family="ekf_2d"))
    assert "--ekf2d-robust-mixture" not in build_pipeline_command(default, tmp_path, Path("c.toml"), Path("k.json"))
    with pytest.raises(ValueError):
        validate_profile(ReconstructionProfile(name="", family="ekf_2d", robust_mixture_outlier_prob=1.0))
