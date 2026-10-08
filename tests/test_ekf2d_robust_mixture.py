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
from vitpose_ekf_pipeline import (
    KP_INDEX,
    initial_state_from_ekf_bootstrap,
    robust_mixture_keypoint_guard,
    robust_mixture_lock_decision,
    robust_mixture_measurement_variances,
    run_ekf,
    validate_robust_mixture_lock_fractions,
)


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
    assert cmd[cmd.index("--ekf2d-robust-lock-fraction") + 1] == "0.5"
    assert cmd[cmd.index("--ekf2d-robust-resume-fraction") + 1] == "0.25"
    default = validate_profile(ReconstructionProfile(name="", family="ekf_2d"))
    assert "--ekf2d-robust-mixture" not in build_pipeline_command(default, tmp_path, Path("c.toml"), Path("k.json"))
    with pytest.raises(ValueError):
        validate_profile(ReconstructionProfile(name="", family="ekf_2d", robust_mixture_outlier_prob=1.0))


def test_lock_decision_hysteresis():
    def weights(n_low, n=20):
        return np.r_[np.full(n_low, 0.01), np.ones(n - n_low)]

    # Active mixture: suspended only above the lock fraction (strictly).
    assert robust_mixture_lock_decision(weights(10), False, 0.5, 0.25) == (False, 0.5)
    assert robust_mixture_lock_decision(weights(11), False, 0.5, 0.25) == (True, 0.55)
    # Suspended mixture: resumes only at or below the resume fraction.
    assert robust_mixture_lock_decision(weights(6), True, 0.5, 0.25) == (True, 0.3)
    assert robust_mixture_lock_decision(weights(5), True, 0.5, 0.25) == (False, 0.25)
    # No weights: the mode is kept.
    suspended, fraction = robust_mixture_lock_decision(np.empty(0), True, 0.5, 0.25)
    assert suspended and np.isnan(fraction)
    # lock_fraction = resume_fraction = 1 restores the unguarded mixture (never suspended after the first frame).
    assert robust_mixture_lock_decision(weights(20), False, 1.0, 1.0) == (False, 1.0)
    assert robust_mixture_lock_decision(weights(20), True, 1.0, 1.0) == (False, 1.0)


def test_lock_fractions_are_validated():
    assert validate_robust_mixture_lock_fractions(0.5, 0.25) == (0.5, 0.25)
    for lock, resume in ((0.0, 0.0), (1.5, 0.2), (0.5, 0.6), (0.5, -0.1)):
        with pytest.raises(ValueError):
            validate_robust_mixture_lock_fractions(lock, resume)
    with pytest.raises(ValueError):
        validate_profile(
            ReconstructionProfile(
                name="",
                family="ekf_2d",
                robust_mixture=True,
                robust_mixture_lock_fraction=0.2,
                robust_mixture_resume_fraction=0.3,
            )
        )
    profile = validate_profile(
        ReconstructionProfile(name="", family="ekf_2d", robust_mixture=True, robust_mixture_lock_fraction=0.6)
    )
    assert profile.name == "ekf_2d_acc_robust_lk0_6"


def _offset_initial_state(model, q_true, offset_m=1.5, rotation_rad=(0.75, 0.5, -1.0)):
    nq = model.nbQ()
    q0 = np.array(q_true[0], copy=True)
    q0[:3] += np.array([1.0, -1.0, 0.5]) * offset_m
    q0[3:6] += np.asarray(rotation_rad)
    return np.concatenate((q0, np.zeros(nq), np.zeros(nq)))


def test_lock_guard_recovers_from_a_wrong_initial_state_with_white_jerk(tmp_path):
    """Miniature of the real divergence: wrong q0 + small white-jerk Q + consistent detections.

    Without the guard, the mixture rejects the detections that disagree with the wrong prediction and
    the filter stays far from the truth; with the guard it is updated like the Gaussian filter until the
    mixture agrees with the prediction, then the mixture resumes.
    """

    model, calibrations, pose_data, reconstruction, q_true = _scene(tmp_path, n_frames=24, outlier=False)
    x0 = _offset_initial_state(model, q_true)
    common = dict(initial_state=x0, process_noise_model="white_jerk")
    frames = np.arange(10, 20)

    def run(**kwargs):
        return run_ekf(
            None,
            calibrations,
            pose_data,
            reconstruction,
            fps=120.0,
            measurement_noise_scale=1.5,
            model=model,
            **{**common, **kwargs},
        )[0]

    gaussian = run()
    unguarded = run(robust_mixture=True, robust_mixture_lock_fraction=1.0, robust_mixture_resume_fraction=1.0)
    guarded = run(robust_mixture=True)
    guarded_legacy = run(robust_mixture=True, update_method="legacy")

    assert _marker_error(model, unguarded["q"], q_true, frames) > 0.3
    assert unguarded["robust_mixture_stats"]["suspended_frames"] == 0
    assert _marker_error(model, guarded["q"], q_true, frames) < 0.05
    stats = guarded["robust_mixture_stats"]
    assert 1 <= stats["suspended_frames"] < 10
    assert stats["applied_downweighted_below_0_5"] < stats["downweighted_below_0_5"]
    # Once the mixture resumes, it behaves like the Gaussian filter on outlier-free data.
    assert (
        _marker_error(model, guarded["q"], q_true, frames) < _marker_error(model, gaussian["q"], q_true, frames) + 0.01
    )
    np.testing.assert_allclose(guarded["q"], guarded_legacy["q"], atol=1e-8)


def test_lock_guard_keeps_the_outlier_rejection_and_is_inert_without_mixture(tmp_path):
    scene = _scene(tmp_path)
    model, q_true = scene[0], scene[4]
    baseline = _run(*scene)
    robust = _run(*scene, robust_mixture=True)
    frames = np.arange(8, 16)
    stats = robust["robust_mixture_stats"]
    assert stats["suspended_frames"] == 0 and stats["lock_events"] == 0
    assert stats["applied_downweighted_below_0_5"] == stats["downweighted_below_0_5"] >= 12
    assert _marker_error(model, robust["q"], q_true, frames) < 0.5 * _marker_error(model, baseline["q"], q_true, frames)
    # Without the mixture the guard parameters are ignored (bit-identical states).
    other = _run(*scene, robust_mixture_lock_fraction=0.1, robust_mixture_resume_fraction=0.0)
    np.testing.assert_array_equal(other["q"], baseline["q"])
    assert other["robust_mixture_stats"] is None


def test_bootstrap_is_not_locked_by_the_mixture(tmp_path):
    model, calibrations, pose_data, reconstruction, q_true = _scene(tmp_path, n_frames=4, outlier=False)
    x0 = _offset_initial_state(model, q_true, offset_m=1.5, rotation_rad=(0.45, 0.3, -0.6))

    def boot(**kwargs):
        state, diagnostics = initial_state_from_ekf_bootstrap(
            model,
            calibrations,
            pose_data,
            reconstruction,
            fps=120.0,
            measurement_noise_scale=1.5,
            initial_state=x0,
            passes=5,
            **kwargs,
        )
        return state, diagnostics

    gaussian, _ = boot()
    guarded, _ = boot(robust_mixture=True)
    unguarded, _ = boot(robust_mixture=True, robust_mixture_lock_fraction=1.0, robust_mixture_resume_fraction=1.0)

    def error(state):
        return _marker_error(model, state[np.newaxis, : model.nbQ()], q_true[:1], np.array([0]))

    assert error(gaussian) < 0.01
    assert error(guarded) == pytest.approx(error(gaussian), abs=1e-3)
    assert error(unguarded) > 0.1


def test_keypoint_guard_restores_keypoints_rejected_in_most_views():
    keypoints = np.array([0, 0, 0, 0, 5, 5, 5, 9, 12])
    weights = np.array([0.01, 0.02, 0.9, 0.03, 0.01, 0.9, 0.95, 0.01, 0.9])
    restore = robust_mixture_keypoint_guard(weights, keypoints, lock_fraction=0.5)
    # kp 0: 3/4 views rejected -> restored ; kp 5: 1/3 -> single-view outlier kept ;
    # kp 9: one view only -> kept ; kp 12: inlier.
    np.testing.assert_array_equal(restore, [True, True, True, True, False, False, False, False, False])
    perm = np.random.default_rng(0).permutation(weights.size)
    np.testing.assert_array_equal(robust_mixture_keypoint_guard(weights[perm], keypoints[perm]), restore[perm])
    assert not np.any(robust_mixture_keypoint_guard(np.zeros(4), np.zeros(4, dtype=int), lock_fraction=1.0))
    with pytest.raises(ValueError):
        robust_mixture_keypoint_guard(np.zeros(3), np.zeros(2, dtype=int))


def test_keypoint_guard_unlocks_a_limb_started_in_a_wrong_pose(tmp_path):
    """Partial lock: one thigh starts 1.2 rad off; its keypoints are rejected in every view."""

    model, calibrations, pose_data, reconstruction, q_true = _scene(tmp_path, n_frames=24, outlier=False)
    nq = model.nbQ()
    names = [
        f"{model.segment(i).name().to_string()}:{model.segment(i).nameDof(j).to_string()}"
        for i in range(model.nbSegment())
        for j in range(model.segment(i).nbDof())
    ]
    q0 = np.array(q_true[0], copy=True)
    q0[names.index("RIGHT_THIGH:RotY")] += 1.2
    x0 = np.concatenate((q0, np.zeros(nq), np.zeros(nq)))
    frames = np.arange(10, 20)

    def run(**kwargs):
        return run_ekf(
            None,
            calibrations,
            pose_data,
            reconstruction,
            fps=120.0,
            measurement_noise_scale=1.5,
            model=model,
            initial_state=x0,
            process_noise_model="white_jerk",
            **kwargs,
        )[0]

    gaussian = run()
    unguarded = run(robust_mixture=True, robust_mixture_lock_fraction=1.0, robust_mixture_resume_fraction=1.0)
    guarded = run(robust_mixture=True)

    assert _marker_error(model, unguarded["q"], q_true, frames) > 0.04
    assert (
        _marker_error(model, guarded["q"], q_true, frames) < _marker_error(model, gaussian["q"], q_true, frames) + 1e-3
    )
    stats = guarded["robust_mixture_stats"]
    assert stats["suspended_frames"] == 0
    assert stats["keypoint_guard_restored"] > 0
    assert unguarded["robust_mixture_stats"]["keypoint_guard_restored"] == 0
