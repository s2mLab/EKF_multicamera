"""Equivalence of the information-form (Woodbury) EKF2D update with the legacy solvers."""

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

import vitpose_ekf_pipeline
from vitpose_ekf_pipeline import (
    MultiViewKinematicEKF,
    apply_measurement_update_batch,
    apply_measurement_update_sequential,
    apply_measurement_update_woodbury,
    normalize_ekf2d_update_method,
    run_ekf,
    stack_measurement_blocks,
)

REL_TOL = 1e-9


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(actual - expected)) / max(float(np.max(np.abs(expected))), 1e-300))


def _random_problem(seed: int, nq: int = 8, block_sizes=(6, 4, 10)):
    rng = np.random.default_rng(seed)
    nx = 3 * nq
    base = rng.normal(size=(nx, nx))
    covariance = base @ base.T + 1e-3 * np.eye(nx)
    state = rng.normal(size=nx)
    blocks = []
    for size in block_sizes:
        blocks.append(
            (
                rng.normal(size=size) * 50.0,
                rng.normal(size=size) * 50.0,
                rng.normal(size=(size, nq)) * 300.0,
                4.0 + 20.0 * rng.random(size),
            )
        )
    return state, covariance, blocks, nq


def _all_solvers(state, covariance, blocks, nq):
    identity_x = np.eye(state.size)
    z, h, H_q, R = stack_measurement_blocks(blocks, nq)
    batch = apply_measurement_update_batch(state, covariance, z, h, H_q, R, nq, identity_x)
    sequential = apply_measurement_update_sequential(state, covariance, blocks, nq, identity_x)
    woodbury = apply_measurement_update_woodbury(state, covariance, z, h, H_q, R, nq)
    return batch, sequential, woodbury


def _assert_equivalent(result, reference):
    assert result is not None and reference is not None
    assert _relative_error(result[0], reference[0]) <= REL_TOL
    assert _relative_error(result[1], reference[1]) <= REL_TOL


@pytest.mark.parametrize("seed", range(6))
def test_woodbury_update_matches_batch_and_sequential_joseph_updates(seed):
    batch, sequential, woodbury = _all_solvers(*_random_problem(seed))

    _assert_equivalent(woodbury, batch)
    _assert_equivalent(woodbury, sequential)
    covariance = woodbury[1]
    np.testing.assert_array_equal(covariance, covariance.T)
    assert np.min(np.linalg.eigvalsh(covariance)) > 0.0


def test_woodbury_update_handles_locked_dofs_and_singular_q_covariance():
    state, covariance, blocks, nq = _random_problem(11)
    # Same pattern as ``_apply_lock_constraints``: zero cross-covariances, tiny variance.
    for q_idx in (2, 5):
        for block in (0, nq, 2 * nq):
            covariance[block + q_idx, :] = 0.0
            covariance[:, block + q_idx] = 0.0
            covariance[block + q_idx, block + q_idx] = 1e-9
    blocks = [(z, h, np.where(np.isin(np.arange(nq), (2, 5)), 0.0, H), r) for z, h, H, r in blocks]
    batch, sequential, woodbury = _all_solvers(state, covariance, blocks, nq)
    _assert_equivalent(woodbury, batch)
    _assert_equivalent(woodbury, sequential)

    # Rank-deficient (PSD, singular) q-block: the Woodbury solver never inverts P_qq.
    rng = np.random.default_rng(3)
    low_rank = rng.normal(size=(3 * nq, nq - 3))
    singular_covariance = low_rank @ low_rank.T
    z, h, H_q, R = stack_measurement_blocks(blocks, nq)
    identity_x = np.eye(3 * nq)
    reference = apply_measurement_update_batch(state, singular_covariance, z, h, H_q, R, nq, identity_x)
    woodbury = apply_measurement_update_woodbury(state, singular_covariance, z, h, H_q, R, nq)
    assert woodbury is not None
    assert _relative_error(woodbury[0], reference[0]) <= 1e-7
    assert _relative_error(woodbury[1], reference[1]) <= 1e-7


def test_woodbury_update_ignores_infinite_variance_and_nan_rows():
    state, covariance, blocks, nq = _random_problem(5)
    z, h, H_q, R = stack_measurement_blocks(blocks, nq)
    keep = np.ones(z.size, dtype=bool)
    keep[[1, 7]] = False
    keep[[3]] = False
    R_masked = np.array(R, copy=True)
    R_masked[[1, 7]] = np.inf
    z_masked = np.array(z, copy=True)
    z_masked[3] = np.nan

    reference = apply_measurement_update_batch(
        state, covariance, z[keep], h[keep], H_q[keep], R[keep], nq, np.eye(state.size)
    )
    woodbury = apply_measurement_update_woodbury(state, covariance, z_masked, h, H_q, R_masked, nq)

    _assert_equivalent(woodbury, reference)


def test_woodbury_update_rejects_degenerate_inputs():
    state, covariance, blocks, nq = _random_problem(1)
    z, h, H_q, R = stack_measurement_blocks(blocks, nq)
    R_bad = np.array(R, copy=True)
    R_bad[0] = 0.0

    assert apply_measurement_update_woodbury(state, covariance, z, h, H_q, R_bad, nq) is None
    assert apply_measurement_update_woodbury(state, covariance, z, h, H_q, np.full_like(R, np.inf), nq) is None
    assert apply_measurement_update_woodbury(state, covariance, z[:0], h[:0], H_q[:0], R[:0], nq) is None


def test_normalize_ekf2d_update_method():
    assert normalize_ekf2d_update_method(None) == "woodbury"
    assert normalize_ekf2d_update_method(" Legacy ") == "legacy"
    with pytest.raises(ValueError):
        normalize_ekf2d_update_method("cholesky")


def test_profile_update_method_is_validated_named_and_forwarded(tmp_path):
    from pathlib import Path

    from reconstruction.reconstruction_profiles import (
        ReconstructionProfile,
        build_pipeline_command,
        canonical_profile_name,
        validate_profile,
    )

    default_profile = validate_profile(ReconstructionProfile(name="", family="ekf_2d"))
    legacy_profile = validate_profile(ReconstructionProfile(name="", family="ekf_2d", ekf2d_update_method="legacy"))

    assert default_profile.ekf2d_update_method == "woodbury"
    assert canonical_profile_name(legacy_profile) == canonical_profile_name(default_profile) + "_upd_legacy"
    default_cmd = build_pipeline_command(default_profile, tmp_path, Path("calib.toml"), Path("kp.json"))
    legacy_cmd = build_pipeline_command(legacy_profile, tmp_path, Path("calib.toml"), Path("kp.json"))
    assert "--ekf2d-update-method" not in default_cmd
    assert legacy_cmd[legacy_cmd.index("--ekf2d-update-method") + 1] == "legacy"
    with pytest.raises(ValueError):
        validate_profile(ReconstructionProfile(name="", family="ekf_2d", ekf2d_update_method="qr"))


def _synthetic_scene(tmp_path, model_variant="single_trunk", n_frames=12, noise_px=1.5):
    model = load_synthetic_model(tmp_path, model_variant=model_variant)
    calibrations = synthetic_cameras(5)
    q_true = synthetic_q_trajectory(model, n_frames)
    points_3d = model_keypoints_3d(model, q_true)
    keypoints = project_keypoints(points_3d, calibrations)
    keypoints += np.random.default_rng(7).normal(scale=noise_px, size=keypoints.shape)
    # Excluded / missing observations: NaN keypoints and zero scores.
    keypoints[1, 3, 5] = np.nan
    scores = np.where(np.all(np.isfinite(keypoints), axis=-1), 0.9, 0.0)
    scores[2, :, 9] = 0.0
    pose_data, reconstruction = make_pose_and_reconstruction(keypoints, points_3d, calibrations, scores=scores)
    return model, calibrations, pose_data, reconstruction, q_true


@pytest.mark.parametrize(
    "model_variant,ankle_bed,dof_locking",
    [("single_trunk", False, False), ("back_3dof", True, True)],
)
def test_ekf_update_woodbury_matches_legacy_on_synthetic_model(tmp_path, model_variant, ankle_bed, dof_locking):
    model, calibrations, pose_data, reconstruction, q_true = _synthetic_scene(tmp_path, model_variant)
    common = dict(
        model=model,
        calibrations=calibrations,
        pose_data=pose_data,
        reconstruction=reconstruction,
        dt=1.0 / 120.0,
        measurement_noise_scale=1.5,
        ankle_bed_pseudo_obs=ankle_bed,
        enable_dof_locking=dof_locking,
    )
    ekf_legacy = MultiViewKinematicEKF(update_method="legacy", **common)
    ekf_woodbury = MultiViewKinematicEKF(update_method="woodbury", **common)
    state = np.concatenate((q_true[0] + 0.02, np.zeros(model.nbQ()), np.zeros(model.nbQ())))
    covariance = np.eye(ekf_legacy.nx) * 1e-2

    predicted_state, predicted_covariance = ekf_legacy.predict(state, covariance, 1)
    legacy = ekf_legacy.update(predicted_state, predicted_covariance, 1)
    woodbury = ekf_woodbury.update(predicted_state, predicted_covariance, 1)

    assert legacy[2] == woodbury[2] == "corrected"
    assert ekf_woodbury.solver_counts == {"woodbury": 1, "woodbury_fallback_legacy": 0}
    _assert_equivalent(woodbury[:2], legacy[:2])


def test_run_ekf_woodbury_trajectory_matches_legacy(tmp_path):
    model, calibrations, pose_data, reconstruction, q_true = _synthetic_scene(tmp_path, n_frames=15)
    initial_state = np.concatenate((q_true[0], np.zeros(model.nbQ()), np.zeros(model.nbQ())))
    kwargs = dict(
        biomod_path=None,
        calibrations=calibrations,
        pose_data=pose_data,
        reconstruction=reconstruction,
        fps=120.0,
        measurement_noise_scale=1.5,
        model=model,
        initial_state=initial_state,
    )
    legacy, _ = run_ekf(update_method="legacy", **kwargs)
    woodbury, _ = run_ekf(**kwargs)

    assert woodbury["update_method"] == "woodbury"
    assert woodbury["update_solver_counts"]["woodbury_fallback_legacy"] == 0
    assert legacy["update_solver_counts"]["woodbury"] == 0
    for key in ("q", "qdot", "qddot"):
        assert _relative_error(woodbury[key], legacy[key]) <= REL_TOL
    assert np.sqrt(np.mean((woodbury["q"][5:] - q_true[5:]) ** 2)) < 0.05


def test_ekf_update_woodbury_falls_back_to_legacy_solver(tmp_path, monkeypatch):
    model, calibrations, pose_data, reconstruction, q_true = _synthetic_scene(tmp_path, n_frames=4)
    ekf = MultiViewKinematicEKF(
        model=model,
        calibrations=calibrations,
        pose_data=pose_data,
        reconstruction=reconstruction,
        dt=1.0 / 120.0,
    )
    monkeypatch.setattr(vitpose_ekf_pipeline, "apply_measurement_update_woodbury", lambda **_kwargs: None)
    state = np.concatenate((q_true[0], np.zeros(model.nbQ()), np.zeros(model.nbQ())))
    predicted_state, predicted_covariance = ekf.predict(state, np.eye(ekf.nx) * 1e-2, 1)

    _state, _covariance, status = ekf.update(predicted_state, predicted_covariance, 1)

    assert status == "corrected"
    assert ekf.solver_counts == {"woodbury": 0, "woodbury_fallback_legacy": 1}
