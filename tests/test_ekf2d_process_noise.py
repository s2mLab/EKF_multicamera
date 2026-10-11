"""Opt-in white-jerk process noise for the EKF2D (``process_noise_model='white_jerk'``)."""

from pathlib import Path

import numpy as np
import pytest
from _ekf2d_synthetic import load_synthetic_model, make_pose_and_reconstruction, synthetic_cameras

from reconstruction.reconstruction_profiles import ReconstructionProfile, build_pipeline_command, validate_profile
from vitpose_ekf_pipeline import (
    MultiViewKinematicEKF,
    legacy_process_noise,
    normalize_process_noise_model,
    white_jerk_noise_block,
    white_jerk_process_noise,
)


def _transition(dt: float) -> np.ndarray:
    return np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize("dt", [1.0 / 120.0, 1.0 / 40.0, 0.5])
def test_white_jerk_block_is_the_exact_discretization_of_continuous_white_jerk(dt):
    # Q = int_0^dt Phi(s) b b^T Phi(s)^T ds with b = [0, 0, 1], by Gauss-Legendre quadrature (exact for polynomials).
    nodes, weights = np.polynomial.legendre.leggauss(8)
    s_values = 0.5 * dt * (nodes + 1.0)
    b = np.array([0.0, 0.0, 1.0])
    reference = sum(0.5 * dt * w * np.outer(_transition(s) @ b, _transition(s) @ b) for s, w in zip(s_values, weights))

    np.testing.assert_allclose(white_jerk_noise_block(dt), reference, rtol=1e-12, atol=0.0)


def test_white_jerk_block_scales_with_dt_and_is_spd():
    dt = 1.0 / 120.0
    block = white_jerk_noise_block(dt)
    doubled = white_jerk_noise_block(2.0 * dt)
    powers = np.array([[5, 4, 3], [4, 3, 2], [3, 2, 1]])

    np.testing.assert_allclose(doubled, block * 2.0**powers, rtol=1e-14)
    np.testing.assert_array_equal(block, block.T)
    assert np.all(np.linalg.eigvalsh(block) > 0.0)
    np.linalg.cholesky(block)
    with pytest.raises(ValueError):
        white_jerk_noise_block(0.0)


def test_white_jerk_process_noise_groups_root_translation_rotation_and_joints():
    q_names = [
        "PELVIS:TransX",
        "PELVIS:TransY",
        "PELVIS:TransZ",
        "PELVIS:RotY",
        "PELVIS:RotX",
        "PELVIS:RotZ",
        "ARM:RotY",
    ]
    nq = len(q_names)
    dt = 1.0 / 60.0
    Q = white_jerk_process_noise(dt, q_names, n_root=6, jerk_psd=(2.0, 3.0, 5.0), process_noise_scale=1.5)
    block = white_jerk_noise_block(dt)

    assert Q.shape == (3 * nq, 3 * nq)
    np.testing.assert_array_equal(Q, Q.T)
    np.linalg.cholesky(Q)
    for dof_idx, psd in enumerate([2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 5.0]):
        idx = [dof_idx, nq + dof_idx, 2 * nq + dof_idx]
        np.testing.assert_allclose(Q[np.ix_(idx, idx)], 1.5 * psd * block, rtol=1e-14)
    # No coupling between different DoFs.
    assert Q[0, nq + 1] == 0.0 and Q[3, 2 * nq + 6] == 0.0
    with pytest.raises(ValueError):
        white_jerk_process_noise(dt, q_names, n_root=6, jerk_psd=(1.0, -1.0, 1.0))


def test_default_jerk_psd_is_the_calibrated_value():
    from vitpose_ekf_pipeline import DEFAULT_PROCESS_NOISE_JERK_PSD, DEFAULT_PROCESS_NOISE_MODEL

    assert DEFAULT_PROCESS_NOISE_MODEL == "legacy"
    assert DEFAULT_PROCESS_NOISE_JERK_PSD == (200.0, 1000.0, 10000.0)
    q_names = ["PELVIS:TransZ", "PELVIS:RotY", "ARM:RotY"]
    Q = white_jerk_process_noise(0.01, q_names, n_root=2)
    np.testing.assert_allclose(np.diag(Q)[6:], [200.0 * 0.01, 1000.0 * 0.01, 10000.0 * 0.01])


def test_normalize_process_noise_model():
    assert normalize_process_noise_model(None) == "legacy"
    assert normalize_process_noise_model("White_Jerk") == "white_jerk"
    with pytest.raises(ValueError):
        normalize_process_noise_model("singer")


def test_ekf_default_process_noise_is_unchanged_and_white_jerk_is_opt_in(tmp_path):
    model = load_synthetic_model(tmp_path)
    calibrations = synthetic_cameras(3)
    nq = model.nbQ()
    keypoints = np.full((3, 2, 17, 2), np.nan)
    pose_data, reconstruction = make_pose_and_reconstruction(keypoints, np.full((2, 17, 3), np.nan), calibrations)
    common = dict(
        model=model,
        calibrations=calibrations,
        pose_data=pose_data,
        reconstruction=reconstruction,
        dt=1.0 / 120.0,
        process_noise_scale=2.0,
    )

    default_ekf = MultiViewKinematicEKF(**common)
    expected_legacy = np.diag(
        np.concatenate((1e-4 * np.ones(nq), 5e-3 * np.ones(nq), 5e-2 * np.ones(nq))) * 2.0
    )  # historical formula
    np.testing.assert_array_equal(default_ekf.process_noise, expected_legacy)
    np.testing.assert_array_equal(legacy_process_noise(nq, 2.0), expected_legacy)

    jerk_ekf = MultiViewKinematicEKF(process_noise_model="white_jerk", process_noise_jerk_psd=(1.0, 2.0, 4.0), **common)
    expected = white_jerk_process_noise(1.0 / 120.0, jerk_ekf.q_names, jerk_ekf.n_root, (1.0, 2.0, 4.0), 2.0)
    np.testing.assert_array_equal(jerk_ekf.process_noise, expected)
    assert jerk_ekf.n_root == 6
    assert jerk_ekf.process_noise[2, 2] == pytest.approx(2.0 * 1.0 * (1.0 / 120.0) ** 5 / 20.0)
    assert jerk_ekf.process_noise[2 * nq + 6, 2 * nq + 6] == pytest.approx(2.0 * 4.0 / 120.0)

    state = np.zeros(3 * nq)
    _pred, predicted_covariance = jerk_ekf.predict(state, np.eye(3 * nq) * 1e-2, 1)
    F = jerk_ekf.transition_matrix()
    np.testing.assert_allclose(predicted_covariance, F @ (np.eye(3 * nq) * 1e-2) @ F.T + expected, rtol=1e-12)


def test_profile_process_noise_model_is_named_and_forwarded(tmp_path):
    profile = validate_profile(
        ReconstructionProfile(
            name="", family="ekf_2d", process_noise_model="white_jerk", process_noise_jerk_psd=[1.0, 2.0, 4.0]
        )
    )
    cmd = build_pipeline_command(profile, tmp_path, Path("c.toml"), Path("k.json"))

    assert "qjerk" in profile.name
    assert cmd[cmd.index("--process-noise-model") + 1] == "white_jerk"
    start = cmd.index("--process-noise-jerk-psd")
    assert [float(value) for value in cmd[start + 1 : start + 4]] == [1.0, 2.0, 4.0]
    default = validate_profile(ReconstructionProfile(name="", family="ekf_2d", process_noise_jerk_psd=[1.0, 2.0, 4.0]))
    assert default.process_noise_jerk_psd is None
    assert "--process-noise-model" not in build_pipeline_command(default, tmp_path, Path("c.toml"), Path("k.json"))
    with pytest.raises(ValueError):
        validate_profile(
            ReconstructionProfile(
                name="", family="ekf_2d", process_noise_model="white_jerk", process_noise_jerk_psd=[1.0]
            )
        )
