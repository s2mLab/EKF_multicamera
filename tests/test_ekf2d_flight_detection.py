"""Opt-in EKF-state flight criterion for the ``dyn`` predictor (``flight_detection='ekf_state'``)."""

import numpy as np
import pytest
from _ekf2d_synthetic import (
    load_synthetic_model,
    make_pose_and_reconstruction,
    model_keypoints_3d,
    project_keypoints,
    synthetic_cameras,
)

from reconstruction.reconstruction_profiles import ReconstructionProfile, build_pipeline_command, validate_profile
from vitpose_ekf_pipeline import MultiViewKinematicEKF, normalize_flight_detection, run_ekf

FPS = 120.0
GRAVITY = 9.81


def test_state_flight_criterion_uses_threshold_hysteresis_and_consecutive_frames():
    ekf = MultiViewKinematicEKF.__new__(MultiViewKinematicEKF)
    ekf.nq = 1
    ekf.flight_height_threshold_m = 1.5
    ekf.flight_hysteresis_m = 0.05
    ekf.flight_min_consecutive_frames = 2
    ekf.flight_com_accel_tolerance = None
    ekf._state_flight_active = False
    ekf._state_flight_candidate_frames = 0
    heights = iter([0.5, 1.6, 1.6, 1.52, 1.47, 1.44, 1.6])
    ekf._lowest_model_marker_height = lambda _q: next(heights)

    flags = [ekf._is_airborne_from_previous_state(np.zeros(3), frame_idx) for frame_idx in range(1, 8)]

    assert flags == [False, False, True, True, True, False, False]
    assert ekf._is_airborne_from_previous_state(np.zeros(3), 0) is False


def test_normalize_flight_detection():
    assert normalize_flight_detection(None) == "triangulation"
    assert normalize_flight_detection("EKF_State") == "ekf_state"
    with pytest.raises(ValueError):
        normalize_flight_detection("com")


def _ballistic_scene(tmp_path, n_stand=15, takeoff_speed=4.0, n_after=10):
    model = load_synthetic_model(tmp_path)
    nq = model.nbQ()
    flight_duration = 2.0 * takeoff_speed / GRAVITY
    n_flight = int(np.floor(flight_duration * FPS))
    n_frames = n_stand + n_flight + n_after
    t = np.arange(n_frames) / FPS
    q = np.zeros((n_frames, nq))
    q[:, 2] = 1.0
    t_flight = np.clip(t - t[n_stand], 0.0, flight_duration)
    q[:, 2] += takeoff_speed * t_flight - 0.5 * GRAVITY * t_flight**2
    joints = np.arange(6, nq)
    q[:, joints] = 0.1 * np.sin(2.0 * np.pi * 0.8 * t[:, np.newaxis] + joints[np.newaxis, :])
    points_3d = model_keypoints_3d(model, q)
    calibrations = synthetic_cameras(5)
    keypoints = project_keypoints(points_3d, calibrations)
    keypoints += np.random.default_rng(1).normal(scale=1.0, size=keypoints.shape)
    # ekf2d_3d_source=first_frame_only: only the first frame has 3D support.
    support = np.full_like(points_3d, np.nan)
    support[0] = points_3d[0]
    pose_data, reconstruction = make_pose_and_reconstruction(keypoints, support, calibrations)
    lowest = np.nanmin(points_3d[..., 2], axis=1)
    threshold = float(lowest[0] + 0.3)
    return model, calibrations, pose_data, reconstruction, q, lowest, threshold, n_stand


def test_ekf_state_flight_detection_activates_dyn_on_ballistic_flight(tmp_path):
    model, calibrations, pose_data, reconstruction, q_true, lowest, threshold, n_stand = _ballistic_scene(tmp_path)
    nq = model.nbQ()
    initial_state = np.concatenate((q_true[0], np.zeros(nq), np.zeros(nq)))
    common = dict(
        biomod_path=None,
        calibrations=calibrations,
        pose_data=pose_data,
        reconstruction=reconstruction,
        fps=FPS,
        measurement_noise_scale=1.5,
        model=model,
        initial_state=initial_state,
        flight_height_threshold_m=threshold,
        flight_min_consecutive_frames=2,
    )

    legacy_dyn, _ = run_ekf(root_flight_dynamics=True, predictor_mode="dyn", **common)
    acc, _ = run_ekf(root_flight_dynamics=False, predictor_mode="acc", **common)
    state_dyn, _ = run_ekf(root_flight_dynamics=True, predictor_mode="dyn", flight_detection="ekf_state", **common)

    # Historical criterion: no triangulated support after frame 0, so dyn never activates and equals acc.
    assert legacy_dyn["flight_detection"] == "triangulation"
    assert not np.any(legacy_dyn["dyn_active_per_frame"])
    np.testing.assert_array_equal(legacy_dyn["q"], acc["q"])
    assert acc["dyn_active_per_frame"] is None

    active = state_dyn["dyn_active_per_frame"]
    truly_airborne = lowest > threshold
    assert active.shape == (q_true.shape[0],)
    assert np.count_nonzero(active) >= 0.8 * np.count_nonzero(truly_airborne)
    assert not np.any(active[: n_stand + 1])
    # Activation (decided on frame t-1, with hysteresis) stays inside the true airborne phase.
    assert np.all(lowest[np.flatnonzero(active) - 1] > threshold - 0.05 - 0.02)
    # During flight the root vertical acceleration is imposed by the free-floating dynamics.
    active_idx = np.flatnonzero(active)[5:-5]
    np.testing.assert_allclose(state_dyn["qddot"][active_idx, 2], -GRAVITY, atol=1.5)
    assert np.sqrt(np.mean((state_dyn["q"][:, :3] - q_true[:, :3]) ** 2)) < 0.05


def test_ekf_state_flight_detection_with_com_acceleration_gate(tmp_path):
    model, calibrations, pose_data, reconstruction, q_true, lowest, threshold, _n_stand = _ballistic_scene(tmp_path)
    nq = model.nbQ()
    kwargs = dict(
        biomod_path=None,
        calibrations=calibrations,
        pose_data=pose_data,
        reconstruction=reconstruction,
        fps=FPS,
        measurement_noise_scale=1.5,
        model=model,
        initial_state=np.concatenate((q_true[0], np.zeros(nq), np.zeros(nq))),
        flight_height_threshold_m=threshold,
        root_flight_dynamics=True,
        predictor_mode="dyn",
        flight_detection="ekf_state",
    )

    tight, _ = run_ekf(flight_com_accel_tolerance=1e-6, **kwargs)
    loose, _ = run_ekf(flight_com_accel_tolerance=5.0, **kwargs)

    assert np.count_nonzero(tight["dyn_active_per_frame"]) < np.count_nonzero(loose["dyn_active_per_frame"])
    assert np.count_nonzero(loose["dyn_active_per_frame"]) > 0


def test_profile_flight_detection_is_named_and_forwarded(tmp_path):
    from pathlib import Path

    profile = validate_profile(
        ReconstructionProfile(name="", family="ekf_2d", predictor="dyn", flight_detection="ekf_state")
    )
    cmd = build_pipeline_command(profile, tmp_path, Path("c.toml"), Path("k.json"))

    assert "flightekf" in profile.name
    assert cmd[cmd.index("--flight-detection") + 1] == "ekf_state"
    default = validate_profile(ReconstructionProfile(name="", family="ekf_2d", predictor="dyn"))
    assert "--flight-detection" not in build_pipeline_command(default, tmp_path, Path("c.toml"), Path("k.json"))
    assert validate_profile(
        ReconstructionProfile(name="", family="ekf_3d", flight_detection="ekf_state")
    ).flight_detection == ("triangulation")
