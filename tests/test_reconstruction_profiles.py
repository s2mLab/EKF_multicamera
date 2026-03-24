from pathlib import Path

import pytest

from reconstruction.reconstruction_profiles import (
    ReconstructionProfile,
    build_pipeline_command,
    canonical_profile_name,
    validate_profile,
)


def test_canonical_profile_name_includes_camera_names():
    profile = validate_profile(
        ReconstructionProfile(
            name="",
            family="ekf_2d",
            predictor="acc",
            camera_names=["M11139", "M11140"],
        )
    )
    assert canonical_profile_name(profile) == "ekf_2d_acc_cams_m11139_m11140"


def test_build_pipeline_command_prefers_profile_camera_names_over_override():
    profile = validate_profile(
        ReconstructionProfile(
            name="ekf_custom_cams",
            family="triangulation",
            camera_names=["M11139", "M11140"],
        )
    )
    cmd = build_pipeline_command(
        profile=profile,
        output_root=Path("outputs"),
        calib=Path("inputs/calibration/Calib.toml"),
        keypoints=Path("inputs/keypoints/1_partie_0429_keypoints.json"),
        pose2sim_trc=Path("inputs/trc/1_partie_0429.trc"),
        camera_names_override=["M11141", "M11458"],
        python_executable="python",
    )
    assert "--camera-names" in cmd
    camera_names_arg = cmd[cmd.index("--camera-names") + 1]
    assert camera_names_arg == "M11139,M11140"


def test_canonical_profile_name_includes_root_pose_bootstrap_flag():
    profile = validate_profile(
        ReconstructionProfile(
            name="",
            family="ekf_2d",
            predictor="acc",
            ekf2d_initial_state_method="root_pose_bootstrap",
        )
    )
    assert canonical_profile_name(profile) == "ekf_2d_acc_rootq0"


def test_canonical_profile_name_includes_frame_stride():
    profile = validate_profile(
        ReconstructionProfile(
            name="",
            family="ekf_2d",
            predictor="acc",
            frame_stride=3,
        )
    )
    assert canonical_profile_name(profile) == "ekf_2d_acc_stride3"


def test_canonical_profile_name_includes_ekf3d_root_pose_flag():
    profile = validate_profile(
        ReconstructionProfile(
            name="",
            family="ekf_3d",
            biorbd_kalman_init_method="root_pose_zero_rest",
        )
    )
    assert canonical_profile_name(profile) == "ekf_3d_rootq0"


def test_build_pipeline_command_includes_ekf3d_init_method():
    profile = validate_profile(
        ReconstructionProfile(
            name="ekf3d_rootq0",
            family="ekf_3d",
            biorbd_kalman_init_method="root_pose_zero_rest",
        )
    )
    cmd = build_pipeline_command(
        profile=profile,
        output_root=Path("outputs"),
        calib=Path("inputs/calibration/Calib.toml"),
        keypoints=Path("inputs/keypoints/1_partie_0429_keypoints.json"),
        pose2sim_trc=Path("inputs/trc/1_partie_0429.trc"),
        python_executable="python",
    )
    assert "--biorbd-kalman-init-method" in cmd
    assert cmd[cmd.index("--biorbd-kalman-init-method") + 1] == "root_pose_zero_rest"


def test_build_pipeline_command_includes_frame_stride():
    profile = validate_profile(
        ReconstructionProfile(
            name="tri_stride2",
            family="triangulation",
            frame_stride=2,
        )
    )
    cmd = build_pipeline_command(
        profile=profile,
        output_root=Path("outputs"),
        calib=Path("inputs/calibration/Calib.toml"),
        keypoints=Path("inputs/keypoints/1_partie_0429_keypoints.json"),
        pose2sim_trc=Path("inputs/trc/1_partie_0429.trc"),
        python_executable="python",
    )
    assert "--frame-stride" in cmd
    assert cmd[cmd.index("--frame-stride") + 1] == "2"


def test_build_pipeline_command_includes_frame_stride_for_ekf2d():
    profile = validate_profile(
        ReconstructionProfile(
            name="ekf2d_stride4",
            family="ekf_2d",
            predictor="acc",
            frame_stride=4,
        )
    )
    cmd = build_pipeline_command(
        profile=profile,
        output_root=Path("outputs"),
        calib=Path("inputs/calibration/Calib.toml"),
        keypoints=Path("inputs/keypoints/1_partie_0429_keypoints.json"),
        pose2sim_trc=Path("inputs/trc/1_partie_0429.trc"),
        python_executable="python",
    )
    assert "--frame-stride" in cmd
    assert cmd[cmd.index("--frame-stride") + 1] == "4"


def test_validate_profile_pose2sim_forces_frame_stride_to_one():
    profile = validate_profile(
        ReconstructionProfile(
            name="pose2sim_stride4",
            family="pose2sim",
            frame_stride=4,
        )
    )
    assert profile.frame_stride == 1


def test_validate_profile_rejects_unsupported_frame_stride():
    with pytest.raises(ValueError, match="Unsupported frame_stride"):
        validate_profile(
            ReconstructionProfile(
                name="bad_stride",
                family="triangulation",
                frame_stride=5,
            )
        )


def test_validate_profile_accepts_once_and_triangulation_once_coherence():
    profile = validate_profile(
        ReconstructionProfile(
            name="",
            family="ekf_2d",
            predictor="acc",
            triangulation_method="once",
            coherence_method="triangulation_once",
        )
    )

    assert profile.triangulation_method == "once"
    assert profile.coherence_method == "triangulation_once"
    assert "coh_triangulation_once" in canonical_profile_name(profile)


def test_build_pipeline_command_includes_explicit_coherence_and_once_triangulation():
    profile = validate_profile(
        ReconstructionProfile(
            name="ekf2d_once",
            family="ekf_2d",
            predictor="acc",
            triangulation_method="once",
            coherence_method="triangulation_greedy",
        )
    )

    cmd = build_pipeline_command(
        profile=profile,
        output_root=Path("outputs"),
        calib=Path("inputs/calibration/Calib.toml"),
        keypoints=Path("inputs/keypoints/1_partie_0429_keypoints.json"),
        pose2sim_trc=Path("inputs/trc/1_partie_0429.trc"),
        python_executable="python",
    )

    assert cmd[cmd.index("--triangulation-method") + 1] == "once"
    assert cmd[cmd.index("--coherence-method") + 1] == "triangulation_greedy"


def test_validate_profile_accepts_epipolar_fast_coherence_and_flip_method():
    profile = validate_profile(
        ReconstructionProfile(
            name="",
            family="ekf_2d",
            predictor="acc",
            coherence_method="epipolar_fast",
            flip=True,
            flip_method="triangulation_greedy",
        )
    )

    assert profile.coherence_method == "epipolar_fast"
    assert profile.flip_method == "triangulation_greedy"
    assert "coh_epipolar_fast" in canonical_profile_name(profile)
    assert "flip_triangulation_greedy" in canonical_profile_name(profile)


def test_validate_profile_accepts_explicit_viterbi_flip_method():
    profile = validate_profile(
        ReconstructionProfile(
            name="",
            family="ekf_2d",
            predictor="acc",
            flip=True,
            flip_method="epipolar_fast_viterbi",
        )
    )

    assert profile.flip_method == "epipolar_fast_viterbi"
    assert "flip_epipolar_fast_viterbi" in canonical_profile_name(profile)


def test_validate_profile_accepts_ekf_prediction_gate_for_ekf2d_only():
    ekf2d_profile = validate_profile(
        ReconstructionProfile(
            name="",
            family="ekf_2d",
            predictor="acc",
            flip=True,
            flip_method="ekf_prediction_gate",
        )
    )

    assert ekf2d_profile.flip_method == "ekf_prediction_gate"
    assert "flip_ekf_prediction_gate" in canonical_profile_name(ekf2d_profile)

    try:
        validate_profile(
            ReconstructionProfile(
                name="",
                family="triangulation",
                flip=True,
                flip_method="ekf_prediction_gate",
            )
        )
    except ValueError as exc:
        assert "ekf_prediction_gate" in str(exc)
    else:
        raise AssertionError("triangulation profiles should reject ekf_prediction_gate")


def test_build_pipeline_command_includes_flip_method():
    profile = validate_profile(
        ReconstructionProfile(
            name="tri_flip_fast",
            family="triangulation",
            flip=True,
            flip_method="epipolar_fast",
        )
    )

    cmd = build_pipeline_command(
        profile=profile,
        output_root=Path("outputs"),
        calib=Path("inputs/calibration/Calib.toml"),
        keypoints=Path("inputs/keypoints/1_partie_0429_keypoints.json"),
        pose2sim_trc=Path("inputs/trc/1_partie_0429.trc"),
        python_executable="python",
    )

    assert "--flip-method" in cmd
    assert cmd[cmd.index("--flip-method") + 1] == "epipolar_fast"
