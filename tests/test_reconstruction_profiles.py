from pathlib import Path

from reconstruction_profiles import ReconstructionProfile, build_pipeline_command, canonical_profile_name, validate_profile


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
        calib=Path("inputs/Calib.toml"),
        keypoints=Path("inputs/1_partie_0429_keypoints.json"),
        pose2sim_trc=Path("inputs/1_partie_0429.trc"),
        camera_names_override=["M11141", "M11458"],
        python_executable="python",
    )
    assert "--camera-names" in cmd
    camera_names_arg = cmd[cmd.index("--camera-names") + 1]
    assert camera_names_arg == "M11139,M11140"
