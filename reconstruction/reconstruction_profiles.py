#!/usr/bin/env python3
"""Profils de reconstruction nommes et serialisables.

Ce module permet de:
- decrire une reconstruction par un profil JSON,
- verifier la compatibilite des options,
- generer automatiquement des combinaisons supportees,
- construire la commande pipeline correspondante.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

from reconstruction.reconstruction_registry import (
    infer_dataset_name,
    reconstruction_output_dir,
    scan_dataset_dirs,
    scan_reconstruction_dirs,
    slugify,
)

SUPPORTED_FAMILIES = ("pose2sim", "triangulation", "ekf_3d", "ekf_2d")
SUPPORTED_PREDICTORS = ("acc", "dyn")
SUPPORTED_EKF2D_3D_SOURCE_MODES = ("full_triangulation", "first_frame_only")
SUPPORTED_MODEL_VARIANTS = (
    "single_trunk",
    "back_flexion_1d",
    "back_3dof",
    "upper_root_back_flexion_1d",
    "upper_root_back_3dof",
)
SUPPORTED_POSE_DATA_MODES = ("raw", "filtered", "cleaned")
SUPPORTED_TRIANGULATION_METHODS = ("once", "greedy", "exhaustive")
SUPPORTED_COHERENCE_METHODS = (
    "epipolar",
    "epipolar_fast",
    "epipolar_framewise",
    "epipolar_fast_framewise",
    "triangulation",
    "triangulation_once",
    "triangulation_greedy",
    "triangulation_exhaustive",
)
SUPPORTED_FLIP_METHODS = (
    "epipolar",
    "epipolar_fast",
    "epipolar_viterbi",
    "epipolar_fast_viterbi",
    "ekf_prediction_gate",
    "triangulation_once",
    "triangulation_greedy",
    "triangulation_exhaustive",
)
SUPPORTED_BIORBD_KALMAN_INIT_METHODS = (
    "none",
    "triangulation_ik",
    "triangulation_ik_root_translation",
    "root_translation_zero_rest",
    "root_pose_zero_rest",
)
SUPPORTED_FRAME_STRIDES = (1, 2, 3, 4)
DEFAULT_POSE2SIM_TRC = Path("inputs/trc/1_partie_0429.trc")


@dataclass
class ReconstructionProfile:
    name: str
    family: str
    camera_names: list[str] | None = None
    use_all_cameras: bool = False
    ekf_model_path: str | None = None
    model_variant: str = "single_trunk"
    symmetrize_limbs: bool = True
    frame_stride: int = 1
    predictor: str | None = None
    ekf2d_3d_source: str = "full_triangulation"
    ekf2d_initial_state_method: str = "ekf_bootstrap"
    ekf2d_bootstrap_passes: int = 5
    flip: bool = False
    flip_method: str = "epipolar"
    flip_improvement_ratio: float = 0.7
    flip_min_gain_px: float = 3.0
    flip_min_other_cameras: int = 2
    flip_restrict_to_outliers: bool = True
    flip_outlier_percentile: float = 85.0
    flip_outlier_floor_px: float = 5.0
    flip_temporal_weight: float = 0.35
    flip_temporal_tau_px: float = 20.0
    flip_temporal_min_valid_keypoints: int = 4
    dof_locking: bool = False
    initial_rotation_correction: bool = False
    pose_data_mode: str = "cleaned"
    triangulation_method: str = "exhaustive"
    coherence_method: str = "epipolar"
    compare_biorbd_kalman: bool = True
    reuse_triangulation: bool = False
    no_root_unwrap: bool = False
    biorbd_kalman_noise_factor: float = 1e-8
    biorbd_kalman_error_factor: float = 1e-4
    biorbd_kalman_init_method: str = "triangulation_ik_root_translation"
    measurement_noise_scale: float = 1.5
    process_noise_scale: float = 1.0
    coherence_confidence_floor: float = 0.35
    upper_back_sagittal_gain: float = 0.2
    upper_back_pseudo_std_deg: float = 10.0
    pose_filter_window: int = 9
    pose_outlier_threshold_ratio: float = 0.10
    pose_amplitude_lower_percentile: float = 5.0
    pose_amplitude_upper_percentile: float = 95.0
    enabled: bool = True
    extra_args: list[str] | None = None


def canonical_profile_name(profile: ReconstructionProfile) -> str:
    parts = [profile.family]
    if profile.model_variant != "single_trunk":
        parts.append(profile.model_variant)
    if not bool(profile.symmetrize_limbs):
        parts.append("asym")
    if profile.family in ("ekf_2d", "ekf_3d") and profile.ekf_model_path:
        parts.append(f"mdl_{Path(profile.ekf_model_path).stem}")
    if profile.family == "pose2sim":
        if profile.initial_rotation_correction:
            parts.append("rotfix")
    elif profile.family == "ekf_2d":
        parts.append(profile.predictor or "acc")
        if profile.ekf2d_3d_source == "first_frame_only":
            parts.append("bootstrap1")
        if profile.ekf2d_initial_state_method == "triangulation_ik":
            parts.append("ikq0")
        elif profile.ekf2d_initial_state_method == "root_pose_bootstrap":
            parts.append("rootq0")
        if int(profile.ekf2d_bootstrap_passes) != 5:
            parts.append(f"boot{int(profile.ekf2d_bootstrap_passes)}")
        if profile.coherence_method != "epipolar":
            parts.append(f"coh_{profile.coherence_method}")
        if not math.isclose(float(profile.upper_back_sagittal_gain), 0.2, rel_tol=0.0, abs_tol=1e-9):
            parts.append(f"ubg{slugify(f'{profile.upper_back_sagittal_gain:.2f}')}")
        if profile.flip:
            if profile.flip_method != "epipolar":
                parts.append(f"flip_{profile.flip_method}")
            parts.append("flip")
        if profile.dof_locking:
            parts.append("lock")
        if profile.initial_rotation_correction:
            parts.append("rotfix")
    elif profile.family == "ekf_3d":
        if profile.biorbd_kalman_init_method == "triangulation_ik":
            parts.append("ikq0")
        elif profile.biorbd_kalman_init_method == "root_translation_zero_rest":
            parts.append("roottransq0")
        elif profile.biorbd_kalman_init_method == "root_pose_zero_rest":
            parts.append("rootq0")
        if profile.flip:
            if profile.flip_method != "epipolar":
                parts.append(f"flip_{profile.flip_method}")
            parts.append("flip")
        if profile.initial_rotation_correction:
            parts.append("rotfix")
    elif profile.family == "triangulation":
        parts.append(profile.triangulation_method)
        if profile.flip:
            if profile.flip_method != "epipolar":
                parts.append(f"flip_{profile.flip_method}")
            parts.append("flip")
        if profile.initial_rotation_correction:
            parts.append("rotfix")
    if profile.flip:
        if abs(float(profile.flip_improvement_ratio) - 0.7) > 1e-9:
            parts.append(f"fr{str(float(profile.flip_improvement_ratio)).replace('.', 'p')}")
        if abs(float(profile.flip_min_gain_px) - 3.0) > 1e-9:
            parts.append(f"fg{str(float(profile.flip_min_gain_px)).replace('.', 'p')}")
        if int(profile.flip_min_other_cameras) != 2:
            parts.append(f"fc{int(profile.flip_min_other_cameras)}")
        if not bool(profile.flip_restrict_to_outliers):
            parts.append("flipall")
        if abs(float(profile.flip_outlier_percentile) - 85.0) > 1e-9:
            parts.append(f"fp{int(round(float(profile.flip_outlier_percentile)))}")
        if abs(float(profile.flip_outlier_floor_px) - 5.0) > 1e-9:
            parts.append(f"ff{str(float(profile.flip_outlier_floor_px)).replace('.', 'p')}")
        if abs(float(profile.flip_temporal_weight) - 0.35) > 1e-9:
            parts.append(f"ftw{str(float(profile.flip_temporal_weight)).replace('.', 'p')}")
        if abs(float(profile.flip_temporal_tau_px) - 20.0) > 1e-9:
            parts.append(f"ftt{str(float(profile.flip_temporal_tau_px)).replace('.', 'p')}")
    if profile.pose_data_mode != "cleaned":
        parts.append(profile.pose_data_mode)
    if int(profile.frame_stride) != 1:
        parts.append(f"stride{int(profile.frame_stride)}")
    if profile.use_all_cameras:
        parts.append("all_cameras")
    elif profile.camera_names:
        parts.append("cams")
        parts.extend(str(name) for name in profile.camera_names)
    return slugify("_".join(parts))


def validate_profile(profile: ReconstructionProfile) -> ReconstructionProfile:
    if profile.family not in SUPPORTED_FAMILIES:
        raise ValueError(f"Unsupported family: {profile.family}")
    if profile.pose_data_mode not in SUPPORTED_POSE_DATA_MODES:
        raise ValueError(f"Unsupported pose_data_mode: {profile.pose_data_mode}")
    if profile.triangulation_method not in SUPPORTED_TRIANGULATION_METHODS:
        raise ValueError(f"Unsupported triangulation_method: {profile.triangulation_method}")
    if profile.coherence_method not in SUPPORTED_COHERENCE_METHODS:
        raise ValueError(f"Unsupported coherence_method: {profile.coherence_method}")
    if profile.flip_method not in SUPPORTED_FLIP_METHODS:
        raise ValueError(f"Unsupported flip_method: {profile.flip_method}")
    if profile.model_variant not in SUPPORTED_MODEL_VARIANTS:
        raise ValueError(f"Unsupported model_variant: {profile.model_variant}")
    if profile.biorbd_kalman_init_method not in SUPPORTED_BIORBD_KALMAN_INIT_METHODS:
        raise ValueError(f"Unsupported biorbd_kalman_init_method: {profile.biorbd_kalman_init_method}")
    if float(profile.upper_back_sagittal_gain) < 0.0:
        raise ValueError("upper_back_sagittal_gain must be >= 0.")
    if float(profile.upper_back_pseudo_std_deg) <= 0.0:
        raise ValueError("upper_back_pseudo_std_deg must be > 0.")
    profile.frame_stride = int(profile.frame_stride)
    if profile.frame_stride not in SUPPORTED_FRAME_STRIDES:
        raise ValueError(f"Unsupported frame_stride: {profile.frame_stride}")
    if profile.camera_names is not None:
        normalized_camera_names: list[str] = []
        seen_camera_names: set[str] = set()
        for camera_name in profile.camera_names:
            name = str(camera_name).strip()
            if not name or name in seen_camera_names:
                continue
            seen_camera_names.add(name)
            normalized_camera_names.append(name)
        profile.camera_names = normalized_camera_names or None
    profile.use_all_cameras = bool(profile.use_all_cameras)
    if profile.use_all_cameras:
        profile.camera_names = None
    if profile.ekf_model_path is not None:
        normalized_model_path = str(profile.ekf_model_path).strip()
        profile.ekf_model_path = normalized_model_path or None
    profile.pose_filter_window = max(3, int(profile.pose_filter_window))
    if profile.pose_filter_window % 2 == 0:
        profile.pose_filter_window += 1
    if not (0.0 < float(profile.pose_outlier_threshold_ratio) <= 1.0):
        raise ValueError("pose_outlier_threshold_ratio must be in (0, 1].")
    if not (
        0.0 <= float(profile.pose_amplitude_lower_percentile) < float(profile.pose_amplitude_upper_percentile) <= 100.0
    ):
        raise ValueError("Amplitude percentiles must satisfy 0 <= lower < upper <= 100.")

    if profile.family == "pose2sim":
        profile.pose_data_mode = "cleaned"
        profile.triangulation_method = "exhaustive"
        profile.frame_stride = 1
    if profile.family == "ekf_2d":
        profile.predictor = profile.predictor or "acc"
        if profile.predictor not in SUPPORTED_PREDICTORS:
            raise ValueError(f"Unsupported predictor: {profile.predictor}")
        if profile.ekf2d_3d_source not in SUPPORTED_EKF2D_3D_SOURCE_MODES:
            raise ValueError(f"Unsupported ekf2d_3d_source: {profile.ekf2d_3d_source}")
        if profile.ekf2d_initial_state_method not in ("triangulation_ik", "ekf_bootstrap", "root_pose_bootstrap"):
            raise ValueError(f"Unsupported ekf2d_initial_state_method: {profile.ekf2d_initial_state_method}")
        profile.ekf2d_bootstrap_passes = max(1, int(profile.ekf2d_bootstrap_passes))
        if profile.ekf2d_3d_source == "first_frame_only" and profile.coherence_method not in (
            "epipolar",
            "epipolar_fast",
            "epipolar_framewise",
            "epipolar_fast_framewise",
        ):
            raise ValueError("ekf2d_3d_source=first_frame_only requires an epipolar coherence method.")
    elif profile.flip and profile.flip_method == "ekf_prediction_gate":
        raise ValueError("flip_method=ekf_prediction_gate is only supported for family='ekf_2d'.")
    else:
        profile.predictor = None
        profile.ekf2d_3d_source = "full_triangulation"
        profile.ekf2d_initial_state_method = "ekf_bootstrap"
        profile.ekf2d_bootstrap_passes = 5
        profile.dof_locking = False
    if profile.family != "ekf_3d":
        profile.biorbd_kalman_init_method = "triangulation_ik_root_translation"
    if profile.family not in ("ekf_2d", "ekf_3d"):
        profile.ekf_model_path = None
        profile.model_variant = "single_trunk"
        profile.symmetrize_limbs = True
        profile.upper_back_sagittal_gain = 0.2
        profile.upper_back_pseudo_std_deg = 10.0
    profile.flip_min_other_cameras = max(1, int(profile.flip_min_other_cameras))
    if not (0.0 < float(profile.flip_improvement_ratio) < 1.0):
        raise ValueError("flip_improvement_ratio must be in (0, 1).")
    if float(profile.flip_min_gain_px) < 0.0:
        raise ValueError("flip_min_gain_px must be >= 0.")
    if not (0.0 <= float(profile.flip_outlier_percentile) <= 100.0):
        raise ValueError("flip_outlier_percentile must be in [0, 100].")
    if float(profile.flip_outlier_floor_px) < 0.0:
        raise ValueError("flip_outlier_floor_px must be >= 0.")
    if not (0.0 <= float(profile.flip_temporal_weight) <= 1.0):
        raise ValueError("flip_temporal_weight must be in [0, 1].")
    if float(profile.flip_temporal_tau_px) <= 0.0:
        raise ValueError("flip_temporal_tau_px must be > 0.")
    profile.flip_temporal_min_valid_keypoints = max(1, int(profile.flip_temporal_min_valid_keypoints))
    if profile.family not in ("triangulation", "ekf_2d", "ekf_3d"):
        profile.flip = False
    if not profile.flip:
        profile.flip_method = "epipolar"

    if profile.family in ("triangulation", "pose2sim"):
        profile.compare_biorbd_kalman = False
    if not profile.name:
        profile.name = canonical_profile_name(profile)
    profile.name = slugify(profile.name)
    return profile


def profile_from_dict(data: dict[str, object]) -> ReconstructionProfile:
    profile = ReconstructionProfile(**data)
    return validate_profile(profile)


def profile_to_dict(profile: ReconstructionProfile) -> dict[str, object]:
    return asdict(validate_profile(profile))


def load_profiles_json(path: Path) -> list[ReconstructionProfile]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        entries = payload.get("profiles", [])
    else:
        entries = payload
    return [profile_from_dict(entry) for entry in entries]


def save_profiles_json(path: Path, profiles: list[ReconstructionProfile]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"profiles": [profile_to_dict(profile) for profile in profiles]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def example_profiles() -> list[ReconstructionProfile]:
    examples = [
        ReconstructionProfile(name="pose2sim", family="pose2sim", initial_rotation_correction=False),
        ReconstructionProfile(name="pose2sim_rotfix", family="pose2sim", initial_rotation_correction=True),
        ReconstructionProfile(
            name="ekf_2d_acc_flip_lock_filtered",
            family="ekf_2d",
            predictor="acc",
            flip=True,
            dof_locking=True,
            pose_data_mode="filtered",
        ),
        ReconstructionProfile(
            name="ekf_2d_acc_flip_lock",
            family="ekf_2d",
            predictor="acc",
            flip=True,
            dof_locking=True,
            pose_data_mode="cleaned",
        ),
        ReconstructionProfile(name="ekf_2d_acc", family="ekf_2d", predictor="acc", pose_data_mode="cleaned"),
        ReconstructionProfile(
            name="ekf_2d_acc_bootstrap1",
            family="ekf_2d",
            predictor="acc",
            pose_data_mode="cleaned",
            ekf2d_3d_source="first_frame_only",
        ),
        ReconstructionProfile(name="ekf_3d_filtered", family="ekf_3d", pose_data_mode="filtered"),
        ReconstructionProfile(name="ekf_3d_flip", family="ekf_3d", flip=True, pose_data_mode="cleaned"),
        ReconstructionProfile(name="ekf_3d", family="ekf_3d", pose_data_mode="cleaned"),
        ReconstructionProfile(
            name="triangulation_exhaustive", family="triangulation", triangulation_method="exhaustive"
        ),
        ReconstructionProfile(
            name="triangulation_exhaustive_flip", family="triangulation", triangulation_method="exhaustive", flip=True
        ),
        ReconstructionProfile(name="triangulation_once", family="triangulation", triangulation_method="once"),
        ReconstructionProfile(name="triangulation_greedy", family="triangulation", triangulation_method="greedy"),
    ]
    return [validate_profile(profile) for profile in examples]


def generate_supported_profiles(
    families: tuple[str, ...] = SUPPORTED_FAMILIES,
    pose_data_modes: tuple[str, ...] = SUPPORTED_POSE_DATA_MODES,
    triangulation_methods: tuple[str, ...] = SUPPORTED_TRIANGULATION_METHODS,
    predictors: tuple[str, ...] = SUPPORTED_PREDICTORS,
    enable_flip: bool = True,
    enable_dof_locking: bool = True,
    enable_initial_rotation_correction: bool = True,
) -> list[ReconstructionProfile]:
    profiles: list[ReconstructionProfile] = []
    for family in families:
        if family == "pose2sim":
            for initial_rotation_correction in ((False, True) if enable_initial_rotation_correction else (False,)):
                profiles.append(
                    validate_profile(
                        ReconstructionProfile(
                            name="",
                            family="pose2sim",
                            initial_rotation_correction=initial_rotation_correction,
                        )
                    )
                )
            continue
        if family == "triangulation":
            for triangulation_method, pose_data_mode, flip, initial_rotation_correction in product(
                triangulation_methods,
                pose_data_modes,
                (False, True) if enable_flip else (False,),
                (False, True) if enable_initial_rotation_correction else (False,),
            ):
                profiles.append(
                    validate_profile(
                        ReconstructionProfile(
                            name="",
                            family="triangulation",
                            triangulation_method=triangulation_method,
                            pose_data_mode=pose_data_mode,
                            flip=flip,
                            initial_rotation_correction=initial_rotation_correction,
                        )
                    )
                )
            continue
        if family == "ekf_3d":
            for pose_data_mode, flip, initial_rotation_correction in product(
                pose_data_modes,
                (False, True) if enable_flip else (False,),
                (False, True) if enable_initial_rotation_correction else (False,),
            ):
                profiles.append(
                    validate_profile(
                        ReconstructionProfile(
                            name="",
                            family="ekf_3d",
                            pose_data_mode=pose_data_mode,
                            flip=flip,
                            initial_rotation_correction=initial_rotation_correction,
                        )
                    )
                )
            continue
        for pose_data_mode, predictor, ekf2d_3d_source, flip, dof_locking, initial_rotation_correction in product(
            pose_data_modes,
            predictors,
            SUPPORTED_EKF2D_3D_SOURCE_MODES,
            (False, True) if enable_flip else (False,),
            (False, True) if enable_dof_locking else (False,),
            (False, True) if enable_initial_rotation_correction else (False,),
        ):
            coherence_methods = (
                ("epipolar", "epipolar_fast", "epipolar_framewise", "epipolar_fast_framewise")
                if ekf2d_3d_source == "first_frame_only"
                else SUPPORTED_COHERENCE_METHODS
            )
            for coherence_method in coherence_methods:
                profiles.append(
                    validate_profile(
                        ReconstructionProfile(
                            name="",
                            family="ekf_2d",
                            predictor=predictor,
                            ekf2d_3d_source=ekf2d_3d_source,
                            flip=flip,
                            dof_locking=dof_locking,
                            pose_data_mode=pose_data_mode,
                            coherence_method=coherence_method,
                            initial_rotation_correction=initial_rotation_correction,
                        )
                    )
                )
    unique: dict[str, ReconstructionProfile] = {}
    for profile in profiles:
        unique[profile.name] = profile
    return [unique[name] for name in sorted(unique)]


def variant_output_dir(
    output_root: Path,
    profile: ReconstructionProfile,
    dataset_name: str | None = None,
    keypoints_path: Path | None = None,
    pose2sim_trc: Path | None = None,
) -> Path:
    profile = validate_profile(profile)
    dataset_name = infer_dataset_name(
        keypoints_path=keypoints_path, pose2sim_trc=pose2sim_trc, dataset_name=dataset_name
    )
    return reconstruction_output_dir(output_root, dataset_name, profile.name)


def build_pipeline_command(
    profile: ReconstructionProfile,
    output_root: Path,
    calib: Path,
    keypoints: Path,
    pose2sim_trc: Path | None = None,
    dataset_name: str | None = None,
    python_executable: str | None = None,
    camera_names_override: list[str] | None = None,
) -> list[str]:
    profile = validate_profile(profile)
    python_executable = python_executable or sys.executable
    out_dir = variant_output_dir(
        output_root,
        profile,
        dataset_name=dataset_name,
        keypoints_path=keypoints,
        pose2sim_trc=pose2sim_trc,
    )
    cmd = [
        python_executable,
        "export_reconstruction_bundle.py",
        "--name",
        profile.name,
        "--family",
        profile.family,
        "--calib",
        str(calib),
        "--keypoints",
        str(keypoints),
        "--output-dir",
        str(out_dir),
        "--pose-data-mode",
        profile.pose_data_mode,
        "--frame-stride",
        str(profile.frame_stride),
        "--triangulation-method",
        profile.triangulation_method,
        "--coherence-method",
        profile.coherence_method,
        "--pose-filter-window",
        str(profile.pose_filter_window),
        "--pose-outlier-threshold-ratio",
        str(profile.pose_outlier_threshold_ratio),
        "--pose-amplitude-lower-percentile",
        str(profile.pose_amplitude_lower_percentile),
        "--pose-amplitude-upper-percentile",
        str(profile.pose_amplitude_upper_percentile),
        "--biorbd-kalman-noise-factor",
        str(profile.biorbd_kalman_noise_factor),
        "--biorbd-kalman-error-factor",
        str(profile.biorbd_kalman_error_factor),
        "--measurement-noise-scale",
        str(profile.measurement_noise_scale),
        "--process-noise-scale",
        str(profile.process_noise_scale),
        "--coherence-confidence-floor",
        str(profile.coherence_confidence_floor),
    ]
    if pose2sim_trc is not None:
        cmd.extend(["--trc-file", str(pose2sim_trc)])
    camera_names = None if profile.use_all_cameras else (profile.camera_names or camera_names_override)
    if camera_names:
        cmd.extend(["--camera-names", ",".join(str(name) for name in camera_names)])
    if profile.initial_rotation_correction:
        cmd.append("--initial-rotation-correction")
    if profile.no_root_unwrap:
        cmd.append("--no-root-unwrap")
    if profile.family == "ekf_2d":
        if profile.ekf_model_path:
            cmd.extend(["--biomod", str(profile.ekf_model_path)])
        cmd.extend(["--model-variant", profile.model_variant])
        if not profile.symmetrize_limbs:
            cmd.append("--no-symmetrize-limbs")
        cmd.extend(["--upper-back-sagittal-gain", str(profile.upper_back_sagittal_gain)])
        cmd.extend(["--upper-back-pseudo-std-deg", str(profile.upper_back_pseudo_std_deg)])
        cmd.extend(["--predictor", str(profile.predictor or "acc")])
        cmd.extend(["--ekf2d-3d-source", profile.ekf2d_3d_source])
        cmd.extend(["--ekf2d-initial-state-method", profile.ekf2d_initial_state_method])
        cmd.extend(["--ekf2d-bootstrap-passes", str(profile.ekf2d_bootstrap_passes)])
        if profile.flip:
            cmd.append("--flip-left-right")
            cmd.extend(["--flip-method", profile.flip_method])
        if profile.dof_locking:
            cmd.append("--enable-dof-locking")
    elif profile.family == "triangulation":
        if profile.flip:
            cmd.append("--flip-left-right")
            cmd.extend(["--flip-method", profile.flip_method])
    elif profile.family == "ekf_3d":
        if profile.ekf_model_path:
            cmd.extend(["--biomod", str(profile.ekf_model_path)])
        cmd.extend(["--model-variant", profile.model_variant])
        if not profile.symmetrize_limbs:
            cmd.append("--no-symmetrize-limbs")
        cmd.extend(["--biorbd-kalman-init-method", profile.biorbd_kalman_init_method])
        if profile.flip:
            cmd.append("--flip-left-right")
            cmd.extend(["--flip-method", profile.flip_method])
    if profile.family in ("triangulation", "ekf_2d", "ekf_3d"):
        cmd.extend(["--flip-improvement-ratio", str(profile.flip_improvement_ratio)])
        cmd.extend(["--flip-min-gain-px", str(profile.flip_min_gain_px)])
        cmd.extend(["--flip-min-other-cameras", str(profile.flip_min_other_cameras)])
        cmd.extend(["--flip-outlier-percentile", str(profile.flip_outlier_percentile)])
        cmd.extend(["--flip-outlier-floor-px", str(profile.flip_outlier_floor_px)])
        cmd.extend(["--flip-temporal-weight", str(profile.flip_temporal_weight)])
        cmd.extend(["--flip-temporal-tau-px", str(profile.flip_temporal_tau_px)])
        cmd.extend(["--flip-temporal-min-valid-keypoints", str(profile.flip_temporal_min_valid_keypoints)])
        if not profile.flip_restrict_to_outliers:
            cmd.append("--flip-test-all-camera-frames")
    if profile.extra_args:
        cmd.extend(profile.extra_args)
    return cmd


def scan_variant_output_dirs(output_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for dataset_dir in scan_dataset_dirs(output_root):
        candidates.extend(scan_reconstruction_dirs(dataset_dir))
    unique = {path.resolve(): path for path in candidates}
    return [unique[key] for key in sorted(unique, key=lambda item: str(item))]
