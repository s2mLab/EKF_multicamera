#!/usr/bin/env python3
"""CLI pour generer un bundle standardise de reconstruction."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from camera_selection import parse_camera_names, subset_calibrations
from reconstruction_bundle import (
    SUPPORTED_EKF2D_3D_SOURCE_MODES,
    build_ekf_2d_bundle,
    build_ekf_3d_bundle,
    build_pose2sim_bundle,
    build_pose_data,
    build_triangulation_bundle,
)
from vitpose_ekf_pipeline import (
    DEFAULT_BIORBD_KALMAN_ERROR_FACTOR,
    DEFAULT_BIORBD_KALMAN_INIT_METHOD,
    DEFAULT_BIORBD_KALMAN_NOISE_FACTOR,
    DEFAULT_CALIB,
    DEFAULT_CAMERA_FPS,
    DEFAULT_COHERENCE_CONFIDENCE_FLOOR,
    DEFAULT_COHERENCE_METHOD,
    DEFAULT_EPIPOLAR_THRESHOLD_PX,
    DEFAULT_FLIGHT_HEIGHT_THRESHOLD_M,
    DEFAULT_FLIGHT_MIN_CONSECUTIVE_FRAMES,
    DEFAULT_FLIP_TEMPORAL_MIN_VALID_KEYPOINTS,
    DEFAULT_FLIP_TEMPORAL_TAU_PX,
    DEFAULT_FLIP_TEMPORAL_WEIGHT,
    DEFAULT_KEYPOINTS,
    DEFAULT_MEASUREMENT_NOISE_SCALE,
    DEFAULT_MIN_CAMERAS_FOR_TRIANGULATION,
    DEFAULT_MIN_FRAME_COHERENCE_FOR_UPDATE,
    DEFAULT_REPROJECTION_THRESHOLD_PX,
    DEFAULT_SUBJECT_MASS_KG,
    DEFAULT_TRIANGULATION_METHOD,
    DEFAULT_TRIANGULATION_WORKERS,
    load_calibrations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genere un bundle standardise pour une reconstruction.")
    parser.add_argument("--name", required=True, help="Nom stable de la reconstruction.")
    parser.add_argument("--family", choices=("pose2sim", "triangulation", "ekf_3d", "ekf_2d"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    parser.add_argument("--keypoints", type=Path, default=DEFAULT_KEYPOINTS)
    parser.add_argument("--pose2sim-trc", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=DEFAULT_CAMERA_FPS)
    parser.add_argument("--camera-names", type=str, default="", help="Liste de cameras a utiliser, separees par des virgules.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--pose-data-mode", choices=("raw", "filtered", "cleaned"), default="cleaned")
    parser.add_argument("--pose-filter-window", type=int, default=9)
    parser.add_argument("--pose-outlier-threshold-ratio", type=float, default=0.10)
    parser.add_argument("--pose-amplitude-lower-percentile", type=float, default=5.0)
    parser.add_argument("--pose-amplitude-upper-percentile", type=float, default=95.0)
    parser.add_argument("--initial-rotation-correction", action="store_true")
    parser.add_argument("--triangulation-method", choices=("exhaustive", "greedy"), default=DEFAULT_TRIANGULATION_METHOD)
    parser.add_argument("--triangulation-workers", type=int, default=DEFAULT_TRIANGULATION_WORKERS)
    parser.add_argument("--reprojection-threshold-px", type=float, default=DEFAULT_REPROJECTION_THRESHOLD_PX)
    parser.add_argument("--epipolar-threshold-px", type=float, default=DEFAULT_EPIPOLAR_THRESHOLD_PX)
    parser.add_argument("--min-cameras-for-triangulation", type=int, default=DEFAULT_MIN_CAMERAS_FOR_TRIANGULATION)
    parser.add_argument("--coherence-method", choices=("epipolar", "triangulation"), default=DEFAULT_COHERENCE_METHOD)
    parser.add_argument("--subject-mass-kg", type=float, default=DEFAULT_SUBJECT_MASS_KG)
    parser.add_argument("--biorbd-kalman-noise-factor", type=float, default=DEFAULT_BIORBD_KALMAN_NOISE_FACTOR)
    parser.add_argument("--biorbd-kalman-error-factor", type=float, default=DEFAULT_BIORBD_KALMAN_ERROR_FACTOR)
    parser.add_argument(
        "--biorbd-kalman-init-method",
        choices=("none", "triangulation_ik", "triangulation_ik_root_translation", "root_translation_zero_rest"),
        default=DEFAULT_BIORBD_KALMAN_INIT_METHOD,
    )
    parser.add_argument("--predictor", choices=("acc", "dyn"), default="acc")
    parser.add_argument("--ekf2d-3d-source", choices=SUPPORTED_EKF2D_3D_SOURCE_MODES, default="full_triangulation")
    parser.add_argument(
        "--ekf2d-initial-state-method",
        choices=("triangulation_ik", "ekf_bootstrap", "root_pose_bootstrap"),
        default="ekf_bootstrap",
    )
    parser.add_argument("--ekf2d-bootstrap-passes", type=int, default=5)
    parser.add_argument("--flip-left-right", action="store_true")
    parser.add_argument("--flip-improvement-ratio", type=float, default=0.7)
    parser.add_argument("--flip-min-gain-px", type=float, default=3.0)
    parser.add_argument("--flip-min-other-cameras", type=int, default=2)
    parser.add_argument("--flip-outlier-percentile", type=float, default=85.0)
    parser.add_argument("--flip-outlier-floor-px", type=float, default=5.0)
    parser.add_argument("--flip-test-all-camera-frames", action="store_true", help="Desactive la restriction du test flip L/R aux outliers du cout nominal.")
    parser.add_argument("--flip-temporal-weight", type=float, default=DEFAULT_FLIP_TEMPORAL_WEIGHT)
    parser.add_argument("--flip-temporal-tau-px", type=float, default=DEFAULT_FLIP_TEMPORAL_TAU_PX)
    parser.add_argument("--flip-temporal-min-valid-keypoints", type=int, default=DEFAULT_FLIP_TEMPORAL_MIN_VALID_KEYPOINTS)
    parser.add_argument("--enable-dof-locking", action="store_true")
    parser.add_argument("--measurement-noise-scale", type=float, default=DEFAULT_MEASUREMENT_NOISE_SCALE)
    parser.add_argument("--process-noise-scale", type=float, default=1.0)
    parser.add_argument("--coherence-confidence-floor", type=float, default=DEFAULT_COHERENCE_CONFIDENCE_FLOOR)
    parser.add_argument("--min-frame-coherence-for-update", type=float, default=DEFAULT_MIN_FRAME_COHERENCE_FOR_UPDATE)
    parser.add_argument("--skip-low-coherence-updates", action="store_true")
    parser.add_argument("--flight-height-threshold-m", type=float, default=DEFAULT_FLIGHT_HEIGHT_THRESHOLD_M)
    parser.add_argument("--flight-min-consecutive-frames", type=int, default=DEFAULT_FLIGHT_MIN_CONSECUTIVE_FRAMES)
    parser.add_argument("--no-root-unwrap", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibrations = load_calibrations(args.calib)
    selected_camera_names = parse_camera_names(args.camera_names)
    if selected_camera_names:
        calibrations = subset_calibrations(calibrations, selected_camera_names)
    if args.family == "pose2sim":
        if args.pose2sim_trc is None:
            raise ValueError("pose2sim reconstruction requires --pose2sim-trc.")
        print("[STEP 1/2] Load 2D data", flush=True)
    elif args.family == "triangulation":
        print("[STEP 1/3] Load 2D data", flush=True)
    else:
        print("[STEP 1/5] Load 2D data", flush=True)
    pose_data_start = time.perf_counter()
    pose_data = build_pose_data(
        keypoints_path=args.keypoints,
        calibrations=calibrations,
        max_frames=args.max_frames,
        pose_data_mode=args.pose_data_mode,
        pose_filter_window=args.pose_filter_window,
        pose_outlier_threshold_ratio=args.pose_outlier_threshold_ratio,
        pose_amplitude_lower_percentile=args.pose_amplitude_lower_percentile,
        pose_amplitude_upper_percentile=args.pose_amplitude_upper_percentile,
    )
    pose_data_compute_time_s = time.perf_counter() - pose_data_start

    if args.family == "pose2sim":
        build_pose2sim_bundle(
            name=args.name,
            output_dir=args.output_dir,
            pose2sim_trc=args.pose2sim_trc,
            calibrations=calibrations,
            pose_data=pose_data,
            pose_data_compute_time_s=pose_data_compute_time_s,
            fps=args.fps,
            initial_rotation_correction=args.initial_rotation_correction,
            unwrap_root=not args.no_root_unwrap,
        )
    elif args.family == "triangulation":
        build_triangulation_bundle(
            name=args.name,
            output_dir=args.output_dir,
            pose_data=pose_data,
            pose_data_compute_time_s=pose_data_compute_time_s,
            calibrations=calibrations,
            fps=args.fps,
            initial_rotation_correction=args.initial_rotation_correction,
            unwrap_root=not args.no_root_unwrap,
            triangulation_method=args.triangulation_method,
            reprojection_threshold_px=args.reprojection_threshold_px,
            min_cameras_for_triangulation=args.min_cameras_for_triangulation,
            epipolar_threshold_px=args.epipolar_threshold_px,
            coherence_method=args.coherence_method,
            triangulation_workers=args.triangulation_workers,
            pose_data_mode=args.pose_data_mode,
            pose_filter_window=args.pose_filter_window,
            pose_outlier_threshold_ratio=args.pose_outlier_threshold_ratio,
            pose_amplitude_lower_percentile=args.pose_amplitude_lower_percentile,
            pose_amplitude_upper_percentile=args.pose_amplitude_upper_percentile,
            flip_left_right=args.flip_left_right,
            flip_improvement_ratio=args.flip_improvement_ratio,
            flip_min_gain_px=args.flip_min_gain_px,
            flip_min_other_cameras=args.flip_min_other_cameras,
            flip_restrict_to_outliers=not args.flip_test_all_camera_frames,
            flip_outlier_percentile=args.flip_outlier_percentile,
            flip_outlier_floor_px=args.flip_outlier_floor_px,
            flip_temporal_weight=args.flip_temporal_weight,
            flip_temporal_tau_px=args.flip_temporal_tau_px,
            flip_temporal_min_valid_keypoints=args.flip_temporal_min_valid_keypoints,
        )
    elif args.family == "ekf_3d":
        build_ekf_3d_bundle(
            name=args.name,
            output_dir=args.output_dir,
            pose_data=pose_data,
            pose_data_compute_time_s=pose_data_compute_time_s,
            calibrations=calibrations,
            fps=args.fps,
            initial_rotation_correction=args.initial_rotation_correction,
            unwrap_root=not args.no_root_unwrap,
            triangulation_method=args.triangulation_method,
            reprojection_threshold_px=args.reprojection_threshold_px,
            min_cameras_for_triangulation=args.min_cameras_for_triangulation,
            epipolar_threshold_px=args.epipolar_threshold_px,
            coherence_method=args.coherence_method,
            triangulation_workers=args.triangulation_workers,
            pose_data_mode=args.pose_data_mode,
            pose_filter_window=args.pose_filter_window,
            pose_outlier_threshold_ratio=args.pose_outlier_threshold_ratio,
            pose_amplitude_lower_percentile=args.pose_amplitude_lower_percentile,
            pose_amplitude_upper_percentile=args.pose_amplitude_upper_percentile,
            flip_left_right=args.flip_left_right,
            flip_improvement_ratio=args.flip_improvement_ratio,
            flip_min_gain_px=args.flip_min_gain_px,
            flip_min_other_cameras=args.flip_min_other_cameras,
            flip_restrict_to_outliers=not args.flip_test_all_camera_frames,
            flip_outlier_percentile=args.flip_outlier_percentile,
            flip_outlier_floor_px=args.flip_outlier_floor_px,
            flip_temporal_weight=args.flip_temporal_weight,
            flip_temporal_tau_px=args.flip_temporal_tau_px,
            flip_temporal_min_valid_keypoints=args.flip_temporal_min_valid_keypoints,
            subject_mass_kg=args.subject_mass_kg,
            biorbd_kalman_noise_factor=args.biorbd_kalman_noise_factor,
            biorbd_kalman_error_factor=args.biorbd_kalman_error_factor,
            biorbd_kalman_init_method=args.biorbd_kalman_init_method,
        )
    else:
        build_ekf_2d_bundle(
            name=args.name,
            output_dir=args.output_dir,
            pose_data=pose_data,
            pose_data_compute_time_s=pose_data_compute_time_s,
            calibrations=calibrations,
            fps=args.fps,
            initial_rotation_correction=args.initial_rotation_correction,
            unwrap_root=not args.no_root_unwrap,
            triangulation_method=args.triangulation_method,
            reprojection_threshold_px=args.reprojection_threshold_px,
            min_cameras_for_triangulation=args.min_cameras_for_triangulation,
            epipolar_threshold_px=args.epipolar_threshold_px,
            coherence_method=args.coherence_method,
            triangulation_workers=args.triangulation_workers,
            pose_data_mode=args.pose_data_mode,
            pose_filter_window=args.pose_filter_window,
            pose_outlier_threshold_ratio=args.pose_outlier_threshold_ratio,
            pose_amplitude_lower_percentile=args.pose_amplitude_lower_percentile,
            pose_amplitude_upper_percentile=args.pose_amplitude_upper_percentile,
            subject_mass_kg=args.subject_mass_kg,
            predictor=args.predictor,
            ekf2d_3d_source=args.ekf2d_3d_source,
            ekf2d_initial_state_method=args.ekf2d_initial_state_method,
            ekf2d_bootstrap_passes=args.ekf2d_bootstrap_passes,
            flip_left_right=args.flip_left_right,
            flip_improvement_ratio=args.flip_improvement_ratio,
            flip_min_gain_px=args.flip_min_gain_px,
            flip_min_other_cameras=args.flip_min_other_cameras,
            flip_restrict_to_outliers=not args.flip_test_all_camera_frames,
            flip_outlier_percentile=args.flip_outlier_percentile,
            flip_outlier_floor_px=args.flip_outlier_floor_px,
            flip_temporal_weight=args.flip_temporal_weight,
            flip_temporal_tau_px=args.flip_temporal_tau_px,
            flip_temporal_min_valid_keypoints=args.flip_temporal_min_valid_keypoints,
            enable_dof_locking=args.enable_dof_locking,
            measurement_noise_scale=args.measurement_noise_scale,
            process_noise_scale=args.process_noise_scale,
            coherence_confidence_floor=args.coherence_confidence_floor,
            min_frame_coherence_for_update=args.min_frame_coherence_for_update,
            skip_low_coherence_updates=args.skip_low_coherence_updates,
            flight_height_threshold_m=args.flight_height_threshold_m,
            flight_min_consecutive_frames=args.flight_min_consecutive_frames,
        )

    print(f"Bundle ecrit dans {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
