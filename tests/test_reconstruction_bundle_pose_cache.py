import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from reconstruction.reconstruction_bundle import (
    build_bundle_payload,
    build_triangulation_bundle,
    epipolar_cache_metadata,
    load_or_build_model_cache,
    load_or_compute_pose_data_variant_cache,
    load_or_compute_triangulation_cache,
    summarize_view_usage,
)
from vitpose_ekf_pipeline import (
    KP_INDEX,
    CameraCalibration,
    PoseData,
    SegmentLengths,
    apply_left_right_flip_corrections,
    biorbd_kalman_cache_metadata,
    calibration_signature,
    metadata_cache_matches,
    model_stage_cache_matches,
    model_stage_metadata,
    reconstruction_cache_metadata,
    save_model_stage,
)


def _make_pose_data() -> PoseData:
    keypoints = np.full((1, 2, 17, 2), np.nan, dtype=float)
    scores = np.zeros((1, 2, 17), dtype=float)
    raw_keypoints = np.full_like(keypoints, np.nan)
    filtered_keypoints = np.full_like(keypoints, np.nan)
    left_idx = KP_INDEX["left_shoulder"]
    right_idx = KP_INDEX["right_shoulder"]
    keypoints[0, 0, left_idx] = [10.0, 1.0]
    keypoints[0, 0, right_idx] = [20.0, 2.0]
    raw_keypoints[0, 0, left_idx] = [11.0, 1.1]
    raw_keypoints[0, 0, right_idx] = [21.0, 2.1]
    filtered_keypoints[0, 0, left_idx] = [12.0, 1.2]
    filtered_keypoints[0, 0, right_idx] = [22.0, 2.2]
    scores[0, 0, left_idx] = 0.9
    scores[0, 0, right_idx] = 0.8
    return PoseData(
        camera_names=["cam0"],
        frames=np.array([100, 101], dtype=int),
        keypoints=keypoints,
        scores=scores,
        raw_keypoints=raw_keypoints,
        filtered_keypoints=filtered_keypoints,
    )


def _make_synthetic_camera(name: str, tx_m: float) -> CameraCalibration:
    intrinsics = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 480.0], [0.0, 0.0, 1.0]], dtype=float)
    rotation = np.eye(3, dtype=float)
    translation = np.array([[tx_m], [0.0], [0.0]], dtype=float)
    return CameraCalibration(
        name=name,
        image_size=(1280, 960),
        K=intrinsics,
        dist=np.zeros(5, dtype=float),
        rvec=np.zeros(3, dtype=float),
        tvec=translation,
        R=rotation,
        P=intrinsics @ np.hstack((rotation, translation)),
    )


def _single_camera_calibrations() -> dict[str, CameraCalibration]:
    return {"cam0": _make_synthetic_camera("cam0", 0.0)}


def _make_synthetic_bundle_inputs() -> tuple[PoseData, dict[str, CameraCalibration], np.ndarray]:
    cameras = [
        _make_synthetic_camera("cam0", 0.0),
        _make_synthetic_camera("cam1", 1.0),
        _make_synthetic_camera("cam2", -0.8),
    ]
    frame_zero = np.array(
        [
            [0.00, 0.00, 5.60],
            [0.02, 0.05, 5.58],
            [0.02, -0.05, 5.58],
            [0.03, 0.12, 5.54],
            [0.03, -0.12, 5.54],
            [0.00, 0.35, 5.00],
            [0.00, -0.35, 5.00],
            [0.08, 0.68, 4.72],
            [0.08, -0.68, 4.72],
            [0.14, 0.92, 4.48],
            [0.14, -0.92, 4.48],
            [0.00, 0.20, 4.00],
            [0.00, -0.20, 4.00],
            [0.06, 0.22, 3.05],
            [0.06, -0.22, 3.05],
            [0.12, 0.23, 2.10],
            [0.12, -0.23, 2.10],
        ],
        dtype=float,
    )
    expected_points = np.stack([frame_zero + np.array([0.02 * frame_idx, 0.0, 0.0]) for frame_idx in range(3)])
    keypoints = np.stack(
        [
            np.stack([[camera.project_point(point) for point in frame_points] for frame_points in expected_points])
            for camera in cameras
        ]
    )
    pose_data = PoseData(
        camera_names=[camera.name for camera in cameras],
        frames=np.array([0, 2, 4], dtype=int),
        keypoints=keypoints,
        scores=np.ones((len(cameras), len(expected_points), len(KP_INDEX)), dtype=float),
        frame_stride=2,
    )
    return pose_data, {camera.name: camera for camera in cameras}, expected_points


def test_apply_left_right_flip_corrections_preserves_raw_and_filtered_variants():
    pose_data = _make_pose_data()
    suspect_mask = np.array([[True, False]], dtype=bool)

    corrected = apply_left_right_flip_corrections(pose_data, suspect_mask)

    left_idx = KP_INDEX["left_shoulder"]
    right_idx = KP_INDEX["right_shoulder"]
    np.testing.assert_allclose(corrected.keypoints[0, 0, left_idx], [20.0, 2.0])
    np.testing.assert_allclose(corrected.keypoints[0, 0, right_idx], [10.0, 1.0])
    np.testing.assert_allclose(corrected.raw_keypoints[0, 0, left_idx], [21.0, 2.1])
    np.testing.assert_allclose(corrected.raw_keypoints[0, 0, right_idx], [11.0, 1.1])
    np.testing.assert_allclose(corrected.filtered_keypoints[0, 0, left_idx], [22.0, 2.2])
    np.testing.assert_allclose(corrected.filtered_keypoints[0, 0, right_idx], [12.0, 1.2])


def test_cache_metadata_changes_after_pose_data_flip():
    pose_data = _make_pose_data()
    corrected = apply_left_right_flip_corrections(pose_data, np.array([[True, False]], dtype=bool))

    reconstruction_metadata = reconstruction_cache_metadata(
        pose_data,
        error_threshold_px=10.0,
        min_cameras_for_triangulation=2,
        epipolar_threshold_px=15.0,
        triangulation_method="exhaustive",
        pose_data_mode="cleaned",
        pose_filter_window=9,
        pose_outlier_threshold_ratio=0.1,
        pose_amplitude_lower_percentile=5.0,
        pose_amplitude_upper_percentile=95.0,
        calibrations=_single_camera_calibrations(),
    )
    corrected_reconstruction_metadata = reconstruction_cache_metadata(
        corrected,
        error_threshold_px=10.0,
        min_cameras_for_triangulation=2,
        epipolar_threshold_px=15.0,
        triangulation_method="exhaustive",
        pose_data_mode="cleaned",
        pose_filter_window=9,
        pose_outlier_threshold_ratio=0.1,
        pose_amplitude_lower_percentile=5.0,
        pose_amplitude_upper_percentile=95.0,
        calibrations=_single_camera_calibrations(),
    )
    assert reconstruction_metadata["pose_data_signature"] != corrected_reconstruction_metadata["pose_data_signature"]

    epipolar_metadata = epipolar_cache_metadata(
        pose_data,
        epipolar_threshold_px=15.0,
        distance_mode="sampson",
        pose_data_mode="cleaned",
        pose_filter_window=9,
        pose_outlier_threshold_ratio=0.1,
        pose_amplitude_lower_percentile=5.0,
        pose_amplitude_upper_percentile=95.0,
        calibrations=_single_camera_calibrations(),
    )
    corrected_epipolar_metadata = epipolar_cache_metadata(
        corrected,
        epipolar_threshold_px=15.0,
        distance_mode="sampson",
        pose_data_mode="cleaned",
        pose_filter_window=9,
        pose_outlier_threshold_ratio=0.1,
        pose_amplitude_lower_percentile=5.0,
        pose_amplitude_upper_percentile=95.0,
        calibrations=_single_camera_calibrations(),
    )
    assert epipolar_metadata["pose_data_signature"] != corrected_epipolar_metadata["pose_data_signature"]


def test_epipolar_cache_metadata_distinguishes_fast_distance_mode():
    pose_data = _make_pose_data()

    sampson_metadata = epipolar_cache_metadata(
        pose_data,
        epipolar_threshold_px=15.0,
        distance_mode="sampson",
        pose_data_mode="cleaned",
        pose_filter_window=9,
        pose_outlier_threshold_ratio=0.1,
        pose_amplitude_lower_percentile=5.0,
        pose_amplitude_upper_percentile=95.0,
        calibrations=_single_camera_calibrations(),
    )
    fast_metadata = epipolar_cache_metadata(
        pose_data,
        epipolar_threshold_px=15.0,
        distance_mode="symmetric",
        pose_data_mode="cleaned",
        pose_filter_window=9,
        pose_outlier_threshold_ratio=0.1,
        pose_amplitude_lower_percentile=5.0,
        pose_amplitude_upper_percentile=95.0,
        calibrations=_single_camera_calibrations(),
    )

    assert sampson_metadata["distance_mode"] == "sampson"
    assert fast_metadata["distance_mode"] == "symmetric"


def test_calibration_signatures_invalidate_epipolar_and_triangulation_metadata():
    pose_data, calibrations, _expected_points = _make_synthetic_bundle_inputs()
    modified_calibrations = dict(calibrations)
    original = calibrations["cam1"]
    modified_intrinsics = np.array(original.K, copy=True)
    modified_intrinsics[0, 0] += 1.0
    modified_calibrations["cam1"] = CameraCalibration(
        name=original.name,
        image_size=original.image_size,
        K=modified_intrinsics,
        dist=np.array(original.dist, copy=True),
        rvec=np.array(original.rvec, copy=True),
        tvec=np.array(original.tvec, copy=True),
        R=np.array(original.R, copy=True),
        P=modified_intrinsics @ np.hstack((original.R, original.tvec)),
    )

    assert calibration_signature(calibrations, pose_data.camera_names) != calibration_signature(
        modified_calibrations, pose_data.camera_names
    )
    epipolar_metadata = epipolar_cache_metadata(
        pose_data,
        epipolar_threshold_px=15.0,
        distance_mode="sampson",
        pose_data_mode="raw",
        pose_filter_window=9,
        pose_outlier_threshold_ratio=0.1,
        pose_amplitude_lower_percentile=5.0,
        pose_amplitude_upper_percentile=95.0,
        calibrations=calibrations,
    )
    modified_epipolar_metadata = epipolar_cache_metadata(
        pose_data,
        epipolar_threshold_px=15.0,
        distance_mode="sampson",
        pose_data_mode="raw",
        pose_filter_window=9,
        pose_outlier_threshold_ratio=0.1,
        pose_amplitude_lower_percentile=5.0,
        pose_amplitude_upper_percentile=95.0,
        calibrations=modified_calibrations,
    )
    triangulation_metadata = reconstruction_cache_metadata(
        pose_data,
        error_threshold_px=15.0,
        min_cameras_for_triangulation=2,
        epipolar_threshold_px=15.0,
        triangulation_method="once",
        pose_data_mode="raw",
        pose_filter_window=9,
        pose_outlier_threshold_ratio=0.1,
        pose_amplitude_lower_percentile=5.0,
        pose_amplitude_upper_percentile=95.0,
        calibrations=calibrations,
    )
    modified_triangulation_metadata = reconstruction_cache_metadata(
        pose_data,
        error_threshold_px=15.0,
        min_cameras_for_triangulation=2,
        epipolar_threshold_px=15.0,
        triangulation_method="once",
        pose_data_mode="raw",
        pose_filter_window=9,
        pose_outlier_threshold_ratio=0.1,
        pose_amplitude_lower_percentile=5.0,
        pose_amplitude_upper_percentile=95.0,
        calibrations=modified_calibrations,
    )

    assert epipolar_metadata["calibration_signature"] != modified_epipolar_metadata["calibration_signature"]
    assert triangulation_metadata["calibration_signature"] != modified_triangulation_metadata["calibration_signature"]


def test_model_and_biorbd_cache_metadata_track_content_changes(tmp_path):
    reconstruction = SimpleNamespace(
        frames=np.array([0, 1], dtype=int), points_3d=np.zeros((2, len(KP_INDEX), 3), dtype=float)
    )
    changed_reconstruction = SimpleNamespace(
        frames=np.array([0, 1], dtype=int), points_3d=np.ones((2, len(KP_INDEX), 3), dtype=float)
    )
    model_metadata = model_stage_metadata(tmp_path / "triangulation.npz", reconstruction, 120.0, 55.0, False)
    changed_model_metadata = model_stage_metadata(
        tmp_path / "triangulation.npz", changed_reconstruction, 120.0, 55.0, False
    )
    assert model_metadata["reconstruction_signature"] != changed_model_metadata["reconstruction_signature"]

    biomod_path = tmp_path / "model.bioMod"
    biomod_path.write_text("version 4", encoding="utf-8")
    first_kalman_metadata = biorbd_kalman_cache_metadata(
        tmp_path / "triangulation.npz", reconstruction, biomod_path, 120.0, 1e-8, 1e-4
    )
    biomod_path.write_text("version 4\nsegment trunk", encoding="utf-8")
    second_kalman_metadata = biorbd_kalman_cache_metadata(
        tmp_path / "triangulation.npz", reconstruction, biomod_path, 120.0, 1e-8, 1e-4
    )
    assert first_kalman_metadata["biomod_signature"] != second_kalman_metadata["biomod_signature"]


def _with_modified_focal_length(
    calibrations: dict[str, CameraCalibration], camera_name: str, delta_px: float
) -> dict[str, CameraCalibration]:
    original = calibrations[camera_name]
    intrinsics = np.array(original.K, copy=True)
    intrinsics[0, 0] += delta_px
    modified = dict(calibrations)
    modified[camera_name] = CameraCalibration(
        name=original.name,
        image_size=original.image_size,
        K=intrinsics,
        dist=np.array(original.dist, copy=True),
        rvec=np.array(original.rvec, copy=True),
        tvec=np.array(original.tvec, copy=True),
        R=np.array(original.R, copy=True),
        P=intrinsics @ np.hstack((original.R, original.tvec)),
    )
    return modified


def test_geometric_cache_metadata_requires_calibrations():
    pose_data = _make_pose_data()

    with pytest.raises(TypeError):
        reconstruction_cache_metadata(
            pose_data,
            error_threshold_px=10.0,
            min_cameras_for_triangulation=2,
            epipolar_threshold_px=15.0,
            triangulation_method="once",
            pose_data_mode="raw",
            pose_filter_window=9,
            pose_outlier_threshold_ratio=0.1,
            pose_amplitude_lower_percentile=5.0,
            pose_amplitude_upper_percentile=95.0,
        )
    with pytest.raises(TypeError):
        epipolar_cache_metadata(
            pose_data,
            epipolar_threshold_px=15.0,
            distance_mode="sampson",
            pose_data_mode="raw",
            pose_filter_window=9,
            pose_outlier_threshold_ratio=0.1,
            pose_amplitude_lower_percentile=5.0,
            pose_amplitude_upper_percentile=95.0,
        )


def test_triangulation_cache_recomputes_after_calibration_change(tmp_path):
    pose_data, calibrations, expected_points = _make_synthetic_bundle_inputs()
    modified_calibrations = _with_modified_focal_length(calibrations, "cam1", 50.0)
    triangulation_kwargs = {
        "output_dir": tmp_path,
        "pose_data": pose_data,
        "coherence_method": "epipolar",
        "triangulation_method": "once",
        "reprojection_threshold_px": None,
        "min_cameras_for_triangulation": 2,
        "epipolar_threshold_px": 15.0,
        "triangulation_workers": 1,
        "pose_data_mode": "raw",
        "pose_filter_window": 9,
        "pose_outlier_threshold_ratio": 0.1,
        "pose_amplitude_lower_percentile": 5.0,
        "pose_amplitude_upper_percentile": 95.0,
    }

    first, first_path, first_epipolar_path, first_source = load_or_compute_triangulation_cache(
        calibrations=calibrations, **triangulation_kwargs
    )
    reused, reused_path, _reused_epipolar_path, reused_source = load_or_compute_triangulation_cache(
        calibrations=calibrations, **triangulation_kwargs
    )
    modified, modified_path, modified_epipolar_path, modified_source = load_or_compute_triangulation_cache(
        calibrations=modified_calibrations, **triangulation_kwargs
    )
    _restored, restored_path, _restored_epipolar_path, restored_source = load_or_compute_triangulation_cache(
        calibrations=calibrations, **triangulation_kwargs
    )

    assert (first_source, reused_source, modified_source, restored_source) == (
        "computed_now",
        "cache",
        "computed_now",
        "cache",
    )
    assert reused_path == first_path == restored_path
    assert modified_path != first_path
    assert modified_epipolar_path != first_epipolar_path
    np.testing.assert_allclose(first.points_3d, expected_points, atol=1e-8)
    np.testing.assert_allclose(reused.points_3d, expected_points, atol=1e-8)
    assert np.nanmax(np.abs(modified.points_3d - expected_points)) > 1e-3


def test_model_stage_cache_matches_requires_existing_unchanged_biomod(tmp_path):
    lengths = SegmentLengths(
        trunk_height=0.6,
        head_length=0.2,
        shoulder_half_width=0.18,
        hip_half_width=0.12,
        upper_arm_length=0.3,
        forearm_length=0.25,
        thigh_length=0.45,
        shank_length=0.4,
        eye_offset_x=0.03,
        eye_offset_y=0.025,
        ear_offset_y=0.06,
    )
    metadata = {"model_stage_version": 1, "model_variant": "single_trunk"}
    biomod_path = tmp_path / "model.bioMod"
    cache_path = tmp_path / "model_stage.npz"

    biomod_path.write_text("version 4\n", encoding="utf-8")
    save_model_stage(cache_path, lengths, biomod_path, metadata)
    assert model_stage_cache_matches(cache_path, metadata, biomod_path)

    biomod_path.write_text("version 4\nsegment trunk\n", encoding="utf-8")
    assert not model_stage_cache_matches(cache_path, metadata, biomod_path)

    biomod_path.unlink()
    assert not model_stage_cache_matches(cache_path, metadata, biomod_path)

    missing_biomod_cache_path = tmp_path / "model_stage_without_biomod.npz"
    save_model_stage(missing_biomod_cache_path, lengths, biomod_path, metadata)
    assert not model_stage_cache_matches(missing_biomod_cache_path, metadata, biomod_path)

    biomod_path.write_text("version 4\n", encoding="utf-8")
    legacy_cache_path = tmp_path / "legacy_model_stage.npz"
    np.savez(legacy_cache_path, metadata=np.asarray(json.dumps(metadata), dtype=object))
    assert metadata_cache_matches(legacy_cache_path, metadata)
    assert not model_stage_cache_matches(legacy_cache_path, metadata, biomod_path)


def test_biorbd_kalman_cache_rejects_changed_reconstruction_and_legacy_metadata(tmp_path):
    reconstruction = SimpleNamespace(
        frames=np.array([0, 1], dtype=int), points_3d=np.zeros((2, len(KP_INDEX), 3), dtype=float)
    )
    changed_reconstruction = SimpleNamespace(
        frames=np.array([0, 1], dtype=int), points_3d=np.full((2, len(KP_INDEX), 3), 0.01, dtype=float)
    )
    biomod_path = tmp_path / "model.bioMod"
    biomod_path.write_text("version 4", encoding="utf-8")
    triangulation_path = tmp_path / "triangulation.npz"
    metadata = biorbd_kalman_cache_metadata(triangulation_path, reconstruction, biomod_path, 120.0, 1e-8, 1e-4)
    changed_metadata = biorbd_kalman_cache_metadata(
        triangulation_path, changed_reconstruction, biomod_path, 120.0, 1e-8, 1e-4
    )
    cache_path = tmp_path / "biorbd_kalman_states.npz"
    np.savez(cache_path, metadata=np.asarray(json.dumps(metadata), dtype=object))
    legacy_metadata = {key: value for key, value in metadata.items() if key != "reconstruction_signature"}
    legacy_cache_path = tmp_path / "legacy_biorbd_kalman_states.npz"
    np.savez(legacy_cache_path, metadata=np.asarray(json.dumps(legacy_metadata), dtype=object))

    assert metadata["reconstruction_frame_signature"] == changed_metadata["reconstruction_frame_signature"]
    assert metadata["reconstruction_signature"] != changed_metadata["reconstruction_signature"]
    assert metadata_cache_matches(cache_path, metadata)
    assert not metadata_cache_matches(cache_path, changed_metadata)
    assert not metadata_cache_matches(legacy_cache_path, metadata)


def test_pose_data_variant_cache_reuses_corrected_flip_variant(tmp_path, monkeypatch):
    pose_data = _make_pose_data()
    calibrations = {"cam0": _make_synthetic_camera("cam0", 0.0)}
    call_count = {"count": 0}

    def fake_flip_cache(**_kwargs):
        call_count["count"] += 1
        suspect_mask = np.array([[True, False]], dtype=bool)
        diagnostics = {"method": "epipolar", "n_suspects": 1}
        return suspect_mask, diagnostics, 0.123, tmp_path / "flip_cache.npz", "computed_now"

    monkeypatch.setattr("reconstruction.reconstruction_bundle.load_or_compute_left_right_flip_cache", fake_flip_cache)

    corrected_a, diagnostics_a, compute_time_a, cache_path, source_a = load_or_compute_pose_data_variant_cache(
        output_dir=tmp_path,
        pose_data=pose_data,
        calibrations=calibrations,
        correction_mode="flip",
        flip_method="epipolar",
        pose_data_mode="cleaned",
        pose_filter_window=9,
        pose_outlier_threshold_ratio=0.1,
        pose_amplitude_lower_percentile=5.0,
        pose_amplitude_upper_percentile=95.0,
    )
    corrected_b, diagnostics_b, compute_time_b, cache_path_b, source_b = load_or_compute_pose_data_variant_cache(
        output_dir=tmp_path,
        pose_data=pose_data,
        calibrations=calibrations,
        correction_mode="flip",
        flip_method="epipolar",
        pose_data_mode="cleaned",
        pose_filter_window=9,
        pose_outlier_threshold_ratio=0.1,
        pose_amplitude_lower_percentile=5.0,
        pose_amplitude_upper_percentile=95.0,
    )

    assert call_count["count"] == 1
    assert cache_path == cache_path_b
    assert source_a == "computed_now"
    assert source_b == "cache"
    assert abs(compute_time_a - 0.123) < 1e-12
    assert abs(compute_time_b - 0.123) < 1e-12
    assert diagnostics_a["method"] == diagnostics_b["method"]
    assert diagnostics_a["n_suspects"] == diagnostics_b["n_suspects"]
    assert diagnostics_a["source"] == "computed_now"
    assert diagnostics_b["source"] == "cache"
    left_idx = KP_INDEX["left_shoulder"]
    right_idx = KP_INDEX["right_shoulder"]
    np.testing.assert_allclose(corrected_a.keypoints[0, 0, left_idx], [20.0, 2.0])
    np.testing.assert_allclose(corrected_b.keypoints[0, 0, right_idx], [10.0, 1.0])


def test_load_or_build_model_cache_records_full_model_stage_time(tmp_path, monkeypatch):
    reconstruction = SimpleNamespace(
        frames=np.array([0, 1, 2], dtype=int), points_3d=np.zeros((3, len(KP_INDEX), 3), dtype=float)
    )
    lengths = SegmentLengths(
        trunk_height=0.6,
        head_length=0.2,
        shoulder_half_width=0.18,
        hip_half_width=0.12,
        upper_arm_length=0.3,
        forearm_length=0.25,
        thigh_length=0.45,
        shank_length=0.4,
        eye_offset_x=0.03,
        eye_offset_y=0.025,
        ear_offset_y=0.06,
    )

    monkeypatch.setattr(
        "reconstruction.reconstruction_bundle.estimate_segment_lengths", lambda *_args, **_kwargs: lengths
    )
    monkeypatch.setattr(
        "reconstruction.reconstruction_bundle.build_biomod",
        lambda _lengths, output_path, **_kwargs: output_path.write_text("version 4", encoding="utf-8"),
    )
    perf_counter_values = iter((10.0, 14.5))
    monkeypatch.setattr("reconstruction.reconstruction_bundle.time.perf_counter", lambda: next(perf_counter_values))

    _cached_lengths, biomod_cache_path, cache_path, bootstrap_frame_idx, compute_time_s, source = (
        load_or_build_model_cache(
            output_dir=tmp_path,
            reconstruction=reconstruction,
            reconstruction_cache_path=tmp_path / "triangulation_stage.npz",
            fps=120.0,
            subject_mass_kg=70.0,
            initial_rotation_correction=True,
            lengths_mode="full_triangulation",
            model_variant="single_trunk",
            symmetrize_limbs=True,
        )
    )

    assert source == "computed_now"
    assert bootstrap_frame_idx == 0
    assert biomod_cache_path.exists()
    assert cache_path.exists()
    assert math.isclose(compute_time_s, 4.5)

    with np.load(cache_path, allow_pickle=True) as data:
        assert math.isclose(float(np.asarray(data["compute_time_s"]).item()), 4.5)


def test_reconstruction_cache_metadata_and_match_support_none_threshold(tmp_path):
    pose_data = _make_pose_data()
    metadata = reconstruction_cache_metadata(
        pose_data,
        error_threshold_px=None,
        min_cameras_for_triangulation=2,
        epipolar_threshold_px=15.0,
        triangulation_method="once",
        pose_data_mode="raw",
        pose_filter_window=9,
        pose_outlier_threshold_ratio=0.1,
        pose_amplitude_lower_percentile=5.0,
        pose_amplitude_upper_percentile=95.0,
        calibrations=_single_camera_calibrations(),
    )
    np.savez(tmp_path / "cache.npz", metadata=np.asarray(json.dumps(metadata), dtype=object))

    assert metadata["reprojection_threshold_px"] is None
    assert metadata_cache_matches(tmp_path / "cache.npz", metadata)


def test_build_bundle_payload_includes_excluded_views():
    excluded_views = np.ones((1, 17, 1), dtype=bool)
    excluded_views[0, 0, 0] = False
    payload = build_bundle_payload(
        name="demo",
        family="triangulation",
        frames=np.array([0], dtype=int),
        time_s=np.array([0.0], dtype=float),
        camera_names=["cam0"],
        points_3d=np.full((1, 17, 3), np.nan, dtype=float),
        q_names=np.array([], dtype=object),
        q=None,
        qdot=None,
        qddot=None,
        q_root=np.zeros((1, 6), dtype=float),
        qdot_root=np.zeros((1, 6), dtype=float),
        reprojection_errors=np.full((1, 17, 1), np.nan, dtype=float),
        summary={},
        excluded_views=excluded_views,
    )

    np.testing.assert_array_equal(payload["excluded_views"], excluded_views)


def test_synthetic_triangulation_bundle_writes_outputs_and_reuses_caches(tmp_path):
    pose_data, calibrations, expected_points = _make_synthetic_bundle_inputs()
    bundle_kwargs = {
        "name": "synthetic_once",
        "output_dir": tmp_path / "reconstruction",
        "pose_data": pose_data,
        "calibrations": calibrations,
        "fps": 120.0,
        "initial_rotation_correction": False,
        "unwrap_root": False,
        "triangulation_method": "once",
        "reprojection_threshold_px": 1e-6,
        "min_cameras_for_triangulation": 2,
        "epipolar_threshold_px": 15.0,
        "coherence_method": "epipolar",
        "triangulation_workers": 1,
        "pose_data_mode": "raw",
        "pose_filter_window": 9,
        "pose_outlier_threshold_ratio": 0.1,
        "pose_amplitude_lower_percentile": 5.0,
        "pose_amplitude_upper_percentile": 95.0,
        "flip_left_right": False,
        "flip_improvement_ratio": 0.7,
        "flip_min_gain_px": 3.0,
        "flip_min_other_cameras": 2,
        "flip_restrict_to_outliers": True,
        "flip_outlier_percentile": 85.0,
        "flip_outlier_floor_px": 5.0,
        "flip_temporal_weight": 0.35,
        "flip_temporal_tau_px": 20.0,
        "flip_temporal_min_valid_keypoints": 4,
    }

    first_bundle, first_reconstruction = build_triangulation_bundle(**bundle_kwargs)
    second_bundle, second_reconstruction = build_triangulation_bundle(**bundle_kwargs)

    np.testing.assert_allclose(first_reconstruction.points_3d, expected_points, atol=1e-8)
    np.testing.assert_allclose(second_reconstruction.points_3d, expected_points, atol=1e-8)
    assert np.all(first_reconstruction.reprojection_error_per_view < 1e-8)
    assert not np.any(first_reconstruction.excluded_views)
    np.testing.assert_allclose(
        first_bundle.payload["q_root"][:, :3],
        expected_points[:, [KP_INDEX["left_hip"], KP_INDEX["right_hip"]]].mean(axis=1),
        atol=1e-8,
    )
    assert first_bundle.summary["family"] == "triangulation"
    assert first_bundle.summary["fps"] == 60.0
    assert first_bundle.summary["source_fps"] == 120.0
    assert first_bundle.summary["frame_stride"] == 2
    assert (bundle_kwargs["output_dir"] / "reconstruction_bundle.npz").exists()
    assert (bundle_kwargs["output_dir"] / "bundle_summary.json").exists()
    assert Path(first_bundle.summary["cache_paths"]["epipolar"]).exists()
    assert Path(first_bundle.summary["cache_paths"]["triangulation"]).exists()
    first_triangulation_stage = next(
        stage for stage in first_bundle.summary["pipeline_timing"]["stages"] if stage["id"] == "triangulation"
    )
    second_triangulation_stage = next(
        stage for stage in second_bundle.summary["pipeline_timing"]["stages"] if stage["id"] == "triangulation"
    )
    assert first_triangulation_stage["source"] == "computed_now"
    assert second_triangulation_stage["source"] == "cache"


def test_summarize_view_usage_reports_included_and_excluded_ratios():
    excluded_views = np.array(
        [
            [[False, True], [True, True]],
            [[False, False], [True, False]],
        ],
        dtype=bool,
    )

    stats = summarize_view_usage(excluded_views, ["cam0", "cam1"])

    assert math.isclose(stats["included_ratio"], 0.5)
    assert math.isclose(stats["excluded_ratio"], 0.5)
    assert math.isclose(stats["per_camera"]["cam0"]["included_ratio"], 0.5)
    assert math.isclose(stats["per_camera"]["cam1"]["excluded_ratio"], 0.5)
