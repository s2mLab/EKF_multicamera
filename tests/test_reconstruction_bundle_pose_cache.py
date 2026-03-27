import json
import math
from types import SimpleNamespace

import numpy as np

from reconstruction.reconstruction_bundle import (
    build_bundle_payload,
    epipolar_cache_metadata,
    load_or_build_model_cache,
    load_or_compute_pose_data_variant_cache,
    summarize_view_usage,
)
from vitpose_ekf_pipeline import (
    KP_INDEX,
    PoseData,
    SegmentLengths,
    apply_left_right_flip_corrections,
    metadata_cache_matches,
    reconstruction_cache_metadata,
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
    )

    assert sampson_metadata["distance_mode"] == "sampson"
    assert fast_metadata["distance_mode"] == "symmetric"


def test_pose_data_variant_cache_reuses_corrected_flip_variant(tmp_path, monkeypatch):
    pose_data = _make_pose_data()
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
        calibrations={},
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
        calibrations={},
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
    reconstruction = SimpleNamespace(frames=np.array([0, 1, 2], dtype=int))
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
