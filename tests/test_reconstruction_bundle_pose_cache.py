import numpy as np

from reconstruction_bundle import epipolar_cache_metadata, load_or_compute_pose_data_variant_cache
from vitpose_ekf_pipeline import KP_INDEX, PoseData, apply_left_right_flip_corrections, reconstruction_cache_metadata


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
        pose_data_mode="cleaned",
        pose_filter_window=9,
        pose_outlier_threshold_ratio=0.1,
        pose_amplitude_lower_percentile=5.0,
        pose_amplitude_upper_percentile=95.0,
    )
    corrected_epipolar_metadata = epipolar_cache_metadata(
        corrected,
        epipolar_threshold_px=15.0,
        pose_data_mode="cleaned",
        pose_filter_window=9,
        pose_outlier_threshold_ratio=0.1,
        pose_amplitude_lower_percentile=5.0,
        pose_amplitude_upper_percentile=95.0,
    )
    assert epipolar_metadata["pose_data_signature"] != corrected_epipolar_metadata["pose_data_signature"]


def test_pose_data_variant_cache_reuses_corrected_flip_variant(tmp_path, monkeypatch):
    pose_data = _make_pose_data()
    call_count = {"count": 0}

    def fake_flip_cache(**_kwargs):
        call_count["count"] += 1
        suspect_mask = np.array([[True, False]], dtype=bool)
        diagnostics = {"method": "epipolar", "n_suspects": 1}
        return suspect_mask, diagnostics, 0.123, tmp_path / "flip_cache.npz", "computed_now"

    monkeypatch.setattr("reconstruction_bundle.load_or_compute_left_right_flip_cache", fake_flip_cache)

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
