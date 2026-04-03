import numpy as np

from judging.jump_cache import get_cached_jump_analysis, jump_segmentation_height_series


def test_jump_segmentation_height_series_prefers_ankles_when_available():
    root_q = np.zeros((3, 6), dtype=float)
    points = np.full((3, 17, 3), np.nan, dtype=float)
    points[:, 15, 2] = np.array([1.2, 1.3, 1.4], dtype=float)
    points[:, 16, 2] = np.array([0.8, 0.7, 0.6], dtype=float)

    series = jump_segmentation_height_series(points, root_q)

    np.testing.assert_allclose(series, np.array([0.8, 0.7, 0.6], dtype=float))


def test_get_cached_jump_analysis_reuses_same_cached_object():
    cache = {}
    root_q = np.zeros((20, 6), dtype=float)
    points = np.full((20, 17, 3), np.nan, dtype=float)
    points[:, 5] = np.array([0.0, 0.2, 1.0], dtype=float)
    points[:, 6] = np.array([0.0, -0.2, 1.0], dtype=float)
    points[:, 11] = np.array([0.0, 0.2, 0.0], dtype=float)
    points[:, 12] = np.array([0.0, -0.2, 0.0], dtype=float)
    points[:, 13] = np.array([0.0, 0.2, -1.0], dtype=float)
    points[:, 14] = np.array([0.0, -0.2, -1.0], dtype=float)
    points[:, 15] = np.array([0.0, 0.2, -2.0], dtype=float)
    points[:, 16] = np.array([0.0, -0.2, -2.0], dtype=float)
    points[5:12, 15, 2] = 1.0
    points[5:12, 16, 2] = 1.0

    first = get_cached_jump_analysis(
        cache,
        reconstruction_name="demo",
        root_q=root_q,
        points_3d=points,
        fps=10.0,
        height_threshold=0.5,
        height_threshold_range_ratio=0.2,
        smoothing_window_s=0.0,
        min_airtime_s=0.2,
        min_gap_s=0.0,
        min_peak_prominence_m=0.2,
        contact_window_s=0.2,
        q_names=["TRUNK:TransZ"],
        angle_mode="euler",
        analysis_start_frame=0,
        require_complete_jumps=True,
    )
    second = get_cached_jump_analysis(
        cache,
        reconstruction_name="demo",
        root_q=root_q,
        points_3d=points,
        fps=10.0,
        height_threshold=0.5,
        height_threshold_range_ratio=0.2,
        smoothing_window_s=0.0,
        min_airtime_s=0.2,
        min_gap_s=0.0,
        min_peak_prominence_m=0.2,
        contact_window_s=0.2,
        q_names=["TRUNK:TransZ"],
        angle_mode="euler",
        analysis_start_frame=0,
        require_complete_jumps=True,
    )

    assert first is second
