import numpy as np

from camera_tools.camera_metrics import compute_camera_metric_rows, suggest_best_camera_names
from vitpose_ekf_pipeline import PoseData


def test_compute_camera_metric_rows_summarizes_expected_ratios():
    scores = np.zeros((2, 3, 17), dtype=float)
    keypoints = np.full((2, 3, 17, 2), np.nan, dtype=float)
    scores[0, :, 0] = [0.8, 0.7, 0.6]
    scores[1, :2, 0] = [0.9, 0.5]
    keypoints[0, :, 0] = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
    keypoints[1, :2, 0] = [[1.0, 2.0], [2.0, 3.0]]
    pose_data = PoseData(camera_names=["cam1", "cam2"], frames=np.array([0, 1, 2]), keypoints=keypoints, scores=scores)

    epipolar = np.full((3, 17, 2), np.nan, dtype=float)
    epipolar[:, 0, 0] = [0.9, 0.8, 0.7]
    epipolar[:2, 0, 1] = [0.5, 0.6]
    reproj = np.full((3, 17, 2), np.nan, dtype=float)
    reproj[:, 0, 0] = [2.0, 3.0, 4.0]
    reproj[:2, 0, 1] = [8.0, 9.0]
    excluded = np.ones((3, 17, 2), dtype=bool)
    excluded[:, 0, 0] = [False, False, True]
    excluded[:2, 0, 1] = [False, True]
    flip_masks = {
        "epipolar": np.array([[False, True, False], [True, False, False]], dtype=bool),
        "epipolar_fast": np.array([[False, False, True], [False, False, False]], dtype=bool),
        "triangulation": np.array([[False, False, False], [True, True, False]], dtype=bool),
    }

    rows = compute_camera_metric_rows(
        pose_data,
        epipolar_coherence=epipolar,
        reprojection_error_per_view=reproj,
        excluded_views=excluded,
        flip_masks=flip_masks,
        good_reprojection_threshold_px=5.0,
    )
    row1, row2 = rows
    assert row1.camera_name == "cam1"
    assert abs(row1.valid_ratio - (3 / (3 * 17))) < 1e-12
    assert abs(row1.mean_score - 0.7) < 1e-12
    assert abs(row1.reprojection_mean_px - 3.0) < 1e-12
    assert abs(row1.reprojection_good_frame_ratio - 1.0) < 1e-12
    assert abs(row1.triangulation_usage_ratio - (2 / 3)) < 1e-12
    assert abs(row1.flip_rate_epipolar - (1 / 3)) < 1e-12
    assert abs(row1.flip_rate_epipolar_fast - (1 / 3)) < 1e-12
    assert row2.reprojection_good_frame_ratio == 0.0


def test_suggest_best_camera_names_prefers_high_confidence_low_flip_rows():
    scores = np.zeros((3, 2, 17), dtype=float)
    keypoints = np.full((3, 2, 17, 2), np.nan, dtype=float)
    scores[0, :, 0] = [0.9, 0.8]
    scores[1, :, 0] = [0.6, 0.5]
    scores[2, :, 0] = [0.95, 0.95]
    keypoints[:, :, 0] = [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]
    pose_data = PoseData(camera_names=["cam1", "cam2", "cam3"], frames=np.array([0, 1]), keypoints=keypoints, scores=scores)
    epipolar = np.full((2, 17, 3), np.nan)
    epipolar[:, 0, 0] = [0.8, 0.8]
    epipolar[:, 0, 1] = [0.6, 0.6]
    epipolar[:, 0, 2] = [0.95, 0.95]
    flip_masks = {
        "epipolar": np.array([[False, False], [True, False], [False, False]], dtype=bool),
        "epipolar_fast": np.array([[False, False], [False, False], [False, True]], dtype=bool),
        "triangulation": np.array([[False, False], [False, True], [False, False]], dtype=bool),
    }
    rows = compute_camera_metric_rows(pose_data, epipolar_coherence=epipolar, flip_masks=flip_masks)
    assert suggest_best_camera_names(rows, 2) == ["cam3", "cam1"]
