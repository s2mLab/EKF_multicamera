import numpy as np

from camera_selection import parse_camera_names, select_camera_names, subset_calibrations, subset_pose_data
from vitpose_ekf_pipeline import PoseData


def test_parse_camera_names_deduplicates_and_strips():
    assert parse_camera_names("cam3, cam1, cam3 ; cam2") == ["cam3", "cam1", "cam2"]


def test_select_camera_names_validates_presence():
    assert select_camera_names(["cam1", "cam2"], ["cam2"]) == ["cam2"]
    try:
        select_camera_names(["cam1"], ["cam9"])
    except ValueError as exc:
        assert "cam9" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_subset_pose_data_slices_all_camera_axes():
    pose_data = PoseData(
        camera_names=["cam1", "cam2", "cam3"],
        frames=np.array([0, 1]),
        keypoints=np.arange(3 * 2 * 17 * 2, dtype=float).reshape(3, 2, 17, 2),
        scores=np.arange(3 * 2 * 17, dtype=float).reshape(3, 2, 17),
        raw_keypoints=np.arange(3 * 2 * 17 * 2, dtype=float).reshape(3, 2, 17, 2) + 100.0,
        filtered_keypoints=np.arange(3 * 2 * 17 * 2, dtype=float).reshape(3, 2, 17, 2) + 200.0,
    )
    subset = subset_pose_data(pose_data, ["cam3", "cam1"])
    assert subset.camera_names == ["cam3", "cam1"]
    np.testing.assert_array_equal(subset.keypoints[0], pose_data.keypoints[2])
    np.testing.assert_array_equal(subset.scores[1], pose_data.scores[0])
    np.testing.assert_array_equal(subset.raw_keypoints[0], pose_data.raw_keypoints[2])
    np.testing.assert_array_equal(subset.filtered_keypoints[1], pose_data.filtered_keypoints[0])


def test_subset_calibrations_preserves_requested_order():
    calibrations = {"cam1": object(), "cam2": object(), "cam3": object()}
    subset = subset_calibrations(calibrations, ["cam3", "cam1"])
    assert list(subset.keys()) == ["cam3", "cam1"]
