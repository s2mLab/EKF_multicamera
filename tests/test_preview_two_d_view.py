import numpy as np

from preview.two_d_view import adjust_image_levels, camera_layout, compute_pose_crop_limits_2d


class _DummyCalibration:
    def __init__(self, image_size: tuple[int, int]) -> None:
        self.image_size = image_size


def test_camera_layout_uses_more_screen_space_for_small_camera_sets():
    assert camera_layout(1) == (1, 1)
    assert camera_layout(2) == (1, 2)
    assert camera_layout(4) == (2, 2)
    assert camera_layout(5) == (2, 3)


def test_compute_pose_crop_limits_2d_fills_missing_frames_from_neighbors():
    raw_2d = np.full((1, 3, 2, 2), np.nan, dtype=float)
    raw_2d[0, 0] = np.array([[100.0, 50.0], [120.0, 90.0]])
    raw_2d[0, 2] = np.array([[140.0, 70.0], [180.0, 110.0]])
    calibrations = {"camA": _DummyCalibration((640, 480))}

    limits = compute_pose_crop_limits_2d(raw_2d, calibrations, ["camA"], margin=0.2)

    camera_limits = limits["camA"]
    assert camera_limits.shape == (3, 4)
    assert np.all(np.isfinite(camera_limits))
    np.testing.assert_allclose(camera_limits[1], camera_limits[0])


def test_adjust_image_levels_keeps_dtype_and_changes_rgb_levels():
    image = np.array(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=np.uint8,
    )

    adjusted = adjust_image_levels(image, brightness=1.1, contrast=1.2)

    assert adjusted.dtype == np.uint8
    assert adjusted.shape == image.shape
    assert not np.array_equal(adjusted, image)
