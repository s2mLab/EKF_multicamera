import numpy as np

from preview import two_d_view
from preview.two_d_view import (
    adjust_image_levels,
    camera_layout,
    compute_pose_crop_limits_2d,
    crop_limits_from_points,
    load_camera_background_image,
)


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


def test_crop_limits_from_points_falls_back_to_full_image_without_visible_points():
    x_limits, y_limits = crop_limits_from_points(np.full((2, 2), np.nan), width=640, height=480, margin=0.2)

    assert x_limits == (0.0, 640.0)
    assert y_limits == (480.0, 0.0)


def test_load_camera_background_image_uses_resolved_path_and_adjustment(monkeypatch, tmp_path):
    image_path = tmp_path / "cam_frame.png"
    image_path.write_bytes(b"fake")
    calls = []

    monkeypatch.setattr(
        two_d_view,
        "resolve_execution_image_path",
        lambda root, camera_name, frame_number: calls.append((root, camera_name, frame_number)) or image_path,
    )

    loaded = load_camera_background_image(
        tmp_path,
        "camA",
        12,
        image_reader=lambda _path: np.ones((2, 2, 3), dtype=float) * 0.5,
        brightness=1.1,
        contrast=0.9,
    )

    assert loaded is not None
    assert loaded.shape == (2, 2, 3)
    assert calls == [(tmp_path, "camA", 12)]
