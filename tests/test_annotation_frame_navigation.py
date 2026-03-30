from pathlib import Path

import numpy as np

from annotation import frame_navigation


def test_resolve_annotation_frame_filter_mode_accepts_label_or_key():
    options = {"all": "All frames", "worst_reproj": "Worst reproj 5%"}

    assert frame_navigation.resolve_annotation_frame_filter_mode("All frames", options) == "all"
    assert frame_navigation.resolve_annotation_frame_filter_mode("worst_reproj", options) == "worst_reproj"
    assert frame_navigation.resolve_annotation_frame_filter_mode("unknown", options) == "all"


def test_fallback_annotation_filtered_indices_uses_all_frames_when_empty():
    assert frame_navigation.fallback_annotation_filtered_indices(4, []) == [0, 1, 2, 3]
    assert frame_navigation.fallback_annotation_filtered_indices(4, [1, 3]) == [1, 3]


def test_step_frame_index_within_subset_wraps_and_skips_gaps():
    candidates = [2, 7, 9]

    assert frame_navigation.step_frame_index_within_subset(7, 1, candidates) == 9
    assert frame_navigation.step_frame_index_within_subset(9, 1, candidates) == 2
    assert frame_navigation.step_frame_index_within_subset(7, -1, candidates) == 2
    assert frame_navigation.step_frame_index_within_subset(2, -1, candidates) == 9
    assert frame_navigation.step_frame_index_within_subset(5, 1, candidates) == 7
    assert frame_navigation.step_frame_index_within_subset(5, -1, candidates) == 2


def test_clamp_index_to_subset_returns_current_or_first():
    assert frame_navigation.clamp_index_to_subset(3, [1, 3, 5]) == 3
    assert frame_navigation.clamp_index_to_subset(2, [1, 3, 5]) == 1
    assert frame_navigation.clamp_index_to_subset(2, []) is None


def test_navigable_annotation_frame_local_indices_filters_with_available_images(monkeypatch, tmp_path):
    frames = np.array([10, 11, 12, 13], dtype=int)
    images_root = tmp_path / "images"
    images_root.mkdir()

    monkeypatch.setattr(
        frame_navigation,
        "available_execution_image_frames",
        lambda _images_root, _camera_names: {"cam0": {10, 12}, "cam1": {13}},
    )

    navigable = frame_navigation.navigable_annotation_frame_local_indices(
        frames,
        filtered_indices=[0, 1, 2, 3],
        camera_names=["cam0"],
        images_root=images_root,
    )
    assert navigable == [0, 2]

    fallback = frame_navigation.navigable_annotation_frame_local_indices(
        frames,
        filtered_indices=[1, 3],
        camera_names=["cam2"],
        images_root=images_root,
    )
    assert fallback == [1, 3]
