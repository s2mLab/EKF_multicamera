from pathlib import Path

import numpy as np

from annotation.annotation_store import (
    apply_annotations_to_pose_arrays,
    clear_annotation_point,
    default_annotation_path,
    get_annotation_point,
    load_annotation_payload,
    save_annotation_payload,
    set_annotation_point,
)
from vitpose_ekf_pipeline import COCO17


def test_default_annotation_path_uses_inputs_annotations_folder():
    keypoints_path = Path("inputs/keypoints/1_partie_0429_keypoints.json")

    assert default_annotation_path(keypoints_path) == Path("inputs/annotations/1_partie_0429_annotations.json")


def test_sparse_annotation_payload_round_trip_and_clear(tmp_path):
    annotation_path = tmp_path / "trial_annotations.json"
    payload = load_annotation_payload(annotation_path, keypoints_path=Path("inputs/keypoints/trial_keypoints.json"))

    set_annotation_point(
        payload,
        camera_name="M11139",
        frame_number=12,
        keypoint_name="left_wrist",
        xy=[123.0, 456.0],
        score=0.95,
    )
    saved_path = save_annotation_payload(annotation_path, payload)
    reloaded = load_annotation_payload(saved_path)

    xy, score = get_annotation_point(
        reloaded,
        camera_name="M11139",
        frame_number=12,
        keypoint_name="left_wrist",
    )

    np.testing.assert_allclose(xy, np.array([123.0, 456.0], dtype=float))
    assert score == 0.95

    clear_annotation_point(
        reloaded,
        camera_name="M11139",
        frame_number=12,
        keypoint_name="left_wrist",
    )
    xy, score = get_annotation_point(
        reloaded,
        camera_name="M11139",
        frame_number=12,
        keypoint_name="left_wrist",
    )
    assert xy is None
    assert score is None


def test_apply_annotations_to_pose_arrays_overlays_sparse_points():
    keypoints = np.zeros((1, 2, 17, 2), dtype=float)
    scores = np.ones((1, 2, 17), dtype=float)
    payload = {"annotations": {"M11139": {"5": {"left_wrist": {"xy": [321.0, 654.0], "score": 0.8}}}}}

    annotated_keypoints, annotated_scores = apply_annotations_to_pose_arrays(
        keypoints=keypoints,
        scores=scores,
        camera_names=["M11139"],
        frames=np.array([4, 5], dtype=int),
        keypoint_names=COCO17,
        payload=payload,
    )

    np.testing.assert_allclose(annotated_keypoints[0, 1, 9], np.array([321.0, 654.0], dtype=float))
    assert annotated_scores[0, 1, 9] == 0.8
    np.testing.assert_allclose(annotated_keypoints[0, 0, 9], np.array([0.0, 0.0], dtype=float))
