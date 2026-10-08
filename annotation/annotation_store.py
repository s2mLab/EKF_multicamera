#!/usr/bin/env python3
"""Sparse multi-view 2D annotation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ANNOTATION_SCHEMA_VERSION = 1


def default_annotation_path(keypoints_path: Path) -> Path:
    """Return the default sparse-annotation JSON path for one keypoints file."""

    keypoints_path = Path(keypoints_path)
    dataset_stem = keypoints_path.stem
    if dataset_stem.endswith("_keypoints"):
        dataset_stem = dataset_stem[: -len("_keypoints")]
    return keypoints_path.parent.parent / "annotations" / f"{dataset_stem}_annotations.json"


def empty_annotation_payload(keypoints_path: Path | None = None) -> dict[str, object]:
    """Create one empty sparse annotation payload."""

    payload: dict[str, object] = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "annotations": {},
    }
    if keypoints_path is not None:
        payload["keypoints_source"] = str(Path(keypoints_path))
    return payload


def load_annotation_payload(path: Path | str | None, *, keypoints_path: Path | None = None) -> dict[str, object]:
    """Load one sparse annotation payload, or return an empty one when missing."""

    if path is None:
        return empty_annotation_payload(keypoints_path)
    annotation_path = Path(path)
    if not annotation_path.exists():
        return empty_annotation_payload(keypoints_path)
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid annotation payload in {annotation_path}")
    payload.setdefault("schema_version", ANNOTATION_SCHEMA_VERSION)
    payload.setdefault("annotations", {})
    if keypoints_path is not None:
        payload.setdefault("keypoints_source", str(Path(keypoints_path)))
    return payload


def save_annotation_payload(path: Path | str, payload: dict[str, object]) -> Path:
    """Persist one sparse annotation payload."""

    annotation_path = Path(path)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(payload)
    serializable["schema_version"] = ANNOTATION_SCHEMA_VERSION
    serializable.setdefault("annotations", {})
    annotation_path.write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")
    return annotation_path


def _annotation_leaf(
    payload: dict[str, object],
    *,
    camera_name: str,
    frame_number: int,
    create: bool,
) -> dict[str, object] | None:
    annotations = payload.setdefault("annotations", {})
    if not isinstance(annotations, dict):
        raise ValueError("Annotation payload must contain a dict under 'annotations'.")
    camera_annotations = annotations.get(str(camera_name))
    if camera_annotations is None:
        if not create:
            return None
        camera_annotations = {}
        annotations[str(camera_name)] = camera_annotations
    if not isinstance(camera_annotations, dict):
        raise ValueError(f"Annotation payload for camera {camera_name} must be a dict.")
    frame_key = str(int(frame_number))
    frame_annotations = camera_annotations.get(frame_key)
    if frame_annotations is None:
        if not create:
            return None
        frame_annotations = {}
        camera_annotations[frame_key] = frame_annotations
    if not isinstance(frame_annotations, dict):
        raise ValueError(f"Annotation payload for frame {frame_key} must be a dict.")
    return frame_annotations


def get_annotation_point(
    payload: dict[str, object],
    *,
    camera_name: str,
    frame_number: int,
    keypoint_name: str,
) -> tuple[np.ndarray, float] | tuple[None, None]:
    """Return one sparse annotation point as ``(xy, score)``."""

    frame_annotations = _annotation_leaf(payload, camera_name=camera_name, frame_number=frame_number, create=False)
    if not frame_annotations:
        return None, None
    value = frame_annotations.get(str(keypoint_name))
    if value is None:
        return None, None
    xy = np.asarray(value.get("xy", [np.nan, np.nan]), dtype=float)
    if xy.shape != (2,) or not np.all(np.isfinite(xy)):
        return None, None
    score = float(value.get("score", 1.0))
    return xy, score


def set_annotation_point(
    payload: dict[str, object],
    *,
    camera_name: str,
    frame_number: int,
    keypoint_name: str,
    xy: np.ndarray | tuple[float, float] | list[float],
    score: float = 1.0,
) -> None:
    """Write or replace one sparse 2D annotation."""

    point = np.asarray(xy, dtype=float).reshape(2)
    frame_annotations = _annotation_leaf(payload, camera_name=camera_name, frame_number=frame_number, create=True)
    frame_annotations[str(keypoint_name)] = {"xy": [float(point[0]), float(point[1])], "score": float(score)}


def clear_annotation_point(
    payload: dict[str, object],
    *,
    camera_name: str,
    frame_number: int,
    keypoint_name: str,
) -> None:
    """Remove one sparse 2D annotation if it exists."""

    annotations = payload.get("annotations")
    if not isinstance(annotations, dict):
        return
    camera_annotations = annotations.get(str(camera_name))
    if not isinstance(camera_annotations, dict):
        return
    frame_key = str(int(frame_number))
    frame_annotations = camera_annotations.get(frame_key)
    if not isinstance(frame_annotations, dict):
        return
    frame_annotations.pop(str(keypoint_name), None)
    if not frame_annotations:
        camera_annotations.pop(frame_key, None)
    if not camera_annotations:
        annotations.pop(str(camera_name), None)


def apply_annotations_to_pose_arrays(
    *,
    keypoints: np.ndarray,
    scores: np.ndarray,
    camera_names: list[str],
    frames: np.ndarray,
    keypoint_names: list[str],
    payload: dict[str, object] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Overlay sparse annotations onto dense 2D arrays."""

    if payload is None:
        return np.asarray(keypoints, dtype=float), np.asarray(scores, dtype=float)
    annotated_keypoints = np.asarray(keypoints, dtype=float).copy()
    annotated_scores = np.asarray(scores, dtype=float).copy()
    frame_to_index = {int(frame): idx for idx, frame in enumerate(np.asarray(frames, dtype=int))}
    keypoint_to_index = {str(name): idx for idx, name in enumerate(keypoint_names)}
    annotations = payload.get("annotations", {})
    if not isinstance(annotations, dict):
        return annotated_keypoints, annotated_scores
    for cam_idx, camera_name in enumerate(camera_names):
        camera_annotations = annotations.get(str(camera_name))
        if not isinstance(camera_annotations, dict):
            continue
        for frame_key, marker_annotations in camera_annotations.items():
            if not isinstance(marker_annotations, dict):
                continue
            frame_idx = frame_to_index.get(int(frame_key))
            if frame_idx is None:
                continue
            for keypoint_name, value in marker_annotations.items():
                kp_idx = keypoint_to_index.get(str(keypoint_name))
                if kp_idx is None or not isinstance(value, dict):
                    continue
                xy = np.asarray(value.get("xy", [np.nan, np.nan]), dtype=float)
                if xy.shape != (2,) or not np.all(np.isfinite(xy)):
                    continue
                annotated_keypoints[cam_idx, frame_idx, kp_idx] = xy
                annotated_scores[cam_idx, frame_idx, kp_idx] = float(value.get("score", 1.0))
    return annotated_keypoints, annotated_scores
