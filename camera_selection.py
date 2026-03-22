#!/usr/bin/env python3
"""Helpers to select and subset camera data consistently."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vitpose_ekf_pipeline import PoseData


def parse_camera_names(raw: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """Parse camera names from CLI/GUI input while preserving their first-seen order."""

    if raw is None:
        return []
    if isinstance(raw, str):
        tokens = [token.strip() for token in raw.replace(";", ",").split(",")]
    else:
        tokens = [str(token).strip() for token in raw]
    seen: OrderedDict[str, None] = OrderedDict()
    for token in tokens:
        if token:
            seen.setdefault(token, None)
    return list(seen.keys())


def format_camera_names(camera_names: list[str] | tuple[str, ...]) -> str:
    """Format camera names for compact display in the GUI and summaries."""

    return ", ".join(str(name) for name in camera_names)


def select_camera_names(available_camera_names: list[str], requested_camera_names: list[str] | tuple[str, ...] | None) -> list[str]:
    """Validate and resolve a requested camera subset against the available names."""

    requested = parse_camera_names(list(requested_camera_names) if requested_camera_names is not None else None)
    if not requested:
        return [str(name) for name in available_camera_names]
    available = {str(name): idx for idx, name in enumerate(available_camera_names)}
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"Unknown camera(s): {', '.join(missing)}")
    return requested


def subset_calibrations(
    calibrations: dict[str, object],
    requested_camera_names: list[str] | tuple[str, ...] | None,
) -> dict[str, object]:
    """Return a calibration mapping restricted to the selected camera subset."""

    selected = select_camera_names(list(calibrations.keys()), requested_camera_names)
    return {camera_name: calibrations[camera_name] for camera_name in selected}


def subset_pose_data(
    pose_data: "PoseData",
    requested_camera_names: list[str] | tuple[str, ...] | None,
) -> "PoseData":
    """Return a shallow PoseData copy containing only the requested cameras."""

    from vitpose_ekf_pipeline import PoseData

    selected = select_camera_names(list(pose_data.camera_names), requested_camera_names)
    index_map = {str(name): idx for idx, name in enumerate(pose_data.camera_names)}
    camera_indices = [index_map[name] for name in selected]
    return PoseData(
        camera_names=selected,
        frames=np.asarray(pose_data.frames, dtype=int),
        keypoints=np.asarray(pose_data.keypoints[camera_indices], dtype=float),
        scores=np.asarray(pose_data.scores[camera_indices], dtype=float),
        raw_keypoints=(
            None
            if pose_data.raw_keypoints is None
            else np.asarray(pose_data.raw_keypoints[camera_indices], dtype=float)
        ),
        filtered_keypoints=(
            None
            if pose_data.filtered_keypoints is None
            else np.asarray(pose_data.filtered_keypoints[camera_indices], dtype=float)
        ),
    )
