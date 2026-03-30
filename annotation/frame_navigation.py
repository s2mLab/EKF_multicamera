from __future__ import annotations

from pathlib import Path

import numpy as np

from judging.execution import available_execution_image_frames


def resolve_annotation_frame_filter_mode(
    selected_label: str | None,
    options: dict[str, str],
) -> str:
    """Resolve one UI label/value to a stable frame filter mode."""

    selected_label = str(selected_label or "").strip()
    for mode, label in options.items():
        if selected_label == label or selected_label == mode:
            return str(mode)
    return "all"


def fallback_annotation_filtered_indices(
    n_frames: int,
    filtered_indices: list[int] | tuple[int, ...] | np.ndarray | None,
) -> list[int]:
    """Return a safe filtered subset, falling back to all frames when empty."""

    fallback = list(range(max(0, int(n_frames))))
    if filtered_indices is None:
        return fallback
    filtered = [int(idx) for idx in filtered_indices]
    return filtered if filtered else fallback


def frame_has_available_image(
    frame_number: int,
    camera_names: list[str],
    available_by_camera: dict[str, set[int]] | None,
) -> bool:
    """Return whether one frame is available in at least one selected camera."""

    if not camera_names or not available_by_camera:
        return False
    frame_number = int(frame_number)
    return any(frame_number in available_by_camera.get(str(camera_name), set()) for camera_name in camera_names)


def navigable_annotation_frame_local_indices(
    frames: np.ndarray,
    filtered_indices: list[int] | tuple[int, ...] | np.ndarray | None,
    camera_names: list[str],
    images_root: Path | None,
) -> list[int]:
    """Return the filtered subset further restricted to frames with images when possible."""

    frames = np.asarray(frames, dtype=int)
    filtered = fallback_annotation_filtered_indices(len(frames), filtered_indices)
    if images_root is None or not camera_names:
        return filtered
    available_by_camera = available_execution_image_frames(images_root, camera_names)
    with_images = [
        int(local_idx)
        for local_idx in filtered
        if 0 <= int(local_idx) < len(frames)
        and frame_has_available_image(int(frames[int(local_idx)]), camera_names, available_by_camera)
    ]
    return with_images or filtered


def step_frame_index_within_subset(
    current_index: int,
    delta: int,
    candidate_indices: list[int] | tuple[int, ...] | np.ndarray | None,
) -> int | None:
    """Move cyclically within one sparse subset of local frame indices."""

    if not candidate_indices:
        return None
    candidates = sorted(int(idx) for idx in candidate_indices)
    current_index = int(current_index)
    if current_index in candidates:
        current_pos = candidates.index(current_index)
        next_pos = (current_pos + (-1 if int(delta) < 0 else 1)) % len(candidates)
        return int(candidates[next_pos])
    if int(delta) < 0:
        previous = [idx for idx in candidates if idx < current_index]
        return int(previous[-1] if previous else candidates[-1])
    following = [idx for idx in candidates if idx > current_index]
    return int(following[0] if following else candidates[0])


def clamp_index_to_subset(
    current_index: int, candidate_indices: list[int] | tuple[int, ...] | np.ndarray | None
) -> int | None:
    """Return the current index if valid, else the first candidate."""

    if not candidate_indices:
        return None
    candidates = [int(idx) for idx in candidate_indices]
    current_index = int(current_index)
    return current_index if current_index in candidates else int(candidates[0])
