from __future__ import annotations

import math

import numpy as np


def camera_layout(n_cameras: int) -> tuple[int, int]:
    n_cameras = max(int(n_cameras), 1)
    ncols = min(4, max(1, int(math.ceil(math.sqrt(n_cameras)))))
    nrows = int(math.ceil(n_cameras / ncols))
    return nrows, ncols


def square_crop_bounds(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    width: float,
    height: float,
    margin: float,
) -> np.ndarray:
    """Return a square crop that preserves the image isoview around visible points."""

    span_x = max(10.0, float(xmax - xmin))
    span_y = max(10.0, float(ymax - ymin))
    half_size = 0.5 * max(span_x, span_y) * (1.0 + float(margin))
    half_size = max(12.0, half_size)
    center_x = 0.5 * (float(xmin) + float(xmax))
    center_y = 0.5 * (float(ymin) + float(ymax))
    x0 = center_x - half_size
    x1 = center_x + half_size
    y0 = center_y - half_size
    y1 = center_y + half_size
    if x0 < 0.0:
        x1 = min(float(width), x1 - x0)
        x0 = 0.0
    if x1 > float(width):
        x0 = max(0.0, x0 - (x1 - float(width)))
        x1 = float(width)
    if y0 < 0.0:
        y1 = min(float(height), y1 - y0)
        y0 = 0.0
    if y1 > float(height):
        y0 = max(0.0, y0 - (y1 - float(height)))
        y1 = float(height)
    final_size = min(float(x1 - x0), float(y1 - y0))
    if final_size <= 0.0:
        return np.array([0.0, float(width), float(height), 0.0], dtype=float)
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    half_final = 0.5 * final_size
    x0 = max(0.0, center_x - half_final)
    x1 = min(float(width), center_x + half_final)
    y0 = max(0.0, center_y - half_final)
    y1 = min(float(height), center_y + half_final)
    return np.array([x0, x1, y1, y0], dtype=float)


def compute_pose_crop_limits_2d(
    raw_2d: np.ndarray,
    calibrations: dict,
    camera_names: list[str],
    margin: float,
) -> dict[str, np.ndarray]:
    """Compute per-frame crop bounds from valid 2D points for each camera."""

    limits: dict[str, np.ndarray] = {}
    for cam_idx, cam_name in enumerate(camera_names):
        width, height = calibrations[cam_name].image_size
        n_frames = raw_2d.shape[1]
        camera_limits = np.full((n_frames, 4), np.nan, dtype=float)
        for frame_idx in range(n_frames):
            points = raw_2d[cam_idx, frame_idx]
            valid = np.all(np.isfinite(points), axis=1)
            if not np.any(valid):
                continue
            xy = points[valid]
            xmin, ymin = np.min(xy, axis=0)
            xmax, ymax = np.max(xy, axis=0)
            camera_limits[frame_idx] = square_crop_bounds(
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                width=width,
                height=height,
                margin=margin,
            )
        valid_frames = np.flatnonzero(np.all(np.isfinite(camera_limits), axis=1))
        if valid_frames.size == 0:
            camera_limits[:] = np.array([0.0, float(width), float(height), 0.0], dtype=float)
        else:
            first_valid = int(valid_frames[0])
            last_valid = int(valid_frames[-1])
            for frame_idx in range(0, first_valid):
                camera_limits[frame_idx] = camera_limits[first_valid]
            for frame_idx in range(first_valid + 1, n_frames):
                if not np.all(np.isfinite(camera_limits[frame_idx])):
                    camera_limits[frame_idx] = camera_limits[frame_idx - 1]
            for frame_idx in range(last_valid + 1, n_frames):
                camera_limits[frame_idx] = camera_limits[last_valid]
        limits[cam_name] = camera_limits
    return limits


def apply_2d_axis_limits(
    ax,
    *,
    crop_mode: str,
    crop_limits: dict[str, np.ndarray],
    cam_name: str,
    frame_idx: int,
    width: float,
    height: float,
) -> None:
    """Apply fixed 2D limits and disable matplotlib autoscale."""

    if crop_mode == "pose":
        x0, x1, y1, y0 = crop_limits[cam_name][frame_idx]
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
    else:
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
    ax.set_autoscale_on(False)


def draw_2d_background_image(ax, image: np.ndarray, width: float, height: float) -> None:
    """Display a 2D image in the same pixel coordinate system as the keypoints."""

    ax.imshow(
        image,
        extent=(0.0, float(width), float(height), 0.0),
        origin="upper",
        interpolation="none",
        zorder=0,
    )


def hide_2d_axes(ax) -> None:
    """Hide pixel axes while preserving the title and data coordinates."""

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)


def adjust_image_levels(
    image: np.ndarray,
    *,
    brightness: float = 1.0,
    contrast: float = 1.0,
) -> np.ndarray:
    """Apply one simple global brightness/contrast adjustment to an RGB(A) image."""

    adjusted = np.asarray(image, dtype=float)
    if adjusted.size == 0:
        return adjusted
    scale = 255.0 if np.nanmax(adjusted) > 1.5 else 1.0
    adjusted = adjusted / scale
    rgb = adjusted[..., :3]
    rgb = (rgb - 0.5) * float(contrast) + 0.5
    rgb = rgb * float(brightness)
    adjusted[..., :3] = np.clip(rgb, 0.0, 1.0)
    if scale > 1.5:
        adjusted = np.round(adjusted * scale).astype(image.dtype, copy=False)
    return adjusted
