from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class SkeletonLayer2D:
    points: np.ndarray
    color: str
    label: str | None = None
    marker_size: float = 12.0
    marker_fill: bool = True
    marker_edge_width: float = 1.4
    line_alpha: float = 0.85
    line_style: str = "-"
    line_width_scale: float = 1.0


@dataclass(frozen=True)
class PointValueOverlay2D:
    label: str
    points: np.ndarray
    values: np.ndarray | None = None
    mask: np.ndarray | None = None
    cmap: str | None = "turbo"
    size: float = 18.0
    edgecolors: str = "white"
    linewidths: float = 0.8
    alpha: float = 0.95
    excluded_color: str = "#111111"
    excluded_marker: str = "x"
    excluded_size: float = 34.0
    excluded_linewidths: float = 1.6


def draw_point_value_overlay(ax, overlay: PointValueOverlay2D):
    """Draw an optional per-point overlay and return the color-mapped scatter when present."""

    points = np.asarray(overlay.points, dtype=float)
    finite_points_mask = np.all(np.isfinite(points), axis=1)
    scatter = None
    if overlay.values is not None:
        values = np.asarray(overlay.values, dtype=float).reshape(-1)
        finite_overlay = finite_points_mask & np.isfinite(values)
        if np.any(finite_overlay):
            scatter = ax.scatter(
                points[finite_overlay, 0],
                points[finite_overlay, 1],
                c=np.asarray(values[finite_overlay], dtype=float),
                cmap=overlay.cmap or "turbo",
                s=float(overlay.size),
                linewidths=float(overlay.linewidths),
                edgecolors=overlay.edgecolors,
                alpha=float(overlay.alpha),
                zorder=6,
            )
    if overlay.mask is not None:
        mask = finite_points_mask & np.asarray(overlay.mask, dtype=bool).reshape(-1)
        if np.any(mask):
            ax.scatter(
                points[mask, 0],
                points[mask, 1],
                marker=overlay.excluded_marker,
                s=float(overlay.excluded_size),
                linewidths=float(overlay.excluded_linewidths),
                c=overlay.excluded_color,
                alpha=float(overlay.alpha),
                zorder=7,
                label=overlay.label if overlay.values is None else None,
            )
    return scatter


def render_camera_frame_2d(
    ax,
    *,
    width: float,
    height: float,
    title: str,
    layers: list[SkeletonLayer2D],
    draw_skeleton_fn: Callable[..., None],
    background_image: np.ndarray | None = None,
    draw_background_fn: Callable[..., None] | None = None,
    crop_mode: str = "full",
    crop_limits: dict[str, np.ndarray] | None = None,
    cam_name: str | None = None,
    frame_idx: int = 0,
    apply_axis_limits_fn: Callable[..., None] | None = None,
    hide_axes: bool = True,
    hide_axes_fn: Callable[..., None] | None = None,
    x_limits: tuple[float, float] | None = None,
    y_limits: tuple[float, float] | None = None,
    show_grid: bool = False,
    grid_alpha: float = 0.18,
    xlabel: str = "",
    ylabel: str = "",
) -> bool:
    """Render one 2D camera frame with shared background/limits/layer logic."""

    has_image_background = False
    if background_image is not None and draw_background_fn is not None:
        draw_background_fn(ax, background_image, width=width, height=height)
        has_image_background = True

    if x_limits is not None and y_limits is not None:
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        if hasattr(ax, "set_autoscale_on"):
            ax.set_autoscale_on(False)
    elif apply_axis_limits_fn is not None and crop_limits is not None and cam_name is not None:
        apply_axis_limits_fn(
            ax,
            crop_mode=crop_mode,
            crop_limits=crop_limits,
            cam_name=cam_name,
            frame_idx=frame_idx,
            width=width,
            height=height,
        )
    else:
        ax.set_xlim(0.0, float(width))
        ax.set_ylim(float(height), 0.0)
        if hasattr(ax, "set_autoscale_on"):
            ax.set_autoscale_on(False)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)

    for layer in layers:
        draw_skeleton_fn(
            ax,
            layer.points,
            layer.color,
            layer.label,
            marker_size=layer.marker_size,
            marker_fill=layer.marker_fill,
            marker_edge_width=layer.marker_edge_width,
            line_alpha=layer.line_alpha,
            line_style=layer.line_style,
            line_width_scale=layer.line_width_scale,
        )

    if hide_axes and hide_axes_fn is not None:
        hide_axes_fn(ax)
    else:
        if show_grid:
            ax.grid(alpha=float(grid_alpha))
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

    return has_image_background
