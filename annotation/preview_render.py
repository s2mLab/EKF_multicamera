"""Rendering helpers for the Matplotlib-based Annotation preview."""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np


def annotation_frame_label_text(
    *,
    frame_idx: int,
    frame_number: int,
    mode: str,
    filtered_indices: list[int],
    mode_labels: dict[str, str],
) -> str:
    """Build the compact annotation frame label for the current filter mode."""

    if str(mode) == "all":
        return f"frame {int(frame_idx)} | raw {int(frame_number)}"
    filtered_position = filtered_indices.index(int(frame_idx)) + 1 if int(frame_idx) in filtered_indices else 0
    return (
        f"frame {int(frame_idx)} | raw {int(frame_number)} | "
        f"{mode_labels[str(mode)]} {filtered_position}/{len(filtered_indices)}"
    )


def render_annotation_camera_view(
    ax,
    *,
    ax_idx: int,
    camera_name: str,
    frame_idx: int,
    frame_number: int,
    width: float,
    height: float,
    crop_mode: str,
    crop_limits: dict[str, np.ndarray],
    background_image: np.ndarray | None,
    current_marker: str,
    current_color,
    keypoint_names: list[str] | tuple[str, ...],
    kp_index: dict[str, int],
    annotation_xy_getter: Callable[[str, int, str], np.ndarray | None],
    pending_reprojection_points: dict[tuple[str, int, str], np.ndarray],
    marker_color_getter: Callable[[str], object],
    marker_shape_getter: Callable[[str], str],
    draw_background_fn: Callable[..., None],
    apply_axis_limits_fn: Callable[..., None],
    hide_axes_fn: Callable[[object], None],
    draw_skeleton_fn: Callable[..., None],
    draw_upper_back_fn: Callable[..., None],
    kinematic_projected_points: np.ndarray | None,
    kinematic_segmented_back_projected: dict[str, np.ndarray] | None,
    reference_projected_points: np.ndarray | None,
    reference_projected_label: str | None,
    reference_projected_color,
    motion_prior_enabled: bool,
    motion_prior_diameter: float,
    motion_prior_center_fn: Callable[[np.ndarray | None, np.ndarray | None], np.ndarray | None],
) -> list[dict[str, object]]:
    """Render one camera subplot for the annotation preview and return hover entries."""

    hover_entries: list[dict[str, object]] = []
    if background_image is not None:
        draw_background_fn(ax, background_image, width=width, height=height)
    apply_axis_limits_fn(
        ax,
        crop_mode=crop_mode,
        crop_limits=crop_limits,
        cam_name=camera_name,
        frame_idx=frame_idx,
        width=width,
        height=height,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(camera_name)
    ax.grid(False)
    hide_axes_fn(ax)

    for keypoint_name in keypoint_names:
        annotated_xy = annotation_xy_getter(camera_name, frame_number, keypoint_name)
        if annotated_xy is None:
            continue
        marker_color = marker_color_getter(keypoint_name)
        display_color = "black" if keypoint_name == current_marker else marker_color
        ax.scatter(
            [annotated_xy[0]],
            [annotated_xy[1]],
            s=(72 if keypoint_name == current_marker else 34),
            c=[display_color],
            marker=marker_shape_getter(keypoint_name),
            linewidths=(1.7 if keypoint_name == current_marker else 1.0),
            zorder=5,
        )
        hover_entries.append(
            {
                "xy": np.asarray(annotated_xy, dtype=float),
                "keypoint_name": str(keypoint_name),
                "source": "annotated",
            }
        )

    if kinematic_projected_points is not None and ax_idx < kinematic_projected_points.shape[0]:
        projected_points = np.asarray(kinematic_projected_points[ax_idx, 0], dtype=float)
        draw_skeleton_fn(
            ax,
            projected_points,
            "#00b8d9",
            "Model reproj",
            marker_size=3.2,
            marker_fill=False,
            marker_edge_width=0.8,
            line_alpha=0.32,
            line_style="--",
            line_width_scale=0.62,
        )
        for keypoint_name in keypoint_names:
            projected_xy = np.asarray(projected_points[kp_index[keypoint_name]], dtype=float)
            if not np.all(np.isfinite(projected_xy)):
                continue
            hover_entries.append(
                {
                    "xy": projected_xy,
                    "keypoint_name": str(keypoint_name),
                    "source": "model reproj",
                }
            )
        segmented = kinematic_segmented_back_projected or {}
        draw_upper_back_fn(
            ax,
            hip_triangle_2d=(
                segmented.get("hip_triangle", np.empty((0, 0, 0)))[ax_idx, 0] if "hip_triangle" in segmented else None
            ),
            shoulder_triangle_2d=(
                segmented.get("shoulder_triangle", np.empty((0, 0, 0)))[ax_idx, 0]
                if "shoulder_triangle" in segmented
                else None
            ),
            mid_back_2d=(
                segmented.get("mid_back", np.empty((0, 0, 0)))[ax_idx, 0, 0] if "mid_back" in segmented else None
            ),
            color="#00b8d9",
            line_width=1.0,
            line_alpha=0.32,
            line_style="--",
            marker_size=18.0,
            marker_line_width=0.9,
            marker_alpha=0.38,
        )

    if reference_projected_points is not None:
        draw_skeleton_fn(
            ax,
            np.asarray(reference_projected_points, dtype=float),
            reference_projected_color,
            reference_projected_label,
            marker_size=2.8,
            marker_fill=False,
            marker_edge_width=0.7,
            line_alpha=0.26,
            line_style="--",
            line_width_scale=0.5,
        )
        for keypoint_name in keypoint_names:
            projected_xy = np.asarray(reference_projected_points[kp_index[keypoint_name]], dtype=float)
            if not np.all(np.isfinite(projected_xy)):
                continue
            hover_entries.append(
                {
                    "xy": projected_xy,
                    "keypoint_name": str(keypoint_name),
                    "source": "reconstruction reproj",
                }
            )

    for (pending_camera, pending_frame, keypoint_name), pending_xy in pending_reprojection_points.items():
        if pending_camera != str(camera_name) or int(pending_frame) != int(frame_number):
            continue
        ax.scatter(
            [pending_xy[0]],
            [pending_xy[1]],
            s=72,
            facecolors="none",
            edgecolors=[marker_color_getter(keypoint_name)],
            marker="o",
            linewidths=1.7,
            alpha=0.95,
            zorder=6,
        )
        hover_entries.append(
            {
                "xy": np.asarray(pending_xy, dtype=float),
                "keypoint_name": str(keypoint_name),
                "source": "pending reproj",
            }
        )

    if motion_prior_enabled:
        previous_xy = annotation_xy_getter(camera_name, int(frame_number - 1), current_marker)
        two_back_xy = annotation_xy_getter(camera_name, int(frame_number - 2), current_marker)
        prior_center = motion_prior_center_fn(previous_xy, two_back_xy)
        if prior_center is not None:
            circle = plt.Circle(
                tuple(prior_center),
                radius=0.5 * float(motion_prior_diameter),
                edgecolor=current_color,
                facecolor="none",
                linestyle="--",
                linewidth=1.3,
                alpha=0.8,
            )
            ax.add_patch(circle)

    return hover_entries
