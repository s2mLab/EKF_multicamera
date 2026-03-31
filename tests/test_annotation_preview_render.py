import numpy as np
from matplotlib.figure import Figure

from annotation.preview_render import annotation_frame_label_text, render_annotation_camera_view


def test_annotation_frame_label_text_formats_filtered_mode():
    text = annotation_frame_label_text(
        frame_idx=7,
        frame_number=42,
        mode="worst_reproj",
        filtered_indices=[1, 7, 9],
        mode_labels={"all": "All frames", "worst_reproj": "Worst reproj 5%"},
    )

    assert text == "frame 7 | raw 42 | Worst reproj 5% 2/3"


def test_render_annotation_camera_view_collects_hover_entries_and_draws_overlays():
    figure = Figure(figsize=(4, 4))
    ax = figure.subplots()
    calls = {"background": 0, "limits": 0, "hide": 0, "skeleton": 0, "upper_back": 0}
    annotated_points = {
        ("cam0", 10, "nose"): np.array([10.0, 20.0], dtype=float),
        ("cam0", 10, "left_wrist"): np.array([30.0, 40.0], dtype=float),
        ("cam0", 9, "nose"): np.array([8.0, 18.0], dtype=float),
        ("cam0", 8, "nose"): np.array([6.0, 16.0], dtype=float),
    }

    hover_entries = render_annotation_camera_view(
        ax,
        ax_idx=0,
        camera_name="cam0",
        frame_idx=0,
        frame_number=10,
        width=640,
        height=480,
        crop_mode="full",
        crop_limits={},
        background_image=np.zeros((10, 10, 3), dtype=float),
        current_marker="nose",
        current_color="red",
        keypoint_names=("nose", "left_wrist"),
        kp_index={"nose": 0, "left_wrist": 1},
        annotation_xy_getter=lambda camera_name, frame_number, keypoint_name: annotated_points.get(
            (camera_name, frame_number, keypoint_name)
        ),
        pending_reprojection_points={("cam0", 10, "left_wrist"): np.array([50.0, 60.0], dtype=float)},
        marker_color_getter=lambda keypoint_name: {"nose": "red", "left_wrist": "blue"}[keypoint_name],
        marker_shape_getter=lambda keypoint_name: "+" if keypoint_name == "nose" else "x",
        draw_background_fn=lambda *_args, **_kwargs: calls.__setitem__("background", calls["background"] + 1),
        apply_axis_limits_fn=lambda *_args, **_kwargs: calls.__setitem__("limits", calls["limits"] + 1),
        hide_axes_fn=lambda *_args, **_kwargs: calls.__setitem__("hide", calls["hide"] + 1),
        draw_skeleton_fn=lambda *_args, **_kwargs: calls.__setitem__("skeleton", calls["skeleton"] + 1),
        draw_upper_back_fn=lambda *_args, **_kwargs: calls.__setitem__("upper_back", calls["upper_back"] + 1),
        kinematic_projected_points=np.zeros((1, 1, 2, 2), dtype=float),
        kinematic_segmented_back_projected={
            "hip_triangle": np.zeros((1, 1, 4, 2), dtype=float),
            "shoulder_triangle": np.zeros((1, 1, 4, 2), dtype=float),
            "mid_back": np.zeros((1, 1, 1, 2), dtype=float),
        },
        motion_prior_enabled=True,
        motion_prior_diameter=15.0,
        motion_prior_center_fn=lambda pt1, pt2: np.array([12.0, 22.0], dtype=float),
    )

    assert calls == {"background": 1, "limits": 1, "hide": 1, "skeleton": 1, "upper_back": 1}
    assert [entry["source"] for entry in hover_entries] == [
        "annotated",
        "annotated",
        "model reproj",
        "model reproj",
        "pending reproj",
    ]
