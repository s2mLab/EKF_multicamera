import numpy as np

from dd_analysis import DDJumpAnalysis, DDSessionAnalysis, JumpSegment
from dd_presenter import build_jump_plot_data, format_dd_summary, jump_list_label


def make_jump() -> DDJumpAnalysis:
    return DDJumpAnalysis(
        segment=JumpSegment(start=10, end=30, peak_index=20),
        somersault_turns=2.25,
        twist_turns=1.5,
        max_tilt_rad=np.deg2rad(25.0),
        mean_tilt_rad=np.deg2rad(10.0),
        classification="double forward, twist 1.5",
        body_shape=None,
        code="82+33",
        twists_per_salto=[0.5, 1.0],
        full_salto_event_indices=[5, 12],
        quarter_salto_event_indices=[2, 5, 8],
        half_twist_event_indices=[4, 10],
        somersault_curve_turns=np.linspace(0.0, 2.25, 16),
        twist_curve_turns=np.linspace(0.0, 1.5, 16),
        tilt_curve_rad=np.linspace(0.0, np.deg2rad(25.0), 16),
        angle_mode="euler",
    )


def test_jump_list_label_formats_expected_text():
    label = jump_list_label(3, make_jump())
    assert label == "S3 | som 2.25 | tw 1.50"


def test_format_dd_summary_handles_missing_body_shape_and_code():
    jump = make_jump()
    analysis = DDSessionAnalysis(
        root_q=np.zeros((40, 6)),
        height=np.linspace(1.0, 2.0, 40),
        smoothed_height=np.linspace(1.0, 2.0, 40),
        height_threshold=1.4,
        airborne_regions=[(12, 25)],
        jump_segments=[jump.segment],
        jumps=[jump],
    )
    summary = format_dd_summary(
        analysis,
        reconstruction_label_text="Pose2Sim",
        height_dof="TRUNK:TransZ",
        angle_mode="euler",
        fps=120.0,
    )
    assert "Reconstruction: Pose2Sim" in summary
    assert "body shape=- | code=82+33" in summary
    assert "twists by salto=[S1: 0.5, S2: 1.0]" in summary


def test_build_jump_plot_data_converts_indices_to_times_and_values():
    jump = make_jump()
    plot_data = build_jump_plot_data(jump, fps=100.0)
    np.testing.assert_allclose(plot_data.full_salto_times, np.array([0.05, 0.12]), atol=1e-12)
    np.testing.assert_allclose(plot_data.quarter_salto_times, np.array([0.02, 0.05, 0.08]), atol=1e-12)
    np.testing.assert_allclose(
        plot_data.quarter_salto_values,
        jump.somersault_curve_turns[[2, 5, 8]],
        atol=1e-12,
    )
    np.testing.assert_allclose(plot_data.half_twist_times, np.array([0.04, 0.10]), atol=1e-12)
    np.testing.assert_allclose(
        plot_data.half_twist_values,
        jump.twist_curve_turns[[4, 10]],
        atol=1e-12,
    )
