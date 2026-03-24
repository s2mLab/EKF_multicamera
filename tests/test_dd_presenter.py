import numpy as np

from judging.dd_analysis import DDJumpAnalysis, DDSessionAnalysis, JumpSegment
from judging.dd_presenter import (
    build_jump_plot_data,
    compare_dd_code_characters,
    compare_dd_to_reference,
    dd_reference_status_color,
    dd_reference_status_text,
    format_dd_summary,
    jump_list_label,
    jump_list_label_with_reference,
    split_dd_code,
)


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


def test_jump_list_label_with_reference_appends_reference_code():
    label = jump_list_label_with_reference(3, make_jump(), "821o")
    assert label == "S3 | som 2.25 | tw 1.50 | ref 821o"


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
        expected_codes_by_jump={1: "821o"},
    )
    assert "Reconstruction: Pose2Sim" in summary
    assert "body shape=- | code=82+33" in summary
    assert "reference code=821o | diff" in summary
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


def test_compare_dd_to_reference_reports_exact_match():
    jump = make_jump()
    jump.code = "821o"
    analysis = DDSessionAnalysis(
        root_q=np.zeros((40, 6)),
        height=np.linspace(1.0, 2.0, 40),
        smoothed_height=np.linspace(1.0, 2.0, 40),
        height_threshold=1.4,
        airborne_regions=[(12, 25)],
        jump_segments=[jump.segment],
        jumps=[jump],
    )
    comparison = compare_dd_to_reference(analysis, {1: "821o"})
    assert comparison.status == "exact"
    assert dd_reference_status_text(comparison) == "1/1"
    assert dd_reference_status_color(comparison) == "ok"


def test_compare_dd_to_reference_reports_partial_match():
    jump_a = make_jump()
    jump_a.code = "821o"
    jump_b = make_jump()
    jump_b.code = "42/"
    jump_b.segment = JumpSegment(start=40, end=60, peak_index=50)
    analysis = DDSessionAnalysis(
        root_q=np.zeros((80, 6)),
        height=np.linspace(1.0, 2.0, 80),
        smoothed_height=np.linspace(1.0, 2.0, 80),
        height_threshold=1.4,
        airborne_regions=[(12, 25), (42, 55)],
        jump_segments=[jump_a.segment, jump_b.segment],
        jumps=[jump_a, jump_b],
    )
    comparison = compare_dd_to_reference(analysis, {1: "821o", 2: "43/"})
    assert comparison.status == "partial"
    assert dd_reference_status_text(comparison) == "1/2"
    assert dd_reference_status_color(comparison) == "partial"


def test_split_dd_code_separates_somersault_twist_and_body_shape():
    assert split_dd_code("821o") == ("82", "1", "o")
    assert split_dd_code("42/") == ("42", "", "/")
    assert split_dd_code("00") == ("00", "", "")


def test_compare_dd_code_characters_marks_only_the_wrong_role_characters():
    comparison = compare_dd_code_characters("821o", "812/")

    assert [(item.role, item.expected_char, item.detected_char, item.matches) for item in comparison] == [
        ("somersault", "8", "8", True),
        ("somersault", "2", "1", False),
        ("twist", "1", "2", False),
        ("body", "o", "/", False),
    ]
