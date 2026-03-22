from reconstruction.reconstruction_timings import (
    build_pipeline_diagram,
    compute_time_seconds,
    format_reconstruction_timing_details,
    format_seconds_brief,
    humanize_stage_name,
    make_timing_stage,
    objective_total_seconds,
    parse_stage_timings,
)


def test_parse_stage_timings_keeps_finite_numeric_values_in_order():
    summary = {
        "stage_timings_s": {
            "triangulation_s": 1.25,
            "bad": "x",
            "model_creation_s": 0.5,
            "missing": None,
            "total_s": 1.75,
        }
    }
    assert parse_stage_timings(summary) == [
        ("triangulation_s", 1.25),
        ("model_creation_s", 0.5),
        ("total_s", 1.75),
    ]


def test_compute_time_seconds_reads_total_stage():
    assert compute_time_seconds({"stage_timings_s": {"triangulation_s": 1.0, "total_s": 3.5}}) == 3.5
    assert compute_time_seconds({"stage_timings_s": {"triangulation_s": 1.0}}) is None


def test_format_reconstruction_timing_details_lists_stage_breakdown():
    summary = {
        "name": "ekf_2d_acc",
        "family": "ekf_2d",
        "n_frames": 1746,
        "duration_s": 14.55,
        "stage_timings_s": {
            "triangulation_s": 1.0,
            "ekf_2d_solve_s": 0.25,
            "total_s": 2.0,
        },
    }
    details = format_reconstruction_timing_details(summary)
    assert "Name: ekf_2d_acc" in details
    assert "Sequence duration: 14.55 s" in details
    assert "Objective compute time: 2.00 s" in details
    assert "Triangulation: 1.00 s" in details
    assert "EKF 2D solve: 0.25 s" in details


def test_format_helpers_cover_known_names_and_long_durations():
    assert humanize_stage_name("epipolar_coherence_s") == "Epipolar coherence"
    assert format_seconds_brief(61.2) == "1 min 1.2 s"


def test_objective_total_seconds_prefers_pipeline_trace_sum():
    summary = {
        "stage_timings_s": {"total_s": 1.2},
        "pipeline_timing": {
            "stages": [
                make_timing_stage("pose_data", "2D cleaning", compute_time_s=0.4, source="cache"),
                make_timing_stage("triangulation", "Triangulation", compute_time_s=1.1),
                make_timing_stage("ekf_2d_solve", "EKF 2D solve", compute_time_s=0.2, include_in_total=False),
            ]
        },
    }
    assert objective_total_seconds(summary) == 1.5


def test_build_pipeline_diagram_marks_cached_stages():
    diagram = build_pipeline_diagram(
        [
            make_timing_stage("pose_data", "2D cleaning", compute_time_s=0.5, source="cache"),
            make_timing_stage("triangulation", "Triangulation", compute_time_s=1.0),
        ]
    )
    assert diagram == "2D cleaning [cache] -> Triangulation"


def test_format_reconstruction_timing_details_uses_pipeline_trace_when_present():
    summary = {
        "name": "ekf_2d_acc",
        "family": "ekf_2d",
        "n_frames": 100,
        "duration_s": 0.83,
        "pipeline_timing": {
            "current_run_wall_s": 0.4,
            "stages": [
                make_timing_stage("pose_data", "2D cleaning", compute_time_s=0.2, source="cache", cache_path="/tmp/p"),
                make_timing_stage("triangulation", "Triangulation", compute_time_s=1.0),
                make_timing_stage("ekf_2d", "EKF 2D", compute_time_s=0.6),
            ],
        },
    }
    details = format_reconstruction_timing_details(summary)
    assert "Objective compute time: 1.80 s" in details
    assert "Current run wall time: 0.40 s" in details
    assert "2D cleaning [cache] -> Triangulation -> EKF 2D" in details
    assert "cache: /tmp/p" in details
