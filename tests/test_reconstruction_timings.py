from reconstruction_timings import (
    compute_time_seconds,
    format_reconstruction_timing_details,
    format_seconds_brief,
    humanize_stage_name,
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
    assert "Compute time: 2.00 s" in details
    assert "Triangulation: 1.00 s" in details
    assert "EKF 2D solve: 0.25 s" in details


def test_format_helpers_cover_known_names_and_long_durations():
    assert humanize_stage_name("epipolar_coherence_s") == "Epipolar coherence"
    assert format_seconds_brief(61.2) == "1 min 1.2 s"
