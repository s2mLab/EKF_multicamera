import math
from pathlib import Path

import numpy as np

from reconstruction.reconstruction_timings import (
    build_pipeline_diagram,
    compute_time_seconds,
    format_reconstruction_timing_details,
    format_seconds_brief,
    humanize_stage_name,
    make_timing_stage,
    model_compute_seconds,
    objective_total_seconds,
    parse_stage_timings,
    reconstruction_run_seconds,
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


def test_model_and_reconstruction_seconds_split_pipeline_total():
    summary = {
        "pipeline_timing": {
            "stages": [
                make_timing_stage("pose_data", "2D cleaning", compute_time_s=0.4, source="cache"),
                make_timing_stage("triangulation", "Triangulation", compute_time_s=1.1),
                make_timing_stage("model_creation", "Model creation", compute_time_s=0.6),
                make_timing_stage("ekf_2d", "EKF 2D", compute_time_s=0.2),
            ]
        }
    }
    assert math.isclose(objective_total_seconds(summary), 2.3)
    assert math.isclose(model_compute_seconds(summary), 0.6)
    assert math.isclose(reconstruction_run_seconds(summary), 1.7)


def test_model_and_reconstruction_seconds_prefer_stage_wall_times():
    summary = {
        "stage_timings_s": {
            "triangulation_s": 2.0,
            "model_creation_s": 0.7,
            "ekf_2d_s": 1.3,
            "total_s": 4.0,
        },
        "pipeline_timing": {
            "stages": [
                make_timing_stage("triangulation", "Triangulation", compute_time_s=2.0),
                make_timing_stage("model_creation", "Model creation", compute_time_s=0.0, source="provided"),
                make_timing_stage("ekf_2d", "EKF 2D", compute_time_s=1.3),
            ]
        },
    }
    assert math.isclose(objective_total_seconds(summary), 3.3)
    assert math.isclose(model_compute_seconds(summary), 0.7)
    assert math.isclose(reconstruction_run_seconds(summary), 2.6)


def test_objective_model_time_can_be_recovered_from_selected_biomod_cache(tmp_path: Path):
    model_dir = tmp_path / "model_demo"
    model_dir.mkdir()
    biomod_path = model_dir / "model_demo.bioMod"
    biomod_path.write_text("version 4", encoding="utf-8")
    np.savez(
        model_dir / "model_stage.npz",
        lengths=np.asarray("{}", dtype=object),
        biomod_path=np.asarray(str(biomod_path), dtype=object),
        compute_time_s=np.asarray(1.25, dtype=float),
        metadata=np.asarray("{}", dtype=object),
    )

    summary = {
        "selected_model": str(biomod_path),
        "pipeline_timing": {
            "stages": [
                make_timing_stage("triangulation", "Triangulation", compute_time_s=2.0),
                make_timing_stage("model_creation", "Model creation", compute_time_s=0.0, source="provided"),
                make_timing_stage("ekf_3d", "EKF 3D", compute_time_s=0.5),
            ]
        },
    }

    assert math.isclose(model_compute_seconds(summary), 1.25)
    assert math.isclose(objective_total_seconds(summary), 3.75)
    assert math.isclose(reconstruction_run_seconds(summary), 2.5)


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
    assert "Reconstruction time (excl. model): 1.80 s" in details
    assert "Model time: -" in details
    assert "Current run wall time: 0.40 s" in details
    assert "2D cleaning [cache] -> Triangulation -> EKF 2D" in details
    assert "cache: /tmp/p" in details


def test_format_reconstruction_timing_details_shows_model_time_when_present():
    summary = {
        "name": "ekf_2d_acc",
        "family": "ekf_2d",
        "n_frames": 100,
        "duration_s": 0.83,
        "pipeline_timing": {
            "stages": [
                make_timing_stage("triangulation", "Triangulation", compute_time_s=1.0),
                make_timing_stage("model_creation", "Model creation", compute_time_s=0.4),
                make_timing_stage("ekf_2d", "EKF 2D", compute_time_s=0.6),
            ],
        },
    }
    details = format_reconstruction_timing_details(summary)
    assert "Objective compute time: 2.00 s" in details
    assert "Reconstruction time (excl. model): 1.60 s" in details
    assert "Model time: 0.40 s" in details


def test_format_reconstruction_timing_details_shows_trc_source_details():
    source_path = "output/1_partie_0429/reconstructions/pose2sim_rotfix/markers_from_q.trc"
    summary = {
        "name": "pose2sim_rotfix",
        "family": "pose2sim",
        "source": source_path,
        "trc_rate_hz": 120.0,
        "n_frames": 1746,
        "fps": 120.0,
        "duration_s": 14.55,
        "stage_timings_s": {"total_s": 0.75},
    }
    details = format_reconstruction_timing_details(summary)
    assert f"Source file: {source_path}" in details
    assert "TRC rate: 120.00 Hz" in details
