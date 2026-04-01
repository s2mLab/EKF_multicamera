import json
from pathlib import Path

import pytest

import batch_run

openpyxl = pytest.importorskip("openpyxl")
load_workbook = openpyxl.load_workbook


def test_discover_keypoints_files_resolves_patterns_and_deduplicates(tmp_path):
    root = tmp_path / "workspace"
    keypoints_dir = root / "inputs" / "keypoints"
    keypoints_dir.mkdir(parents=True)
    trial_a = keypoints_dir / "trial_a_keypoints.json"
    trial_b = keypoints_dir / "trial_b_keypoints.json"
    trial_a.write_text("{}", encoding="utf-8")
    trial_b.write_text("{}", encoding="utf-8")

    discovered = batch_run.discover_keypoints_files(
        ["inputs/keypoints/*.json", "inputs/keypoints/trial_a_keypoints.json"],
        root=root,
    )

    assert discovered == [trial_a, trial_b]


def test_infer_optional_inputs_from_keypoints(tmp_path):
    keypoints_path = tmp_path / "inputs" / "keypoints" / "trial_keypoints.json"
    trc_path = tmp_path / "inputs" / "trc" / "trial.trc"
    annotations_path = tmp_path / "inputs" / "annotations" / "trial_annotations.json"
    keypoints_path.parent.mkdir(parents=True)
    trc_path.parent.mkdir(parents=True)
    annotations_path.parent.mkdir(parents=True)
    keypoints_path.write_text("{}", encoding="utf-8")
    trc_path.write_text("dummy", encoding="utf-8")
    annotations_path.write_text("{}", encoding="utf-8")

    assert batch_run.infer_pose2sim_trc_for_keypoints(keypoints_path) == trc_path
    assert batch_run.infer_annotations_for_keypoints(keypoints_path) == annotations_path


def test_collect_manifest_rows_flattens_summary_and_profile(tmp_path):
    output_root = tmp_path / "output"
    recon_dir = output_root / "trial" / "reconstructions" / "demo"
    recon_dir.mkdir(parents=True)
    (recon_dir / "profile.json").write_text(
        json.dumps(
            {
                "name": "demo",
                "family": "ekf_2d",
                "pose_data_mode": "cleaned",
                "predictor": "dyn",
                "triangulation_method": "exhaustive",
                "coherence_method": "epipolar",
                "frame_stride": 1,
            }
        ),
        encoding="utf-8",
    )
    batch_manifest = {
        "output_root": str(output_root),
        "calib": "inputs/calibration/Calib.toml",
        "fps": 120.0,
        "triangulation_workers": 6,
        "datasets": [
            {
                "dataset_name": "trial",
                "keypoints": "inputs/keypoints/trial_keypoints.json",
                "annotations_path": "inputs/annotations/trial_annotations.json",
                "pose2sim_trc": "inputs/trc/trial.trc",
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "command": "python run_reconstruction_profiles.py ...",
                "manifest": {
                    "dataset_name": "trial",
                    "keypoints": "inputs/keypoints/trial_keypoints.json",
                    "annotations_path": "inputs/annotations/trial_annotations.json",
                    "pose2sim_trc": "inputs/trc/trial.trc",
                    "runs": [
                        {
                            "name": "demo",
                            "family": "ekf_2d",
                            "returncode": 0,
                            "output_dir": str(recon_dir),
                            "latest_family_version": 8,
                            "bundle_summary_path": str(recon_dir / "bundle_summary.json"),
                            "bundle_summary": {
                                "n_frames": 100,
                                "pose_data_mode": "cleaned",
                                "triangulation_method": "exhaustive",
                                "coherence_method": "epipolar",
                                "reprojection_px": {
                                    "mean": 12.5,
                                    "std": 3.4,
                                    "per_camera": {"cam0": {"mean_px": 10.0, "std_px": 2.0}},
                                    "per_keypoint": {"left_wrist": {"mean_px": 15.0, "std_px": 5.0, "n_samples": 80}},
                                },
                                "stage_timings_s": {"triangulation_s": 5.2, "ekf_2d_s": 1.4},
                            },
                        }
                    ],
                },
            }
        ],
    }

    run_rows, stage_rows, camera_rows, keypoint_rows, recognition_or_failures = batch_run.collect_manifest_rows(
        batch_manifest
    )

    assert len(run_rows) == 1
    assert run_rows[0]["dataset_name"] == "trial"
    assert run_rows[0]["profile_family"] == "ekf_2d"
    assert run_rows[0]["reprojection_mean_px"] == 12.5
    assert len(stage_rows) == 2
    assert stage_rows[0]["dataset_name"] == "trial"
    assert len(camera_rows) == 1
    assert camera_rows[0]["camera_name"] == "cam0"
    assert len(keypoint_rows) == 1
    assert keypoint_rows[0]["keypoint_name"] == "left_wrist"
    assert recognition_or_failures == []


def test_write_batch_workbook_creates_expected_sheets(tmp_path):
    workbook_path = tmp_path / "batch.xlsx"
    batch_manifest = {
        "output_root": str(tmp_path / "output"),
        "calib": "inputs/calibration/Calib.toml",
        "fps": 120.0,
        "triangulation_workers": 6,
        "datasets": [],
    }

    written = batch_run.write_batch_workbook(batch_manifest, workbook_path)

    assert written == workbook_path
    assert workbook_path.exists()
    workbook = load_workbook(workbook_path)
    assert workbook.sheetnames == [
        "Runs",
        "StageTimings",
        "ReprojectionPerCamera",
        "ReprojectionPerKeypoint",
        "Recognition",
        "Failures",
    ]


def test_run_batch_export_only_reads_existing_manifest(tmp_path):
    output_root = tmp_path / "output"
    dataset_dir = output_root / "trial"
    dataset_dir.mkdir(parents=True)
    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"dataset_name": "trial", "keypoints": "inputs/keypoints/trial_keypoints.json", "runs": []}),
        encoding="utf-8",
    )
    keypoints_path = tmp_path / "inputs" / "keypoints" / "trial_keypoints.json"
    keypoints_path.parent.mkdir(parents=True)
    keypoints_path.write_text("{}", encoding="utf-8")

    manifest = batch_run.run_batch(
        keypoints_files=[keypoints_path],
        config_path=tmp_path / "profiles.json",
        output_root=output_root,
        calib_path=tmp_path / "Calib.toml",
        fps=120.0,
        triangulation_workers=6,
        selected_profiles=None,
        continue_on_error=False,
        export_only=True,
        python_executable="python",
    )

    assert len(manifest["datasets"]) == 1
    assert manifest["datasets"][0]["dataset_name"] == "trial"
    assert manifest["datasets"][0]["manifest"]["dataset_name"] == "trial"
