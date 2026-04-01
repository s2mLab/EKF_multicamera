#!/usr/bin/env python3
"""Batch execution helper for reconstruction profiles plus Excel synthesis export."""

from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from openpyxl import Workbook

from annotation.annotation_store import default_annotation_path
from reconstruction.reconstruction_profiles import ReconstructionProfile, build_pipeline_command, validate_profile
from reconstruction.reconstruction_registry import infer_dataset_name, normalize_output_root

ROOT = Path(__file__).resolve().parent
LOCAL_MPLCONFIG = ROOT / ".cache" / "matplotlib"
LOCAL_MPLCONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_MPLCONFIG))

DEFAULT_CONFIG = Path("reconstruction_profiles.json")
DEFAULT_OUTPUT_ROOT = Path("output")
DEFAULT_CALIB = Path("inputs/calibration/Calib.toml")
DEFAULT_KEYPOINTS_GLOB = "inputs/keypoints/*.json"
DEFAULT_EXCEL_OUTPUT = Path("output/batch_summary.xlsx")


def _require_openpyxl():
    try:
        from openpyxl import Workbook
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Excel export requires `openpyxl`. Install it in the environment or skip the batch workbook export."
        ) from exc
    return Workbook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reconstruction profiles over many keypoint files.")
    parser.add_argument(
        "--keypoints-glob",
        action="append",
        default=[DEFAULT_KEYPOINTS_GLOB],
        help="Glob pattern for source keypoints JSON files. Can be repeated.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Profiles JSON file.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--triangulation-workers", type=int, default=6)
    parser.add_argument("--profile", action="append", default=None, help="Optional subset of profiles to run.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--excel-output", type=Path, default=DEFAULT_EXCEL_OUTPUT)
    parser.add_argument("--batch-name", type=str, default="")
    parser.add_argument("--export-only", action="store_true", help="Skip execution and export from existing manifests.")
    return parser.parse_args()


def discover_keypoints_files(patterns: list[str], *, root: Path = ROOT) -> list[Path]:
    """Resolve glob patterns or explicit paths into one sorted unique list of keypoints JSON files."""

    discovered: list[Path] = []
    seen: set[Path] = set()
    for raw_pattern in patterns:
        pattern = str(raw_pattern).strip()
        if not pattern:
            continue
        has_wildcards = any(char in pattern for char in "*?[]")
        if has_wildcards:
            if Path(pattern).is_absolute():
                matches = [Path(match) for match in glob.glob(pattern)]
            else:
                matches = [Path(match) for match in glob.glob(str(root / pattern))]
        else:
            candidate = Path(pattern)
            if not candidate.is_absolute():
                candidate = root / candidate
            matches = [candidate]
        for match in sorted(matches):
            resolved = Path(match)
            if not resolved.exists() or resolved.suffix.lower() != ".json":
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            discovered.append(resolved)
    return discovered


def infer_pose2sim_trc_for_keypoints(keypoints_path: Path) -> Path | None:
    dataset_name = infer_dataset_name(keypoints_path=keypoints_path)
    candidate = keypoints_path.parent.parent / "trc" / f"{dataset_name}.trc"
    return candidate if candidate.exists() else None


def infer_annotations_for_keypoints(keypoints_path: Path) -> Path | None:
    candidate = default_annotation_path(keypoints_path)
    return candidate if candidate.exists() else None


def build_dataset_run_command(
    *,
    keypoints_path: Path,
    config_path: Path,
    output_root: Path,
    calib_path: Path,
    fps: float,
    triangulation_workers: int,
    selected_profiles: list[str] | None,
    continue_on_error: bool,
    python_executable: str | None = None,
) -> list[str]:
    dataset_name = infer_dataset_name(keypoints_path=keypoints_path)
    pose2sim_trc = infer_pose2sim_trc_for_keypoints(keypoints_path)
    annotations_path = infer_annotations_for_keypoints(keypoints_path)
    cmd = [
        python_executable or sys.executable,
        "run_reconstruction_profiles.py",
        "--config",
        str(config_path),
        "--output-root",
        str(normalize_output_root(output_root)),
        "--dataset-name",
        dataset_name,
        "--calib",
        str(calib_path),
        "--keypoints",
        str(keypoints_path),
        "--fps",
        str(float(fps)),
        "--triangulation-workers",
        str(int(triangulation_workers)),
    ]
    if pose2sim_trc is not None:
        cmd.extend(["--trc-file", str(pose2sim_trc)])
    if annotations_path is not None:
        cmd.extend(["--annotations-path", str(annotations_path)])
    if continue_on_error:
        cmd.append("--continue-on-error")
    for profile_name in selected_profiles or []:
        cmd.extend(["--profile", str(profile_name)])
    return cmd


def _load_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_scalars(
    data: dict[str, Any] | None,
    *,
    prefix: str,
    exclude: set[str] | None = None,
) -> dict[str, Any]:
    exclude = exclude or set()
    flat: dict[str, Any] = {}
    if not isinstance(data, dict):
        return flat
    for key, value in data.items():
        if key in exclude:
            continue
        if isinstance(value, dict):
            for child_key, child_value in _flatten_scalars(value, prefix=f"{prefix}_{key}", exclude=set()).items():
                flat[child_key] = child_value
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flat[f"{prefix}_{key}"] = value
        else:
            flat[f"{prefix}_{key}"] = json.dumps(value, ensure_ascii=True, sort_keys=True)
    return flat


def load_recognition_rows(reconstruction_dir: Path) -> list[dict[str, Any]]:
    """Load optional recognition outputs if future JSON exports are present."""

    rows: list[dict[str, Any]] = []
    for filename, kind in (
        ("dd_analysis.json", "dd"),
        ("execution_analysis.json", "execution"),
        ("recognition_summary.json", "recognition"),
    ):
        path = reconstruction_dir / filename
        payload = _load_json_if_exists(path)
        if payload is None:
            continue
        row = {"kind": kind, "source_file": str(path)}
        row.update(_flatten_scalars(payload, prefix=kind))
        rows.append(row)
    return rows


def collect_manifest_rows(
    batch_manifest: dict[str, Any],
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    """Flatten one batch manifest into workbook sheets."""

    run_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    camera_rows: list[dict[str, Any]] = []
    keypoint_rows: list[dict[str, Any]] = []
    recognition_rows: list[dict[str, Any]] = []

    for dataset_run in batch_manifest.get("datasets", []):
        manifest = dataset_run.get("manifest") or {}
        dataset_name = dataset_run.get("dataset_name") or manifest.get("dataset_name")
        keypoints = dataset_run.get("keypoints")
        annotations_path = dataset_run.get("annotations_path") or manifest.get("annotations_path")
        pose2sim_trc = dataset_run.get("pose2sim_trc") or manifest.get("pose2sim_trc")
        for run in manifest.get("runs", []):
            output_dir = Path(run.get("output_dir", ""))
            summary = run.get("bundle_summary") or {}
            profile_payload = _load_json_if_exists(output_dir / "profile.json") or {}
            profile = validate_profile(ReconstructionProfile(**profile_payload)) if profile_payload else None
            repro_command = None
            if profile is not None:
                repro_command = shlex.join(
                    build_pipeline_command(
                        profile,
                        output_root=Path(batch_manifest.get("output_root", DEFAULT_OUTPUT_ROOT)),
                        calib=Path(batch_manifest.get("calib", DEFAULT_CALIB)),
                        keypoints=Path(keypoints),
                        pose2sim_trc=(None if pose2sim_trc in (None, "") else Path(str(pose2sim_trc))),
                        dataset_name=dataset_name,
                        python_executable=batch_manifest.get("python_executable", sys.executable),
                    )
                    + (["--annotations-path", str(annotations_path)] if annotations_path not in (None, "") else [])
                    + [
                        "--fps",
                        str(batch_manifest.get("fps", 120.0)),
                        "--triangulation-workers",
                        str(batch_manifest.get("triangulation_workers", 6)),
                    ]
                )

            run_row = {
                "dataset_name": dataset_name,
                "keypoints": keypoints,
                "annotations_path": annotations_path,
                "pose2sim_trc": pose2sim_trc,
                "profile_name": run.get("name"),
                "family": run.get("family"),
                "returncode": run.get("returncode"),
                "output_dir": str(output_dir),
                "bundle_summary_path": run.get("bundle_summary_path"),
                "latest_family_version": run.get("latest_family_version"),
                "manifest_stdout": dataset_run.get("stdout", ""),
                "manifest_stderr": dataset_run.get("stderr", ""),
                "repro_command": repro_command,
            }
            run_row.update(
                _flatten_scalars(
                    profile_payload,
                    prefix="profile",
                )
            )
            run_row.update(
                _flatten_scalars(
                    summary,
                    prefix="summary",
                    exclude={
                        "reprojection_px",
                        "stage_timings_s",
                        "pipeline_timing",
                        "cache_paths",
                        "algorithm_versions",
                        "left_right_flip_diagnostics",
                        "ekf2d_initial_state_diagnostics",
                    },
                )
            )
            reprojection = summary.get("reprojection_px") or {}
            run_row["reprojection_mean_px"] = reprojection.get("mean")
            run_row["reprojection_std_px"] = reprojection.get("std")
            run_rows.append(run_row)

            for stage_name, stage_duration in (summary.get("stage_timings_s") or {}).items():
                stage_rows.append(
                    {
                        "dataset_name": dataset_name,
                        "profile_name": run.get("name"),
                        "family": run.get("family"),
                        "output_dir": str(output_dir),
                        "stage_name": stage_name,
                        "duration_s": stage_duration,
                    }
                )

            for camera_name, camera_summary in (reprojection.get("per_camera") or {}).items():
                camera_rows.append(
                    {
                        "dataset_name": dataset_name,
                        "profile_name": run.get("name"),
                        "family": run.get("family"),
                        "camera_name": camera_name,
                        "mean_px": camera_summary.get("mean_px"),
                        "std_px": camera_summary.get("std_px"),
                    }
                )

            for keypoint_name, keypoint_summary in (reprojection.get("per_keypoint") or {}).items():
                keypoint_rows.append(
                    {
                        "dataset_name": dataset_name,
                        "profile_name": run.get("name"),
                        "family": run.get("family"),
                        "keypoint_name": keypoint_name,
                        "mean_px": keypoint_summary.get("mean_px"),
                        "std_px": keypoint_summary.get("std_px"),
                        "n_samples": keypoint_summary.get("n_samples"),
                    }
                )

            for recognition_row in load_recognition_rows(output_dir):
                recognition_rows.append(
                    {
                        "dataset_name": dataset_name,
                        "profile_name": run.get("name"),
                        "family": run.get("family"),
                        "output_dir": str(output_dir),
                        **recognition_row,
                    }
                )

    failures = [
        {
            "dataset_name": dataset_run.get("dataset_name"),
            "keypoints": dataset_run.get("keypoints"),
            "returncode": dataset_run.get("returncode"),
            "command": dataset_run.get("command"),
            "stdout": dataset_run.get("stdout", ""),
            "stderr": dataset_run.get("stderr", ""),
        }
        for dataset_run in batch_manifest.get("datasets", [])
        if int(dataset_run.get("returncode", 0)) != 0
    ]
    return run_rows, stage_rows, camera_rows, keypoint_rows, failures + recognition_rows


def write_rows_sheet(workbook: Any, title: str, rows: list[dict[str, Any]]) -> None:
    worksheet = workbook.create_sheet(title=title)
    if not rows:
        worksheet.append(["empty"])
        return
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                columns.append(key)
    worksheet.append(columns)
    for row in rows:
        worksheet.append([row.get(column) for column in columns])
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions


def write_batch_workbook(batch_manifest: dict[str, Any], workbook_path: Path) -> Path:
    Workbook = _require_openpyxl()
    workbook = Workbook()
    default_sheet = workbook.active
    workbook.remove(default_sheet)

    run_rows, stage_rows, camera_rows, keypoint_rows, recognition_or_failures = collect_manifest_rows(batch_manifest)
    write_rows_sheet(workbook, "Runs", run_rows)
    write_rows_sheet(workbook, "StageTimings", stage_rows)
    write_rows_sheet(workbook, "ReprojectionPerCamera", camera_rows)
    write_rows_sheet(workbook, "ReprojectionPerKeypoint", keypoint_rows)
    write_rows_sheet(workbook, "Recognition", [row for row in recognition_or_failures if "kind" in row])
    write_rows_sheet(workbook, "Failures", [row for row in recognition_or_failures if "kind" not in row])

    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(workbook_path)
    return workbook_path


def run_batch(
    *,
    keypoints_files: list[Path],
    config_path: Path,
    output_root: Path,
    calib_path: Path,
    fps: float,
    triangulation_workers: int,
    selected_profiles: list[str] | None,
    continue_on_error: bool,
    export_only: bool = False,
    python_executable: str | None = None,
) -> dict[str, Any]:
    output_root = normalize_output_root(output_root)
    batch_manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(config_path),
        "output_root": str(output_root),
        "calib": str(calib_path),
        "fps": float(fps),
        "triangulation_workers": int(triangulation_workers),
        "python_executable": python_executable or sys.executable,
        "datasets": [],
    }
    for keypoints_path in keypoints_files:
        dataset_name = infer_dataset_name(keypoints_path=keypoints_path)
        pose2sim_trc = infer_pose2sim_trc_for_keypoints(keypoints_path)
        annotations_path = infer_annotations_for_keypoints(keypoints_path)
        command = build_dataset_run_command(
            keypoints_path=keypoints_path,
            config_path=config_path,
            output_root=output_root,
            calib_path=calib_path,
            fps=fps,
            triangulation_workers=triangulation_workers,
            selected_profiles=selected_profiles,
            continue_on_error=continue_on_error,
            python_executable=python_executable,
        )
        completed = None
        if not export_only:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
        manifest_path = output_root / dataset_name / "manifest.json"
        manifest_payload = _load_json_if_exists(manifest_path) or {
            "dataset_name": dataset_name,
            "keypoints": str(keypoints_path),
            "annotations_path": None if annotations_path is None else str(annotations_path),
            "pose2sim_trc": None if pose2sim_trc is None else str(pose2sim_trc),
            "runs": [],
        }
        dataset_row = {
            "dataset_name": dataset_name,
            "keypoints": str(keypoints_path),
            "annotations_path": None if annotations_path is None else str(annotations_path),
            "pose2sim_trc": None if pose2sim_trc is None else str(pose2sim_trc),
            "manifest_path": str(manifest_path),
            "command": shlex.join(command),
            "returncode": 0 if completed is None else int(completed.returncode),
            "stdout": "" if completed is None else completed.stdout,
            "stderr": "" if completed is None else completed.stderr,
            "manifest": manifest_payload,
        }
        batch_manifest["datasets"].append(dataset_row)
        if completed is not None and completed.returncode != 0 and not continue_on_error:
            break
    return batch_manifest


def main() -> None:
    args = parse_args()
    keypoints_files = discover_keypoints_files(args.keypoints_glob)
    if not keypoints_files:
        raise SystemExit("No keypoints JSON files matched the requested patterns.")
    batch_manifest = run_batch(
        keypoints_files=keypoints_files,
        config_path=args.config,
        output_root=args.output_root,
        calib_path=args.calib,
        fps=args.fps,
        triangulation_workers=args.triangulation_workers,
        selected_profiles=args.profile,
        continue_on_error=args.continue_on_error,
        export_only=args.export_only,
        python_executable=sys.executable,
    )
    batch_name = args.batch_name.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = normalize_output_root(args.output_root) / "_batch" / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = batch_dir / "batch_manifest.json"
    manifest_path.write_text(json.dumps(batch_manifest, indent=2), encoding="utf-8")
    workbook_path = args.excel_output
    if not workbook_path.is_absolute():
        workbook_path = ROOT / workbook_path
    write_batch_workbook(batch_manifest, workbook_path)
    print(f"Batch manifest written to {manifest_path}", flush=True)
    print(f"Batch workbook written to {workbook_path}", flush=True)


if __name__ == "__main__":
    main()
