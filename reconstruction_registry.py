#!/usr/bin/env python3
"""Helpers legers pour organiser les sorties par dataset."""

from __future__ import annotations

import re
from pathlib import Path


BUNDLE_SCHEMA_VERSION = 1
ALGORITHM_VERSIONS = {
    "pose2sim": 1,
    "triangulation": 3,
    "ekf_3d": 5,
    "ekf_2d": 5,
    "pose_cleaning": 1,
    "root_rotation_correction": 1,
}

DEFAULT_MODEL_SUBJECT_MASS_KG = 55.0
DEFAULT_POSE_FILTER_WINDOW = 9
DEFAULT_POSE_OUTLIER_THRESHOLD_RATIO = 0.10
DEFAULT_POSE_AMPLITUDE_LOWER_PERCENTILE = 5.0
DEFAULT_POSE_AMPLITUDE_UPPER_PERCENTILE = 95.0


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "dataset"


def canonical_dataset_name(path_or_name: str | Path) -> str:
    name = Path(path_or_name).stem if isinstance(path_or_name, Path) else str(path_or_name)
    for suffix in ("_keypoints", "_points", "_detections", "_2d"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return slugify(name)


def infer_dataset_name(keypoints_path: Path | None = None, pose2sim_trc: Path | None = None, dataset_name: str | None = None) -> str:
    if dataset_name:
        return canonical_dataset_name(dataset_name)
    if keypoints_path is not None:
        return canonical_dataset_name(keypoints_path)
    if pose2sim_trc is not None:
        return canonical_dataset_name(pose2sim_trc)
    return "dataset"


def dataset_output_dir(output_root: Path, dataset_name: str) -> Path:
    return output_root / canonical_dataset_name(dataset_name)


def dataset_models_dir(output_root: Path, dataset_name: str) -> Path:
    return dataset_output_dir(output_root, dataset_name) / "models"


def dataset_reconstructions_dir(output_root: Path, dataset_name: str) -> Path:
    return dataset_output_dir(output_root, dataset_name) / "reconstructions"


def dataset_figures_dir(output_root: Path, dataset_name: str) -> Path:
    return dataset_output_dir(output_root, dataset_name) / "figures"


def reconstruction_output_dir(output_root: Path, dataset_name: str, reconstruction_name: str) -> Path:
    return dataset_reconstructions_dir(output_root, dataset_name) / slugify(reconstruction_name)


def _format_mass_slug(subject_mass_kg: float | None) -> str | None:
    if subject_mass_kg is None:
        return None
    if abs(float(subject_mass_kg) - DEFAULT_MODEL_SUBJECT_MASS_KG) < 1e-9:
        return None
    return f"m{str(float(subject_mass_kg)).replace('.', 'p')}"


def default_model_stem(
    pose_data_mode: str,
    triangulation_method: str,
    *,
    initial_rotation_correction: bool = False,
    max_frames: int | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
    subject_mass_kg: float | None = None,
    pose_filter_window: int = DEFAULT_POSE_FILTER_WINDOW,
    pose_outlier_threshold_ratio: float = DEFAULT_POSE_OUTLIER_THRESHOLD_RATIO,
    pose_amplitude_lower_percentile: float = DEFAULT_POSE_AMPLITUDE_LOWER_PERCENTILE,
    pose_amplitude_upper_percentile: float = DEFAULT_POSE_AMPLITUDE_UPPER_PERCENTILE,
) -> str:
    tokens = ["model", "2d", slugify(pose_data_mode), slugify(triangulation_method)]
    if initial_rotation_correction:
        tokens.append("rotfix")
    if frame_start is not None or frame_end is not None:
        start = "start" if frame_start is None else str(int(frame_start))
        end = "end" if frame_end is None else str(int(frame_end))
        tokens.append(f"r{start}_{end}")
    if max_frames is not None:
        tokens.append(f"f{int(max_frames)}")
    mass_slug = _format_mass_slug(subject_mass_kg)
    if mass_slug is not None:
        tokens.append(mass_slug)
    if int(pose_filter_window) != DEFAULT_POSE_FILTER_WINDOW:
        tokens.append(f"w{int(pose_filter_window)}")
    if abs(float(pose_outlier_threshold_ratio) - DEFAULT_POSE_OUTLIER_THRESHOLD_RATIO) > 1e-9:
        ratio_pct = int(round(float(pose_outlier_threshold_ratio) * 100.0))
        tokens.append(f"thr{ratio_pct}")
    if (
        abs(float(pose_amplitude_lower_percentile) - DEFAULT_POSE_AMPLITUDE_LOWER_PERCENTILE) > 1e-9
        or abs(float(pose_amplitude_upper_percentile) - DEFAULT_POSE_AMPLITUDE_UPPER_PERCENTILE) > 1e-9
    ):
        tokens.append(
            f"p{int(round(float(pose_amplitude_lower_percentile)))}_{int(round(float(pose_amplitude_upper_percentile)))}"
        )
    return "_".join(tokens)


def model_output_dir(
    output_root: Path,
    dataset_name: str,
    *,
    pose_data_mode: str,
    triangulation_method: str,
    initial_rotation_correction: bool = False,
    max_frames: int | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
    subject_mass_kg: float | None = None,
    pose_filter_window: int = DEFAULT_POSE_FILTER_WINDOW,
    pose_outlier_threshold_ratio: float = DEFAULT_POSE_OUTLIER_THRESHOLD_RATIO,
    pose_amplitude_lower_percentile: float = DEFAULT_POSE_AMPLITUDE_LOWER_PERCENTILE,
    pose_amplitude_upper_percentile: float = DEFAULT_POSE_AMPLITUDE_UPPER_PERCENTILE,
) -> Path:
    stem = default_model_stem(
        pose_data_mode,
        triangulation_method,
        initial_rotation_correction=initial_rotation_correction,
        max_frames=max_frames,
        frame_start=frame_start,
        frame_end=frame_end,
        subject_mass_kg=subject_mass_kg,
        pose_filter_window=pose_filter_window,
        pose_outlier_threshold_ratio=pose_outlier_threshold_ratio,
        pose_amplitude_lower_percentile=pose_amplitude_lower_percentile,
        pose_amplitude_upper_percentile=pose_amplitude_upper_percentile,
    )
    return dataset_models_dir(output_root, dataset_name) / stem


def model_biomod_path(
    output_root: Path,
    dataset_name: str,
    *,
    pose_data_mode: str,
    triangulation_method: str,
    initial_rotation_correction: bool = False,
    max_frames: int | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
    subject_mass_kg: float | None = None,
    pose_filter_window: int = DEFAULT_POSE_FILTER_WINDOW,
    pose_outlier_threshold_ratio: float = DEFAULT_POSE_OUTLIER_THRESHOLD_RATIO,
    pose_amplitude_lower_percentile: float = DEFAULT_POSE_AMPLITUDE_LOWER_PERCENTILE,
    pose_amplitude_upper_percentile: float = DEFAULT_POSE_AMPLITUDE_UPPER_PERCENTILE,
) -> Path:
    model_dir = model_output_dir(
        output_root,
        dataset_name,
        pose_data_mode=pose_data_mode,
        triangulation_method=triangulation_method,
        initial_rotation_correction=initial_rotation_correction,
        max_frames=max_frames,
        frame_start=frame_start,
        frame_end=frame_end,
        subject_mass_kg=subject_mass_kg,
        pose_filter_window=pose_filter_window,
        pose_outlier_threshold_ratio=pose_outlier_threshold_ratio,
        pose_amplitude_lower_percentile=pose_amplitude_lower_percentile,
        pose_amplitude_upper_percentile=pose_amplitude_upper_percentile,
    )
    return model_dir / f"{model_dir.name}.bioMod"


def scan_dataset_dirs(output_root: Path) -> list[Path]:
    if not output_root.exists():
        return []
    dataset_dirs = []
    for child in sorted(output_root.iterdir()):
        if not child.is_dir():
            continue
        recon_dir = child / "reconstructions"
        has_reconstructions = recon_dir.exists() and any(
            (grandchild / "bundle_summary.json").exists() for grandchild in recon_dir.iterdir() if grandchild.is_dir()
        )
        legacy_reconstructions = any((grandchild / "bundle_summary.json").exists() for grandchild in child.iterdir() if grandchild.is_dir())
        if (child / "manifest.json").exists() or has_reconstructions or legacy_reconstructions:
            dataset_dirs.append(child)
    return dataset_dirs


def scan_reconstruction_dirs(dataset_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    if (dataset_dir / "bundle_summary.json").exists():
        candidates.append(dataset_dir)
    reconstructions_dir = dataset_dir / "reconstructions"
    if reconstructions_dir.exists():
        for child in sorted(reconstructions_dir.iterdir()):
            if child.is_dir() and (child / "bundle_summary.json").exists():
                candidates.append(child)
    if dataset_dir.exists():
        for child in sorted(dataset_dir.iterdir()):
            if child.name in {"models", "reconstructions", "figures"}:
                continue
            if child.is_dir() and (child / "bundle_summary.json").exists():
                candidates.append(child)
    return candidates


def scan_model_dirs(dataset_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    models_dir = dataset_dir / "models"
    if models_dir.exists():
        for child in sorted(models_dir.iterdir()):
            if child.is_dir() and any(child.glob("*.bioMod")):
                candidates.append(child)
    legacy_preview = dataset_dir / "model_preview"
    if legacy_preview.exists() and any(legacy_preview.glob("*.bioMod")):
        candidates.append(legacy_preview)
    if any(dataset_dir.glob("*.bioMod")):
        candidates.append(dataset_dir)
    unique = {path.resolve(): path for path in candidates}
    return [unique[key] for key in sorted(unique, key=lambda item: str(item))]


def latest_version_for_family(family: str) -> int | None:
    return ALGORITHM_VERSIONS.get(family)
