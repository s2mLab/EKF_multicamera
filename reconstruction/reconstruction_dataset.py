#!/usr/bin/env python3
"""Utilitaires partages pour les datasets de reconstructions bundle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reconstruction.reconstruction_registry import scan_reconstruction_dirs

DEFAULT_CALIB = Path("inputs/calibration/Calib.toml")

KNOWN_RECONSTRUCTION_LABELS = {
    "pose2sim": "Pose2Sim",
    "triangulation_adaptive": "Triangulation adaptive",
    "triangulation_fast": "Triangulation fast",
    "triangulation_exhaustive": "Triangulation exhaustive",
    "triangulation_greedy": "Triangulation greedy",
    "ekf_2d_acc": "EKF 2D ACC",
    "ekf_2d_flip_acc": "EKF 2D FLIP ACC",
    "ekf_2d_dyn": "EKF 2D DYN",
    "ekf_2d_flip_dyn": "EKF 2D FLIP DYN",
    "ekf_3d": "EKF 3D",
    "raw": "Brut 2D",
}

KNOWN_RECONSTRUCTION_COLORS = {
    "pose2sim": "black",
    "triangulation_adaptive": "#dd8452",
    "triangulation_fast": "#f2a104",
    "triangulation_exhaustive": "#dd8452",
    "triangulation_greedy": "#f2a104",
    "ekf_2d_acc": "#c44e52",
    "ekf_2d_flip_acc": "#937860",
    "ekf_2d_dyn": "#8172b3",
    "ekf_2d_flip_dyn": "#da8bc3",
    "ekf_3d": "#55a868",
    "raw": "#7a7a7a",
}

FALLBACK_COLORS = [
    "#4c72b0",
    "#dd8452",
    "#55a868",
    "#c44e52",
    "#8172b3",
    "#937860",
    "#da8bc3",
    "#8c8c8c",
]

PREFERRED_MASTER_NAMES = (
    "pose2sim",
    "triangulation_exhaustive",
    "triangulation_adaptive",
    "triangulation_greedy",
    "triangulation_fast",
    "ekf_2d_flip_acc",
    "ekf_2d_acc",
    "ekf_3d",
)
PREFERRED_TRIANGULATION_NAMES = (
    "triangulation_exhaustive",
    "triangulation_adaptive",
    "triangulation_greedy",
    "triangulation_fast",
)


def load_json_if_exists(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_dataset_dir(path: Path) -> Path:
    path = Path(path)
    if (path / "reconstruction_bundle.npz").exists():
        return path.parent
    return path


def reconstruction_dirs_for_path(path: Path) -> list[Path]:
    path = Path(path)
    if (path / "reconstruction_bundle.npz").exists():
        return [path]
    return [
        bundle_dir
        for bundle_dir in scan_reconstruction_dirs(path)
        if (bundle_dir / "reconstruction_bundle.npz").exists()
    ]


def dataset_manifest(dataset_dir: Path) -> dict[str, object]:
    return load_json_if_exists(Path(dataset_dir) / "manifest.json")


def dataset_name_from_dir(dataset_dir: Path) -> str:
    manifest = dataset_manifest(dataset_dir)
    return str(manifest.get("dataset_name", Path(dataset_dir).name))


def dataset_source_paths(
    dataset_dir: Path,
    *,
    calib: Path | None = None,
    keypoints: Path | None = None,
    pose2sim_trc: Path | None = None,
) -> dict[str, Path]:
    dataset_dir = Path(dataset_dir)
    dataset_name = dataset_name_from_dir(dataset_dir)
    manifest = dataset_manifest(dataset_dir)
    default_keypoints = Path("inputs") / f"{dataset_name}_keypoints.json"
    default_trc = Path("inputs") / f"{dataset_name}.trc"
    return {
        "dataset_name": Path(dataset_name),
        "calib": Path(calib) if calib is not None else Path(manifest.get("calib", DEFAULT_CALIB)),
        "keypoints": Path(keypoints) if keypoints is not None else Path(manifest.get("keypoints", default_keypoints)),
        "pose2sim_trc": (
            Path(pose2sim_trc) if pose2sim_trc is not None else Path(manifest.get("pose2sim_trc", default_trc))
        ),
    }


def reconstruction_label(name: str) -> str:
    if name in KNOWN_RECONSTRUCTION_LABELS:
        return KNOWN_RECONSTRUCTION_LABELS[name]
    return name.replace("_", " ").strip()


def reconstruction_color(name: str) -> str:
    if name in KNOWN_RECONSTRUCTION_COLORS:
        return KNOWN_RECONSTRUCTION_COLORS[name]
    color_idx = sum(name.encode("utf-8")) % len(FALLBACK_COLORS)
    return FALLBACK_COLORS[color_idx]


def preferred_triangulation_name(available_names: list[str]) -> str | None:
    available = set(available_names)
    for candidate in PREFERRED_TRIANGULATION_NAMES:
        if candidate in available:
            return candidate
    return None


def preferred_master_name(available_names: list[str]) -> str | None:
    available = set(available_names)
    for candidate in PREFERRED_MASTER_NAMES:
        if candidate in available:
            return candidate
    return available_names[0] if available_names else None


def default_show_names(available_names: list[str]) -> list[str]:
    preferred = []
    for candidate in (
        preferred_triangulation_name(available_names),
        "pose2sim",
        "ekf_3d",
        "ekf_2d_flip_acc",
        "ekf_2d_acc",
    ):
        if candidate is not None and candidate in available_names and candidate not in preferred:
            preferred.append(candidate)
    if preferred:
        return preferred
    return available_names[: min(4, len(available_names))]


def resolve_requested_names(requested: list[str] | tuple[str, ...] | None, available_names: list[str]) -> list[str]:
    if not requested:
        return default_show_names(available_names)

    available = list(available_names)
    resolved: list[str] = []
    for name in requested:
        mapped = name
        if name == "triangulation":
            mapped = preferred_triangulation_name(available) or name
        elif name == "ekf_2d":
            mapped = "ekf_2d_acc" if "ekf_2d_acc" in available else name
        elif name == "biorbd_kalman":
            mapped = "ekf_3d" if "ekf_3d" in available else name
        if mapped in available and mapped not in resolved:
            resolved.append(mapped)
    return resolved


def align_array_to_frames(
    array: np.ndarray, source_frames: np.ndarray, target_frames: np.ndarray, fill_value: float = np.nan
) -> np.ndarray:
    source_frames = np.asarray(source_frames, dtype=int)
    target_frames = np.asarray(target_frames, dtype=int)
    aligned_shape = (len(target_frames),) + tuple(array.shape[1:])
    aligned = np.full(aligned_shape, fill_value, dtype=float)
    frame_to_index = {int(frame): idx for idx, frame in enumerate(source_frames)}
    for out_idx, frame in enumerate(target_frames):
        source_idx = frame_to_index.get(int(frame))
        if source_idx is not None:
            aligned[out_idx] = array[source_idx]
    return aligned


def load_bundle_entries(path: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for bundle_dir in reconstruction_dirs_for_path(path):
        bundle_path = bundle_dir / "reconstruction_bundle.npz"
        if not bundle_path.exists():
            continue
        data = np.load(bundle_path, allow_pickle=True)
        name = str(np.asarray(data["bundle_name"]).item()) if "bundle_name" in data else bundle_dir.name
        frames = (
            np.asarray(data["frames"], dtype=int)
            if "frames" in data
            else np.arange(np.asarray(data["points_3d"]).shape[0], dtype=int)
        )
        time_s = np.asarray(data["time_s"], dtype=float) if "time_s" in data else frames.astype(float) / 120.0
        q_names = np.asarray(data["q_names"], dtype=object) if "q_names" in data else np.array([], dtype=object)
        q_root = (
            np.asarray(data["q_root"], dtype=float) if "q_root" in data else np.empty((len(frames), 0), dtype=float)
        )
        qdot_root = (
            np.asarray(data["qdot_root"], dtype=float)
            if "qdot_root" in data
            else np.empty((len(frames), 0), dtype=float)
        )
        summary = load_json_if_exists(bundle_dir / "bundle_summary.json")
        points_3d = (
            np.asarray(data["points_3d"], dtype=float)
            if "points_3d" in data
            else np.empty((len(frames), 0, 3), dtype=float)
        )
        support_points_3d = (
            np.asarray(data["support_points_3d"], dtype=float)
            if "support_points_3d" in data
            else np.empty((len(frames), 0, 3), dtype=float)
        )
        q = np.asarray(data["q"], dtype=float) if "q" in data else np.empty((len(frames), 0), dtype=float)
        qdot = np.asarray(data["qdot"], dtype=float) if "qdot" in data else np.empty((len(frames), 0), dtype=float)
        entries.append(
            {
                "name": name,
                "path": bundle_dir,
                "frames": frames,
                "time_s": time_s,
                "summary": summary,
                "points_3d": points_3d,
                "support_points_3d": support_points_3d,
                "q": q,
                "qdot": qdot,
                "q_root": q_root,
                "qdot_root": qdot_root,
                "q_names": q_names,
                "points_3d_source": str(summary.get("points_3d_source", "")),
            }
        )
    entries.sort(
        key=lambda entry: (
            list(PREFERRED_MASTER_NAMES).index(entry["name"]) if entry["name"] in PREFERRED_MASTER_NAMES else 999
        )
    )
    return entries


def write_trc_file(
    output_path: Path,
    marker_names: list[str],
    points_3d: np.ndarray,
    frames: np.ndarray,
    time_s: np.ndarray,
    *,
    data_rate: float,
    units: str = "m",
) -> Path:
    """Write a simple TRC file from 3D marker trajectories.

    The output follows the subset of the TRC format already parsed by the
    project utilities: two header rows, a marker-name row, a coordinate-label
    row, then one row per frame with `X/Y/Z` triplets.
    """

    output_path = Path(output_path)
    marker_names = [str(name) for name in marker_names]
    points_3d = np.asarray(points_3d, dtype=float)
    frames = np.asarray(frames, dtype=int)
    time_s = np.asarray(time_s, dtype=float)
    if points_3d.ndim != 3 or points_3d.shape[2] != 3:
        raise ValueError("points_3d must have shape (n_frames, n_markers, 3).")
    if points_3d.shape[0] != len(frames) or points_3d.shape[0] != len(time_s):
        raise ValueError("frames/time_s length must match the first points_3d dimension.")
    if points_3d.shape[1] != len(marker_names):
        raise ValueError("marker_names length must match the second points_3d dimension.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_keys = [
        "DataRate",
        "CameraRate",
        "NumFrames",
        "NumMarkers",
        "Units",
        "OrigDataRate",
        "OrigDataStartFrame",
        "OrigNumFrames",
    ]
    metadata_values = [
        f"{float(data_rate):g}",
        f"{float(data_rate):g}",
        str(int(points_3d.shape[0])),
        str(int(points_3d.shape[1])),
        str(units),
        f"{float(data_rate):g}",
        str(int(frames[0]) if len(frames) else 1),
        str(int(points_3d.shape[0])),
    ]
    marker_row = ["Frame#", "Time"]
    coord_row = ["", ""]
    for marker_idx, marker_name in enumerate(marker_names, start=1):
        marker_row.extend([marker_name, "", ""])
        coord_row.extend([f"X{marker_idx}", f"Y{marker_idx}", f"Z{marker_idx}"])

    lines = [
        f"PathFileType\t4\t(X/Y/Z)\t{output_path.name}",
        "\t".join(metadata_keys),
        "\t".join(metadata_values),
        "\t".join(marker_row),
        "\t".join(coord_row),
    ]
    for frame_idx in range(points_3d.shape[0]):
        row = [str(int(frames[frame_idx])), f"{float(time_s[frame_idx]):.8f}"]
        for point in points_3d[frame_idx]:
            for value in point:
                row.append("" if not np.isfinite(value) else f"{float(value):.8f}")
        lines.append("\t".join(row))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def trc_root_kinematics_sidecar_path(trc_path: Path) -> Path:
    """Return the sidecar path used to persist root kinematics alongside one TRC."""

    trc_path = Path(trc_path)
    return trc_path.with_name(f"{trc_path.stem}_root_kinematics.npz")


def write_trc_root_kinematics_sidecar(
    trc_path: Path,
    q_root: np.ndarray,
    qdot_root: np.ndarray,
    frames: np.ndarray,
    time_s: np.ndarray,
) -> Path:
    """Persist exported root kinematics next to one TRC file.

    The sidecar is used when a TRC generated from model coordinates is later
    re-imported as a ``TRC file`` reconstruction. In that case, reusing the
    original root DoFs yields an exact round-trip, whereas re-extracting the
    root geometrically from markers only gives an approximation.
    """

    q_root = np.asarray(q_root, dtype=float)
    qdot_root = np.asarray(qdot_root, dtype=float)
    frames = np.asarray(frames, dtype=int)
    time_s = np.asarray(time_s, dtype=float)
    if q_root.ndim != 2 or q_root.shape[1] != 6:
        raise ValueError("q_root must have shape (n_frames, 6).")
    if qdot_root.shape != q_root.shape:
        raise ValueError("qdot_root must match q_root shape.")
    if q_root.shape[0] != len(frames) or q_root.shape[0] != len(time_s):
        raise ValueError("frames/time_s length must match q_root length.")

    output_path = trc_root_kinematics_sidecar_path(trc_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, q_root=q_root, qdot_root=qdot_root, frames=frames, time_s=time_s)
    return output_path


def load_trc_root_kinematics_sidecar(trc_path: Path) -> dict[str, np.ndarray] | None:
    """Load the optional root-kinematics sidecar written next to one TRC export."""

    sidecar_path = trc_root_kinematics_sidecar_path(trc_path)
    if not sidecar_path.exists():
        return None
    data = np.load(sidecar_path, allow_pickle=False)
    q_root = np.asarray(data["q_root"], dtype=float)
    qdot_root = np.asarray(data["qdot_root"], dtype=float)
    frames = np.asarray(data["frames"], dtype=int)
    time_s = np.asarray(data["time_s"], dtype=float)
    if q_root.ndim != 2 or q_root.shape[1] != 6:
        return None
    if qdot_root.shape != q_root.shape:
        return None
    if q_root.shape[0] != len(frames) or q_root.shape[0] != len(time_s):
        return None
    return {
        "q_root": q_root,
        "qdot_root": qdot_root,
        "frames": frames,
        "time_s": time_s,
    }
