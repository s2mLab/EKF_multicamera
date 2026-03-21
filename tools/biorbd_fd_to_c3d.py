#!/usr/bin/env python3
"""
Forward-kinematics trajectory, animation, and C3D export from biorbd model + MAT file.

Typical use:
python biorbd_fd_to_c3d.py \
  --model "/Volumes/10.89.24.15-2/Projet_Trampo/André/data/JeCh/Model/JeCh_201.s2mMod" \
  --q-mat "/Volumes/10.89.24.15-2/Projet_Trampo/André/data/JeCh/Q/Je_833_5_MOD201.13_GenderM_JeChg_Q.mat" \
  --output-c3d "Je_833_5_MOD201.13_markers_joint_centers.c3d" \
  --show
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import biorbd
import ezc3d


def _try_loadmat(path: Path):
    try:
        from scipy.io import loadmat

        return loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception:
        try:
            import h5py
        except Exception as exc:
            raise RuntimeError(
                "Could not read .mat file with scipy.io.loadmat and h5py is unavailable."
            ) from exc

        data = {}
        with h5py.File(path, "r") as f:
            for key in f.keys():
                val = np.array(f[key])
                if val.ndim == 2:
                    val = val.T
                data[key] = val
        return data


def _pick_numeric_array(
    mat_data: dict,
    preferred_keys: Iterable[str],
    explicit_key: Optional[str] = None,
) -> tuple[str, np.ndarray]:
    if explicit_key:
        if explicit_key not in mat_data:
            raise KeyError(f"Requested key '{explicit_key}' not found in MAT file.")
        arr = np.asarray(mat_data[explicit_key], dtype=float)
        return explicit_key, arr

    for key in preferred_keys:
        if key in mat_data:
            arr = np.asarray(mat_data[key], dtype=float)
            if arr.ndim in (1, 2):
                return key, arr

    skip = {"__header__", "__version__", "__globals__"}
    candidates = []
    for key, value in mat_data.items():
        if key in skip:
            continue
        try:
            arr = np.asarray(value, dtype=float)
        except Exception:
            continue
        if arr.ndim in (1, 2) and arr.size > 0:
            candidates.append((key, arr))

    if not candidates:
        raise ValueError("No usable numeric vector/matrix found in MAT file.")

    candidates.sort(key=lambda kv: kv[1].size, reverse=True)
    return candidates[0]


def _ensure_dof_by_frames(arr: np.ndarray, ndof: int, name: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        if arr.size == ndof:
            arr = arr[:, None]
        else:
            raise ValueError(f"{name} is 1D with size {arr.size}, expected {ndof}.")
    elif arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got ndim={arr.ndim}.")

    if arr.shape[0] == ndof:
        return arr
    if arr.shape[1] == ndof:
        return arr.T

    raise ValueError(
        f"{name} shape {arr.shape} does not match model DoF ({ndof}) in either axis."
    )


def _to_np(v) -> np.ndarray:
    return np.asarray(v.to_array() if hasattr(v, "to_array") else v, dtype=float)


def _str_name(n) -> str:
    if hasattr(n, "to_string"):
        return n.to_string()
    return str(n)


def _segment_origin(model: biorbd.Model, q: np.ndarray, iseg: int) -> np.ndarray:
    rt = model.globalJCS(q, iseg)
    rt_np = _to_np(rt)
    if rt_np.shape == (4, 4):
        return rt_np[:3, 3]
    if rt_np.size >= 3:
        return rt_np[:3]
    raise ValueError(f"Unexpected RotoTrans shape for segment {iseg}: {rt_np.shape}")


def _collect_points(
    model: biorbd.Model, q_traj: np.ndarray
) -> tuple[np.ndarray, list[str], int, int]:
    marker_names = [_str_name(n) for n in model.markerNames()]
    seg_names = [_str_name(model.segment(i).name()) for i in range(model.nbSegment())]
    jc_names = [f"JC_{name}" for name in seg_names]

    n_markers = len(marker_names)
    n_jc = len(jc_names)
    n_frames = q_traj.shape[1]
    n_points = n_markers + n_jc

    points = np.zeros((3, n_points, n_frames))
    for i in range(n_frames):
        q = q_traj[:, i]
        markers = model.markers(q)
        for m in range(n_markers):
            points[:, m, i] = _to_np(markers[m])[:3]
        for s in range(n_jc):
            points[:, n_markers + s, i] = _segment_origin(model, q, s)

    labels = marker_names + jc_names
    return points, labels, n_markers, n_jc


def _write_c3d(points_xyz: np.ndarray, labels: list[str], fps: float, out_path: Path) -> None:
    n_points, n_frames = points_xyz.shape[1], points_xyz.shape[2]
    c3d = ezc3d.c3d()
    c3d["parameters"]["POINT"]["LABELS"]["value"] = labels
    c3d["parameters"]["POINT"]["UNITS"]["value"] = ["m"]
    c3d["parameters"]["POINT"]["RATE"]["value"] = [float(fps)]
    c3d["parameters"]["POINT"]["USED"]["value"] = [int(n_points)]

    data_points = np.zeros((4, n_points, n_frames))
    data_points[:3, :, :] = points_xyz
    data_points[3, :, :] = 0.0
    c3d["data"]["points"] = data_points
    c3d.write(str(out_path))


def _animate(points_xyz: np.ndarray, n_markers: int, fps: float, save_mp4: Optional[Path]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    n_frames = points_xyz.shape[2]
    xyz_min = points_xyz.reshape(3, -1).min(axis=1)
    xyz_max = points_xyz.reshape(3, -1).max(axis=1)
    center = 0.5 * (xyz_min + xyz_max)
    radius = max(float(np.max(xyz_max - xyz_min)) * 0.6, 0.5)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Markers (blue) and Joint Centers (red)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))

    marker_scatter = ax.scatter([], [], [], c="tab:blue", s=20, label="Markers")
    jc_scatter = ax.scatter([], [], [], c="tab:red", s=20, label="Joint centers")
    ax.legend(loc="upper right")

    def _update(frame: int):
        xyz = points_xyz[:, :, frame]
        m = xyz[:, :n_markers]
        j = xyz[:, n_markers:]
        marker_scatter._offsets3d = (m[0], m[1], m[2])
        jc_scatter._offsets3d = (j[0], j[1], j[2])
        ax.set_title(f"Frame {frame + 1}/{n_frames}")
        return marker_scatter, jc_scatter

    anim = FuncAnimation(
        fig,
        _update,
        frames=n_frames,
        interval=1000.0 / fps,
        blit=False,
        repeat=True,
    )

    if save_mp4:
        anim.save(str(save_mp4), fps=fps, dpi=150)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load biorbd model and generalized coordinates from MAT, run forward kinematics, "
            "animate markers/joint centers, and export to C3D."
        )
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to .s2mMod/.bioMod model.")
    parser.add_argument("--q-mat", type=Path, required=True, help="Path to MAT file with generalized coordinates q.")
    parser.add_argument("--q-key", type=str, default=None, help="Optional MAT key for q.")
    parser.add_argument("--fps", type=float, default=100.0, help="Frame rate used for finite differences and C3D.")
    parser.add_argument(
        "--output-c3d",
        type=Path,
        default=Path("markers_and_joint_centers.c3d"),
        help="Output C3D path.",
    )
    parser.add_argument("--show", action="store_true", help="Display animation window.")
    parser.add_argument("--save-mp4", type=Path, default=None, help="Optional MP4 animation output.")
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not args.q_mat.exists():
        raise FileNotFoundError(f"q MAT file not found: {args.q_mat}")

    model = biorbd.Model(str(args.model))
    nq = model.nbQ()

    q_data = _try_loadmat(args.q_mat)
    q_key, q_raw = _pick_numeric_array(q_data, preferred_keys=("Q", "q", "q_opt"), explicit_key=args.q_key)
    q = _ensure_dof_by_frames(q_raw, nq, f"q (key={q_key})")
    n_frames = q.shape[1]

    points_xyz, labels, n_markers, n_jc = _collect_points(model, q)

    _write_c3d(points_xyz, labels, args.fps, args.output_c3d)

    print(f"Model DoF: {nq}")
    print(f"Frames: {n_frames}")
    print(f"Markers: {n_markers}")
    print(f"Joint centers (segment origins): {n_jc}")
    print(f"C3D written: {args.output_c3d}")

    if args.show or args.save_mp4:
        _animate(points_xyz, n_markers=n_markers, fps=args.fps, save_mp4=args.save_mp4)


if __name__ == "__main__":
    main()
