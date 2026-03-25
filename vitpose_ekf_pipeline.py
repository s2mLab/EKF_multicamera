#!/usr/bin/env python3
"""Pipeline de reconstruction cinematique a partir de keypoints 2D multi-vues.

Le script enchaine quatre etapes:
1. lecture des calibrations camera et des detections 2D VITPose/COCO17,
2. triangulation 3D initiale ponderee par la confiance des keypoints,
3. construction d'un modele `.bioMod` minimal avec `biobuddy`,
4. estimation cinematique avec un EKF multi-vues base sur `biorbd`, puis
   comparaison optionnelle avec le Kalman marqueurs classique de `biorbd`.

Le but n'est pas de reproduire toute la chaine Pose2Sim/biobuddy, mais de
fournir un pipeline compact et modifiable autour des donnees du projet.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import functools
import hashlib
import itertools as it
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from camera_tools.camera_selection import parse_camera_names, subset_calibrations
from kinematics.root_kinematics import compute_trunk_dofs_from_points, root_z_correction_angle_from_points

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - python<3.11 fallback
    import tomli as tomllib

try:
    import cv2
except ImportError:  # pragma: no cover - optional but useful for distortion-aware projection
    cv2 = None


DEFAULT_CALIB = Path("inputs/calibration/Calib.toml")
DEFAULT_KEYPOINTS = Path("inputs/keypoints/1_partie_0429_keypoints.json")
LOCAL_BIOBUDDY = Path("/Users/mickaelbegon/Documents/GIT/biobuddy")
LOCAL_MPLCONFIG = Path("/Users/mickaelbegon/Documents/Playground/.cache/matplotlib")
LOCAL_MPLCONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_MPLCONFIG))
DEFAULT_CAMERA_FPS = 120.0
DEFAULT_REPROJECTION_THRESHOLD_PX = 15.0
DEFAULT_EPIPOLAR_THRESHOLD_PX = 15.0
DEFAULT_MIN_CAMERAS_FOR_TRIANGULATION = 3
DEFAULT_MIN_FRAME_COHERENCE_FOR_UPDATE = 0.0
DEFAULT_COHERENCE_METHOD = "epipolar"
DEFAULT_BIORBD_KALMAN_NOISE_FACTOR = 1e-8
DEFAULT_BIORBD_KALMAN_ERROR_FACTOR = 1e-4
DEFAULT_BIORBD_KALMAN_INIT_METHOD = "triangulation_ik_root_translation"
DEFAULT_MEASUREMENT_NOISE_SCALE = 1.5
DEFAULT_TRIANGULATION_METHOD = "exhaustive"
DEFAULT_TRIANGULATION_WORKERS = 6
SUPPORTED_TRIANGULATION_METHODS = ("once", "greedy", "exhaustive")
SUPPORTED_COHERENCE_METHODS = (
    "epipolar",
    "epipolar_fast",
    "triangulation",
    "triangulation_once",
    "triangulation_greedy",
    "triangulation_exhaustive",
)
SUPPORTED_FLIP_METHODS = (
    "epipolar",
    "epipolar_fast",
    "epipolar_viterbi",
    "epipolar_fast_viterbi",
    "ekf_prediction_gate",
    "triangulation_once",
    "triangulation_greedy",
    "triangulation_exhaustive",
)
DEFAULT_COHERENCE_CONFIDENCE_FLOOR = 0.35
DEFAULT_SUBJECT_MASS_KG = 55.0
DEFAULT_FLIGHT_HEIGHT_THRESHOLD_M = 1.5
DEFAULT_FLIGHT_MIN_CONSECUTIVE_FRAMES = 1
DEFAULT_EKF2D_INITIAL_STATE_METHOD = "ekf_bootstrap"
DEFAULT_EKF2D_BOOTSTRAP_PASSES = 5
DEFAULT_FLIP_IMPROVEMENT_RATIO = 0.7
DEFAULT_FLIP_MIN_GAIN_PX = 3.0
DEFAULT_FLIP_MIN_OTHER_CAMERAS = 2
DEFAULT_FLIP_RESTRICT_TO_OUTLIERS = True
DEFAULT_FLIP_OUTLIER_PERCENTILE = 85.0
DEFAULT_FLIP_OUTLIER_FLOOR_PX = 5.0
DEFAULT_FLIP_TEMPORAL_WEIGHT = 0.35
DEFAULT_FLIP_TEMPORAL_TAU_PX = 20.0
DEFAULT_FLIP_TEMPORAL_MIN_VALID_KEYPOINTS = 4
DEFAULT_EKF_PREDICTION_GATE_MIN_VALID_KEYPOINTS = 4
DEFAULT_EKF_PREDICTION_GATE_ERROR_THRESHOLD_PX = 12.0
DEFAULT_EKF_PREDICTION_GATE_ERROR_DELTA_THRESHOLD_PX = 4.0
DEFAULT_FLIP_EPIPOLAR_SMOOTH_WINDOW = 5
DEFAULT_FLIP_EPIPOLAR_PAIR_WEIGHT_EXPONENT = 0.5
MODEL_STAGE_VERSION = 7

COCO17 = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
KP_INDEX = {name: i for i, name in enumerate(COCO17)}
LEFT_RIGHT_SWAP_PAIRS = [
    ("left_eye", "right_eye"),
    ("left_ear", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
]
FLIP_PROXIMAL_KEYPOINT_WEIGHTS = {
    "left_shoulder": 2.0,
    "right_shoulder": 2.0,
    "left_hip": 2.0,
    "right_hip": 2.0,
    "left_elbow": 1.5,
    "right_elbow": 1.5,
    "left_knee": 1.5,
    "right_knee": 1.5,
    "left_wrist": 0.75,
    "right_wrist": 0.75,
    "left_ankle": 0.75,
    "right_ankle": 0.75,
    "nose": 0.5,
    "left_eye": 0.4,
    "right_eye": 0.4,
    "left_ear": 0.3,
    "right_ear": 0.3,
}


@dataclass
class CameraCalibration:
    """Parametres intrinsèques/extrinseques d'une camera.

    Les attributs sont stockes une fois pour eviter de recalculer la meme
    geometrie a chaque projection dans l'EKF.
    """

    name: str
    image_size: tuple[int, int]
    K: np.ndarray
    dist: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    R: np.ndarray
    P: np.ndarray

    def project_point(self, point_world: np.ndarray) -> np.ndarray:
        """Projette un point 3D monde en coordonnees pixels sans distortion explicite."""
        point_cam = self.world_to_camera(point_world)
        z = max(point_cam[2], 1e-9)
        uv_h = self.K @ np.array([point_cam[0] / z, point_cam[1] / z, 1.0])
        return uv_h[:2]

    def world_to_camera(self, point_world: np.ndarray) -> np.ndarray:
        """Transforme un point du repere monde vers le repere camera."""
        return self.R @ np.asarray(point_world, dtype=float).reshape(3) + self.tvec.reshape(3)

    def projection_jacobian(self, point_world: np.ndarray) -> np.ndarray:
        """Jacobienne analytique de la projection perspective par rapport au point 3D monde."""
        point_cam = self.world_to_camera(point_world)
        return self.projection_jacobian_from_camera_point(point_cam)

    def projection_jacobian_from_camera_point(self, point_cam: np.ndarray) -> np.ndarray:
        """Jacobienne analytique de la projection perspective a partir du point camera."""
        x, y, z = point_cam
        z = max(z, 1e-9)
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        jac_cam = np.array(
            [
                [fx / z, 0.0, -fx * x / (z * z)],
                [0.0, fy / z, -fy * y / (z * z)],
            ]
        )
        return jac_cam @ self.R

    def project_point_and_jacobian(self, point_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Retourne en une seule passe la projection 2D et sa jacobienne."""
        point_cam = self.world_to_camera(point_world)
        z = max(point_cam[2], 1e-9)
        uv_h = self.K @ np.array([point_cam[0] / z, point_cam[1] / z, 1.0])
        return uv_h[:2], self.projection_jacobian_from_camera_point(point_cam)

    def project_points_and_jacobians(self, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Projette plusieurs points 3D et leurs jacobiennes en une seule passe."""
        points_world = np.asarray(points_world, dtype=float)
        if points_world.ndim != 2 or points_world.shape[1] != 3:
            return np.empty((0, 2), dtype=float), np.empty((0, 2, 3), dtype=float)
        point_cam = points_world @ self.R.T + self.tvec.reshape(1, 3)
        z = np.maximum(point_cam[:, 2], 1e-9)
        normalized = np.column_stack(
            (point_cam[:, 0] / z, point_cam[:, 1] / z, np.ones(point_cam.shape[0], dtype=float))
        )
        uv_h = normalized @ self.K.T
        fx = float(self.K[0, 0])
        fy = float(self.K[1, 1])
        jac_cam = np.zeros((points_world.shape[0], 2, 3), dtype=float)
        jac_cam[:, 0, 0] = fx / z
        jac_cam[:, 0, 2] = -fx * point_cam[:, 0] / (z * z)
        jac_cam[:, 1, 1] = fy / z
        jac_cam[:, 1, 2] = -fy * point_cam[:, 1] / (z * z)
        jac_world = np.einsum("mab,bc->mac", jac_cam, self.R)
        return uv_h[:, :2], jac_world


@dataclass
class PoseData:
    """Keypoints 2D et scores de confiance synchronises sur un nombre de frames commun."""

    camera_names: list[str]
    frames: np.ndarray
    keypoints: np.ndarray  # (n_cam, n_frames, 17, 2)
    scores: np.ndarray  # (n_cam, n_frames, 17)
    frame_stride: int = 1
    raw_keypoints: np.ndarray | None = None  # (n_cam, n_frames, 17, 2)
    filtered_keypoints: np.ndarray | None = None  # (n_cam, n_frames, 17, 2)


@dataclass
class SegmentLengths:
    """Longueurs segmentaires estimees a partir de la reconstruction 3D initiale."""

    trunk_height: float
    head_length: float
    shoulder_half_width: float
    hip_half_width: float
    upper_arm_length: float
    forearm_length: float
    thigh_length: float
    shank_length: float
    eye_offset_x: float
    eye_offset_y: float
    ear_offset_y: float


@dataclass
class ReconstructionResult:
    """Resultat de la triangulation initiale avant filtrage cinematique."""

    frames: np.ndarray
    points_3d: np.ndarray  # (n_frames, 17, 3)
    mean_confidence: np.ndarray  # (n_frames, 17)
    reprojection_error: np.ndarray  # (n_frames, 17)
    reprojection_error_per_view: np.ndarray  # (n_frames, 17, n_cam)
    multiview_coherence: np.ndarray  # (n_frames, 17, n_cam), score actif utilise par l'EKF
    epipolar_coherence: np.ndarray  # (n_frames, 17, n_cam)
    triangulation_coherence: np.ndarray  # (n_frames, 17, n_cam)
    excluded_views: np.ndarray  # (n_frames, 17, n_cam)
    coherence_method: str
    epipolar_coherence_compute_time_s: float = 0.0
    triangulation_compute_time_s: float = 0.0


@dataclass
class ComparisonResult:
    """Conteneur des sorties pour la comparaison EKF multi-vues vs Kalman `biorbd`."""

    q_ekf: np.ndarray
    q_ekf_3d: np.ndarray
    qdot_ekf_3d: np.ndarray
    qddot_ekf_3d: np.ndarray
    rmse_per_dof: np.ndarray
    mae_per_dof: np.ndarray
    ekf_2d_reprojection_mean_px: float
    ekf_2d_reprojection_std_px: float
    ekf_3d_reprojection_mean_px: float
    ekf_3d_reprojection_std_px: float
    q_names: np.ndarray


def save_single_ekf_state(path: Path, ekf_result: dict[str, np.ndarray]) -> None:
    """Sauvegarde une trajectoire EKF dans un NPZ simple."""
    payload = {
        "q": ekf_result["q"],
        "qdot": ekf_result["qdot"],
        "qddot": ekf_result["qddot"],
        "q_names": ekf_result["q_names"],
    }
    if "update_status_per_frame" in ekf_result:
        payload["update_status_per_frame"] = ekf_result["update_status_per_frame"]
    np.savez(path, **payload)


def comparison_to_npz_payload(comparison: ComparisonResult) -> dict[str, np.ndarray]:
    """Convertit une comparaison en payload `np.savez`."""
    return {
        "q_ekf": comparison.q_ekf,
        "q_ekf_3d": comparison.q_ekf_3d,
        "qdot_ekf_3d": comparison.qdot_ekf_3d,
        "qddot_ekf_3d": comparison.qddot_ekf_3d,
        "q_biorbd_kalman": comparison.q_ekf_3d,
        "qdot_biorbd_kalman": comparison.qdot_ekf_3d,
        "qddot_biorbd_kalman": comparison.qddot_ekf_3d,
        "rmse_per_dof": comparison.rmse_per_dof,
        "mae_per_dof": comparison.mae_per_dof,
        "ekf_2d_reprojection_mean_px": np.asarray(comparison.ekf_2d_reprojection_mean_px),
        "ekf_2d_reprojection_std_px": np.asarray(comparison.ekf_2d_reprojection_std_px),
        "ekf_3d_reprojection_mean_px": np.asarray(comparison.ekf_3d_reprojection_mean_px),
        "ekf_3d_reprojection_std_px": np.asarray(comparison.ekf_3d_reprojection_std_px),
        "q_names": comparison.q_names,
    }


def comparison_to_summary_dict(
    comparison: ComparisonResult,
    biorbd_kalman_noise_factor: float,
    biorbd_kalman_error_factor: float,
) -> dict[str, object]:
    """Construit le resume JSON associe a une comparaison."""
    return {
        "mean_rmse_rad_or_m": float(np.mean(comparison.rmse_per_dof)),
        "mean_mae_rad_or_m": float(np.mean(comparison.mae_per_dof)),
        "ekf_2d_reprojection_px": {
            "mean": float(comparison.ekf_2d_reprojection_mean_px),
            "std": float(comparison.ekf_2d_reprojection_std_px),
        },
        "ekf_3d_reprojection_px": {
            "mean": float(comparison.ekf_3d_reprojection_mean_px),
            "std": float(comparison.ekf_3d_reprojection_std_px),
        },
        "ekf_3d_parameters": {
            "noise_factor": float(biorbd_kalman_noise_factor),
            "error_factor": float(biorbd_kalman_error_factor),
        },
        "rmse_per_dof": {str(name): float(val) for name, val in zip(comparison.q_names, comparison.rmse_per_dof)},
        "mae_per_dof": {str(name): float(val) for name, val in zip(comparison.q_names, comparison.mae_per_dof)},
    }


def frame_signature(frames: np.ndarray) -> str:
    """Construit une signature stable a partir des ids de frame."""
    frames = np.asarray(frames, dtype=np.int64)
    return hashlib.sha1(frames.tobytes()).hexdigest()


def pose_data_signature(pose_data: PoseData) -> str:
    """Construit une signature stable a partir du contenu utile des observations 2D."""
    hasher = hashlib.sha1()
    hasher.update(np.asarray(pose_data.frames, dtype=np.int64).tobytes())
    hasher.update(np.asarray(pose_data.keypoints, dtype=np.float64).tobytes())
    hasher.update(np.asarray(pose_data.scores, dtype=np.float64).tobytes())
    for camera_name in pose_data.camera_names:
        hasher.update(str(camera_name).encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()[:16]


def select_active_coherence(
    epipolar_coherence: np.ndarray,
    triangulation_coherence: np.ndarray,
    coherence_method: str,
) -> np.ndarray:
    """Selectionne le score de coherence actif selon la methode demandee."""
    coherence_method = canonical_coherence_method(coherence_method)
    if coherence_method in {"epipolar", "epipolar_fast"}:
        return np.array(epipolar_coherence, copy=True)
    if coherence_method in {"triangulation_once", "triangulation_greedy", "triangulation_exhaustive"}:
        return np.array(triangulation_coherence, copy=True)
    raise ValueError(f"Unknown coherence method: {coherence_method}")


def canonical_triangulation_method(triangulation_method: str) -> str:
    """Normalize triangulation method names while keeping legacy aliases working."""

    method = str(triangulation_method).strip().lower()
    if method == "raw":
        return "once"
    if method not in SUPPORTED_TRIANGULATION_METHODS:
        raise ValueError(f"Unknown triangulation method: {triangulation_method}")
    return method


def canonical_coherence_method(coherence_method: str, triangulation_method: str = DEFAULT_TRIANGULATION_METHOD) -> str:
    """Normalize coherence-method aliases to the explicit internal representation."""

    method = str(coherence_method).strip().lower()
    if method == "triangulation":
        return f"triangulation_{canonical_triangulation_method(triangulation_method)}"
    if method not in SUPPORTED_COHERENCE_METHODS:
        raise ValueError(f"Unknown coherence method: {coherence_method}")
    return method


def triangulation_method_from_coherence_method(
    coherence_method: str,
    triangulation_method: str = DEFAULT_TRIANGULATION_METHOD,
) -> str:
    """Return the triangulation variant required by the active coherence method."""

    normalized = canonical_coherence_method(coherence_method, triangulation_method)
    if normalized in {"epipolar", "epipolar_fast"}:
        return canonical_triangulation_method(triangulation_method)
    return normalized.replace("triangulation_", "", 1)


def smooth_valid_1d(values: np.ndarray, valid_mask: np.ndarray, window: int = 9) -> np.ndarray:
    """Lisse une coordonnee 1D en interpolant d'abord les trous valides.

    Le lissage est un moyenneur centre simple, applique sur une trajectoire
    reconstituee uniquement pour estimer une reference filtrée. Les points
    invalides restent `NaN` dans la sortie.
    """
    values = np.asarray(values, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    filtered = np.full(values.shape, np.nan, dtype=float)
    if not np.any(valid_mask):
        return filtered

    valid_idx = np.flatnonzero(valid_mask)
    if valid_idx.size == 1:
        filtered[valid_mask] = values[valid_mask]
        return filtered

    x = np.arange(values.shape[0], dtype=float)
    interpolated = np.interp(x, valid_idx.astype(float), values[valid_mask])
    window = max(1, int(window))
    window = min(window, interpolated.shape[0])
    if window % 2 == 0:
        window = max(1, window - 1)
    if window <= 1:
        smoothed = interpolated
    else:
        kernel = np.ones(window, dtype=float) / float(window)
        smoothed = np.convolve(interpolated, kernel, mode="same")
    filtered[valid_mask] = smoothed[valid_mask]
    return filtered


def filter_pose_keypoints(
    keypoints: np.ndarray,
    scores: np.ndarray,
    smoothing_window: int = 9,
    outlier_threshold_ratio: float = 0.10,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construit une version filtrée des 2D puis rejette les outliers trop éloignés.

    Règle de rejet demandée:
    - on calcule une version filtrée de chaque coordonnée,
    - on calcule la norme 2D entre brut et filtré à chaque instant,
    - on met la coordonnée brute à `NaN` si cette norme dépasse 10% d'une
      amplitude robuste estimée par les percentiles 5-95 sur `(x, y)`.
    """
    keypoints = np.asarray(keypoints, dtype=float)
    scores = np.asarray(scores, dtype=float)
    filtered_keypoints = np.full_like(keypoints, np.nan)
    cleaned_keypoints = np.array(keypoints, copy=True)
    cleaned_scores = np.array(scores, copy=True)

    n_cam, _, n_kp, _ = keypoints.shape
    for cam_idx in range(n_cam):
        for kp_idx in range(n_kp):
            xy = keypoints[cam_idx, :, kp_idx, :]
            score = scores[cam_idx, :, kp_idx]
            valid = np.all(np.isfinite(xy), axis=1) & (score > 0)
            if np.count_nonzero(valid) < 2:
                if np.count_nonzero(valid) == 1:
                    filtered_keypoints[cam_idx, valid, kp_idx, :] = xy[valid]
                continue

            filtered_x = smooth_valid_1d(xy[:, 0], valid, window=smoothing_window)
            filtered_y = smooth_valid_1d(xy[:, 1], valid, window=smoothing_window)
            filtered_xy = np.column_stack((filtered_x, filtered_y))
            filtered_keypoints[cam_idx, :, kp_idx, :] = filtered_xy

            p_low = np.nanpercentile(xy[valid], lower_percentile, axis=0)
            p_high = np.nanpercentile(xy[valid], upper_percentile, axis=0)
            amplitude = float(np.linalg.norm(p_high - p_low))
            threshold = outlier_threshold_ratio * amplitude
            if not np.isfinite(threshold) or threshold <= 0:
                continue

            distance = np.full(xy.shape[0], np.nan, dtype=float)
            valid_filtered = valid & np.all(np.isfinite(filtered_xy), axis=1)
            distance[valid_filtered] = np.linalg.norm(xy[valid_filtered] - filtered_xy[valid_filtered], axis=1)
            outliers = valid_filtered & (distance > threshold)
            if np.any(outliers):
                cleaned_keypoints[cam_idx, outliers, kp_idx, :] = np.nan
                cleaned_scores[cam_idx, outliers, kp_idx] = 0.0

    return cleaned_keypoints, cleaned_scores, filtered_keypoints


def ensure_local_imports() -> None:
    """Configure un cache Matplotlib local et rend le checkout `biobuddy` importable."""
    LOCAL_MPLCONFIG.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_MPLCONFIG))
    if str(LOCAL_BIOBUDDY) not in sys.path:
        sys.path.insert(0, str(LOCAL_BIOBUDDY))


def load_calibrations(calib_path: Path) -> dict[str, CameraCalibration]:
    """Charge le fichier TOML de calibration et construit les matrices utiles.

    Le format attendu est celui fourni dans `Calib.toml`, avec un bloc par
    camera contenant `matrix`, `rotation` et `translation`.
    """
    with calib_path.open("rb") as f:
        raw = tomllib.load(f)

    calibrations: dict[str, CameraCalibration] = {}
    for key, value in raw.items():
        if key == "metadata":
            continue
        K = np.asarray(value["matrix"], dtype=float)
        rvec = np.asarray(value["rotation"], dtype=float).reshape(3, 1)
        tvec = np.asarray(value["translation"], dtype=float).reshape(3, 1)
        if cv2 is not None:
            R, _ = cv2.Rodrigues(rvec)
        else:
            theta = np.linalg.norm(rvec)
            if theta < 1e-12:
                R = np.eye(3)
            else:
                k = (rvec[:, 0] / theta).reshape(3)
                Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
                R = np.eye(3) + math.sin(theta) * Kx + (1 - math.cos(theta)) * (Kx @ Kx)
        P = K @ np.hstack((R, tvec))
        calibrations[key] = CameraCalibration(
            name=value["name"],
            image_size=tuple(value["size"]),
            K=K,
            dist=np.asarray(value["distortions"], dtype=float).reshape(-1),
            rvec=rvec,
            tvec=tvec,
            R=R,
            P=P,
        )
    return calibrations


def load_pose_data(
    keypoints_path: Path,
    calibrations: dict[str, CameraCalibration],
    max_frames: int | None = None,
    frame_stride: int = 1,
    frame_start: int | None = None,
    frame_end: int | None = None,
    data_mode: str = "cleaned",
    smoothing_window: int = 9,
    outlier_threshold_ratio: float = 0.10,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
) -> PoseData:
    """Charge les keypoints 2D et les aligne camera par camera.

    L'alignement se fait sur l'union des indices de frame presents dans les
    differentes vues. Une camera absente sur une frame donnee est remplie avec
    des `NaN` et des scores nuls, afin de conserver toute la timeline sans
    supprimer des frames entieres du pipeline.
    """
    if keypoints_path.suffix.lower() != ".json":
        raise ValueError(
            f"2D keypoints must come from a JSON file, got '{keypoints_path.name}'. "
            "If you selected a .trc file by mistake, choose the matching '*_keypoints.json' file instead."
        )

    try:
        with keypoints_path.open("r") as f:
            raw = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Could not parse 2D keypoints from '{keypoints_path.name}'. "
            "Expected a JSON file with detections. If you selected a .trc file, choose the matching '*_keypoints.json' file instead."
        ) from exc

    ordered_items: list[tuple[str, dict]] = []
    for camera_label, content in raw.items():
        calib_name = camera_label.split("_")[-1]
        if calib_name not in calibrations:
            raise ValueError(f"Camera {camera_label} is absent from calibration file")
        ordered_items.append((calib_name, content))

    # Keep every frame seen by at least one camera. Missing camera observations
    # are represented as NaN/0 so downstream stages can keep the full timeline.
    frame_lists = [np.asarray(content["frames"], dtype=int) for _, content in ordered_items]
    all_frames = set()
    for frame_list in frame_lists:
        all_frames.update(frame_list.tolist())
    if not all_frames:
        raise ValueError("No frame indices were found in the 2D detections.")

    frames = np.array(sorted(all_frames), dtype=int)
    if frame_start is not None:
        frames = frames[frames >= int(frame_start)]
    if frame_end is not None:
        frames = frames[frames <= int(frame_end)]
    frame_stride = max(1, int(frame_stride))
    if frame_stride > 1:
        frames = frames[::frame_stride]
    if max_frames is not None:
        frames = sample_frames_uniformly(frames, int(max_frames))
    n_frames = len(frames)
    if n_frames == 0:
        raise ValueError("No 2D frames remain after applying frame_start/frame_end/max_frames.")

    keypoints = np.full((len(ordered_items), n_frames, len(COCO17), 2), np.nan, dtype=float)
    scores = np.zeros((len(ordered_items), n_frames, len(COCO17)), dtype=float)

    for i_cam, (_, content) in enumerate(ordered_items):
        frame_to_idx = {int(frame): idx for idx, frame in enumerate(np.asarray(content["frames"], dtype=int))}
        kp_all = np.asarray(content["keypoints"], dtype=float)
        score_all = np.asarray(content["scores"], dtype=float)
        for out_idx, frame in enumerate(frames):
            source_idx = frame_to_idx.get(int(frame))
            if source_idx is None:
                continue
            keypoints[i_cam, out_idx] = kp_all[source_idx]
            scores[i_cam, out_idx] = score_all[source_idx]

    raw_keypoints = np.array(keypoints, copy=True)
    raw_scores = np.array(scores, copy=True)
    cleaned_keypoints, cleaned_scores, filtered_keypoints = filter_pose_keypoints(
        keypoints,
        scores,
        smoothing_window=smoothing_window,
        outlier_threshold_ratio=outlier_threshold_ratio,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )

    filtered_scores = np.array(raw_scores, copy=True)
    filtered_scores[~np.all(np.isfinite(filtered_keypoints), axis=3)] = 0.0

    if data_mode == "raw":
        selected_keypoints = raw_keypoints
        selected_scores = raw_scores
    elif data_mode == "filtered":
        selected_keypoints = filtered_keypoints
        selected_scores = filtered_scores
    elif data_mode == "cleaned":
        selected_keypoints = cleaned_keypoints
        selected_scores = cleaned_scores
    else:
        raise ValueError(f"Unsupported pose data mode: {data_mode}")

    return PoseData(
        camera_names=[name for name, _ in ordered_items],
        frames=frames,
        keypoints=selected_keypoints,
        scores=selected_scores,
        frame_stride=frame_stride,
        raw_keypoints=raw_keypoints,
        filtered_keypoints=filtered_keypoints,
    )


def sample_frames_uniformly(frames: np.ndarray, max_frames: int | None) -> np.ndarray:
    """Sous-échantillonne uniformément des frames déjà triées."""
    frames = np.asarray(frames, dtype=int)
    if max_frames is None:
        return np.array(frames, copy=True)
    max_frames = int(max_frames)
    if max_frames <= 0 or frames.size <= max_frames:
        return np.array(frames, copy=True)
    sample_idx = np.linspace(0, frames.size - 1, num=max_frames)
    sample_idx = np.asarray(np.round(sample_idx), dtype=int)
    sample_idx = np.clip(sample_idx, 0, frames.size - 1)
    sample_idx = np.unique(sample_idx)
    if sample_idx.size < max_frames:
        missing = max_frames - sample_idx.size
        remaining = np.setdiff1d(np.arange(frames.size, dtype=int), sample_idx, assume_unique=False)
        if remaining.size:
            extra_positions = np.linspace(0, remaining.size - 1, num=min(missing, remaining.size))
            extra_idx = remaining[np.asarray(np.round(extra_positions), dtype=int)]
            sample_idx = np.sort(np.concatenate((sample_idx, extra_idx)))
    return frames[sample_idx[:max_frames]]


def weighted_triangulation(
    projections: Iterable[np.ndarray], observations: np.ndarray, confidences: np.ndarray
) -> np.ndarray:
    """Triangule un keypoint 3D par DLT ponderee.

    Chaque camera contribue proportionnellement a son score de confiance 2D.
    Cela reproduit l'idee de la triangulation ponderee utilisee dans Pose2Sim.
    """
    projections_array = np.asarray(list(projections), dtype=float)
    observations = np.asarray(observations, dtype=float)
    confidences = np.asarray(confidences, dtype=float)
    if projections_array.ndim != 3 or projections_array.shape[1:] != (3, 4):
        return np.full(3, np.nan)
    valid = (
        np.isfinite(observations[:, 0]) & np.isfinite(observations[:, 1]) & np.isfinite(confidences) & (confidences > 0)
    )
    if np.count_nonzero(valid) < 2:
        return np.full(3, np.nan)
    projections_valid = projections_array[valid]
    observations_valid = observations[valid]
    weights_valid = confidences[valid][:, np.newaxis]
    u = observations_valid[:, 0][:, np.newaxis]
    v = observations_valid[:, 1][:, np.newaxis]
    rows0 = (projections_valid[:, 0, :] - u * projections_valid[:, 2, :]) * weights_valid
    rows1 = (projections_valid[:, 1, :] - v * projections_valid[:, 2, :]) * weights_valid
    A = np.empty((2 * projections_valid.shape[0], 4), dtype=float)
    A[0::2] = rows0
    A[1::2] = rows1
    _, _, vh = np.linalg.svd(A)
    q_h = vh[-1]
    if abs(q_h[3]) < 1e-12:
        return np.full(3, np.nan)
    return q_h[:3] / q_h[3]


@functools.lru_cache(maxsize=64)
def exclusion_combinations(n_valid: int, nb_cams_off: int) -> tuple[tuple[int, ...], ...]:
    return tuple(it.combinations(range(int(n_valid)), int(nb_cams_off)))


def triangulation_reference_from_other_views(
    raw_2d_frame: np.ndarray,
    raw_scores_frame: np.ndarray,
    ordered_calibrations: list[CameraCalibration],
    *,
    min_other_cameras: int = 2,
) -> np.ndarray:
    """Precompute reprojection references for flip diagnostics in triangulation mode.

    For each tested camera and keypoint, the 3D point is triangulated from the
    *other* valid views only once, then reprojected back in the tested camera.
    This avoids recomputing the same triangulation for nominal and swapped
    hypotheses.
    """
    n_cams, n_keypoints = raw_2d_frame.shape[:2]
    reprojected = np.full((n_cams, n_keypoints, 2), np.nan, dtype=float)
    projections = [calibration.P for calibration in ordered_calibrations]
    for cam_idx in range(n_cams):
        other_mask = np.ones(n_cams, dtype=bool)
        other_mask[cam_idx] = False
        for kp_idx in range(n_keypoints):
            valid_other = (
                other_mask & np.all(np.isfinite(raw_2d_frame[:, kp_idx]), axis=1) & (raw_scores_frame[:, kp_idx] > 0)
            )
            if np.count_nonzero(valid_other) < max(2, int(min_other_cameras)):
                continue
            point_3d = weighted_triangulation(
                [projections[idx] for idx in np.flatnonzero(valid_other)],
                raw_2d_frame[valid_other, kp_idx],
                raw_scores_frame[valid_other, kp_idx],
            )
            if not np.all(np.isfinite(point_3d)):
                continue
            reproj = ordered_calibrations[cam_idx].project_point(point_3d)
            if np.all(np.isfinite(reproj)):
                reprojected[cam_idx, kp_idx] = reproj
    return reprojected


def project_point_with_projection_matrices(projection_matrices: np.ndarray, point_world: np.ndarray) -> np.ndarray:
    """Project one 3D point with many `3x4` projection matrices at once."""
    projection_matrices = np.asarray(projection_matrices, dtype=float)
    if projection_matrices.ndim != 3 or projection_matrices.shape[1:] != (3, 4):
        return np.empty((0, 2), dtype=float)
    point_h = np.append(np.asarray(point_world, dtype=float).reshape(3), 1.0)
    uv_h = projection_matrices @ point_h
    z = uv_h[:, 2]
    projected = np.full((projection_matrices.shape[0], 2), np.nan, dtype=float)
    valid = np.isfinite(z) & (np.abs(z) > 1e-12)
    if np.any(valid):
        projected[valid, 0] = uv_h[valid, 0] / z[valid]
        projected[valid, 1] = uv_h[valid, 1] / z[valid]
    return projected


def solve_regularized_kalman_gain(
    predicted_covariance: np.ndarray,
    H_q: np.ndarray,
    R_diag_array: np.ndarray,
    nq: int,
) -> np.ndarray | None:
    """Solve the Kalman gain for one linearized measurement block."""
    if H_q.size == 0 or R_diag_array.size == 0:
        return None
    P_q = predicted_covariance[:, :nq]
    P_qq = predicted_covariance[:nq, :nq]
    PHT = P_q @ H_q.T
    S = H_q @ P_qq @ H_q.T
    diag_idx = np.diag_indices_from(S)
    S[diag_idx] += R_diag_array
    S = 0.5 * (S + S.T)
    if not np.all(np.isfinite(S)):
        return None
    regularization_values = (0.0, 1e-9, 1e-7, 1e-5, 1e-3)
    K = None
    for regularization in regularization_values:
        S_reg = np.array(S, copy=True)
        if regularization > 0.0:
            S_reg[diag_idx] += regularization
        try:
            chol = np.linalg.cholesky(S_reg)
            y = np.linalg.solve(chol, PHT.T)
            K = np.linalg.solve(chol.T, y).T
            break
        except np.linalg.LinAlgError:
            try:
                K = PHT @ np.linalg.pinv(S_reg, rcond=1e-10)
                if np.all(np.isfinite(K)):
                    break
            except np.linalg.LinAlgError:
                K = None
    if K is None or not np.all(np.isfinite(K)):
        return None
    return K


def apply_joseph_measurement_update(
    predicted_state: np.ndarray,
    predicted_covariance: np.ndarray,
    K: np.ndarray,
    innovation: np.ndarray,
    H_q: np.ndarray,
    R_diag_array: np.ndarray,
    nq: int,
    identity_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one Joseph-form covariance update for a measurement block."""
    updated_state = predicted_state + K @ innovation
    I_minus_KH = np.array(identity_x, copy=True)
    I_minus_KH[:, :nq] -= K @ H_q
    updated_covariance = I_minus_KH @ predicted_covariance @ I_minus_KH.T
    updated_covariance += (K * R_diag_array[np.newaxis, :]) @ K.T
    updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)
    return updated_state, updated_covariance


def apply_measurement_update_batch(
    predicted_state: np.ndarray,
    predicted_covariance: np.ndarray,
    z: np.ndarray,
    h: np.ndarray,
    H_q: np.ndarray,
    R_diag_array: np.ndarray,
    nq: int,
    identity_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Apply one batch Kalman correction over all frame measurements."""
    K = solve_regularized_kalman_gain(predicted_covariance, H_q, R_diag_array, nq)
    if K is None:
        return None
    return apply_joseph_measurement_update(
        predicted_state=predicted_state,
        predicted_covariance=predicted_covariance,
        K=K,
        innovation=z - h,
        H_q=H_q,
        R_diag_array=R_diag_array,
        nq=nq,
        identity_x=identity_x,
    )


def apply_measurement_update_sequential(
    predicted_state: np.ndarray,
    predicted_covariance: np.ndarray,
    measurement_blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    nq: int,
    identity_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Apply sequential Kalman corrections over measurement blocks."""
    state = np.array(predicted_state, copy=True)
    covariance = np.array(predicted_covariance, copy=True)
    reference_q = np.asarray(predicted_state[:nq], dtype=float)
    applied = False
    for z, h, H_q, R_diag_array in measurement_blocks:
        if z.size == 0 or H_q.size == 0:
            continue
        current_h = h + H_q @ (np.asarray(state[:nq], dtype=float) - reference_q)
        K = solve_regularized_kalman_gain(covariance, H_q, R_diag_array, nq)
        if K is None:
            return None
        state, covariance = apply_joseph_measurement_update(
            predicted_state=state,
            predicted_covariance=covariance,
            K=K,
            innovation=z - current_h,
            H_q=H_q,
            R_diag_array=R_diag_array,
            nq=nq,
            identity_x=identity_x,
        )
        applied = True
    return (state, covariance) if applied else None


def skew(vector: np.ndarray) -> np.ndarray:
    """Construit la matrice antisymetrique associee a un vecteur 3D."""
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def fundamental_matrix(calib_a: CameraCalibration, calib_b: CameraCalibration) -> np.ndarray:
    """Calcule la matrice fondamentale entre deux cameras calibrees."""
    R_ba = calib_b.R @ calib_a.R.T
    t_ba = calib_b.tvec.reshape(3) - R_ba @ calib_a.tvec.reshape(3)
    E = skew(t_ba) @ R_ba
    K_a_inv = np.linalg.inv(calib_a.K)
    K_b_inv = np.linalg.inv(calib_b.K)
    return K_b_inv.T @ E @ K_a_inv


def sampson_error_pixels(point_a: np.ndarray, point_b: np.ndarray, F_ab: np.ndarray) -> float:
    """Retourne l'erreur de Sampson pour une paire de detections 2D."""
    if not (np.all(np.isfinite(point_a)) and np.all(np.isfinite(point_b))):
        return np.nan
    xa = np.array([point_a[0], point_a[1], 1.0], dtype=float)
    xb = np.array([point_b[0], point_b[1], 1.0], dtype=float)
    Fxa = F_ab @ xa
    Ftxb = F_ab.T @ xb
    denom = Fxa[0] ** 2 + Fxa[1] ** 2 + Ftxb[0] ** 2 + Ftxb[1] ** 2
    if denom <= 1e-12:
        return np.nan
    numer = xb @ F_ab @ xa
    return float(abs(numer) / math.sqrt(denom))


def swap_left_right_keypoints(points_2d: np.ndarray) -> np.ndarray:
    """Construit une hypothese alternative via un swap gauche/droite global."""
    swapped = np.array(points_2d, copy=True)
    for left_name, right_name in LEFT_RIGHT_SWAP_PAIRS:
        left_idx = KP_INDEX[left_name]
        right_idx = KP_INDEX[right_name]
        swapped[left_idx], swapped[right_idx] = np.array(points_2d[right_idx], copy=True), np.array(
            points_2d[left_idx], copy=True
        )
    return swapped


def swap_left_right_keypoint_values(values: np.ndarray) -> np.ndarray:
    """Swap left/right entries for any keypoint-indexed array.

    The first axis must match the COCO17 keypoint order. This is used for
    both 2D coordinates and per-keypoint scalar arrays such as variances.
    """

    swapped = np.array(values, copy=True)
    for left_name, right_name in LEFT_RIGHT_SWAP_PAIRS:
        left_idx = KP_INDEX[left_name]
        right_idx = KP_INDEX[right_name]
        swapped[left_idx], swapped[right_idx] = np.array(values[right_idx], copy=True), np.array(
            values[left_idx], copy=True
        )
    return swapped


def choose_ekf_prediction_gate_measurements(
    frame_keypoints: np.ndarray,
    frame_variances: np.ndarray,
    predicted_uv: np.ndarray,
    keypoint_indices: np.ndarray,
    *,
    improvement_ratio: float,
    min_gain_px: float,
    min_valid_keypoints: int = DEFAULT_EKF_PREDICTION_GATE_MIN_VALID_KEYPOINTS,
    activation_error_threshold_px: float = DEFAULT_EKF_PREDICTION_GATE_ERROR_THRESHOLD_PX,
    activation_error_delta_threshold_px: float = DEFAULT_EKF_PREDICTION_GATE_ERROR_DELTA_THRESHOLD_PX,
    previous_nominal_rms_px: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Choose between raw and globally swapped 2D measurements for one camera.

    The decision is made against the current EKF prediction in image space.
    We keep the existing acceptance rule semantics:
    - swapped RMS reprojection error must beat `improvement_ratio * nominal`
    - and the absolute RMS gain must be at least `min_gain_px`
    """

    keypoint_indices = np.asarray(keypoint_indices, dtype=int).reshape(-1)
    predicted_uv = np.asarray(predicted_uv, dtype=float).reshape(-1, 2)
    raw_points = np.asarray(frame_keypoints, dtype=float)
    raw_variances = np.asarray(frame_variances, dtype=float)
    swapped_points = swap_left_right_keypoints(raw_points)
    swapped_variances = swap_left_right_keypoint_values(raw_variances)

    raw_selected_points = raw_points[keypoint_indices]
    raw_selected_variances = raw_variances[keypoint_indices]
    swapped_selected_points = swapped_points[keypoint_indices]
    swapped_selected_variances = swapped_variances[keypoint_indices]

    finite_prediction = np.all(np.isfinite(predicted_uv), axis=1)
    raw_valid = (
        finite_prediction & np.all(np.isfinite(raw_selected_points), axis=1) & np.isfinite(raw_selected_variances)
    )
    raw_valid &= raw_selected_variances < np.inf
    swapped_valid = (
        finite_prediction
        & np.all(np.isfinite(swapped_selected_points), axis=1)
        & np.isfinite(swapped_selected_variances)
        & (swapped_selected_variances < np.inf)
    )
    comparable = raw_valid & swapped_valid

    diagnostics: dict[str, object] = {
        "used_swapped": False,
        "decision": "raw",
        "n_comparable_keypoints": int(np.sum(comparable)),
        "n_raw_valid_keypoints": int(np.sum(raw_valid)),
        "n_swapped_valid_keypoints": int(np.sum(swapped_valid)),
        "nominal_rms_px": float("nan"),
        "swapped_rms_px": float("nan"),
        "previous_nominal_rms_px": (
            float(previous_nominal_rms_px) if previous_nominal_rms_px is not None else float("nan")
        ),
        "nominal_rms_delta_px": float("nan"),
        "activation_error_threshold_px": float(activation_error_threshold_px),
        "activation_error_delta_threshold_px": float(activation_error_delta_threshold_px),
    }

    if int(np.sum(comparable)) >= int(max(1, min_valid_keypoints)):
        raw_errors = np.linalg.norm(raw_selected_points[comparable] - predicted_uv[comparable], axis=1)
        swapped_errors = np.linalg.norm(swapped_selected_points[comparable] - predicted_uv[comparable], axis=1)
        nominal_rms_px = float(np.sqrt(np.mean(raw_errors**2)))
        swapped_rms_px = float(np.sqrt(np.mean(swapped_errors**2)))
        diagnostics["nominal_rms_px"] = nominal_rms_px
        diagnostics["swapped_rms_px"] = swapped_rms_px
        if previous_nominal_rms_px is not None and np.isfinite(previous_nominal_rms_px):
            diagnostics["nominal_rms_delta_px"] = nominal_rms_px - float(previous_nominal_rms_px)
        if nominal_rms_px < float(activation_error_threshold_px):
            diagnostics["decision"] = "raw_below_error_threshold"
            return raw_valid, raw_selected_points, raw_selected_variances, diagnostics
        if previous_nominal_rms_px is None or not np.isfinite(previous_nominal_rms_px):
            diagnostics["decision"] = "raw_no_previous_error"
            return raw_valid, raw_selected_points, raw_selected_variances, diagnostics
        if diagnostics["nominal_rms_delta_px"] < float(activation_error_delta_threshold_px):
            diagnostics["decision"] = "raw_below_error_delta"
            return raw_valid, raw_selected_points, raw_selected_variances, diagnostics
        use_swapped = (
            np.isfinite(nominal_rms_px)
            and np.isfinite(swapped_rms_px)
            and (swapped_rms_px < float(improvement_ratio) * nominal_rms_px)
            and ((nominal_rms_px - swapped_rms_px) >= float(min_gain_px))
        )
        if use_swapped:
            diagnostics["used_swapped"] = True
            diagnostics["decision"] = "swapped"
            final_mask = swapped_valid
            return final_mask, swapped_selected_points, swapped_selected_variances, diagnostics

    if int(np.sum(comparable)) < int(max(1, min_valid_keypoints)):
        diagnostics["decision"] = "raw_insufficient_support"
    return raw_valid, raw_selected_points, raw_selected_variances, diagnostics


def build_temporal_reference_points(pose_data: PoseData) -> tuple[np.ndarray, np.ndarray]:
    """Construit, pour chaque camera-frame-keypoint, une prediction temporelle 2D.

    La prediction privilegie une interpolation entre la vue valide precedente et
    suivante. A defaut, une extrapolation vitesse constante unilaterale est
    utilisee si deux observations sont disponibles du meme cote.
    """
    n_cams, n_frames, n_keypoints = pose_data.keypoints.shape[:3]
    frame_numbers = np.asarray(pose_data.frames, dtype=float)
    references = np.full((n_cams, n_frames, n_keypoints, 2), np.nan, dtype=float)
    support_counts = np.zeros((n_cams, n_frames, n_keypoints), dtype=int)

    for cam_idx in range(n_cams):
        camera_points = pose_data.keypoints[cam_idx]
        camera_scores = pose_data.scores[cam_idx]
        for kp_idx in range(n_keypoints):
            valid_mask = np.all(np.isfinite(camera_points[:, kp_idx]), axis=1) & (camera_scores[:, kp_idx] > 0)
            valid_indices = np.flatnonzero(valid_mask)
            if valid_indices.size < 2:
                continue
            for frame_idx in range(n_frames):
                previous = valid_indices[valid_indices < frame_idx]
                following = valid_indices[valid_indices > frame_idx]
                prediction = None

                if previous.size >= 1 and following.size >= 1:
                    prev_idx = int(previous[-1])
                    next_idx = int(following[0])
                    dt = frame_numbers[next_idx] - frame_numbers[prev_idx]
                    if dt > 0:
                        alpha = (frame_numbers[frame_idx] - frame_numbers[prev_idx]) / dt
                        prediction = (1.0 - alpha) * camera_points[prev_idx, kp_idx] + alpha * camera_points[
                            next_idx, kp_idx
                        ]
                elif previous.size >= 2:
                    prev_idx = int(previous[-1])
                    prev2_idx = int(previous[-2])
                    dt = frame_numbers[prev_idx] - frame_numbers[prev2_idx]
                    if dt > 0:
                        velocity = (camera_points[prev_idx, kp_idx] - camera_points[prev2_idx, kp_idx]) / dt
                        prediction = camera_points[prev_idx, kp_idx] + velocity * (
                            frame_numbers[frame_idx] - frame_numbers[prev_idx]
                        )
                elif following.size >= 2:
                    next_idx = int(following[0])
                    next2_idx = int(following[1])
                    dt = frame_numbers[next2_idx] - frame_numbers[next_idx]
                    if dt > 0:
                        velocity = (camera_points[next2_idx, kp_idx] - camera_points[next_idx, kp_idx]) / dt
                        prediction = camera_points[next_idx, kp_idx] - velocity * (
                            frame_numbers[next_idx] - frame_numbers[frame_idx]
                        )

                if prediction is not None and np.all(np.isfinite(prediction)):
                    references[cam_idx, frame_idx, kp_idx] = prediction
                    support_counts[cam_idx, frame_idx, kp_idx] = 2

    return references, support_counts


def camera_center_world(calibration: CameraCalibration) -> np.ndarray:
    """Retourne le centre camera dans le repere monde."""
    return -(calibration.R.T @ calibration.tvec.reshape(3))


def build_flip_epipolar_pair_weights(
    calibrations: list[CameraCalibration],
    exponent: float = DEFAULT_FLIP_EPIPOLAR_PAIR_WEIGHT_EXPONENT,
) -> dict[tuple[int, int], float]:
    """Construit des poids de paires favorisant les baselines informatives."""
    centers = np.asarray([camera_center_world(calibration) for calibration in calibrations], dtype=float)
    weights: dict[tuple[int, int], float] = {}
    baselines = []
    for i_cam in range(len(calibrations)):
        for j_cam in range(len(calibrations)):
            if i_cam == j_cam:
                continue
            baseline = float(np.linalg.norm(centers[i_cam] - centers[j_cam]))
            baselines.append(baseline)
            weights[(i_cam, j_cam)] = baseline
    if not baselines:
        return weights
    baseline_scale = max(float(np.median(np.asarray(baselines, dtype=float))), 1e-9)
    exponent = float(max(exponent, 0.0))
    for key, baseline in list(weights.items()):
        normalized = max(baseline / baseline_scale, 1e-6)
        weights[key] = float(normalized**exponent)
    return weights


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Calcule une mediane ponderee robuste."""
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan
    valid_values = values[valid]
    valid_weights = weights[valid]
    order = np.argsort(valid_values)
    sorted_values = valid_values[order]
    sorted_weights = valid_weights[order]
    cumulative = np.cumsum(sorted_weights)
    threshold = 0.5 * cumulative[-1]
    idx = int(np.searchsorted(cumulative, threshold, side="left"))
    return float(sorted_values[min(idx, sorted_values.size - 1)])


def build_fundamental_matrix_array(
    calibrations: list[CameraCalibration],
) -> np.ndarray:
    """Empile les matrices fondamentales dans un tenseur `(cam_i, cam_j, 3, 3)`."""
    n_cams = len(calibrations)
    matrices = np.full((n_cams, n_cams, 3, 3), np.nan, dtype=float)
    for i_cam in range(n_cams):
        for j_cam in range(n_cams):
            if i_cam == j_cam:
                continue
            matrices[i_cam, j_cam] = fundamental_matrix(calibrations[i_cam], calibrations[j_cam])
    return matrices


def build_flip_epipolar_pair_weight_array(
    calibrations: list[CameraCalibration],
    exponent: float = DEFAULT_FLIP_EPIPOLAR_PAIR_WEIGHT_EXPONENT,
) -> np.ndarray:
    """Version tensorielle des poids de paires pour le diagnostic flip épipolaire."""
    weight_dict = build_flip_epipolar_pair_weights(calibrations, exponent=exponent)
    n_cams = len(calibrations)
    weights = np.zeros((n_cams, n_cams), dtype=float)
    for (i_cam, j_cam), value in weight_dict.items():
        weights[i_cam, j_cam] = float(value)
    return weights


def smooth_camera_time_series(values: np.ndarray, window: int = DEFAULT_FLIP_EPIPOLAR_SMOOTH_WINDOW) -> np.ndarray:
    """Lisse chaque serie camera dans le temps par moyenne locale robuste aux NaN."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("Expected a 2D array shaped (n_cams, n_frames).")
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window <= 1:
        return np.array(values, copy=True)
    half_window = window // 2
    smoothed = np.array(values, copy=True)
    for cam_idx in range(values.shape[0]):
        for frame_idx in range(values.shape[1]):
            start = max(0, frame_idx - half_window)
            stop = min(values.shape[1], frame_idx + half_window + 1)
            local_values = values[cam_idx, start:stop]
            finite_values = local_values[np.isfinite(local_values)]
            if finite_values.size == 0:
                smoothed[cam_idx, frame_idx] = np.nan
            else:
                smoothed[cam_idx, frame_idx] = float(np.mean(finite_values))
    return smoothed


def viterbi_flip_state_path(
    nominal_costs: np.ndarray,
    swapped_costs: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    transition_cost: float,
) -> np.ndarray:
    """Decode a temporally consistent normal/flipped path for one camera.

    State 0 corresponds to the nominal left/right assignment, while state 1
    corresponds to the swapped hypothesis. The decoder combines the per-frame
    costs already computed by the flip detector with a constant transition
    penalty so isolated local positives do not fragment the sequence.
    """

    nominal_costs = np.asarray(nominal_costs, dtype=float)
    swapped_costs = np.asarray(swapped_costs, dtype=float)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if nominal_costs.ndim != 1 or swapped_costs.ndim != 1 or candidate_mask.ndim != 1:
        raise ValueError("Viterbi flip decoding expects 1D arrays per camera.")
    if nominal_costs.shape != swapped_costs.shape or nominal_costs.shape != candidate_mask.shape:
        raise ValueError("Nominal, swapped, and candidate arrays must share the same shape.")

    n_frames = nominal_costs.shape[0]
    if n_frames == 0:
        return np.zeros(0, dtype=bool)

    transition_cost = max(float(transition_cost), 1e-6)
    unavailable_flip_penalty = transition_cost
    start_flipped_penalty = transition_cost
    end_flipped_penalty = transition_cost

    emissions = np.full((2, n_frames), np.inf, dtype=float)
    finite_nominal = np.isfinite(nominal_costs)
    finite_swapped = np.isfinite(swapped_costs)

    flipped_available = candidate_mask & finite_swapped
    comparable = finite_nominal & flipped_available
    baseline = np.full(n_frames, np.nan, dtype=float)
    baseline[comparable] = np.minimum(nominal_costs[comparable], swapped_costs[comparable])

    emissions[0, finite_nominal] = 0.0
    emissions[1, flipped_available] = unavailable_flip_penalty
    emissions[0, comparable] = nominal_costs[comparable] - baseline[comparable]
    emissions[1, comparable] = swapped_costs[comparable] - baseline[comparable]

    fallback_flipped = finite_nominal & ~flipped_available
    emissions[1, fallback_flipped] = unavailable_flip_penalty

    missing_frames = ~finite_nominal & ~finite_swapped
    emissions[:, missing_frames] = 0.0

    cumulative_cost = np.full((2, n_frames), np.inf, dtype=float)
    backpointers = np.zeros((2, n_frames), dtype=np.int8)
    cumulative_cost[0, 0] = emissions[0, 0]
    cumulative_cost[1, 0] = emissions[1, 0] + start_flipped_penalty

    transition_matrix = np.array([[0.0, transition_cost], [transition_cost, 0.0]], dtype=float)
    for frame_idx in range(1, n_frames):
        for state_idx in range(2):
            candidate_costs = cumulative_cost[:, frame_idx - 1] + transition_matrix[:, state_idx]
            best_prev_state = int(np.argmin(candidate_costs))
            cumulative_cost[state_idx, frame_idx] = emissions[state_idx, frame_idx] + candidate_costs[best_prev_state]
            backpointers[state_idx, frame_idx] = best_prev_state

    final_state = int(np.argmin(cumulative_cost[:, -1] + np.array([0.0, end_flipped_penalty], dtype=float)))
    states = np.zeros(n_frames, dtype=np.int8)
    states[-1] = final_state
    for frame_idx in range(n_frames - 1, 0, -1):
        states[frame_idx - 1] = backpointers[states[frame_idx], frame_idx]
    return states.astype(bool)


def sampson_error_pixels_vectorized(
    candidate_points: np.ndarray,
    other_points: np.ndarray,
    fundamental_matrices: np.ndarray,
) -> np.ndarray:
    """Calcule l'erreur de Sampson pour toutes les paires `(autre camera, keypoint)`."""
    candidate_points = np.asarray(candidate_points, dtype=float)
    other_points = np.asarray(other_points, dtype=float)
    fundamental_matrices = np.asarray(fundamental_matrices, dtype=float)
    if candidate_points.ndim != 2 or candidate_points.shape[1] != 2:
        raise ValueError("candidate_points must have shape (n_keypoints, 2).")
    if other_points.ndim != 3 or other_points.shape[2] != 2:
        raise ValueError("other_points must have shape (n_other_cams, n_keypoints, 2).")
    if fundamental_matrices.ndim != 3 or fundamental_matrices.shape[1:] != (3, 3):
        raise ValueError("fundamental_matrices must have shape (n_other_cams, 3, 3).")

    n_other_cams, n_keypoints = other_points.shape[:2]
    candidate_h = np.concatenate((candidate_points, np.ones((n_keypoints, 1), dtype=float)), axis=1)
    other_h = np.concatenate((other_points, np.ones((n_other_cams, n_keypoints, 1), dtype=float)), axis=2)
    Fx1 = np.einsum("oab,kb->oka", fundamental_matrices, candidate_h, optimize=True)
    Ftx2 = np.einsum("oba,okb->oka", fundamental_matrices, other_h, optimize=True)
    numerator = np.abs(np.einsum("okb,okb->ok", other_h, Fx1, optimize=True))
    denominator = np.sqrt(Fx1[..., 0] ** 2 + Fx1[..., 1] ** 2 + Ftx2[..., 0] ** 2 + Ftx2[..., 1] ** 2)
    errors = np.full((n_other_cams, n_keypoints), np.nan, dtype=float)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 1e-12)
    errors[valid] = numerator[valid] / denominator[valid]
    return errors


def symmetric_epipolar_distance_vectorized(
    candidate_points: np.ndarray,
    other_points: np.ndarray,
    fundamental_matrices: np.ndarray,
) -> np.ndarray:
    """Compute the symmetric epipolar distance for all `(camera, keypoint)` pairs."""

    candidate_points = np.asarray(candidate_points, dtype=float)
    other_points = np.asarray(other_points, dtype=float)
    fundamental_matrices = np.asarray(fundamental_matrices, dtype=float)
    if candidate_points.ndim != 2 or candidate_points.shape[1] != 2:
        raise ValueError("candidate_points must have shape (n_keypoints, 2).")
    if other_points.ndim != 3 or other_points.shape[2] != 2:
        raise ValueError("other_points must have shape (n_other_cams, n_keypoints, 2).")
    if fundamental_matrices.ndim != 3 or fundamental_matrices.shape[1:] != (3, 3):
        raise ValueError("fundamental_matrices must have shape (n_other_cams, 3, 3).")

    n_other_cams, n_keypoints = other_points.shape[:2]
    candidate_h = np.concatenate((candidate_points, np.ones((n_keypoints, 1), dtype=float)), axis=1)
    other_h = np.concatenate((other_points, np.ones((n_other_cams, n_keypoints, 1), dtype=float)), axis=2)
    Fx1 = np.einsum("oab,kb->oka", fundamental_matrices, candidate_h, optimize=True)
    Ftx2 = np.einsum("oba,okb->oka", fundamental_matrices, other_h, optimize=True)
    numerator = np.abs(np.einsum("okb,okb->ok", other_h, Fx1, optimize=True))
    denom1 = np.sqrt(Fx1[..., 0] ** 2 + Fx1[..., 1] ** 2)
    denom2 = np.sqrt(Ftx2[..., 0] ** 2 + Ftx2[..., 1] ** 2)
    errors = np.full((n_other_cams, n_keypoints), np.nan, dtype=float)
    valid = np.isfinite(numerator) & np.isfinite(denom1) & np.isfinite(denom2) & (denom1 > 1e-12) & (denom2 > 1e-12)
    errors[valid] = numerator[valid] / denom1[valid] + numerator[valid] / denom2[valid]
    return errors


def compute_camera_epipolar_costs_vectorized(
    camera_idx: int,
    candidate_points: np.ndarray,
    raw_2d_frame: np.ndarray,
    raw_scores_frame: np.ndarray,
    fundamental_matrices_array: np.ndarray,
    *,
    pair_weights_array: np.ndarray | None = None,
    keypoint_weights: np.ndarray | None = None,
    min_other_cameras: int = DEFAULT_FLIP_MIN_OTHER_CAMERAS,
    distance_mode: str = "sampson",
) -> tuple[float, float]:
    """Retourne les coûts épipolaires pondéré et legacy pour une caméra candidate."""
    n_cams = raw_2d_frame.shape[0]
    other_indices = [idx for idx in range(n_cams) if idx != camera_idx]
    if not other_indices:
        return np.nan, np.nan

    candidate_points = np.asarray(candidate_points, dtype=float)
    candidate_valid = np.all(np.isfinite(candidate_points), axis=1) & (raw_scores_frame[camera_idx] > 0)
    if not np.any(candidate_valid):
        return np.nan, np.nan

    other_points = np.asarray(raw_2d_frame[other_indices], dtype=float)
    other_scores = np.asarray(raw_scores_frame[other_indices], dtype=float)
    other_valid = np.all(np.isfinite(other_points), axis=2) & (other_scores > 0)
    valid_mask = other_valid & candidate_valid[np.newaxis, :]
    informative_camera_mask = np.any(valid_mask, axis=1)
    if int(np.count_nonzero(informative_camera_mask)) < max(1, int(min_other_cameras)):
        return np.nan, np.nan

    fundamental_blocks = fundamental_matrices_array[camera_idx, other_indices]
    if distance_mode == "sampson":
        errors = sampson_error_pixels_vectorized(candidate_points, other_points, fundamental_blocks)
    elif distance_mode == "symmetric":
        errors = symmetric_epipolar_distance_vectorized(candidate_points, other_points, fundamental_blocks)
    else:
        raise ValueError(f"Unknown epipolar distance mode: {distance_mode}")
    valid_mask &= np.isfinite(errors)
    if not np.any(valid_mask):
        return np.nan, np.nan

    if keypoint_weights is None:
        kp_weights = np.ones(candidate_points.shape[0], dtype=float)
    else:
        kp_weights = np.asarray(keypoint_weights, dtype=float)
    if pair_weights_array is None:
        pair_weights = np.ones(len(other_indices), dtype=float)
    else:
        pair_weights = np.asarray(pair_weights_array[camera_idx, other_indices], dtype=float)

    confidence_weights = np.minimum(raw_scores_frame[camera_idx][np.newaxis, :], other_scores)
    combined_weights = pair_weights[:, np.newaxis] * kp_weights[np.newaxis, :] * np.maximum(confidence_weights, 1e-6)
    valid_weights = combined_weights[valid_mask]
    weighted_cost = weighted_median(errors[valid_mask], np.maximum(valid_weights, 1e-9))
    legacy_cost = float(np.median(errors[valid_mask])) if np.any(valid_mask) else np.nan
    return weighted_cost, legacy_cost


def compute_camera_temporal_cost(
    camera_idx: int,
    frame_idx: int,
    candidate_points: np.ndarray,
    raw_scores_frame: np.ndarray,
    temporal_references: np.ndarray,
    temporal_support_counts: np.ndarray,
    min_valid_keypoints: int = DEFAULT_FLIP_TEMPORAL_MIN_VALID_KEYPOINTS,
) -> float:
    """Mesure la coherence temporelle locale d'une hypothese 2D pour une camera."""
    reference_points = temporal_references[camera_idx, frame_idx]
    support = temporal_support_counts[camera_idx, frame_idx]
    valid = (
        np.all(np.isfinite(candidate_points), axis=1)
        & np.all(np.isfinite(reference_points), axis=1)
        & (raw_scores_frame[camera_idx] > 0)
        & (support > 0)
    )
    if int(np.count_nonzero(valid)) < max(1, int(min_valid_keypoints)):
        return np.nan
    errors = np.linalg.norm(candidate_points[valid] - reference_points[valid], axis=1)
    if errors.size == 0:
        return np.nan
    return float(np.median(errors))


def combine_flip_costs(
    geometric_costs: np.ndarray,
    temporal_costs: np.ndarray,
    *,
    geometry_tau_px: float,
    temporal_tau_px: float,
    temporal_weight: float,
) -> np.ndarray | float:
    """Combine un cout geometrique et un cout temporel en pixels effectifs."""
    geometry_tau_px = max(float(geometry_tau_px), 1e-6)
    temporal_tau_px = max(float(temporal_tau_px), 1e-6)
    temporal_weight = float(np.clip(temporal_weight, 0.0, 1.0))
    geometric = np.asarray(geometric_costs, dtype=float)
    temporal = np.asarray(temporal_costs, dtype=float)
    combined = np.array(geometric, copy=True, dtype=float)
    if temporal_weight <= 0.0:
        return float(combined) if combined.ndim == 0 else combined
    valid = np.isfinite(geometric) & np.isfinite(temporal)
    combined = np.where(
        valid,
        geometry_tau_px
        * ((1.0 - temporal_weight) * (geometric / geometry_tau_px) + temporal_weight * (temporal / temporal_tau_px)),
        geometric,
    )
    return float(combined) if combined.ndim == 0 else combined


def compute_camera_epipolar_cost(
    camera_idx: int,
    candidate_points: np.ndarray,
    raw_2d_frame: np.ndarray,
    raw_scores_frame: np.ndarray,
    fundamental_matrices: dict[tuple[int, int], np.ndarray],
    *,
    pair_weights: dict[tuple[int, int], float] | None = None,
    keypoint_weights: np.ndarray | None = None,
    min_other_cameras: int = DEFAULT_FLIP_MIN_OTHER_CAMERAS,
) -> float:
    """Mesure la coherence epipolaire d'une camera candidate avec les autres vues."""
    errors = []
    weights = []
    valid_candidate = np.all(np.isfinite(candidate_points), axis=1) & (raw_scores_frame[camera_idx] > 0)
    informative_other_cameras: set[int] = set()
    for kp_idx in range(candidate_points.shape[0]):
        if not valid_candidate[kp_idx]:
            continue
        point_i = candidate_points[kp_idx]
        kp_weight = 1.0 if keypoint_weights is None else float(keypoint_weights[kp_idx])
        if kp_weight <= 0:
            continue
        for other_idx in range(raw_2d_frame.shape[0]):
            if other_idx == camera_idx:
                continue
            if raw_scores_frame[other_idx, kp_idx] <= 0:
                continue
            point_j = raw_2d_frame[other_idx, kp_idx]
            if not np.all(np.isfinite(point_j)):
                continue
            err = sampson_error_pixels(point_i, point_j, fundamental_matrices[(camera_idx, other_idx)])
            if np.isfinite(err):
                errors.append(err)
                informative_other_cameras.add(other_idx)
                pair_weight = 1.0 if pair_weights is None else float(pair_weights.get((camera_idx, other_idx), 1.0))
                confidence_weight = float(
                    min(raw_scores_frame[camera_idx, kp_idx], raw_scores_frame[other_idx, kp_idx])
                )
                weights.append(max(pair_weight * kp_weight * max(confidence_weight, 1e-6), 1e-9))
    if len(informative_other_cameras) < max(1, int(min_other_cameras)):
        return np.nan
    if not errors:
        return np.nan
    return weighted_median(np.asarray(errors, dtype=float), np.asarray(weights, dtype=float))


def compute_camera_epipolar_cost_legacy(
    camera_idx: int,
    candidate_points: np.ndarray,
    raw_2d_frame: np.ndarray,
    raw_scores_frame: np.ndarray,
    fundamental_matrices: dict[tuple[int, int], np.ndarray],
) -> float:
    """Version non pondérée, conservée comme garde-fou pour ne pas sous-détecter."""
    errors = []
    valid_candidate = np.all(np.isfinite(candidate_points), axis=1) & (raw_scores_frame[camera_idx] > 0)
    for kp_idx in range(candidate_points.shape[0]):
        if not valid_candidate[kp_idx]:
            continue
        point_i = candidate_points[kp_idx]
        for other_idx in range(raw_2d_frame.shape[0]):
            if other_idx == camera_idx:
                continue
            if raw_scores_frame[other_idx, kp_idx] <= 0:
                continue
            point_j = raw_2d_frame[other_idx, kp_idx]
            if not np.all(np.isfinite(point_j)):
                continue
            err = sampson_error_pixels(point_i, point_j, fundamental_matrices[(camera_idx, other_idx)])
            if np.isfinite(err):
                errors.append(err)
    if not errors:
        return np.nan
    return float(np.median(np.asarray(errors, dtype=float)))


def compute_camera_triangulation_cost(
    camera_idx: int,
    candidate_points: np.ndarray,
    raw_2d_frame: np.ndarray,
    raw_scores_frame: np.ndarray,
    ordered_calibrations: list[CameraCalibration],
    min_other_cameras: int = 2,
    precomputed_reprojected_points: np.ndarray | None = None,
    triangulation_variant: str = "raw",
    error_threshold_px: float = DEFAULT_REPROJECTION_THRESHOLD_PX,
) -> float:
    """Mesure la coherence d'une camera candidate via triangulation des autres vues.

    Pour chaque keypoint, on triangule en excluant la camera testee, puis on
    compare la reprojection dans cette camera pour l'hypothese nominale et la
    version swappee.
    """
    triangulation_variant = canonical_triangulation_method(triangulation_variant)
    if triangulation_variant not in {"once", "greedy", "exhaustive"}:
        raise ValueError(f"Unknown triangulation variant: {triangulation_variant}")

    if (
        triangulation_variant == "once"
        and precomputed_reprojected_points is not None
        and precomputed_reprojected_points.ndim == 3
        and camera_idx < precomputed_reprojected_points.shape[0]
    ):
        references = np.asarray(precomputed_reprojected_points[camera_idx], dtype=float)
        valid = (
            np.all(np.isfinite(candidate_points), axis=1)
            & np.all(np.isfinite(references), axis=1)
            & (raw_scores_frame[camera_idx] > 0)
        )
        if not np.any(valid):
            return np.nan
        return float(np.median(np.linalg.norm(candidate_points[valid] - references[valid], axis=1)))

    errors = []
    calibration = ordered_calibrations[camera_idx]
    valid_candidate = np.all(np.isfinite(candidate_points), axis=1) & (raw_scores_frame[camera_idx] > 0)
    for kp_idx in range(candidate_points.shape[0]):
        if not valid_candidate[kp_idx]:
            continue
        projections = []
        observations = []
        confidences = []
        supporting_calibrations = []
        for other_idx in range(raw_2d_frame.shape[0]):
            if other_idx == camera_idx:
                continue
            if raw_scores_frame[other_idx, kp_idx] <= 0:
                continue
            point_other = raw_2d_frame[other_idx, kp_idx]
            if not np.all(np.isfinite(point_other)):
                continue
            projections.append(ordered_calibrations[other_idx].P)
            observations.append(point_other)
            confidences.append(raw_scores_frame[other_idx, kp_idx])
            supporting_calibrations.append(ordered_calibrations[other_idx])
        if len(observations) < max(2, int(min_other_cameras)):
            continue
        observations_array = np.asarray(observations, dtype=float)
        confidences_array = np.asarray(confidences, dtype=float)
        if triangulation_variant == "once":
            point_3d = weighted_triangulation(
                projections,
                observations_array,
                confidences_array,
            )
        else:
            triangulate_fn = (
                robust_triangulation_from_best_cameras
                if triangulation_variant == "exhaustive"
                else greedy_triangulation_from_best_cameras
            )
            point_3d, _mean_error, _per_view, _coherence, _excluded = triangulate_fn(
                projections,
                observations_array,
                confidences_array,
                supporting_calibrations,
                error_threshold_px=float(error_threshold_px),
                min_cameras_for_triangulation=max(2, int(min_other_cameras)),
            )
        if not np.all(np.isfinite(point_3d)):
            continue
        reprojected = calibration.project_point(point_3d)
        if not np.all(np.isfinite(reprojected)):
            continue
        errors.append(float(np.linalg.norm(candidate_points[kp_idx] - reprojected)))
    if not errors:
        return np.nan
    return float(np.median(np.asarray(errors, dtype=float)))


def detect_left_right_flip_diagnostics(
    pose_data: PoseData,
    calibrations: dict[str, CameraCalibration],
    method: str = "epipolar",
    improvement_ratio: float = DEFAULT_FLIP_IMPROVEMENT_RATIO,
    min_gain_px: float = DEFAULT_FLIP_MIN_GAIN_PX,
    min_other_cameras: int = DEFAULT_FLIP_MIN_OTHER_CAMERAS,
    restrict_to_outliers: bool = DEFAULT_FLIP_RESTRICT_TO_OUTLIERS,
    outlier_percentile: float = DEFAULT_FLIP_OUTLIER_PERCENTILE,
    outlier_floor_px: float = DEFAULT_FLIP_OUTLIER_FLOOR_PX,
    geometry_tau_px: float = DEFAULT_EPIPOLAR_THRESHOLD_PX,
    temporal_weight: float = DEFAULT_FLIP_TEMPORAL_WEIGHT,
    temporal_tau_px: float = DEFAULT_FLIP_TEMPORAL_TAU_PX,
    temporal_min_valid_keypoints: int = DEFAULT_FLIP_TEMPORAL_MIN_VALID_KEYPOINTS,
) -> tuple[np.ndarray, dict[str, object], dict[str, np.ndarray]]:
    """Diagnostique les frames ou un swap gauche/droite semblerait plus coherent.

    Le diagnostic est calcule camera par camera. Une frame est consideree
    suspecte des qu'au moins une camera beneficie nettement d'un swap global
    gauche/droite selon le cout median choisi (`epipolar` ou `triangulation`).
    """
    n_cams, n_frames = pose_data.keypoints.shape[:2]
    ordered_calibrations = [calibrations[name] for name in pose_data.camera_names]
    fundamental_matrices = None
    pair_weights = None
    keypoint_weights = np.asarray([FLIP_PROXIMAL_KEYPOINT_WEIGHTS.get(name, 1.0) for name in COCO17], dtype=float)
    local_epipolar_methods = {"epipolar", "epipolar_fast"}
    viterbi_epipolar_methods = {"epipolar_viterbi", "epipolar_fast_viterbi"}
    epipolar_family = method in local_epipolar_methods | viterbi_epipolar_methods
    use_viterbi_decoding = method in viterbi_epipolar_methods
    triangulation_family = method in {
        "triangulation",
        "triangulation_once",
        "triangulation_raw",
        "triangulation_greedy",
        "triangulation_exhaustive",
    }
    triangulation_variant = (
        "once"
        if method in {"triangulation", "triangulation_once", "triangulation_raw"}
        else ("greedy" if method == "triangulation_greedy" else "exhaustive")
    )
    epipolar_distance_mode = (
        "sampson"
        if method in {"epipolar", "epipolar_viterbi"}
        else ("symmetric" if method in {"epipolar_fast", "epipolar_fast_viterbi"} else "reprojection")
    )
    smoothing_window = DEFAULT_FLIP_EPIPOLAR_SMOOTH_WINDOW if epipolar_family else 1
    if epipolar_family:
        fundamental_matrices = build_fundamental_matrix_array(ordered_calibrations)
        pair_weights = build_flip_epipolar_pair_weight_array(ordered_calibrations)
    elif not triangulation_family:
        raise ValueError(f"Unknown flip diagnostic method: {method}")
    nominal_costs = np.full((n_cams, n_frames), np.nan, dtype=float)
    nominal_legacy_costs = np.full((n_cams, n_frames), np.nan, dtype=float)
    nominal_temporal_costs = np.full((n_cams, n_frames), np.nan, dtype=float)
    swapped_costs = np.full((n_cams, n_frames), np.nan, dtype=float)
    swapped_legacy_costs = np.full((n_cams, n_frames), np.nan, dtype=float)
    swapped_temporal_costs = np.full((n_cams, n_frames), np.nan, dtype=float)
    temporal_references, temporal_support_counts = build_temporal_reference_points(pose_data)

    for frame_idx in range(n_frames):
        raw_points_frame = pose_data.keypoints[:, frame_idx]
        raw_scores_frame = pose_data.scores[:, frame_idx]
        triangulation_references = (
            triangulation_reference_from_other_views(
                raw_points_frame,
                raw_scores_frame,
                ordered_calibrations,
                min_other_cameras=min_other_cameras,
            )
            if triangulation_family and triangulation_variant == "raw"
            else None
        )
        for cam_idx in range(n_cams):
            if epipolar_family:
                nominal_costs[cam_idx, frame_idx], nominal_legacy_costs[cam_idx, frame_idx] = (
                    compute_camera_epipolar_costs_vectorized(
                        cam_idx,
                        raw_points_frame[cam_idx],
                        raw_points_frame,
                        raw_scores_frame,
                        fundamental_matrices,
                        pair_weights_array=pair_weights,
                        keypoint_weights=keypoint_weights,
                        min_other_cameras=min_other_cameras,
                        distance_mode=epipolar_distance_mode,
                    )
                )
            else:
                nominal_costs[cam_idx, frame_idx] = compute_camera_triangulation_cost(
                    cam_idx,
                    raw_points_frame[cam_idx],
                    raw_points_frame,
                    raw_scores_frame,
                    ordered_calibrations,
                    min_other_cameras=min_other_cameras,
                    precomputed_reprojected_points=triangulation_references,
                    triangulation_variant=triangulation_variant,
                    error_threshold_px=geometry_tau_px,
                )
            nominal_temporal_costs[cam_idx, frame_idx] = compute_camera_temporal_cost(
                cam_idx,
                frame_idx,
                raw_points_frame[cam_idx],
                raw_scores_frame,
                temporal_references,
                temporal_support_counts,
                min_valid_keypoints=temporal_min_valid_keypoints,
            )

    effective_nominal_costs = combine_flip_costs(
        nominal_costs,
        nominal_temporal_costs,
        geometry_tau_px=geometry_tau_px,
        temporal_tau_px=temporal_tau_px,
        temporal_weight=temporal_weight,
    )
    smoothed_nominal_costs = smooth_camera_time_series(effective_nominal_costs, window=smoothing_window)

    outlier_thresholds_by_camera = np.full(n_cams, float(outlier_floor_px), dtype=float)
    legacy_outlier_thresholds_by_camera = np.full(n_cams, float(outlier_floor_px), dtype=float)
    candidate_mask = np.isfinite(effective_nominal_costs)
    legacy_candidate_mask = np.zeros_like(candidate_mask, dtype=bool)
    if restrict_to_outliers:
        candidate_mask[:] = False
        for cam_idx in range(n_cams):
            valid_costs = effective_nominal_costs[cam_idx, np.isfinite(effective_nominal_costs[cam_idx])]
            if valid_costs.size == 0:
                continue
            threshold = max(float(outlier_floor_px), float(np.percentile(valid_costs, float(outlier_percentile))))
            outlier_thresholds_by_camera[cam_idx] = threshold
            raw_outliers = np.isfinite(effective_nominal_costs[cam_idx]) & (
                effective_nominal_costs[cam_idx] >= threshold
            )
            if epipolar_family:
                smooth_outliers = np.isfinite(smoothed_nominal_costs[cam_idx]) & (
                    smoothed_nominal_costs[cam_idx] >= threshold
                )
                candidate_mask[cam_idx] = raw_outliers | smooth_outliers
                legacy_valid_costs = nominal_legacy_costs[cam_idx, np.isfinite(nominal_legacy_costs[cam_idx])]
                if legacy_valid_costs.size > 0:
                    legacy_threshold = max(
                        float(outlier_floor_px),
                        float(np.percentile(legacy_valid_costs, float(outlier_percentile))),
                    )
                    legacy_outlier_thresholds_by_camera[cam_idx] = legacy_threshold
                    legacy_candidate_mask[cam_idx] = np.isfinite(nominal_legacy_costs[cam_idx]) & (
                        nominal_legacy_costs[cam_idx] >= legacy_threshold
                    )
                    candidate_mask[cam_idx] |= legacy_candidate_mask[cam_idx]
            else:
                candidate_mask[cam_idx] = raw_outliers
    elif epipolar_family:
        legacy_candidate_mask = np.isfinite(nominal_legacy_costs)

    for frame_idx in range(n_frames):
        raw_points_frame = pose_data.keypoints[:, frame_idx]
        raw_scores_frame = pose_data.scores[:, frame_idx]
        triangulation_references = (
            triangulation_reference_from_other_views(
                raw_points_frame,
                raw_scores_frame,
                ordered_calibrations,
                min_other_cameras=min_other_cameras,
            )
            if triangulation_family and triangulation_variant == "raw"
            else None
        )
        for cam_idx in range(n_cams):
            if not candidate_mask[cam_idx, frame_idx]:
                continue
            swapped_candidate = swap_left_right_keypoints(raw_points_frame[cam_idx])
            if epipolar_family:
                swapped_costs[cam_idx, frame_idx], swapped_legacy_costs[cam_idx, frame_idx] = (
                    compute_camera_epipolar_costs_vectorized(
                        cam_idx,
                        swapped_candidate,
                        raw_points_frame,
                        raw_scores_frame,
                        fundamental_matrices,
                        pair_weights_array=pair_weights,
                        keypoint_weights=keypoint_weights,
                        min_other_cameras=min_other_cameras,
                        distance_mode=epipolar_distance_mode,
                    )
                )
            else:
                swapped_costs[cam_idx, frame_idx] = compute_camera_triangulation_cost(
                    cam_idx,
                    swapped_candidate,
                    raw_points_frame,
                    raw_scores_frame,
                    ordered_calibrations,
                    min_other_cameras=min_other_cameras,
                    precomputed_reprojected_points=triangulation_references,
                    triangulation_variant=triangulation_variant,
                    error_threshold_px=geometry_tau_px,
                )
            swapped_temporal_costs[cam_idx, frame_idx] = compute_camera_temporal_cost(
                cam_idx,
                frame_idx,
                swapped_candidate,
                raw_scores_frame,
                temporal_references,
                temporal_support_counts,
                min_valid_keypoints=temporal_min_valid_keypoints,
            )
            swapped_cost = combine_flip_costs(
                swapped_costs[cam_idx, frame_idx],
                swapped_temporal_costs[cam_idx, frame_idx],
                geometry_tau_px=geometry_tau_px,
                temporal_tau_px=temporal_tau_px,
                temporal_weight=temporal_weight,
            )

    effective_swapped_costs = combine_flip_costs(
        swapped_costs,
        swapped_temporal_costs,
        geometry_tau_px=geometry_tau_px,
        temporal_tau_px=temporal_tau_px,
        temporal_weight=temporal_weight,
    )
    if epipolar_family:
        legacy_decision = (
            candidate_mask
            & np.isfinite(nominal_legacy_costs)
            & np.isfinite(swapped_legacy_costs)
            & (nominal_legacy_costs > 0.0)
            & (swapped_legacy_costs < float(improvement_ratio) * nominal_legacy_costs)
            & ((nominal_legacy_costs - swapped_legacy_costs) >= float(min_gain_px))
        )
    else:
        legacy_decision = np.zeros_like(candidate_mask, dtype=bool)
    gain_margin_px = effective_nominal_costs - effective_swapped_costs - float(min_gain_px)
    ratio_margin_px = float(improvement_ratio) * effective_nominal_costs - effective_swapped_costs
    decision_score = np.minimum(gain_margin_px, ratio_margin_px)
    decision_score[~candidate_mask] = np.nan
    smoothed_decision_score = smooth_camera_time_series(
        np.where(np.isfinite(decision_score), decision_score, 0.0),
        window=smoothing_window,
    )
    strong_positive_margin = 0.5 * float(min_gain_px) if epipolar_family else 0.0
    if epipolar_family and use_viterbi_decoding:
        viterbi_transition_cost = max(float(min_gain_px), 0.25 * float(geometry_tau_px))
        viterbi_state_mask = np.zeros_like(candidate_mask, dtype=bool)
        for cam_idx in range(n_cams):
            viterbi_state_mask[cam_idx] = viterbi_flip_state_path(
                effective_nominal_costs[cam_idx],
                effective_swapped_costs[cam_idx],
                candidate_mask[cam_idx],
                transition_cost=viterbi_transition_cost,
            )
        suspect_mask = (
            candidate_mask
            & viterbi_state_mask
            & np.isfinite(decision_score)
            & ((smoothed_decision_score > 0.0) | (decision_score >= strong_positive_margin))
        )
    elif epipolar_family:
        viterbi_transition_cost = 0.0
        viterbi_state_mask = np.zeros_like(candidate_mask, dtype=bool)
        suspect_mask = candidate_mask & np.isfinite(decision_score) & (decision_score > 0.0)
    else:
        viterbi_transition_cost = 0.0
        viterbi_state_mask = np.zeros_like(candidate_mask, dtype=bool)
        suspect_mask = (
            candidate_mask
            & np.isfinite(decision_score)
            & (((decision_score > 0.0) & (smoothed_decision_score > 0.0)) | (decision_score >= strong_positive_margin))
        )
    suspect_mask |= legacy_decision
    frames_with_any_suspect = np.any(suspect_mask, axis=0)
    temporal_support_mask = np.isfinite(nominal_temporal_costs)
    candidate_temporal_support_mask = candidate_mask & temporal_support_mask
    diagnostics = {
        "method": method,
        "distance_mode": epipolar_distance_mode,
        "n_frames_with_any_flip_suspect": int(frames_with_any_suspect.sum()),
        "n_camera_frame_flip_suspects": int(suspect_mask.sum()),
        "n_candidate_camera_frames_tested": int(np.count_nonzero(candidate_mask)),
        "improvement_ratio": float(improvement_ratio),
        "min_gain_px": float(min_gain_px),
        "min_other_cameras": int(min_other_cameras),
        "restrict_to_outliers": bool(restrict_to_outliers),
        "outlier_percentile": float(outlier_percentile),
        "outlier_floor_px": float(outlier_floor_px),
        "geometry_tau_px": float(geometry_tau_px),
        "temporal_tau_px": float(temporal_tau_px),
        "temporal_weight": float(temporal_weight),
        "temporal_min_valid_keypoints": int(temporal_min_valid_keypoints),
        "combination_method": "weighted_normalized_cost",
        "epipolar_pair_weighting": "baseline_confidence_weighted" if epipolar_family else "n/a",
        "epipolar_keypoint_weighting": "torso_proximal_priority" if epipolar_family else "n/a",
        "temporal_smoothing_window": int(smoothing_window),
        "temporal_decision_method": ("viterbi_two_state" if use_viterbi_decoding else "local_threshold"),
        "viterbi_transition_cost_px": float(viterbi_transition_cost),
        "n_camera_frame_temporal_support": int(np.count_nonzero(temporal_support_mask)),
        "n_candidate_camera_frames_with_temporal_support": int(np.count_nonzero(candidate_temporal_support_mask)),
        "outlier_thresholds_by_camera": {
            pose_data.camera_names[cam_idx]: float(outlier_thresholds_by_camera[cam_idx]) for cam_idx in range(n_cams)
        },
        "camera_frame_temporal_support": {
            pose_data.camera_names[cam_idx]: int(np.count_nonzero(temporal_support_mask[cam_idx]))
            for cam_idx in range(n_cams)
        },
        "frames_with_any_flip_suspect": pose_data.frames[frames_with_any_suspect].astype(int).tolist(),
        "camera_frame_flip_suspects": {
            pose_data.camera_names[cam_idx]: int(suspect_mask[cam_idx].sum()) for cam_idx in range(n_cams)
        },
        "camera_frame_flip_suspect_frames": {
            pose_data.camera_names[cam_idx]: pose_data.frames[suspect_mask[cam_idx]].astype(int).tolist()
            for cam_idx in range(n_cams)
        },
        "camera_frame_flip_candidates": {
            pose_data.camera_names[cam_idx]: int(candidate_mask[cam_idx].sum()) for cam_idx in range(n_cams)
        },
        "camera_frame_flip_legacy_candidates": {
            pose_data.camera_names[cam_idx]: int(legacy_candidate_mask[cam_idx].sum()) for cam_idx in range(n_cams)
        },
        "cost_summaries_px": {
            "legacy_nominal_median": (
                float(np.nanmedian(nominal_legacy_costs)) if np.isfinite(nominal_legacy_costs).any() else None
            ),
            "legacy_swapped_candidate_median": (
                float(np.nanmedian(swapped_legacy_costs[candidate_mask]))
                if np.isfinite(swapped_legacy_costs[candidate_mask]).any()
                else None
            ),
            "geometric_nominal_median": (
                float(np.nanmedian(nominal_costs)) if np.isfinite(nominal_costs).any() else None
            ),
            "temporal_nominal_median": (
                float(np.nanmedian(nominal_temporal_costs)) if np.isfinite(nominal_temporal_costs).any() else None
            ),
            "combined_nominal_median": (
                float(np.nanmedian(effective_nominal_costs)) if np.isfinite(effective_nominal_costs).any() else None
            ),
            "combined_nominal_smoothed_median": (
                float(np.nanmedian(smoothed_nominal_costs)) if np.isfinite(smoothed_nominal_costs).any() else None
            ),
            "combined_swapped_candidate_median": (
                float(np.nanmedian(effective_swapped_costs[candidate_mask]))
                if np.isfinite(effective_swapped_costs[candidate_mask]).any()
                else None
            ),
            "decision_score_median": (
                float(np.nanmedian(decision_score[candidate_mask]))
                if np.isfinite(decision_score[candidate_mask]).any()
                else None
            ),
        },
    }
    detail_arrays = {
        "nominal_geometric_costs": nominal_costs,
        "nominal_legacy_costs": nominal_legacy_costs,
        "nominal_temporal_costs": nominal_temporal_costs,
        "nominal_combined_costs": effective_nominal_costs,
        "nominal_combined_costs_smoothed": smoothed_nominal_costs,
        "swapped_geometric_costs": swapped_costs,
        "swapped_legacy_costs": swapped_legacy_costs,
        "swapped_temporal_costs": swapped_temporal_costs,
        "swapped_combined_costs": effective_swapped_costs,
        "legacy_decision_mask": legacy_decision.astype(bool),
        "decision_scores": decision_score,
        "decision_scores_smoothed": smoothed_decision_score,
        "strong_positive_margin": np.full_like(decision_score, strong_positive_margin, dtype=float),
        "viterbi_state_mask": viterbi_state_mask.astype(bool),
        "candidate_mask": candidate_mask.astype(bool),
        "legacy_candidate_mask": legacy_candidate_mask.astype(bool),
        "temporal_support_mask": temporal_support_mask.astype(bool),
    }
    return suspect_mask, diagnostics, detail_arrays


def apply_left_right_flip_to_points(points_2d: np.ndarray, suspect_mask: np.ndarray) -> np.ndarray:
    """Applique un swap gauche/droite a un tenseur `(n_cam, n_frames, 17, d)`."""
    corrected = np.array(points_2d, copy=True)
    for cam_idx in range(suspect_mask.shape[0]):
        suspect_frames = np.flatnonzero(suspect_mask[cam_idx])
        for frame_idx in suspect_frames:
            corrected[cam_idx, frame_idx] = swap_left_right_keypoints(corrected[cam_idx, frame_idx])
    return corrected


def apply_left_right_flip_corrections(pose_data: PoseData, suspect_mask: np.ndarray) -> PoseData:
    """Retourne une copie de `pose_data` ou les vues suspectes sont swappees.

    La correction s'applique camera par camera et frame par frame, uniquement
    quand le diagnostic epipolaire juge le swap gauche/droite plus coherent.
    """
    corrected_keypoints = apply_left_right_flip_to_points(pose_data.keypoints, suspect_mask)
    corrected_scores = apply_left_right_flip_to_points(pose_data.scores[..., np.newaxis], suspect_mask)[..., 0]
    return PoseData(
        camera_names=list(pose_data.camera_names),
        frames=np.array(pose_data.frames, copy=True),
        keypoints=corrected_keypoints,
        scores=corrected_scores,
        frame_stride=int(getattr(pose_data, "frame_stride", 1)),
        raw_keypoints=(
            None
            if pose_data.raw_keypoints is None
            else apply_left_right_flip_to_points(pose_data.raw_keypoints, suspect_mask)
        ),
        filtered_keypoints=(
            None
            if pose_data.filtered_keypoints is None
            else apply_left_right_flip_to_points(pose_data.filtered_keypoints, suspect_mask)
        ),
    )


def multiview_coherence_score(reprojection_error_px: float, threshold_px: float) -> float:
    """Convertit une erreur de reprojection en score de coherence dans `[0, 1]`.

    Le score vaut 1 pour une vue parfaitement coherente et decroit rapidement
    lorsque l'erreur depasse le seuil de triangulation retenu.
    """
    if not np.isfinite(reprojection_error_px):
        return 0.0
    threshold_px = max(threshold_px, 1e-6)
    return float(np.exp(-0.5 * (reprojection_error_px / threshold_px) ** 2))


def robust_triangulation_from_best_cameras(
    projections: list[np.ndarray],
    observations: np.ndarray,
    confidences: np.ndarray,
    calibrations: list[CameraCalibration],
    error_threshold_px: float,
    min_cameras_for_triangulation: int,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """Triangulation robuste inspiree de Pose2Sim.

    L'algorithme retire progressivement des cameras, teste toutes les
    combinaisons possibles et conserve la configuration avec l'erreur moyenne
    de reprojection la plus faible. A la fin, on derive:
    - l'erreur de reprojection par vue,
    - un masque des vues exclues,
    - un score de coherence par vue.
    """
    n_cams = len(projections)
    valid = np.isfinite(observations[:, 0]) & np.isfinite(observations[:, 1]) & (confidences > 0)

    if np.count_nonzero(valid) < min_cameras_for_triangulation:
        nan_views = np.full(n_cams, np.nan)
        return np.full(3, np.nan), np.nan, nan_views, np.zeros(n_cams), ~valid

    error_min = np.inf
    best_point = np.full(3, np.nan)
    best_excluded = np.array([], dtype=int)
    best_included = np.where(valid)[0]
    valid_indices = np.where(valid)[0]

    max_cams_off = max(0, np.count_nonzero(valid) - min_cameras_for_triangulation)
    for nb_cams_off in range(max_cams_off + 1):
        local_exclusions = exclusion_combinations(valid_indices.size, nb_cams_off)
        candidate_exclusions = [tuple(valid_indices[list(local_exclusion)]) for local_exclusion in local_exclusions]
        if not candidate_exclusions:
            candidate_exclusions = [tuple()]

        for excluded in candidate_exclusions:
            candidate_mask = valid.copy()
            candidate_mask[list(excluded)] = False
            included_indices = np.where(candidate_mask)[0]
            if included_indices.size < min_cameras_for_triangulation:
                continue

            point = weighted_triangulation(
                [projections[i] for i in included_indices],
                observations[included_indices],
                confidences[included_indices],
            )
            if not np.all(np.isfinite(point)):
                continue

            projected = project_point_with_projection_matrices(
                np.asarray([projections[i] for i in included_indices], dtype=float), point
            )
            errors = np.linalg.norm(observations[included_indices] - projected, axis=1)
            candidate_error = float(np.mean(errors)) if errors.size else np.inf

            if candidate_error < error_min:
                error_min = candidate_error
                best_point = point
                best_excluded = np.asarray(excluded, dtype=int)
                best_included = included_indices

        if error_min <= error_threshold_px:
            break

    reprojection_error_per_view = np.full(n_cams, np.nan)
    coherence_per_view = np.zeros(n_cams)
    excluded_views = np.ones(n_cams, dtype=bool)

    if not np.all(np.isfinite(best_point)):
        return best_point, np.nan, reprojection_error_per_view, coherence_per_view, excluded_views

    valid_indices = np.where(valid)[0]
    projected_valid = project_point_with_projection_matrices(
        np.asarray([projections[i] for i in valid_indices], dtype=float), best_point
    )
    reprojection_error_per_view[valid_indices] = np.linalg.norm(observations[valid_indices] - projected_valid, axis=1)

    excluded_views[:] = True
    excluded_views[best_included] = False
    for i_cam in best_included:
        coherence_per_view[i_cam] = multiview_coherence_score(reprojection_error_per_view[i_cam], error_threshold_px)

    if error_min > error_threshold_px:
        best_point = np.full(3, np.nan)

    return best_point, error_min, reprojection_error_per_view, coherence_per_view, excluded_views


def greedy_triangulation_from_best_cameras(
    projections: list[np.ndarray],
    observations: np.ndarray,
    confidences: np.ndarray,
    calibrations: list[CameraCalibration],
    error_threshold_px: float,
    min_cameras_for_triangulation: int,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """Version plus rapide de la triangulation robuste.

    On part de toutes les vues valides puis on retire iterativement la vue avec
    la plus mauvaise erreur de reprojection jusqu'a atteindre le seuil ou le
    minimum de cameras.
    """
    n_cams = len(projections)
    valid = np.isfinite(observations[:, 0]) & np.isfinite(observations[:, 1]) & (confidences > 0)

    if np.count_nonzero(valid) < min_cameras_for_triangulation:
        nan_views = np.full(n_cams, np.nan)
        return np.full(3, np.nan), np.nan, nan_views, np.zeros(n_cams), ~valid

    included_indices = np.where(valid)[0].tolist()
    best_point = np.full(3, np.nan)
    best_error = np.inf
    best_included = np.asarray(included_indices, dtype=int)

    while len(included_indices) >= min_cameras_for_triangulation:
        point = weighted_triangulation(
            [projections[i] for i in included_indices],
            observations[included_indices],
            confidences[included_indices],
        )
        if not np.all(np.isfinite(point)):
            break

        projected = project_point_with_projection_matrices(
            np.asarray([projections[i] for i in included_indices], dtype=float), point
        )
        errors = np.linalg.norm(observations[included_indices] - projected, axis=1)
        mean_error = float(np.mean(errors))
        if mean_error < best_error:
            best_error = mean_error
            best_point = point
            best_included = np.asarray(included_indices, dtype=int)
        if mean_error <= error_threshold_px or len(included_indices) == min_cameras_for_triangulation:
            break

        worst_local_idx = int(np.nanargmax(errors))
        del included_indices[worst_local_idx]

    reprojection_error_per_view = np.full(n_cams, np.nan)
    coherence_per_view = np.zeros(n_cams)
    excluded_views = np.ones(n_cams, dtype=bool)

    if not np.all(np.isfinite(best_point)):
        return best_point, np.nan, reprojection_error_per_view, coherence_per_view, excluded_views

    valid_indices = np.where(valid)[0]
    projected_valid = project_point_with_projection_matrices(
        np.asarray([projections[i] for i in valid_indices], dtype=float), best_point
    )
    reprojection_error_per_view[valid_indices] = np.linalg.norm(observations[valid_indices] - projected_valid, axis=1)

    excluded_views[best_included] = False
    for i_cam in best_included:
        coherence_per_view[i_cam] = multiview_coherence_score(reprojection_error_per_view[i_cam], error_threshold_px)

    if best_error > error_threshold_px:
        best_point = np.full(3, np.nan)

    return best_point, best_error, reprojection_error_per_view, coherence_per_view, excluded_views


def once_triangulation_from_best_cameras(
    projections: list[np.ndarray],
    observations: np.ndarray,
    confidences: np.ndarray,
    calibrations: list[CameraCalibration],
    error_threshold_px: float,
    min_cameras_for_triangulation: int,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate once from all valid cameras without view rejection."""

    _ = calibrations
    n_cams = len(projections)
    valid = np.isfinite(observations[:, 0]) & np.isfinite(observations[:, 1]) & (confidences > 0)
    if np.count_nonzero(valid) < min_cameras_for_triangulation:
        nan_views = np.full(n_cams, np.nan)
        return np.full(3, np.nan), np.nan, nan_views, np.zeros(n_cams), ~valid

    included_indices = np.where(valid)[0]
    point = weighted_triangulation(
        [projections[i] for i in included_indices],
        observations[included_indices],
        confidences[included_indices],
    )
    reprojection_error_per_view = np.full(n_cams, np.nan)
    coherence_per_view = np.zeros(n_cams)
    excluded_views = ~valid
    if not np.all(np.isfinite(point)):
        return point, np.nan, reprojection_error_per_view, coherence_per_view, excluded_views

    projected = project_point_with_projection_matrices(
        np.asarray([projections[i] for i in included_indices], dtype=float), point
    )
    errors = np.linalg.norm(observations[included_indices] - projected, axis=1)
    mean_error = float(np.mean(errors)) if errors.size else np.nan
    reprojection_error_per_view[included_indices] = errors
    for i_cam in included_indices:
        coherence_per_view[i_cam] = multiview_coherence_score(reprojection_error_per_view[i_cam], error_threshold_px)
    if mean_error > error_threshold_px:
        point = np.full(3, np.nan)
    return point, mean_error, reprojection_error_per_view, coherence_per_view, excluded_views


def compute_epipolar_coherence(
    pose_data: PoseData,
    fundamental_matrices: dict[tuple[int, int], np.ndarray],
    threshold_px: float,
) -> np.ndarray:
    """Calcule un score de coherence epipolaire par frame, keypoint et vue.

    Le score d'une vue est obtenu a partir des erreurs de Sampson avec toutes
    les autres vues valides. Cela donne un indicateur geometrique de
    compatibilite multivue sans utiliser la triangulation comme critere.
    """
    n_cams, n_frames, n_keypoints, _ = pose_data.keypoints.shape
    threshold_px = max(threshold_px, 1e-6)
    coherence = np.zeros((n_frames, n_keypoints, n_cams), dtype=float)
    homogeneous_keypoints = np.concatenate(
        (pose_data.keypoints, np.ones((n_cams, n_frames, n_keypoints, 1), dtype=float)),
        axis=-1,
    )
    valid_measurements = pose_data.scores > 0
    valid_measurements &= np.all(np.isfinite(pose_data.keypoints), axis=-1)

    # On vectorise le calcul paire par paire: pour chaque couple dirige
    # (camera i -> camera j), on calcule toute la carte d'erreur Sampson d'un
    # bloc `(frames, keypoints)` puis on agrège par mediane sur les partenaires.
    pair_errors_by_camera: list[list[np.ndarray]] = [[] for _ in range(n_cams)]
    for i_cam in range(n_cams):
        xi = homogeneous_keypoints[i_cam]
        valid_i = valid_measurements[i_cam]
        for j_cam in range(n_cams):
            if i_cam == j_cam:
                continue
            xj = homogeneous_keypoints[j_cam]
            valid_ij = valid_i & valid_measurements[j_cam]
            if not np.any(valid_ij):
                continue
            Fij = fundamental_matrices[(i_cam, j_cam)]
            Fxi = xi @ Fij.T
            Ftxj = xj @ Fij
            numer = np.sum(xj * Fxi, axis=-1)
            denom = Fxi[..., 0] ** 2 + Fxi[..., 1] ** 2 + Ftxj[..., 0] ** 2 + Ftxj[..., 1] ** 2
            pair_error = np.full((n_frames, n_keypoints), np.nan, dtype=float)
            valid_denom = valid_ij & (denom > 1e-12) & np.isfinite(numer) & np.isfinite(denom)
            pair_error[valid_denom] = np.abs(numer[valid_denom]) / np.sqrt(denom[valid_denom])
            pair_errors_by_camera[i_cam].append(pair_error)

    for i_cam in range(n_cams):
        if not pair_errors_by_camera[i_cam]:
            continue
        stacked_errors = np.stack(pair_errors_by_camera[i_cam], axis=0)
        any_finite = np.any(np.isfinite(stacked_errors), axis=0)
        median_error = np.full((n_frames, n_keypoints), np.nan, dtype=float)
        if np.any(any_finite):
            median_error[any_finite] = np.nanmedian(stacked_errors[:, any_finite], axis=0)
        valid_error = np.isfinite(median_error)
        coherence_cam = coherence[:, :, i_cam]
        coherence_cam[valid_error] = np.exp(-0.5 * (median_error[valid_error] / threshold_px) ** 2)
        coherence[:, :, i_cam] = coherence_cam

    return coherence


def triangulate_pose2sim_like(
    pose_data: PoseData,
    calibrations: dict[str, CameraCalibration],
    error_threshold_px: float = DEFAULT_REPROJECTION_THRESHOLD_PX,
    min_cameras_for_triangulation: int = DEFAULT_MIN_CAMERAS_FOR_TRIANGULATION,
    coherence_method: str = DEFAULT_COHERENCE_METHOD,
    epipolar_threshold_px: float = DEFAULT_EPIPOLAR_THRESHOLD_PX,
    triangulation_method: str = DEFAULT_TRIANGULATION_METHOD,
    n_workers: int = DEFAULT_TRIANGULATION_WORKERS,
    precomputed_epipolar_coherence: np.ndarray | None = None,
    precomputed_epipolar_time_s: float | None = None,
) -> ReconstructionResult:
    """Effectue une reconstruction 3D frame-par-frame de tous les keypoints."""
    triangulation_method = canonical_triangulation_method(triangulation_method)
    coherence_method = canonical_coherence_method(coherence_method, triangulation_method)
    projections = [calibrations[name].P for name in pose_data.camera_names]
    ordered_calibrations = [calibrations[name] for name in pose_data.camera_names]
    n_frames = pose_data.keypoints.shape[1]
    n_keypoints = pose_data.keypoints.shape[2]
    points_3d = np.full((n_frames, n_keypoints, 3), np.nan)
    mean_confidence = np.full((n_frames, n_keypoints), np.nan)
    reprojection_error = np.full((n_frames, n_keypoints), np.nan)
    reprojection_error_per_view = np.full((n_frames, n_keypoints, len(pose_data.camera_names)), np.nan)
    triangulation_coherence = np.zeros((n_frames, n_keypoints, len(pose_data.camera_names)))
    excluded_views = np.ones((n_frames, n_keypoints, len(pose_data.camera_names)), dtype=bool)
    triangulate_one = (
        once_triangulation_from_best_cameras
        if triangulation_method == "once"
        else (
            robust_triangulation_from_best_cameras
            if triangulation_method == "exhaustive"
            else greedy_triangulation_from_best_cameras
        )
    )

    fundamental_matrices = {
        (i_cam, j_cam): fundamental_matrix(ordered_calibrations[i_cam], ordered_calibrations[j_cam])
        for i_cam in range(len(ordered_calibrations))
        for j_cam in range(len(ordered_calibrations))
        if i_cam != j_cam
    }

    def process_frame(frame_idx: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        frame_points = np.full((n_keypoints, 3), np.nan)
        frame_mean_confidence = np.full(n_keypoints, np.nan)
        frame_reprojection_error = np.full(n_keypoints, np.nan)
        frame_reprojection_per_view = np.full((n_keypoints, len(pose_data.camera_names)), np.nan)
        frame_triangulation_coherence = np.zeros((n_keypoints, len(pose_data.camera_names)))
        frame_excluded_views = np.ones((n_keypoints, len(pose_data.camera_names)), dtype=bool)
        for kp_idx in range(n_keypoints):
            observations = pose_data.keypoints[:, frame_idx, kp_idx, :]
            confidences = pose_data.scores[:, frame_idx, kp_idx]
            point, mean_error, errors_view, coherence_view, excluded_view = triangulate_one(
                projections=projections,
                observations=observations,
                confidences=confidences,
                calibrations=ordered_calibrations,
                error_threshold_px=error_threshold_px,
                min_cameras_for_triangulation=min_cameras_for_triangulation,
            )
            frame_points[kp_idx] = point
            frame_reprojection_error[kp_idx] = mean_error
            frame_reprojection_per_view[kp_idx] = errors_view
            frame_triangulation_coherence[kp_idx] = coherence_view
            frame_excluded_views[kp_idx] = excluded_view
            valid = confidences > 0
            if np.any(valid):
                frame_mean_confidence[kp_idx] = float(np.mean(confidences[valid]))
        return (
            frame_idx,
            frame_points,
            frame_mean_confidence,
            frame_reprojection_error,
            frame_reprojection_per_view,
            frame_triangulation_coherence,
            frame_excluded_views,
        )

    if n_workers <= 1:
        frame_results = [process_frame(frame_idx) for frame_idx in range(n_frames)]
    else:
        with cf.ThreadPoolExecutor(max_workers=n_workers) as executor:
            frame_results = list(executor.map(process_frame, range(n_frames)))

    for (
        frame_idx,
        frame_points,
        frame_mean_confidence,
        frame_reprojection_error,
        frame_reprojection_per_view,
        frame_triangulation_coherence,
        frame_excluded_views,
    ) in frame_results:
        points_3d[frame_idx] = frame_points
        mean_confidence[frame_idx] = frame_mean_confidence
        reprojection_error[frame_idx] = frame_reprojection_error
        reprojection_error_per_view[frame_idx] = frame_reprojection_per_view
        triangulation_coherence[frame_idx] = frame_triangulation_coherence
        excluded_views[frame_idx] = frame_excluded_views

    if precomputed_epipolar_coherence is None:
        t_epipolar = time.perf_counter()
        epipolar_coherence = compute_epipolar_coherence(
            pose_data,
            fundamental_matrices,
            threshold_px=epipolar_threshold_px,
        )
        epipolar_time_s = time.perf_counter() - t_epipolar
    else:
        epipolar_coherence = np.asarray(precomputed_epipolar_coherence, dtype=float)
        epipolar_time_s = float(precomputed_epipolar_time_s) if precomputed_epipolar_time_s is not None else 0.0
    multiview_coherence = select_active_coherence(
        epipolar_coherence=epipolar_coherence,
        triangulation_coherence=triangulation_coherence,
        coherence_method=coherence_method,
    )

    return ReconstructionResult(
        frames=pose_data.frames,
        points_3d=points_3d,
        mean_confidence=mean_confidence,
        reprojection_error=reprojection_error,
        reprojection_error_per_view=reprojection_error_per_view,
        multiview_coherence=multiview_coherence,
        epipolar_coherence=epipolar_coherence,
        triangulation_coherence=triangulation_coherence,
        excluded_views=excluded_views,
        coherence_method=coherence_method,
        epipolar_coherence_compute_time_s=epipolar_time_s,
        triangulation_compute_time_s=0.0,
    )


def median_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """Distance mediane robuste entre deux series temporelles de points 3D."""
    distances = np.linalg.norm(points_a - points_b, axis=1)
    valid = np.isfinite(distances)
    if not np.any(valid):
        return np.nan
    return float(np.nanmedian(distances[valid]))


def estimate_segment_lengths(reconstruction: ReconstructionResult, fps: float, window_s: float = 2.0) -> SegmentLengths:
    """Estime des longueurs segmentaires a partir des premieres secondes du mouvement.

    Les longueurs sont derivees de couples de keypoints COCO17. Des valeurs de
    secours sont prevues pour les cas ou la triangulation est trop incomplete.
    """
    n_frames = min(reconstruction.points_3d.shape[0], max(1, int(round(fps * window_s))))
    pts = reconstruction.points_3d[:n_frames]

    l_shoulder = pts[:, KP_INDEX["left_shoulder"], :]
    r_shoulder = pts[:, KP_INDEX["right_shoulder"], :]
    l_hip = pts[:, KP_INDEX["left_hip"], :]
    r_hip = pts[:, KP_INDEX["right_hip"], :]
    shoulder_center = 0.5 * (l_shoulder + r_shoulder)
    hip_center = 0.5 * (l_hip + r_hip)

    trunk_height = median_distance(shoulder_center, hip_center)
    head_length = median_distance(pts[:, KP_INDEX["nose"], :], shoulder_center)
    shoulder_width = median_distance(l_shoulder, r_shoulder)
    hip_width = median_distance(l_hip, r_hip)
    upper_arm_length = np.nanmedian(
        [
            median_distance(pts[:, KP_INDEX["left_shoulder"], :], pts[:, KP_INDEX["left_elbow"], :]),
            median_distance(pts[:, KP_INDEX["right_shoulder"], :], pts[:, KP_INDEX["right_elbow"], :]),
        ]
    )
    forearm_length = np.nanmedian(
        [
            median_distance(pts[:, KP_INDEX["left_elbow"], :], pts[:, KP_INDEX["left_wrist"], :]),
            median_distance(pts[:, KP_INDEX["right_elbow"], :], pts[:, KP_INDEX["right_wrist"], :]),
        ]
    )
    thigh_length = np.nanmedian(
        [
            median_distance(pts[:, KP_INDEX["left_hip"], :], pts[:, KP_INDEX["left_knee"], :]),
            median_distance(pts[:, KP_INDEX["right_hip"], :], pts[:, KP_INDEX["right_knee"], :]),
        ]
    )
    shank_length = np.nanmedian(
        [
            median_distance(pts[:, KP_INDEX["left_knee"], :], pts[:, KP_INDEX["left_ankle"], :]),
            median_distance(pts[:, KP_INDEX["right_knee"], :], pts[:, KP_INDEX["right_ankle"], :]),
        ]
    )

    left_eye = pts[:, KP_INDEX["left_eye"], :]
    right_eye = pts[:, KP_INDEX["right_eye"], :]
    nose = pts[:, KP_INDEX["nose"], :]
    left_ear = pts[:, KP_INDEX["left_ear"], :]
    right_ear = pts[:, KP_INDEX["right_ear"], :]
    eye_offset_x = np.nanmedian(np.linalg.norm(0.5 * (left_eye + right_eye) - nose, axis=1))
    eye_offset_y = 0.5 * median_distance(left_eye, right_eye)
    ear_offset_y = 0.5 * median_distance(left_ear, right_ear)

    fallback = {
        "trunk_height": 0.55,
        "head_length": 0.24,
        "shoulder_half_width": 0.18,
        "hip_half_width": 0.12,
        "upper_arm_length": 0.29,
        "forearm_length": 0.27,
        "thigh_length": 0.42,
        "shank_length": 0.43,
        "eye_offset_x": 0.05,
        "eye_offset_y": 0.03,
        "ear_offset_y": 0.07,
    }
    values = {
        "trunk_height": trunk_height,
        "head_length": head_length,
        "shoulder_half_width": shoulder_width / 2.0,
        "hip_half_width": hip_width / 2.0,
        "upper_arm_length": upper_arm_length,
        "forearm_length": forearm_length,
        "thigh_length": thigh_length,
        "shank_length": shank_length,
        "eye_offset_x": eye_offset_x,
        "eye_offset_y": eye_offset_y,
        "ear_offset_y": ear_offset_y,
    }

    for key, default_value in fallback.items():
        if not np.isfinite(values[key]) or values[key] <= 0:
            values[key] = default_value

    return SegmentLengths(**values)


def female_deleva_inertia_parameters(lengths: SegmentLengths, total_mass_kg: float):
    """Construit des inerties segmentaires via le modele proportionnel de de Leva.

    Le modele cinematique courant ne comporte ni segments de main ni de pied.
    On reprojecte donc de facon pragmatique:
    - la masse/inertie de la main sur l'avant-bras,
    - la masse/inertie du pied sur la jambe.
    """
    ensure_local_imports()
    from biobuddy import InertiaParametersReal
    from biobuddy.characteristics.de_leva import DeLevaTable, SegmentName, Sex

    shoulder_height = lengths.shank_length + lengths.thigh_length + lengths.trunk_height
    hip_height = lengths.shank_length + lengths.thigh_length
    knee_height = lengths.shank_length
    ankle_height = 0.0
    total_height = shoulder_height + lengths.head_length
    shoulder_span = 2.0 * lengths.shoulder_half_width
    elbow_span = shoulder_span + 2.0 * lengths.upper_arm_length
    wrist_span = elbow_span + 2.0 * lengths.forearm_length
    hand_length = (0.1701 / 1.735) * total_height
    finger_span = wrist_span + 2.0 * hand_length
    foot_length = (0.2283 / 1.735) * total_height
    hip_width = 2.0 * lengths.hip_half_width

    table = DeLevaTable(total_mass=total_mass_kg, sex=Sex.FEMALE)
    table.from_measurements(
        total_height=total_height,
        ankle_height=ankle_height,
        knee_height=knee_height,
        hip_height=hip_height,
        shoulder_height=shoulder_height,
        finger_span=finger_span,
        wrist_span=wrist_span,
        elbow_span=elbow_span,
        shoulder_span=shoulder_span,
        foot_length=foot_length,
        hip_width=hip_width,
    )

    def to_real(segment_name: SegmentName) -> InertiaParametersReal:
        generic = table[segment_name]
        mass = float(generic.relative_mass(None, None))
        com = np.asarray(generic.center_of_mass(None, None), dtype=float).reshape(-1)[:3]
        inertia_diag = np.asarray(generic.inertia(None, None), dtype=float).reshape(-1)[:3]
        return InertiaParametersReal(mass=mass, center_of_mass=com, inertia=np.diag(inertia_diag))

    def aggregate_distal(
        primary: InertiaParametersReal, distal: InertiaParametersReal, distal_offset: np.ndarray
    ) -> InertiaParametersReal:
        primary_mass = float(primary.mass)
        distal_mass = float(distal.mass)
        total_mass = primary_mass + distal_mass

        primary_com = np.asarray(primary.center_of_mass, dtype=float)[:3, 0]
        distal_com = distal_offset + np.asarray(distal.center_of_mass, dtype=float)[:3, 0]
        total_com = (primary_mass * primary_com + distal_mass * distal_com) / total_mass

        I_primary = np.asarray(primary.inertia, dtype=float)[:3, :3]
        I_distal = np.asarray(distal.inertia, dtype=float)[:3, :3]
        d_primary = primary_com - total_com
        d_distal = distal_com - total_com
        I_primary_shifted = I_primary + primary_mass * (
            (d_primary @ d_primary) * np.eye(3) - np.outer(d_primary, d_primary)
        )
        I_distal_shifted = I_distal + distal_mass * ((d_distal @ d_distal) * np.eye(3) - np.outer(d_distal, d_distal))
        return InertiaParametersReal(
            mass=total_mass, center_of_mass=total_com, inertia=I_primary_shifted + I_distal_shifted
        )

    trunk = to_real(SegmentName.TRUNK)
    head = to_real(SegmentName.HEAD)
    upper_arm = to_real(SegmentName.UPPER_ARM)
    forearm = aggregate_distal(
        to_real(SegmentName.LOWER_ARM),
        to_real(SegmentName.HAND),
        np.array([0.0, 0.0, -lengths.forearm_length]),
    )
    thigh = to_real(SegmentName.THIGH)
    shank = aggregate_distal(
        to_real(SegmentName.SHANK),
        to_real(SegmentName.FOOT),
        np.array([0.0, 0.0, -lengths.shank_length]),
    )
    return {
        "TRUNK": trunk,
        "HEAD": head,
        "UPPER_ARM": upper_arm,
        "FOREARM": forearm,
        "THIGH": thigh,
        "SHANK": shank,
        "TOTAL_HEIGHT_M": total_height,
    }


def build_biomod(
    lengths: SegmentLengths,
    output_path: Path,
    subject_mass_kg: float = DEFAULT_SUBJECT_MASS_KG,
    reconstruction: ReconstructionResult | None = None,
    apply_initial_root_rotation_correction: bool = True,
) -> Path:
    """Construit un modele `.bioMod` minimal compatible avec les keypoints COCO17.

    Des parametres inertiels sont ajoutes a partir du modele proportionnel de
    de Leva pour une femme. Comme le modele courant ne comporte pas de segments
    main/pied, leurs inerties sont agrégées aux segments avant-bras/jambe.
    """
    ensure_local_imports()
    from biobuddy import (
        BiomechanicalModelReal,
        MarkerReal,
        MeshReal,
        Rotations,
        SegmentCoordinateSystemReal,
        SegmentReal,
        Translations,
    )

    inertia = female_deleva_inertia_parameters(lengths, total_mass_kg=subject_mass_kg)
    model = BiomechanicalModelReal()

    def mesh_with_axes(base_points: list[tuple[float, float, float]], axis_scale: float) -> MeshReal:
        """Construit un mesh polyline affichant aussi le repere local du segment.

        On evite d'ajouter des marqueurs supplementaires au modele pour ne pas
        perturber l'IK, le Kalman ou les comparaisons qui s'appuient sur les
        marqueurs COCO17. Les axes locaux sont donc encodes dans le mesh
        visuel: X en avant, Y a gauche, Z du repere local du segment.
        """
        origin = (0.0, 0.0, 0.0)
        axis_len = max(float(axis_scale), 0.05)
        axis_points = [
            origin,
            (axis_len, 0.0, 0.0),
            origin,
            (0.0, axis_len, 0.0),
            origin,
            (0.0, 0.0, axis_len),
        ]
        return MeshReal(list(base_points) + axis_points)

    model.add_segment(
        SegmentReal(
            name="TRUNK",
            translations=Translations.XYZ,
            rotations=Rotations.YXZ,
            inertia_parameters=inertia["TRUNK"],
            mesh=mesh_with_axes(
                [
                    (0.0, -lengths.hip_half_width, 0.0),
                    (0.0, lengths.hip_half_width, 0.0),
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, lengths.trunk_height),
                ],
                axis_scale=0.25 * lengths.trunk_height,
            ),
        )
    )
    trunk = model.segments["TRUNK"]
    trunk.add_marker(MarkerReal(name="left_hip", parent_name="TRUNK", position=[0, lengths.hip_half_width, 0]))
    trunk.add_marker(MarkerReal(name="right_hip", parent_name="TRUNK", position=[0, -lengths.hip_half_width, 0]))
    trunk.add_marker(
        MarkerReal(
            name="left_shoulder", parent_name="TRUNK", position=[0, lengths.shoulder_half_width, lengths.trunk_height]
        )
    )
    trunk.add_marker(
        MarkerReal(
            name="right_shoulder", parent_name="TRUNK", position=[0, -lengths.shoulder_half_width, lengths.trunk_height]
        )
    )

    model.add_segment(
        SegmentReal(
            name="HEAD",
            parent_name="TRUNK",
            segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                np.zeros(3), "xyz", np.array([0, 0, lengths.trunk_height]), is_scs_local=True
            ),
            rotations=Rotations.XYZ,
            inertia_parameters=inertia["HEAD"],
            mesh=mesh_with_axes(
                [(0.0, 0.0, 0.0), (lengths.head_length, 0.0, lengths.head_length)],
                axis_scale=0.35 * lengths.head_length,
            ),
        )
    )
    head = model.segments["HEAD"]
    head.add_marker(MarkerReal(name="nose", parent_name="HEAD", position=[lengths.head_length, 0, lengths.head_length]))
    head.add_marker(
        MarkerReal(
            name="left_eye",
            parent_name="HEAD",
            position=[lengths.head_length - lengths.eye_offset_x, lengths.eye_offset_y, lengths.head_length],
        )
    )
    head.add_marker(
        MarkerReal(
            name="right_eye",
            parent_name="HEAD",
            position=[lengths.head_length - lengths.eye_offset_x, -lengths.eye_offset_y, lengths.head_length],
        )
    )
    head.add_marker(
        MarkerReal(name="left_ear", parent_name="HEAD", position=[0, lengths.ear_offset_y, 0.7 * lengths.head_length])
    )
    head.add_marker(
        MarkerReal(name="right_ear", parent_name="HEAD", position=[0, -lengths.ear_offset_y, 0.7 * lengths.head_length])
    )

    for side, sign in (("left", 1.0), ("right", -1.0)):
        shoulder_offset = (0, sign * lengths.shoulder_half_width, lengths.trunk_height)
        hip_offset = (0, sign * lengths.hip_half_width, 0)

        upper_name = f"{side.upper()}_UPPER_ARM"
        forearm_name = f"{side.upper()}_FOREARM"
        thigh_name = f"{side.upper()}_THIGH"
        shank_name = f"{side.upper()}_SHANK"

        model.add_segment(
            SegmentReal(
                name=upper_name,
                parent_name="TRUNK",
                segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                    np.zeros(3), "xyz", np.asarray(shoulder_offset, dtype=float), is_scs_local=True
                ),
                rotations=Rotations.YX,
                inertia_parameters=inertia["UPPER_ARM"],
                mesh=mesh_with_axes(
                    [(0.0, 0.0, 0.0), (0.0, 0.0, -lengths.upper_arm_length)],
                    axis_scale=0.3 * lengths.upper_arm_length,
                ),
            )
        )
        upper_arm = model.segments[upper_name]
        upper_arm.add_marker(
            MarkerReal(name=f"{side}_elbow", parent_name=upper_name, position=[0, 0, -lengths.upper_arm_length])
        )

        model.add_segment(
            SegmentReal(
                name=forearm_name,
                parent_name=upper_name,
                segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                    np.zeros(3), "xyz", np.array([0, 0, -lengths.upper_arm_length]), is_scs_local=True
                ),
                rotations=Rotations.ZY,
                inertia_parameters=inertia["FOREARM"],
                mesh=mesh_with_axes(
                    [(0.0, 0.0, 0.0), (0.0, 0.0, -lengths.forearm_length)],
                    axis_scale=0.3 * lengths.forearm_length,
                ),
            )
        )
        forearm = model.segments[forearm_name]
        forearm.add_marker(
            MarkerReal(name=f"{side}_wrist", parent_name=forearm_name, position=[0, 0, -lengths.forearm_length])
        )

        model.add_segment(
            SegmentReal(
                name=thigh_name,
                parent_name="TRUNK",
                segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                    np.zeros(3), "xyz", np.asarray(hip_offset, dtype=float), is_scs_local=True
                ),
                rotations=Rotations.YXZ,
                inertia_parameters=inertia["THIGH"],
                mesh=mesh_with_axes(
                    [(0.0, 0.0, 0.0), (0.0, 0.0, -lengths.thigh_length)],
                    axis_scale=0.25 * lengths.thigh_length,
                ),
            )
        )
        thigh = model.segments[thigh_name]
        thigh.add_marker(
            MarkerReal(name=f"{side}_knee", parent_name=thigh_name, position=[0, 0, -lengths.thigh_length])
        )

        model.add_segment(
            SegmentReal(
                name=shank_name,
                parent_name=thigh_name,
                segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                    np.zeros(3), "xyz", np.array([0, 0, -lengths.thigh_length]), is_scs_local=True
                ),
                rotations=Rotations.Y,
                inertia_parameters=inertia["SHANK"],
                mesh=mesh_with_axes(
                    [(0.0, 0.0, 0.0), (0.0, 0.0, -lengths.shank_length)],
                    axis_scale=0.25 * lengths.shank_length,
                ),
            )
        )
        shank = model.segments[shank_name]
        shank.add_marker(
            MarkerReal(name=f"{side}_ankle", parent_name=shank_name, position=[0, 0, -lengths.shank_length])
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.to_biomod(str(output_path), with_mesh=True)
    remove_ghost_root_segment_from_biomod(output_path)
    # Si le repere tronc initial est mal aligne en lacet par rapport au repere
    # global, on applique directement dans le bloc RT du TRUNK une rotation
    # autour de Z arrondie au multiple de pi/2 le plus proche.
    correction_angle = trunk_root_z_correction_angle(reconstruction)
    if apply_initial_root_rotation_correction and abs(correction_angle) > 1e-8:
        rotate_trunk_rt_about_z_in_biomod(output_path, correction_angle)
    return output_path


def remove_ghost_root_segment_from_biomod(biomod_path: Path) -> None:
    """Supprime le segment `root` ajoute par l'export et rattache le tronc a `base`.

    Dans ce projet, ce segment intermediaire identite n'apporte rien. Il
    complique au contraire la lecture du modele et interfère avec
    l'identification d'une base flottante utile pour les diagnostics.
    """
    text = biomod_path.read_text()
    ghost_block = (
        "segment\troot\n"
        "\tparent\tbase\n"
        "\tRTinMatrix\t1\n"
        "\tRT\n"
        "\t\t1.000000\t0.000000\t0.000000\t0.000000\n"
        "\t\t0.000000\t1.000000\t0.000000\t0.000000\n"
        "\t\t0.000000\t0.000000\t1.000000\t0.000000\n"
        "\t\t0.000000\t0.000000\t0.000000\t1.000000\n"
        "endsegment\n"
    )
    if ghost_block in text:
        text = text.replace(ghost_block, "", 1)
    text = text.replace("parent\troot", "parent\tbase")
    biomod_path.write_text(text)


def trunk_root_z_correction_angle(reconstruction: ReconstructionResult | None) -> float:
    """Calcule une correction de lacet de la racine, arrondie au pi/2 le plus proche.

    L'angle est estime a partir de l'axe medio-lateral du tronc (axe `y`
    reconstruit avec les hanches et les epaules), projete dans le plan
    horizontal. On arrondit ensuite cet angle au multiple de pi/2 le plus
    proche pour garder un repere cardinal simple.
    """
    if reconstruction is None or reconstruction.points_3d.shape[0] == 0:
        return 0.0
    return root_z_correction_angle_from_points(
        reconstruction.points_3d,
        left_hip_idx=KP_INDEX["left_hip"],
        right_hip_idx=KP_INDEX["right_hip"],
        left_shoulder_idx=KP_INDEX["left_shoulder"],
        right_shoulder_idx=KP_INDEX["right_shoulder"],
    )


def rotate_trunk_rt_about_z_in_biomod(biomod_path: Path, angle_rad: float) -> None:
    """Applique au segment TRUNK une rotation autour de z dans son bloc RT."""
    cos_a = math.cos(float(angle_rad))
    sin_a = math.sin(float(angle_rad))
    text = biomod_path.read_text()
    old_block = (
        "segment\tTRUNK\n"
        "\tparent\tbase\n"
        "\tRTinMatrix\t1\n"
        "\tRT\n"
        "\t\t1.000000\t0.000000\t0.000000\t0.000000\n"
        "\t\t0.000000\t1.000000\t0.000000\t0.000000\n"
        "\t\t0.000000\t0.000000\t1.000000\t0.000000\n"
        "\t\t0.000000\t0.000000\t0.000000\t1.000000\n"
    )
    new_block = (
        "segment\tTRUNK\n"
        "\tparent\tbase\n"
        "\tRTinMatrix\t1\n"
        "\tRT\n"
        f"\t\t{cos_a:.6f}\t{-sin_a:.6f}\t0.000000\t0.000000\n"
        f"\t\t{sin_a:.6f}\t{cos_a:.6f}\t0.000000\t0.000000\n"
        "\t\t0.000000\t0.000000\t1.000000\t0.000000\n"
        "\t\t0.000000\t0.000000\t0.000000\t1.000000\n"
    )
    if old_block in text:
        text = text.replace(old_block, new_block, 1)
        biomod_path.write_text(text)


def marker_name_list(model) -> list[str]:
    """Retourne les noms de marqueurs du modele dans l'ordre interne `biorbd`."""
    return [name.to_string() for name in model.markerNames()]


def unwrap_with_gaps(values: np.ndarray) -> np.ndarray:
    """Applique un unwrap par DoF sans propager les NaN entre segments valides."""
    array = np.asarray(values, dtype=float)
    squeeze = array.ndim == 1
    if squeeze:
        array = array[:, np.newaxis]
    unwrapped = np.array(array, copy=True)
    for col_idx in range(unwrapped.shape[1]):
        column = unwrapped[:, col_idx]
        valid_idx = np.flatnonzero(np.isfinite(column))
        if valid_idx.size == 0:
            continue
        split_points = np.where(np.diff(valid_idx) > 1)[0] + 1
        segments = np.split(valid_idx, split_points)
        for segment in segments:
            if segment.size == 0:
                continue
            unwrapped[segment, col_idx] = np.unwrap(column[segment])
    return unwrapped[:, 0] if squeeze else unwrapped


def unwrap_root_rotations(q: np.ndarray, q_names: np.ndarray | list[str]) -> np.ndarray:
    """Applique un unwrap temporel aux trois rotations de la racine.

    Le but est d'eviter des sauts artificiels de type `+-2pi` sur les trois
    rotations du segment racine (`TRUNK:RotX`, `TRUNK:RotY`, `TRUNK:RotZ`).
    """
    q_unwrapped = np.array(q, copy=True)
    name_to_index = {str(name): i for i, name in enumerate(q_names)}
    root_rotation_names = ["TRUNK:RotX", "TRUNK:RotY", "TRUNK:RotZ"]
    for dof_name in root_rotation_names:
        if dof_name in name_to_index:
            q_unwrapped[:, name_to_index[dof_name]] = unwrap_with_gaps(q_unwrapped[:, name_to_index[dof_name]])
    return q_unwrapped


def debug_state_summary(state: np.ndarray, q_names: np.ndarray | list[str], nq: int, prefix: str) -> str:
    """Construit un resume compact d'un etat EKF pour le debug console."""
    q_names = [str(name) for name in q_names]
    q = np.asarray(state[:nq], dtype=float)
    qdot = np.asarray(state[nq : 2 * nq], dtype=float)
    qddot = np.asarray(state[2 * nq : 3 * nq], dtype=float)

    def _top_entries(values: np.ndarray) -> str:
        magnitudes = np.nan_to_num(np.abs(values), nan=-np.inf)
        top_indices = np.argsort(magnitudes)[::-1][:3]
        parts = []
        for idx in top_indices:
            if idx >= len(q_names):
                continue
            value = values[idx]
            parts.append(f"{q_names[idx]}={'nan' if not np.isfinite(value) else f'{value:.3f}'}")
        return "[" + ", ".join(parts) + "]"

    return (
        f"{prefix} | "
        f"q_max={float(np.nanmax(np.abs(q))):.3f}, "
        f"qdot_max={float(np.nanmax(np.abs(qdot))):.3f}, "
        f"qddot_max={float(np.nanmax(np.abs(qddot))):.3f}, "
        f"qdot_top={_top_entries(qdot)}"
    )


def validate_ekf_state_or_raise(
    state: np.ndarray,
    covariance: np.ndarray,
    q_names: np.ndarray | list[str],
    nq: int,
    frame_idx: int,
    stage: str,
    max_abs_q: float = 1e3,
    max_abs_qdot: float = 1e3,
    max_abs_qddot: float = 1e4,
) -> None:
    """Arrete le debug EKF si l'etat/covariance devient non fini ou explose."""
    if not np.all(np.isfinite(state)):
        raise RuntimeError(f"[DEBUG EKF 2D DYN] Etat non fini a la frame {frame_idx} ({stage}).")
    if not np.all(np.isfinite(covariance)):
        raise RuntimeError(f"[DEBUG EKF 2D DYN] Covariance non finie a la frame {frame_idx} ({stage}).")
    q = np.asarray(state[:nq], dtype=float)
    qdot = np.asarray(state[nq : 2 * nq], dtype=float)
    qddot = np.asarray(state[2 * nq : 3 * nq], dtype=float)
    if float(np.nanmax(np.abs(q))) > max_abs_q:
        raise RuntimeError(f"[DEBUG EKF 2D DYN] q explose a la frame {frame_idx} ({stage}).")
    if float(np.nanmax(np.abs(qdot))) > max_abs_qdot:
        raise RuntimeError(f"[DEBUG EKF 2D DYN] qdot explose a la frame {frame_idx} ({stage}).")
    if float(np.nanmax(np.abs(qddot))) > max_abs_qddot:
        raise RuntimeError(f"[DEBUG EKF 2D DYN] qddot explose a la frame {frame_idx} ({stage}).")


def points_to_marker_tensor(model, point_frame: np.ndarray) -> np.ndarray:
    """Mappe un frame de keypoints COCO17 vers le tenseur de marqueurs attendu par `biorbd`."""
    names = marker_name_list(model)
    markers = np.full((3, len(names), 1), np.nan)
    for i_marker, name in enumerate(names):
        if name in KP_INDEX and np.all(np.isfinite(point_frame[KP_INDEX[name]])):
            markers[:, i_marker, 0] = point_frame[KP_INDEX[name]]
    return markers


def first_valid_marker_tensor_from_reconstruction(
    model, reconstruction: ReconstructionResult
) -> tuple[int, np.ndarray] | tuple[None, None]:
    """Retourne la premiere frame triangulee exploitable et son tenseur marqueurs."""
    for frame_idx in range(reconstruction.points_3d.shape[0]):
        marker_positions = points_to_marker_tensor(model, reconstruction.points_3d[frame_idx])
        if np.any(np.isfinite(marker_positions)):
            return frame_idx, marker_positions
    return None, None


class MultiViewKinematicEKF:
    """EKF multi-vues avec etat `[q, qdot, qddot]`.

    La prediction suit un modele a acceleration constante. L'observation est
    definie dans l'espace image: les marqueurs 3D du modele sont reprojetes
    dans chaque camera, et la jacobienne combine `markersJacobian` de `biorbd`
    avec la jacobienne analytique de projection camera.
    """

    def __init__(
        self,
        model,
        calibrations: dict[str, CameraCalibration],
        pose_data: PoseData,
        reconstruction: ReconstructionResult,
        dt: float,
        measurement_noise_scale: float = 1.0,
        process_noise_scale: float = 1.0,
        min_frame_coherence_for_update: float = DEFAULT_MIN_FRAME_COHERENCE_FOR_UPDATE,
        skip_low_coherence_updates: bool = False,
        coherence_confidence_floor: float = DEFAULT_COHERENCE_CONFIDENCE_FLOOR,
        enable_dof_locking: bool = False,
        root_flight_dynamics: bool = False,
        flight_height_threshold_m: float = DEFAULT_FLIGHT_HEIGHT_THRESHOLD_M,
        flight_min_consecutive_frames: int = DEFAULT_FLIGHT_MIN_CONSECUTIVE_FRAMES,
        flip_method: str | None = None,
        flip_improvement_ratio: float = DEFAULT_FLIP_IMPROVEMENT_RATIO,
        flip_min_gain_px: float = DEFAULT_FLIP_MIN_GAIN_PX,
        flip_min_valid_keypoints: int = DEFAULT_EKF_PREDICTION_GATE_MIN_VALID_KEYPOINTS,
        flip_error_threshold_px: float = DEFAULT_EKF_PREDICTION_GATE_ERROR_THRESHOLD_PX,
        flip_error_delta_threshold_px: float = DEFAULT_EKF_PREDICTION_GATE_ERROR_DELTA_THRESHOLD_PX,
    ):
        self.model = model
        self.calibrations = calibrations
        self.pose_data = pose_data
        self.reconstruction = reconstruction
        self.dt = dt
        self.measurement_noise_scale = measurement_noise_scale
        self.process_noise_scale = process_noise_scale
        self.min_frame_coherence_for_update = min_frame_coherence_for_update
        self.skip_low_coherence_updates = skip_low_coherence_updates
        self.coherence_confidence_floor = coherence_confidence_floor
        self.enable_dof_locking = enable_dof_locking
        self.root_flight_dynamics = root_flight_dynamics
        self.flight_height_threshold_m = flight_height_threshold_m
        self.flight_min_consecutive_frames = max(1, int(flight_min_consecutive_frames))
        self.flip_method = None if flip_method is None else str(flip_method)
        self.flip_improvement_ratio = float(flip_improvement_ratio)
        self.flip_min_gain_px = float(flip_min_gain_px)
        self.flip_min_valid_keypoints = max(1, int(flip_min_valid_keypoints))
        self.flip_error_threshold_px = float(flip_error_threshold_px)
        self.flip_error_delta_threshold_px = float(flip_error_delta_threshold_px)
        self.use_prediction_flip_gate = self.flip_method == "ekf_prediction_gate"
        self.nq = model.nbQ()
        self.nx = 3 * self.nq
        self.n_root = int(model.nbRoot()) if hasattr(model, "nbRoot") else 0
        self.root_indices = np.arange(self.n_root, dtype=int)
        self.joint_indices = np.arange(self.n_root, self.nq, dtype=int)
        self.identity_x = np.eye(self.nx)
        self.marker_names = marker_name_list(model)
        self.marker_pairs = [
            (marker_idx, KP_INDEX[marker_name])
            for marker_idx, marker_name in enumerate(self.marker_names)
            if marker_name in KP_INDEX
        ]
        self.marker_pair_keypoint_indices = np.asarray([kp_idx for _, kp_idx in self.marker_pairs], dtype=int)
        self.camera_order = pose_data.camera_names
        self.camera_calibrations = [self.calibrations[cam_name] for cam_name in self.camera_order]
        self.q_names = self._make_q_names()
        self.lock_map = {name: i for i, name in enumerate(self.q_names)}
        self.locked_q_indices: set[int] = set()
        base_process_noise = np.concatenate((1e-4 * np.ones(self.nq), 5e-3 * np.ones(self.nq), 5e-2 * np.ones(self.nq)))
        self.process_noise = np.diag(base_process_noise * self.process_noise_scale)
        self.multiview_coherence: np.ndarray | None = reconstruction.multiview_coherence
        self.skip_correction_countdown = 0
        self.update_status = {
            "corrected": 0,
            "pred_only_no_measurement": 0,
            "pred_only_low_coherence": 0,
            "pred_only_cooldown": 0,
            "flip_prediction_gate_swapped": 0,
            "flip_prediction_gate_raw": 0,
            "flip_prediction_gate_raw_insufficient_support": 0,
        }
        self.profiling = {
            "predict_s": 0.0,
            "update_s": 0.0,
            "markers_s": 0.0,
            "marker_jacobians_s": 0.0,
            "assembly_s": 0.0,
            "solve_s": 0.0,
            "flip_gate_s": 0.0,
        }
        self.previous_prediction_gate_nominal_rms_px = np.full(len(self.camera_calibrations), np.nan, dtype=float)
        self.effective_confidences, self.measurement_variances = self._precompute_measurement_variances()

    def _make_q_names(self) -> list[str]:
        q_names = []
        for i_segment in range(self.model.nbSegment()):
            seg = self.model.segment(i_segment)
            seg_name = seg.name().to_string()
            dof_names = [seg.nameDof(i).to_string() for i in range(seg.nbDof())]
            for dof_name in dof_names:
                q_names.append(f"{seg_name}:{dof_name}")
        return q_names

    def _precompute_measurement_variances(self) -> tuple[np.ndarray, np.ndarray]:
        """Pre-calcule les poids et variances de mesure pour toute la sequence."""
        scores = np.asarray(self.pose_data.scores, dtype=float)
        if self.multiview_coherence is None:
            effective_confidences = scores
        else:
            coherence_cam_frame_kp = np.transpose(np.asarray(self.multiview_coherence, dtype=float), (2, 0, 1))
            blended_coherence = (
                self.coherence_confidence_floor + (1.0 - self.coherence_confidence_floor) * coherence_cam_frame_kp
            )
            effective_confidences = scores * blended_coherence
        effective_confidences = np.asarray(effective_confidences, dtype=float)
        measurement_variances = self.measurement_noise_scale * (4.0 / np.maximum(effective_confidences, 1e-3)) ** 2
        measurement_variances[effective_confidences <= 1e-3] = np.inf
        return effective_confidences, measurement_variances

    def transition_matrix(self) -> np.ndarray:
        """Matrice d'etat du modele discret a acceleration constante."""
        dt = self.dt
        eye = np.eye(self.nq)
        return np.block(
            [
                [eye, dt * eye, 0.5 * dt * dt * eye],
                [np.zeros_like(eye), eye, dt * eye],
                [np.zeros_like(eye), np.zeros_like(eye), eye],
            ]
        )

    def _is_airborne_from_previous_frame(self, frame_idx: int) -> bool:
        """Retourne vrai si le critere de vol est satisfait sur assez de frames precedentes."""
        if frame_idx <= 0:
            return False
        start_idx = max(0, frame_idx - self.flight_min_consecutive_frames)
        for previous_idx in range(start_idx, frame_idx):
            previous_points = self.reconstruction.points_3d[previous_idx]
            previous_z = previous_points[:, 2]
            valid = np.isfinite(previous_z)
            if not np.any(valid):
                return False
            if not np.all(previous_z[valid] > self.flight_height_threshold_m):
                return False
        return (frame_idx - start_idx) >= self.flight_min_consecutive_frames

    def _build_biorbd_state_vector(self, vector_type: str, values: np.ndarray):
        """Construit un vecteur d'etat `biorbd` a partir d'un tableau numpy.

        Les wrappers Python de `biorbd` ne se comportent pas tous pareil selon
        les versions. On essaie donc les formats les plus frequents avant de
        retomber sur un constructeur base sur le modele.
        """
        import biorbd

        constructor = getattr(biorbd, vector_type)
        array_1d = np.asarray(values, dtype=float)
        array_2d = array_1d[:, np.newaxis]
        for candidate in (array_1d, array_2d):
            try:
                return constructor(candidate)
            except TypeError:
                continue
        return constructor(self.model)

    @staticmethod
    def _biorbd_to_numpy(vector) -> np.ndarray:
        """Convertit un retour `biorbd` en tableau 1D numpy."""
        if hasattr(vector, "to_array"):
            return np.asarray(vector.to_array(), dtype=float).reshape(-1)
        return np.asarray(vector, dtype=float).reshape(-1)

    def _compute_root_acceleration_from_flight_dynamics(
        self, q: np.ndarray, qdot: np.ndarray, qddot_joint: np.ndarray
    ) -> np.ndarray:
        """Calcule `qddot_root` en phase aerienne.

        Par defaut, on utilise `biorbd.ForwardDynamicsFreeFloatingBase`, comme
        dans `bioptim`. Si ce chemin n'est pas disponible ou echoue pour une
        raison numerique, on retombe sur la resolution manuelle du sous-systeme
        racine:

        On resout:
            M_rr qddot_r = -(M_rj qddot_j + h_r)

        avec `np.linalg.solve` plutot qu'une inversion explicite. C'est plus
        stable numeriquement, et c'est adapte ici car `M_rr` est un sous-bloc
        principal de la matrice de masse, donc en principe symetrique definie
        positive.
        """
        if self.n_root == 0 or self.joint_indices.size == 0:
            return np.zeros(self.n_root)

        q_biorbd = self._build_biorbd_state_vector("GeneralizedCoordinates", q)
        qdot_biorbd = self._build_biorbd_state_vector("GeneralizedVelocity", qdot)

        if hasattr(self.model, "ForwardDynamicsFreeFloatingBase"):
            try:
                qddot_root = self.model.ForwardDynamicsFreeFloatingBase(
                    q_biorbd, qdot_biorbd, np.asarray(qddot_joint, dtype=float)
                )
                qddot_root = self._biorbd_to_numpy(qddot_root)
                if qddot_root.size == self.nq:
                    qddot_root = qddot_root[self.root_indices]
                if qddot_root.size == self.n_root and np.all(np.isfinite(qddot_root)):
                    return qddot_root
            except Exception:
                pass

        mass_matrix = self.model.massMatrix(q_biorbd).to_array()
        nonlinear_effects = self.model.NonLinearEffect(q_biorbd, qdot_biorbd).to_array().reshape(-1)
        M_rr = mass_matrix[np.ix_(self.root_indices, self.root_indices)]
        M_rj = mass_matrix[np.ix_(self.root_indices, self.joint_indices)]
        rhs = -(M_rj @ qddot_joint + nonlinear_effects[self.root_indices])
        return np.linalg.solve(M_rr, rhs)

    def predict(self, state: np.ndarray, covariance: np.ndarray, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Etape de prediction EKF, suivie du gel optionnel de certains DoF."""
        t_predict = time.perf_counter()
        F = self.transition_matrix()
        predicted_state = F @ state
        predicted_covariance = F @ covariance @ F.T + self.process_noise
        if self.root_flight_dynamics and self._is_airborne_from_previous_frame(frame_idx):
            q_prev = state[: self.nq]
            qdot_prev = state[self.nq : 2 * self.nq]
            qddot_joint = predicted_state[2 * self.nq + self.n_root : 3 * self.nq]
            qddot_root = self._compute_root_acceleration_from_flight_dynamics(q_prev, qdot_prev, qddot_joint)
            predicted_state[2 * self.nq : 2 * self.nq + self.n_root] = qddot_root
            predicted_state[self.nq : self.nq + self.n_root] = qdot_prev[self.root_indices] + self.dt * qddot_root
            predicted_state[: self.n_root] = (
                q_prev[self.root_indices]
                + self.dt * qdot_prev[self.root_indices]
                + 0.5 * self.dt * self.dt * qddot_root
            )
        self._update_locked_dofs(predicted_state)
        self._apply_lock_constraints(predicted_state, predicted_covariance)
        self.profiling["predict_s"] += time.perf_counter() - t_predict
        return predicted_state, predicted_covariance

    def _update_locked_dofs(self, predicted_state: np.ndarray) -> None:
        """Active des verrous de DoF pres de configurations jugees singulieres.

        Ici, a titre d'exemple, on gèle la rotation longitudinale de l'avant-
        bras quand le coude est proche de l'extension.
        """
        self.locked_q_indices.clear()
        if not self.enable_dof_locking:
            return
        deg5 = np.deg2rad(5.0)
        for side in ("LEFT", "RIGHT"):
            flex_name = f"{side}_FOREARM:RotY"
            rot_name = f"{side}_FOREARM:RotZ"
            if flex_name in self.lock_map and rot_name in self.lock_map:
                flex_idx = self.lock_map[flex_name]
                rot_idx = self.lock_map[rot_name]
                if abs(predicted_state[flex_idx]) <= deg5:
                    self.locked_q_indices.add(rot_idx)

    def _apply_lock_constraints(self, state: np.ndarray, covariance: np.ndarray) -> None:
        """Impose un DoF verrouille en annulant sa variance et ses derivees."""
        for q_idx in self.locked_q_indices:
            for block in (0, self.nq, 2 * self.nq):
                full_idx = q_idx + block
                if block > 0:
                    state[full_idx] = 0.0
                covariance[full_idx, :] = 0.0
                covariance[:, full_idx] = 0.0
                covariance[full_idx, full_idx] = 1e-9

    def update(
        self, predicted_state: np.ndarray, predicted_covariance: np.ndarray, frame_idx: int
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Etape de correction EKF dans l'espace image 2D.

        Le score de confiance 2D est converti en bruit de mesure: une detection
        peu fiable est gardee mais influence moins fortement l'estimation.
        """
        t_update = time.perf_counter()
        if self.skip_correction_countdown > 0:
            self.skip_correction_countdown -= 1
            self.update_status["pred_only_cooldown"] += 1
            self.profiling["update_s"] += time.perf_counter() - t_update
            return predicted_state, predicted_covariance, "pred_only_cooldown"

        if self.skip_low_coherence_updates and self.multiview_coherence is not None:
            frame_coherence = self.multiview_coherence[frame_idx]
            valid_frame_coherence = frame_coherence[frame_coherence > 0]
            mean_frame_coherence = float(np.mean(valid_frame_coherence)) if valid_frame_coherence.size else 0.0
            if mean_frame_coherence < self.min_frame_coherence_for_update:
                # On saute la correction pour cette frame et la suivante afin
                # d'eviter qu'une mauvaise coherence multivue injecte du bruit.
                self.skip_correction_countdown = 1
                self.update_status["pred_only_low_coherence"] += 1
                self.profiling["update_s"] += time.perf_counter() - t_update
                return predicted_state, predicted_covariance, "pred_only_low_coherence"

        q = predicted_state[: self.nq]
        t_markers = time.perf_counter()
        marker_positions_all = self.model.markers(q)
        self.profiling["markers_s"] += time.perf_counter() - t_markers
        t_jac = time.perf_counter()
        marker_jacobians_all = self.model.markersJacobian(q)
        self.profiling["marker_jacobians_s"] += time.perf_counter() - t_jac

        # On convertit les sorties `biorbd` une seule fois par frame puis on
        # les reutilise pour toutes les cameras. Sans cela, les memes
        # `to_array()` sont executes encore et encore pour chaque vue.
        marker_points = []
        marker_jacobians = []
        for marker_idx, _ in self.marker_pairs:
            marker_points.append(marker_positions_all[marker_idx].to_array())
            marker_jacobians.append(marker_jacobians_all[marker_idx].to_array())
        marker_points_array = np.asarray(marker_points, dtype=float)
        marker_jacobians_array = np.asarray(marker_jacobians, dtype=float)
        finite_marker_points = np.all(np.isfinite(marker_points_array), axis=1)

        measurement_blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        locked_indices = tuple(self.locked_q_indices)
        locked_q_columns = np.asarray(locked_indices, dtype=int) if locked_indices else np.empty(0, dtype=int)
        t_assembly = time.perf_counter()

        for cam_idx, calibration in enumerate(self.camera_calibrations):
            frame_keypoints = self.pose_data.keypoints[cam_idx, frame_idx]
            frame_variances = self.measurement_variances[cam_idx, frame_idx]
            valid_keypoints = (
                np.all(np.isfinite(frame_keypoints), axis=1) & np.isfinite(frame_variances) & (frame_variances < np.inf)
            )
            if self.use_prediction_flip_gate:
                swapped_keypoints = swap_left_right_keypoints(frame_keypoints)
                swapped_variances = swap_left_right_keypoint_values(frame_variances)
                valid_keypoints |= (
                    np.all(np.isfinite(swapped_keypoints), axis=1)
                    & np.isfinite(swapped_variances)
                    & (swapped_variances < np.inf)
                )
            if not np.any(valid_keypoints):
                continue
            valid_pairs = finite_marker_points & valid_keypoints[self.marker_pair_keypoint_indices]
            if not np.any(valid_pairs):
                continue
            pair_indices = np.flatnonzero(valid_pairs)
            keypoint_indices = self.marker_pair_keypoint_indices[pair_indices]
            projected_uv, projected_jac = calibration.project_points_and_jacobians(marker_points_array[pair_indices])
            H_q_blocks = np.einsum("mab,mbq->maq", projected_jac, marker_jacobians_array[pair_indices], optimize=True)
            if locked_q_columns.size:
                H_q_blocks = np.array(H_q_blocks, copy=True)
                H_q_blocks[:, :, locked_q_columns] = 0.0
            finite_pairs = np.all(np.isfinite(projected_uv), axis=1) & np.all(
                np.isfinite(H_q_blocks.reshape(H_q_blocks.shape[0], -1)), axis=1
            )
            if not np.any(finite_pairs):
                continue
            keypoint_indices = keypoint_indices[finite_pairs]
            projected_uv = projected_uv[finite_pairs]
            H_q_blocks = H_q_blocks[finite_pairs]
            gate_t0 = time.perf_counter()
            if self.use_prediction_flip_gate:
                selected_mask, selected_points, selected_variances, gate_diagnostics = (
                    choose_ekf_prediction_gate_measurements(
                        frame_keypoints,
                        frame_variances,
                        projected_uv,
                        keypoint_indices,
                        improvement_ratio=self.flip_improvement_ratio,
                        min_gain_px=self.flip_min_gain_px,
                        min_valid_keypoints=self.flip_min_valid_keypoints,
                        activation_error_threshold_px=self.flip_error_threshold_px,
                        activation_error_delta_threshold_px=self.flip_error_delta_threshold_px,
                        previous_nominal_rms_px=self.previous_prediction_gate_nominal_rms_px[cam_idx],
                    )
                )
                self.profiling["flip_gate_s"] += time.perf_counter() - gate_t0
                current_nominal_rms_px = float(gate_diagnostics.get("nominal_rms_px", float("nan")))
                if np.isfinite(current_nominal_rms_px):
                    self.previous_prediction_gate_nominal_rms_px[cam_idx] = current_nominal_rms_px
                decision_label = str(gate_diagnostics.get("decision", "raw"))
                status_key = f"flip_prediction_gate_{decision_label}"
                if status_key in self.update_status:
                    self.update_status[status_key] += 1
                elif bool(gate_diagnostics.get("used_swapped")):
                    self.update_status["flip_prediction_gate_swapped"] += 1
                else:
                    self.update_status["flip_prediction_gate_raw"] += 1
            else:
                selected_mask = np.all(np.isfinite(frame_keypoints[keypoint_indices]), axis=1) & np.isfinite(
                    frame_variances[keypoint_indices]
                )
                selected_mask &= frame_variances[keypoint_indices] < np.inf
                selected_points = frame_keypoints[keypoint_indices]
                selected_variances = frame_variances[keypoint_indices]
            if not np.any(selected_mask):
                continue
            measurement_blocks.append(
                (
                    selected_points[selected_mask].reshape(-1),
                    projected_uv[selected_mask].reshape(-1),
                    H_q_blocks[selected_mask].reshape(-1, self.nq),
                    np.repeat(selected_variances[selected_mask], 2).astype(float, copy=False),
                )
            )
        self.profiling["assembly_s"] += time.perf_counter() - t_assembly

        if not measurement_blocks:
            self.update_status["pred_only_no_measurement"] += 1
            self.profiling["update_s"] += time.perf_counter() - t_update
            return predicted_state, predicted_covariance, "pred_only_no_measurement"

        t_solve = time.perf_counter()
        update_result = apply_measurement_update_sequential(
            predicted_state=predicted_state,
            predicted_covariance=predicted_covariance,
            measurement_blocks=measurement_blocks,
            nq=self.nq,
            identity_x=self.identity_x,
        )
        if update_result is None:
            self.update_status["pred_only_no_measurement"] += 1
            self.profiling["solve_s"] += time.perf_counter() - t_solve
            self.profiling["update_s"] += time.perf_counter() - t_update
            return predicted_state, predicted_covariance, "pred_only_no_measurement"
        updated_state, updated_covariance = update_result
        self.profiling["solve_s"] += time.perf_counter() - t_solve
        self._apply_lock_constraints(updated_state, updated_covariance)
        self.update_status["corrected"] += 1
        self.profiling["update_s"] += time.perf_counter() - t_update
        return updated_state, updated_covariance, "corrected"


def initial_state_from_triangulation(model, reconstruction: ReconstructionResult) -> np.ndarray:
    """Initialise l'etat `[q, qdot, qddot]` a partir d'une IK sur les marqueurs 3D."""
    import biorbd

    _frame_idx, marker_positions = first_valid_marker_tensor_from_reconstruction(model, reconstruction)
    if marker_positions is None:
        return np.zeros(3 * model.nbQ())
    try:
        q0 = np.asarray(biorbd.InverseKinematics(model, marker_positions).solve()).reshape(-1)
    except Exception:
        q0 = np.zeros(model.nbQ())
    return np.concatenate((q0, np.zeros(model.nbQ()), np.zeros(model.nbQ())))


def q_names_from_model(model) -> list[str]:
    """Retourne les noms de DoF sous la forme `SEGMENT:DoF`."""
    return [
        f"{model.segment(i_seg).name().to_string()}:{model.segment(i_seg).nameDof(i_dof).to_string()}"
        for i_seg in range(model.nbSegment())
        for i_dof in range(model.segment(i_seg).nbDof())
    ]


def _segment_rotation_sequence_and_offsets(segment_dof_names: list[str]) -> tuple[str | None, np.ndarray]:
    """Infer a contiguous Euler rotation block from one segment DoF list."""

    rotation_offsets: list[int] = []
    rotation_axes: list[str] = []
    for dof_idx, dof_name in enumerate(segment_dof_names):
        dof_name = str(dof_name)
        if dof_name.startswith("Rot") and len(dof_name) == 4 and dof_name[-1] in {"X", "Y", "Z"}:
            rotation_offsets.append(dof_idx)
            rotation_axes.append(dof_name[-1].lower())
    if not rotation_offsets:
        return None, np.empty(0, dtype=int)
    offsets_array = np.asarray(rotation_offsets, dtype=int)
    expected_offsets = np.arange(offsets_array[0], offsets_array[0] + offsets_array.size, dtype=int)
    if not np.array_equal(offsets_array, expected_offsets):
        return None, np.empty(0, dtype=int)
    return "".join(rotation_axes), offsets_array


def canonicalize_model_q_rotation_branches(model, q_values: np.ndarray) -> np.ndarray:
    """Re-extract segment Euler blocks on a canonical branch.

    The EKF bootstrap can converge to numerically large Euler representations
    that still encode the same joint orientation. We rebuild each segment
    rotation matrix and re-extract the angles with the same sequence so the
    warm start stays equivalent in 3D while returning to more physiological
    numeric ranges.
    """

    q_array = np.asarray(q_values, dtype=float).reshape(-1)
    canonical_q = np.array(q_array, copy=True)
    q_cursor = 0
    for segment_idx in range(model.nbSegment()):
        segment = model.segment(segment_idx)
        segment_dof_names = [segment.nameDof(dof_idx).to_string() for dof_idx in range(segment.nbDof())]
        sequence, local_rotation_offsets = _segment_rotation_sequence_and_offsets(segment_dof_names)
        if sequence is not None and local_rotation_offsets.size:
            block_indices = q_cursor + local_rotation_offsets
            block_values = canonical_q[block_indices]
            if np.all(np.isfinite(block_values)):
                if len(sequence) == 1:
                    angle_value = float(block_values[0])
                    canonical_q[block_indices[0]] = math.atan2(math.sin(angle_value), math.cos(angle_value))
                else:
                    scipy_sequence = sequence.upper()
                    if len(sequence) == 2:
                        extra_axis = next(axis for axis in "xyz" if axis not in set(sequence))
                        extended_sequence = f"{sequence}{extra_axis}"
                        scipy_extended_sequence = extended_sequence.upper()
                        extended_values = np.concatenate((block_values, np.zeros(1, dtype=float)))
                        rotation_matrix = Rotation.from_euler(
                            scipy_extended_sequence, extended_values, degrees=False
                        ).as_matrix()
                        canonical_q[block_indices] = Rotation.from_matrix(rotation_matrix).as_euler(
                            scipy_extended_sequence,
                            degrees=False,
                        )[:2]
                    else:
                        rotation_matrix = Rotation.from_euler(scipy_sequence, block_values, degrees=False).as_matrix()
                        canonical_q[block_indices] = Rotation.from_matrix(rotation_matrix).as_euler(
                            scipy_sequence,
                            degrees=False,
                        )
        q_cursor += segment.nbDof()
    return canonical_q


def first_valid_root_translation_from_triangulation(
    reconstruction: ReconstructionResult,
) -> tuple[int | None, np.ndarray | None]:
    """Estime `TransX/Y/Z` de la racine au milieu des hanches triangulees."""
    left_idx = KP_INDEX["left_hip"]
    right_idx = KP_INDEX["right_hip"]
    for frame_idx in range(reconstruction.points_3d.shape[0]):
        left_hip = reconstruction.points_3d[frame_idx, left_idx]
        right_hip = reconstruction.points_3d[frame_idx, right_idx]
        if np.all(np.isfinite(left_hip)) and np.all(np.isfinite(right_hip)):
            return frame_idx, 0.5 * (left_hip + right_hip)
    return None, None


def root_translation_from_triangulation_frame(
    reconstruction: ReconstructionResult,
    frame_idx: int,
) -> np.ndarray | None:
    """Estimate the root translation from one specific triangulated frame."""

    if not hasattr(reconstruction, "points_3d"):
        return None
    frame_idx = int(frame_idx)
    if frame_idx < 0 or frame_idx >= reconstruction.points_3d.shape[0]:
        return None
    left_idx = KP_INDEX["left_hip"]
    right_idx = KP_INDEX["right_hip"]
    left_hip = reconstruction.points_3d[frame_idx, left_idx]
    right_hip = reconstruction.points_3d[frame_idx, right_idx]
    if np.all(np.isfinite(left_hip)) and np.all(np.isfinite(right_hip)):
        return 0.5 * (left_hip + right_hip)
    return None


def first_valid_root_pose_from_triangulation(
    reconstruction: ReconstructionResult,
) -> tuple[int | None, np.ndarray | None]:
    """Estime les 6 DoF de la racine a partir du tronc triangule."""
    root_q = compute_trunk_dofs_from_points(reconstruction.points_3d, unwrap_rotations=False)
    for frame_idx in range(root_q.shape[0]):
        if np.all(np.isfinite(root_q[frame_idx])):
            return frame_idx, np.asarray(root_q[frame_idx], dtype=float)
    return None, None


def apply_root_translation_guess_to_state(model, state: np.ndarray, translation_xyz: np.ndarray | None) -> np.ndarray:
    """Injecte une translation racine estimee dans un etat `[q, qdot, qddot]`."""
    if translation_xyz is None or not np.all(np.isfinite(translation_xyz)):
        return np.array(state, copy=True)
    q_names = q_names_from_model(model)
    updated_state = np.array(state, copy=True)
    for axis_idx, axis_name in enumerate(("TransX", "TransY", "TransZ")):
        target_name = f"TRUNK:{axis_name}"
        if target_name in q_names:
            updated_state[q_names.index(target_name)] = float(translation_xyz[axis_idx])
    return updated_state


def apply_root_pose_guess_to_state(model, state: np.ndarray, root_pose: np.ndarray | None) -> np.ndarray:
    """Injecte une pose racine `[TransX, TransY, TransZ, RotY, RotX, RotZ]` dans l'etat."""
    if root_pose is None:
        return np.array(state, copy=True)
    root_pose = np.asarray(root_pose, dtype=float).reshape(-1)
    if root_pose.size < 6 or not np.all(np.isfinite(root_pose[:6])):
        return np.array(state, copy=True)
    q_names = q_names_from_model(model)
    updated_state = np.array(state, copy=True)
    for value, target_name in zip(
        root_pose[:6], ("TRUNK:TransX", "TRUNK:TransY", "TRUNK:TransZ", "TRUNK:RotY", "TRUNK:RotX", "TRUNK:RotZ")
    ):
        if target_name in q_names:
            updated_state[q_names.index(target_name)] = float(value)
    return updated_state


def align_root_translation_guess_to_frame_zero(
    model,
    state: np.ndarray,
    reconstruction: ReconstructionResult,
    *,
    source_frame_idx: int | None,
) -> np.ndarray:
    """Re-anchor root translation to frame 0 when the hips are available there.

    The biorbd warm-start may be solved from the first triangulated frame with
    enough markers, which is not always the first frame of the sequence. When
    the athlete already drifts horizontally, this produces a transient bias on
    `TRUNK:TransX/TransY` during the first reconstructed frames. If frame 0 has
    valid hips, we shift the guessed root translation back to that frame.
    """

    if source_frame_idx is None:
        return np.array(state, copy=True)
    source_translation = root_translation_from_triangulation_frame(reconstruction, int(source_frame_idx))
    target_translation = root_translation_from_triangulation_frame(reconstruction, 0)
    if source_translation is None or target_translation is None:
        return np.array(state, copy=True)
    return apply_root_translation_guess_to_state(model, state, target_translation)


def compute_biorbd_kalman_initial_state(
    model,
    reconstruction: ReconstructionResult,
    method: str = DEFAULT_BIORBD_KALMAN_INIT_METHOD,
) -> tuple[np.ndarray | None, dict[str, object]]:
    """Construit l'etat initial du Kalman marqueurs `biorbd`."""
    if method == "none":
        return None, {"method": "none", "used_init_state": False}

    state = initial_state_from_triangulation(model, reconstruction)
    diagnostics: dict[str, object] = {
        "method": str(method),
        "used_init_state": True,
        "used_triangulation_ik": bool(method in {"triangulation_ik", "triangulation_ik_root_translation"}),
        "used_root_translation_mid_hips": False,
        "bootstrap_frame_idx": None,
    }

    if method == "triangulation_ik":
        aligned_state = align_root_translation_guess_to_frame_zero(
            model,
            state,
            reconstruction,
            source_frame_idx=diagnostics["bootstrap_frame_idx"],
        )
        diagnostics["aligned_root_translation_to_frame_zero"] = not np.allclose(aligned_state, state, equal_nan=True)
        return aligned_state, diagnostics
    if method == "triangulation_ik_root_translation":
        frame_idx, root_translation = first_valid_root_translation_from_triangulation(reconstruction)
        diagnostics["bootstrap_frame_idx"] = None if frame_idx is None else int(frame_idx)
        diagnostics["used_root_translation_mid_hips"] = bool(root_translation is not None)
        updated_state = apply_root_translation_guess_to_state(model, state, root_translation)
        aligned_state = align_root_translation_guess_to_frame_zero(
            model,
            updated_state,
            reconstruction,
            source_frame_idx=diagnostics["bootstrap_frame_idx"],
        )
        diagnostics["aligned_root_translation_to_frame_zero"] = not np.allclose(
            aligned_state, updated_state, equal_nan=True
        )
        return aligned_state, diagnostics
    if method == "root_pose_zero_rest":
        zero_state = np.zeros(3 * model.nbQ())
        frame_idx, root_pose = first_valid_root_pose_from_triangulation(reconstruction)
        diagnostics["bootstrap_frame_idx"] = None if frame_idx is None else int(frame_idx)
        diagnostics["used_triangulation_ik"] = False
        diagnostics["used_root_translation_mid_hips"] = bool(root_pose is not None)
        diagnostics["used_root_pose_guess"] = bool(root_pose is not None)
        updated_state = apply_root_pose_guess_to_state(model, zero_state, root_pose)
        aligned_state = align_root_translation_guess_to_frame_zero(
            model,
            updated_state,
            reconstruction,
            source_frame_idx=diagnostics["bootstrap_frame_idx"],
        )
        diagnostics["aligned_root_translation_to_frame_zero"] = not np.allclose(
            aligned_state, updated_state, equal_nan=True
        )
        return aligned_state, diagnostics
    if method == "root_translation_zero_rest":
        zero_state = np.zeros(3 * model.nbQ())
        frame_idx, root_translation = first_valid_root_translation_from_triangulation(reconstruction)
        diagnostics["bootstrap_frame_idx"] = None if frame_idx is None else int(frame_idx)
        diagnostics["used_triangulation_ik"] = False
        diagnostics["used_root_translation_mid_hips"] = bool(root_translation is not None)
        updated_state = apply_root_translation_guess_to_state(model, zero_state, root_translation)
        aligned_state = align_root_translation_guess_to_frame_zero(
            model,
            updated_state,
            reconstruction,
            source_frame_idx=diagnostics["bootstrap_frame_idx"],
        )
        diagnostics["aligned_root_translation_to_frame_zero"] = not np.allclose(
            aligned_state, updated_state, equal_nan=True
        )
        return aligned_state, diagnostics
    raise ValueError(f"Unsupported biorbd kalman init method: {method}")


def initial_state_from_ekf_bootstrap(
    model,
    calibrations: dict[str, CameraCalibration],
    pose_data: PoseData,
    reconstruction: ReconstructionResult,
    fps: float,
    measurement_noise_scale: float = 1.0,
    process_noise_scale: float = 1.0,
    min_frame_coherence_for_update: float = DEFAULT_MIN_FRAME_COHERENCE_FOR_UPDATE,
    skip_low_coherence_updates: bool = False,
    coherence_confidence_floor: float = DEFAULT_COHERENCE_CONFIDENCE_FLOOR,
    enable_dof_locking: bool = False,
    passes: int = DEFAULT_EKF2D_BOOTSTRAP_PASSES,
    tolerance: float = 1e-6,
    initial_state: np.ndarray | None = None,
    flip_method: str | None = None,
    flip_improvement_ratio: float = DEFAULT_FLIP_IMPROVEMENT_RATIO,
    flip_min_gain_px: float = DEFAULT_FLIP_MIN_GAIN_PX,
    flip_error_threshold_px: float = DEFAULT_EKF_PREDICTION_GATE_ERROR_THRESHOLD_PX,
    flip_error_delta_threshold_px: float = DEFAULT_EKF_PREDICTION_GATE_ERROR_DELTA_THRESHOLD_PX,
) -> tuple[np.ndarray, dict[str, object]]:
    """Affine `q0` par corrections EKF repetees sur une seule frame.

    A chaque passe, on reutilise uniquement `q` comme graine et on remet
    `qdot`/`qddot` a zero. Cela donne un bootstrap 2D plus proche de la
    formulation du filtre que l'IK 3D pure, tout en restant tres local.
    """
    ik_state = (
        np.array(initial_state, copy=True)
        if initial_state is not None
        else initial_state_from_triangulation(model, reconstruction)
    )
    ik_state[: model.nbQ()] = canonicalize_model_q_rotation_branches(model, ik_state[: model.nbQ()])
    frame_idx, _marker_positions = first_valid_marker_tensor_from_reconstruction(model, reconstruction)
    diagnostics: dict[str, object] = {
        "method": "ekf_bootstrap",
        "fallback_method": "triangulation_ik",
        "bootstrap_frame_idx": None if frame_idx is None else int(frame_idx),
        "requested_passes": int(max(1, passes)),
        "completed_passes": 0,
        "converged": False,
        "update_statuses": [],
        "q_delta_norms": [],
        "used_fallback": False,
    }
    if frame_idx is None:
        diagnostics["used_fallback"] = True
        diagnostics["reason"] = "no_valid_triangulated_frame"
        return ik_state, diagnostics

    ekf = MultiViewKinematicEKF(
        model=model,
        calibrations=calibrations,
        pose_data=pose_data,
        reconstruction=reconstruction,
        dt=1.0 / fps,
        measurement_noise_scale=measurement_noise_scale,
        process_noise_scale=process_noise_scale,
        min_frame_coherence_for_update=min_frame_coherence_for_update,
        skip_low_coherence_updates=skip_low_coherence_updates,
        coherence_confidence_floor=coherence_confidence_floor,
        enable_dof_locking=enable_dof_locking,
        root_flight_dynamics=False,
        flight_height_threshold_m=DEFAULT_FLIGHT_HEIGHT_THRESHOLD_M,
        flight_min_consecutive_frames=DEFAULT_FLIGHT_MIN_CONSECUTIVE_FRAMES,
        flip_method=flip_method,
        flip_improvement_ratio=flip_improvement_ratio,
        flip_min_gain_px=flip_min_gain_px,
        flip_error_threshold_px=flip_error_threshold_px,
        flip_error_delta_threshold_px=flip_error_delta_threshold_px,
    )
    state = np.array(ik_state, copy=True)
    base_covariance = np.eye(ekf.nx) * 1e-2
    passes = max(1, int(passes))
    for _pass_idx in range(passes):
        seed_q = canonicalize_model_q_rotation_branches(model, state[: ekf.nq])
        seed_state = np.concatenate((seed_q, np.zeros(ekf.nq), np.zeros(ekf.nq)))
        covariance = np.array(base_covariance, copy=True)
        ekf.skip_correction_countdown = 0
        predicted_state, predicted_covariance = ekf.predict(seed_state, covariance, frame_idx)
        corrected_state, _corrected_covariance, update_status = ekf.update(
            predicted_state, predicted_covariance, frame_idx
        )
        diagnostics["update_statuses"].append(str(update_status))
        diagnostics["completed_passes"] = int(diagnostics["completed_passes"]) + 1
        if update_status != "corrected" or not np.all(np.isfinite(corrected_state)):
            diagnostics["reason"] = f"bootstrap_{update_status}"
            if int(diagnostics["completed_passes"]) <= 1:
                diagnostics["used_fallback"] = True
                return ik_state, diagnostics
            diagnostics["used_fallback"] = False
            diagnostics["stopped_early"] = True
            diagnostics["final_q_norm"] = float(np.linalg.norm(state[: ekf.nq]))
            return state, diagnostics
        corrected_q = canonicalize_model_q_rotation_branches(model, corrected_state[: ekf.nq])
        q_delta_norm = float(np.linalg.norm(corrected_q - seed_state[: ekf.nq]))
        diagnostics["q_delta_norms"].append(q_delta_norm)
        state = np.concatenate((corrected_q, np.zeros(ekf.nq), np.zeros(ekf.nq)))
        if q_delta_norm <= float(tolerance):
            diagnostics["converged"] = True
            break

    diagnostics["final_q_norm"] = float(np.linalg.norm(state[: ekf.nq]))
    return state, diagnostics


def initial_state_from_root_pose_bootstrap(
    model,
    calibrations: dict[str, CameraCalibration],
    pose_data: PoseData,
    reconstruction: ReconstructionResult,
    fps: float,
    measurement_noise_scale: float = 1.0,
    process_noise_scale: float = 1.0,
    min_frame_coherence_for_update: float = DEFAULT_MIN_FRAME_COHERENCE_FOR_UPDATE,
    skip_low_coherence_updates: bool = False,
    coherence_confidence_floor: float = DEFAULT_COHERENCE_CONFIDENCE_FLOOR,
    enable_dof_locking: bool = False,
    passes: int = DEFAULT_EKF2D_BOOTSTRAP_PASSES,
    flip_method: str | None = None,
    flip_improvement_ratio: float = DEFAULT_FLIP_IMPROVEMENT_RATIO,
    flip_min_gain_px: float = DEFAULT_FLIP_MIN_GAIN_PX,
    flip_error_threshold_px: float = DEFAULT_EKF_PREDICTION_GATE_ERROR_THRESHOLD_PX,
    flip_error_delta_threshold_px: float = DEFAULT_EKF_PREDICTION_GATE_ERROR_DELTA_THRESHOLD_PX,
) -> tuple[np.ndarray, dict[str, object]]:
    """Initialise l'EKF 2D depuis une pose racine geometrique, puis bootstrappe."""
    zero_state = np.zeros(3 * model.nbQ(), dtype=float)
    frame_idx, root_pose = first_valid_root_pose_from_triangulation(reconstruction)
    diagnostics_seed = {
        "method": "root_pose_bootstrap",
        "root_pose_frame_idx": None if frame_idx is None else int(frame_idx),
        "used_root_pose_guess": bool(root_pose is not None),
        "used_fallback": False,
    }
    if root_pose is None:
        diagnostics_seed["used_fallback"] = True
        diagnostics_seed["fallback_reason"] = "no_valid_root_pose"
        return initial_state_from_ekf_bootstrap(
            model=model,
            calibrations=calibrations,
            pose_data=pose_data,
            reconstruction=reconstruction,
            fps=fps,
            measurement_noise_scale=measurement_noise_scale,
            process_noise_scale=process_noise_scale,
            min_frame_coherence_for_update=min_frame_coherence_for_update,
            skip_low_coherence_updates=skip_low_coherence_updates,
            coherence_confidence_floor=coherence_confidence_floor,
            enable_dof_locking=enable_dof_locking,
            passes=passes,
            flip_method=flip_method,
            flip_improvement_ratio=flip_improvement_ratio,
            flip_min_gain_px=flip_min_gain_px,
            flip_error_threshold_px=flip_error_threshold_px,
            flip_error_delta_threshold_px=flip_error_delta_threshold_px,
        )

    root_seed_state = apply_root_pose_guess_to_state(model, zero_state, root_pose)
    state, diagnostics = initial_state_from_ekf_bootstrap(
        model=model,
        calibrations=calibrations,
        pose_data=pose_data,
        reconstruction=reconstruction,
        fps=fps,
        measurement_noise_scale=measurement_noise_scale,
        process_noise_scale=process_noise_scale,
        min_frame_coherence_for_update=min_frame_coherence_for_update,
        skip_low_coherence_updates=skip_low_coherence_updates,
        coherence_confidence_floor=coherence_confidence_floor,
        enable_dof_locking=enable_dof_locking,
        passes=passes,
        initial_state=root_seed_state,
        flip_method=flip_method,
        flip_improvement_ratio=flip_improvement_ratio,
        flip_min_gain_px=flip_min_gain_px,
        flip_error_threshold_px=flip_error_threshold_px,
        flip_error_delta_threshold_px=flip_error_delta_threshold_px,
    )
    diagnostics = dict(diagnostics)
    diagnostics["method"] = "root_pose_bootstrap"
    diagnostics["root_pose_frame_idx"] = None if frame_idx is None else int(frame_idx)
    diagnostics["used_root_pose_guess"] = True
    diagnostics.setdefault("fallback_method", "triangulation_ik")
    return state, diagnostics


def compute_ekf2d_initial_state(
    model,
    calibrations: dict[str, CameraCalibration],
    pose_data: PoseData,
    reconstruction: ReconstructionResult,
    fps: float,
    measurement_noise_scale: float,
    process_noise_scale: float,
    min_frame_coherence_for_update: float,
    skip_low_coherence_updates: bool,
    coherence_confidence_floor: float,
    enable_dof_locking: bool,
    method: str = DEFAULT_EKF2D_INITIAL_STATE_METHOD,
    bootstrap_passes: int = DEFAULT_EKF2D_BOOTSTRAP_PASSES,
    flip_method: str | None = None,
    flip_improvement_ratio: float = DEFAULT_FLIP_IMPROVEMENT_RATIO,
    flip_min_gain_px: float = DEFAULT_FLIP_MIN_GAIN_PX,
    flip_error_threshold_px: float = DEFAULT_EKF_PREDICTION_GATE_ERROR_THRESHOLD_PX,
    flip_error_delta_threshold_px: float = DEFAULT_EKF_PREDICTION_GATE_ERROR_DELTA_THRESHOLD_PX,
) -> tuple[np.ndarray, dict[str, object]]:
    """Selectionne et calcule l'etat initial des EKF 2D."""
    if method == "triangulation_ik":
        state = initial_state_from_triangulation(model, reconstruction)
        frame_idx, _marker_positions = first_valid_marker_tensor_from_reconstruction(model, reconstruction)
        return state, {
            "method": "triangulation_ik",
            "bootstrap_frame_idx": None if frame_idx is None else int(frame_idx),
            "requested_passes": 0,
            "completed_passes": 0,
            "converged": False,
            "update_statuses": [],
            "q_delta_norms": [],
            "used_fallback": False,
        }
    if method == "ekf_bootstrap":
        return initial_state_from_ekf_bootstrap(
            model=model,
            calibrations=calibrations,
            pose_data=pose_data,
            reconstruction=reconstruction,
            fps=fps,
            measurement_noise_scale=measurement_noise_scale,
            process_noise_scale=process_noise_scale,
            min_frame_coherence_for_update=min_frame_coherence_for_update,
            skip_low_coherence_updates=skip_low_coherence_updates,
            coherence_confidence_floor=coherence_confidence_floor,
            enable_dof_locking=enable_dof_locking,
            passes=bootstrap_passes,
            flip_method=flip_method,
            flip_improvement_ratio=flip_improvement_ratio,
            flip_min_gain_px=flip_min_gain_px,
            flip_error_threshold_px=flip_error_threshold_px,
            flip_error_delta_threshold_px=flip_error_delta_threshold_px,
        )
    if method == "root_pose_bootstrap":
        return initial_state_from_root_pose_bootstrap(
            model=model,
            calibrations=calibrations,
            pose_data=pose_data,
            reconstruction=reconstruction,
            fps=fps,
            measurement_noise_scale=measurement_noise_scale,
            process_noise_scale=process_noise_scale,
            min_frame_coherence_for_update=min_frame_coherence_for_update,
            skip_low_coherence_updates=skip_low_coherence_updates,
            coherence_confidence_floor=coherence_confidence_floor,
            enable_dof_locking=enable_dof_locking,
            passes=bootstrap_passes,
            flip_method=flip_method,
            flip_improvement_ratio=flip_improvement_ratio,
            flip_min_gain_px=flip_min_gain_px,
            flip_error_threshold_px=flip_error_threshold_px,
            flip_error_delta_threshold_px=flip_error_delta_threshold_px,
        )
    raise ValueError(f"Unsupported ekf2d initial state method: {method}")


def run_ekf(
    biomod_path: Path | None,
    calibrations: dict[str, CameraCalibration],
    pose_data: PoseData,
    reconstruction: ReconstructionResult,
    fps: float,
    measurement_noise_scale: float = 1.0,
    process_noise_scale: float = 1.0,
    min_frame_coherence_for_update: float = DEFAULT_MIN_FRAME_COHERENCE_FOR_UPDATE,
    skip_low_coherence_updates: bool = False,
    coherence_confidence_floor: float = DEFAULT_COHERENCE_CONFIDENCE_FLOOR,
    enable_dof_locking: bool = False,
    root_flight_dynamics: bool = False,
    flight_height_threshold_m: float = DEFAULT_FLIGHT_HEIGHT_THRESHOLD_M,
    flight_min_consecutive_frames: int = DEFAULT_FLIGHT_MIN_CONSECUTIVE_FRAMES,
    unwrap_root: bool = True,
    debug_label: str | None = None,
    debug_console: bool = False,
    initial_state: np.ndarray | None = None,
    model=None,
    flip_method: str | None = None,
    flip_improvement_ratio: float = DEFAULT_FLIP_IMPROVEMENT_RATIO,
    flip_min_gain_px: float = DEFAULT_FLIP_MIN_GAIN_PX,
    flip_error_threshold_px: float = DEFAULT_EKF_PREDICTION_GATE_ERROR_THRESHOLD_PX,
    flip_error_delta_threshold_px: float = DEFAULT_EKF_PREDICTION_GATE_ERROR_DELTA_THRESHOLD_PX,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Execute l'EKF multi-vues sur toute la sequence.

    Retourne les etats estimes ainsi qu'un petit dictionnaire de timings pour
    distinguer l'initialisation du cout de boucle pure.
    """
    import biorbd

    timings = {"init_s": 0.0, "loop_s": 0.0}
    t_init = time.perf_counter()
    if model is None:
        if biomod_path is None:
            raise ValueError("Either `biomod_path` or `model` must be provided to run_ekf.")
        model = biorbd.Model(str(biomod_path))
    ekf = MultiViewKinematicEKF(
        model=model,
        calibrations=calibrations,
        pose_data=pose_data,
        reconstruction=reconstruction,
        dt=1.0 / fps,
        measurement_noise_scale=measurement_noise_scale,
        process_noise_scale=process_noise_scale,
        min_frame_coherence_for_update=min_frame_coherence_for_update,
        skip_low_coherence_updates=skip_low_coherence_updates,
        coherence_confidence_floor=coherence_confidence_floor,
        enable_dof_locking=enable_dof_locking,
        root_flight_dynamics=root_flight_dynamics,
        flight_height_threshold_m=flight_height_threshold_m,
        flight_min_consecutive_frames=flight_min_consecutive_frames,
        flip_method=flip_method,
        flip_improvement_ratio=flip_improvement_ratio,
        flip_min_gain_px=flip_min_gain_px,
        flip_error_threshold_px=flip_error_threshold_px,
        flip_error_delta_threshold_px=flip_error_delta_threshold_px,
    )
    state = (
        np.array(initial_state, copy=True)
        if initial_state is not None
        else initial_state_from_triangulation(model, reconstruction)
    )
    covariance = np.eye(ekf.nx) * 1e-2
    timings["init_s"] = time.perf_counter() - t_init

    states = np.zeros((pose_data.frames.shape[0], ekf.nx))
    update_status_per_frame: list[str] = []
    t_loop = time.perf_counter()
    for frame_idx in range(pose_data.frames.shape[0]):
        predicted_state, predicted_covariance = ekf.predict(state, covariance, frame_idx)
        if debug_console:
            print(
                debug_state_summary(
                    predicted_state, ekf.q_names, ekf.nq, f"[{debug_label or 'EKF'}] frame {frame_idx} predicted"
                )
            )
            validate_ekf_state_or_raise(
                predicted_state, predicted_covariance, ekf.q_names, ekf.nq, frame_idx, "predict"
            )
        state, covariance, update_status = ekf.update(predicted_state, predicted_covariance, frame_idx)
        if debug_console:
            print(
                debug_state_summary(
                    state,
                    ekf.q_names,
                    ekf.nq,
                    f"[{debug_label or 'EKF'}] frame {frame_idx} corrected ({update_status})",
                )
            )
            validate_ekf_state_or_raise(state, covariance, ekf.q_names, ekf.nq, frame_idx, "update")
        states[frame_idx] = state
        update_status_per_frame.append(update_status)
    timings["loop_s"] = time.perf_counter() - t_loop
    timings.update({key: float(value) for key, value in ekf.profiling.items()})
    q = (
        unwrap_root_rotations(states[:, : ekf.nq], ekf.q_names)
        if unwrap_root
        else np.array(states[:, : ekf.nq], copy=True)
    )
    return (
        {
            "q": q,
            "qdot": states[:, ekf.nq : 2 * ekf.nq],
            "qddot": states[:, 2 * ekf.nq :],
            "q_names": np.asarray(ekf.q_names, dtype=object),
            "update_status_per_frame": np.asarray(update_status_per_frame, dtype=object),
            "update_status_counts": dict(ekf.update_status),
            "flip_diagnostics": (
                {
                    "method": str(flip_method),
                    "decision_stage": "ekf_update_prediction_gate",
                    "improvement_ratio": float(flip_improvement_ratio),
                    "min_gain_px": float(flip_min_gain_px),
                    "min_valid_keypoints": int(ekf.flip_min_valid_keypoints),
                    "error_threshold_px": float(flip_error_threshold_px),
                    "error_delta_threshold_px": float(flip_error_delta_threshold_px),
                    "n_camera_updates_swapped": int(ekf.update_status.get("flip_prediction_gate_swapped", 0)),
                    "n_camera_updates_raw": int(ekf.update_status.get("flip_prediction_gate_raw", 0)),
                    "n_camera_updates_raw_insufficient_support": int(
                        ekf.update_status.get("flip_prediction_gate_raw_insufficient_support", 0)
                    ),
                    "compute_time_s": float(ekf.profiling.get("flip_gate_s", 0.0)),
                }
                if flip_method == "ekf_prediction_gate"
                else None
            ),
        },
        timings,
    )


def warmup_ekf_runtime(
    model,
    calibrations: dict[str, CameraCalibration],
    pose_data: PoseData,
    reconstruction: ReconstructionResult,
    fps: float,
    measurement_noise_scale: float,
    process_noise_scale: float,
    coherence_confidence_floor: float,
    min_frame_coherence_for_update: float,
    skip_low_coherence_updates: bool,
    enable_dof_locking: bool,
    root_flight_dynamics: bool,
    flight_height_threshold_m: float,
    flight_min_consecutive_frames: int,
    initial_state: np.ndarray,
) -> float:
    """Execute une frame jetable pour externaliser le warm-up du premier EKF.

    L'objectif est de sortir du temps des variantes les premiers appels reels a
    `markers(q)`, `markersJacobian(q)` et aux routines lineaires associees.
    """
    if pose_data.frames.shape[0] == 0:
        return 0.0

    ekf = MultiViewKinematicEKF(
        model=model,
        calibrations=calibrations,
        pose_data=pose_data,
        reconstruction=reconstruction,
        dt=1.0 / fps,
        measurement_noise_scale=measurement_noise_scale,
        process_noise_scale=process_noise_scale,
        min_frame_coherence_for_update=min_frame_coherence_for_update,
        skip_low_coherence_updates=skip_low_coherence_updates,
        coherence_confidence_floor=coherence_confidence_floor,
        enable_dof_locking=enable_dof_locking,
        root_flight_dynamics=root_flight_dynamics,
        flight_height_threshold_m=flight_height_threshold_m,
        flight_min_consecutive_frames=flight_min_consecutive_frames,
    )
    covariance = np.eye(ekf.nx) * 1e-2
    state = np.array(initial_state, copy=True)
    t0 = time.perf_counter()
    predicted_state, predicted_covariance = ekf.predict(state, covariance, 0)
    _ = ekf.update(predicted_state, predicted_covariance, 0)
    return time.perf_counter() - t0


def compute_dyn_activation_and_root_qddot_diff(
    reconstruction: ReconstructionResult,
    ekf_result_acc: dict[str, np.ndarray],
    ekf_result_dyn: dict[str, np.ndarray] | None,
    flight_height_threshold_m: float,
    flight_min_consecutive_frames: int,
    n_root: int,
) -> dict[str, object] | None:
    """Calcule deux diagnostics pour comparer ACC et DYN.

    - nombre de frames ou la branche DYN est active
    - norme de `qddot_root_dyn - qddot_root_acc`
    """
    if ekf_result_dyn is None:
        return None

    n_frames = reconstruction.points_3d.shape[0]
    activation_mask = np.zeros(n_frames, dtype=bool)
    for frame_idx in range(n_frames):
        if frame_idx <= 0:
            continue
        start_idx = max(0, frame_idx - max(1, int(flight_min_consecutive_frames)))
        active = True
        for previous_idx in range(start_idx, frame_idx):
            previous_points = reconstruction.points_3d[previous_idx]
            previous_z = previous_points[:, 2]
            valid = np.isfinite(previous_z)
            if not np.any(valid) or not np.all(previous_z[valid] > flight_height_threshold_m):
                active = False
                break
        activation_mask[frame_idx] = active and (frame_idx - start_idx) >= max(1, int(flight_min_consecutive_frames))

    qddot_acc = np.asarray(ekf_result_acc["qddot"], dtype=float)
    qddot_dyn = np.asarray(ekf_result_dyn["qddot"], dtype=float)
    root_diff = qddot_dyn[:, :n_root] - qddot_acc[:, :n_root]
    root_diff_norm = np.linalg.norm(root_diff, axis=1)
    return {
        "n_root": int(n_root),
        "dyn_branch_activated_frames": int(np.sum(activation_mask)),
        "dyn_branch_activation_ratio": float(np.mean(activation_mask)),
        "dyn_branch_first_frame_indices": np.flatnonzero(activation_mask)[:50].astype(int).tolist(),
        "qddot_root_dyn_minus_acc_norm_mean": float(np.mean(root_diff_norm)),
        "qddot_root_dyn_minus_acc_norm_max": float(np.max(root_diff_norm)),
        "qddot_root_dyn_minus_acc_norm_nonzero_frames": int(np.sum(root_diff_norm > 1e-12)),
    }


def run_biorbd_marker_kalman(model, reconstruction: ReconstructionResult, fps: float) -> dict[str, np.ndarray]:
    """Lance le Kalman marqueurs classique de `biorbd` sur les marqueurs 3D triangules."""
    return run_biorbd_marker_kalman_with_parameters(
        model,
        reconstruction,
        fps,
        noise_factor=DEFAULT_BIORBD_KALMAN_NOISE_FACTOR,
        error_factor=DEFAULT_BIORBD_KALMAN_ERROR_FACTOR,
        unwrap_root=True,
        initial_state_method=DEFAULT_BIORBD_KALMAN_INIT_METHOD,
    )


def compute_model_reprojection_errors(
    model,
    q_trajectory: np.ndarray,
    calibrations: dict[str, CameraCalibration],
    pose_data: PoseData,
) -> np.ndarray:
    """Calcule les erreurs de reprojection 2D d'une trajectoire `q` sur toutes les vues."""
    marker_names = marker_name_list(model)
    marker_kp_pairs = [(marker_name, KP_INDEX[marker_name]) for marker_name in marker_names if marker_name in KP_INDEX]
    errors = []
    n_frames = min(q_trajectory.shape[0], pose_data.keypoints.shape[1])

    for frame_idx in range(n_frames):
        q = q_trajectory[frame_idx]
        marker_positions = {name: marker.to_array() for name, marker in zip(marker_names, model.markers(q))}
        for cam_idx, cam_name in enumerate(pose_data.camera_names):
            calibration = calibrations[cam_name]
            frame_keypoints = pose_data.keypoints[cam_idx, frame_idx]
            frame_scores = pose_data.scores[cam_idx, frame_idx]
            for marker_name, kp_idx in marker_kp_pairs:
                if frame_scores[kp_idx] <= 0:
                    continue
                z_uv = frame_keypoints[kp_idx]
                if not np.all(np.isfinite(z_uv)):
                    continue
                point_3d = marker_positions[marker_name]
                if not np.all(np.isfinite(point_3d)):
                    continue
                h_uv = calibration.project_point(point_3d)
                errors.append(float(np.linalg.norm(z_uv - h_uv)))

    return np.asarray(errors, dtype=float)


def compare_kalman_filters(
    biomod_path: Path,
    calibrations: dict[str, CameraCalibration],
    pose_data: PoseData,
    reconstruction: ReconstructionResult,
    ekf_result: dict[str, np.ndarray],
    fps: float,
    biorbd_kalman_noise_factor: float = DEFAULT_BIORBD_KALMAN_NOISE_FACTOR,
    biorbd_kalman_error_factor: float = DEFAULT_BIORBD_KALMAN_ERROR_FACTOR,
    biorbd_kalman_init_method: str = DEFAULT_BIORBD_KALMAN_INIT_METHOD,
    classic_result: dict[str, np.ndarray] | None = None,
    unwrap_root: bool = True,
) -> ComparisonResult:
    """Compare les `q` du nouvel EKF avec ceux du Kalman `biorbd`."""
    import biorbd

    model = biorbd.Model(str(biomod_path))
    if classic_result is None:
        classic = run_biorbd_marker_kalman_with_parameters(
            model,
            reconstruction,
            fps,
            noise_factor=biorbd_kalman_noise_factor,
            error_factor=biorbd_kalman_error_factor,
            unwrap_root=unwrap_root,
            initial_state_method=biorbd_kalman_init_method,
        )
    else:
        classic = classic_result

    n_frames = min(ekf_result["q"].shape[0], classic["q"].shape[0], pose_data.keypoints.shape[1])
    ekf_q = ekf_result["q"][:n_frames]
    classic_q = classic["q"][:n_frames]
    classic_qdot = classic["qdot"][:n_frames]
    classic_qddot = classic["qddot"][:n_frames]

    ekf_2d_reprojection_errors = compute_model_reprojection_errors(model, ekf_q, calibrations, pose_data)
    ekf_3d_reprojection_errors = compute_model_reprojection_errors(model, classic_q, calibrations, pose_data)
    diff = ekf_q - classic_q
    rmse = np.sqrt(np.mean(diff**2, axis=0))
    mae = np.mean(np.abs(diff), axis=0)
    return ComparisonResult(
        q_ekf=ekf_q,
        q_ekf_3d=classic_q,
        qdot_ekf_3d=classic_qdot,
        qddot_ekf_3d=classic_qddot,
        rmse_per_dof=rmse,
        mae_per_dof=mae,
        ekf_2d_reprojection_mean_px=float(np.nanmean(ekf_2d_reprojection_errors)),
        ekf_2d_reprojection_std_px=float(np.nanstd(ekf_2d_reprojection_errors)),
        ekf_3d_reprojection_mean_px=float(np.nanmean(ekf_3d_reprojection_errors)),
        ekf_3d_reprojection_std_px=float(np.nanstd(ekf_3d_reprojection_errors)),
        q_names=ekf_result["q_names"],
    )


def reconstruction_cache_metadata(
    pose_data: PoseData,
    error_threshold_px: float,
    min_cameras_for_triangulation: int,
    epipolar_threshold_px: float,
    triangulation_method: str,
    pose_data_mode: str,
    pose_filter_window: int,
    pose_outlier_threshold_ratio: float,
    pose_amplitude_lower_percentile: float,
    pose_amplitude_upper_percentile: float,
    pose_correction_mode: str = "none",
) -> dict[str, object]:
    """Construit les metadonnees necessaires pour valider un cache de triangulation."""
    return {
        "camera_names": list(pose_data.camera_names),
        "n_frames": int(pose_data.frames.shape[0]),
        "frame_signature": frame_signature(pose_data.frames),
        "pose_data_signature": pose_data_signature(pose_data),
        "reprojection_threshold_px": float(error_threshold_px),
        "min_cameras_for_triangulation": int(min_cameras_for_triangulation),
        "epipolar_threshold_px": float(epipolar_threshold_px),
        "triangulation_method": triangulation_method,
        "pose_data_mode": pose_data_mode,
        "pose_correction_mode": str(pose_correction_mode),
        "pose_filter_window": int(pose_filter_window),
        "pose_outlier_threshold_ratio": float(pose_outlier_threshold_ratio),
        "pose_amplitude_lower_percentile": float(pose_amplitude_lower_percentile),
        "pose_amplitude_upper_percentile": float(pose_amplitude_upper_percentile),
    }


def model_stage_metadata(
    reconstruction_cache_path: Path,
    reconstruction: ReconstructionResult,
    fps: float,
    subject_mass_kg: float,
    initial_rotation_correction: bool,
) -> dict[str, object]:
    """Metadonnees de validite du stage modele."""
    return {
        "model_stage_version": int(MODEL_STAGE_VERSION),
        "reconstruction_cache_path": str(reconstruction_cache_path),
        "reconstruction_n_frames": int(reconstruction.frames.shape[0]),
        "reconstruction_frame_signature": frame_signature(reconstruction.frames),
        "fps": float(fps),
        "subject_mass_kg": float(subject_mass_kg),
        "initial_rotation_correction": bool(initial_rotation_correction),
    }


def save_model_stage(
    cache_path: Path,
    lengths: SegmentLengths,
    biomod_path: Path,
    metadata: dict[str, object],
    compute_time_s: float = 0.0,
) -> None:
    """Sauvegarde les longueurs et le chemin du biomod dans un cache dedie."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        lengths=np.asarray(json.dumps(lengths.__dict__), dtype=object),
        biomod_path=np.asarray(str(biomod_path), dtype=object),
        compute_time_s=np.asarray(float(compute_time_s), dtype=float),
        metadata=np.asarray(json.dumps(metadata), dtype=object),
    )


def load_model_stage(cache_path: Path) -> tuple[SegmentLengths, Path, float]:
    """Recharge le stage modele sans recalculer les longueurs ni reexporter le biomod."""
    with np.load(cache_path, allow_pickle=True) as data:
        lengths_dict = json.loads(data["lengths"].item())
        biomod_path = Path(data["biomod_path"].item())
        compute_time_s = float(np.asarray(data["compute_time_s"]).item()) if "compute_time_s" in data else 0.0
    return SegmentLengths(**lengths_dict), biomod_path, compute_time_s


def metadata_cache_matches(cache_path: Path, expected_metadata: dict[str, object]) -> bool:
    """Compare des metadonnees JSON stockees dans un cache `.npz`."""
    if not cache_path.exists():
        return False
    try:
        with np.load(cache_path, allow_pickle=True) as data:
            if "metadata" not in data:
                return False
            cached_metadata = json.loads(data["metadata"].item())
    except Exception:
        return False
    for key, expected_value in expected_metadata.items():
        cached_value = cached_metadata.get(key)
        if cached_value is None:
            return False
        if isinstance(expected_value, float):
            if not math.isclose(float(cached_value), expected_value, rel_tol=1e-9, abs_tol=1e-9):
                return False
        else:
            if cached_value != expected_value:
                return False
    return True


def save_reconstruction_cache(
    cache_path: Path,
    reconstruction: ReconstructionResult,
    metadata: dict[str, object],
) -> None:
    """Sauvegarde la reconstruction initiale dans un cache reutilisable."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        frames=reconstruction.frames,
        points_3d=reconstruction.points_3d,
        mean_confidence=reconstruction.mean_confidence,
        reprojection_error=reconstruction.reprojection_error,
        reprojection_error_per_view=reconstruction.reprojection_error_per_view,
        multiview_coherence=reconstruction.multiview_coherence,
        epipolar_coherence=reconstruction.epipolar_coherence,
        triangulation_coherence=reconstruction.triangulation_coherence,
        excluded_views=reconstruction.excluded_views,
        coherence_method=np.asarray(reconstruction.coherence_method, dtype=object),
        epipolar_coherence_compute_time_s=np.asarray(reconstruction.epipolar_coherence_compute_time_s, dtype=float),
        triangulation_compute_time_s=np.asarray(reconstruction.triangulation_compute_time_s, dtype=float),
        keypoint_names=np.asarray(COCO17, dtype=object),
        metadata=np.asarray(json.dumps(metadata), dtype=object),
    )


def load_reconstruction_cache(cache_path: Path, coherence_method: str) -> ReconstructionResult:
    """Charge un cache de triangulation et active la source de coherence demandee."""
    with np.load(cache_path, allow_pickle=True) as data:
        frames = np.asarray(data["frames"])
        points_3d = np.asarray(data["points_3d"])
        mean_confidence = np.asarray(data["mean_confidence"])
        reprojection_error = np.asarray(data["reprojection_error"])
        reprojection_error_per_view = np.asarray(data["reprojection_error_per_view"])
        excluded_views = np.asarray(data["excluded_views"], dtype=bool)

        if "epipolar_coherence" in data:
            epipolar_coherence = np.asarray(data["epipolar_coherence"])
        else:
            epipolar_coherence = np.zeros_like(reprojection_error_per_view)

        if "triangulation_coherence" in data:
            triangulation_coherence = np.asarray(data["triangulation_coherence"])
        else:
            triangulation_coherence = np.asarray(data["multiview_coherence"])
        epipolar_time_s = (
            float(np.asarray(data["epipolar_coherence_compute_time_s"]).item())
            if "epipolar_coherence_compute_time_s" in data
            else 0.0
        )
        triangulation_time_s = (
            float(np.asarray(data["triangulation_compute_time_s"]).item())
            if "triangulation_compute_time_s" in data
            else 0.0
        )

        multiview_coherence = select_active_coherence(
            epipolar_coherence=epipolar_coherence,
            triangulation_coherence=triangulation_coherence,
            coherence_method=coherence_method,
        )

    return ReconstructionResult(
        frames=frames,
        points_3d=points_3d,
        mean_confidence=mean_confidence,
        reprojection_error=reprojection_error,
        reprojection_error_per_view=reprojection_error_per_view,
        multiview_coherence=multiview_coherence,
        epipolar_coherence=epipolar_coherence,
        triangulation_coherence=triangulation_coherence,
        excluded_views=excluded_views,
        coherence_method=coherence_method,
        epipolar_coherence_compute_time_s=epipolar_time_s,
        triangulation_compute_time_s=triangulation_time_s,
    )


def reconstruction_distance_stats(
    reference: ReconstructionResult, alternative: ReconstructionResult
) -> dict[str, float]:
    """Compare deux triangulations en distance 3D point a point."""
    n_frames = min(reference.points_3d.shape[0], alternative.points_3d.shape[0])
    ref = reference.points_3d[:n_frames]
    alt = alternative.points_3d[:n_frames]
    valid = np.all(np.isfinite(ref), axis=2) & np.all(np.isfinite(alt), axis=2)
    if not np.any(valid):
        return {"mean_distance_m": float("nan"), "median_distance_m": float("nan")}
    distances = np.linalg.norm(ref - alt, axis=2)[valid]
    return {
        "mean_distance_m": float(np.mean(distances)),
        "median_distance_m": float(np.median(distances)),
    }


def reconstruction_cache_matches(cache_path: Path, expected_metadata: dict[str, object]) -> bool:
    """Verifie si un cache existant correspond bien aux parametres demandes."""
    return metadata_cache_matches(cache_path, expected_metadata)


def run_biorbd_marker_kalman_with_parameters(
    model,
    reconstruction: ReconstructionResult,
    fps: float,
    noise_factor: float,
    error_factor: float,
    unwrap_root: bool = True,
    initial_state_method: str = DEFAULT_BIORBD_KALMAN_INIT_METHOD,
) -> dict[str, np.ndarray]:
    """Version parametree du Kalman `biorbd` pour faciliter le tuning du lissage."""
    import biorbd

    params = biorbd.KalmanParam(fps, noiseFactor=noise_factor, errorFactor=error_factor)
    kalman = biorbd.KalmanReconsMarkers(model, params)
    initial_state, initial_state_diagnostics = compute_biorbd_kalman_initial_state(
        model,
        reconstruction,
        method=initial_state_method,
    )
    if initial_state is not None:
        kalman.setInitState(
            biorbd.GeneralizedCoordinates(np.asarray(initial_state[: model.nbQ()], dtype=float)),
            biorbd.GeneralizedVelocity(np.asarray(initial_state[model.nbQ() : 2 * model.nbQ()], dtype=float)),
            biorbd.GeneralizedAcceleration(np.asarray(initial_state[2 * model.nbQ() :], dtype=float)),
        )

    q_all = np.zeros((reconstruction.points_3d.shape[0], model.nbQ()))
    qdot_all = np.zeros_like(q_all)
    qddot_all = np.zeros_like(q_all)

    for frame_idx in range(reconstruction.points_3d.shape[0]):
        marker_tensor = points_to_marker_tensor(model, reconstruction.points_3d[frame_idx])
        markers = []
        for i_marker in range(model.nbMarkers()):
            xyz = marker_tensor[:, i_marker, 0]
            markers.append(biorbd.NodeSegment(xyz if np.all(np.isfinite(xyz)) else np.array([np.nan, np.nan, np.nan])))
        q = biorbd.GeneralizedCoordinates(model)
        qdot = biorbd.GeneralizedVelocity(model)
        qddot = biorbd.GeneralizedAcceleration(model)
        kalman.reconstructFrame(model, markers, q, qdot, qddot)
        q_all[frame_idx, :] = q.to_array()
        qdot_all[frame_idx, :] = qdot.to_array()
        qddot_all[frame_idx, :] = qddot.to_array()

    q_names = q_names_from_model(model)
    q_all = unwrap_root_rotations(q_all, q_names) if unwrap_root else q_all

    return {
        "q": q_all,
        "qdot": qdot_all,
        "qddot": qddot_all,
        "initial_state_method": np.asarray(initial_state_method, dtype=object),
        "initial_state_diagnostics": initial_state_diagnostics,
    }


def biorbd_kalman_cache_metadata(
    reconstruction_cache_path: Path,
    reconstruction: ReconstructionResult,
    biomod_path: Path,
    fps: float,
    noise_factor: float,
    error_factor: float,
    initial_state_method: str = DEFAULT_BIORBD_KALMAN_INIT_METHOD,
) -> dict[str, object]:
    """Metadonnees du cache Kalman `biorbd`."""
    return {
        "reconstruction_cache_path": str(reconstruction_cache_path),
        "reconstruction_n_frames": int(reconstruction.frames.shape[0]),
        "reconstruction_frame_signature": frame_signature(reconstruction.frames),
        "biomod_path": str(biomod_path),
        "fps": float(fps),
        "noise_factor": float(noise_factor),
        "error_factor": float(error_factor),
        "initial_state_method": str(initial_state_method),
    }


def save_biorbd_kalman_cache(cache_path: Path, result: dict[str, np.ndarray], metadata: dict[str, object]) -> None:
    """Sauvegarde le resultat du Kalman `biorbd` pour reutilisation."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        q=result["q"],
        qdot=result["qdot"],
        qddot=result["qddot"],
        initial_state_method=np.asarray(
            str(result.get("initial_state_method", DEFAULT_BIORBD_KALMAN_INIT_METHOD)), dtype=object
        ),
        initial_state_diagnostics=np.asarray(json.dumps(result.get("initial_state_diagnostics", {})), dtype=object),
        metadata=np.asarray(json.dumps(metadata), dtype=object),
    )


def load_biorbd_kalman_cache(cache_path: Path) -> dict[str, np.ndarray]:
    """Recharge un resultat de Kalman `biorbd` depuis le cache."""
    with np.load(cache_path, allow_pickle=True) as data:
        return {
            "q": np.asarray(data["q"]),
            "qdot": np.asarray(data["qdot"]),
            "qddot": np.asarray(data["qddot"]),
            "initial_state_method": (
                np.asarray(data["initial_state_method"]).item()
                if "initial_state_method" in data
                else DEFAULT_BIORBD_KALMAN_INIT_METHOD
            ),
            "initial_state_diagnostics": (
                json.loads(str(np.asarray(data["initial_state_diagnostics"]).item()))
                if "initial_state_diagnostics" in data
                else {}
            ),
        }


def try_export_pyorerun_animation(biomod_path: Path, q: np.ndarray, fps: float, output_dir: Path) -> str | None:
    """Tente d'exporter une animation `pyorerun` si la bibliotheque est disponible."""
    try:
        import pyorerun
    except ImportError:
        return None

    time_vector = np.arange(q.shape[0]) / fps
    viz = pyorerun.PhaseRerun(time_vector)
    viz_model = pyorerun.BiorbdModel(str(biomod_path))
    viz_model.options.transparent_mesh = False
    viz_model.options.show_gravity = False
    viz_model.options.show_marker_labels = False
    viz_model.options.show_center_of_mass_labels = False
    viz.add_animated_model(viz_model, q.T)

    rrd_path = output_dir / "ekf_animation.rrd"
    if hasattr(viz, "save"):
        viz.save(str(rrd_path))
    elif hasattr(viz, "rerun_by_frame"):
        viz.rerun_by_frame("EKF animation")
    else:
        return None
    return str(rrd_path) if rrd_path.exists() else "interactive"


def save_outputs(
    output_dir: Path,
    reconstruction: ReconstructionResult,
    reconstruction_cache_path: Path,
    reconstruction_cache_metadata_dict: dict[str, object],
    reconstruction_fast: ReconstructionResult | None,
    reconstruction_fast_cache_path: Path | None,
    lengths: SegmentLengths,
    biomod_path: Path,
    ekf_result_acc: dict[str, np.ndarray],
    ekf_result_dyn: dict[str, np.ndarray] | None = None,
    ekf_result_flip_acc: dict[str, np.ndarray] | None = None,
    ekf_result_flip_dyn: dict[str, np.ndarray] | None = None,
    comparison_acc: ComparisonResult | None = None,
    comparison_dyn: ComparisonResult | None = None,
    comparison_flip_acc: ComparisonResult | None = None,
    comparison_flip_dyn: ComparisonResult | None = None,
    dyn_vs_acc_diagnostics: dict[str, object] | None = None,
    animation_target: str | None = None,
    biorbd_kalman_noise_factor: float = DEFAULT_BIORBD_KALMAN_NOISE_FACTOR,
    biorbd_kalman_error_factor: float = DEFAULT_BIORBD_KALMAN_ERROR_FACTOR,
    subject_mass_kg: float = DEFAULT_SUBJECT_MASS_KG,
    root_flight_dynamics: bool = False,
    flight_height_threshold_m: float = DEFAULT_FLIGHT_HEIGHT_THRESHOLD_M,
    flight_min_consecutive_frames: int = DEFAULT_FLIGHT_MIN_CONSECUTIVE_FRAMES,
    measurement_noise_scale: float = 1.0,
    coherence_confidence_floor: float = DEFAULT_COHERENCE_CONFIDENCE_FLOOR,
    pose_data_mode: str = "cleaned",
    pose_filter_window: int = 9,
    pose_outlier_threshold_ratio: float = 0.10,
    pose_amplitude_lower_percentile: float = 5.0,
    pose_amplitude_upper_percentile: float = 95.0,
    ekf2d_initial_state_method: str = DEFAULT_EKF2D_INITIAL_STATE_METHOD,
    ekf2d_bootstrap_passes: int = DEFAULT_EKF2D_BOOTSTRAP_PASSES,
    shared_initial_state_diagnostics: dict[str, object] | None = None,
    shared_initial_state_flip_acc_diagnostics: dict[str, object] | None = None,
    stage_timings_s: dict[str, float] | None = None,
    triangulation_comparison: dict[str, float] | None = None,
    left_right_flip_diagnostics: dict[str, object] | None = None,
) -> None:
    """Sauvegarde les sorties numeriques et un resume JSON compact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_reconstruction_cache(reconstruction_cache_path, reconstruction, reconstruction_cache_metadata_dict)
    if reconstruction_fast is not None and reconstruction_fast_cache_path is not None:
        reconstruction_fast_metadata_dict = dict(reconstruction_cache_metadata_dict)
        reconstruction_fast_metadata_dict["triangulation_method"] = "greedy"
        save_reconstruction_cache(
            reconstruction_fast_cache_path, reconstruction_fast, reconstruction_fast_metadata_dict
        )
    save_single_ekf_state(output_dir / "ekf_states_acc.npz", ekf_result_acc)
    ekf_combined_payload = {
        "q": ekf_result_acc["q"],
        "qdot": ekf_result_acc["qdot"],
        "qddot": ekf_result_acc["qddot"],
        "q_names": ekf_result_acc["q_names"],
        "update_status_per_frame_ekf_2d_acc": ekf_result_acc["update_status_per_frame"],
        "q_ekf_2d_acc": ekf_result_acc["q"],
        "qdot_ekf_2d_acc": ekf_result_acc["qdot"],
        "qddot_ekf_2d_acc": ekf_result_acc["qddot"],
    }
    if ekf_result_dyn is not None:
        save_single_ekf_state(output_dir / "ekf_states_dyn.npz", ekf_result_dyn)
        ekf_combined_payload.update(
            {
                "update_status_per_frame_ekf_2d_dyn": ekf_result_dyn["update_status_per_frame"],
                "q_ekf_2d_dyn": ekf_result_dyn["q"],
                "qdot_ekf_2d_dyn": ekf_result_dyn["qdot"],
                "qddot_ekf_2d_dyn": ekf_result_dyn["qddot"],
            }
        )
    if ekf_result_flip_acc is not None:
        save_single_ekf_state(output_dir / "ekf_states_flip_acc.npz", ekf_result_flip_acc)
        ekf_combined_payload.update(
            {
                "update_status_per_frame_ekf_2d_flip_acc": ekf_result_flip_acc["update_status_per_frame"],
                "q_ekf_2d_flip_acc": ekf_result_flip_acc["q"],
                "qdot_ekf_2d_flip_acc": ekf_result_flip_acc["qdot"],
                "qddot_ekf_2d_flip_acc": ekf_result_flip_acc["qddot"],
            }
        )
    if ekf_result_flip_dyn is not None:
        save_single_ekf_state(output_dir / "ekf_states_flip_dyn.npz", ekf_result_flip_dyn)
        ekf_combined_payload.update(
            {
                "update_status_per_frame_ekf_2d_flip_dyn": ekf_result_flip_dyn["update_status_per_frame"],
                "q_ekf_2d_flip_dyn": ekf_result_flip_dyn["q"],
                "qdot_ekf_2d_flip_dyn": ekf_result_flip_dyn["qdot"],
                "qddot_ekf_2d_flip_dyn": ekf_result_flip_dyn["qddot"],
            }
        )
    np.savez(output_dir / "ekf_states.npz", **ekf_combined_payload)

    nan_mask = ~np.all(np.isfinite(reconstruction.points_3d), axis=2)
    missing_3d_keypoint_frames = nan_mask.sum(axis=0)
    frames_with_any_missing_3d = int(np.any(nan_mask, axis=1).sum())
    frames_with_all_missing_3d = int(np.all(nan_mask, axis=1).sum())

    if comparison_acc is not None:
        payload_acc = comparison_to_npz_payload(comparison_acc)
        np.savez(output_dir / "kalman_comparison_acc.npz", **payload_acc)
        np.savez(output_dir / "kalman_comparison.npz", **payload_acc)
    if comparison_dyn is not None:
        np.savez(output_dir / "kalman_comparison_dyn.npz", **comparison_to_npz_payload(comparison_dyn))
    if comparison_flip_acc is not None:
        np.savez(output_dir / "kalman_comparison_flip_acc.npz", **comparison_to_npz_payload(comparison_flip_acc))
    if comparison_flip_dyn is not None:
        np.savez(output_dir / "kalman_comparison_flip_dyn.npz", **comparison_to_npz_payload(comparison_flip_dyn))
    summary = {
        "biomod_path": str(biomod_path),
        "segment_lengths_m": lengths.__dict__,
        "n_frames": int(reconstruction.frames.shape[0]),
        "pose_data_mode": pose_data_mode,
        "pose_filter_window": int(pose_filter_window),
        "pose_outlier_threshold_ratio": float(pose_outlier_threshold_ratio),
        "pose_amplitude_lower_percentile": float(pose_amplitude_lower_percentile),
        "pose_amplitude_upper_percentile": float(pose_amplitude_upper_percentile),
        "mean_reprojection_error_px": float(np.nanmean(reconstruction.reprojection_error)),
        "mean_multiview_coherence": float(np.nanmean(reconstruction.multiview_coherence)),
        "mean_epipolar_coherence": float(np.nanmean(reconstruction.epipolar_coherence)),
        "mean_triangulation_coherence": float(np.nanmean(reconstruction.triangulation_coherence)),
        "epipolar_coherence_compute_time_s": float(reconstruction.epipolar_coherence_compute_time_s),
        "mean_reprojection_error_fast_px": (
            float(np.nanmean(reconstruction_fast.reprojection_error)) if reconstruction_fast is not None else None
        ),
        "triangulation_defaults": {
            "reprojection_threshold_px": DEFAULT_REPROJECTION_THRESHOLD_PX,
            "epipolar_threshold_px": DEFAULT_EPIPOLAR_THRESHOLD_PX,
            "min_cameras_for_triangulation": DEFAULT_MIN_CAMERAS_FOR_TRIANGULATION,
            "min_frame_coherence_for_update": DEFAULT_MIN_FRAME_COHERENCE_FOR_UPDATE,
            "coherence_method": DEFAULT_COHERENCE_METHOD,
            "triangulation_method": DEFAULT_TRIANGULATION_METHOD,
        },
        "coherence_method_used": reconstruction.coherence_method,
        "reconstruction_cache_path": str(reconstruction_cache_path),
        "reconstruction_fast_cache_path": (
            str(reconstruction_fast_cache_path) if reconstruction_fast_cache_path is not None else None
        ),
        "subject_mass_kg": float(subject_mass_kg),
        "measurement_noise_scale": float(measurement_noise_scale),
        "coherence_confidence_floor": float(coherence_confidence_floor),
        "root_flight_dynamics": bool(root_flight_dynamics),
        "flight_height_threshold_m": float(flight_height_threshold_m),
        "flight_min_consecutive_frames": int(flight_min_consecutive_frames),
        "ekf2d_initial_state_method": str(ekf2d_initial_state_method),
        "ekf2d_bootstrap_passes": int(ekf2d_bootstrap_passes),
        "ekf2d_initial_state_diagnostics": shared_initial_state_diagnostics,
        "triangulation_missing_3d": {
            "frames_with_any_missing_keypoint": frames_with_any_missing_3d,
            "frames_with_all_keypoints_missing": frames_with_all_missing_3d,
            "missing_frames_per_keypoint": {
                name: int(count) for name, count in zip(COCO17, missing_3d_keypoint_frames)
            },
        },
        "ekf_2d_acc_frame_diagnostics": {
            "total_frames": int(ekf_result_acc["q"].shape[0]),
            "update_status_counts": {
                key: int(value) for key, value in ekf_result_acc.get("update_status_counts", {}).items()
            },
        },
    }
    if stage_timings_s is not None:
        summary["stage_timings_s"] = {str(key): float(value) for key, value in stage_timings_s.items()}
    if triangulation_comparison is not None:
        summary["triangulation_comparison"] = triangulation_comparison
    if left_right_flip_diagnostics is not None:
        summary["left_right_flip_diagnostics"] = left_right_flip_diagnostics
    if dyn_vs_acc_diagnostics is not None:
        summary["dyn_vs_acc_diagnostics"] = dyn_vs_acc_diagnostics
    if comparison_acc is not None:
        summary["kalman_comparison"] = comparison_to_summary_dict(
            comparison_acc,
            biorbd_kalman_noise_factor=biorbd_kalman_noise_factor,
            biorbd_kalman_error_factor=biorbd_kalman_error_factor,
        )
        summary["kalman_comparison_acc"] = summary["kalman_comparison"]
    if comparison_dyn is not None:
        summary["kalman_comparison_dyn"] = comparison_to_summary_dict(
            comparison_dyn,
            biorbd_kalman_noise_factor=biorbd_kalman_noise_factor,
            biorbd_kalman_error_factor=biorbd_kalman_error_factor,
        )
    if comparison_flip_acc is not None:
        summary["kalman_comparison_flip_acc"] = comparison_to_summary_dict(
            comparison_flip_acc,
            biorbd_kalman_noise_factor=biorbd_kalman_noise_factor,
            biorbd_kalman_error_factor=biorbd_kalman_error_factor,
        )
    if comparison_flip_dyn is not None:
        summary["kalman_comparison_flip_dyn"] = comparison_to_summary_dict(
            comparison_flip_dyn,
            biorbd_kalman_noise_factor=biorbd_kalman_noise_factor,
            biorbd_kalman_error_factor=biorbd_kalman_error_factor,
        )
    if ekf_result_dyn is not None:
        summary["ekf_2d_dyn_frame_diagnostics"] = {
            "total_frames": int(ekf_result_dyn["q"].shape[0]),
            "update_status_counts": {
                key: int(value) for key, value in ekf_result_dyn.get("update_status_counts", {}).items()
            },
        }
    if ekf_result_flip_acc is not None:
        summary["ekf_2d_flip_acc_frame_diagnostics"] = {
            "total_frames": int(ekf_result_flip_acc["q"].shape[0]),
            "update_status_counts": {
                key: int(value) for key, value in ekf_result_flip_acc.get("update_status_counts", {}).items()
            },
        }
        summary["ekf2d_flip_acc_initial_state_diagnostics"] = shared_initial_state_flip_acc_diagnostics
    if ekf_result_flip_dyn is not None:
        summary["ekf_2d_flip_dyn_frame_diagnostics"] = {
            "total_frames": int(ekf_result_flip_dyn["q"].shape[0]),
            "update_status_counts": {
                key: int(value) for key, value in ekf_result_flip_dyn.get("update_status_counts", {}).items()
            },
        }
    if animation_target is not None:
        summary["pyorerun_animation"] = animation_target
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


def parse_args() -> argparse.Namespace:
    """Construit l'interface CLI du pipeline."""
    parser = argparse.ArgumentParser(
        description="Triangulation 3D initiale + bioMod + EKF multi-vues pour keypoints 2D."
    )
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    parser.add_argument("--keypoints", type=Path, default=DEFAULT_KEYPOINTS)
    parser.add_argument(
        "--camera-names", type=str, default="", help="Liste de cameras a utiliser, separees par des virgules."
    )
    parser.add_argument(
        "--pose-data-mode",
        choices=("raw", "filtered", "cleaned"),
        default="cleaned",
        help="Source 2D utilisee apres chargement: brut, filtre lisse, ou nettoye avec rejet des outliers.",
    )
    parser.add_argument(
        "--pose-correction-mode",
        choices=(
            "none",
            "flip_epipolar",
            "flip_epipolar_fast",
            "flip_epipolar_viterbi",
            "flip_epipolar_fast_viterbi",
            "flip_triangulation",
        ),
        default="none",
        help="Correction optionnelle des 2D apres chargement: aucune, flip L/R local par epipolaire Sampson, local par epipolaire rapide (distance symétrique), variante Viterbi explicite, ou flip L/R detecte par triangulation/reprojection.",
    )
    parser.add_argument(
        "--pose-filter-window",
        type=int,
        default=9,
        help="Fenetre de lissage temporel utilisee pour construire la reference filtree des 2D.",
    )
    parser.add_argument(
        "--pose-outlier-threshold-ratio",
        type=float,
        default=0.10,
        help="Seuil de rejet des outliers 2D, exprime comme fraction de l'amplitude robuste.",
    )
    parser.add_argument(
        "--pose-amplitude-lower-percentile",
        type=float,
        default=5.0,
        help="Percentile inferieur utilise pour definir l'amplitude robuste des 2D.",
    )
    parser.add_argument(
        "--pose-amplitude-upper-percentile",
        type=float,
        default=95.0,
        help="Percentile superieur utilise pour definir l'amplitude robuste des 2D.",
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_CAMERA_FPS, help="Frequence d'acquisition camera en Hz.")
    parser.add_argument("--frame-start", type=int, default=None, help="Premiere frame incluse apres chargement des 2D.")
    parser.add_argument("--frame-end", type=int, default=None, help="Derniere frame incluse apres chargement des 2D.")
    parser.add_argument(
        "--subject-mass-kg",
        type=float,
        default=DEFAULT_SUBJECT_MASS_KG,
        help="Masse du sujet utilisee pour les parametres inertiels de de Leva (femme).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "vitpose_ekf")
    parser.add_argument("--biomod", type=Path, default=Path("outputs") / "vitpose_ekf" / "vitpose_chain.bioMod")
    parser.add_argument("--model-cache", type=Path, default=None, help="Cache NPZ du stage modele.")
    parser.add_argument("--biorbd-kalman-cache", type=Path, default=None, help="Cache NPZ du Kalman marqueurs biorbd.")
    parser.add_argument(
        "--reconstruction-cache",
        type=Path,
        default=None,
        help="Cache NPZ de triangulation robuste. Par defaut: <output-dir>/triangulation_pose2sim_like.npz.",
    )
    parser.add_argument(
        "--reuse-triangulation",
        action="store_true",
        help="Recharge le cache de triangulation s'il est compatible au lieu de recalculer.",
    )
    parser.add_argument(
        "--triangulate-only",
        action="store_true",
        help="Calcule uniquement la reconstruction initiale et son cache, puis s'arrete.",
    )
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="Calcule ou recharge le stage modele (longueurs + biomod), puis s'arrete.",
    )
    parser.add_argument(
        "--initial-rotation-correction",
        action="store_true",
        help="Si le repere racine reconstruit a t0 est oppose au repere global, applique une rotation de pi autour de Z lors de la construction du bioMod.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Limite optionnelle pour valider rapidement le pipeline."
    )
    parser.add_argument(
        "--reprojection-threshold-px",
        type=float,
        default=DEFAULT_REPROJECTION_THRESHOLD_PX,
        help="Seuil de reprojection pour la triangulation robuste type TRC-style.",
    )
    parser.add_argument(
        "--epipolar-threshold-px",
        type=float,
        default=DEFAULT_EPIPOLAR_THRESHOLD_PX,
        help="Seuil de conversion erreur epipolaire -> score de coherence.",
    )
    parser.add_argument(
        "--min-cameras-for-triangulation",
        type=int,
        default=DEFAULT_MIN_CAMERAS_FOR_TRIANGULATION,
        help="Nombre minimal de cameras a conserver pour trianguler un keypoint.",
    )
    parser.add_argument(
        "--coherence-method",
        choices=SUPPORTED_COHERENCE_METHODS,
        default=DEFAULT_COHERENCE_METHOD,
        help="Source du score de coherence multivue utilise dans l'EKF.",
    )
    parser.add_argument(
        "--triangulation-method",
        choices=SUPPORTED_TRIANGULATION_METHODS,
        default=DEFAULT_TRIANGULATION_METHOD,
        help="Strategie de triangulation: une seule DLT ponderee, suppression gloutonne, ou recherche exhaustive.",
    )
    parser.add_argument(
        "--triangulation-workers",
        type=int,
        default=DEFAULT_TRIANGULATION_WORKERS,
        help="Nombre de workers pour paralléliser la triangulation par frame.",
    )
    parser.add_argument(
        "--measurement-noise-scale",
        type=float,
        default=DEFAULT_MEASUREMENT_NOISE_SCALE,
        help="Facteur multiplicatif sur la covariance de mesure. Plus grand = plus de poids sur la prediction.",
    )
    parser.add_argument(
        "--process-noise-scale",
        type=float,
        default=1.0,
        help="Facteur multiplicatif sur la covariance de processus. Plus petit = plus de poids sur la prediction.",
    )
    parser.add_argument(
        "--ekf2d-initial-state-method",
        choices=("triangulation_ik", "ekf_bootstrap", "root_pose_bootstrap"),
        default=DEFAULT_EKF2D_INITIAL_STATE_METHOD,
        help="Methode pour initialiser q0 de l'EKF 2D: IK 3D sur la triangulation, bootstrap EKF sur une frame, ou bootstrap depuis une pose racine geometrique (hanches/epaules).",
    )
    parser.add_argument(
        "--ekf2d-bootstrap-passes",
        type=int,
        default=DEFAULT_EKF2D_BOOTSTRAP_PASSES,
        help="Nombre de passes EKF utilisees pour affiner q0 quand ekf2d-initial-state-method=ekf_bootstrap.",
    )
    parser.add_argument(
        "--coherence-confidence-floor",
        type=float,
        default=DEFAULT_COHERENCE_CONFIDENCE_FLOOR,
        help="Part minimale de confiance conservee meme si la coherence multivue est faible.",
    )
    parser.add_argument(
        "--min-frame-coherence-for-update",
        type=float,
        default=DEFAULT_MIN_FRAME_COHERENCE_FOR_UPDATE,
        help="Seuil minimal de coherence multivue moyen pour autoriser la correction Kalman sur une frame.",
    )
    parser.add_argument(
        "--skip-low-coherence-updates",
        action="store_true",
        help="Saute la correction EKF quand la coherence moyenne de frame passe sous le seuil.",
    )
    parser.add_argument(
        "--enable-dof-locking",
        action="store_true",
        help="Active le verrouillage automatique de certains DoF pres de singularites.",
    )
    parser.add_argument(
        "--root-flight-dynamics",
        action="store_true",
        help="En phase aerienne, utilise une prediction dynamique de la racine basee sur la matrice de masse et les effets non lineaires.",
    )
    parser.add_argument(
        "--run-ekf-2d-flip-acc",
        action="store_true",
        help="Lance une variante EKF 2D ACC ou les vues suspectes face/dos sont auto-corrigees par swap gauche/droite.",
    )
    parser.add_argument(
        "--run-ekf-2d-flip-dyn",
        action="store_true",
        help="Lance une variante EKF 2D DYN avec auto-correction des flips gauche/droite sur les vues suspectes.",
    )
    parser.add_argument(
        "--flip-improvement-ratio",
        type=float,
        default=DEFAULT_FLIP_IMPROVEMENT_RATIO,
        help="Accepte un flip L/R si le cout swappe devient inferieur a ce ratio du cout nominal.",
    )
    parser.add_argument(
        "--flip-min-gain-px",
        type=float,
        default=DEFAULT_FLIP_MIN_GAIN_PX,
        help="Gain minimal en pixels entre cout nominal et cout swappe pour accepter un flip L/R.",
    )
    parser.add_argument(
        "--flip-min-other-cameras",
        type=int,
        default=DEFAULT_FLIP_MIN_OTHER_CAMERAS,
        help="Nombre minimal d'autres cameras valides pour evaluer un flip L/R en mode triangulation.",
    )
    parser.add_argument(
        "--flip-outlier-percentile",
        type=float,
        default=DEFAULT_FLIP_OUTLIER_PERCENTILE,
        help="Percentile de cout nominal utilise pour restreindre le test flip L/R aux outliers par camera.",
    )
    parser.add_argument(
        "--flip-outlier-floor-px",
        type=float,
        default=DEFAULT_FLIP_OUTLIER_FLOOR_PX,
        help="Plancher en pixels pour definir les outliers de cout nominal avant de tester un flip L/R.",
    )
    parser.add_argument(
        "--flip-test-all-camera-frames",
        action="store_true",
        help="Teste le flip L/R sur tous les camera-frames au lieu de le restreindre aux outliers de cout nominal.",
    )
    parser.add_argument(
        "--flip-temporal-weight",
        type=float,
        default=DEFAULT_FLIP_TEMPORAL_WEIGHT,
        help="Poids du cout temporel 2D par camera dans le cout combine de decision flip L/R (0 desactive l'apport temporel).",
    )
    parser.add_argument(
        "--flip-temporal-tau-px",
        type=float,
        default=DEFAULT_FLIP_TEMPORAL_TAU_PX,
        help="Echelle en pixels du cout temporel pour normaliser sa contribution dans le score combine flip L/R.",
    )
    parser.add_argument(
        "--flip-temporal-min-valid-keypoints",
        type=int,
        default=DEFAULT_FLIP_TEMPORAL_MIN_VALID_KEYPOINTS,
        help="Nombre minimal de keypoints avec support temporel valide pour evaluer le cout temporel d'un camera-frame.",
    )
    parser.add_argument(
        "--flight-height-threshold-m",
        type=float,
        default=DEFAULT_FLIGHT_HEIGHT_THRESHOLD_M,
        help="Seuil vertical pour declarer la phase aerienne quand tous les marqueurs triangules de la frame precedente sont au-dessus de cette valeur.",
    )
    parser.add_argument(
        "--flight-min-consecutive-frames",
        type=int,
        default=DEFAULT_FLIGHT_MIN_CONSECUTIVE_FRAMES,
        help="Nombre minimal de frames consecutives au-dessus du seuil avant d'activer la dynamique de vol.",
    )
    parser.add_argument(
        "--biorbd-kalman-noise-factor",
        type=float,
        default=DEFAULT_BIORBD_KALMAN_NOISE_FACTOR,
        help="Parametre noiseFactor du Kalman marqueurs `biorbd`. Plus petit = plus lisse.",
    )
    parser.add_argument(
        "--biorbd-kalman-error-factor",
        type=float,
        default=DEFAULT_BIORBD_KALMAN_ERROR_FACTOR,
        help="Parametre errorFactor du Kalman marqueurs `biorbd`. Plus grand = moins de confiance dans l'etat precedent.",
    )
    parser.add_argument(
        "--biorbd-kalman-init-method",
        choices=(
            "none",
            "triangulation_ik",
            "triangulation_ik_root_translation",
            "root_translation_zero_rest",
            "root_pose_zero_rest",
        ),
        default=DEFAULT_BIORBD_KALMAN_INIT_METHOD,
        help="Strategie de warm-start du Kalman marqueurs `biorbd` via setInitState.",
    )
    parser.add_argument(
        "--no-root-unwrap",
        action="store_true",
        help="Desactive l'unwrap temporel des trois rotations de la racine dans EKF 2D et EKF 3D.",
    )
    parser.add_argument(
        "--debug-ekf-2d-dyn",
        action="store_true",
        help="Affiche les etats predits/corriges de EKF 2D DYN et arrete en cas de NaN ou de divergence evidente.",
    )
    parser.add_argument(
        "--compare-biorbd-kalman", action="store_true", help="Lance aussi le Kalman marqueurs classique de biorbd."
    )
    parser.add_argument("--animate", action="store_true", help="Exporte/lance une animation pyorerun si disponible.")
    return parser.parse_args()


def main() -> None:
    """Point d'entree CLI."""
    args = parse_args()
    stage_timings_s: dict[str, float] = {}
    calibrations = load_calibrations(args.calib)
    selected_camera_names = parse_camera_names(args.camera_names)
    if selected_camera_names:
        calibrations = subset_calibrations(calibrations, selected_camera_names)
    pose_data = load_pose_data(
        args.keypoints,
        calibrations,
        max_frames=args.max_frames,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        data_mode=args.pose_data_mode,
        smoothing_window=args.pose_filter_window,
        outlier_threshold_ratio=args.pose_outlier_threshold_ratio,
        lower_percentile=args.pose_amplitude_lower_percentile,
        upper_percentile=args.pose_amplitude_upper_percentile,
    )
    if args.pose_correction_mode != "none":
        if args.pose_correction_mode == "flip_epipolar":
            flip_method = "epipolar"
        elif args.pose_correction_mode == "flip_epipolar_fast":
            flip_method = "epipolar_fast"
        elif args.pose_correction_mode == "flip_epipolar_viterbi":
            flip_method = "epipolar_viterbi"
        elif args.pose_correction_mode == "flip_epipolar_fast_viterbi":
            flip_method = "epipolar_fast_viterbi"
        else:
            flip_method = "triangulation"
        t0 = time.perf_counter()
        left_right_flip_suspect_mask, _left_right_flip_diagnostics, _left_right_flip_details = (
            detect_left_right_flip_diagnostics(
                pose_data,
                calibrations,
                improvement_ratio=args.flip_improvement_ratio,
                min_gain_px=args.flip_min_gain_px,
                min_other_cameras=args.flip_min_other_cameras,
                restrict_to_outliers=not args.flip_test_all_camera_frames,
                outlier_percentile=args.flip_outlier_percentile,
                outlier_floor_px=args.flip_outlier_floor_px,
                geometry_tau_px=(
                    args.epipolar_threshold_px
                    if flip_method in {"epipolar", "epipolar_fast"}
                    else args.reprojection_threshold_px
                ),
                method=flip_method,
                temporal_weight=args.flip_temporal_weight,
                temporal_tau_px=args.flip_temporal_tau_px,
                temporal_min_valid_keypoints=args.flip_temporal_min_valid_keypoints,
            )
        )
        pose_data = apply_left_right_flip_corrections(pose_data, left_right_flip_suspect_mask)
        stage_timings_s["pose_correction_s"] = time.perf_counter() - t0

    output_dir = args.output_dir
    selected_triangulation_method = canonical_triangulation_method(args.triangulation_method)
    selected_coherence_method = canonical_coherence_method(args.coherence_method, selected_triangulation_method)
    reconstruction_cache_path = args.reconstruction_cache or (args.output_dir / "triangulation_pose2sim_like.npz")
    reconstruction_fast_cache_path = output_dir / "triangulation_pose2sim_like_fast.npz"
    reconstruction_once_cache_path = output_dir / "triangulation_pose2sim_like_once.npz"
    model_cache_path = args.model_cache or (output_dir / "model_stage.npz")
    biorbd_kalman_cache_path = args.biorbd_kalman_cache or (output_dir / "biorbd_kalman_states.npz")
    reconstruction_once_metadata_dict = reconstruction_cache_metadata(
        pose_data=pose_data,
        error_threshold_px=args.reprojection_threshold_px,
        min_cameras_for_triangulation=args.min_cameras_for_triangulation,
        epipolar_threshold_px=args.epipolar_threshold_px,
        triangulation_method="once",
        pose_data_mode=args.pose_data_mode,
        pose_correction_mode=args.pose_correction_mode,
        pose_filter_window=args.pose_filter_window,
        pose_outlier_threshold_ratio=args.pose_outlier_threshold_ratio,
        pose_amplitude_lower_percentile=args.pose_amplitude_lower_percentile,
        pose_amplitude_upper_percentile=args.pose_amplitude_upper_percentile,
    )
    reconstruction_metadata_dict = reconstruction_cache_metadata(
        pose_data=pose_data,
        error_threshold_px=args.reprojection_threshold_px,
        min_cameras_for_triangulation=args.min_cameras_for_triangulation,
        epipolar_threshold_px=args.epipolar_threshold_px,
        triangulation_method="exhaustive",
        pose_data_mode=args.pose_data_mode,
        pose_correction_mode=args.pose_correction_mode,
        pose_filter_window=args.pose_filter_window,
        pose_outlier_threshold_ratio=args.pose_outlier_threshold_ratio,
        pose_amplitude_lower_percentile=args.pose_amplitude_lower_percentile,
        pose_amplitude_upper_percentile=args.pose_amplitude_upper_percentile,
    )
    reconstruction_fast_metadata_dict = reconstruction_cache_metadata(
        pose_data=pose_data,
        error_threshold_px=args.reprojection_threshold_px,
        min_cameras_for_triangulation=args.min_cameras_for_triangulation,
        epipolar_threshold_px=args.epipolar_threshold_px,
        triangulation_method="greedy",
        pose_data_mode=args.pose_data_mode,
        pose_correction_mode=args.pose_correction_mode,
        pose_filter_window=args.pose_filter_window,
        pose_outlier_threshold_ratio=args.pose_outlier_threshold_ratio,
        pose_amplitude_lower_percentile=args.pose_amplitude_lower_percentile,
        pose_amplitude_upper_percentile=args.pose_amplitude_upper_percentile,
    )

    if args.reuse_triangulation and reconstruction_cache_matches(
        reconstruction_once_cache_path, reconstruction_once_metadata_dict
    ):
        t0 = time.perf_counter()
        reconstruction_once = load_reconstruction_cache(
            reconstruction_once_cache_path, coherence_method=selected_coherence_method
        )
        stage_timings_s["triangulation_once_s"] = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
        reconstruction_once = triangulate_pose2sim_like(
            pose_data,
            calibrations,
            error_threshold_px=args.reprojection_threshold_px,
            min_cameras_for_triangulation=args.min_cameras_for_triangulation,
            coherence_method=selected_coherence_method,
            epipolar_threshold_px=args.epipolar_threshold_px,
            triangulation_method="once",
            n_workers=args.triangulation_workers,
        )
        stage_timings_s["triangulation_once_s"] = time.perf_counter() - t0
        save_reconstruction_cache(
            reconstruction_once_cache_path, reconstruction_once, reconstruction_once_metadata_dict
        )

    if args.reuse_triangulation and reconstruction_cache_matches(
        reconstruction_cache_path, reconstruction_metadata_dict
    ):
        t0 = time.perf_counter()
        reconstruction_adaptive = load_reconstruction_cache(
            reconstruction_cache_path, coherence_method=selected_coherence_method
        )
        stage_timings_s["triangulation_adaptive_s"] = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
        reconstruction_adaptive = triangulate_pose2sim_like(
            pose_data,
            calibrations,
            error_threshold_px=args.reprojection_threshold_px,
            min_cameras_for_triangulation=args.min_cameras_for_triangulation,
            coherence_method=selected_coherence_method,
            epipolar_threshold_px=args.epipolar_threshold_px,
            triangulation_method="exhaustive",
            n_workers=args.triangulation_workers,
        )
        stage_timings_s["triangulation_adaptive_s"] = time.perf_counter() - t0
        save_reconstruction_cache(reconstruction_cache_path, reconstruction_adaptive, reconstruction_metadata_dict)

    if args.reuse_triangulation and reconstruction_cache_matches(
        reconstruction_fast_cache_path, reconstruction_fast_metadata_dict
    ):
        t0 = time.perf_counter()
        reconstruction_fast = load_reconstruction_cache(
            reconstruction_fast_cache_path, coherence_method=selected_coherence_method
        )
        stage_timings_s["triangulation_fast_s"] = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
        reconstruction_fast = triangulate_pose2sim_like(
            pose_data,
            calibrations,
            error_threshold_px=args.reprojection_threshold_px,
            min_cameras_for_triangulation=args.min_cameras_for_triangulation,
            coherence_method=selected_coherence_method,
            epipolar_threshold_px=args.epipolar_threshold_px,
            triangulation_method="greedy",
            n_workers=args.triangulation_workers,
        )
        stage_timings_s["triangulation_fast_s"] = time.perf_counter() - t0
        save_reconstruction_cache(
            reconstruction_fast_cache_path, reconstruction_fast, reconstruction_fast_metadata_dict
        )

    reconstruction_by_method = {
        "once": (reconstruction_once, reconstruction_once_cache_path, reconstruction_once_metadata_dict),
        "greedy": (reconstruction_fast, reconstruction_fast_cache_path, reconstruction_fast_metadata_dict),
        "exhaustive": (reconstruction_adaptive, reconstruction_cache_path, reconstruction_metadata_dict),
    }
    reconstruction, selected_reconstruction_cache_path, selected_reconstruction_metadata_dict = (
        reconstruction_by_method[selected_triangulation_method]
    )
    triangulation_comparison = reconstruction_distance_stats(reconstruction_adaptive, reconstruction_fast)
    stage_timings_s["epipolar_coherence_s"] = float(reconstruction.epipolar_coherence_compute_time_s)

    if args.triangulate_only:
        print(f"Frames traitées: {pose_data.frames.shape[0]}")
        print(
            f"Triangulation once: {stage_timings_s['triangulation_once_s']:.2f} s | erreur moyenne {np.nanmean(reconstruction_once.reprojection_error):.2f} px"
        )
        print(
            f"Triangulation adaptive: {stage_timings_s['triangulation_adaptive_s']:.2f} s | erreur moyenne {np.nanmean(reconstruction_adaptive.reprojection_error):.2f} px"
        )
        print(
            f"Triangulation fast: {stage_timings_s['triangulation_fast_s']:.2f} s | erreur moyenne {np.nanmean(reconstruction_fast.reprojection_error):.2f} px"
        )
        print(f"Ecart 3D adaptive vs fast: {triangulation_comparison['mean_distance_m']:.4f} m (moyenne)")
        print(f"Cache once: {reconstruction_once_cache_path}")
        print(f"Cache adaptive: {reconstruction_cache_path}")
        print(f"Cache fast: {reconstruction_fast_cache_path}")
        return

    model_metadata = model_stage_metadata(
        selected_reconstruction_cache_path,
        reconstruction,
        args.fps,
        args.subject_mass_kg,
        args.initial_rotation_correction,
    )
    if metadata_cache_matches(model_cache_path, model_metadata) and args.biomod.exists():
        t0 = time.perf_counter()
        lengths, biomod_path, _compute_time_s = load_model_stage(model_cache_path)
        biomod_path = args.biomod
        stage_timings_s["model_stage_s"] = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
        lengths = estimate_segment_lengths(reconstruction, fps=args.fps)
        biomod_path = build_biomod(
            lengths,
            output_path=args.biomod,
            subject_mass_kg=args.subject_mass_kg,
            reconstruction=reconstruction,
            apply_initial_root_rotation_correction=args.initial_rotation_correction,
        )
        model_compute_time_s = time.perf_counter() - t0
        save_model_stage(model_cache_path, lengths, biomod_path, model_metadata, compute_time_s=model_compute_time_s)
        stage_timings_s["model_stage_s"] = model_compute_time_s

    if args.model_only:
        print(f"Cache triangulation utilise: {selected_reconstruction_cache_path}")
        print(f"Cache modele: {model_cache_path}")
        print(f"bioMod exporte: {biomod_path}")
        return

    import biorbd

    shared_biorbd_model = biorbd.Model(str(biomod_path))
    # Warm-up neutre pour ne pas avantager le deuxieme EKF lance.
    _ = shared_biorbd_model.markers(np.zeros(shared_biorbd_model.nbQ()))
    _ = shared_biorbd_model.markersJacobian(np.zeros(shared_biorbd_model.nbQ()))

    left_right_flip_suspect_mask, left_right_flip_diagnostics, _left_right_flip_details = (
        detect_left_right_flip_diagnostics(
            pose_data,
            calibrations,
            improvement_ratio=args.flip_improvement_ratio,
            min_gain_px=args.flip_min_gain_px,
            min_other_cameras=args.flip_min_other_cameras,
            restrict_to_outliers=not args.flip_test_all_camera_frames,
            outlier_percentile=args.flip_outlier_percentile,
            outlier_floor_px=args.flip_outlier_floor_px,
            geometry_tau_px=args.epipolar_threshold_px,
            temporal_weight=args.flip_temporal_weight,
            temporal_tau_px=args.flip_temporal_tau_px,
            temporal_min_valid_keypoints=args.flip_temporal_min_valid_keypoints,
        )
    )
    left_right_flip_diagnostics = dict(left_right_flip_diagnostics)
    left_right_flip_diagnostics["tau_px"] = float(args.epipolar_threshold_px)
    pose_data_flip = apply_left_right_flip_corrections(pose_data, left_right_flip_suspect_mask)
    t0 = time.perf_counter()
    shared_initial_state, shared_initial_state_diagnostics = compute_ekf2d_initial_state(
        model=shared_biorbd_model,
        calibrations=calibrations,
        pose_data=pose_data,
        reconstruction=reconstruction,
        fps=args.fps,
        measurement_noise_scale=args.measurement_noise_scale,
        process_noise_scale=args.process_noise_scale,
        min_frame_coherence_for_update=args.min_frame_coherence_for_update,
        skip_low_coherence_updates=args.skip_low_coherence_updates,
        coherence_confidence_floor=args.coherence_confidence_floor,
        enable_dof_locking=args.enable_dof_locking,
        method=args.ekf2d_initial_state_method,
        bootstrap_passes=args.ekf2d_bootstrap_passes,
    )
    stage_timings_s["ekf_initial_state_s"] = time.perf_counter() - t0
    shared_initial_state_flip_acc = shared_initial_state
    shared_initial_state_flip_acc_diagnostics = shared_initial_state_diagnostics
    if args.run_ekf_2d_flip_acc and args.ekf2d_initial_state_method in {"ekf_bootstrap", "root_pose_bootstrap"}:
        t0 = time.perf_counter()
        shared_initial_state_flip_acc, shared_initial_state_flip_acc_diagnostics = compute_ekf2d_initial_state(
            model=shared_biorbd_model,
            calibrations=calibrations,
            pose_data=pose_data_flip,
            reconstruction=reconstruction,
            fps=args.fps,
            measurement_noise_scale=args.measurement_noise_scale,
            process_noise_scale=args.process_noise_scale,
            min_frame_coherence_for_update=args.min_frame_coherence_for_update,
            skip_low_coherence_updates=args.skip_low_coherence_updates,
            coherence_confidence_floor=args.coherence_confidence_floor,
            enable_dof_locking=args.enable_dof_locking,
            method=args.ekf2d_initial_state_method,
            bootstrap_passes=args.ekf2d_bootstrap_passes,
        )
        stage_timings_s["ekf_initial_state_flip_acc_s"] = time.perf_counter() - t0

    stage_timings_s["ekf_warmup_acc_s"] = warmup_ekf_runtime(
        model=shared_biorbd_model,
        calibrations=calibrations,
        pose_data=pose_data,
        reconstruction=reconstruction,
        fps=args.fps,
        measurement_noise_scale=args.measurement_noise_scale,
        process_noise_scale=args.process_noise_scale,
        coherence_confidence_floor=args.coherence_confidence_floor,
        min_frame_coherence_for_update=args.min_frame_coherence_for_update,
        skip_low_coherence_updates=args.skip_low_coherence_updates,
        enable_dof_locking=args.enable_dof_locking,
        root_flight_dynamics=False,
        flight_height_threshold_m=args.flight_height_threshold_m,
        flight_min_consecutive_frames=args.flight_min_consecutive_frames,
        initial_state=shared_initial_state,
    )

    ekf_result_acc, ekf_acc_timings = run_ekf(
        biomod_path=None,
        calibrations=calibrations,
        pose_data=pose_data,
        reconstruction=reconstruction,
        fps=args.fps,
        measurement_noise_scale=args.measurement_noise_scale,
        process_noise_scale=args.process_noise_scale,
        coherence_confidence_floor=args.coherence_confidence_floor,
        min_frame_coherence_for_update=args.min_frame_coherence_for_update,
        skip_low_coherence_updates=args.skip_low_coherence_updates,
        enable_dof_locking=args.enable_dof_locking,
        root_flight_dynamics=False,
        flight_height_threshold_m=args.flight_height_threshold_m,
        flight_min_consecutive_frames=args.flight_min_consecutive_frames,
        unwrap_root=not args.no_root_unwrap,
        initial_state=shared_initial_state,
        model=shared_biorbd_model,
    )
    stage_timings_s["ekf_2d_acc_init_s"] = ekf_acc_timings["init_s"]
    stage_timings_s["ekf_2d_acc_loop_s"] = ekf_acc_timings["loop_s"]
    stage_timings_s["ekf_2d_acc_s"] = ekf_acc_timings["init_s"] + ekf_acc_timings["loop_s"]
    stage_timings_s["ekf_2d_acc_predict_s"] = ekf_acc_timings.get("predict_s", 0.0)
    stage_timings_s["ekf_2d_acc_update_s"] = ekf_acc_timings.get("update_s", 0.0)
    stage_timings_s["ekf_2d_acc_markers_s"] = ekf_acc_timings.get("markers_s", 0.0)
    stage_timings_s["ekf_2d_acc_marker_jacobians_s"] = ekf_acc_timings.get("marker_jacobians_s", 0.0)
    stage_timings_s["ekf_2d_acc_assembly_s"] = ekf_acc_timings.get("assembly_s", 0.0)
    stage_timings_s["ekf_2d_acc_solve_s"] = ekf_acc_timings.get("solve_s", 0.0)
    ekf_result_dyn = None
    ekf_result_flip_acc = None
    ekf_result_flip_dyn = None
    if args.root_flight_dynamics or args.debug_ekf_2d_dyn or args.run_ekf_2d_flip_dyn:
        print("Note: les variantes EKF 2D DYN sont temporairement desactivees; seul EKF 2D ACC est execute.")
    if args.run_ekf_2d_flip_acc:
        stage_timings_s["ekf_warmup_flip_acc_s"] = warmup_ekf_runtime(
            model=shared_biorbd_model,
            calibrations=calibrations,
            pose_data=pose_data_flip,
            reconstruction=reconstruction,
            fps=args.fps,
            measurement_noise_scale=args.measurement_noise_scale,
            process_noise_scale=args.process_noise_scale,
            coherence_confidence_floor=args.coherence_confidence_floor,
            min_frame_coherence_for_update=args.min_frame_coherence_for_update,
            skip_low_coherence_updates=args.skip_low_coherence_updates,
            enable_dof_locking=args.enable_dof_locking,
            root_flight_dynamics=False,
            flight_height_threshold_m=args.flight_height_threshold_m,
            flight_min_consecutive_frames=args.flight_min_consecutive_frames,
            initial_state=shared_initial_state_flip_acc,
        )
        ekf_result_flip_acc, ekf_flip_acc_timings = run_ekf(
            biomod_path=None,
            calibrations=calibrations,
            pose_data=pose_data_flip,
            reconstruction=reconstruction,
            fps=args.fps,
            measurement_noise_scale=args.measurement_noise_scale,
            process_noise_scale=args.process_noise_scale,
            coherence_confidence_floor=args.coherence_confidence_floor,
            min_frame_coherence_for_update=args.min_frame_coherence_for_update,
            skip_low_coherence_updates=args.skip_low_coherence_updates,
            enable_dof_locking=args.enable_dof_locking,
            root_flight_dynamics=False,
            flight_height_threshold_m=args.flight_height_threshold_m,
            flight_min_consecutive_frames=args.flight_min_consecutive_frames,
            unwrap_root=not args.no_root_unwrap,
            initial_state=shared_initial_state_flip_acc,
            model=shared_biorbd_model,
        )
        stage_timings_s["ekf_2d_flip_acc_init_s"] = ekf_flip_acc_timings["init_s"]
        stage_timings_s["ekf_2d_flip_acc_loop_s"] = ekf_flip_acc_timings["loop_s"]
        stage_timings_s["ekf_2d_flip_acc_s"] = ekf_flip_acc_timings["init_s"] + ekf_flip_acc_timings["loop_s"]
        stage_timings_s["ekf_2d_flip_acc_predict_s"] = ekf_flip_acc_timings.get("predict_s", 0.0)
        stage_timings_s["ekf_2d_flip_acc_update_s"] = ekf_flip_acc_timings.get("update_s", 0.0)
        stage_timings_s["ekf_2d_flip_acc_markers_s"] = ekf_flip_acc_timings.get("markers_s", 0.0)
        stage_timings_s["ekf_2d_flip_acc_marker_jacobians_s"] = ekf_flip_acc_timings.get("marker_jacobians_s", 0.0)
        stage_timings_s["ekf_2d_flip_acc_assembly_s"] = ekf_flip_acc_timings.get("assembly_s", 0.0)
        stage_timings_s["ekf_2d_flip_acc_solve_s"] = ekf_flip_acc_timings.get("solve_s", 0.0)
    comparison_acc = None
    comparison_dyn = None
    comparison_flip_acc = None
    comparison_flip_dyn = None
    dyn_vs_acc_diagnostics = None
    classic_result = None
    if args.compare_biorbd_kalman:
        biorbd_metadata = biorbd_kalman_cache_metadata(
            selected_reconstruction_cache_path,
            reconstruction,
            biomod_path,
            args.fps,
            args.biorbd_kalman_noise_factor,
            args.biorbd_kalman_error_factor,
            args.biorbd_kalman_init_method,
        )
        if metadata_cache_matches(biorbd_kalman_cache_path, biorbd_metadata):
            t0 = time.perf_counter()
            classic_result = load_biorbd_kalman_cache(biorbd_kalman_cache_path)
            stage_timings_s["ekf_3d_s"] = time.perf_counter() - t0
        else:
            import biorbd

            t0 = time.perf_counter()
            classic_result = run_biorbd_marker_kalman_with_parameters(
                biorbd.Model(str(biomod_path)),
                reconstruction,
                args.fps,
                noise_factor=args.biorbd_kalman_noise_factor,
                error_factor=args.biorbd_kalman_error_factor,
                unwrap_root=not args.no_root_unwrap,
                initial_state_method=args.biorbd_kalman_init_method,
            )
            save_biorbd_kalman_cache(biorbd_kalman_cache_path, classic_result, biorbd_metadata)
            stage_timings_s["ekf_3d_s"] = time.perf_counter() - t0

        comparison_acc = compare_kalman_filters(
            biomod_path,
            calibrations,
            pose_data,
            reconstruction,
            ekf_result_acc,
            fps=args.fps,
            biorbd_kalman_noise_factor=args.biorbd_kalman_noise_factor,
            biorbd_kalman_error_factor=args.biorbd_kalman_error_factor,
            biorbd_kalman_init_method=args.biorbd_kalman_init_method,
            classic_result=classic_result,
            unwrap_root=not args.no_root_unwrap,
        )
        if ekf_result_dyn is not None:
            comparison_dyn = compare_kalman_filters(
                biomod_path,
                calibrations,
                pose_data,
                reconstruction,
                ekf_result_dyn,
                fps=args.fps,
                biorbd_kalman_noise_factor=args.biorbd_kalman_noise_factor,
                biorbd_kalman_error_factor=args.biorbd_kalman_error_factor,
                classic_result=classic_result,
                unwrap_root=not args.no_root_unwrap,
            )
        if ekf_result_flip_acc is not None:
            comparison_flip_acc = compare_kalman_filters(
                biomod_path,
                calibrations,
                pose_data_flip,
                reconstruction,
                ekf_result_flip_acc,
                fps=args.fps,
                biorbd_kalman_noise_factor=args.biorbd_kalman_noise_factor,
                biorbd_kalman_error_factor=args.biorbd_kalman_error_factor,
                classic_result=classic_result,
                unwrap_root=not args.no_root_unwrap,
            )
        if ekf_result_flip_dyn is not None:
            comparison_flip_dyn = compare_kalman_filters(
                biomod_path,
                calibrations,
                pose_data_flip,
                reconstruction,
                ekf_result_flip_dyn,
                fps=args.fps,
                biorbd_kalman_noise_factor=args.biorbd_kalman_noise_factor,
                biorbd_kalman_error_factor=args.biorbd_kalman_error_factor,
                classic_result=classic_result,
                unwrap_root=not args.no_root_unwrap,
            )
    animation_target = (
        try_export_pyorerun_animation(biomod_path, ekf_result_acc["q"], args.fps, args.output_dir)
        if args.animate
        else None
    )
    dyn_vs_acc_diagnostics = compute_dyn_activation_and_root_qddot_diff(
        reconstruction=reconstruction,
        ekf_result_acc=ekf_result_acc,
        ekf_result_dyn=ekf_result_dyn,
        flight_height_threshold_m=args.flight_height_threshold_m,
        flight_min_consecutive_frames=args.flight_min_consecutive_frames,
        n_root=int(shared_biorbd_model.nbRoot()) if hasattr(shared_biorbd_model, "nbRoot") else 0,
    )
    save_outputs(
        args.output_dir,
        reconstruction,
        selected_reconstruction_cache_path,
        selected_reconstruction_metadata_dict,
        reconstruction_fast=reconstruction_fast,
        reconstruction_fast_cache_path=reconstruction_fast_cache_path,
        lengths=lengths,
        biomod_path=biomod_path,
        ekf_result_acc=ekf_result_acc,
        ekf_result_dyn=ekf_result_dyn,
        ekf_result_flip_acc=ekf_result_flip_acc,
        ekf_result_flip_dyn=ekf_result_flip_dyn,
        comparison_acc=comparison_acc,
        comparison_dyn=comparison_dyn,
        comparison_flip_acc=comparison_flip_acc,
        comparison_flip_dyn=comparison_flip_dyn,
        dyn_vs_acc_diagnostics=dyn_vs_acc_diagnostics,
        animation_target=animation_target,
        biorbd_kalman_noise_factor=args.biorbd_kalman_noise_factor,
        biorbd_kalman_error_factor=args.biorbd_kalman_error_factor,
        subject_mass_kg=args.subject_mass_kg,
        measurement_noise_scale=args.measurement_noise_scale,
        coherence_confidence_floor=args.coherence_confidence_floor,
        root_flight_dynamics=args.root_flight_dynamics,
        flight_height_threshold_m=args.flight_height_threshold_m,
        flight_min_consecutive_frames=args.flight_min_consecutive_frames,
        pose_data_mode=args.pose_data_mode,
        pose_filter_window=args.pose_filter_window,
        pose_outlier_threshold_ratio=args.pose_outlier_threshold_ratio,
        pose_amplitude_lower_percentile=args.pose_amplitude_lower_percentile,
        pose_amplitude_upper_percentile=args.pose_amplitude_upper_percentile,
        ekf2d_initial_state_method=args.ekf2d_initial_state_method,
        ekf2d_bootstrap_passes=args.ekf2d_bootstrap_passes,
        shared_initial_state_diagnostics=shared_initial_state_diagnostics,
        shared_initial_state_flip_acc_diagnostics=shared_initial_state_flip_acc_diagnostics,
        stage_timings_s=stage_timings_s,
        triangulation_comparison=triangulation_comparison,
        left_right_flip_diagnostics=left_right_flip_diagnostics,
    )

    print(f"Frames traitées: {pose_data.frames.shape[0]}")
    print(f"Erreur de reprojection moyenne: {np.nanmean(reconstruction.reprojection_error):.2f} px")
    print("")
    print("Temps de calcul")
    print(f"{'Etape':<24} {'Temps (s)':>10}")
    print(f"{'-' * 24} {'-' * 10}")
    print(f"{'Triangulation adaptive':<24} {stage_timings_s['triangulation_adaptive_s']:>10.2f}")
    print(f"{'Triangulation fast':<24} {stage_timings_s['triangulation_fast_s']:>10.2f}")
    print(f"{'Coherence epipolaire':<24} {stage_timings_s.get('epipolar_coherence_s', float('nan')):>10.2f}")
    print(f"{'Model creation':<24} {stage_timings_s.get('model_stage_s', float('nan')):>10.2f}")
    print(f"{'EKF initial state':<24} {stage_timings_s.get('ekf_initial_state_s', float('nan')):>10.2f}")
    if args.run_ekf_2d_flip_acc and "ekf_initial_state_flip_acc_s" in stage_timings_s:
        print(f"{'EKF init FLIP ACC':<24} {stage_timings_s.get('ekf_initial_state_flip_acc_s', float('nan')):>10.2f}")
    print(f"{'EKF warmup ACC':<24} {stage_timings_s.get('ekf_warmup_acc_s', float('nan')):>10.2f}")
    print(f"{'EKF 2D ACC':<24} {stage_timings_s.get('ekf_2d_acc_s', float('nan')):>10.2f}")
    print(f"{'EKF 2D ACC loop':<24} {stage_timings_s.get('ekf_2d_acc_loop_s', float('nan')):>10.2f}")
    if stage_timings_s.get("ekf_2d_acc_markers_s", 0.0) > 0.0:
        print(f"{'  ACC predict':<24} {stage_timings_s.get('ekf_2d_acc_predict_s', float('nan')):>10.2f}")
        print(f"{'  ACC update':<24} {stage_timings_s.get('ekf_2d_acc_update_s', float('nan')):>10.2f}")
        print(f"{'  ACC markers':<24} {stage_timings_s.get('ekf_2d_acc_markers_s', float('nan')):>10.2f}")
        print(f"{'  ACC jacobians':<24} {stage_timings_s.get('ekf_2d_acc_marker_jacobians_s', float('nan')):>10.2f}")
        print(f"{'  ACC assembly':<24} {stage_timings_s.get('ekf_2d_acc_assembly_s', float('nan')):>10.2f}")
        print(f"{'  ACC solve':<24} {stage_timings_s.get('ekf_2d_acc_solve_s', float('nan')):>10.2f}")
    if args.root_flight_dynamics:
        print(f"{'EKF warmup DYN':<24} {stage_timings_s.get('ekf_warmup_dyn_s', float('nan')):>10.2f}")
        print(f"{'EKF 2D DYN':<24} {stage_timings_s.get('ekf_2d_dyn_s', float('nan')):>10.2f}")
        print(f"{'EKF 2D DYN loop':<24} {stage_timings_s.get('ekf_2d_dyn_loop_s', float('nan')):>10.2f}")
    if args.run_ekf_2d_flip_acc:
        print(f"{'EKF warmup FLIP ACC':<24} {stage_timings_s.get('ekf_warmup_flip_acc_s', float('nan')):>10.2f}")
        print(f"{'EKF 2D FLIP ACC':<24} {stage_timings_s.get('ekf_2d_flip_acc_s', float('nan')):>10.2f}")
        print(f"{'EKF 2D FLIP ACC loop':<24} {stage_timings_s.get('ekf_2d_flip_acc_loop_s', float('nan')):>10.2f}")
    if args.run_ekf_2d_flip_dyn:
        print(f"{'EKF warmup FLIP DYN':<24} {stage_timings_s.get('ekf_warmup_flip_dyn_s', float('nan')):>10.2f}")
        print(f"{'EKF 2D FLIP DYN':<24} {stage_timings_s.get('ekf_2d_flip_dyn_s', float('nan')):>10.2f}")
        print(f"{'EKF 2D FLIP DYN loop':<24} {stage_timings_s.get('ekf_2d_flip_dyn_loop_s', float('nan')):>10.2f}")
    if args.compare_biorbd_kalman:
        print(f"{'EKF 3D':<24} {stage_timings_s.get('ekf_3d_s', float('nan')):>10.2f}")
    print("")
    q0_str = np.array2string(
        np.asarray(shared_initial_state[: shared_biorbd_model.nbQ()], dtype=float),
        precision=4,
        suppress_small=False,
        max_line_width=120,
    )
    print(
        "q0 initial "
        f"({args.ekf2d_initial_state_method}, frame "
        f"{shared_initial_state_diagnostics.get('bootstrap_frame_idx', '-')}"
        f"): {q0_str}"
    )
    print(
        "Ecart 3D adaptive vs fast: "
        f"{triangulation_comparison['mean_distance_m']:.4f} m (moyenne), "
        f"{triangulation_comparison['median_distance_m']:.4f} m (mediane)"
    )
    print(f"bioMod exporte: {biomod_path}")
    print(f"Outputs: {args.output_dir}")
    print(f"Cache modele: {model_cache_path}")
    if comparison_acc is not None:
        print(f"RMSE moyenne q (EKF 2D ACC vs EKF 3D): {np.mean(comparison_acc.rmse_per_dof):.5f}")
        print(
            f"Erreur de reprojection EKF 2D ACC: {comparison_acc.ekf_2d_reprojection_mean_px:.2f} +/- "
            f"{comparison_acc.ekf_2d_reprojection_std_px:.2f} px"
        )
        print(
            f"Erreur de reprojection EKF 3D: {comparison_acc.ekf_3d_reprojection_mean_px:.2f} +/- "
            f"{comparison_acc.ekf_3d_reprojection_std_px:.2f} px"
        )
    if comparison_dyn is not None:
        print(f"RMSE moyenne q (EKF 2D DYN vs EKF 3D): {np.mean(comparison_dyn.rmse_per_dof):.5f}")
        print(
            f"Erreur de reprojection EKF 2D DYN: {comparison_dyn.ekf_2d_reprojection_mean_px:.2f} +/- "
            f"{comparison_dyn.ekf_2d_reprojection_std_px:.2f} px"
        )
    if comparison_flip_acc is not None:
        print(f"RMSE moyenne q (EKF 2D FLIP ACC vs EKF 3D): {np.mean(comparison_flip_acc.rmse_per_dof):.5f}")
        print(
            f"Erreur de reprojection EKF 2D FLIP ACC: {comparison_flip_acc.ekf_2d_reprojection_mean_px:.2f} +/- "
            f"{comparison_flip_acc.ekf_2d_reprojection_std_px:.2f} px"
        )
    if comparison_flip_dyn is not None:
        print(f"RMSE moyenne q (EKF 2D FLIP DYN vs EKF 3D): {np.mean(comparison_flip_dyn.rmse_per_dof):.5f}")
        print(
            f"Erreur de reprojection EKF 2D FLIP DYN: {comparison_flip_dyn.ekf_2d_reprojection_mean_px:.2f} +/- "
            f"{comparison_flip_dyn.ekf_2d_reprojection_std_px:.2f} px"
        )
    missing_3d_mask = ~np.all(np.isfinite(reconstruction.points_3d), axis=2)
    print("")
    print("Diagnostic frames")
    print(
        "Triangulation 3D: "
        f"{int(np.any(missing_3d_mask, axis=1).sum())} frames avec au moins un keypoint manquant, "
        f"{int(np.all(missing_3d_mask, axis=1).sum())} frames sans aucun keypoint 3D"
    )
    acc_counts = ekf_result_acc.get("update_status_counts", {})
    print(
        "EKF 2D ACC: "
        f"{int(acc_counts.get('corrected', 0))} frames corrigees, "
        f"{int(acc_counts.get('pred_only_no_measurement', 0))} prediction seule (pas de mesure), "
        f"{int(acc_counts.get('pred_only_low_coherence', 0))} prediction seule (coherence faible), "
        f"{int(acc_counts.get('pred_only_cooldown', 0))} prediction seule (cooldown)"
    )
    if ekf_result_dyn is not None:
        dyn_counts = ekf_result_dyn.get("update_status_counts", {})
        print(
            "EKF 2D DYN: "
            f"{int(dyn_counts.get('corrected', 0))} frames corrigees, "
            f"{int(dyn_counts.get('pred_only_no_measurement', 0))} prediction seule (pas de mesure), "
            f"{int(dyn_counts.get('pred_only_low_coherence', 0))} prediction seule (coherence faible), "
            f"{int(dyn_counts.get('pred_only_cooldown', 0))} prediction seule (cooldown)"
        )
    if dyn_vs_acc_diagnostics is not None:
        print(
            "ACC vs DYN: "
            f"nbRoot = {int(dyn_vs_acc_diagnostics['n_root'])}, "
            f"{int(dyn_vs_acc_diagnostics['dyn_branch_activated_frames'])} frames avec branche DYN active, "
            f"||qddot_root_dyn - qddot_root_acc|| moyen = {dyn_vs_acc_diagnostics['qddot_root_dyn_minus_acc_norm_mean']:.6g}, "
            f"max = {dyn_vs_acc_diagnostics['qddot_root_dyn_minus_acc_norm_max']:.6g}, "
            f"non-zero = {int(dyn_vs_acc_diagnostics['qddot_root_dyn_minus_acc_norm_nonzero_frames'])}"
        )
        if int(dyn_vs_acc_diagnostics["n_root"]) == 0:
            print(
                "Avertissement: `biorbd` rapporte nbRoot = 0 pour ce bioMod, donc la branche DYN n'agit actuellement sur aucun DoF racine."
            )
    if ekf_result_flip_acc is not None:
        flip_acc_counts = ekf_result_flip_acc.get("update_status_counts", {})
        print(
            "EKF 2D FLIP ACC: "
            f"{int(flip_acc_counts.get('corrected', 0))} frames corrigees, "
            f"{int(flip_acc_counts.get('pred_only_no_measurement', 0))} prediction seule (pas de mesure), "
            f"{int(flip_acc_counts.get('pred_only_low_coherence', 0))} prediction seule (coherence faible), "
            f"{int(flip_acc_counts.get('pred_only_cooldown', 0))} prediction seule (cooldown)"
        )
    if ekf_result_flip_dyn is not None:
        flip_dyn_counts = ekf_result_flip_dyn.get("update_status_counts", {})
        print(
            "EKF 2D FLIP DYN: "
            f"{int(flip_dyn_counts.get('corrected', 0))} frames corrigees, "
            f"{int(flip_dyn_counts.get('pred_only_no_measurement', 0))} prediction seule (pas de mesure), "
            f"{int(flip_dyn_counts.get('pred_only_low_coherence', 0))} prediction seule (coherence faible), "
            f"{int(flip_dyn_counts.get('pred_only_cooldown', 0))} prediction seule (cooldown)"
        )
    print(
        "Flip left/right suspect: "
        f"{int(left_right_flip_diagnostics['n_frames_with_any_flip_suspect'])} frames suspectes, "
        f"{int(left_right_flip_diagnostics['n_camera_frame_flip_suspects'])} couples camera-frame suspects"
    )
    print(
        "  seuils flip: "
        f"ratio={float(left_right_flip_diagnostics.get('improvement_ratio', float('nan'))):.2f}, "
        f"gain_min={float(left_right_flip_diagnostics.get('min_gain_px', float('nan'))):.2f}px, "
        f"outlier={'oui' if bool(left_right_flip_diagnostics.get('restrict_to_outliers', True)) else 'non'}, "
        f"Q={float(left_right_flip_diagnostics.get('outlier_percentile', float('nan'))):.1f}, "
        f"floor={float(left_right_flip_diagnostics.get('outlier_floor_px', float('nan'))):.1f}px, "
        f"tau={float(left_right_flip_diagnostics.get('tau_px', float('nan'))):.1f}px, "
        f"temp_w={float(left_right_flip_diagnostics.get('temporal_weight', float('nan'))):.2f}, "
        f"temp_tau={float(left_right_flip_diagnostics.get('temporal_tau_px', float('nan'))):.1f}px"
    )
    if args.animate and animation_target is None:
        print("Animation pyorerun non exportee: pyorerun indisponible dans l'environnement.")


if __name__ == "__main__":
    main()
