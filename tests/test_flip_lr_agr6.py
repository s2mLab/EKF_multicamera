"""Tests du detecteur de flip gauche/droite `epipolar_agr6` (variante A6, sans terme temporel).

Squelette 3D synthetique projete dans les 8 cameras de `inputs/calibration/Calib.toml` (sans bruit de detection) ; une vue
inversee gauche/droite sur 10 images consecutives.
"""

from pathlib import Path

import numpy as np
import pytest

import vitpose_ekf_pipeline as V
from reconstruction.flip_lr_agr6 import (
    FLIP_AGR6_GAIN_PX,
    FLIP_AGR6_RATIO,
    MIRROR,
    detect_agr6,
    flip_agr6_parameters,
)
from reconstruction.reconstruction_bundle import flip_cache_metadata

CALIB = Path(V.__file__).resolve().parent / "inputs" / "calibration" / "Calib.toml"
KEYPOINT_WEIGHTS = np.asarray([V.FLIP_PROXIMAL_KEYPOINT_WEIGHTS.get(name, 1.0) for name in V.COCO17], dtype=float)

BASE_SKELETON = np.array(
    [
        [0.0, 0.0, 1.70],
        [0.03, 0.05, 1.72],
        [0.03, -0.05, 1.72],
        [0.0, 0.09, 1.70],
        [0.0, -0.09, 1.70],
        [0.0, 0.18, 1.45],
        [0.0, -0.18, 1.45],
        [0.05, 0.25, 1.22],
        [0.05, -0.25, 1.22],
        [0.12, 0.28, 1.0],
        [0.12, -0.28, 1.0],
        [0.0, 0.10, 1.0],
        [0.0, -0.10, 1.0],
        [0.08, 0.11, 0.55],
        [0.08, -0.11, 0.55],
        [0.0, 0.12, 0.1],
        [0.0, -0.12, 0.1],
    ]
)


def _skeleton(n_frames):
    rng = np.random.default_rng(0)
    out = np.zeros((n_frames, 17, 3))
    for f in range(n_frames):
        ang = 0.15 * f  # la rotation autour de la verticale change l'orientation du sujet d'une image a l'autre
        rot = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
        out[f] = BASE_SKELETON @ rot.T + np.array([1.0, 1.5, 0.0]) + 0.01 * rng.standard_normal((17, 3))
    return out


@pytest.fixture(scope="module")
def pose_and_calib():
    calib = V.load_calibrations(CALIB)
    names = sorted(calib)
    n_frames = 40
    skeleton = _skeleton(n_frames)
    points = np.stack(
        [
            np.array([[calib[name].project_point(skeleton[f, k]) for k in range(17)] for f in range(n_frames)])
            for name in names
        ]
    )
    scores = np.ones(points.shape[:3])
    return V.PoseData(camera_names=names, frames=np.arange(n_frames), keypoints=points, scores=scores), calib


def _detect(pose, calib):
    return detect_agr6(
        pose, calib, V.build_fundamental_matrix_array, V.build_flip_epipolar_pair_weight_array, KEYPOINT_WEIGHTS
    )


def test_agr6_flags_persistent_flip_without_false_alarm(pose_and_calib):
    pose, calib = pose_and_calib
    clean, _, _ = _detect(pose, calib)
    assert clean.shape == (len(pose.camera_names), len(pose.frames))
    assert clean.dtype == bool
    assert not clean.any()

    keypoints = np.array(pose.keypoints, copy=True)
    keypoints[2, 10:20] = keypoints[2, 10:20][:, MIRROR]
    flipped = V.PoseData(camera_names=pose.camera_names, frames=pose.frames, keypoints=keypoints, scores=pose.scores)
    mask, diagnostics, detail = _detect(flipped, calib)
    expected = np.zeros_like(mask)
    expected[2, 10:20] = True
    assert not (mask & ~expected).any()  # aucune fausse alerte (autres vues, images saines)
    assert mask.sum() >= 5  # candidats = percentile 85 de la camera : les images de fin de flip peuvent echapper
    assert diagnostics["method"] == "epipolar_agr6"
    assert diagnostics["temporal_weight"] == 0.0
    assert diagnostics["n_camera_frame_flip_suspects"] == int(mask.sum())
    assert diagnostics["variant_parameters"] == flip_agr6_parameters()
    assert detail["nominal_geometric_costs"].shape == mask.shape


def test_agr6_handles_missing_points(pose_and_calib):
    pose, calib = pose_and_calib
    keypoints = np.array(pose.keypoints, copy=True)
    scores = np.array(pose.scores, copy=True)
    keypoints[1, :, :] = np.nan
    scores[1, :, :] = 0.0
    mask, _, _ = _detect(V.PoseData(pose.camera_names, pose.frames, keypoints, scores), calib)
    assert not mask[1].any()


def test_dispatcher_routes_epipolar_agr6_and_is_selectable(pose_and_calib):
    pose, calib = pose_and_calib
    assert "epipolar_agr6" in V.SUPPORTED_FLIP_METHODS
    via_pipeline, diagnostics, _ = V.detect_left_right_flip_diagnostics(pose, calib, method="epipolar_agr6")
    direct, _, _ = _detect(pose, calib)
    assert (via_pipeline == direct).all()
    assert diagnostics["method"] == "epipolar_agr6"


def test_flip_cache_metadata_tracks_agr6_method_and_parameters(pose_and_calib):
    pose, calib = pose_and_calib
    common = dict(
        pose_data_mode="cleaned",
        pose_filter_window=9,
        pose_outlier_threshold_ratio=0.1,
        pose_amplitude_lower_percentile=5.0,
        pose_amplitude_upper_percentile=95.0,
        improvement_ratio=0.7,
        min_gain_px=3.0,
        min_other_cameras=2,
        restrict_to_outliers=True,
        outlier_percentile=85.0,
        outlier_floor_px=5.0,
        tau_px=None,
        temporal_weight=0.35,
        temporal_tau_px=5.0,
        temporal_min_valid_keypoints=3,
        calibrations=calib,
    )
    default = flip_cache_metadata(pose, method="epipolar", **common)
    agr6 = flip_cache_metadata(pose, method="epipolar_agr6", **common)
    assert agr6["method"] == "epipolar_agr6"
    assert agr6 != default
    parameters = agr6["flip_agr6_parameters"]
    assert parameters["improvement_ratio"] == FLIP_AGR6_RATIO
    assert parameters["min_gain_px"] == FLIP_AGR6_GAIN_PX
    assert "flip_agr6_parameters" not in default  # les caches des autres methodes restent valides
    assert agr6["epipolar_distance_mode"] == "sampson"
