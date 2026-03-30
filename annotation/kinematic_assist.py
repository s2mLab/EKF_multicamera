from __future__ import annotations

import numpy as np

from vitpose_ekf_pipeline import (
    COCO17,
    KP_INDEX,
    MultiViewKinematicEKF,
    PoseData,
    ReconstructionResult,
    canonicalize_model_q_rotation_branches,
)


def _canonicalize_annotation_q(model, q_values: np.ndarray) -> np.ndarray:
    """Canonicalize q when the model exposes full segment metadata, else keep q as-is."""

    q_values = np.asarray(q_values, dtype=float).reshape(-1)
    if not hasattr(model, "nbSegment"):
        return np.array(q_values, copy=True)
    try:
        return canonicalize_model_q_rotation_branches(model, q_values)
    except Exception:
        return np.array(q_values, copy=True)


def annotation_relevant_q_prefixes(keypoint_name: str) -> tuple[str, ...]:
    """Return the segment prefixes most relevant to one annotated keypoint."""

    name = str(keypoint_name)
    prefixes = ["TRUNK", "LOWER_TRUNK", "UPPER_BACK"]
    if name.startswith("left_"):
        prefixes.extend(
            ["LEFT_UPPER_ARM", "LEFT_LOWER_ARM"]
            if "shoulder" in name or "elbow" in name or "wrist" in name
            else ["LEFT_THIGH", "LEFT_SHANK"]
        )
    elif name.startswith("right_"):
        prefixes.extend(
            ["RIGHT_UPPER_ARM", "RIGHT_LOWER_ARM"]
            if "shoulder" in name or "elbow" in name or "wrist" in name
            else ["RIGHT_THIGH", "RIGHT_SHANK"]
        )
    else:
        prefixes.append("HEAD")
    return tuple(prefixes)


def annotation_blend_q_by_relevance(
    q_names: list[str] | np.ndarray,
    previous_q: np.ndarray | None,
    estimated_q: np.ndarray,
    keypoint_name: str,
) -> np.ndarray:
    """Keep unrelated DoFs from the previous state and update only the relevant subtree."""

    estimated_q = np.asarray(estimated_q, dtype=float).reshape(-1)
    if previous_q is None:
        return estimated_q
    previous_q = np.asarray(previous_q, dtype=float).reshape(-1)
    if previous_q.shape != estimated_q.shape:
        return estimated_q
    prefixes = annotation_relevant_q_prefixes(keypoint_name)
    blended = np.array(previous_q, copy=True)
    for idx, q_name in enumerate(q_names):
        q_name = str(q_name)
        if any(q_name.startswith(f"{prefix}:") for prefix in prefixes):
            blended[idx] = estimated_q[idx]
    return blended


def annotation_reconstruction_from_points(
    points_3d: np.ndarray, frame_number: int, n_cameras: int
) -> ReconstructionResult:
    """Wrap one annotation frame into a minimal ReconstructionResult."""

    points_3d = np.asarray(points_3d, dtype=float).reshape(1, len(COCO17), 3)
    return ReconstructionResult(
        frames=np.asarray([int(frame_number)], dtype=int),
        points_3d=points_3d,
        mean_confidence=np.full((1, len(COCO17)), np.nan, dtype=float),
        reprojection_error=np.full((1, len(COCO17)), np.nan, dtype=float),
        reprojection_error_per_view=np.full((1, len(COCO17), int(n_cameras)), np.nan, dtype=float),
        multiview_coherence=np.full((1, len(COCO17), int(n_cameras)), np.nan, dtype=float),
        epipolar_coherence=np.full((1, len(COCO17), int(n_cameras)), np.nan, dtype=float),
        triangulation_coherence=np.full((1, len(COCO17), int(n_cameras)), np.nan, dtype=float),
        excluded_views=np.ones((1, len(COCO17), int(n_cameras)), dtype=bool),
        coherence_method="epipolar_fast_framewise",
    )


def annotation_state_from_q(
    model,
    q_values: np.ndarray,
) -> np.ndarray:
    nq = int(model.nbQ())
    q_values = _canonicalize_annotation_q(model, np.asarray(q_values, dtype=float).reshape(nq))
    return np.concatenate((q_values, np.zeros(nq, dtype=float), np.zeros(nq, dtype=float)))


def normalize_annotation_kinematic_state(model, state: np.ndarray | None) -> np.ndarray | None:
    if state is None:
        return None
    nq = int(model.nbQ())
    array = np.asarray(state, dtype=float).reshape(-1)
    if array.size == nq:
        return annotation_state_from_q(model, array)
    if array.size != 3 * nq:
        return None
    normalized = np.array(array, copy=True)
    normalized[:nq] = _canonicalize_annotation_q(model, normalized[:nq])
    return normalized


def propagate_annotation_kinematic_state(
    model,
    state: np.ndarray,
    *,
    dt: float,
    frame_delta: int,
) -> np.ndarray:
    nq = int(model.nbQ())
    state = normalize_annotation_kinematic_state(model, state)
    if state is None:
        raise ValueError("Invalid kinematic state.")
    total_dt = float(dt) * float(frame_delta)
    propagated = np.array(state, copy=True)
    q = propagated[:nq]
    qdot = propagated[nq : 2 * nq]
    qddot = propagated[2 * nq :]
    propagated[:nq] = q + total_dt * qdot + 0.5 * total_dt * total_dt * qddot
    propagated[nq : 2 * nq] = qdot + total_dt * qddot
    propagated[:nq] = _canonicalize_annotation_q(model, propagated[:nq])
    return propagated


def refine_annotation_q_with_local_ekf(
    *,
    model,
    calibrations: dict[str, object],
    pose_data: PoseData,
    frame_number: int,
    seed_state: np.ndarray,
    fps: float,
    passes: int,
    measurement_noise_scale: float = 1.0,
    process_noise_scale: float = 1.0,
    epipolar_threshold_px: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Refine one frame from sparse 2D annotations using local EKF2D corrections."""

    nq = int(model.nbQ())
    seed_state = np.asarray(seed_state, dtype=float).reshape(-1)
    if seed_state.size == nq:
        seed_state = np.concatenate((seed_state, np.zeros(nq, dtype=float), np.zeros(nq, dtype=float)))
    elif seed_state.size != 3 * nq:
        raise ValueError("seed_state must contain either q or [q, qdot, qddot].")
    seed_state = np.array(seed_state, copy=True)
    seed_state[:nq] = _canonicalize_annotation_q(model, seed_state[:nq])
    reconstruction = annotation_reconstruction_from_points(
        np.full((len(COCO17), 3), np.nan, dtype=float),
        frame_number=frame_number,
        n_cameras=len(pose_data.camera_names),
    )
    ekf = MultiViewKinematicEKF(
        model=model,
        calibrations={str(name): calibrations[str(name)] for name in pose_data.camera_names},
        pose_data=pose_data,
        reconstruction=reconstruction,
        dt=1.0 / float(fps),
        measurement_noise_scale=float(measurement_noise_scale),
        process_noise_scale=float(process_noise_scale),
        min_frame_coherence_for_update=0.0,
        skip_low_coherence_updates=False,
        coherence_confidence_floor=0.0,
        epipolar_threshold_px=float(epipolar_threshold_px),
        enable_dof_locking=False,
        root_flight_dynamics=False,
    )
    state = np.array(seed_state, copy=True)
    covariance = np.eye(ekf.nx) * 1e-2
    diagnostics: dict[str, object] = {
        "method": "annotation_local_ekf",
        "requested_passes": int(max(1, passes)),
        "completed_passes": 0,
        "update_statuses": [],
        "converged": False,
        "used_fallback": False,
    }
    for _pass_idx in range(max(1, int(passes))):
        state_iter = np.array(state, copy=True)
        state_iter[:nq] = canonicalize_model_q_rotation_branches(model, state_iter[:nq])
        predicted_state, predicted_covariance = ekf.predict(state_iter, covariance, 0)
        corrected_state, corrected_covariance, update_status = ekf.update(predicted_state, predicted_covariance, 0)
        diagnostics["update_statuses"].append(str(update_status))
        diagnostics["completed_passes"] = int(diagnostics["completed_passes"]) + 1
        if update_status != "corrected" or not np.all(np.isfinite(corrected_state)):
            diagnostics["used_fallback"] = True
            diagnostics["reason"] = f"local_{update_status}"
            return np.array(seed_state, copy=True), diagnostics
        corrected_state = np.asarray(corrected_state, dtype=float).reshape(3 * nq)
        corrected_state[:nq] = _canonicalize_annotation_q(model, corrected_state[:nq])
        q_delta_norm = float(np.linalg.norm(corrected_state[:nq] - state_iter[:nq]))
        diagnostics.setdefault("q_delta_norms", []).append(q_delta_norm)
        state = corrected_state
        covariance = corrected_covariance
        if q_delta_norm <= 1e-6:
            diagnostics["converged"] = True
            break
    return np.array(state, copy=True), diagnostics
