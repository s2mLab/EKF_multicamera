from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vitpose_ekf_pipeline import (
    COCO17,
    KP_INDEX,
    MultiViewKinematicEKF,
    PoseData,
    ReconstructionResult,
    apply_measurement_update_batch,
    canonicalize_model_q_rotation_branches,
    stack_measurement_blocks,
)


@dataclass(frozen=True)
class AnnotationKinematicStateInfo:
    state: np.ndarray | None
    source_frame: int | None
    is_exact: bool


def _canonicalize_annotation_q(model, q_values: np.ndarray) -> np.ndarray:
    """Canonicalize q when the model exposes full segment metadata, else keep q as-is."""

    q_values = np.asarray(q_values, dtype=float).reshape(-1)
    if not hasattr(model, "nbSegment"):
        return np.array(q_values, copy=True)
    try:
        return canonicalize_model_q_rotation_branches(model, q_values)
    except Exception:
        return np.array(q_values, copy=True)


def resolve_annotation_kinematic_state_info(
    frame_states: dict[tuple[str, int], np.ndarray] | None,
    *,
    model_label: str,
    frame_number: int,
    model,
) -> AnnotationKinematicStateInfo:
    """Return the normalized exact or nearest saved state for one model/frame."""

    model_label = str(model_label).strip()
    if not model_label:
        return AnnotationKinematicStateInfo(state=None, source_frame=None, is_exact=False)
    frame_states = frame_states or {}
    exact = frame_states.get((model_label, int(frame_number)))
    if exact is not None:
        normalized = normalize_annotation_kinematic_state(model, exact)
        return AnnotationKinematicStateInfo(
            state=normalized,
            source_frame=int(frame_number) if normalized is not None else None,
            is_exact=normalized is not None,
        )
    candidates = [
        (
            abs(int(saved_frame) - int(frame_number)),
            int(saved_frame),
            normalize_annotation_kinematic_state(model, saved_state),
        )
        for (saved_model, saved_frame), saved_state in frame_states.items()
        if str(saved_model) == model_label
    ]
    candidates = [item for item in candidates if item[2] is not None]
    if not candidates:
        return AnnotationKinematicStateInfo(state=None, source_frame=None, is_exact=False)
    candidates.sort(key=lambda item: item[0])
    _distance, source_frame, source_state = candidates[0]
    return AnnotationKinematicStateInfo(
        state=np.asarray(source_state, dtype=float),
        source_frame=int(source_frame),
        is_exact=False,
    )


def store_annotation_kinematic_state(
    frame_states: dict[tuple[str, int], np.ndarray] | None,
    *,
    model_label: str,
    frame_number: int,
    model,
    state: np.ndarray,
) -> np.ndarray:
    """Normalize one state, store it in-place, and return the stored value."""

    normalized_state = normalize_annotation_kinematic_state(model, state)
    if normalized_state is None:
        raise ValueError("Invalid kinematic state.")
    if frame_states is None:
        raise ValueError("frame_states must be initialized before storing.")
    frame_states[(str(model_label).strip(), int(frame_number))] = np.asarray(normalized_state, dtype=float)
    return np.asarray(normalized_state, dtype=float)


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


def annotation_relevant_q_mask(
    q_names: list[str] | np.ndarray,
    keypoint_name: str,
) -> np.ndarray:
    """Return a boolean mask selecting the DoFs relevant to one annotated keypoint."""

    prefixes = annotation_relevant_q_prefixes(keypoint_name)
    return np.asarray(
        [any(str(q_name).startswith(f"{prefix}:") for prefix in prefixes) for q_name in q_names],
        dtype=bool,
    )


def constrain_annotation_state_to_mask(
    state: np.ndarray,
    reference_state: np.ndarray,
    active_q_mask: np.ndarray | None,
) -> np.ndarray:
    """Freeze inactive generalized coordinates and their derivatives to the reference state."""

    state = np.asarray(state, dtype=float).reshape(-1)
    reference_state = np.asarray(reference_state, dtype=float).reshape(-1)
    if active_q_mask is None:
        return np.array(state, copy=True)
    active_q_mask = np.asarray(active_q_mask, dtype=bool).reshape(-1)
    if active_q_mask.size == 0:
        return np.array(reference_state, copy=True)
    nq = active_q_mask.size
    constrained = np.array(state, copy=True)
    inactive = ~active_q_mask
    constrained[:nq][inactive] = reference_state[:nq][inactive]
    if constrained.size >= 2 * nq:
        constrained[nq : 2 * nq][inactive] = reference_state[nq : 2 * nq][inactive]
    if constrained.size >= 3 * nq:
        constrained[2 * nq : 3 * nq][inactive] = reference_state[2 * nq : 3 * nq][inactive]
    return constrained


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
    q_names: list[str] | np.ndarray | None = None,
    keypoint_name: str | None = None,
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
    active_q_mask = (
        annotation_relevant_q_mask(q_names or [], keypoint_name)
        if keypoint_name is not None and q_names is not None
        else None
    )
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
        state_iter[:nq] = _canonicalize_annotation_q(model, state_iter[:nq])
        predicted_state, predicted_covariance = ekf.predict(state_iter, covariance, 0)
        corrected_state, corrected_covariance, update_status = ekf.update(predicted_state, predicted_covariance, 0)
        diagnostics["update_statuses"].append(str(update_status))
        diagnostics["completed_passes"] = int(diagnostics["completed_passes"]) + 1
        if update_status != "corrected" or not np.all(np.isfinite(corrected_state)):
            diagnostics["used_fallback"] = True
            diagnostics["reason"] = f"local_{update_status}"
            return np.array(seed_state, copy=True), diagnostics
        corrected_state = np.asarray(corrected_state, dtype=float).reshape(3 * nq)
        corrected_state = constrain_annotation_state_to_mask(corrected_state, seed_state, active_q_mask)
        corrected_state[:nq] = _canonicalize_annotation_q(model, corrected_state[:nq])
        q_delta_norm = float(np.linalg.norm(corrected_state[:nq] - state_iter[:nq]))
        diagnostics.setdefault("q_delta_norms", []).append(q_delta_norm)
        state = corrected_state
        covariance = corrected_covariance
        if q_delta_norm <= 1e-6:
            diagnostics["converged"] = True
            break
    return np.array(state, copy=True), diagnostics


def refine_annotation_q_with_direct_measurements(
    *,
    model,
    calibrations: dict[str, object],
    pose_data: PoseData,
    seed_state: np.ndarray,
    passes: int,
    measurement_std_px: float = 2.0,
    q_names: list[str] | np.ndarray | None = None,
    keypoint_name: str | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply a direct image-space correction from sparse annotated 2D points.

    This is a lightweight fallback/polish pass for annotation clicks when the
    regular local EKF path is too conservative or rejects the update.
    """

    nq = int(model.nbQ())
    nx = 3 * nq
    seed_state = np.asarray(seed_state, dtype=float).reshape(-1)
    if seed_state.size == nq:
        seed_state = np.concatenate((seed_state, np.zeros(nq, dtype=float), np.zeros(nq, dtype=float)))
    elif seed_state.size != nx:
        raise ValueError("seed_state must contain either q or [q, qdot, qddot].")
    marker_pairs = [
        (marker_idx, KP_INDEX[marker_name.to_string()])
        for marker_idx, marker_name in enumerate(model.markerNames())
        if marker_name.to_string() in KP_INDEX
    ]
    marker_pair_keypoint_indices = np.asarray([kp_idx for _, kp_idx in marker_pairs], dtype=int)
    if marker_pair_keypoint_indices.size == 0:
        return np.array(seed_state, copy=True), {
            "method": "annotation_direct_measurements",
            "requested_passes": int(max(1, passes)),
            "completed_passes": 0,
            "used_fallback": True,
            "reason": "no_marker_pairs",
        }

    state = np.array(seed_state, copy=True)
    state[:nq] = _canonicalize_annotation_q(model, state[:nq])
    active_q_mask = (
        annotation_relevant_q_mask(q_names or [], keypoint_name)
        if keypoint_name is not None and q_names is not None
        else None
    )
    covariance = np.eye(nx) * 1e-2
    identity_x = np.eye(nx)
    diagnostics: dict[str, object] = {
        "method": "annotation_direct_measurements",
        "requested_passes": int(max(1, passes)),
        "completed_passes": 0,
        "used_fallback": False,
        "converged": False,
    }

    for _pass_idx in range(max(1, int(passes))):
        q = state[:nq]
        marker_positions_all = model.markers(q)
        marker_jacobians_all = model.markersJacobian(q)
        marker_points = [marker_positions_all[marker_idx].to_array() for marker_idx, _ in marker_pairs]
        marker_jacobians = [marker_jacobians_all[marker_idx].to_array() for marker_idx, _ in marker_pairs]
        marker_points_array = np.asarray(marker_points, dtype=float)
        marker_jacobians_array = np.asarray(marker_jacobians, dtype=float)
        finite_marker_points = np.all(np.isfinite(marker_points_array), axis=1)
        measurement_blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

        for cam_idx, camera_name in enumerate(pose_data.camera_names):
            calibration = calibrations.get(str(camera_name))
            if calibration is None:
                continue
            frame_keypoints = np.asarray(pose_data.keypoints[cam_idx, 0], dtype=float)
            frame_scores = np.asarray(pose_data.scores[cam_idx, 0], dtype=float)
            valid_keypoints = (
                np.all(np.isfinite(frame_keypoints), axis=1) & np.isfinite(frame_scores) & (frame_scores > 0)
            )
            if not np.any(valid_keypoints):
                continue
            valid_pairs = finite_marker_points & valid_keypoints[marker_pair_keypoint_indices]
            if not np.any(valid_pairs):
                continue
            pair_indices = np.flatnonzero(valid_pairs)
            keypoint_indices = marker_pair_keypoint_indices[pair_indices]
            projected_uv, projected_jac = calibration.project_points_and_jacobians(marker_points_array[pair_indices])
            H_q_blocks = np.einsum("mab,mbq->maq", projected_jac, marker_jacobians_array[pair_indices], optimize=True)
            finite_pairs = np.all(np.isfinite(projected_uv), axis=1) & np.all(
                np.isfinite(H_q_blocks.reshape(H_q_blocks.shape[0], -1)), axis=1
            )
            if not np.any(finite_pairs):
                continue
            keypoint_indices = keypoint_indices[finite_pairs]
            projected_uv = projected_uv[finite_pairs]
            H_q_blocks = H_q_blocks[finite_pairs]
            measured_points = frame_keypoints[keypoint_indices]
            measured_scores = frame_scores[keypoint_indices]
            selected_mask = (
                np.all(np.isfinite(measured_points), axis=1) & np.isfinite(measured_scores) & (measured_scores > 0)
            )
            if not np.any(selected_mask):
                continue
            selected_scores = np.maximum(measured_scores[selected_mask], 1e-3)
            H_q_selected = np.asarray(H_q_blocks[selected_mask], dtype=float)
            if active_q_mask is not None:
                H_q_selected[..., ~active_q_mask] = 0.0
            variances = np.repeat((float(measurement_std_px) / selected_scores) ** 2, 2).astype(float, copy=False)
            measurement_blocks.append(
                (
                    measured_points[selected_mask].reshape(-1),
                    projected_uv[selected_mask].reshape(-1),
                    H_q_selected.reshape(-1, nq),
                    variances,
                )
            )

        stacked = stack_measurement_blocks(measurement_blocks, nq)
        if stacked is None:
            diagnostics["used_fallback"] = True
            diagnostics["reason"] = "no_measurement"
            return np.array(seed_state, copy=True), diagnostics
        update_result = apply_measurement_update_batch(
            predicted_state=state,
            predicted_covariance=covariance,
            z=stacked[0],
            h=stacked[1],
            H_q=stacked[2],
            R_diag_array=stacked[3],
            nq=nq,
            identity_x=identity_x,
        )
        if update_result is None:
            diagnostics["used_fallback"] = True
            diagnostics["reason"] = "singular_gain"
            return np.array(seed_state, copy=True), diagnostics
        updated_state, covariance = update_result
        updated_state = np.asarray(updated_state, dtype=float).reshape(nx)
        updated_state = constrain_annotation_state_to_mask(updated_state, seed_state, active_q_mask)
        updated_state[:nq] = _canonicalize_annotation_q(model, updated_state[:nq])
        q_delta_norm = float(np.linalg.norm(updated_state[:nq] - state[:nq]))
        diagnostics.setdefault("q_delta_norms", []).append(q_delta_norm)
        diagnostics["completed_passes"] = int(diagnostics["completed_passes"]) + 1
        state = updated_state
        if q_delta_norm <= 1e-7:
            diagnostics["converged"] = True
            break

    return np.array(state, copy=True), diagnostics


def refine_annotation_window_states(
    *,
    model,
    calibrations: dict[str, object],
    pose_data_by_frame: dict[int, PoseData],
    center_frame_number: int,
    seed_state: np.ndarray,
    fps: float,
    passes: int,
    epipolar_threshold_px: float,
    q_names: list[str] | np.ndarray | None = None,
) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    """Refine a short temporal window of sparse annotation states around one frame.

    The window is initialized from the center state, propagated to neighboring
    frames with the constant-acceleration model, then corrected frame-by-frame
    where annotated 2D measurements exist.
    """

    frame_numbers = sorted(int(frame_number) for frame_number in pose_data_by_frame)
    if not frame_numbers:
        return {}, {
            "method": "annotation_window_local_ekf",
            "requested_passes": int(max(1, passes)),
            "completed_frames": 0,
            "used_fallback": True,
            "reason": "no_frames",
        }

    center_frame_number = int(center_frame_number)
    if center_frame_number not in frame_numbers:
        raise ValueError("center_frame_number must be present in pose_data_by_frame.")

    center_state = normalize_annotation_kinematic_state(model, seed_state)
    if center_state is None:
        raise ValueError("Invalid seed_state for local annotation window.")

    refined_states: dict[int, np.ndarray] = {center_frame_number: np.asarray(center_state, dtype=float)}
    diagnostics: dict[str, object] = {
        "method": "annotation_window_local_ekf",
        "requested_passes": int(max(1, passes)),
        "completed_frames": 0,
        "used_fallback": False,
        "frame_statuses": {},
    }
    center_idx = frame_numbers.index(center_frame_number)

    def _refine_one_frame(frame_number: int, seed: np.ndarray) -> np.ndarray:
        pose_data = pose_data_by_frame.get(int(frame_number))
        if pose_data is None or float(np.sum(np.asarray(pose_data.scores) > 0.0)) <= 0.0:
            diagnostics["frame_statuses"][int(frame_number)] = "propagated"
            return np.asarray(seed, dtype=float)
        refined_state, frame_diag = refine_annotation_q_with_local_ekf(
            model=model,
            calibrations=calibrations,
            pose_data=pose_data,
            frame_number=int(frame_number),
            seed_state=np.asarray(seed, dtype=float),
            fps=float(fps),
            passes=max(1, int(passes)),
            measurement_noise_scale=1.0,
            process_noise_scale=1.0,
            epipolar_threshold_px=float(epipolar_threshold_px),
            q_names=q_names,
            keypoint_name=None,
        )
        diagnostics["completed_frames"] = int(diagnostics["completed_frames"]) + 1
        diagnostics["frame_statuses"][int(frame_number)] = str(
            "fallback" if frame_diag.get("used_fallback") else "corrected"
        )
        if bool(frame_diag.get("used_fallback")):
            return np.asarray(seed, dtype=float)
        return np.asarray(refined_state, dtype=float)

    state = refined_states[center_frame_number]
    for frame_number in frame_numbers[center_idx + 1 :]:
        previous_frame = frame_numbers[frame_numbers.index(frame_number) - 1]
        state = propagate_annotation_kinematic_state(
            model,
            state,
            dt=1.0 / float(fps),
            frame_delta=int(frame_number) - int(previous_frame),
        )
        state = _refine_one_frame(frame_number, state)
        refined_states[int(frame_number)] = np.asarray(state, dtype=float)

    state = refined_states[center_frame_number]
    for reverse_idx in range(center_idx - 1, -1, -1):
        frame_number = frame_numbers[reverse_idx]
        next_frame = frame_numbers[reverse_idx + 1]
        state = propagate_annotation_kinematic_state(
            model,
            state,
            dt=1.0 / float(fps),
            frame_delta=int(frame_number) - int(next_frame),
        )
        state = _refine_one_frame(frame_number, state)
        refined_states[int(frame_number)] = np.asarray(state, dtype=float)

    return refined_states, diagnostics
