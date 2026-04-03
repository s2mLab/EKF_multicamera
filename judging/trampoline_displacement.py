#!/usr/bin/env python3
"""Helpers to score horizontal displacement on the trampoline bed."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from judging.dd_analysis import DDSessionAnalysis

BED_X_MAX = 2.525
BED_Y_MAX = 1.455


@dataclass(frozen=True)
class TrampolineGeometry:
    """Reference 2D geometry of the judged trampoline zones."""

    center: np.ndarray
    cross: dict[str, np.ndarray]
    small_square: dict[str, np.ndarray]
    big_rectangle: dict[str, np.ndarray]


def _reference_marker_xy() -> dict[str, np.ndarray]:
    """Return the reference trampoline markers extracted from the TRC example."""

    return {
        "Center": np.array([0.043268113558143854, -0.05300411860982727], dtype=float),
        "Cross_1": np.array([-0.27952443398574717, -0.04696764728727904], dtype=float),
        "Cross_2": np.array([0.37652836700158177, -0.04795804673711549], dtype=float),
        "Cross_3": np.array([0.05005495872135084, -0.3823917151684797], dtype=float),
        "Cross_4": np.array([0.04356532271660445, 0.27947414593708636], dtype=float),
        "SmallSquare_1": np.array([-0.45678352093271957, -0.558066501261472], dtype=float),
        "SmallSquare_2": np.array([-0.46609020457361006, 0.46948830182957657], dtype=float),
        "SmallSquare_3": np.array([0.564131366062569, -0.5636934229359234], dtype=float),
        "SmallSquare_4": np.array([0.55429137445568, 0.4711246832757012], dtype=float),
        "BigRect_1": np.array([-0.9598132124496965, -0.5786406873658397], dtype=float),
        "BigRect_2": np.array([-0.9982785248343781, 0.47024547295091973], dtype=float),
        "BigRect_3": np.array([1.0873758536665161, -0.5780009194576564], dtype=float),
        "BigRect_4": np.array([1.08636698254981, 0.46828409287201767], dtype=float),
    }


def _reference_marker_xyz() -> dict[str, np.ndarray]:
    """Return the full reference trampoline markers extracted from the TRC example."""

    return {
        "Center": np.array([0.043268113558143854, -0.05300411860982727, 1.2440734991150493], dtype=float),
        "Cross_1": np.array([-0.27952443398574717, -0.04696764728727904, 1.2391880887638858], dtype=float),
        "Cross_2": np.array([0.37652836700158177, -0.04795804673711549, 1.2402354305239016], dtype=float),
        "Cross_3": np.array([0.05005495872135084, -0.3823917151684797, 1.2399942447962533], dtype=float),
        "Cross_4": np.array([0.04356532271660445, 0.27947414593708636, 1.245396056095114], dtype=float),
        "SmallSquare_1": np.array([-0.45678352093271957, -0.558066501261472, 1.2396003056682545], dtype=float),
        "SmallSquare_2": np.array([-0.46609020457361006, 0.46948830182957657, 1.2484197798034131], dtype=float),
        "SmallSquare_3": np.array([0.564131366062569, -0.5636934229359234, 1.2398627229190031], dtype=float),
        "SmallSquare_4": np.array([0.55429137445568, 0.4711246832757012, 1.2490164829519783], dtype=float),
        "BigRect_1": np.array([-0.9598132124496965, -0.5786406873658397, 1.2396327440493142], dtype=float),
        "BigRect_2": np.array([-0.9982785248343781, 0.47024547295091973, 1.2461274907323676], dtype=float),
        "BigRect_3": np.array([1.0873758536665161, -0.5780009194576564, 1.2474209618855885], dtype=float),
        "BigRect_4": np.array([1.08636698254981, 0.46828409287201767, 1.2550467553780242], dtype=float),
    }


def trampoline_geometry_from_reference() -> TrampolineGeometry:
    """Build a centered geometry model from the reference TRC marker set."""

    markers = _reference_marker_xy()
    center = markers["Center"]
    normalized = {name: value - center for name, value in markers.items()}
    return TrampolineGeometry(
        center=np.zeros(2, dtype=float),
        cross={
            "left": normalized["Cross_1"],
            "right": normalized["Cross_2"],
            "bottom": normalized["Cross_3"],
            "top": normalized["Cross_4"],
        },
        small_square={
            "bottom_left": normalized["SmallSquare_1"],
            "top_left": normalized["SmallSquare_2"],
            "bottom_right": normalized["SmallSquare_3"],
            "top_right": normalized["SmallSquare_4"],
        },
        big_rectangle={
            "bottom_left": normalized["BigRect_1"],
            "top_left": normalized["BigRect_2"],
            "bottom_right": normalized["BigRect_3"],
            "top_right": normalized["BigRect_4"],
        },
    )


def rectangle_half_extents(corners: dict[str, np.ndarray]) -> tuple[float, float]:
    """Return symmetric half-extents from a set of rectangle corner markers."""

    xy = np.asarray(list(corners.values()), dtype=float)
    return float(np.max(np.abs(xy[:, 0]))), float(np.max(np.abs(xy[:, 1])))


TRAMPOLINE_GEOMETRY = trampoline_geometry_from_reference()
X_INNER, Y_INNER = rectangle_half_extents(TRAMPOLINE_GEOMETRY.small_square)
X_MAX, Y_MAX = rectangle_half_extents(TRAMPOLINE_GEOMETRY.big_rectangle)
TRAMPOLINE_BED_HEIGHT_M = float(
    np.mean(np.asarray([point[2] for point in _reference_marker_xyz().values()], dtype=float))
)


@dataclass
class TrampolineContact:
    """Summary of one contact interval used for horizontal-displacement judging."""

    index: int
    start: int
    end: int
    center_frame: int
    left_x: float
    left_y: float
    right_x: float
    right_y: float
    x: float
    y: float
    penalty: float | None


def trampoline_penalty_refined(x: float, y: float) -> float:
    """Return the FIG-style landing penalty associated with one bed position."""

    if abs(x) > X_MAX or abs(y) > Y_MAX:
        return float("nan")
    if abs(x) <= X_INNER and abs(y) <= Y_INNER:
        return 0.0
    if (abs(x) <= X_INNER and abs(y) > Y_INNER) or (abs(x) > X_INNER and abs(y) <= Y_INNER):
        return 0.1
    if abs(x) > X_INNER and abs(y) > Y_INNER:
        return 0.3
    return 0.2


def judged_trampoline_zone_xy(x: float, y: float) -> np.ndarray | None:
    """Return the judged trampoline rectangle containing one contact point."""

    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    x = float(x)
    y = float(y)
    if abs(x) > X_MAX or abs(y) > Y_MAX:
        return None
    if abs(x) <= X_INNER and abs(y) <= Y_INNER:
        min_x, max_x = -X_INNER, X_INNER
        min_y, max_y = -Y_INNER, Y_INNER
    elif abs(x) <= X_INNER:
        min_x, max_x = -X_INNER, X_INNER
        if y >= 0.0:
            min_y, max_y = Y_INNER, Y_MAX
        else:
            min_y, max_y = -Y_MAX, -Y_INNER
    elif abs(y) <= Y_INNER:
        min_y, max_y = -Y_INNER, Y_INNER
        if x >= 0.0:
            min_x, max_x = X_INNER, X_MAX
        else:
            min_x, max_x = -X_MAX, -X_INNER
    else:
        if x >= 0.0:
            min_x, max_x = X_INNER, X_MAX
        else:
            min_x, max_x = -X_MAX, -X_INNER
        if y >= 0.0:
            min_y, max_y = Y_INNER, Y_MAX
        else:
            min_y, max_y = -Y_MAX, -Y_INNER
    return np.array(
        [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
        ],
        dtype=float,
    )


def contact_segments_between_jumps(session: DDSessionAnalysis) -> list[tuple[int, int]]:
    """Return the contact intervals located between consecutive DD jumps."""

    segments: list[tuple[int, int]] = []
    for current, following in zip(session.jump_segments[:-1], session.jump_segments[1:]):
        start = int(current.end)
        end = int(following.start)
        if end >= start:
            segments.append((start, end))
    return segments


def analyze_trampoline_contacts(
    session: DDSessionAnalysis,
    contact_series: np.ndarray,
) -> list[TrampolineContact]:
    """Score contact intervals from either XY contacts or full-body 3D marker trajectories."""

    points = np.asarray(contact_series, dtype=float)
    if points.ndim == 2 and points.shape[1] == 2:
        point_mode = "xy"
    elif points.ndim == 3 and points.shape[1] >= 17 and points.shape[2] >= 2:
        point_mode = "feet_3d"
    else:
        raise ValueError("contact_series must have shape (n_frames, 2) or (n_frames, 17, 3)")
    contacts: list[TrampolineContact] = []
    for idx, (start, end) in enumerate(contact_segments_between_jumps(session), start=1):
        segment = points[start : end + 1]
        if point_mode == "xy":
            finite_mask = np.all(np.isfinite(segment), axis=1)
            if not np.any(finite_mask):
                contacts.append(
                    TrampolineContact(
                        index=idx,
                        start=start,
                        end=end,
                        center_frame=(start + end) // 2,
                        left_x=float("nan"),
                        left_y=float("nan"),
                        right_x=float("nan"),
                        right_y=float("nan"),
                        x=float("nan"),
                        y=float("nan"),
                        penalty=None,
                    )
                )
                continue
            segment_finite = segment[finite_mask]
            center = np.nanmedian(segment_finite, axis=0)
            penalty = trampoline_penalty_refined(float(center[0]), float(center[1]))
            contacts.append(
                TrampolineContact(
                    index=idx,
                    start=start,
                    end=end,
                    center_frame=(start + end) // 2,
                    left_x=float(center[0]),
                    left_y=float(center[1]),
                    right_x=float(center[0]),
                    right_y=float(center[1]),
                    x=float(center[0]),
                    y=float(center[1]),
                    penalty=None if not np.isfinite(penalty) else float(penalty),
                )
            )
            continue

        # Keep the strongest foot penalty for now, which matches the current judging approximation.
        left_foot = np.asarray(segment[:, 15, :2], dtype=float)
        right_foot = np.asarray(segment[:, 16, :2], dtype=float)
        left_valid = np.all(np.isfinite(left_foot), axis=1)
        right_valid = np.all(np.isfinite(right_foot), axis=1)
        left_center = (
            np.nanmedian(left_foot[left_valid], axis=0)
            if np.any(left_valid)
            else np.array([np.nan, np.nan], dtype=float)
        )
        right_center = (
            np.nanmedian(right_foot[right_valid], axis=0)
            if np.any(right_valid)
            else np.array([np.nan, np.nan], dtype=float)
        )
        left_penalty = (
            trampoline_penalty_refined(float(left_center[0]), float(left_center[1]))
            if np.all(np.isfinite(left_center))
            else float("nan")
        )
        right_penalty = (
            trampoline_penalty_refined(float(right_center[0]), float(right_center[1]))
            if np.all(np.isfinite(right_center))
            else float("nan")
        )
        penalties = [value for value in (left_penalty, right_penalty) if np.isfinite(value)]
        if penalties:
            penalty = float(max(penalties))
            if np.isfinite(left_penalty) and left_penalty >= penalty:
                applied_center = left_center
            elif np.isfinite(right_penalty):
                applied_center = right_center
            else:
                applied_center = np.array([np.nan, np.nan], dtype=float)
        else:
            penalty = None
            applied_center = np.array([np.nan, np.nan], dtype=float)
        contacts.append(
            TrampolineContact(
                index=idx,
                start=start,
                end=end,
                center_frame=(start + end) // 2,
                left_x=float(left_center[0]),
                left_y=float(left_center[1]),
                right_x=float(right_center[0]),
                right_y=float(right_center[1]),
                x=float(applied_center[0]),
                y=float(applied_center[1]),
                penalty=penalty,
            )
        )
    return contacts


def total_trampoline_penalty(contacts: list[TrampolineContact]) -> float:
    """Sum all finite trampoline-contact penalties."""

    penalties = [contact.penalty for contact in contacts if contact.penalty is not None]
    if not penalties:
        return 0.0
    return float(np.sum(penalties))
