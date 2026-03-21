#!/usr/bin/env python3
"""Helpers to score horizontal displacement on the trampoline bed."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dd_analysis import DDSessionAnalysis

X_MAX = 2.525
Y_MAX = 1.455
X_INNER = 1.01
Y_INNER = 0.582


@dataclass
class TrampolineContact:
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
    if abs(x) > X_MAX or abs(y) > Y_MAX:
        return float("nan")
    if abs(x) <= X_INNER and abs(y) <= Y_INNER:
        return 0.0
    if (abs(x) <= X_INNER and abs(y) > Y_INNER) or (abs(x) > X_INNER and abs(y) <= Y_INNER):
        return 0.1
    if abs(x) > X_INNER and abs(y) > Y_INNER:
        return 0.3
    return 0.2


def contact_segments_between_jumps(session: DDSessionAnalysis) -> list[tuple[int, int]]:
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

        left_foot = np.asarray(segment[:, 15, :2], dtype=float)
        right_foot = np.asarray(segment[:, 16, :2], dtype=float)
        left_valid = np.all(np.isfinite(left_foot), axis=1)
        right_valid = np.all(np.isfinite(right_foot), axis=1)
        left_center = np.nanmedian(left_foot[left_valid], axis=0) if np.any(left_valid) else np.array([np.nan, np.nan], dtype=float)
        right_center = np.nanmedian(right_foot[right_valid], axis=0) if np.any(right_valid) else np.array([np.nan, np.nan], dtype=float)
        left_penalty = trampoline_penalty_refined(float(left_center[0]), float(left_center[1])) if np.all(np.isfinite(left_center)) else float("nan")
        right_penalty = trampoline_penalty_refined(float(right_center[0]), float(right_center[1])) if np.all(np.isfinite(right_center)) else float("nan")
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
    penalties = [contact.penalty for contact in contacts if contact.penalty is not None]
    if not penalties:
        return 0.0
    return float(np.sum(penalties))
