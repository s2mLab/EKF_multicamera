#!/usr/bin/env python3
"""Presentation helpers for DD analysis views."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from judging.dd_analysis import DDJumpAnalysis, DDSessionAnalysis


@dataclass
class DDJumpPlotData:
    """Precomputed event annotations used by the DD plots."""

    local_t: np.ndarray
    full_salto_times: np.ndarray
    quarter_salto_times: np.ndarray
    quarter_salto_values: np.ndarray
    half_twist_times: np.ndarray
    half_twist_values: np.ndarray


def jump_list_label(index: int, jump: DDJumpAnalysis) -> str:
    """Build the compact label shown in the DD jump list."""

    return f"S{index} | som {jump.somersault_turns:.2f} | tw {jump.twist_turns:.2f}"


def format_dd_summary(
    analysis: DDSessionAnalysis | None,
    *,
    reconstruction_label_text: str | None,
    height_dof: str,
    angle_mode: str,
    fps: float,
) -> str:
    """Format the textual DD summary panel for the selected reconstruction."""

    if analysis is None or reconstruction_label_text is None:
        return "Aucune reconstruction sélectionnée.\n"

    lines = [
        f"Reconstruction: {reconstruction_label_text}",
        f"Height DoF: {height_dof}",
        f"Angle mode: {angle_mode}",
        f"Threshold: {analysis.height_threshold:.3f}",
        f"Jumps detected: {len(analysis.jumps)}",
        "",
    ]
    for idx, jump in enumerate(analysis.jumps, start=1):
        start_t = jump.segment.start / fps
        end_t = jump.segment.end / fps
        peak_h = float(np.nanmax(analysis.height[jump.segment.start : jump.segment.end + 1]))
        lines.append(
            f"S{idx}: frames {jump.segment.start}-{jump.segment.end} "
            f"({start_t:.2f}-{end_t:.2f} s), peak {peak_h:.2f} m"
        )
        lines.append(
            f"  somersault={jump.somersault_turns:.2f} turns | "
            f"twist={jump.twist_turns:.2f} turns | "
            f"tilt max={np.rad2deg(jump.max_tilt_rad):.1f} deg | "
            f"{jump.classification}"
        )
        code_text = jump.code if jump.code is not None else "-"
        body_shape = jump.body_shape if jump.body_shape is not None else "-"
        lines.append(f"  body shape={body_shape} | code={code_text}")
        if jump.twists_per_salto:
            twists_str = ", ".join(f"S{idx}: {value:.1f}" for idx, value in enumerate(jump.twists_per_salto, start=1))
            lines.append(f"  twists by salto=[{twists_str}]")
        lines.append("")
    return "\n".join(lines) + "\n"


def build_jump_plot_data(jump: DDJumpAnalysis, fps: float) -> DDJumpPlotData:
    """Extract event times and values used to annotate salto and twist curves."""

    local_t = np.arange(jump.somersault_curve_turns.shape[0], dtype=float) / float(fps)

    def event_times(indices: list[int]) -> np.ndarray:
        if not indices:
            return np.array([], dtype=float)
        idx = np.asarray(indices, dtype=int)
        valid = (idx >= 0) & (idx < local_t.shape[0])
        return local_t[idx[valid]]

    def event_values(indices: list[int], curve: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not indices:
            return np.array([], dtype=float), np.array([], dtype=float)
        idx = np.asarray(indices, dtype=int)
        valid = (idx >= 0) & (idx < curve.shape[0])
        idx = idx[valid]
        return local_t[idx], np.asarray(curve[idx], dtype=float)

    quarter_salto_times, quarter_salto_values = event_values(
        jump.quarter_salto_event_indices,
        jump.somersault_curve_turns,
    )
    half_twist_times, half_twist_values = event_values(
        jump.half_twist_event_indices,
        jump.twist_curve_turns,
    )
    return DDJumpPlotData(
        local_t=local_t,
        full_salto_times=event_times(jump.full_salto_event_indices),
        quarter_salto_times=quarter_salto_times,
        quarter_salto_values=quarter_salto_values,
        half_twist_times=half_twist_times,
        half_twist_values=half_twist_values,
    )
