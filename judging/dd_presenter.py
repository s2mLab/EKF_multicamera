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


@dataclass
class DDReferenceComparison:
    """Summary of how one reconstruction matches the expected DD reference codes."""

    matched_count: int
    expected_count: int
    detected_count: int
    status: str
    detected_codes: list[str]
    expected_codes: list[str]


@dataclass
class DDCodeCharacterComparison:
    """Character-level DD comparison annotated with its semantic role."""

    role: str
    expected_char: str
    detected_char: str
    matches: bool


def compare_dd_to_reference(
    analysis: DDSessionAnalysis | None,
    expected_codes_by_jump: dict[int, str] | None,
) -> DDReferenceComparison:
    """Compare one DD analysis against the optional reference-code file."""

    detected_codes = [str(jump.code or "-") for jump in (analysis.jumps if analysis is not None else [])]
    expected_codes = [code for _, code in sorted((expected_codes_by_jump or {}).items())]
    if not expected_codes:
        return DDReferenceComparison(
            matched_count=0,
            expected_count=0,
            detected_count=len(detected_codes),
            status="no_reference",
            detected_codes=detected_codes,
            expected_codes=[],
        )
    expected_count = len(expected_codes)
    matched_count = sum(
        1
        for jump_idx, expected_code in sorted(expected_codes_by_jump.items())
        if analysis is not None
        and 1 <= jump_idx <= len(analysis.jumps)
        and str(analysis.jumps[jump_idx - 1].code or "") == str(expected_code)
    )
    if matched_count == expected_count and len(detected_codes) == expected_count:
        status = "exact"
    elif matched_count > 0:
        status = "partial"
    else:
        status = "mismatch"
    return DDReferenceComparison(
        matched_count=matched_count,
        expected_count=expected_count,
        detected_count=len(detected_codes),
        status=status,
        detected_codes=detected_codes,
        expected_codes=expected_codes,
    )


def split_dd_code(code: str | None) -> tuple[str, str, str]:
    """Split one DD code into somersault, twist, and body-shape parts."""

    raw_code = "" if code is None else str(code).strip()
    if not raw_code or raw_code == "-":
        return "", "", ""
    body_shape = raw_code[-1] if (not raw_code[-1].isdigit() and raw_code[-1] != "+") else ""
    core = raw_code[:-1] if body_shape else raw_code
    somersault = core[:2]
    twist = core[2:]
    return somersault, twist, body_shape


def compare_dd_code_characters(expected_code: str | None, detected_code: str | None) -> list[DDCodeCharacterComparison]:
    """Compare two DD codes character by character inside each semantic block."""

    expected_som, expected_twist, expected_body = split_dd_code(expected_code)
    detected_som, detected_twist, detected_body = split_dd_code(detected_code)
    comparisons: list[DDCodeCharacterComparison] = []
    for role, expected_part, detected_part in (
        ("somersault", expected_som, detected_som),
        ("twist", expected_twist, detected_twist),
        ("body", expected_body, detected_body),
    ):
        block_len = max(len(expected_part), len(detected_part))
        for idx in range(block_len):
            expected_char = expected_part[idx] if idx < len(expected_part) else "_"
            detected_char = detected_part[idx] if idx < len(detected_part) else "_"
            comparisons.append(
                DDCodeCharacterComparison(
                    role=role,
                    expected_char=expected_char,
                    detected_char=detected_char,
                    matches=expected_char == detected_char and expected_char != "_" and detected_char != "_",
                )
            )
    return comparisons


def dd_reference_status_text(comparison: DDReferenceComparison) -> str:
    """Format one short status label for the DD comparison table."""

    if comparison.status == "no_reference":
        return "No ref"
    return f"{comparison.matched_count}/{comparison.expected_count}"


def dd_reference_status_color(comparison: DDReferenceComparison) -> str:
    """Map a DD comparison status to a GUI-friendly color token."""

    return {
        "exact": "ok",
        "partial": "partial",
        "mismatch": "bad",
        "no_reference": "neutral",
    }.get(comparison.status, "neutral")


def jump_list_label(index: int, jump: DDJumpAnalysis) -> str:
    """Build the compact label shown in the DD jump list."""

    return f"S{index} | som {jump.somersault_turns:.2f} | tw {jump.twist_turns:.2f}"


def jump_list_label_with_reference(index: int, jump: DDJumpAnalysis, expected_code: str | None = None) -> str:
    """Build one jump-list label and append the expected DD code when available."""

    label = jump_list_label(index, jump)
    if expected_code:
        return f"{label} | ref {expected_code}"
    return label


def format_dd_summary(
    analysis: DDSessionAnalysis | None,
    *,
    reconstruction_label_text: str | None,
    height_dof: str,
    angle_mode: str,
    fps: float,
    expected_codes_by_jump: dict[int, str] | None = None,
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
        expected_code = (expected_codes_by_jump or {}).get(idx)
        if expected_code:
            match_text = "match" if jump.code == expected_code else "diff"
            lines.append(f"  reference code={expected_code} | {match_text}")
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
