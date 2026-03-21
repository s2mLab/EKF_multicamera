#!/usr/bin/env python3
"""Pure helpers for frame navigation in preview widgets."""

from __future__ import annotations


def frame_from_slider_click(x: float, width: int, from_value: float, to_value: float) -> int:
    safe_width = max(int(width), 1)
    ratio = min(max(float(x) / float(safe_width), 0.0), 1.0)
    return int(round(float(from_value) + ratio * (float(to_value) - float(from_value))))


def clamp_frame_index(frame: int, max_frame: int) -> int:
    return max(0, min(int(frame), int(max_frame)))


def step_frame_index(current: int, delta: int, max_frame: int) -> int:
    return clamp_frame_index(int(current) + int(delta), max_frame)
