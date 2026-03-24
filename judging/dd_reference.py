#!/usr/bin/env python3
"""Reference-code helpers for DD analysis."""

from __future__ import annotations

import json
from pathlib import Path


def default_dd_reference_path(keypoints_path: Path) -> Path:
    """Return the default `*_DD.json` path associated with one keypoints file."""

    if keypoints_path.name.endswith("_keypoints.json"):
        stem = keypoints_path.name[: -len("_keypoints.json")]
        return keypoints_path.with_name(f"{stem}_DD.json")
    return keypoints_path.with_name(f"{keypoints_path.stem}_DD.json")


def load_dd_reference_codes(path: Path) -> dict[int, str]:
    """Load expected DD codes from a JSON file.

    Supported payloads:
    - `{"jumps": [{"jump": 1, "code": "821o"}, ...]}`
    - `{"jumps": ["821o", "42/", ...]}`
    - `{"1": "821o", "2": "42/"}` or `{"jump_1": "821o", ...}`
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("jumps"), list):
        jumps = payload["jumps"]
        if all(isinstance(item, str) for item in jumps):
            return {index: str(code).strip() for index, code in enumerate(jumps, start=1) if str(code).strip()}
        result: dict[int, str] = {}
        for item in jumps:
            if not isinstance(item, dict):
                continue
            jump_index = int(item.get("jump"))
            code = str(item.get("code", "")).strip()
            if jump_index >= 1 and code:
                result[jump_index] = code
        return result
    if isinstance(payload, dict):
        result = {}
        for raw_key, raw_value in payload.items():
            key = str(raw_key).strip().lower().replace("jump", "").replace("_", "").replace("-", "")
            if not key.isdigit():
                continue
            code = str(raw_value).strip()
            if code:
                result[int(key)] = code
        return result
    raise ValueError("Unsupported DD reference JSON format.")
