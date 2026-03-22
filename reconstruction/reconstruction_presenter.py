#!/usr/bin/env python3
"""Helpers to present dataset reconstruction selections consistently."""

from __future__ import annotations


def bundle_available_reconstruction_names(
    bundle: dict[str, object],
    *,
    include_3d: bool = True,
    include_q: bool = True,
    include_q_root: bool = False,
) -> list[str]:
    """Collect reconstruction names exposed by a preview bundle."""

    names: set[str] = set()
    if include_3d:
        names.update(bundle.get("recon_3d", {}).keys())
    if include_q:
        names.update(bundle.get("recon_q", {}).keys())
    if include_q_root:
        names.update(bundle.get("recon_q_root", {}).keys())
    return sorted(str(name) for name in names)


def catalog_rows_for_names(
    catalog: list[dict[str, object]],
    available_names: list[str] | set[str],
    *,
    extra_rows: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Filter catalog rows to the reconstructions that are both cached and available."""

    available = {str(name) for name in available_names}
    rows: list[dict[str, object]] = list(extra_rows or [])
    rows.extend(row for row in catalog if row.get("cached") and str(row.get("name")) in available)
    return rows


def default_selection(
    available_names: list[str] | set[str],
    preferred_names: list[str],
    *,
    fallback_count: int,
) -> list[str]:
    """Choose preferred default selections with a fallback to the first available rows."""

    available = [str(name) for name in available_names]
    available_set = set(available)
    defaults = [name for name in preferred_names if name in available_set]
    return defaults or available[:fallback_count]
