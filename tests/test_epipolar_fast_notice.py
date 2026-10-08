"""The ``*_fast*`` epipolar modes are flagged as using the symmetric distance."""

import subprocess
import sys
from pathlib import Path

from vitpose_ekf_pipeline import (
    SUPPORTED_COHERENCE_METHODS,
    SUPPORTED_FLIP_METHODS,
    epipolar_distance_mode_for_method,
    epipolar_fast_notice,
)

ROOT = Path(__file__).resolve().parents[1]


def test_epipolar_distance_mode_for_all_supported_methods():
    for method in (*SUPPORTED_COHERENCE_METHODS, *SUPPORTED_FLIP_METHODS):
        mode = epipolar_distance_mode_for_method(method)
        if method.startswith("epipolar"):
            assert mode == ("symmetric" if "fast" in method else "sampson")
            if mode == "symmetric":
                sampson_equivalent = method.replace("_fast", "")
                assert sampson_equivalent in (*SUPPORTED_COHERENCE_METHODS, *SUPPORTED_FLIP_METHODS)
                assert epipolar_distance_mode_for_method(sampson_equivalent) == "sampson"
        else:
            assert mode is None
    assert epipolar_distance_mode_for_method(None) is None


def test_epipolar_fast_notice_only_for_symmetric_modes():
    assert epipolar_fast_notice("epipolar", "epipolar_viterbi") is None
    assert epipolar_fast_notice("triangulation_once", None) is None
    notice = epipolar_fast_notice("epipolar_fast_framewise", "epipolar_fast_viterbi")
    assert "coherence=epipolar_fast_framewise (Sampson: epipolar_framewise)" in notice
    assert "flip=epipolar_fast_viterbi (Sampson: epipolar_viterbi)" in notice
    assert "+3.7%" in notice


def test_export_cli_help_documents_symmetric_fast_modes():
    output = subprocess.run(
        [sys.executable, "export_reconstruction_bundle.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "symetrique" in output and "Sampson" in output
