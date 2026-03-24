from pathlib import Path

import numpy as np

from reconstruction.reconstruction_bundle import parse_trc_points


def _write_trc(path: Path, marker_row: str, data_row: str) -> Path:
    """Write a minimal TRC fixture to disk."""

    content = "\n".join(
        [
            f"PathFileType\t4\t(X/Y/Z)\t{path.name}",
            "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
            "120\t120\t1\t2\tm\t120\t0\t1",
            marker_row,
            "\t\tX1\tY1\tZ1\tX2\tY2\tZ2",
            data_row,
        ]
    )
    path.write_text(content, encoding="utf-8")
    return path


def test_parse_trc_points_accepts_pose2sim_header_names(tmp_path: Path):
    trc_path = _write_trc(
        tmp_path / "pose2sim_style.trc",
        "Frame#\tTime\tNose\tL Hip",
        "0\t0.0\t0.1\t0.2\t0.3\t1.0\t1.1\t1.2",
    )

    frames, time_s, points_3d, data_rate = parse_trc_points(trc_path)

    np.testing.assert_array_equal(frames, np.array([0], dtype=int))
    np.testing.assert_allclose(time_s, np.array([0.0], dtype=float))
    assert data_rate == 120.0
    np.testing.assert_allclose(points_3d[0, 0], np.array([0.1, 0.2, 0.3], dtype=float))
    np.testing.assert_allclose(points_3d[0, 11], np.array([1.0, 1.1, 1.2], dtype=float))


def test_parse_trc_points_accepts_exported_marker_names_with_underscores(tmp_path: Path):
    trc_path = _write_trc(
        tmp_path / "export_style.trc",
        "Frame#\tTime\tleft_hip\tright_shoulder",
        "0\t0.0\t0.4\t0.5\t0.6\t1.3\t1.4\t1.5",
    )

    frames, time_s, points_3d, data_rate = parse_trc_points(trc_path)

    np.testing.assert_array_equal(frames, np.array([0], dtype=int))
    np.testing.assert_allclose(time_s, np.array([0.0], dtype=float))
    assert data_rate == 120.0
    np.testing.assert_allclose(points_3d[0, 11], np.array([0.4, 0.5, 0.6], dtype=float))
    np.testing.assert_allclose(points_3d[0, 6], np.array([1.3, 1.4, 1.5], dtype=float))
