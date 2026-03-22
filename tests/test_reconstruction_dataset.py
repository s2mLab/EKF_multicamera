from pathlib import Path

import numpy as np

from reconstruction.reconstruction_dataset import write_trc_file


def test_write_trc_file_outputs_expected_marker_layout(tmp_path: Path):
    marker_names = ["left_hip", "right_hip"]
    points = np.array(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
        ],
        dtype=float,
    )
    frames = np.array([10, 11], dtype=int)
    time_s = np.array([0.0, 1.0 / 120.0], dtype=float)

    output_path = write_trc_file(
        tmp_path / "markers.trc",
        marker_names,
        points,
        frames,
        time_s,
        data_rate=120.0,
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("PathFileType")
    assert "NumFrames" in lines[1]
    assert "left_hip" in lines[3]
    assert "right_hip" in lines[3]
    assert lines[5].startswith("10\t0.00000000")
    assert "0.10000000" in lines[5]
    assert lines[6].startswith("11\t0.00833333")
