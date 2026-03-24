import json
from pathlib import Path

from judging.dd_reference import default_dd_reference_path, load_dd_reference_codes


def test_default_dd_reference_path_matches_keypoints_name():
    keypoints_path = Path("inputs/keypoints/1_partie_0429_keypoints.json")
    assert default_dd_reference_path(keypoints_path) == Path("inputs/dd/1_partie_0429_DD.json")


def test_load_dd_reference_codes_from_structured_json(tmp_path: Path):
    path = tmp_path / "sample_DD.json"
    path.write_text(
        json.dumps(
            {
                "jumps": [
                    {"jump": 1, "code": "821o"},
                    {"jump": 2, "code": "42/"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_dd_reference_codes(path) == {1: "821o", 2: "42/"}
