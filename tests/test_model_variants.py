from pathlib import Path

from reconstruction.reconstruction_registry import default_model_stem
from vitpose_ekf_pipeline import SegmentLengths, build_biomod


def test_default_model_stem_distinguishes_back_variant():
    stem_default = default_model_stem("cleaned", "exhaustive")
    stem_back = default_model_stem("cleaned", "exhaustive", model_variant="back_3dof")

    assert stem_default == "model_2d_cleaned_exhaustive"
    assert stem_back == "model_2d_cleaned_exhaustive_back_3dof"


def test_build_biomod_back_3dof_creates_upper_back_segment(tmp_path: Path):
    output_path = tmp_path / "back_3dof.bioMod"
    lengths = SegmentLengths(
        trunk_height=0.6,
        head_length=0.2,
        shoulder_half_width=0.18,
        hip_half_width=0.12,
        upper_arm_length=0.3,
        forearm_length=0.25,
        thigh_length=0.45,
        shank_length=0.4,
        eye_offset_x=0.03,
        eye_offset_y=0.025,
        ear_offset_y=0.06,
    )

    build_biomod(lengths, output_path, model_variant="back_3dof")

    text = output_path.read_text()
    assert "segment\tUPPER_BACK" in text
    assert "parent\tTRUNK" in text
    assert "marker\tleft_shoulder" in text
    assert "parent\tUPPER_BACK" in text
