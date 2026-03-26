from pathlib import Path

from reconstruction.reconstruction_registry import default_model_stem
from vitpose_ekf_pipeline import SegmentLengths, build_biomod, segment_length_for_side


def test_default_model_stem_distinguishes_back_variant():
    stem_default = default_model_stem("cleaned", "exhaustive")
    stem_back_flex = default_model_stem("cleaned", "exhaustive", model_variant="back_flexion_1d")
    stem_back = default_model_stem("cleaned", "exhaustive", model_variant="back_3dof")

    assert stem_default == "model_2d_cleaned_exhaustive"
    assert stem_back_flex == "model_2d_cleaned_exhaustive_back_flexion_1d"
    assert stem_back == "model_2d_cleaned_exhaustive_back_3dof"


def _lengths() -> SegmentLengths:
    return SegmentLengths(
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
        left_upper_arm_length=0.31,
        right_upper_arm_length=0.27,
        left_forearm_length=0.26,
        right_forearm_length=0.24,
        left_thigh_length=0.47,
        right_thigh_length=0.43,
        left_shank_length=0.41,
        right_shank_length=0.39,
    )


def test_segment_length_for_side_symmetrizes_by_default():
    lengths = _lengths()

    assert segment_length_for_side(lengths, side="left", base_name="upper_arm_length") == lengths.upper_arm_length
    assert segment_length_for_side(lengths, side="right", base_name="upper_arm_length") == lengths.upper_arm_length


def test_segment_length_for_side_preserves_side_specific_lengths_when_requested():
    lengths = _lengths()

    assert segment_length_for_side(lengths, side="left", base_name="upper_arm_length", symmetrize_limbs=False) == 0.31
    assert segment_length_for_side(lengths, side="right", base_name="upper_arm_length", symmetrize_limbs=False) == 0.27


def test_build_biomod_back_flexion_1d_creates_upper_back_segment(tmp_path: Path):
    output_path = tmp_path / "back_flexion_1d.bioMod"

    build_biomod(_lengths(), output_path, model_variant="back_flexion_1d")

    text = output_path.read_text()
    assert "segment\tUPPER_BACK" in text
    trunk_start = text.index("segment\tTRUNK")
    trunk_end = text.index("endsegment", trunk_start)
    trunk_block = text[trunk_start:trunk_end]
    upper_back_start = text.index("segment\tUPPER_BACK")
    upper_back_end = text.index("endsegment", upper_back_start)
    upper_back_block = text[upper_back_start:upper_back_end]
    assert "rotations\ty" in upper_back_block.lower()
    assert "parent\tTRUNK" in text
    assert "marker\tleft_shoulder" in text
    assert "parent\tUPPER_BACK" in text
    assert "mesh\t0.000000\t-0.120000\t0.000000" in trunk_block
    assert "mesh\t0.000000\t0.000000\t0.000000" in trunk_block
    assert "mesh\t0.000000\t0.120000\t0.000000" in trunk_block
    assert "mesh\t0.000000\t0.300000\t0.300000" not in upper_back_block
    assert "mesh\t0.000000\t0.180000\t0.300000" in upper_back_block
    assert "mesh\t0.000000\t-0.180000\t0.300000" in upper_back_block
    assert upper_back_block.count("mesh\t0.000000\t0.000000\t0.300000") >= 2


def test_build_biomod_back_3dof_creates_upper_back_segment(tmp_path: Path):
    output_path = tmp_path / "back_3dof.bioMod"

    build_biomod(_lengths(), output_path, model_variant="back_3dof")

    text = output_path.read_text()
    assert "segment\tUPPER_BACK" in text
    upper_back_start = text.index("segment\tUPPER_BACK")
    upper_back_end = text.index("endsegment", upper_back_start)
    upper_back_block = text[upper_back_start:upper_back_end]
    assert "parent\tTRUNK" in text
    assert "marker\tleft_shoulder" in text
    assert "parent\tUPPER_BACK" in text
    assert "mesh\t0.000000\t0.180000\t0.300000" in upper_back_block
    assert "mesh\t0.000000\t-0.180000\t0.300000" in upper_back_block
