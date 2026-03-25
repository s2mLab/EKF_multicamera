from pathlib import Path

from reconstruction.reconstruction_registry import default_model_stem, model_output_dir, normalize_output_root


def test_default_model_stem_includes_pose_correction_mode() -> None:
    base = default_model_stem("cleaned", "exhaustive")
    flip_epi = default_model_stem("cleaned", "exhaustive", pose_correction_mode="flip_epipolar")
    flip_tri = default_model_stem("cleaned", "exhaustive", pose_correction_mode="flip_triangulation")

    assert base == "model_2d_cleaned_exhaustive"
    assert flip_epi == "model_2d_cleaned_exhaustive_flip_epipolar"
    assert flip_tri == "model_2d_cleaned_exhaustive_flip_triangulation"


def test_model_output_dir_changes_with_pose_correction_mode() -> None:
    output_root = Path("outputs")
    dataset_name = "demo"

    base = model_output_dir(output_root, dataset_name, pose_data_mode="cleaned", triangulation_method="exhaustive")
    flip_epi = model_output_dir(
        output_root,
        dataset_name,
        pose_data_mode="cleaned",
        triangulation_method="exhaustive",
        pose_correction_mode="flip_epipolar",
    )
    flip_tri = model_output_dir(
        output_root,
        dataset_name,
        pose_data_mode="cleaned",
        triangulation_method="exhaustive",
        pose_correction_mode="flip_triangulation",
    )

    assert base != flip_epi
    assert base != flip_tri
    assert flip_epi != flip_tri


def test_model_output_dir_normalizes_legacy_outputs_root() -> None:
    output_root = Path("outputs")
    dataset_name = "demo"

    model_dir = model_output_dir(output_root, dataset_name, pose_data_mode="cleaned", triangulation_method="exhaustive")

    assert model_dir == Path("output/demo/models/model_2d_cleaned_exhaustive")
    assert normalize_output_root(Path("outputs")) == Path("output")
