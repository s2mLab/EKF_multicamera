from reconstruction_presenter import (
    bundle_available_reconstruction_names,
    catalog_rows_for_names,
    default_selection,
)


def test_bundle_available_reconstruction_names_merges_selected_sources():
    bundle = {
        "recon_3d": {"pose2sim": object(), "triangulation_exhaustive": object()},
        "recon_q": {"ekf_2d_acc": object()},
        "recon_q_root": {"pose2sim": object(), "legacy_root": object()},
    }
    names = bundle_available_reconstruction_names(bundle, include_3d=True, include_q=True, include_q_root=True)
    assert names == ["ekf_2d_acc", "legacy_root", "pose2sim", "triangulation_exhaustive"]


def test_catalog_rows_for_names_keeps_extra_rows_and_filters_cached_catalog():
    catalog = [
        {"name": "pose2sim", "cached": True},
        {"name": "ekf_2d_acc", "cached": False},
        {"name": "ekf_3d", "cached": True},
    ]
    rows = catalog_rows_for_names(catalog, {"pose2sim", "ekf_2d_acc"}, extra_rows=[{"name": "raw", "cached": True}])
    assert rows == [{"name": "raw", "cached": True}, {"name": "pose2sim", "cached": True}]


def test_default_selection_prefers_known_order_and_falls_back_to_first_items():
    assert default_selection(["pose2sim", "ekf_3d"], ["ekf_3d", "ekf_2d_acc"], fallback_count=3) == ["ekf_3d"]
    assert default_selection(["b", "a", "c"], ["x"], fallback_count=2) == ["b", "a"]
