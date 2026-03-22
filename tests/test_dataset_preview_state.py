from preview.dataset_preview_state import build_dataset_preview_state


def test_build_dataset_preview_state_uses_bundle_names_and_bundle_frame_count():
    catalog = [
        {"name": "pose2sim", "cached": True, "family": "pose2sim"},
        {"name": "ekf_3d", "cached": True, "family": "ekf_3d"},
        {"name": "stale", "cached": True, "family": "triangulation"},
    ]
    bundle = {
        "recon_3d": {"pose2sim": object()},
        "recon_q": {"ekf_3d": object()},
        "frames": [10, 11, 12],
    }
    state = build_dataset_preview_state(
        catalog=catalog,
        bundle=bundle,
        preferred_names=["ekf_3d", "pose2sim"],
        fallback_count=2,
    )
    assert [row["name"] for row in state.rows] == ["pose2sim", "ekf_3d"]
    assert state.defaults == ["ekf_3d", "pose2sim"]
    assert state.available_names == ["ekf_3d", "pose2sim"]
    assert state.max_frame == 2


def test_build_dataset_preview_state_falls_back_to_catalog_and_keeps_extra_rows():
    catalog = [
        {"name": "pose2sim", "cached": True, "family": "pose2sim"},
        {"name": "ekf_2d_acc", "cached": False, "family": "ekf_2d"},
        {"name": "triangulation_exhaustive", "cached": True, "family": "triangulation"},
    ]
    state = build_dataset_preview_state(
        catalog=catalog,
        bundle=None,
        preferred_names=["raw", "pose2sim"],
        fallback_count=2,
        include_q=False,
        extra_rows=[{"name": "raw", "cached": True, "family": "2d"}],
    )
    assert [row["name"] for row in state.rows] == ["raw", "pose2sim", "triangulation_exhaustive"]
    assert state.defaults == ["raw", "pose2sim"]
    assert state.available_names == []
    assert state.max_frame == 0
