from pathlib import Path

from dataset_preview_loader import load_dataset_preview_resources


def test_load_dataset_preview_resources_combines_sources_catalog_bundle_and_state():
    calls: dict[str, object] = {}

    def fake_sources(output_dir: Path, *, calib=None, keypoints=None, pose2sim_trc=None):
        calls["sources"] = {
            "output_dir": output_dir,
            "calib": calib,
            "keypoints": keypoints,
            "pose2sim_trc": pose2sim_trc,
        }
        return {
            "calib": "/tmp/calib.toml",
            "keypoints": "/tmp/keypoints.json",
            "pose2sim_trc": "/tmp/trial.trc",
        }

    def fake_catalog(output_dir: Path, pose2sim_trc: Path | None):
        calls["catalog"] = {"output_dir": output_dir, "pose2sim_trc": pose2sim_trc}
        return [
            {"name": "pose2sim", "cached": True},
            {"name": "ekf_3d", "cached": True},
        ]

    def fake_bundle_loader(output_dir: Path, biomod_path: Path | None, pose2sim_trc: Path | None, align_root: bool):
        calls["bundle"] = {
            "output_dir": output_dir,
            "biomod_path": biomod_path,
            "pose2sim_trc": pose2sim_trc,
            "align_root": align_root,
        }
        return {
            "frames": [0, 1],
            "recon_3d": {"pose2sim": object()},
            "recon_q": {"ekf_3d": object()},
        }

    result = load_dataset_preview_resources(
        output_dir=Path("/tmp/output"),
        preferred_names=["ekf_3d", "pose2sim"],
        fallback_count=2,
        dataset_source_paths_fn=fake_sources,
        discover_catalog_fn=fake_catalog,
        bundle_loader_fn=fake_bundle_loader,
        pose2sim_trc=Path("/tmp/override.trc"),
        biomod_path=Path("/tmp/model.bioMod"),
        align_root=False,
    )

    assert calls["sources"] == {
        "output_dir": Path("/tmp/output"),
        "calib": None,
        "keypoints": None,
        "pose2sim_trc": Path("/tmp/override.trc"),
    }
    assert calls["catalog"] == {
        "output_dir": Path("/tmp/output"),
        "pose2sim_trc": Path("/tmp/override.trc"),
    }
    assert calls["bundle"] == {
        "output_dir": Path("/tmp/output"),
        "biomod_path": Path("/tmp/model.bioMod"),
        "pose2sim_trc": Path("/tmp/trial.trc"),
        "align_root": False,
    }
    assert result.sources["pose2sim_trc"] == "/tmp/trial.trc"
    assert [row["name"] for row in result.preview_state.rows] == ["pose2sim", "ekf_3d"]
    assert result.preview_state.defaults == ["ekf_3d", "pose2sim"]
    assert result.preview_state.max_frame == 1
