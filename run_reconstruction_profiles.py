#!/usr/bin/env python3
"""Execute une serie de profils de reconstruction nommes."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCAL_MPLCONFIG = ROOT / ".cache" / "matplotlib"
LOCAL_MPLCONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_MPLCONFIG))

from reconstruction.reconstruction_registry import infer_dataset_name, latest_version_for_family
from reconstruction.reconstruction_profiles import (
    build_pipeline_command,
    example_profiles,
    load_profiles_json,
    profile_to_dict,
    save_profiles_json,
    validate_profile,
    variant_output_dir,
)


DEFAULT_CONFIG = Path("reconstruction_profiles.json")
DEFAULT_OUTPUT_ROOT = Path("outputs")
DEFAULT_CALIB = Path("inputs/Calib.toml")
DEFAULT_KEYPOINTS = Path("inputs/1_partie_0429_keypoints.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lance une liste de profils de reconstruction.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Fichier JSON des profils.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Dossier racine des sorties.")
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    parser.add_argument("--keypoints", type=Path, default=DEFAULT_KEYPOINTS)
    parser.add_argument("--pose2sim-trc", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--triangulation-workers", type=int, default=6)
    parser.add_argument("--camera-names", type=str, default="", help="Liste de cameras a utiliser, separees par des virgules.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Nom du dataset; sinon derive du fichier 2D.")
    parser.add_argument("--profile", action="append", default=None, help="Nom de profil a lancer. Peut etre repete.")
    parser.add_argument("--write-example-config", action="store_true", help="Ecrit un fichier de profils d'exemple puis s'arrete.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue meme si un profil echoue.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_example_config:
        save_profiles_json(args.config, example_profiles())
        print(f"Configuration d'exemple ecrite dans {args.config}")
        return

    dataset_name = infer_dataset_name(keypoints_path=args.keypoints, pose2sim_trc=args.pose2sim_trc, dataset_name=args.dataset_name)
    profiles = load_profiles_json(args.config)
    if args.profile:
        selected = set(args.profile)
        profiles = [profile for profile in profiles if profile.name in selected]
    if not profiles:
        raise SystemExit("Aucun profil a executer.")

    manifest = []
    for profile_idx, profile in enumerate(profiles, start=1):
        profile = validate_profile(profile)
        out_dir = variant_output_dir(
            args.output_root,
            profile,
            dataset_name=dataset_name,
            keypoints_path=args.keypoints,
            pose2sim_trc=args.pose2sim_trc,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "profile.json").write_text(json.dumps(profile_to_dict(profile), indent=2), encoding="utf-8")
        cmd = build_pipeline_command(
            profile,
            output_root=args.output_root,
            calib=args.calib,
            keypoints=args.keypoints,
            pose2sim_trc=args.pose2sim_trc,
            dataset_name=dataset_name,
            python_executable=sys.executable,
            camera_names_override=[name.strip() for name in args.camera_names.split(",") if name.strip()] or None,
        )
        cmd.extend(["--fps", str(args.fps), "--triangulation-workers", str(args.triangulation_workers)])
        print(f"\n[PROFILE {profile_idx}/{len(profiles)}] {profile.name}", flush=True)
        print(shlex.join(cmd), flush=True)
        completed = subprocess.run(cmd, cwd=Path(__file__).resolve().parent)
        bundle_summary_path = out_dir / "bundle_summary.json"
        bundle_summary = json.loads(bundle_summary_path.read_text()) if bundle_summary_path.exists() else {}
        manifest.append(
            {
                "name": profile.name,
                "family": profile.family,
                "returncode": completed.returncode,
                "output_dir": str(out_dir),
                "latest_family_version": latest_version_for_family(profile.family),
                "bundle_summary_path": str(bundle_summary_path) if bundle_summary_path.exists() else None,
                "bundle_summary": bundle_summary,
            }
        )
        if completed.returncode != 0 and not args.continue_on_error:
            raise SystemExit(completed.returncode)

    manifest_path = args.output_root / dataset_name / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_name": dataset_name,
                "calib": str(args.calib),
                "keypoints": str(args.keypoints),
                "pose2sim_trc": None if args.pose2sim_trc is None else str(args.pose2sim_trc),
                "fps": float(args.fps),
                "triangulation_workers": int(args.triangulation_workers),
                "runs": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nManifest ecrit dans {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
