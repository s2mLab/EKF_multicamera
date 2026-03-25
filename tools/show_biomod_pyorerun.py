#!/usr/bin/env python3
"""Visualise un modele `.bioMod` avec `pyorerun`.

Deux usages:
- sans `--states`, le script ouvre une pose neutre sur quelques frames ;
- avec `--states`, il anime les coordonnees generalisees chargees depuis
  `ekf_states.npz` (champ `q`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


def parse_args() -> argparse.Namespace:
    """Construit l'interface CLI du visualiseur."""
    parser = argparse.ArgumentParser(description="Visualise un fichier .bioMod dans pyorerun.")
    parser.add_argument(
        "--biomod",
        type=Path,
        default=Path("output/vitpose_full/vitpose_chain.bioMod"),
        help="Modele .bioMod a visualiser.",
    )
    parser.add_argument(
        "--states",
        type=Path,
        default=Path("output/vitpose_full/ekf_states.npz"),
        help="NPZ contenant `q`. Si absent, une pose neutre est affichee.",
    )
    parser.add_argument("--fps", type=float, default=120.0, help="Frequence d'echantillonnage des etats.")
    parser.add_argument("--stride", type=int, default=1, help="Sous-echantillonnage temporel.")
    parser.add_argument(
        "--mode",
        choices=("trajectory", "neutral"),
        default="trajectory",
        help="`trajectory` anime les q, `neutral` montre une pose neutre.",
    )
    parser.add_argument(
        "--rrd",
        type=Path,
        default=None,
        help="Si fourni, exporte un fichier `.rrd` au lieu d'ouvrir seulement la vue.",
    )
    return parser.parse_args()


def load_q(states_path: Path, stride: int) -> np.ndarray:
    """Charge la trajectoire q depuis un fichier NPZ."""
    data = np.load(states_path, allow_pickle=True)
    return np.asarray(data["q"], dtype=float)[:: max(1, stride)]


def main() -> None:
    """Lance la visualisation pyorerun."""
    import pyorerun

    args = parse_args()
    model = pyorerun.BiorbdModel(str(args.biomod))
    model.options.transparent_mesh = False
    model.options.show_gravity = False
    model.options.show_marker_labels = False
    model.options.show_center_of_mass_labels = False

    if args.mode == "trajectory" and args.states.exists():
        q = load_q(args.states, args.stride).T
        t = np.arange(q.shape[1]) / args.fps * max(1, args.stride)
    else:
        q = np.zeros((model.nb_q, 20))
        t = np.linspace(0.0, 1.0, q.shape[1])

    viz = pyorerun.PhaseRerun(t)
    viz.add_animated_model(model, q)

    if args.rrd is not None:
        args.rrd.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(viz, "save"):
            viz.save(str(args.rrd))
            print(f"Animation pyorerun exportee dans: {args.rrd}")
            return

    if hasattr(viz, "rerun_by_frame"):
        viz.rerun_by_frame("Biomechanical model")
    elif hasattr(viz, "save") and args.rrd is None:
        fallback_rrd = args.biomod.with_suffix(".rrd")
        viz.save(str(fallback_rrd))
        print(f"pyorerun n'a pas de vue live disponible ici, export dans: {fallback_rrd}")
    else:
        raise RuntimeError("pyorerun ne fournit ni vue live ni export .rrd dans cette installation.")


if __name__ == "__main__":
    main()
