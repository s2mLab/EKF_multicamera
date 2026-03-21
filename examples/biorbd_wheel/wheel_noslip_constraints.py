#!/usr/bin/env python3
"""
Roue-sol en biorbd: contact rigide avec roulement sans glissement (contraintes dures).

Ce script montre la structure complète avec:
- dynamique contrainte,
- récupération des forces de contact,
- intégration simple en temps.

Note:
- Les noms de méthodes biorbd peuvent varier selon version.
- Les wrappers ci-dessous cherchent plusieurs signatures usuelles.
"""

from __future__ import annotations

import argparse
from typing import Iterable, Any

import numpy as np

try:
    import biorbd
except ImportError as e:
    raise SystemExit("biorbd n'est pas installé dans cet environnement.") from e


def call_first(obj: Any, names: Iterable[str], *args):
    """
    Appelle la première méthode existante dans `names`.
    Permet de rester compatible avec plusieurs versions de biorbd.
    """
    for name in names:
        fn = getattr(obj, name, None)
        if callable(fn):
            return fn(*args)
    tried = ", ".join(names)
    raise AttributeError(f"Aucune méthode disponible parmi: {tried}")


def vec_to_np(v: Any) -> np.ndarray:
    """
    Conversion robuste des types biorbd vers ndarray colonne.
    """
    if hasattr(v, "to_array"):
        arr = np.array(v.to_array(), dtype=float)
    else:
        arr = np.array(v, dtype=float)
    return arr.reshape((-1, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Chemin du .bioMod")
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()

    model = biorbd.Model(args.model)
    nq = model.nbQ()
    nv = model.nbQdot()
    ntau = model.nbGeneralizedTorque()

    # Etat initial (a adapter):
    q = np.zeros((nq, 1))
    qdot = np.zeros((nv, 1))
    tau = np.zeros((ntau, 1))

    # Exemple: petit couple de roue pour avancer.
    # Index a adapter selon ton modele.
    if ntau > 0:
        tau[0, 0] = 2.0

    for k in range(args.steps):
        # 1) Dynamique contrainte:
        #    M qddot + h = tau + Jc^T lambda
        qddot = call_first(
            model,
            [
                "ForwardDynamicsConstraintsDirect",
                "ForwardDynamicsConstraints",
            ],
            q,
            qdot,
            tau,
        )
        qddot = vec_to_np(qddot)

        # 2) Forces de contact associees (lambda):
        #    vecteur des réactions sur les axes de contact definis dans le .bioMod.
        contact_forces = call_first(
            model,
            [
                "contactForcesFromForwardDynamicsConstraintsDirect",
                "contactForcesFromForwardDynamicsConstraints",
            ],
            q,
            qdot,
            tau,
        )
        contact_forces = vec_to_np(contact_forces)

        # 3) Intégration semi-implicite minimale (exemple).
        #    Pour un vrai projet: integrateur plus robuste.
        qdot = qdot + args.dt * qddot
        q = q + args.dt * qdot

        if k % 200 == 0:
            # Impression compacte: premières composantes de q et lambda
            q_flat = q[:, 0]
            f_flat = contact_forces[:, 0]
            print(f"step={k:5d} q0={q_flat[0]: .4f} qdot0={qdot[0,0]: .4f} contact={f_flat[:min(6, len(f_flat))]}")


if __name__ == "__main__":
    main()
