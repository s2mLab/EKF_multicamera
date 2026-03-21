#!/usr/bin/env python3
"""
Roue-sol en biorbd: glissement autorisé + friction Coulomb approchée.

Idée:
1) Calculer la dynamique contrainte pour obtenir la réaction normale.
2) Evaluer la vitesse tangentielle du point de contact.
3) Ajouter une force de friction tangentielle opposee au glissement:
   ||Ft|| <= mu * N (régularisé pour éviter les discontinuités numériques).

Ce script est volontairement explicite et très commenté.
"""

from __future__ import annotations

import argparse
from typing import Any, Iterable, Optional

import numpy as np

try:
    import biorbd
except ImportError as e:
    raise SystemExit("biorbd n'est pas installé dans cet environnement.") from e


def call_first(obj: Any, names: Iterable[str], *args):
    for name in names:
        fn = getattr(obj, name, None)
        if callable(fn):
            return fn(*args)
    tried = ", ".join(names)
    raise AttributeError(f"Aucune méthode disponible parmi: {tried}")


def vec_to_np(v: Any) -> np.ndarray:
    if hasattr(v, "to_array"):
        return np.array(v.to_array(), dtype=float).reshape((-1, 1))
    return np.array(v, dtype=float).reshape((-1, 1))


def marker_pos(model: Any, q: np.ndarray, marker_idx: int) -> np.ndarray:
    """
    Position monde d'un marker (3,).
    """
    m = call_first(model, ["marker", "markers"], q, marker_idx) if hasattr(model, "marker") else model.markers(q)[marker_idx]
    if hasattr(m, "to_array"):
        return np.array(m.to_array(), dtype=float).reshape((3,))
    return np.array(m, dtype=float).reshape((3,))


def marker_jacobian(model: Any, q: np.ndarray, marker_idx: int) -> np.ndarray:
    """
    Jacobien position du marker (3 x nq).
    """
    jac = call_first(
        model,
        [
            "technicalMarkersJacobian",
            "markersJacobian",
        ],
        q,
    )
    # Selon versions, jac est une liste par marker
    J = jac[marker_idx]
    if hasattr(J, "to_array"):
        J = np.array(J.to_array(), dtype=float)
    else:
        J = np.array(J, dtype=float)
    return J


def smooth_sign(v: np.ndarray, eps: float) -> np.ndarray:
    """
    Approximation lisse de v / ||v||.
    """
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / np.sqrt(n * n + eps * eps)


def normal_force_from_contact_vector(contact_forces: np.ndarray, normal_index: int) -> float:
    """
    Extrait la composante normale (N) dans le vecteur contactForces.

    normal_index depend du mapping de tes contacts dans le .bioMod.
    Exemple:
    - si le premier axe du premier contact est la normale: normal_index = 0
    """
    return max(0.0, float(contact_forces[normal_index, 0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Chemin du .bioMod")
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--mu", type=float, default=0.8, help="Coefficient Coulomb")
    parser.add_argument("--eps-slip", type=float, default=1e-4, help="Regularisation direction friction")
    parser.add_argument("--contact-marker", type=int, default=0, help="Marker au point de contact de la roue")
    parser.add_argument("--normal-index", type=int, default=0, help="Indice lambda normal dans contactForces")
    args = parser.parse_args()

    model = biorbd.Model(args.model)
    nq = model.nbQ()
    nv = model.nbQdot()
    ntau = model.nbGeneralizedTorque()

    q = np.zeros((nq, 1))
    qdot = np.zeros((nv, 1))
    tau_cmd = np.zeros((ntau, 1))
    if ntau > 0:
        tau_cmd[0, 0] = 2.0

    # Base sol (monde): normale z, tangentes x/y.
    n_world = np.array([0.0, 0.0, 1.0])
    t1_world = np.array([1.0, 0.0, 0.0])
    t2_world = np.array([0.0, 1.0, 0.0])

    for k in range(args.steps):
        # A) Première passe: dynamique contrainte avec commande seule.
        qddot_0 = call_first(
            model,
            ["ForwardDynamicsConstraintsDirect", "ForwardDynamicsConstraints"],
            q,
            qdot,
            tau_cmd,
        )
        qddot_0 = vec_to_np(qddot_0)

        contact_forces = call_first(
            model,
            ["contactForcesFromForwardDynamicsConstraintsDirect", "contactForcesFromForwardDynamicsConstraints"],
            q,
            qdot,
            tau_cmd,
        )
        contact_forces = vec_to_np(contact_forces)

        # B) Extraire N (réaction normale).
        N = normal_force_from_contact_vector(contact_forces, args.normal_index)

        # C) Vitesse du point de contact (marker) via jacobien.
        #    vC = J_marker(q) * qdot
        Jm = marker_jacobian(model, q, args.contact_marker)  # 3 x nq
        vC = (Jm @ qdot).reshape((3,))

        # D) Vitesse tangentielle au sol.
        v_t = np.array([np.dot(vC, t1_world), np.dot(vC, t2_world)])

        # E) Direction friction opposée au glissement (régularisée).
        dir_t = -smooth_sign(v_t, args.eps_slip)  # (2,)

        # F) Norme Coulomb bornée par mu*N.
        Ft_mag = args.mu * N
        Ft_tangent = Ft_mag * dir_t  # composantes dans base tangentielle (t1, t2)

        # G) Force monde appliquée au marker (pas de composante normale ici,
        #    on ne double pas la normale déjà gérée par la contrainte).
        F_world = Ft_tangent[0] * t1_world + Ft_tangent[1] * t2_world

        # H) Convertir cette force en effort généralisé additionnel.
        #    Q_fric = Jm^T * F_world
        tau_fric = (Jm.T @ F_world.reshape((3, 1)))

        # I) Deuxième passe: dynamique contrainte avec commande + friction.
        tau_total = tau_cmd + tau_fric
        qddot = call_first(
            model,
            ["ForwardDynamicsConstraintsDirect", "ForwardDynamicsConstraints"],
            q,
            qdot,
            tau_total,
        )
        qddot = vec_to_np(qddot)

        # J) Intégration simple.
        qdot = qdot + args.dt * qddot
        q = q + args.dt * qdot

        if k % 200 == 0:
            print(
                f"step={k:5d} N={N: .3f} "
                f"vt=({v_t[0]: .3e},{v_t[1]: .3e}) "
                f"Ft=({Ft_tangent[0]: .3f},{Ft_tangent[1]: .3f})"
            )


if __name__ == "__main__":
    main()
