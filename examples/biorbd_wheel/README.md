# Exemples biorbd: roue-sol

Ce dossier contient deux approches complémentaires pour le contact roue-sol en `biorbd`.

## 1) `wheel_noslip_constraints.py`

Objectif:
- Contact rigide avec contraintes tangentielles dures (roulement sans glissement).

Principe:
- Le solveur de dynamique contrainte résout:
  - `M(q) qddot + h(q, qdot) = tau + Jc(q)^T lambda`
  - `Jc(q) qddot + Jdotc(q, qdot) qdot = 0`
- Les forces de contact proviennent de `lambda`.

Ce qu'il faut côté `.bioMod`:
- Définir le contact roue-sol avec:
  - normale (verticale),
  - tangentielle(s) (longitudinale et/ou latérale) si tu veux du sans glissement.

## 2) `wheel_slip_coulomb_approx.py`

Objectif:
- Autoriser le glissement tangent et ajouter une friction type Coulomb (approchée).

Principe:
- Le contact normal reste géré par la dynamique contrainte (ou via ta logique de contact normale).
- La friction tangentielle est ajoutée comme force externe généralisée:
  - direction opposée à la vitesse tangentielle du point de contact,
  - norme bornée par `mu * N`.

Important:
- Cette approche est une approximation continue (régularisée), pas une LCP/MCP stricte.
- Pour une friction "exacte" stick/slip avec complémentarité, il faut un solveur dédié.

## Adaptation API

`biorbd` a des variations de noms de méthodes selon version.
Les scripts utilisent des wrappers (recherche dynamique de méthode) et lèvent une erreur explicite si une méthode est absente.

## Exécution (exemple)

```bash
python examples/biorbd_wheel/wheel_noslip_constraints.py --model /chemin/roue.bioMod
python examples/biorbd_wheel/wheel_slip_coulomb_approx.py --model /chemin/roue.bioMod --mu 0.8
```
