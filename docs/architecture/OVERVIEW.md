# Architecture actuelle

## Statut et sources

Cette vue decrit l'architecture observee en septembre 2026. Le code source et
les tests priment sur ce document. Les faits sont tires des modules et de la
CI; les risques sont des points a verifier avant une modification, pas des
defauts prouves.

Le depot fournit une reconstruction biomecanique multi-camera de sequences de
trampoline : inspection et correction des detections 2D, reconstruction 3D,
construction de modele, EKF, analyse et exports. Python 3.11 est la version
utilisee par `pyproject.toml` et la CI.

## Flux principaux

```text
Calib.toml + keypoints JSON (+ annotations, TRC, images)
    -> PoseData: 2D raw / cleaned / annotated
    -> correction optionnelle des inversions gauche/droite
    -> coherence epipolaire ou triangulation + reprojection
    -> points 3D et/ou modele .bioMod
    -> EKF 3D ou EKF 2D
    -> reconstruction bundle, caches .npz, manifest/summary
    -> GUI, analyses cinematiques, QA calibration, DD, exports GIF/XLSX
```

Les points d'entree sont `pipeline_gui.py` (GUI Tk),
`export_reconstruction_bundle.py` (un bundle),
`run_reconstruction_profiles.py` (profils nommes), `batch_run.py` (lots) et
`vitpose_ekf_pipeline.py` (CLI historique et algorithmes centraux).

## Responsabilites des modules

| Zone | Responsabilite observee |
| --- | --- |
| `vitpose_ekf_pipeline.py` | structures `PoseData`/`ReconstructionResult`, nettoyage 2D, geometrie epipolaire, triangulation, modele bioMod, EKF et sorties historiques |
| `reconstruction/` | profils, registre de noms, construction de bundles, caches, manifestes et compatibilite des sorties |
| `annotation/` | stockage d'annotations 2D eparses, navigation, rendu et assistance cinematique |
| `calibration_qc.py`, `camera_tools/` | QA epipolaire/reprojection et selection de cameras |
| `pipeline_gui.py` | integration Tk de la selection de donnees, caches, profils, reconstructions et analyses |
| `preview/` | chargement, navigation et rendu des bundles dans le GUI |
| `kinematics/`, `judging/`, `observability/`, `analysis/`, `animation/` | analyses et exports derives |

`pipeline_gui.py` et `vitpose_ekf_pipeline.py` sont de grands modules
integrateurs. Preferer une extension dans un sous-package existant lorsqu'elle
respecte une responsabilite deja isolee; eviter un refactoring transversal sans
objectif explicite.

## Contrats de donnees et invariants

- `PoseData.keypoints` est organise comme `(camera, frame, keypoint, xy)` et
  `PoseData.scores` comme `(camera, frame, keypoint)`; les keypoints suivent
  l'ordre `COCO17`.
- Les observations 2D et erreurs epipolaires/reprojection sont en pixels. Les
  trajectoires 3D, longueurs de segment et pseudo-observations de contact sont
  exprimees en metres; les rotations et ecarts angulaires en radians, sauf les
  parametres explicitement libelles en degres.
- Les donnees manquantes sont representees par `NaN`; les scores nuls et les
  masques de vues exclues font partie du contrat. Ne pas les remplacer
  silencieusement par des zeros numeriques.
- La calibration fixe projection, distorsion et repere monde. La racine
  geometrique utilise la convention Euler `YXZ`; la correction initiale de
  rotation et l'unwrapping doivent etre preserves dans les comparaisons.
- Le FPS est transporte dans les bundles; le defaut camera est 120 Hz. Un
  changement de stride modifie le FPS effectif et les derivees temporelles.

## Corrections 2D, coherence et reconstruction

Le nettoyage temporel et le rejet d'outliers sont implementes dans
`filter_pose_keypoints`. Les corrections L/R sont calculees puis mises en cache
par `reconstruction.reconstruction_bundle`; elles sont des entrees logiques des
stages suivants. La coherence epipolaire utilise des matrices fondamentales et
des erreurs de Sampson ou une distance symetrique selon la methode selectionnee.

La triangulation propose `once`, `greedy` et `exhaustive`; le cout augmente avec
le nombre de cameras, particulierement pour `exhaustive`. Les bundles portent
des erreurs de reprojection par vue, des coherences et des masques d'exclusion.
Les profils de reconstruction sont donc des parametres scientifiques, pas de la
simple configuration d'interface.

## EKF 2D

`MultiViewKinematicEKF` (etat `[q, qdot, qddot]`, `nx = 3 nq`) corrige dans
l'espace image avec `H = [H_q, 0, 0]` et `R` diagonal (variances en px^2,
`inf` pour une mesure exclue, qui n'entre pas dans les blocs). Le solveur de
correction est choisi par `update_method` (`--ekf2d-update-method`, champ de
profil `ekf2d_update_method`, trace dans `summary.filter_parameters` et
`update_solver_counts`) :

- `woodbury` (defaut) : forme information, `G = H_q^T R^-1 H_q`,
  `C = (I + G P_qq)^-1`, `x+ = x + P_q C g`, Joseph reduit ; aucun inverse de
  `P_qq` (DoF verrouilles et `P_qq` singulier geres). Equivalent a `legacy` a
  l'arrondi pres (tests `tests/test_ekf2d_woodbury_update.py`), repli
  automatique sur `legacy` si le systeme reduit echoue.
- `legacy` : espace innovation, sequentiel par camera, ou batch si des
  pseudo-observations sont actives.

## Caches et effets de bord

Les caches de pose corrigee, flip, coherence epipolaire, triangulation, modele
et EKF se trouvent sous `output/<dataset>/...`, avec des metadonnees destinees a
invalider une sortie incompatible. Toute nouvelle option qui affecte un resultat
doit etre incluse dans la cle ou les metadonnees du cache correspondant et etre
couverte par un test de non-reutilisation.

Le GUI a aussi des caches en memoire pour calibration, pose, apercus et analyses
de saut, centralises dans un etat partage. Les onglets de commande lancent des
sous-processus. Une interaction GUI peut lire/ecrire des profils, annotations,
caches et sorties; les tests unitaires n'equivalent pas a un smoke test Tk.

## Validation existante

La CI GitHub Actions utilise Python 3.11, installe `pip install -e .[test]`, puis
execute `isort --check-only`, `black --check`, `flake8` et `pytest -q`. Les tests
couvrent notamment les hotspots du pipeline, caches de reconstruction, profils,
QA calibration, annotations, preview et GUI. Les smoke tests GUI necessitent
`RUN_PIPELINE_GUI_SMOKE=1`, Tk et un affichage disponible.

## Risques et limites observes

- Les dependances Conda scientifiques (`biorbd`, OpenCV, `biobuddy`, etc.) ne
  sont pas toutes declarees dans `pyproject.toml`; la CI n'exerce donc pas tout
  le workflow de reconstruction scientifique.
- Les projections et Jacobiennes suivent le modele pinhole alors que les
  calibrations chargent aussi une distortion. Avant de modifier geometrie ou
  reprojection, confirmer que les detections 2D sont deja undistorted.
- Les metadonnees des caches geometriques (epipolaire, flip, pose corrigee par
  flip, triangulation) exigent les calibrations et stockent
  `calibration_signature`, calculee sur les parametres parses de chaque camera.
  Le stage modele stocke `reconstruction_signature` et `biomod_signature`, et
  n'est reutilise que si le `.bioMod` existe et n'a pas change; le cache Kalman
  `biorbd` stocke aussi ces deux signatures. Un ancien cache sans ces champs est
  rejete et recalcule.
- Cote GUI, la cle du cache de calibration, la cle du cache de pose et le cache
  d'apercu q0 incluent une signature du contenu de `Calib.toml`. Le fichier de
  keypoints reste identifie par son chemin seulement dans le cache de pose du
  GUI : une modification en place pendant une session doit etre rechargee
  explicitement.
- Deux fichiers `environment*.yml` partagent le nom d'environnement
  `vitpose-ekf` mais different par leurs dependances. Ne pas les fusionner ou
  remplacer sans une demande dediee.
- Le README ne doit contenir que des chemins relatifs a la racine Git; ne pas
  reintroduire de chemins absolus propres a une machine.
- Les donnees versionnees de `inputs/` et `reconstruction_profiles*.json` peuvent
  contenir du travail experimental local. Les traiter comme des artefacts a
  proteger jusqu'a instruction contraire.
