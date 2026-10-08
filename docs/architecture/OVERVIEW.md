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

Les variantes `epipolar_fast`, `epipolar_fast_framewise` (coherence) et
`epipolar_fast`, `epipolar_fast_viterbi` (flip) ne different de leurs
equivalents Sampson (`epipolar`, `epipolar_framewise`, `epipolar_viterbi`) que
par la distance epipolaire symetrique, mesuree +3,7 % pire que Sampson sur
donnees reelles. Les profils existants ne sont pas modifies ; utiliser le mode
sans `_fast` pour Sampson (aucune option supplementaire necessaire).
`export_reconstruction_bundle.py` affiche une note (`epipolar_fast_notice`)
quand un mode symetrique est choisi.

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

Le predicteur `dyn`/`dyn_history3` remplace l'acceleration racine par la
dynamique flottante quand un critere de vol est vrai (`flight_detection`,
`--flight-detection`, champ de profil) :

- `triangulation` (defaut, historique) : tous les points 3D triangules finis
  des `flight_min_consecutive_frames` frames precedentes au-dessus de
  `flight_height_threshold_m`. Avec `ekf2d_3d_source=first_frame_only` les
  points sont `NaN` apres la frame 0 et `dyn` ne s'active jamais (0/900 frames
  mesurees sur `1_partie_0429`).
- `ekf_state` (opt-in) : plus bas marqueur du modele a l'etat EKF corrige de la
  frame precedente au-dessus du seuil pendant `flight_min_consecutive_frames`
  frames, sortie sous `seuil - flight_hysteresis_m` (defaut 0.05 m), garde
  optionnelle `|CoMddot_z - g_z| <= flight_com_accel_tolerance`. Le masque
  d'activation reel est exporte (`dyn_active_per_frame`, `dyn_active_frames`).

Bruit de processus (`process_noise_model`, `--process-noise-model`, champ de
profil, trace dans `summary.filter_parameters`) :

- `legacy` (defaut) : `diag(1e-4, 5e-3, 5e-2) * process_noise_scale` pour
  `(q, qdot, qddot)`, independant de `dt` (donc du stride/FPS effectif).
- `white_jerk` (opt-in) : jerk blanc continu discretise exactement,
  `Q(dt) = q_c [[dt^5/20, dt^4/8, dt^3/6], [dt^4/8, dt^3/3, dt^2/2],
  [dt^3/6, dt^2/2, dt]]` par DoF, avec `q_c` par groupe (translation racine
  m^2/s^5, rotation racine et articulations rad^2/s^5 ;
  `--process-noise-jerk-psd`, defaut `200 1000 10000`, valeur calee par une
  analyse independante : synthetique 3 graines -13,4 % d'erreur marqueurs,
  reel `1_partie_0429` en leave-one-camera-out -3 %). Aucun calage automatique
  dans le code. Sur les 240 premieres frames reelles, la reprojection mediane
  passe de 12,91 px (`legacy`) a 12,26 px (l'ancien defaut `6 6 6` donnait
  21,86 px).

A priori articulaire (`joint_prior`, `--ekf2d-joint-prior`, opt-in ; convention
du modele : genou flechi `SHANK:RotY > 0`, coude flechi `FOREARM:RotY < 0`).
Les marqueurs distaux etant sur l'axe du segment, `(RotZ + pi, -RotY)` pour
`FOREARM` et `(THIGH:RotZ + pi, -SHANK:RotY)` pour le genou donnent exactement les
memes marqueurs : ces branches miroir sont inobservables.

1. Apres chaque correction : une flexion du mauvais cote de plus de 5 deg est
   reflechie dans l'autre branche (symetrie exacte, `x <- T x + c`,
   `P <- T P T^T`) ; une violation residuelle de la borne (coude <= -1 deg,
   genou >= +1 deg) declenche une pseudo-observation d'inegalite
   `RotY = borne` (sigma 0,5 deg), active seulement si violee.
2. Pseudo-observations lineaires `FOREARM:RotZ` et `THIGH:RotZ ~ N(0, sigma^2)`
   (`--ekf2d-joint-prior-axial-std-deg`, defaut 30 deg), ajoutees aux blocs de
   pseudo-observations (chemin Woodbury par defaut ; `legacy` bascule en batch).
3. Export (`run_ekf`) : `canonicalize_joint_mirror_branches` ramene q (et le
   signe de qdot/qddot de la flexion) dans la branche canonique et replie
   `RotZ` dans `[-pi, pi)` ; formats et formes inchanges.

Sur `1_partie_0429` (900 frames, triangulation exhaustive, `acc`) : frames en
miroir (au moins un membre) 98,3 % -> 0 % (par membre ~48 % -> 0 %), amplitude
max de `FOREARM:RotZ` 2115 -> 251 deg (45 deg : 356 deg), reprojection mediane
11,86 -> 12,07 px.

Melange robuste (`robust_mixture`, `--ekf2d-robust-mixture`, opt-in) : pour
chaque keypoint, `S_k = H_k P_qq H_k^T + r_k I` (2x2, etat predit),
`w_k = pi_in N(y_k; 0, S_k) / (pi_in N + pi_out / A_img)` avec
`pi_out = 0.03` (`--ekf2d-robust-outlier-prob`) et `A_img` = largeur x hauteur de
la calibration ; variance effective = diagonale de `S_k / w_k - H_k P_qq H_k^T`
(diagonale pour les solveurs, >= `r_k`, `w_k >= 1e-6`). Independant de l'ordre
des cameras et identique pour `woodbury` et `legacy`. Statistiques dans
`robust_mixture_stats`. Sur 900 frames reelles : 6,3 % des keypoints ont
`w < 0,5`, reprojection mediane 11,86 -> 11,66 px (moyenne 20,43 -> 22,36 px :
les outliers ne sont plus suivis).

Garde-fou anti-verrouillage du melange (actif des que `robust_mixture`) : le
melange juge chaque mesure contre la prediction ; si la prediction est fausse
(graine IK du bootstrap a ~240 px, membre parti dans une mauvaise pose), il
rejette precisement les detections qui la corrigeraient et le filtre
s'auto-confirme. Sur `1_partie_0429_001` (detecteur `best`, 120 Hz), le bootstrap
avec melange restait sur la graine (2/3 des keypoints a `w < 0,5`, etat faux de
~1,7 m) ; avec `white_jerk`, `P_qq` se contracte plus vite et le filtre ne se
recalait jamais (96 % de `w < 0,5`, 2238 px / 56 m aux frames annotees) ; un `R`
plus petit (scores eleves) aggrave le risque (`R x 4` : recalage ; `ECCV`/`base`
avec `R / 4` : divergence). Deux regles, independantes de l'ordre des cameras et
du solveur : (1) frame : si la fraction de `w < 0,5` depasse
`robust_mixture_lock_fraction` (0,5, `--ekf2d-robust-lock-fraction`), la frame
est corrigee avec les variances nominales jusqu'a ce que la fraction redescende
a `robust_mixture_resume_fraction` (0,25) ; chaque EKF, bootstrap compris,
demarre suspendu ; (2) keypoint : un keypoint rejete dans plus de
`lock_fraction` de ses vues (au moins 2) reprend sa variance nominale (une erreur
de detecteur est propre a une vue ; un rejet majoritaire signale un point 3D
predit faux). `1 1` restaure le melange non garde. Compteurs :
`suspended_frames`, `lock_events`, `keypoint_guard_restored`,
`applied_downweighted_below_0_5` dans `robust_mixture_stats` (et dans les
diagnostics du bootstrap). `white_jerk` + melange donne alors 8,61 px / 38,2 mm
(`white_jerk` seul : 8,59 px / 38,4 mm) ; la combinaison `white_jerk` +
`undistort` + `joint_prior` + melange, 8,41 px / 42,5 mm, et aucune
divergence avec les detecteurs `best`, `ECCV` et `base` (sans melange :
resultats identiques au bit pres).

Limite connue non corrigee : en `dyn`, `history3` et `dyn_history3`, la moyenne
predite est recalculee (dynamique ou extrapolation d'historique) mais la
covariance reste propagee avec le `F` a acceleration constante ; `P` n'est donc
pas coherente avec la prediction de la moyenne.

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
  calibrations chargent aussi une distortion (OpenCV `k1, k2, p1, p2[, k3..k6]`).
  Par defaut les keypoints ne sont pas dedistordus (comportement historique).
  L'option opt-in `undistort_keypoints` (`--undistort-keypoints`, champ de
  profil, `load_pose_data(..., undistort_keypoints=True)`) dedistord une fois au
  chargement (Newton vectorise, equivalent `cv2.undistortPoints(..., P=K)`)
  les keypoints bruts et annotes avant nettoyage, coherence, triangulation et
  EKF. Les calibrations sont alors marquees `keypoints_undistorted=True` via
  `calibrations_with_undistorted_keypoints`, ce qui modifie
  `calibration_signature` (qui contient deja `dist`) et invalide les caches
  geometriques; la signature par defaut est inchangee. Les overlays 2D sur les
  images brutes (GUI) restent en coordonnees distordues et ne sont pas
  compatibles avec des bundles produits avec cette option; le GUI n'expose pas
  encore l'option.
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
