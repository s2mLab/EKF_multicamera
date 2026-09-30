# Transfert et demarrage d'un agent sur une autre machine

Ce guide permet a un agent de reprendre ce depot sans dependre des chemins,
caches ou sorties de la machine d'origine. Il complete les instructions locales
dans `AGENTS.md`, la vue d'architecture et la matrice de validation.

## Objectif et limites du depot

Ce projet reconstruit des mouvements de trampoline a partir de poses 2D
multi-camera. Les entrees experimentales, les profils de reconstruction et les
sorties peuvent etre locaux ou en cours de travail. Un agent ne doit donc pas
les modifier, les reformatter ou les committer sans demande explicite.

Les caches et resultats sont regenerables sous `output/` et `.cache/`; ils ne
doivent pas etre transferes pour developper ou lancer les tests. Les donnees
necessaires a un essai scientifique doivent en revanche etre transferees par
un canal choisi par le proprietaire du projet, en conservant les chemins
relatifs attendus sous `inputs/`.

## Installation reproductible

Depuis le repertoire dans lequel le depot a ete clone :

```bash
conda env create -f environment.vitpose-ekf.yml
conda activate vitpose-ekf
pip install -e .[test]
```

La reference est `environment.vitpose-ekf.yml` avec Python 3.11. Ne pas utiliser
`environment.yml` comme substitut : il a le meme nom d'environnement mais un
ensemble de dependances different.

Le fichier de reference epingle `biobuddy` par installation pip depuis GitHub,
au commit `85ec552` (`pyomeca/biobuddy`, branche `main`), et `isort` a la
version 9.0.2 (idem dans l'extra `test` de `pyproject.toml`). Ce commit de
`biobuddy` expose `DeLevaTable.from_measurements(..., pelvis_height=...)`, que
`build_biomod` appelle ; une version plus ancienne exposant `hip_height`
provoque un `TypeError` dans `tests/test_model_variants.py`. `plotly`, requis
par `biobuddy`, est liste dans l'environnement. Le code peut aussi utiliser un
checkout local : `vitpose_ekf_pipeline.py` ajoute au `sys.path` le depot
designe par `BIOBUDDY_ROOT`, ou a defaut le depot frere `../biobuddy`.
Avec `isort` 9.0.2, `isort . --check-only --profile black` signale encore
`batch_run.py`, `pipeline_gui.py`, `tests/test_calibration_qc.py` et
`tests/test_preview_frame_2d_render.py` (ordre des imports, non corrige).

L'installation minimale `pip install -e .[test]` suffit aux tests CI. Les
reconstructions biomecaniques et certaines vues du GUI exigent aussi les
dependances Conda, notamment `biorbd`; le GUI requiert une installation Python
avec Tk. L'export Excel requiert `openpyxl`, qui n'est pas une dependance de
base du paquet.

## Verification initiale

Apres l'installation, confirmer l'environnement sans lancer de reconstruction
sur des donnees de production :

```bash
python --version
git status --short --branch
pytest -q tests/test_reconstruction_bundle_pose_cache.py
pytest -q tests/test_pipeline_gui_file_dialog.py
```

Le premier resultat doit utiliser Python 3.11. Un arbre de travail non vide au
depart est un fait a preserver et non une invitation a nettoyer ou a committer.
Pour valider toute la couche commune, lancer ensuite `pytest -q`. Sur une
machine munie de Tk et d'un affichage, le smoke test GUI est :

```bash
RUN_PIPELINE_GUI_SMOKE=1 pytest -q tests/test_pipeline_gui_launch.py
```

## Premiere lecture avant une modification

Lire dans cet ordre :

1. `AGENTS.md` pour les regles locales et scientifiques.
2. `docs/architecture/OVERVIEW.md` pour le flux de donnees, les invariants et
   les caches.
3. `docs/architecture/LLM_CONTEXT.md` pour choisir le test cible correspondant
   a la zone modifiee.
4. Le module concerne et son test avant toute modification.

Les contrats critiques sont l'ordre COCO17, les formes de `PoseData`, le
traitement des `NaN`, les unites (pixels, metres, radians), le FPS et le repere
de calibration. Une modification de filtrage, de flip gauche/droite, de
coherence, de triangulation ou d'EKF doit aussi verifier l'invalidation des
caches et fournir un test numerique cible.

## Reprendre un travail en cours

Avant d'editer, inspecter `git status --short --branch` et `git diff`. Isoler
les fichiers du travail courant de ceux qui etaient deja modifies. En
particulier, ne pas toucher sans instruction explicite :

- `inputs/` : donnees experimentales suivies par Git ;
- `reconstruction_profiles*.json` : reglages scientifiques potentiellement en
  cours ;
- `output/` et `.cache/` : artefacts locaux regenerables.

Les entrees principales sont `pipeline_gui.py` pour le GUI,
`vitpose_ekf_pipeline.py` pour les algorithmes centraux, et
`reconstruction/` pour les bundles, profils et caches. Les commandes de base
sont :

```bash
python pipeline_gui.py
python export_reconstruction_bundle.py --help
python run_reconstruction_profiles.py --help
```

Ne lancer une reconstruction complete sur une sequence reelle qu'apres avoir
confirme les fichiers d'entree, les options du profil et le dossier de sortie.

Options numeriques de l'EKF 2D (detail dans `OVERVIEW.md`, section EKF 2D) :
`--ekf2d-update-method {woodbury,legacy}` (defaut `woodbury`, equivalent a
l'arrondi pres a `legacy`). Test cible :
`pytest -q tests/test_ekf2d_woodbury_update.py`.
`--undistort-keypoints` (opt-in, toutes familles sauf `pose2sim`) dedistord les
keypoints 2D au chargement et invalide les caches geometriques. Test cible :
`pytest -q tests/test_keypoint_undistortion.py`.
`--flight-detection {triangulation,ekf_state}` (defaut `triangulation`),
`--flight-hysteresis-m`, `--flight-com-accel-tolerance` pilotent l'activation
du predicteur `dyn`. Test cible : `pytest -q tests/test_ekf2d_flight_detection.py`.
`--process-noise-model {legacy,white_jerk}` (defaut `legacy`) et
`--process-noise-jerk-psd ROOT_TRANS ROOT_ROT JOINTS` choisissent la matrice `Q`.
Test cible : `pytest -q tests/test_ekf2d_process_noise.py`.
`--ekf2d-joint-prior` (opt-in) et `--ekf2d-joint-prior-axial-std-deg` (defaut 30)
activent limites coude/genou, a priori axial et export canonique. Test cible :
`pytest -q tests/test_ekf2d_joint_prior.py`. Les tests EKF 2D
construisant un modele exigent `biorbd` et `biobuddy` (sinon ils sont sautes) ;
depuis un worktree, definir `BIOBUDDY_ROOT` si le depot frere `../biobuddy`
n'existe pas a cote du worktree.

## Livraison d'un changement

Executer d'abord le test cible de la matrice, puis les controles adequats a la
portee du changement. Pour les fichiers Python modifies, la CI attend :

```bash
isort <fichiers_modifies> --check-only --profile black
black <fichiers_modifies> --check
flake8 <fichiers_modifies>
```

Avant de remettre le travail, verifier `git diff --check`, `git diff` et
`git status --short --branch`. Rapporter distinctement les controles executes,
ceux non executes, les dependances indisponibles et tout impact scientifique
observable. Ne pas committer, pousser, fusionner ou modifier la CI sans une
demande explicite.
