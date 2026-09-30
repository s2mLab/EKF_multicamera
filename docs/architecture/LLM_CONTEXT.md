# Contexte de travail LLM

Lire cette matrice apres `OVERVIEW.md`. Les commandes sont celles observees dans
le depot; choisir le test cible avant la suite complete.

| Tache | Lire d'abord | Valider d'abord |
| --- | --- | --- |
| Nettoyage 2D ou outliers | `vitpose_ekf_pipeline.py`, `tests/test_vitpose_ekf_hotspots.py` | `pytest -q tests/test_vitpose_ekf_hotspots.py` |
| Coherence epipolaire ou geometrie | `vitpose_ekf_pipeline.py`, `calibration_qc.py`, tests hotspots/QC | `pytest -q tests/test_vitpose_ekf_hotspots.py tests/test_calibration_qc.py` |
| Detection ou correction flip L/R | `vitpose_ekf_pipeline.py`, `reconstruction/reconstruction_bundle.py`, tests hotspots/cache | `pytest -q tests/test_vitpose_ekf_hotspots.py tests/test_reconstruction_bundle_pose_cache.py` |
| Triangulation ou reprojection | `vitpose_ekf_pipeline.py`, `reconstruction/reconstruction_bundle.py` | `pytest -q tests/test_vitpose_ekf_hotspots.py tests/test_reconstruction_bundle_pose_cache.py` |
| EKF, modele ou initialisation | `vitpose_ekf_pipeline.py`, `reconstruction/reconstruction_bundle.py`, `tests/test_model_variants.py` | tests touches, puis un cas scientifique de reference si `biorbd` est disponible |
| Metadonnees ou invalidation cache | `reconstruction/reconstruction_bundle.py`, `tests/test_reconstruction_bundle_pose_cache.py` | `pytest -q tests/test_reconstruction_bundle_pose_cache.py` |
| Profil ou CLI de reconstruction | `reconstruction/reconstruction_profiles.py`, registre, scripts CLI | `pytest -q tests/test_reconstruction_profiles.py tests/test_batch_run.py` |
| Annotation 2D | `annotation/`, `pipeline_gui.py`, tests annotation | `pytest -q tests/test_annotation_store.py tests/test_annotation_kinematic_assist.py tests/test_annotation_frame_navigation.py` |
| Calibration QA / choix camera | `calibration_qc.py`, `camera_tools/`, GUI Cameras | `pytest -q tests/test_calibration_qc.py tests/test_camera_metrics.py tests/test_camera_selection.py` |
| Preview, navigation ou rendu 2D | `preview/`, `pipeline_gui.py`, tests preview | `pytest -q tests/test_preview_bundle.py tests/test_preview_navigation.py tests/test_preview_two_d_view.py` |
| GUI Tk ou mise en page | `pipeline_gui.py`, `tests/test_pipeline_gui_launch.py` | `pytest -q tests/test_pipeline_gui_launch.py`; puis `RUN_PIPELINE_GUI_SMOKE=1 pytest -q tests/test_pipeline_gui_launch.py` si Tk/affichage disponibles |
| Analyses cinematiques, DD ou trampoline | le sous-package concerne et son test nomme | le ou les tests de ce sous-package, puis une comparaison de sortie sur un bundle existant |
| Packaging, environnement ou CI | `pyproject.toml`, `environment*.yml`, `.github/workflows/ci.yml` | `pytest -q`; ne modifier la CI/dependances qu'avec une demande explicite |

## Checklist scientifique

Avant de livrer une modification de calcul, consigner : forme et axes des
tableaux, unites, convention de rotation/repere, traitement des `NaN`, seuils
en pixels, FPS/stride, impact sur les caches, et comparaison avec un cas
synthetique ou une sortie de reference. Distinguer les regressions physiques
des variations de backend, de plateforme ou de tolerance numerique.

## Baseline et livraison

La suite complete est `pytest -q`. Le format/lint CI est :

```bash
isort . --check-only --profile black
black . --check --verbose
flake8 .
```

Ne pas lancer ces commandes globales si elles reformateraient des changements
utilisateur non lies. Preferer des chemins de fichiers modifies. Avant toute
livraison, verifier `git diff` et `git status --short --branch`, en distinguant
les changements de cette tache des donnees et profils deja modifies.
