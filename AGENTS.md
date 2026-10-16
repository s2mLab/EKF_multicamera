# Instructions locales

Ce depot est un outil Python de reconstruction biomecanique multi-camera pour
des sequences de trampoline. Lire `docs/architecture/OVERVIEW.md` et
`docs/architecture/LLM_CONTEXT.md` avant toute modification substantielle.

## Demarrage et perimetre

- Executer `git status --short --branch` avant de modifier quoi que ce soit.
- Ne pas modifier, reformater ou regenerer `inputs/`, `output/`, `.cache/` ou
  `reconstruction_profiles*.json` sans demande explicite. Les entrees suivies
  par Git sont des donnees experimentales; `output/` et `.cache/` sont des
  artefacts locaux ignores.
- Utiliser Python 3.11. L'environnement de reference est `vitpose-ekf`, cree
  avec `environment.vitpose-ekf.yml`; `pip install -e .[test]` couvre la suite
  de tests de CI, pas toutes les fonctionnalites `biorbd`/GUI.
- Ne pas ajouter de dependance runtime pendant un audit ou une correction
  localisee sans validation explicite du besoin.

## Contraintes scientifiques

- Preserver l'ordre COCO17 et les formes de tableaux exposees par `PoseData`.
- Verifier explicitement les axes, le repere monde, les pixels, metres,
  radians, FPS, NaN et masques de vues exclues lorsqu'un calcul est modifie.
- Pour les changements de filtrage, flip L/R, coherence, triangulation ou EKF,
  verifier les metadonnees de cache et un test numerique cible avant tout essai
  sur une sequence reelle.
- Ne pas modifier un seuil, une convention de rotation ou un parametre EKF sans
  rendre le changement observable dans les profils, sorties ou tests associes.

## Validation

- Lancer d'abord le test cible indique dans `LLM_CONTEXT.md`, puis `pytest -q`
  si la modification est transverse ou touche un contrat partage.
- Pour le GUI, lancer les tests du module; sur une machine avec Tk et affichage,
  ajouter `RUN_PIPELINE_GUI_SMOKE=1 pytest -q tests/test_pipeline_gui_launch.py`.
- Formater/verifier seulement les fichiers modifies avec `isort`, `black` et
  `flake8`; la CI execute ensuite ces trois controles et `pytest -q` sous
  Python 3.11.
- Inspecter `git diff` et `git status` avant de conclure. Ne pas commit, push ou
  modifier la CI sans demande explicite.
