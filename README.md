# VitPose / EKF Playground

Multi-view pose reconstruction, model building, and kinematic analysis tools for trampoline sequences.

This repository contains:

- a desktop GUI to inspect 2D detections, reconstructions, and analyses
- command-line tools to generate reconstruction bundles and run named profiles
- annotation and calibration-QA tools for interactive multiview inspection
- a batch runner with Excel synthesis export
- analysis utilities for root kinematics, DD estimation, execution deductions, trampoline displacement, observability, and 3D segment analysis

## Repository Overview

Main entry points:

- [pipeline_gui.py](/Users/mickaelbegon/Documents/Playground/pipeline_gui.py): main graphical interface
- [vitpose_ekf_pipeline.py](/Users/mickaelbegon/Documents/Playground/vitpose_ekf_pipeline.py): end-to-end pipeline and core algorithms
- [export_reconstruction_bundle.py](/Users/mickaelbegon/Documents/Playground/export_reconstruction_bundle.py): generate one standardized reconstruction bundle
- [run_reconstruction_profiles.py](/Users/mickaelbegon/Documents/Playground/run_reconstruction_profiles.py): run a set of named reconstruction profiles

Main packages:

- [annotation](/Users/mickaelbegon/Documents/Playground/annotation): sparse 2D annotation storage, navigation, kinematic assist, preview rendering
- [reconstruction](/Users/mickaelbegon/Documents/Playground/reconstruction): bundle generation, dataset handling, timings, profiles, naming
- [kinematics](/Users/mickaelbegon/Documents/Playground/kinematics): root kinematics and 3D analysis
- [camera_tools](/Users/mickaelbegon/Documents/Playground/camera_tools): camera metrics and camera selection helpers
- [judging](/Users/mickaelbegon/Documents/Playground/judging): DD analysis, trampoline displacement, reference codes
- [preview](/Users/mickaelbegon/Documents/Playground/preview): preview bundle loading and frame navigation
- [observability](/Users/mickaelbegon/Documents/Playground/observability): Jacobian-rank analysis
- [analysis](/Users/mickaelbegon/Documents/Playground/analysis): standalone plotting and exploration scripts
- [animation](/Users/mickaelbegon/Documents/Playground/animation): GIF export scripts

## Installation

### 1. Create the environment from the YAML file

The simplest setup is:

```bash
cd /Users/mickaelbegon/Documents/Playground
conda env create -f environment.vitpose-ekf.yml
conda activate vitpose-ekf
```

If the environment already exists:

```bash
conda env update -f environment.vitpose-ekf.yml --prune
conda activate vitpose-ekf
```

These environment files already include the developer tools used in the repo:

- `black`
- `isort`
- `flake8`

### 2. Install the project in editable mode

Minimal install:

```bash
pip install -e .
```

With test dependencies:

```bash
pip install -e .[test]
```

With a few extra utilities:

```bash
pip install -e .[full]
```

### 3. Optional: install non-PyPI dependencies manually

Some parts of the project depend on `biorbd`, and optionally on OpenSim-related tooling depending on your workflow.

For `biorbd`, a Conda install is the safest route:

```bash
conda install -c conda-forge biorbd
```

If you use the GUI, make sure your Python installation has Tk support.

### 4. Optional developer tools

Formatting and tests:

```bash
pip install black isort flake8 pytest
```

## Input Data

Typical inputs are organized under `inputs/`:

- calibration file: `inputs/calibration/Calib.toml`
- 2D detections: `inputs/keypoints/<trial>_keypoints.json`
- optional sparse 2D annotations: `inputs/annotations/<trial>_annotations.json`
- optional TRC file: `inputs/trc/<trial>.trc`
- optional DD reference file: `inputs/dd/<trial>_DD.json`
- optional images or extracted frames: typically `inputs/images/<trial>/...` or another sibling folder inferred from the keypoint file

Example:

- [inputs/keypoints/1_partie_0429_keypoints.json](/Users/mickaelbegon/Documents/Playground/inputs/keypoints/1_partie_0429_keypoints.json)
- [inputs/trc/1_partie_0429.trc](/Users/mickaelbegon/Documents/Playground/inputs/trc/1_partie_0429.trc)
- [inputs/dd/1_partie_0429_DD.json](/Users/mickaelbegon/Documents/Playground/inputs/dd/1_partie_0429_DD.json)

Outputs are typically written under:

- `output/<dataset>/models`
- `output/<dataset>/reconstructions`
- `output/<dataset>/figures`

## Launching the GUI

Run:

```bash
python /Users/mickaelbegon/Documents/Playground/pipeline_gui.py
```

The GUI now uses a shared reconstruction selector at the top of the window. Most analysis tabs reuse that selector instead of maintaining their own reconstruction table.

At startup, the GUI shows a small splash/status window while shared caches and preview resources are loaded.

If you try to quit while annotations or reconstruction profiles contain unsaved
changes, the GUI now asks for confirmation before closing.

Typical workflow:

1. Choose the 2D input in `2D analysis`
2. Inspect cameras, flips, and candidate issues in `Cameras`
3. Create or refine sparse 2D points in `Annotation`
4. Generate models in `Models`
5. Define named reconstruction profiles in `Profiles`
6. Run and inspect bundles in `Reconstructions`
7. Inspect calibration quality in `Calibration`
8. Run multi-dataset/profile batches in `Batch`
9. Compare outputs in the analysis tabs

Main analysis tabs:

- `Cameras`: camera ranking, L/R flip inspection, reprojection overlays, QA overlays on top of images
- `Annotation`: sparse 2D multiview annotation with crop, epipolar guides, reprojection helpers, drag editing, and a first kinematic-assist mode
- `Calibration`: 2D epipolar QA + 3D reprojection QA, worst frames, pairwise camera matrix, and spatial quality maps
- `Batch`: scan several keypoint files, run selected profiles, and export an Excel synthesis workbook
- `3D animation`: export comparative 3D GIFs
- `2D multiview`: export multi-camera 2D GIFs
- `DD`: jump segmentation and DD estimation
- `Execution`: localized execution deductions with 3D and 2D overlays, ready for future image overlays
- `Toile`: horizontal displacement scoring on the trampoline bed
- `Racine`: root translations, rotations, or rotation matrices
- `Autres DoF`: left/right joint comparison
- `3D analysis`: segment-length boxplots and angular momentum
- `Observabilité`: Jacobian rank across frames

## Running the Main CLI Tools

### Run one reconstruction bundle

Example:

```bash
python /Users/mickaelbegon/Documents/Playground/export_reconstruction_bundle.py \
  --name triangulation_exhaustive_flip_rotfix \
  --family triangulation \
  --calib inputs/calibration/Calib.toml \
  --keypoints inputs/keypoints/1_partie_0429_keypoints.json \
  --output-dir output/1_partie_0429/reconstructions/triangulation_exhaustive_flip_rotfix \
  --pose-data-mode cleaned \
  --triangulation-method exhaustive \
  --flip-method epipolar_fast \
  --flip-left-right \
  --initial-rotation-correction \
  --fps 120 \
  --triangulation-workers 6
```

Supported families:

- `pose2sim` (TRC file import)
- `triangulation`
- `ekf_3d`
- `ekf_2d`

The CLI and GUI both support `raw`, `cleaned`, and, when available, `annotated` 2D inputs.

### Run a list of named profiles

Example:

```bash
python /Users/mickaelbegon/Documents/Playground/run_reconstruction_profiles.py \
  --config reconstruction_profiles.json \
  --dataset-name 1_partie_0429 \
  --calib inputs/calibration/Calib.toml \
  --keypoints inputs/keypoints/1_partie_0429_keypoints.json \
  --trc-file inputs/trc/1_partie_0429.trc \
  --fps 120 \
  --triangulation-workers 6
```

To run only some profiles:

```bash
python /Users/mickaelbegon/Documents/Playground/run_reconstruction_profiles.py \
  --config reconstruction_profiles.json \
  --dataset-name 1_partie_0429 \
  --calib inputs/calibration/Calib.toml \
  --keypoints inputs/keypoints/1_partie_0429_keypoints.json \
  --trc-file inputs/trc/1_partie_0429.trc \
  --profile ekf_2d_acc_rootq0_boot15_flip_rotfix \
  --profile triangulation_exhaustive_flip_rotfix
```

## Main Algorithms

### 1. 2D pose preprocessing

The pipeline can work from:

- `raw` detections
- `cleaned` detections
- sparse `annotated` detections

Cleaning includes temporal smoothing and outlier rejection based on a robust motion amplitude estimate.

### 2. Left/right flip detection

The project includes several strategies to detect left/right label swaps:

- epipolar Sampson-based scoring
- fast epipolar symmetric-distance scoring
- triangulation + reprojection scoring with `once`, `greedy`, or `exhaustive` variants

Corrected 2D variants are cached, so downstream stages can reuse:

- raw
- cleaned without flip
- cleaned + epipolar flip
- cleaned + fast epipolar flip
- cleaned + triangulation-based flip
- annotated-only sparse observations in the GUI calibration and annotation workflows

For epipolar-family methods, the current implementation also applies a simple
2-state Viterbi decoding (`normal` / `flipped`) only when you explicitly pick
`epipolar_viterbi` or `epipolar_fast_viterbi`.

### 3. Triangulation

Three triangulation modes are available:

- `once`: one weighted triangulation pass using the currently available views.
- `greedy`: starts from all available views and removes the worst ones step by step.
- `exhaustive`: tests more camera combinations and is the most robust, but also the slowest.

The triangulation stage also stores:

- per-frame reprojection error
- view usage
- excluded-camera patterns
- per-frame/keypoint/camera excluded-view masks usable in the GUI
- coherence scores

### 4. Root orientation extraction

For geometric reconstructions such as triangulation or a TRC file import:

- the trunk frame is built from hips and shoulders
- the root orientation is expressed with the `YXZ` Euler sequence
- an optional initial yaw correction (`rotfix`) aligns the trunk to the global frame by snapping to the nearest right angle

For model-based reconstructions:

- root rotations are read from the model generalized coordinates
- the GUI can also display the corresponding rotation matrix components

### 5. EKF 3D

The 3D EKF uses model-based kinematics driven by 3D marker trajectories.

Recent initialization strategies include:

- triangulation-based initialization
- root-only initialization with zero rest of the body (`root_pose_zero_rest`)

The GUI root-analysis views also support short-gap interpolation before unwrap for visualization/export.

### 6. EKF 2D

The 2D EKF combines:

- the articulated model
- 2D observations in all cameras
- multiview coherence weighting
- configurable predictor (`acc`, `dyn`, `history3`, or `dyn_history3`)

Important improvements already integrated in the codebase:

- sequential camera updates
- vectorized measurement assembly
- root-pose bootstrap initialization (`root_pose_bootstrap`)
- configurable coherence families: `epipolar`, `epipolar_fast`, `triangulation_once`, `triangulation_greedy`, `triangulation_exhaustive`
- runtime left/right gate mode inside the EKF2D loop (`ekf_prediction_gate`)
- optional segmented-back model variants, including upper-trunk-root variants
- storage of excluded views for later QA overlays in the GUI
- higher-order history-based prediction from the last three corrected states
- optional trampoline-contact pseudo-observations for the ankles
- lower confidence on views detected as left/right-flipped so they still help
  the filter without dominating it

### 7. Model building

The `Models` tab and bundle generation code support:

- several trunk/back structures, including segmented-back variants
- optional left/right limb symmetrization (`Symmetrize limbs`)
- model creation from `raw`, `cleaned`, or `annotated` 2D observations
- preview of the segmented back with a dedicated `mid_back` marker and 2-triangle back geometry

### 8. Annotation workflow

The `Annotation` tab provides:

- sparse per-camera / per-frame / per-marker JSON annotation storage
- image-backed multiview annotation with brightness/contrast, crop `+20%`, zoom and pan
- epipolar guides, triangulated reprojection hints, and reprojection from the selected reconstruction
- frame subsets such as `Flipped L/R` and `Worst reproj 5%`
- `Reproject -> Confirm` replacement of already annotated points only
- drag editing of existing 2D points with optional snap to reprojection or epipolar guides
- a first `Kinematic assist` mode:
  - choose an existing `.bioMod`
  - estimate an initial `q` on the current frame
  - keep and propagate local kinematic states across frames
  - run short local EKF/direct-fit corrections when annotated points are edited

### 9. Calibration QA

The `Calibration` tab provides:

- pairwise epipolar consistency per camera pair
- global trimming of the worst `2D` samples before aggregation
- per-camera and per-frame diagnostics
- a `Worst frames` list with quick jump to `Cameras`
- 3D reprojection summaries from the selected reconstruction
- spatial non-uniformity metrics across 3D space, including binned `X/Z` maps
- local choice of the 2D source: `raw`, `cleaned`, or `annotated`

### 10. DD estimation

The `DD` tab and [judging/dd_analysis.py](/Users/mickaelbegon/Documents/Playground/judging/dd_analysis.py) provide:

- jump segmentation from root height
- salto / tilt / twist analysis
- DD code inference
- comparison with expected codes loaded from `*_DD.json`
- comparison of each reconstruction against the expected DD reference with color-coded status

### 11. Execution deductions

The `Execution` tab and [judging/execution.py](/Users/mickaelbegon/Documents/Playground/judging/execution.py) provide:

- per-jump localized deductions
- a synchronized 3D view and 2D camera overlay
- session-level time-of-flight scoring
- a structure ready for direct image overlays as soon as camera frames are available

### 12. Trampoline displacement

The `Toile` tab estimates horizontal displacement penalties:

- contact windows are inferred between jumps segmented in the DD analysis
- contact position currently uses the feet as a proxy
- the bed geometry is based on a calibrated set of trampoline reference markers

### 13. Batch execution and synthesis

The batch backend and `Batch` tab support:

- scanning a set of keypoint files
- selecting a profiles file
- running the chosen reconstructions for all detected trials
- exporting an Excel workbook summarizing:
  - reconstruction options
  - timings by stage
  - reprojection metrics
  - recognition / failure summaries

### 14. Observability analysis

The `Observabilité` tab computes frame-wise ranks of:

- `J_markers_3D(q)`
- `J_obs_2D(q)`

This helps visualize when the marker or image Jacobians lose rank.

## Project Conventions

- default worker count is `6`
- default output root is `output/`
- formatting uses `isort` + `black`
- linting uses `flake8`
- tests live in [tests](/Users/mickaelbegon/Documents/Playground/tests)
- the project uses a local Matplotlib cache under `.cache/matplotlib`

## Development

Run tests:

```bash
pytest -q
```

Format code:

```bash
isort . --profile black
black .
```

Lint code:

```bash
flake8 .
```

The repository contains a single CI workflow under:

- [.github/workflows/ci.yml](/Users/mickaelbegon/Documents/Playground/.github/workflows/ci.yml)

## TODO

- Improve hip and knee flexion handling for EKF outputs to distinguish `piked` and `grouped` body shapes more robustly.
- Compute hip and knee flexion angles directly from triangulated 3D data.
- Continue developing the execution-error analysis module.
- Improve the annotation kinematic assist with a short local temporal EKF window around the current frame.
- Export calibration QA summaries directly in the batch Excel workflow.
- Better control the foot position on the trampoline bed: when a foot is in contact with the bed, keep it fixed in the horizontal plane with a high-confidence pseudo-observation during the whole contact phase.

## Notes

- The GUI is the easiest entry point if you want to explore reconstructions interactively.
- The CLI tools are better when you want reproducible named runs and cached bundles.
- Some advanced analyses require reconstructions with `q`, `qdot`, and an associated `.bioMod`.
