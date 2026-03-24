# VitPose / EKF Playground

Multi-view pose reconstruction, model building, and kinematic analysis tools for trampoline sequences.

This repository contains:

- a desktop GUI to inspect 2D detections, reconstructions, and analyses
- command-line tools to generate reconstruction bundles and run named profiles
- analysis utilities for root kinematics, DD estimation, trampoline displacement, observability, and 3D segment analysis

## Repository Overview

Main entry points:

- [pipeline_gui.py](/Users/mickaelbegon/Documents/Playground/pipeline_gui.py): main graphical interface
- [vitpose_ekf_pipeline.py](/Users/mickaelbegon/Documents/Playground/vitpose_ekf_pipeline.py): end-to-end pipeline and core algorithms
- [export_reconstruction_bundle.py](/Users/mickaelbegon/Documents/Playground/export_reconstruction_bundle.py): generate one standardized reconstruction bundle
- [run_reconstruction_profiles.py](/Users/mickaelbegon/Documents/Playground/run_reconstruction_profiles.py): run a set of named reconstruction profiles

Main packages:

- [reconstruction](/Users/mickaelbegon/Documents/Playground/reconstruction): bundle generation, dataset handling, timings, profiles, naming
- [kinematics](/Users/mickaelbegon/Documents/Playground/kinematics): root kinematics and 3D analysis
- [camera_tools](/Users/mickaelbegon/Documents/Playground/camera_tools): camera metrics and camera selection helpers
- [judging](/Users/mickaelbegon/Documents/Playground/judging): DD analysis, trampoline displacement, reference codes
- [preview](/Users/mickaelbegon/Documents/Playground/preview): preview bundle loading and frame navigation
- [observability](/Users/mickaelbegon/Documents/Playground/observability): Jacobian-rank analysis
- [analysis](/Users/mickaelbegon/Documents/Playground/analysis): standalone plotting and exploration scripts
- [animation](/Users/mickaelbegon/Documents/Playground/animation): GIF export scripts

## Installation

### 1. Create a Python environment

The project targets Python `>= 3.11`.

Example with `venv`:

```bash
cd /Users/mickaelbegon/Documents/Playground
python3.11 -m venv .venv
source .venv/bin/activate
```

Example with Conda:

```bash
conda create -n vitpose-ekf python=3.11
conda activate vitpose-ekf
```

### 2. Install Python dependencies

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

### 3. Install non-PyPI dependencies

Some parts of the project depend on `biorbd`, and optionally on OpenSim-related tooling depending on your workflow.

For `biorbd`, a Conda install is the safest route:

```bash
conda install -c conda-forge biorbd
```

If you use the GUI, make sure your Python installation has Tk support.

### 4. Optional developer tools

Formatting and tests:

```bash
pip install black pytest
```

## Input Data

Typical inputs:

- calibration file: `inputs/Calib.toml`
- 2D detections: `inputs/<trial>_keypoints.json`
- optional Pose2Sim TRC: `inputs/<trial>.trc`
- optional DD reference file: `inputs/<trial>_DD.json`

Example:

- [inputs/1_partie_0429_keypoints.json](/Users/mickaelbegon/Documents/Playground/inputs/1_partie_0429_keypoints.json)
- [inputs/1_partie_0429.trc](/Users/mickaelbegon/Documents/Playground/inputs/1_partie_0429.trc)
- [inputs/1_partie_0429_DD.json](/Users/mickaelbegon/Documents/Playground/inputs/1_partie_0429_DD.json)

Outputs are typically written under:

- `outputs/<dataset>/models`
- `outputs/<dataset>/reconstructions`
- `outputs/<dataset>/figures`

## Launching the GUI

Run:

```bash
python /Users/mickaelbegon/Documents/Playground/pipeline_gui.py
```

The GUI is organized around a single workflow:

1. Choose the 2D input and dataset root in `2D explorer`
2. Inspect cameras, flips, and candidate issues in `Caméras`
3. Generate models in `Modèle`
4. Define named reconstruction profiles in `Profiles`
5. Run and inspect bundles in `Reconstructions`
6. Compare outputs in the analysis tabs

Main analysis tabs:

- `3D animation`: export comparative 3D GIFs
- `2D multiview`: export multi-camera 2D GIFs
- `DD`: jump segmentation and DD estimation
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
  --calib inputs/Calib.toml \
  --keypoints inputs/1_partie_0429_keypoints.json \
  --output-dir outputs/1_partie_0429/reconstructions/triangulation_exhaustive_flip_rotfix \
  --pose-data-mode cleaned \
  --triangulation-method exhaustive \
  --flip-left-right \
  --initial-rotation-correction \
  --fps 120 \
  --triangulation-workers 6
```

Supported families:

- `pose2sim`
- `triangulation`
- `ekf_3d`
- `ekf_2d`

### Run a list of named profiles

Example:

```bash
python /Users/mickaelbegon/Documents/Playground/run_reconstruction_profiles.py \
  --config reconstruction_profiles.json \
  --output-root outputs \
  --dataset-name 1_partie_0429 \
  --calib inputs/Calib.toml \
  --keypoints inputs/1_partie_0429_keypoints.json \
  --pose2sim-trc inputs/1_partie_0429.trc \
  --fps 120 \
  --triangulation-workers 6
```

To run only some profiles:

```bash
python /Users/mickaelbegon/Documents/Playground/run_reconstruction_profiles.py \
  --config reconstruction_profiles.json \
  --output-root outputs \
  --dataset-name 1_partie_0429 \
  --calib inputs/Calib.toml \
  --keypoints inputs/1_partie_0429_keypoints.json \
  --pose2sim-trc inputs/1_partie_0429.trc \
  --profile ekf_2d_acc_rootq0_boot15_flip_rotfix \
  --profile triangulation_exhaustive_flip_rotfix
```

## Main Algorithms

### 1. 2D pose preprocessing

The pipeline can work from:

- `raw` detections
- `filtered` detections
- `cleaned` detections

Cleaning includes temporal smoothing and outlier rejection based on a robust motion amplitude estimate.

### 2. Left/right flip detection

The project includes several strategies to detect left/right label swaps:

- epipolar Sampson-based scoring
- fast epipolar symmetric-distance scoring
- triangulation + reprojection scoring

Corrected 2D variants are cached, so downstream stages can reuse:

- raw
- cleaned without flip
- cleaned + epipolar flip
- cleaned + fast epipolar flip
- cleaned + triangulation-based flip

### 3. Triangulation

Two main triangulation modes are available:

- `greedy`
- `exhaustive`

The triangulation stage also stores:

- per-frame reprojection error
- view usage
- excluded-camera patterns
- coherence scores

### 4. Root orientation extraction

For geometric reconstructions such as triangulation or Pose2Sim:

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

### 6. EKF 2D

The 2D EKF combines:

- the articulated model
- 2D observations in all cameras
- multiview coherence weighting
- configurable predictor (`acc` or `dyn`)

Important improvements already integrated in the codebase:

- sequential camera updates
- vectorized measurement assembly
- root-pose bootstrap initialization (`root_pose_bootstrap`)

### 7. DD estimation

The `DD` tab and [judging/dd_analysis.py](/Users/mickaelbegon/Documents/Playground/judging/dd_analysis.py) provide:

- jump segmentation from root height
- salto / tilt / twist analysis
- DD code inference
- comparison with expected codes loaded from `*_DD.json`

### 8. Trampoline displacement

The `Toile` tab estimates horizontal displacement penalties:

- contact windows are inferred between jumps segmented in the DD analysis
- contact position currently uses the feet as a proxy
- the bed geometry is based on a calibrated set of trampoline reference markers

### 9. Observability analysis

The `Observabilité` tab computes frame-wise ranks of:

- `J_markers_3D(q)`
- `J_obs_2D(q)`

This helps visualize when the marker or image Jacobians lose rank.

## Project Conventions

- default worker count is `6`
- formatting uses `black` with line length `120`
- tests live in [tests](/Users/mickaelbegon/Documents/Playground/tests)
- the project uses a local Matplotlib cache under `.cache/matplotlib`

## Development

Run tests:

```bash
pytest -q
```

Format code:

```bash
black .
```

The repository also contains CI workflows under:

- [.github/workflows/black.yml](/Users/mickaelbegon/Documents/Playground/.github/workflows/black.yml)
- [.github/workflows/tests.yml](/Users/mickaelbegon/Documents/Playground/.github/workflows/tests.yml)

## Notes

- The GUI is the easiest entry point if you want to explore reconstructions interactively.
- The CLI tools are better when you want reproducible named runs and cached bundles.
- Some advanced analyses require reconstructions with `q`, `qdot`, and an associated `.bioMod`.
