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

- [pipeline_gui.py](./pipeline_gui.py): main graphical interface
- [vitpose_ekf_pipeline.py](./vitpose_ekf_pipeline.py): end-to-end pipeline and core algorithms
- [export_reconstruction_bundle.py](./export_reconstruction_bundle.py): generate one standardized reconstruction bundle
- [run_reconstruction_profiles.py](./run_reconstruction_profiles.py): run a set of named reconstruction profiles

Main packages:

- [annotation](./annotation): sparse 2D annotation storage, navigation, kinematic assist, preview rendering
- [reconstruction](./reconstruction): bundle generation, dataset handling, timings, profiles, naming
- [kinematics](./kinematics): root kinematics and 3D analysis
- [camera_tools](./camera_tools): camera metrics and camera selection helpers
- [judging](./judging): DD analysis, trampoline displacement, reference codes
- [preview](./preview): preview bundle loading and frame navigation
- [observability](./observability): Jacobian-rank analysis
- [analysis](./analysis): standalone plotting and exploration scripts
- [animation](./animation): GIF export scripts

For contributors and coding agents, see [the handoff guide](./docs/AGENT_HANDOFF.md),
[the architecture overview](./docs/architecture/OVERVIEW.md), [the task-to-validation matrix](./docs/architecture/LLM_CONTEXT.md),
and the local [instructions](./AGENTS.md).

## Installation

### 1. Create the environment from the YAML file

The simplest setup is:

```bash
cd path/to/EKF_multicamera  # root of your clone
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

Installation levels:

| Usage | Supported installation |
| --- | --- |
| Core modules and CI test suite | `pip install -e .[test]` |
| GUI and biomechanical reconstruction | `conda env create -f environment.vitpose-ekf.yml`, then `pip install -e .[test]` |
| Batch Excel export | Full Conda environment with `openpyxl` available |
| Pyorerun visualization | Full Conda environment |

`.[full]` only provides the PyPI extras (`opencv-python`, `pandas`, and
`plotly`). It does not install `biorbd`, `pyorerun`, `bioviz`, `rerun-sdk`, Tk,
or `openpyxl`.

`environment.vitpose-ekf.yml` is the reference environment for this repository.
`environment.yml` is a historical alternative with the same Conda environment
name and a different dependency set; do not treat the two files as equivalent.

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

- [inputs/keypoints/1_partie_0429_keypoints.json](./inputs/keypoints/1_partie_0429_keypoints.json)
- [inputs/trc/1_partie_0429.trc](./inputs/trc/1_partie_0429.trc)
- [inputs/dd/1_partie_0429_DD.json](./inputs/dd/1_partie_0429_DD.json)

Outputs are typically written under:

- `output/<dataset>/models`
- `output/<dataset>/reconstructions`
- `output/<dataset>/figures`

## Launching the GUI

Run:

```bash
python pipeline_gui.py
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
python export_reconstruction_bundle.py \
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

`--undistort-keypoints` (opt-in, profile field `undistort_keypoints`, not for
`pose2sim`) undistorts the 2D keypoints once at load time with the
`distortions` coefficients of `Calib.toml`, so that coherence, triangulation
and EKF use an exact pinhole model. By default the keypoints are used as
detected (historical behavior). Toggling the option invalidates the geometric
caches. On the first 300 frames of `1_partie_0429` the keypoints move by 2.5 px
on average and the exhaustive triangulation reprojection error drops from
9.55 to 9.38 px (mean).

### Run a list of named profiles

Example:

```bash
python run_reconstruction_profiles.py \
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
python run_reconstruction_profiles.py \
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

Measurement update solver (`--ekf2d-update-method`, profile field
`ekf2d_update_method`):

- `woodbury` (default): information-form batch update that only solves
  `Q x Q` systems (`G = H_q^T R^-1 H_q`) and keeps the Joseph covariance form.
  It is algebraically identical to the historical solver (relative differences
  around `1e-14` on a real 240-frame sequence) and about 4x faster on the whole
  EKF2D loop. It falls back to `legacy` automatically if the reduced system is
  not solvable.
- `legacy`: innovation-space update (sequential per camera, batch when
  pseudo-observations are active).

Flight criterion of the `dyn` / `dyn_history3` predictors
(`--flight-detection`, profile field `flight_detection`):

- `triangulation` (default): every triangulated point of the previous frames
  above `--flight-height-threshold-m`. With `--ekf2d-3d-source first_frame_only`
  there is no 3D support after frame 0, so `dyn` never activates.
- `ekf_state`: the lowest model marker of the previous corrected EKF state is
  above the threshold (with `--flight-hysteresis-m`, default 0.05 m, and an
  optional ballistic gate `--flight-com-accel-tolerance`). On the first 900
  frames of `1_partie_0429` in `first_frame_only`-like mode, `dyn` becomes active
  on 740 frames instead of 0.

Process noise (`--process-noise-model`, profile field `process_noise_model`):

- `legacy` (default): diagonal `Q` independent of the time step.
- `white_jerk`: exact discretization of continuous white jerk,
  `Q(dt) = q_c * [[dt^5/20, dt^4/8, dt^3/6], [dt^4/8, dt^3/3, dt^2/2], [dt^3/6, dt^2/2, dt]]`
  per DoF, with one density per group given by
  `--process-noise-jerk-psd ROOT_TRANS ROOT_ROT JOINTS` (default
  `200 1000 10000`, in m^2/s^5 and rad^2/s^5, calibrated offline; on the first
  240 frames of `1_partie_0429` the median reprojection error is 12.26 px versus
  12.91 px with `legacy`).

Joint prior (`--ekf2d-joint-prior`, opt-in, profile field `joint_prior`): the
elbow and knee have an unobservable mirror branch (`RotZ + pi`, `-RotY`). The
option reflects the state into the anatomical branch, enforces the flexion sign
(elbow <= -1 deg, knee >= +1 deg), adds `FOREARM:RotZ` / `THIGH:RotZ ~ N(0, sigma)`
pseudo-observations (`--ekf2d-joint-prior-axial-std-deg`, default 30) and
exports q in the canonical branch. On 900 real frames, mirrored frames drop from
about 48 % per limb to 0 % and the forearm axial range from 2115 to 251 deg.

Robust measurements (`--ekf2d-robust-mixture`, opt-in, profile field
`robust_mixture`): each 2D keypoint gets an inlier/outlier weight from a
Gaussian-plus-uniform mixture (`--ekf2d-robust-outlier-prob`, default 0.03,
uniform over the image area) and its variance is inflated accordingly, so gross
detection errors are neutralized. A lock guard keeps the mixture from rejecting
the very detections that would correct a wrong prediction: a frame with more than
`--ekf2d-robust-lock-fraction` (default 0.5) of its keypoints at `w < 0.5` is
updated with the nominal variances until the fraction falls to
`--ekf2d-robust-resume-fraction` (default 0.25), every EKF (bootstrap included)
starts in that suspended mode, and a keypoint rejected in more than half of its
views gets back its nominal variance. `1 1` disables the guard (the unguarded
mixture diverged with `white_jerk` on a real sequence). Counters are in
`robust_mixture_stats` (`suspended_frames`, `lock_events`,
`keypoint_guard_restored`, `applied_downweighted_below_0_5`).

Known limitation: in `dyn`/`history3` modes the covariance is still propagated
with the constant-acceleration transition matrix.

### 6.b Complexity overview

The dominant asymptotic costs below use:

- `F`: number of frames
- `C`: number of cameras
- `M`: number of observed model markers or keypoints per frame
- `Q`: number of generalized coordinates / DoF
- `L = 2 * C * M`: approximate 2D measurement dimension for one frame

| Method | Main cost per frame | Full-sequence cost | Notes |
| --- | --- | --- | --- |
| Triangulation `once` | `O(M * C)` | `O(F * M * C)` | One weighted triangulation and one reprojection pass per marker. |
| Triangulation `greedy` | `O(M * C^2)` | `O(F * M * C^2)` | Repeatedly removes the worst view, so the camera loop is effectively quadratic. |
| Triangulation `exhaustive` | `O(M * 2^C)` worst case | `O(F * M * 2^C)` worst case | Tries many camera subsets; practical cost depends on the number of valid views. |
| EKF2D `acc` | `O(Q^3 + L^3 + L^2 * Q)` | `O(F * (Q^3 + L^3 + L^2 * Q))` | Prediction is dominated by covariance propagation; update is dominated by the Kalman solve on image measurements. |
| EKF2D `dyn` | `O(Q^3 + L^3 + L^2 * Q)` | `O(F * (Q^3 + L^3 + L^2 * Q))` | Same asymptotic order as `acc`, with a larger constant when root flight dynamics are active. |
| EKF2D `history3` | `O(Q^3 + L^3 + L^2 * Q)` | `O(F * (Q^3 + L^3 + L^2 * Q))` | Same asymptotic order as `acc`; the higher-order predictor adds only `O(Q)` state-history work. |
| EKF2D `dyn_history3` | `O(Q^3 + L^3 + L^2 * Q)` | `O(F * (Q^3 + L^3 + L^2 * Q))` | Same asymptotic order as `dyn`; root uses `dyn`, joints use the smoothed history-based predictor. |

With the default `woodbury` update, the `L^3 + L^2 * Q` update terms become
`O(L * Q^2 + Q^3)`; the table keeps the `legacy` costs.

In practice:

- triangulation cost scales mostly with the number of cameras and valid keypoints
- EKF2D cost scales most sharply with the measurement dimension `L`, so reducing
  cameras or sparse observations can change runtime more than reducing `Q`
- `history3` and `dyn_history3` are intended as better predictors, not faster
  filters; their overhead is small compared with the Kalman update itself

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

The `DD` tab and [judging/dd_analysis.py](./judging/dd_analysis.py) provide:

- jump segmentation from root height
- salto / tilt / twist analysis
- DD code inference
- comparison with expected codes loaded from `*_DD.json`
- comparison of each reconstruction against the expected DD reference with color-coded status

### 11. Execution deductions

The `Execution` tab and [judging/execution.py](./judging/execution.py) provide:

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
- tests live in [tests](./tests)
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

- [.github/workflows/ci.yml](./.github/workflows/ci.yml)

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
