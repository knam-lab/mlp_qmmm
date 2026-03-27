# MLP_QMMM v0.1 — User Guide

Machine-learning QM/MM framework with explicit handling of QM atoms and embedded MM charges.  
Implements descriptor + electrostatic feature networks (`DeepPotMM`), flexible YAML-driven training, and reproducible testing.

<br>

**Major Pitfalls**
- Ensure your training data files are correctly formatted as per the **Training Data Structure** section.
- The current DeepPotMM model requires the QM region to be same across all frames listed in same order.

---

## Contents

- [Installation](#installation)
- [Training Data Structure](#training-data-structure)
- [Config Files Overview](#config-files-overview)
  - [Option meanings (quick reference)](#option-meanings-quick-reference)
- [Train Models](#train-models)
  - [Basic configs (MLP / ΔMLP)](#basic-configs-mlp--δmlp)
  - [Intermediate configs (MLP / ΔMLP)](#intermediate-configs-mlp--δmlp)
  - [Full‑featured config (works for both)](#fullfeatured-config-works-for-both)
- [Test Models](#test-models)
- [Artifacts Written by Training](#artifacts-written-by-training)
- [Notes & Tips](#notes--tips)

---

<div style="page-break-before: always;"></div>

## Installation

**Note**
The mlp_qmmm is already setup on node21, node22 and node31. Follow the instructions below only if you want to use that environment.
```bash
source /usr/local/miniforge3/activate_conda_env.sh
conda activate mlpqmmmv01env
```

**Otherwise, to install from scratch, follow these steps:**
```bash
# Miniconda (Linux x86_64)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Environment
conda create -n mymlpqmmmv01env python=3.13 -y  # should be below 3.13
conda activate mymlpqmmmv01env

# Core deps
pip install ipython numpy scipy matplotlib mdanalysis multiprocess tqdm pandas pyyaml setuptools

# PyTorch (CUDA 12.x example; pick from pytorch.org for your system)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install this repo (editable)
cd <package_base_dir>
pip install -e .
```

Two CLIs become available:

- `qmmm-train` → trains a model from a YAML
- `qmmm-test` → evaluates a trained run folder (wrapper of `c_testing.py`)

---

<div style="page-break-before: always;"></div>

## Training Data Structure

Your parser accepts **sequential per‑frame blocks** either in a **single file** (blocks back‑to‑back with no blank lines) **or** in **multiple files** (one or more blocks per file).

**Per‑frame template**

```
!E:    <E_high>    <E_low>
!QM region: <N_qm>
<idx> <Z> <Q>  <x> <y> <z> <gx_high> <gy_high> <gz_high> <gx_low> <gy_low> <gz_low>
...
!MM region: <N_mm>
<idx> 1 <Q> <x> <y> <z> <gx_high> <gy_high> <gz_high> <gx_low> <gy_low> <gz_low>
...
!End
```

**Interpretation**

- `idx` — integer index within the frame.  
- `Z` — atomic number (QM only). For **MM lines** the second column is **`1`** meaning “real MM site”; padded rows (added during parsing to reach `max_mm`) use **type `0`** and are masked out by `mm_type`.
- `Q` — partial charge.  
- Coordinates are in Å. Gradients/forces are ∂E/∂R.  
- The parser emits predictor tensors and masks:
  - `qm_coords (B, N_qm, 3)`, `qm_Z (B, N_qm)`, `atom_types (B, N_qm)`
  - `mm_coords (B, M_max, 3)`, `mm_Q (B, M_max)`, `mm_type (B, M_max)` with `mm_type>0` ⇒ real
  - target key fields (energies/gradients) depending on `adapter_kwargs` which include:
    - `energy (B,)` — E_high or ΔE based on energy_mode
    - `qm_grad_high (B, N_qm, 3)`, `qm_grad_low (B, N_qm, 3)`
    - `mm_grad_high (B, M_max, 3)`, `mm_grad_low (B, M_max, 3)`
    - `qm_dgrad (B, N_qm, 3)`, `mm_dgrad (B, M_max, 3)` — Δ(∂E/∂R)
    - `mm_espgrad_high (B, M_max, 3)`, `mm_espgrad_low (B, M_max, 3)` — (∂E/∂R)/q
    - `mm_espgrad_d (B, M_max, 3)` — Δ(∂E/∂R)/q

---

<div style="page-break-before: always;"></div>

## Config Files Overview

All configs are YAML. Unspecified options fall back to code defaults. Below, each example includes **inline comments** describing what each option does.

### Option meanings (quick reference)

- **Top‑level**
  - `adapter` — name/module of the reader in `mlp_qmmm.a_input_types`.
  - `input` — folder or file glob for the frame text files.
  - `adapter_kwargs` — parsing knobs: energy target mode, outlier filters, padding, multi‑process reading, summaries.
  - `model` — import path, class, input keys, (optional) network widths.
  - `losses` — per‑key losses with base weights.
  - `weighting` — (optional) uncertainty MTL (learned sigmas).
  - `weight_schedule` — LR‑ratio‑based reweighting of *groups* (energy vs gradients).
  - `optim` — Adam + LR scheduler (`plateau | exponential | mixed | onecycle`).
  - `trainer` — runtime controls, early/goal stopping, logging, printing, AMP, etc.
  - `masks` — map/wildcards from prediction key patterns to mask tensors (e.g., `mm_*: mm_type`).

- **Energy modes**
  - **MLP**: `adapter_kwargs.energy_mode: high_demean`, `compute_deltas: false`  
    Train on high‑level properties: `energy`, `qm_grad_high`, `mm_espgrad_high`.
  - **ΔMLP**: `adapter_kwargs.energy_mode: diff_demean`, `compute_deltas: true`  
    Train on **differences**: `energy` (ΔE), `qm_dgrad`, `mm_espgrad_d`.

---

## Train Models

Run training with:

```bash
# If testing, make sure to "pkill qmmm-train" and "pkill python*" if you have a previous instance running to avoid memory overload!
qmmm-train path/to/config.yml
```

The model used here (`DeepPotMM`) returns a dict with keys compatible with either MLP or ΔMLP losses.

<div style="page-break-before: always;"></div>

### Basic configs (MLP / ΔMLP)

#### MLP — **Basic** (high‑level targets)

```yaml
# =============================================
# BASIC — MLP (high-level targets; minimal)
# =============================================
adapter: charmmmndo97mts                 # reader module (auto-resolved by a_parser)
input: ./data_100ps_39k                 # folder or file/glob with frame text

adapter_kwargs:
  workers: 8                    # number of parser processes
  mp_chunk: 2                   # files per worker "wave" (load balancing)
  verbose: true                 # print parsing progress
  energy_mode: high_demean              # use E_high demeaned as energy target
  compute_deltas: false                 # don't emit Δ targets
  write_summary: true                   # write parse summaries (safe to leave on)
  summary_json_path: runs/mlp_basic/parse_summary.json  # output location
  summary_txt_path:  runs/mlp_basic/parse_summary.txt   # output location

model:
  module: mlp_qmmm.b_nn_types.dp_qmmm_implicit  # model import path
  class: DeepPotMM                              # class name
  inputs: [qm_coords, atom_types, mm_coords, mm_Q, mm_type]  # forward(...) inputs
  # widths default internally; omit unless overriding:
  # descriptor:   { n_types: 6, neuron: [25,50,100], axis_neuron: 4 }
  # esp:          { n_types: 6, neuron: [5,10,20],   axis_neuron: 4 }
  # fitting_net:  { n_types: 6, neuron: [240,240,240] }

losses:
  energy:          { type: mse, weight: 1.0 }   # E_high (demeaned)
  qm_grad_high:    { type: mse, weight: 1.0 }   # QM ∂E/∂R (high)
  mm_espgrad_high: { type: mse, weight: 10.0 }  # (∂E/∂R)/q (high)

trainer:
  out_dir: runs/mlp_basic                # where to write artifacts
  save_path: runs/mlp_basic/best.pt      # best checkpoint path

masks:
  mm_*: mm_type                                # mask out padded MM (mm_type==0)
```

<div style="page-break-before: always;"></div>

#### ΔMLP — **Basic** (difference targets)

```yaml
# =============================================
# BASIC — ΔMLP (difference targets; minimal)
# =============================================
adapter: charmmmndo97mts
input: ./data_100ps_39k

adapter_kwargs:
  workers: 8                    # number of parser processes
  mp_chunk: 2                   # files per worker "wave" (load balancing)
  verbose: true                 # print parsing progress
  energy_mode: diff_demean              # use ΔE (demeaned)
  compute_deltas: true                  # emit Δ grads/espgrads
  write_summary: true
  summary_json_path: runs/dmlp_basic/parse_summary.json
  summary_txt_path:  runs/dmlp_basic/parse_summary.txt

model:
  module: mlp_qmmm.b_nn_types.dp_qmmm_implicit
  class: DeepPotMM
  inputs: [qm_coords, atom_types, mm_coords, mm_Q, mm_type]
  # widths default internally; omit unless overriding:
  # descriptor:   { n_types: 6, neuron: [25,50,100], axis_neuron: 4 }
  # esp:          { n_types: 6, neuron: [5,10,20],   axis_neuron: 4 }
  # fitting_net:  { n_types: 6, neuron: [240,240,240] }

losses:
  energy:        { type: mse, weight: 1.0 }   # ΔE
  qm_dgrad:      { type: mse, weight: 1.0 }   # Δ(∂E/∂R) on QM
  mm_espgrad_d:  { type: mse, weight: 10.0 }  # (∂ΔE/∂R)/q

trainer:
  out_dir: runs/dmlp_basic
  save_path: runs/dmlp_basic/best.pt

masks:
  mm_*: mm_type                                # mask out padded MM (mm_type==0)
```
---

<div style="page-break-before: always;"></div>


### Intermediate configs (MLP / ΔMLP)

Adds: faster parse knobs, outlier filters, explicit widths, base weights, LR‑group weighting, exponential LR, runtime knobs, compact printing.

#### MLP — **Intermediate**

```yaml
# ------------------------------------------------
# INTERMEDIATE — MLP (E_high demeaned)
# ------------------------------------------------
adapter: charmmmndo97mts
input: ./data_100ps_39k

adapter_kwargs:
  # --- faster parsing ---
  workers: 8                    # number of parser processes based CPU cores availability
  mp_chunk: 2                   # files per worker "wave" (load balancing)
  verbose: true                 # print parsing progress

  energy_mode: high_demean
  compute_deltas: false
  max_mm: 3000                  # pad/truncate MM to this many sites (tensor width)
  # --- outlier filtering ---
  energy_outlier: true
  energy_outlier_fence_k: 3.5   # Tukey fence constant
  grad_outlier: true
  grad_outlier_fence_k: 3.5
  grad_outlier_metric: mean_l2  # mean_l2 | max_l2 | median_l2 (vector norm over atoms)
  grad_outlier_keys: ["qm_grad_high","qm_grad_low","mm_grad_high","mm_grad_low"]  # keys used only to score outliers
  write_summary: true
  summary_json_path: runs/mlp_inter/parse_summary.json
  summary_txt_path:  runs/mlp_inter/parse_summary.txt

model:
  module: mlp_qmmm.b_nn_types.dp_qmmm_implicit
  class: DeepPotMM
  inputs: [qm_coords, atom_types, mm_coords, mm_Q, mm_type]
  # --- explicit widths (override defaults if desired) ---
  descriptor:   { n_types: 4, neuron: [25,50,100], axis_neuron: 4 }
  esp:          { n_types: 4, neuron: [5,10,20],   axis_neuron: 4 }
  fitting_net:  { n_types: 4, neuron: [240,240,240] }

losses:
  energy:            { type: mse, weight: 1.0 }
  qm_grad_high:      { type: mse, weight: 1.0 }
  mm_espgrad_high:   { type: mse, weight: 10.0 }

# LR-ratio-based reweighting — stronger emphasis on gradient losses early
weight_schedule:
  energy_keys: ["energy","dE"]  # patterns treated as “energy group”
  grad_keys:   ["qm_*","mm_*"]  # patterns treated as “gradient group”
  w_ene: 1.0
  w_grad_max: 100.0             # at start_lr ratio=1

optim:
  lr: 5.0e-4
  weight_decay: 0.0
  schedule: exponential          # only this one is active
  exponential:
    factor: 0.99                 # per-epoch decay
    min_lr: 1.0e-6               # clamp floor

trainer:
  batch_size: 32
  epochs: 500
  val_fraction: 0.20
  test_fraction: 0.10
  seed: 2
  print_units: kcal              # printing units (training still in eV)
  print_detail: short            # short: RMSE only; full: include means/σ
  print_keys: ["energy","qm_grad_high","mm_espgrad_high"]  # visible keys on console
  out_dir: runs/mlp_inter                # where to write artifacts
  save_path: runs/mlp_inter/best.pt      # best checkpoint path

masks:
  mm_*: mm_type                                # mask out padded MM (mm_type==0)

```

<div style="page-break-before: always;"></div>


#### ΔMLP — **Intermediate**

```yaml
# ------------------------------------------------
# INTERMEDIATE — ΔMLP (difference targets)
# ------------------------------------------------
adapter: charmmmndo97mts
input: ./data_100ps_39k

adapter_kwargs:
  workers: 8                    # number of parser processes based CPU cores availability
  mp_chunk: 2                   # files per worker "wave" (load balancing)
  verbose: true                 # print parsing progress
  energy_mode: diff_demean
  compute_deltas: true
  max_mm: 3000
  energy_outlier: true
  energy_outlier_fence_k: 3.5
  grad_outlier: true
  grad_outlier_fence_k: 3.5
  grad_outlier_metric: mean_l2
  grad_outlier_keys: ["qm_grad_high","qm_grad_low","mm_grad_high","mm_grad_low"]  # still OK to use high/low for scoring
  write_summary: true
  summary_json_path: runs/dmlp_inter/parse_summary.json
  summary_txt_path:  runs/dmlp_inter/parse_summary.txt

model:
  module: mlp_qmmm.b_nn_types.dp_qmmm_implicit
  class: DeepPotMM
  inputs: [qm_coords, atom_types, mm_coords, mm_Q, mm_type]
  # --- explicit widths (override defaults if desired) ---
  descriptor:   { n_types: 4, neuron: [25,50,100], axis_neuron: 4 }
  esp:          { n_types: 4, neuron: [5,10,20],   axis_neuron: 4 }
  fitting_net:  { n_types: 4, neuron: [240,240,240] }

losses:
  energy:        { type: mse, weight: 1.0 }   # ΔE
  qm_dgrad:      { type: mse, weight: 1.0 }   # Δ(∂E/∂R)_QM
  mm_espgrad_d:  { type: mse, weight: 10.0 }  # (∂ΔE/∂R)/q

weight_schedule:
  energy_keys: ["energy","dE"]
  grad_keys:   ["qm_*","mm_*"]
  w_ene: 1.0
  w_grad_max: 100.0

optim:
  lr: 5.0e-4
  weight_decay: 0.0
  schedule: exponential
  exponential:
    factor: 0.99
    min_lr: 1.0e-6

trainer:
  batch_size: 32
  epochs: 500
  val_fraction: 0.20
  test_fraction: 0.10
  seed: 2
  print_units: kcal
  print_detail: short
  print_keys: ["energy","qm_dgrad","mm_espgrad_d"]
  out_dir: runs/dmlp_inter                # where to write artifacts
  save_path: runs/dmlp_inter/best.pt      # best checkpoint path

masks:
  mm_*: mm_type                                # mask out padded MM (mm_type==0)
```

---

<div style="page-break-before: always;"></div>

### Full‑featured config (works for both)

Contains **all** options with comments. Only the scheduler named in `optim.schedule` is active; other scheduler blocks are safely ignored.

```yaml
# ============================================================
# FULL FEATURED — switch MLP/ΔMLP via energy_mode + losses
# ============================================================

# -----------------------------
# DATA / ADAPTER
# -----------------------------
adapter: charmmmndo97mts                    # module autoloaded via a_parser
input: ./dataset/                          # folder with frame text files
adapter_kwargs:
  # --- parallel parsing ---
  workers: 8                    # number of parser processes based CPU cores availability
  mp_chunk: 2                   # files per worker "wave" (load balancing)
  verbose: true                 # print parsing progress
  # --- target mode ---
  energy_mode: diff_demean                 # Options: 
  compute_deltas: true                     # emit Δ targets (false for plain MLP)
  demean: true                             # mean-center energies (safe default)

  # --- outliers ---
  energy_outlier: true                     # Tukey filter on energies
  energy_outlier_fence_k: 3.5
  de_outlier: false                        # optional Tukey on ΔE (off by default)
  grad_outlier: true                       # gradient outlier filtering
  grad_outlier_fence_k: 3.5
  grad_outlier_metric: mean_l2             # mean_l2 | max_l2 | median_l2
  grad_outlier_keys: ["qm_grad_high","qm_grad_low","mm_grad_high","mm_grad_low"]

  # --- padding & charges ---
  max_mm: 5000                             # MM width after pad/truncate
  mm_charge_zero_eps: 0.0                  # below → treated as zero (masked)

  # --- IO ---
  write_summary: true                      # write summary files
  verbose: false                           # extra parse logs
  summary_json_path: runs/full/parse_summary.json   # optional explicit path
  summary_txt_path:  runs/full/parse_summary.txt

# -----------------------------
# MODEL
# -----------------------------
model:
  module: mlp_qmmm.b_nn_types.dp_qmmm_implicit
  class: DeepPotMM
  inputs: [qm_coords, atom_types, mm_coords, mm_Q, mm_type]
  # Optional width overrides:
  descriptor:   { n_types: 4, neuron: [25,50,100], axis_neuron: 4 }
  esp:          { n_types: 4, neuron: [5,10,20],   axis_neuron: 4 }
  fitting_net:  { n_types: 4, neuron: [240,240,240] }

# -----------------------------
# LOSSES  (pick MLP or ΔMLP block)
# -----------------------------

# ---- ΔMLP (differences) ----
losses:
  energy:        { type: mse, weight: 1.0 }   # ΔE
  qm_dgrad:      { type: mse, weight: 1.0 }   # Δ(∂E/∂R)_QM
  mm_espgrad_d:  { type: mse, weight: 10.0 }  # (∂ΔE/∂R)/q

# ---- MLP (high-level) ----
# losses:
#   energy:          { type: mse, weight: 1.0 }   # E_high (demeaned)
#   qm_grad_high:    { type: mse, weight: 1.0 }   # QM ∂E/∂R (high)
#   mm_espgrad_high: { type: mse, weight: 10.0 }  # (∂E/∂R)/q (high)

# -----------------------------
# LOSS WEIGHTING
# -----------------------------
weighting:
  uncertainty: false                         # learned σ per key (uncertainty MTL). True=enable

weight_schedule:                              # LR-ratio group weighting (active in train/eval)
  energy_keys: ["energy","dE"]                # which keys belong to the “energy” group
  grad_keys:   ["qm_*","mm_*"]                # which keys belong to the “gradient” group
  w_ene: 1.0                                  # base energy weight
  w_grad_max: 100.0                           # max gradient weight near start_lr

# -----------------------------
# OPTIMIZER & LR SCHEDULE
# -----------------------------
optim:
  lr: 5e-4                                    # Adam LR
  weight_decay: 0.0
  schedule: exponential                       # active scheduler: plateau | exponential | mixed | onecycle

  # ---- (1) Plateau (driven by val_unweighted) ----
  plateau:
    factor: 0.5
    patience: 10
    threshold: 1.0e-4
    min_lr: 1.0e-6

  # ---- (2) Exponential ----
  exponential:
    factor: 0.99                               # multiplicative per-epoch decay
    min_lr: 1.0e-6

  # ---- (3) Mixed (Exp + Plateau fallback) ----
  mixed:
    exp_factor: 0.99
    min_lr: 1.0e-6
    patience: 8                                 # bad epochs before fallback
    threshold: 1.0e-4                           # improvement threshold
    plat_factor: 0.5                             # LR × factor on fallback
    cooldown: 2                                  # cooldown epochs after fallback

  # ---- (4) OneCycle ----
  onecycle:
    max_lr: 2.5e-3                              # ~5× base LR
    pct_start: 0.3
    div_factor: 25.0
    final_div_factor: 1.0e4
    anneal: cos                                  # cos | linear

# -----------------------------
# TRAINER / RUNTIME
# -----------------------------
trainer:
  device: cuda                                 # or 'cpu'
  batch_size: 32
  epochs: 500
  val_fraction: 0.2
  test_fraction: 0.1
  seed: 42
  grad_clip: 5.0                                # max global grad-norm
  mm_equal_frame_weight: true                   # per-frame equalization for mm_* losses
  rot_aug: false                                # random rotations on inputs (train)
  center: false                                 # center on QM centroid (train/eval)
  amp: off                                      # off | bf16 | fp16  (CUDA only)
  dataparallel: false                           # torch.nn.DataParallel

  print_units: kcal                             # console print units (kcal or eV)
  print_detail: short                           # short (RMSE) | full (also means & σ)
  print_keys: ["energy","qm_dgrad","mm_espgrad_d"]  # visible keys (adjust for MLP)

  out_dir: runs/full
  save_path: runs/full/ckpt.pt

  # Early stopping (driven by UNWEIGHTED validation loss)
  early_stop_patience: 100
  early_stop_min_delta: 1.0e-4

  # Goal-based stopping (optional)
  # goal_stop:
  #   enabled: true
  #   check_every: 1
  #   patience: 2
  #   units: kcal
  #   thresholds:
  #     energy: 1              # kcal/mol
  #     qm_dgrad: 1           # kcal/mol/Å

  # Logging / exports
  save_progress: true                           # mirror console to progress.txt
  progress_txt: runs/full/progress.txt
  save_script_every: 50                         # TorchScript snapshot cadence (0=off)
  val_rot_eval: false                           # also evaluate val with random rotations
  test_rot_eval: false                          # likewise for test

  # Threading (optional)
  torch_num_threads: 4
  torch_interop_threads: 1

# -----------------------------
# MASKS
# -----------------------------
masks:
  mm_*: mm_type                                # mask out padded MM (mm_type==0)
  # other examples:
  # qm_grad*: qm_mask
  # mm_espgrad*: mm_mask
```

---

<div style="page-break-before: always;"></div>

## Test Models

The tester consumes a **trained run folder** and produces plots/metrics and bridge metadata.

```bash
qmmm-test runs/mlp_inter \    # path to trained run folder where artifacts are saved during training
  --device cuda \             
  --batch_size 32 \           # inference batch size (optional)
  --vector_mode magnitude \   # magnitude | component for vector flattening
  --units kcal \              # ev | kcal (derived units respected)
  --max_points 200000 \       # subsample for plots (optionla)
  --save_data_dir runs/mlp_inter/test_csvs \  # optional save path for per-key CSVs
  --debug                     # optional debug prints on model load and inference
```

**Expected inputs in `out_dir`:**

- `model_script.pt` — TorchScript (from training)
- `model_state.pt` — raw `state_dict()`
- `model_io.json` — input keys, loss keys, masks, species, amp
- `test_data.npz` — held‑out subset (written by training)
- `logs/train_val_metrics.csv` — curves (if present, plotted)

**What it does:**

- Loads `model_script.pt` and builds batched inputs from `test_data.npz` using `model_io.json.input_keys`.
- Inference produces predictions for keys listed in `model_io.json.loss_keys`.
- If `mm_Q` is available, **derives `mm_grad_*` from `mm_espgrad_*`** by multiplying by `q`, so both are comparable.
- Applies masks using `model_io.json.masks` (supports wildcards). Defaults to `mm_*` masked by `mm_type`.
- Converts units (energies, grads, ESP-grad per charge) to **eV** or **kcal/mol** families for metrics/plots.
- Flattens vectors by **magnitude** or **components** for MAE/RMSE/R².
- Dumps dense hexbin and scatter plots, a metrics CSV, and optional per‑key CSVs of true/pred pairs.
- Writes `bridge_meta.txt` with exactly **four lines** for the C/Fortran bridge:
  ```
  n_qm <int>
  max_mm <int>
  n_types <int>
  species_z_ordered <Z1> <Z2> ...
  ```

**Outputs:**

- `plots_test/*.png` — per‑key density/scatter plots
- `test_metrics.csv` — MAE/RMSE/R² summary
- `predictions_test.npz` — flattened (masked/converted) true/pred arrays
- `model_stats.json` — parameter counts + artifact sizes
- `bridge_meta.txt` — 4‑line bridge meta


---

<div style="page-break-before: always;"></div>


## Artifacts Written by Training

Inside `out_dir` (e.g., `runs/mlp_inter`):

- `best.pt` — best checkpoint (when unweighted validation improves)
- `model_state.pt` — `state_dict()`
- `model_script.pt` — TorchScript (if scripting succeeds)
- `model_io.json` — IO meta (input keys, loss keys, masks, species, amp)
- `test_data.npz` — held‑out subset used by tester
- `logs/train_val_metrics.csv` — training/validation curves + weights
- `training_summary.json` — run summary (host env, timings, best metrics, sizes)
- `progress.txt` — mirrored console log (if enabled)
- `plots/` — optional training curve plots (generated by tester from CSV)

---

## Notes & Tips

- **Defaults & widths**  
  If you omit model sub‑blocks, defaults are used:  
  `descriptor.neuron=[25,50,100], axis_neuron=4; esp.neuron=[5,10,20], axis_neuron=4; fitting_net.neuron=[240,240,240]; n_types=6`.

- **MM masking**  
  `mm_type>0` ⇒ real; padded rows are `0` and ignored by losses/metrics. Keep `masks: { mm_*: mm_type }` unless you have a custom mask.

- **Schedulers**  
  You may keep all scheduler blocks present; **only** the one named in `optim.schedule` is active. Others are ignored.

- **Outliers**  
  `energy_outlier` and `grad_outlier` act during parsing. `grad_outlier_keys` are *only* used to compute outlier scores; they don’t change which loss keys you train unless you also configure `losses` accordingly.

- **Units**  
  Training computes losses in **eV**; console/test printing can be switched to **kcal** via `trainer.print_units: kcal` and `qmmm-test --units kcal`.

- **Re‑using parsed folders**  
  The parser reads only frame text files; any `parse_summary.*` you wrote are ignored next time and get **overwritten** if paths are identical.
