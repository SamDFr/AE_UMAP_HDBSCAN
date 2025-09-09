# AE_UMAP_HDBSCAN

Pipeline for **unsupervised defect detection and clustering** in atomistic simulations using:

- **SOAP descriptors** (DScribe)  
- **Autoencoders (AE)** for outlier/defect detection  
- **UMAP** for nonlinear dimensionality reduction  
- **HDBSCAN** for density-based clustering  
- Optional **XYZ export** of subsets/groups for inspection and post-processing

This repository targets large simulation outputs (up to millions of atoms) and is organized as a step-by-step workflow from descriptor generation to cluster-aware structure extraction.

---

## Features

- **Step 0 – SOAP**: Generate per-atom SOAP descriptors from trajectory/structure files.
- **Step 1 – AE training**: Train an autoencoder on descriptors; save model + latent representations + reconstruction errors.
- **Step 2 – Defect detection**: Identify outliers/defects from AE outputs; write filtered `.npy` sets and optional `.xyz`.
- **Step 2bis – Merge**: Batch-merge many per-run `.npy` files into consolidated arrays.
- **Step 3 – UMAP + HDBSCAN**: Embed and cluster defective atoms; save embeddings, labels, and plots.
- **Step 4 – XYZ export**: Write XYZ files per cluster / group for visual analysis (e.g., OVITO, VMD).

---

## Repository Structure (main scripts)

```
AE_UMAP_HDBSCAN/
├── 0_Dscribe_SOAP_gen.py              # SOAP descriptor generation from structure/trajectory
├── 01_autoencoder_training.py         # Train the autoencoder + export latent/recon error
├── 02_defect_detection.py             # Detect defects/outliers using AE outputs
├── 02bis_merge_npy.py                 # Merge many .npy chunks into full_data sets
├── 03_UMAP_HDBSCAN.py                 # UMAP embedding + HDBSCAN clustering + plots
└── 04_xyz_gen_from_UMAP_HDBSCAN.py    # Export XYZ per cluster / selection + diagnostics
```

## 0) SOAP descriptor generation — `0_Dscribe_SOAP_gen.py`

**Purpose**  
Compute per-atom **SOAP** descriptors (DScribe) from OVITO-readable structures/trajectories, and log the descriptor hyperparameters.

**Inputs & Assumptions**
- One or more atomistic files readable by **OVITO** (e.g., `.gz`, `.xyz`, `.lammpstrj`).
- Species/type map consistent with your simulations.
- SOAP hyperparameters set in the script (e.g., `r_cut`, `n_max`, `l_max`, `sigma`).

**What the script does**
1. Loads trajectory/frames via `ovito.io.import_file(...)`.
2. Converts frames to **ASE** atoms (`ovito_to_ase`) for descriptor calculation.
3. Builds a DScribe `SOAP` object with the chosen hyperparameters.
4. Computes a dense descriptor matrix per atom and (optionally) per frame.
5. Saves descriptors to `.npy` and writes a `params_*.txt` containing the final feature length and SOAP settings.

**Outputs**
- `SOAP_<run_id>_<...>.npy` — float array of shape `[n_atoms, n_features]` (or concatenated across frames).
- `params_<run_id>.txt` — the SOAP parameter summary.

**Usage (example)**
```bash
python 0_Dscribe_SOAP_gen.py <filename>
```


## 1) Autoencoder training — `01_autoencoder_training.py`

**Purpose**  
Train a **PyTorch** Autoencoder (AE) on SOAP descriptors; export the model, latent codes, reconstruction errors, and a reproducibility log.

**Inputs & Assumptions**
- Descriptor files (`*.npy`) located in the training folder (e.g., `./run/data/dataset/AE_training/`).
- AE architecture/hyperparameters configured in the script (layers, latent size, batch size, epochs, learning rate).
- Random seeds are fixed for reproducibility.

**What the script does**
1. Loads descriptor arrays and builds `DataLoader`s for train/validation.
2. Defines an AE: encoder → latent → decoder (fully connected).
3. Trains with MSE reconstruction loss; logs/plots losses over epochs.
4. After training:
   - Saves **model weights/checkpoints**.
   - Computes and saves **latent embeddings** for all samples.
   - Computes and saves **reconstruction errors** (per sample).
   - Writes `training/ae/hyperparameters.txt` with all key settings and paths.

**Outputs**
- `training/ae/checkpoints/*.pt` (or similar) — trained weights.
- `training/ae/latent_*.npy` — latent space representation.
- `training/ae/recon_error_*.npy` — per-sample reconstruction errors.
- `training/ae/loss_curve.pdf` — training/validation loss figure.
- `training/ae/hyperparameters.txt` — full run log (existing file is overwritten to avoid appends).

**Usage (example)**
```bash
python 01_autoencoder_training.py 
```

## 2) Defect detection via AE errors — `02_defect_detection.py`

**Purpose**  
Identify **defective atoms** as **outliers** based on AE reconstruction error, and export their indices, descriptors, and optional XYZ subsets.

**Inputs & Assumptions**
- Directory tree under `./run/data/dataset/` with per-run subfolders.
- Each subfolder contains:
  - The original trajectory (e.g., first `*.gz` found).
  - AE outputs: reconstruction errors and descriptors.
- A **threshold** on reconstruction error (script argument or constant) to flag defects.

**What the script does**
1. Iterates over subdirectories matching a pattern (HPC-friendly).
2. For each run:
   - Loads the first trajectory (`*.gz`) and AE arrays.
   - Selects atoms with `recon_error >= threshold`.
   - Saves:
     - `detected_defects_AE_<thr>_.npy` — `[index, recon_error]` pairs.
     - `detected_defects_AE_<thr>_desc.npy` — descriptors for selected atoms (skipped if `thr == 0`).
     - `detected_defects_AE_<thr>_.xyz` — optional XYZ export of the flagged atoms.

**Outputs (per run)**
- `desc/detected_defects_AE_<thr>_desc.npy`
- `recon_err/detected_defects_AE_<thr>_.npy`
- `xyz/detected_defects_AE_<thr>_.xyz` (optional)

**Usage (example)**
```bash
python 02_defect_detection.py \
  --base_dir ./run/data/dataset \
  --pattern "ni10cr20V_.*" \
  --threshold 5.0
```

## 2bis) Merge per-run arrays — `02bis_merge_npy.py`

**Purpose**  
Aggregate many per-run defect arrays into consolidated `full_data` sets **per Y** (a value parsed from filenames), and create **global** merges across runs.

**Inputs & Assumptions**
- A base dataset directory (e.g., `./run/data/dataset/`) containing many run subfolders.
- In each run subfolder, `desc/detected_defects_AE_*_desc.npy` exist, with a name that encodes **Y**.
- A `pattern`/`pattern_prefix` that identifies the run subfolders and lets the script extract `XXX` identifiers.

**What the script does**
1. For each run directory matching `--pattern`:
   - Creates or **cleans** `full_data/`.
   - Scans `desc/` to discover all available **Y** values (or use `--y` to select).
   - For each Y:
     - Loads the matching **defect descriptors** and **recon errors**.
     - Concatenates them and saves `full_data/XXX_Y_full_data.npy`.
2. After all runs are processed:
   - Builds **global merged** arrays per Y at the dataset level.

**Outputs**
- Per run: `full_data/XXX_Y_full_data.npy`
- Global (under dataset root): merged arrays for each Y across all runs.

**Usage (example)**
```bash
python 02bis_merge_npy.py \
  --base_dir ./run/data \
  --pattern ni10cr20V_NiFeCr_stoller_100k_bx98_80kev_test1_rnd_ \
  --y 10 20 40
  ```

## 3) UMAP + HDBSCAN — `03_UMAP_HDBSCAN.py`

**Purpose**  
Embed merged/filtered descriptors with **UMAP**, then **cluster** in the embedded space with **HDBSCAN**. Save embedding, labels, and publication-ready plots.

**Inputs & Assumptions**
- One or more input arrays of shape `[n_samples, n_features]` (e.g., merged outputs).
- The script standardizes features via `StandardScaler` before UMAP.
- UMAP/HDBSCAN hyperparameters are set in the script (or via arguments, if exposed).

**What the script does**
1. Loads descriptors and standardizes them.
2. Fits **UMAP** to produce a 2D embedding (configurable `n_components`).
3. Runs **HDBSCAN** on the UMAP space → cluster labels (`-1` = noise).
4. Computes **silhouette score** (on a sample if needed).
5. Saves:
   - `umap_with_clusters.npy` with columns like `[UMAP_x, UMAP_y, last3_original_features, label]`.
   - `umap_clusters.pdf` (and/or `.png`) — colored scatter with legend and metrics.

**Outputs**
- `<out_dir>/umap_with_clusters.npy`
- `<out_dir>/umap_clusters.pdf` (optionally `.png`)

**Usage (example)**
```bash
python 03_UMAP_HDBSCAN.py 
```

## 4) Cluster-aware XYZ export & diagnostics — `04_xyz_gen_from_UMAP_HDBSCAN.py`

**Purpose**  
Map **UMAP/HDBSCAN** group assignments back to the original trajectories and export **cluster-specific XYZ** files. Produce per-cluster composition tables and diagnostic plots.

**Inputs & Assumptions**
- Per-run subfolders containing:
  - Original trajectories (`*.gz`).
  - UMAP/HDBSCAN results (`umap_with_clusters.npy` or similar).
- Indices saved during earlier steps let the script map clusters → atoms.

**What the script does**
1. For each run and each cluster label:
   - Gathers the corresponding atom indices.
   - Writes **`cluster_<label>.xyz`** (or similar) under an export folder.
2. Computes **per-cluster atom fractions** (per element/species) and saves tables.
3. Generates **diagnostic plots** (e.g., log-scaled histograms/heatmaps of reconstruction errors).

**Outputs**
- Per cluster per run: `exported_clusters/cluster_<id>.xyz`
- CSV/NPY summaries of per-cluster composition.
- Diagnostic PDFs (e.g., `atom_perc_cluster/*.pdf`).

**Usage (example)**
```bash
python 04_xyz_gen_from_UMAP_HDBSCAN.py 
```

## 🔗 End-to-end Workflow

1. **SOAP generation** → `0_Dscribe_SOAP_gen.py` → `SOAP_*.npy`, `params_*.txt`  
2. **AE training** → `01_autoencoder_training.py` → AE weights, **latent**, **recon_error**  
3. **Defect detection** → `02_defect_detection.py` → indices, **defect descriptors**, optional **XYZ**  
4. **Merging** → `02bis_merge_npy.py` → `full_data/XXX_Y_full_data.npy` + **global merges**  
5. **UMAP + HDBSCAN** → `03_UMAP_HDBSCAN.py` → `umap_with_clusters.npy`, **plots**  
6. **XYZ export & diagnostics** → `04_xyz_gen_from_UMAP_HDBSCAN.py` → cluster XYZ + **stats/figures**

---

## 📋 Requirements

Python ≥ 3.9 is recommended.

Core dependencies (based on the imports across scripts):

- **Numerics & Utils**: `numpy`, `pandas`, `joblib`, `glob`, `argparse`
- **ML**: `torch` (PyTorch), `scikit-learn`, `umap-learn`, `hdbscan`
- **Descriptors & IO**: `dscribe`, `ase`, `ovito` (incl. `ovito.io.ase`)
- **Plotting**: `matplotlib`, `seaborn`

### Install (conda + pip example)

```bash
# Create env
conda create -n ae_umap_hdbscan python=3.10 -y
conda activate ae_umap_hdbscan

# Core scientific stack
pip install numpy pandas joblib scikit-learn matplotlib seaborn

# ML
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or your CUDA build

# Manifold + clustering
pip install umap-learn hdbscan

# Atomistic IO & descriptors
pip install ase ovito dscribe
```

---

## 🚀 Quick Start

### 0) Generate SOAP descriptors

```bash
python 0_Dscribe_SOAP_gen.py
```

### 1) Train the Autoencoder

```bash
python 01_autoencoder_training.py
```

### 2) Detect Defects / Outliers

```bash
python 02_defect_detection.py --base_dir /path/to/runs_root --pattern "ni10cr20V_.*" --threshold 5.0
```

### 2bis) Merge many `.npy` chunks

```bash
python 02bis_merge_npy.py --base_dir /path/to/runs_root --pattern_prefix "bx98_" --selected_ys 10 20 40
```

### 3) UMAP + HDBSCAN

```bash
python 03_UMAP_HDBSCAN.py
```

### 4) Export XYZ from UMAP/HDBSCAN Selections

```bash
python 04_xyz_gen_from_UMAP_HDBSCAN.py
```

---
