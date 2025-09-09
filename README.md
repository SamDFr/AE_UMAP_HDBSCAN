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
- **Step 1 – AE training**: Train an autoencoder on descriptors; save **standard scaler**, **model checkpoint**, **loss logs**, **training plot**, and **hyperparameter record**.
- **Step 2 – Defect detection**: Identify outliers/defects from AE **reconstruction errors**; write filtered `.npy` sets and optional `.xyz`.
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
Train a **PyTorch** Autoencoder (AE) on SOAP descriptors; perform (optional) **Optuna** hyperparameter search; save the fitted **standard scaler**, AE **model checkpoint**, **loss logs**, a **training plot**, and a **hyperparameter record**.

**Inputs & Assumptions**
- Descriptor files (`*.npy`) located under: `./run/data/dataset/AE_training/`
  - The script automatically picks the **first** `.npy` file it finds there.
- No CLI flags: **edit parameters inside the script** (latent size, batch size, epochs, learning rate, etc.).
- The script sets random seeds for reproducibility (`numpy`, `random`, `torch`).

**What the script does**
1. Locates the first descriptor file under `./run/data/dataset/AE_training/` and loads it.
2. Splits the data into **train/val/test** (default ratios in the script).
3. Runs an **Optuna** search (`run_optuna_search`) to suggest best hyperparameters  
   *(latent ratio, batch size, learning rate, num epochs)*, **then trains** the AE with those settings.  
   If you disable/comment the search, the script uses the **fallback defaults** defined inside it.
4. Standardizes inputs with `StandardScaler` and saves the scaler.
5. Trains the AE (MSE loss, Adam) and evaluates on the test set.
6. Saves artifacts and a clean hyperparameter log.

**Outputs (paths are defined in the script)**
- `./training/ae/standard_scaler.pkl` — fitted `StandardScaler` (joblib)
- `./training/ae/autoencoder_model.pth` — serialized AE model (via `torch.save`)
- `./training/ae/losses.csv` — per-epoch training/validation losses (CSV)
- `./training/ae/training_errors.pdf` — loss curves (PDF)
- `./training/ae/hyperparameters.txt` — run settings (input_dim, latent_dim, batch_size, num_epochs, lr, etc.)

**Usage**
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
# Single threshold (e.g., 5.0)
python 02_defect_detection.py 5.0

# Multiple thresholds at once
python 02_defect_detection.py 0,1.5,3,5
```

## 2bis) Merge per-run arrays — `02bis_merge_npy.py`

**Purpose**  
Aggregate many per-run defect outputs into consolidated `full_data` sets **per Y** (a value parsed from filenames like `detected_defects_AE_<Y>_desc.npy`), and also produce **global** merges across all runs.

**Inputs & Assumptions**
- Base directory (default: `./run/data`) that contains a `dataset/` folder with multiple run subfolders.
- In each run subfolder, the script expects files such as:
  - `desc/detected_defects_AE_<Y>_desc.npy` (defect descriptors)
  - `recon_err/detected_defects_AE_<Y>.npy` (indices + reconstruction errors)
- The **Y** value is extracted from filenames (the integer after `AE_` and before `_desc.npy`).

**Command-line Arguments**
The script uses `argparse` with:
- `--base_dir` (default: `./run/data`) → path containing the `dataset/` directory.
- `--pattern` (**required**) → prefix identifying run directories to process  
  *(e.g., `ni10cr20V_NiFeCr_stoller_100k_bx98_80kev_test1_rnd_` — the trailing run id is appended to this)*.
- `--y` (optional, one or more integers) → **only** merge these Y values; if omitted, the script merges **all detected** Ys.

**What the script does**
1. Enumerates run directories under `./run/data/dataset/` whose names start with `--pattern`.
2. For each such run directory:
   - Creates or **cleans** a local `full_data/` folder to avoid mixing results.
   - Discovers all available **Y** values from `desc/detected_defects_AE_<Y>_desc.npy`  
     (or uses the list given via `--y`).
   - For each Y:
     - Loads `desc/detected_defects_AE_<Y>_desc.npy` and `recon_err/detected_defects_AE_<Y>.npy`.
     - Concatenates into a single array and saves `full_data/<RUNID>_<Y>_full_data.npy`.
3. After all runs are processed, creates **global merged arrays** per Y at the dataset level.

**Outputs**
- Per run: `full_data/<RUNID>_<Y>_full_data.npy`
- Global (under `./run/data/dataset/`): merged arrays per Y across all matching runs.

**Usage**
```bash
# Merge all available Y values for all runs matching the pattern
python 02bis_merge_npy.py \
  --base_dir ./run/data \
  --pattern ni10cr20V_NiFeCr_stoller_100k_bx98_80kev_test1_rnd_

# Merge only selected Y values (e.g., 10, 20, 40)
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
Map **UMAP/HDBSCAN** cluster labels back to original trajectories and export **cluster-specific XYZ** files for visualization (OVITO/VMD). Also computes per-cluster composition tables and generates diagnostic plots (e.g., histograms/heatmaps of reconstruction errors or sizes).

**Inputs & Assumptions**
- Per-run subfolders under your dataset root, each containing:
  - Original trajectory (e.g., first `*.gz` or other OVITO-readable file).
  - Results from step 3, e.g. `umap_with_clusters.npy` (embedding + labels).
  - (Optionally) arrays with reconstruction errors or indices linking descriptors ↔ atoms.
- Paths, cluster selections, and plotting toggles are **set inside the script** — there is **no CLI**.

**What the script does**
1. Iterates over run directories and loads:
   - The UMAP/HDBSCAN outputs (labels per atom/sample).
   - The corresponding trajectory (to reconstruct atomic subsets).
2. For each **cluster label** found (including or excluding `-1` noise depending on your settings):
   - Gathers atom indices mapped from your pipeline (descriptor/AE indices → atom IDs).
   - Writes **cluster-specific XYZ** files (e.g., `exported_clusters/cluster_<label>.xyz`).
3. Aggregates **per-cluster composition** (element/species percentages) and saves CSV/NPY tables.
4. Generates **diagnostic figures** (PDF/PNG), e.g.:
   - Cluster size distributions
   - Log-scaled histograms / heatmaps of reconstruction error by cluster
   - Bar plots of per-cluster composition

**Outputs**
- `exported_clusters/cluster_<label>.xyz` per cluster per run
- `atom_perc_cluster/*.csv` or `*.npy` with per-cluster composition summaries
- Diagnostic plots (PDF/PNG) under `atom_perc_cluster/` or a run-specific figures directory
- (Optional) a consolidated index → cluster mapping saved as `.npy` for reuse

**Usage**
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

Python ≥ 3.8 is recommended.

Core dependencies:

- **Numerics & Utils**: `numpy`, `pandas`, `joblib`, `glob`, `argparse`
- **ML**: `torch` (PyTorch), `scikit-learn`, `umap-learn`, `hdbscan`, `optuna`
- **Descriptors & IO**: `dscribe`, `ase`, `ovito` (incl. `ovito.io.ase`)
- **Plotting**: `matplotlib`, `seaborn`
