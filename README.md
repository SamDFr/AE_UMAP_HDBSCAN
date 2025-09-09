# AE_UMAP_HDBSCAN

Pipeline for **unsupervised defect detection and clustering** in atomistic simulations using:

- **SOAP descriptors** (DScribe)  
- **Autoencoders (AE)** for outlier/defect detection  
- **UMAP** for nonlinear dimensionality reduction  
- **HDBSCAN** for density-based clustering  
- Optional **XYZ export** of subsets/groups for inspection and post-processing

This repository targets large simulation outputs (up to millions of atoms) and is organized as a step-by-step workflow from descriptor generation to cluster-aware structure extraction.

---

## ✨ Features

- **Step 0 – SOAP**: Generate per-atom SOAP descriptors from trajectory/structure files.
- **Step 1 – AE training**: Train an autoencoder on descriptors; save model + latent representations + reconstruction errors.
- **Step 2 – Defect detection**: Identify outliers/defects from AE outputs; write filtered `.npy` sets and optional `.xyz`.
- **Step 2bis – Merge**: Batch-merge many per-run `.npy` files into consolidated arrays.
- **Step 3 – UMAP + HDBSCAN**: Embed and cluster defective atoms; save embeddings, labels, and plots.
- **Step 4 – XYZ export**: Write XYZ files per cluster / group for visual analysis (e.g., OVITO, VMD).

---

## 📦 Repository Structure (main scripts)

```
AE_UMAP_HDBSCAN/
├── 0_Dscribe_SOAP_gen.py              # SOAP descriptor generation from structure/trajectory
├── 01_autoencoder_training.py         # Train the autoencoder + export latent/recon error
├── 02_defect_detection.py             # Detect defects/outliers using AE outputs
├── 02bis_merge_npy.py                 # Merge many .npy chunks into full_data sets
├── 03_UMAP_HDBSCAN.py                 # UMAP embedding + HDBSCAN clustering + plots
└── 04_xyz_gen_from_UMAP_HDBSCAN.py    # Export XYZ per cluster / selection + diagnostics
```

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
