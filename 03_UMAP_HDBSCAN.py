# %%
import data_loading.load_data as load_data
import torch
import numpy as np
import matplotlib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from matplotlib.colors import ListedColormap, BoundaryNorm
import os
import joblib
import random
from sklearn.cluster import  HDBSCAN
from sklearn.metrics import silhouette_score
import glob 
from umap import UMAP
import sys


# %%
# This is a UMAP + HDBSCAN on the data

# %%
print(torch.__config__.parallel_info())

# %%
# Fixer la seed pour la reproductibilité
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# check whether GPU is available, otherwise use CPU 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# %%
matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14)
matplotlib.use('Agg')  # Définir le backend en mode headless
mplstyle.use('fast') #The fast style set simplification and chunking parameters to reasonable settings to speed up plotting large amounts of data

# %%
#Training descriptor matrix

# Define the base data directory
curr_dir = os.getcwd()
data_dir = os.path.join(curr_dir, '../run/data/dataset')

if not data_dir:
    raise FileNotFoundError("No '../run/data/dataset' found.")

# Automatically list all subdirectories in the data directory
subdirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

if not subdirs:
    raise FileNotFoundError("No subdirectories found in the '/run/data/dataset' directory.")

# Process each subdirectory
for subdir in subdirs:
    if not os.path.basename(subdir).startswith('recon'):
        print(f"Skipping directory: {subdir} (does not start with 'recon')")
        continue
    
    print(f"Processing directory: {subdir}")

     #Check for other .npy file to avoid confusion 
    output_file_umap_dir = os.path.join(curr_dir, 'UMAP_HDBSCAN')
    if not os.path.exists(output_file_umap_dir):
        os.makedirs(output_file_umap_dir)
        print(f"Created directory: {output_file_umap_dir}")

    output_file_umap = os.path.join(output_file_umap_dir, 'umap_with_clusters.npy')
    if os.path.exists(output_file_umap):
        os.remove(output_file_umap)
        print(f"{output_file_umap} already exists and has been deleted.")

    # Find the descriptor file (first file ending with '.npy') in the current subdirectory
    npy_files = glob.glob(os.path.join(subdir, "*.npy"))
    if npy_files:
        desc_file_defect = npy_files[0]  # selects the first found .npy file
        print("Selected desc file:", desc_file_defect)
    else:
        raise FileNotFoundError(f"No .npy file found in directory {subdir}")

    descriptor = load_data.load_data_single(desc_file_defect) # the last colomns is a label (index of the cascade)

    norma = True
    if norma:
        # Standardize the descriptors
        scaler = StandardScaler()
        
        # Instead of using torch.tensor(), use clone().detach() to copy the existing tensor
        descriptor_no_scale = descriptor[:, :-3].clone().detach().to(torch.float32)
        
        # Convert the tensor to a numpy array (if it's on GPU, move it to CPU first)
        descriptor_np = descriptor_no_scale.cpu().numpy()
        
        # Fit the scaler and transform the data
        descriptor_scaled = scaler.fit_transform(descriptor_np)
        
        # Convert the scaled numpy array back to a PyTorch tensor
        descriptor = torch.tensor(descriptor_scaled, dtype=torch.float32)
        
        # Save the fitted scaler to a file
        scaler_file = output_file_umap_dir+'/standard_scaler.pkl'
        if os.path.exists(scaler_file):
            os.remove(scaler_file)
            print(f"{scaler_file} already exists and has been deleted.")
        joblib.dump(scaler, scaler_file)
        print(f"Scaler saved to {scaler_file}")
    else:
        descriptor_no_scale = descriptor[:, :-3].clone().detach().to(torch.float32)
        descriptor_scaled = descriptor_no_scale.cpu().numpy()
        descriptor = torch.tensor(descriptor_scaled, dtype=torch.float32)

    # Hyperparameters
    input_dim = descriptor.shape[1] # input dimension for the Autoencoder 
    # Parse command-line arguments for latent_dim
    latent_dim = 10  # Default value
    for arg in sys.argv:
        if arg.startswith("latent_dim="):
            latent_dim = int(arg.split("=")[1])
            break
  

    print()
    print(f'the number of UMAP components: {latent_dim}') 
    print()

# %%
# Convert the descriptor tensor to a numpy array if it's not already
descriptor_np = descriptor.cpu().numpy() if hasattr(descriptor, "cpu") else descriptor.numpy()

# Perform UMAP decomposition to reduce dimensions to 2

# Parse command-line arguments for n_neighbors
n_neighbors = 15  # Default value
for arg in sys.argv:
    if arg.startswith("n_neighbors="):
        min_cluster_size = int(arg.split("=")[1])
        break

umap = UMAP(n_neighbors=n_neighbors, 
            n_components=latent_dim, 
            random_state=seed, 
            min_dist=0,
            n_jobs=-1)
descriptor_umap = umap.fit_transform(descriptor_np)

# save the UMAP model to a file

save_UMAP_model = False
if save_UMAP_model:
    umap_file = output_file_umap_dir + f'/umap_model_{latent_dim}.pkl'
    if os.path.exists(umap_file):
        os.remove(umap_file)
        print(f"{umap_file} already exists and has been deleted.")

    joblib.dump(umap, umap_file)
    print(f"UMAP model saved to {umap_file}")

# Plot the UMAP results (without clustering labels)
plot_without_labels = False
if plot_without_labels:
    plt.figure(figsize=(8, 6))
    plt.scatter(descriptor_umap[:, 0], descriptor_umap[:, 1], s=10, alpha=0.7, cmap='viridis')
    plt.xlabel('UMAP Component 1', fontsize=14)
    plt.ylabel('UMAP Component 2', fontsize=14)
    #plt.title('UMAP: Component 1 vs Component 2', fontsize=16)
    plt.grid(True)

# Perform HDBSCAN clustering on the UMAP representations

# Parse command-line arguments for min_cluster_size
min_cluster_size = 10  # Default value
for arg in sys.argv:
    if arg.startswith("min_cluster_size="):
        min_cluster_size = int(arg.split("=")[1])
        break

print(f"Performing HDBSCAN clustering with min_cluster_size={min_cluster_size}...")
clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method='eom')
umap_clusters = clusterer.fit_predict(descriptor_umap)

# --- Save HDBSCAN model (with flag) ---
save_HDBSCAN_model = True
if save_HDBSCAN_model:
    hdbscan_file = os.path.join(output_file_umap_dir, f"hdbscan_model_{latent_dim}_mcs{min_cluster_size}.pkl")
    if os.path.exists(hdbscan_file):
        os.remove(hdbscan_file)
        print(f"{hdbscan_file} already exists and has been deleted.")

    joblib.dump(clusterer, hdbscan_file)
    print(f"HDBSCAN model saved to {hdbscan_file}")

total_points = len(umap_clusters)
for label in np.unique(umap_clusters):
    percentage = (umap_clusters == label).sum() / total_points * 100
    print(f"Cluster {label}: {percentage:.5f}% of the data")

# Remap cluster labels to a continuous range for a discrete colormap
unique_clusters = np.unique(umap_clusters)  # e.g., array([-1, 0, 1, 2, ...])
n_clusters = len(unique_clusters)
label_mapping = {label: idx for idx, label in enumerate(unique_clusters)}
mapped_clusters = np.array([label_mapping[label] for label in umap_clusters])

# Create a discrete colormap based on 'viridis'
base_cmap = plt.get_cmap('viridis', n_clusters)
discrete_cmap = ListedColormap(base_cmap(np.linspace(0, 1, n_clusters)))
norm = BoundaryNorm(np.arange(n_clusters + 1) - 0.5, n_clusters)

# Plot the clustering results with the discrete colormap

# Plot the clustering results with the discrete colormap
# Sort clusters by size (smallest to largest)
cluster_sizes = {label: (umap_clusters == label).sum() for label in unique_clusters}
sorted_clusters = sorted(unique_clusters, key=lambda label: cluster_sizes[label])

# Create a sorted index based on cluster sizes
sorted_indices = np.argsort([cluster_sizes[label] for label in umap_clusters])

# Reorder the UMAP data and cluster labels based on the sorted indices
descriptor_umap_sorted = descriptor_umap[sorted_indices]
mapped_clusters_sorted = mapped_clusters[sorted_indices]

# Plot the reordered data
plt.figure(figsize=(8, 6))
scatter = plt.scatter(descriptor_umap_sorted[:, 0], descriptor_umap_sorted[:, 1],
                      c=mapped_clusters_sorted, cmap=discrete_cmap, norm=norm,
                      s=10, alpha=0.7)

cbar = plt.colorbar(scatter, ticks=np.arange(n_clusters))
cbar.set_label('Cluster label', fontsize=14)
cbar.ax.set_yticklabels([str(label) for label in unique_clusters])
plt.xlabel('UMAP Component 1', fontsize=14)
plt.ylabel('UMAP Component 2', fontsize=14)
plt.title('HDBSCAN Clustering on UMAP Representations', fontsize=16)
plt.grid(True)

# Save the plot to a file
plot_pdf = False    
if plot_pdf:
    plot_file_umap = os.path.join(output_file_umap_dir, 'umap_clusters.pdf')
    if os.path.exists(plot_file_umap):
        os.remove(plot_file_umap)
        print(f"{plot_file_umap} already exists and has been deleted.")
    plt.savefig(plot_file_umap, format='pdf', bbox_inches='tight')

plot_png = False
if plot_png:
    plot_file_umap_png = os.path.join(output_file_umap_dir, 'umap_clusters.png')
    if os.path.exists(plot_file_umap_png):
        os.remove(plot_file_umap_png)
        print(f"{plot_file_umap_png} already exists and has been deleted.")
    plt.savefig(plot_file_umap_png, format='png', bbox_inches='tight')

# Calculate the silhouette score for the clustering
silhouette_avg = silhouette_score(descriptor_umap, umap_clusters, sample_size=len(descriptor)//10, random_state=seed)
print(f"Silhouette Score for UMAP-based clustering: {silhouette_avg}")

# Ensure cluster assignments are a column vector
umap_clusters = umap_clusters.reshape(-1, 1)

# Append the cluster assignments as a new column to the UMAP matrix
#umap_with_clusters = np.hstack([descriptor_umap, umap_clusters])

# Extract the last three columns of the original descriptor matrix
last_three_columns = descriptor[:, -3:].cpu().numpy()

# Stack 'descriptor_umap', the last three columns, and 'umap_clusters'
umap_with_clusters = np.hstack([descriptor_umap, last_three_columns, umap_clusters])

# Save the resulting array as a .npy file
np.save(output_file_umap, umap_with_clusters)
print(f"UMAP matrix with clusters saved to {output_file_umap}")



