# %%
import numpy as np
import pandas as pd
import data_loading.load_data as load_data
import numpy as np
import os
import glob
import ase.io
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib.colors import LogNorm
import sys
#import matplotlib.style as mplstyle


# %%
matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16)
#matplotlib.use('Agg')  # Définir le backend en mode headless
#mplstyle.use('fast') #The fast style set simplification and chunking parameters to reasonable settings to speed up plotting large amounts of data

# %%
#Load the data that contains the clustering results after UMAP

results_UMAP_HDBSCAN_root = sys.argv[1]

print(f"Loading UMAP results from {results_UMAP_HDBSCAN_root}")
if not os.path.exists(results_UMAP_HDBSCAN_root):
    raise FileNotFoundError(f"The file {results_UMAP_HDBSCAN_root} does not exist.")
if not os.path.isfile(results_UMAP_HDBSCAN_root):
    raise ValueError(f"The path {results_UMAP_HDBSCAN_root} is not a file.")
results_UMAP_HDBSCAN = np.load(results_UMAP_HDBSCAN_root)

cluster_labels = results_UMAP_HDBSCAN[:, -1].astype(int) # Last column contains cluster labels
recon_err = results_UMAP_HDBSCAN[:, -2]  
id_traj = results_UMAP_HDBSCAN[:, -4].astype(int)
id_atoms = results_UMAP_HDBSCAN[:, -3].astype(int) 

# Create a DataFrame with the extracted information
df_results = pd.DataFrame({
    'id_traj': id_traj,
    'id_atoms': id_atoms,
    'recon_err': recon_err,
    'cluster_labels': cluster_labels
})

#print(df_results.head())

# %%
# ==============================
# PART 1: Processing Trajectories
# ==============================

# Define the base data directory
data_dir = os.path.join(os.getcwd(), 'run/data')

# Define the directory to save the atomic percentage data
directory_save_perc = os.path.join(data_dir, 'atom_perc_cluster')
os.makedirs(directory_save_perc, exist_ok=True)

# Automatically list all subdirectories in the data directory
subdirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

if not subdirs:
    raise FileNotFoundError("No subdirectories found in the 'data' directory.")

# Create a list to store atomic percentage data for all clusters across trajectories
all_atomic_percentage_data = []

# Process each subdirectory (trajectory)
for subdir in subdirs:
    print(f"Processing directory: {subdir}")

    # Find the simulation file (first file ending with '.gz') in the current subdirectory
    gz_files = glob.glob(os.path.join(subdir, "*.gz"))
    if gz_files:
        sim_file_defect = gz_files[0]  # selects the first found .gz file
        print("Selected sim file:", sim_file_defect)
    else:
        print(f"No .gz file found in directory {subdir} \n")
        continue
        
    # Extract the id_traj from the filename
    id_traj_from_file = int(os.path.basename(sim_file_defect).split('_')[0])
    if id_traj_from_file not in id_traj:
        raise ValueError(f"The id_traj {id_traj_from_file} extracted from the file name is not in the loaded id_traj array.")
    
    print(f"Processing id_traj: {id_traj_from_file}")
    
    # Load simulation data from the .gz file
    data = load_data.load_rawdata_single(sim_file_defect)
    positions_defect = data["positions"]
    positions_ase_defect = data["positions_ase"]
    particle_type_defect = data["particle_types"]
    type_definitions_defect = data["type_definitions"]

    traj_data = df_results[df_results['id_traj'] == id_traj_from_file]

    # Define the output directory for the cluster XYZ files and clear existing files
    directory_umap_hdbscan = os.path.join(subdir, 'defect_results', 'UMAP_HDBSCAN_XYZ')
    os.makedirs(directory_umap_hdbscan, exist_ok=True)
    for filename in os.listdir(directory_umap_hdbscan):
        file_path = os.path.join(directory_umap_hdbscan, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    print(f"Cleared files in: {directory_umap_hdbscan}\n")

    # Generate XYZ files for each cluster for this trajectory
    clusters = traj_data['cluster_labels'].unique()
    for cluster in clusters:
        # Get the atom indices corresponding to this cluster
        cluster_data = traj_data[traj_data['cluster_labels'] == cluster]
        atom_indices = cluster_data['id_atoms'].tolist()
        
        # Create a subset ASE Atoms object using the indices
        cluster_atoms = positions_ase_defect[atom_indices]
        
        # Define the output file name (e.g., cluster_0.xyz)
        output_file = os.path.join(directory_umap_hdbscan, f"cluster_{cluster}.xyz")
        ase.io.write(output_file, cluster_atoms)
        print(f"Written cluster {cluster} with {len(atom_indices)} atoms to {output_file}")
    
        # ===== NEW FEATURE: Calculate atomic composition percentages =====

        
        symbols_cluster = [atom.symbol for atom in cluster_atoms]

            # Check if multiple species are present
        unique_species = set(symbols_cluster)
        
        if len(unique_species) > 1:
            # ===== Calculate atomic composition percentages =====
            from collections import Counter
            counts = Counter(symbols_cluster)
            total_atoms = len(symbols_cluster)
            percentages = {elem: (count / total_atoms) * 100 for elem, count in counts.items()}
            print(f"Atomic composition percentages for cluster {cluster} in traj {id_traj_from_file}: {percentages}")
            
            # Store the percentage information for later aggregation and plotting
            all_atomic_percentage_data.append({
                'traj': id_traj_from_file,
                'cluster': cluster,
                'percentages': percentages,
                'number_atoms': total_atoms,
            })
        else:
            # If there is only one species, skip composition analysis
            print(f"Cluster {cluster} in traj {id_traj_from_file} has a single species ({unique_species}). Skipping composition analysis.")
    
    
    # Create an aggregated XYZ file with an extra column for cluster ID.
    all_atom_indices = traj_data['id_atoms'].tolist()
    cluster_mapping = dict(zip(traj_data['id_atoms'], traj_data['cluster_labels']))
    
    positions = positions_ase_defect.get_positions()  # numpy array (n_atoms, 3)
    symbols = positions_ase_defect.get_chemical_symbols()  # list of chemical symbols
    cell_size = positions_ase_defect.get_cell()
    
    output_all = os.path.join(directory_umap_hdbscan, "all_clusters.xyz")
    with open(output_all, 'w') as f:
        f.write(f"{len(all_atom_indices)}\n")
        f.write(f'Lattice="{cell_size[0][0]} 0.0 0.0 0.0 {cell_size[1][1]} 0.0 0.0 0.0 {cell_size[2][2]}" Properties=species:S:1:pos:R:3:ClusterID:R:1 \n')
        for atom_index in all_atom_indices:
            symbol = symbols[atom_index]
            x, y, z = positions[atom_index]
            cluster_label = cluster_mapping[atom_index]
            f.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f} {cluster_label}\n")
    print(f"Written aggregated file with all clusters to {output_all}\n")

# %%
# ==========================================
# PART 2: Save Atomic Percentage Results to CSV
# ==========================================
if len(all_atomic_percentage_data) == 0:
    print("No atomic percentage data to save.")
else:
    print(f"Collected atomic percentage data for clusters across all trajectories.")
    # Convert the nested percentage data into a flat table
    records = []
    for entry in all_atomic_percentage_data:
        traj = entry['traj']
        cluster = entry['cluster']
        for species, percentage in entry['percentages'].items():
            records.append({
                'traj': traj,
                'cluster': cluster,
                'species': species,
                'percentage': percentage
            })

    # Create a DataFrame from the records
    df_atomic_percentages = pd.DataFrame(records)

    # Save the DataFrame to a CSV file for later use
    directory_save_perc = os.path.join(data_dir, 'atom_perc_cluster')
    os.makedirs(directory_save_perc, exist_ok=True)

    for filename in os.listdir(directory_save_perc):
        file_path = os.path.join(directory_save_perc, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        
    print(f"Cleared files in: {directory_save_perc}")

    csv_filename = os.path.join(directory_save_perc,"atomic_percentage_results.csv")
    df_atomic_percentages.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")

    # ==========================================
    # PART 3: Reload Data and Plot Per-Cluster Statistics
    # ==========================================

    # Later, when you want to reload and plot statistics, do:
    df_loaded = pd.read_csv(csv_filename)

    # Aggregate statistics: group by cluster and species to compute average and standard deviation
    cluster_stats = {}
    for cluster, group in df_loaded.groupby('cluster'):
        species_dict = {}
        for species, sp_group in group.groupby('species'):
            avg = sp_group['percentage'].mean()
            std = sp_group['percentage'].std()
            species_dict[species] = {'avg': avg, 'std': std}
        cluster_stats[cluster] = species_dict

    # Plot one figure per cluster with a bar chart for each atomic species
    for cluster, stats in cluster_stats.items():
        species_list = list(stats.keys())
        avg_percentages = [stats[s]['avg'] for s in species_list]
        std_percentages = [stats[s]['std'] for s in species_list]

            # Define a different color for each species using a colormap (here tab10)
        #colors = plt.cm.tab10(np.linspace(0, 1, len(species_list)))
        colors = ['purple', 'red', 'green', 'blue', 'red', 'pink', 'brown', 'gray', 'cyan', 'magenta'][:len(species_list)]
        
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(species_list, avg_percentages, yerr=std_percentages, capsize=5, color=colors, alpha=0.6)
        plt.xlabel('Atomic Species', fontsize=16)
        plt.ylabel('Average Percentage', fontsize=16)
        plt.title(f'Atomic Composition for Cluster {cluster}', fontsize=16)
        plt.ylim(0, 100)
        
        # Annotate each bar with "avg±std%" above the bar
        # Annotate each bar with "avg±std%" above the error bar
        offset = 2  # offset in percentage points
        for bar, avg, std in zip(bars, avg_percentages, std_percentages):
            # Compute annotation y position as top of error bar + offset
            annotation_y = bar.get_height() + std + offset
            annotation = f"{avg:.1f} ± {std:.1f}%"
            plt.text(bar.get_x() + bar.get_width() / 2, annotation_y, annotation, 
                    ha='center', va='bottom', fontsize=16)
            
        plt.savefig(os.path.join(directory_save_perc, f"cluster_{cluster}_composition.pdf"), format='pdf', bbox_inches='tight')
        plt.close()


# %%
df_results['log_recon_err'] = np.log(df_results['recon_err'])

plt.figure(figsize=(8, 6))
for cluster in clusters:
    subset = df_results.loc[df_results['cluster_labels'] == cluster, 'log_recon_err']
    sns.kdeplot(subset, fill=True, label=f'Cluster {cluster}', alpha=0.3)

plt.xlabel('Log(Reconstruction Error)', fontsize=16)
plt.ylabel('Density', fontsize=16)
plt.title('Distribution of Log(Reconstruction Error) by Cluster', fontsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(directory_save_perc, f"KDE_reconn_err.pdf"), format='pdf', bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
for cluster in clusters:
    if cluster != -1:
        subset = df_results.loc[df_results['cluster_labels'] == cluster, 'log_recon_err']
        sns.kdeplot(subset, fill=True, label=f'Cluster {cluster}', alpha=0.3)

plt.xlabel('Log(Reconstruction Error)', fontsize=16)
plt.ylabel('Density', fontsize=16)
plt.title('Distribution of Log(Reconstruction Error) by Cluster', fontsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(directory_save_perc, f"KDE_reconn_err_nonoise.pdf"), format='pdf', bbox_inches='tight')
plt.close()

# %%


# Assume df_results is your DataFrame and it contains 'recon_err' and 'cluster_labels'.

# 1. Compute the minimum and maximum reconstruction errors.
min_recon_err = df_results['recon_err'].min()  # > 5 by assumption
print(f"Minimum reconstruction error: {min_recon_err}")
max_recon_err = df_results['recon_err'].max()
print(f"Maximum reconstruction error: {max_recon_err}")

# 2. Compute the log of these values.
log_min = np.log(min_recon_err)
log_max = np.log(max_recon_err)

# 3. Create log-spaced bin edges that exactly span the data.
n_bins = 200  # This gives n_bins-1 bins.
bin_edges_log = np.linspace(log_min, log_max, n_bins)

# 4. Create a new column with log-transformed reconstruction error.
df_results['log_recon_err'] = np.log(df_results['recon_err'])

# 5. Identify unique clusters and sort them.
clusters_unique = sorted(df_results['cluster_labels'].unique())

# 6. Build a 2D histogram: rows for log-recon_err bins, columns for clusters.
# data2d will have shape: (n_bins - 1, number of clusters)
data2d = np.zeros((n_bins - 1, len(clusters_unique)))

for j, cl in enumerate(clusters_unique):
    subset = df_results[df_results['cluster_labels'] == cl]['log_recon_err']
    hist, _ = np.histogram(subset, bins=bin_edges_log)
    data2d[:, j] = hist

# 7. Mask the zero counts so they don't affect the color scale.
data2d_masked = np.ma.masked_where(data2d == 0, data2d)

# 8. Determine vmin and vmax for LogNorm using only nonzero counts.
nonzero_counts = data2d[data2d > 0]
vmin = nonzero_counts.min()
vmax = data2d.max()

# 9. Create x and y bin edge arrays.
# For clusters: use indices [0, 1, ..., number_of_clusters]
x_edges = np.arange(len(clusters_unique) + 1)
# y_edges are our log-transformed bin edges.
y_edges = bin_edges_log

# 10. Create a meshgrid from these 1D edge arrays.
X, Y = np.meshgrid(x_edges, y_edges)  # X, Y shapes: (n_bins, number_of_clusters+1)

plt.figure(figsize=(8, 6))

# 11. Plot the heatmap with pcolormesh.
pcm = plt.pcolormesh(
    X, Y, data2d_masked, 
    cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax),
    shading='auto'
)

# 12. Set the y-axis limits to exactly the bin range.
plt.ylim(bin_edges_log[0], bin_edges_log[-1])

# 13. Adjust y-axis tick labels to show the original reconstruction error scale.
#yticks = plt.yticks()[0]
#plt.yticks(yticks, [f"{np.exp(tick):.1f}" for tick in yticks])

plt.xlabel('Cluster Label', fontsize=16)
plt.ylabel('Reconstruction Error', fontsize=16)
plt.title('Heatmap of Log-transformed Reconstruction Error Distribution by Cluster', fontsize=16)

# Center the x-tick labels: for clusters, place them at 0.5, 1.5, etc.
plt.xticks(np.arange(len(clusters_unique)) + 0.5, clusters_unique, fontsize=14)

cb = plt.colorbar(pcm)
cb.set_label('Counts (log scale)', fontsize=16)
cb.ax.tick_params(labelsize=14)

plt.tight_layout()
plt.savefig(os.path.join(directory_save_perc, f"heatmap_recon_err.pdf"), format='pdf', bbox_inches='tight')
plt.close()



