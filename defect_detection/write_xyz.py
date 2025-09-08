# autoencoder_utils.py

import os

def write_xyz(filename, positions, types, atom_file, cell_size, reconstruction_error):

    # Check if the file already exists and delete it
    if os.path.exists(filename):
        os.remove(filename)
        print(f"{filename} already exists. It has been deleted.")


    with open(filename, 'w') as f:
        f.write(f"{len(positions)}\n")
        # Ajoutez des informations sur la cellule de la structure (si nécessaire)
        f.write(f'Lattice="{cell_size[0][0]} 0.0 0.0 0.0 {cell_size[1][1]} 0.0 0.0 0.0 {cell_size[2][2]}" Properties=species:S:1:pos:R:3:ReconError:R:1  \n')
        
        # Écrire les positions des atomes et leurs types
        for atom_type, pos, err in zip(types, positions, reconstruction_error):
            f.write(f"{atom_file} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {err:.6f} \n")
    
    print(f"{filename} generated")

def write_xyz_cluster(base_filename, positions, types, atom_file, cell_size, cluster_labels):
    """
    Save each cluster's atoms into separate XYZ files.

    Args:
        base_filename (str): Base name for the output files.
        positions (ndarray): Atomic positions (N x 3 array).
        types (list): List of atomic types.
        atom_file (str): Name of the atom type (e.g., "Ni").
        cell_size (float): Size of the lattice cell.
        cluster_labels (ndarray): Cluster labels for each atom (length N).
    """
    # Get unique cluster labels
    unique_clusters = set(cluster_labels)
    
    for cluster in unique_clusters:
        # Filter atoms belonging to the current cluster
        cluster_indices = (cluster_labels == cluster)
        cluster_positions = positions[cluster_indices]
        cluster_types = [types[i] for i in range(len(types)) if cluster_indices[i]]
        
        # Create a filename for the cluster
        cluster_filename = f"{base_filename}_cluster_{cluster}.xyz"
        
        # Write XYZ file for the cluster
        with open(cluster_filename, 'w') as f:
            f.write(f"{len(cluster_positions)}\n")
            f.write(f'Lattice="{cell_size[0][0]} 0.0 0.0 0.0 {cell_size[1][1]} 0.0 0.0 0.0 {cell_size[2][2]}" Properties=species:S:1:pos:R:3 \n')
            for atom_type, pos in zip(cluster_types, cluster_positions):
                f.write(f"{atom_file} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
        
        print(f"{cluster_filename} generated")
    
    # Create a combined file with all atoms and ClusterID
    combined_filename = f"{base_filename}_all_clusters.xyz"
    with open(combined_filename, 'w') as f:
        f.write(f"{len(positions)}\n")
        f.write(f'Lattice="{cell_size[0][0]} 0.0 0.0 0.0 {cell_size[1][1]} 0.0 0.0 0.0 {cell_size[2][2]}" Properties=species:S:1:pos:R:3:ClusterID:R:1 \n')
        for atom_type, pos, label in zip(types, positions, cluster_labels):
            f.write(f"{atom_file} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {label}\n")
    
    print(f"{combined_filename} generated")
   