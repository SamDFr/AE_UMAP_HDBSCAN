# autoencoder_utils.py

import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import ovito.io  # Ensure that you have the correct ovito module installed

# Define the Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Input to hidden_dim
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),  # Hidden_dim to hidden_dim // 2
            nn.ReLU()
        )
        self.fc_latent = nn.Linear(hidden_dim // 2, latent_dim)  # Hidden_dim // 2 to latent_dim
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),  # Latent_dim to hidden_dim // 2
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),  # Hidden_dim // 2 to hidden_dim
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)  # Hidden_dim to input_dim
        )

    def forward(self, x):
        # Encoder
        hidden = self.encoder(x)  # Pass through the encoder
        latent = self.fc_latent(hidden)  # Compress to latent space

        # Decoder
        reconstructed = self.decoder(latent)  # Reconstruct the input from latent space

        return reconstructed, latent  # Return reconstructed data and latent representation
        
def load_rawdata(sim_file,sim_file_defect):

    pipeline = ovito.io.import_file(sim_file)
    print("Pipeline correctly loaded")

    # Access the atomic positions and species
    data = pipeline.compute()
    number_of_atoms = data.particles.count
    print(f"Number of atoms: {number_of_atoms}")

    positions = data.particles.positions

    # Access the associated ParticleType property manager
    particle_types = data.particles['Particle Type']

    # Access the definitions of the particle types
    type_definitions = data.particles.particle_type.types

    print("Defined particle types:")
    for particle_type in type_definitions:
        print(f"Type ID: {particle_type.id}, Name: {particle_type.name}, Mass: {particle_type.mass}")

    #convert to ASE atoms object (to use the dscribe)
    positions_ase = ovito.io.ase.ovito_to_ase(data)
    positions_ase.symbols = ["Ni" if symbol == "Au" else symbol for symbol in positions_ase.get_chemical_symbols()]

    print("ASE type Atoms object generated")
    print("Atoms type have benn changed from Au to Ni \n")

    pipeline_defect = ovito.io.import_file(sim_file_defect)
    print("Pipeline for defect structures correctly loaded")

        # Access the atomic positions and species
    data_defect = pipeline_defect.compute()
    number_of_atoms_defect = data_defect.particles.count
    print(f"Number of atoms: {number_of_atoms_defect}")

    positions_defect = data_defect.particles.positions

    # Access the associated ParticleType property manager
    particle_types_defect = data_defect.particles['Particle Type']

    # Access the definitions of the particle types
    type_definitions_defect = data_defect.particles.particle_type.types

    print("Defined particle types:")
    for particle_type_defect in type_definitions_defect:
        print(f"Type ID: {particle_type_defect.id}, Name: {particle_type_defect.name}, Mass: {particle_type_defect.mass}")

    #convert to ASE atoms object (to use the dscribe)
    positions_ase_defect = ovito.io.ase.ovito_to_ase(data_defect)
    positions_ase_defect.symbols = ["Ni" if symbol == "Au" else symbol for symbol in positions_ase_defect.get_chemical_symbols()]

    print("ASE type Atoms object generated")
    print("Atoms type have benn changed from Au to Ni \n")

        # Return the relevant data
    return {
        "positions": positions,
        "positions_ase": positions_ase,
        "particle_types": particle_types,
        "type_definitions": type_definitions,
        "positions_defect": positions_defect,
        "positions_ase_defect": positions_ase_defect,
        "particle_types_defect": particle_types_defect,
        "type_definitions_defect": type_definitions_defect
    }

    # Load the descriptors
def load_data(desc_file,desc_file_defect):
    # Assuming `descriptor` and `descriptors_defect` are numpy arrays
    # Replace these lines with actual data loading
    descriptor = np.load(desc_file)
# Example defect-free descriptors
    descriptors_defect = np.load(desc_file_defect)
# Example defective descriptors
    return torch.tensor(descriptor, dtype=torch.float32), torch.tensor(descriptors_defect, dtype=torch.float32)

    # Load the descriptors for one structure only
def load_data_single(desc_file):
    # Replace these lines with actual data loading
    descriptor = np.load(desc_file)
# Example defective descriptors
    return torch.tensor(descriptor, dtype=torch.float32)


# Function to split the dataset into train, validation, and test sets
def train_val_test_split(data, val_size=0.1, test_size=0.1, random_seed=42):
    # Split into train + val and test
    train_val, test = train_test_split(data, test_size=test_size, random_state=random_seed)
    # Split the train+val set into train and validation
    train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=random_seed)
    return train, val, test


# Modify the training function to include validation
def train_autoencoder(autoencoder, train_loader, val_loader, num_epochs, criterion, optimizer):
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        autoencoder.train()
        epoch_train_loss = 0.0
        for data in train_loader:
            inputs = data[0]

            # Forward pass
            reconstructed, _ = autoencoder(inputs)
            loss = criterion(reconstructed, inputs)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        # Store the average training loss for the epoch
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation loss after each epoch
        autoencoder.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                inputs = data[0]
                reconstructed, _ = autoencoder(inputs)
                loss = criterion(reconstructed, inputs)
                epoch_val_loss += loss.item()

        # Store the average validation loss for the epoch
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Print epoch statistics
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.8f}, Validation Loss: {avg_val_loss:.8f}")
        
    return train_losses, val_losses

def test_autoencoder(autoencoder, test_loader, criterion):
    autoencoder.eval()
    test_loss = 0.0

    with torch.no_grad():
        for data in test_loader:
            inputs = data[0]
            reconstructed, _ = autoencoder(inputs)
            loss = criterion(reconstructed, inputs)

            # Accumulate the loss values
            test_loss += loss.item()

    # Compute average test loss for all batches
    avg_test_loss = test_loss / len(test_loader)

    # Print out the test loss, reconstruction loss, and KL divergence
    print(f"Test Loss: {avg_test_loss:.8f}")


# Detecting defects
def detect_defects(autoencoder, defect_descriptors, threshold):
    """
    Detects defects using the reconstruction error from the autoencoder.

    Args:
        autoencoder: Trained autoencoder model.
        defect_descriptors: Tensor of defect descriptors.
        threshold: Reconstruction error threshold for detecting defects.

    Returns:
        defect_indices: Indices of atoms considered defective.
        reconstruction_error: Reconstruction error for all atoms.
    """
    autoencoder.eval()
    with torch.no_grad():
        # Forward pass to get the reconstructed descriptors
        reconstructions, _ = autoencoder(defect_descriptors)  # Extract the reconstructed output
        # Calculate the reconstruction error
        reconstruction_error = torch.mean((reconstructions - defect_descriptors) ** 2, dim=1)

    # Detect defects based on the threshold
    defect_indices = torch.where(reconstruction_error > threshold)[0]
    return defect_indices, reconstruction_error


def write_xyz(filename, positions, types, atom_file, cell_size, reconstruction_error):

    # Check if the file already exists and delete it
    if os.path.exists(filename):
        os.remove(filename)
        print(f"{filename} already exists. It has been deleted.")


    with open(filename, 'w') as f:
        f.write(f"{len(positions)}\n")
        # Ajoutez des informations sur la cellule de la structure (si nécessaire)
        f.write(f'Lattice="{cell_size} 0.0 0.0 0.0 {cell_size} 0.0 0.0 0.0 {cell_size}" Properties=species:S:1:pos:R:3:ReconError:R:1  \n')
        
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
            f.write(f'Lattice="{cell_size} 0.0 0.0 0.0 {cell_size} 0.0 0.0 0.0 {cell_size}" Properties=species:S:1:pos:R:3 \n')
            for atom_type, pos in zip(cluster_types, cluster_positions):
                f.write(f"{atom_file} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
        
        print(f"{cluster_filename} generated")
    
    # Create a combined file with all atoms and ClusterID
    combined_filename = f"{base_filename}_all_clusters.xyz"
    with open(combined_filename, 'w') as f:
        f.write(f"{len(positions)}\n")
        f.write(f'Lattice="{cell_size} 0.0 0.0 0.0 {cell_size} 0.0 0.0 0.0 {cell_size}" Properties=species:S:1:pos:R:3:ClusterID:R:1 \n')
        for atom_type, pos, label in zip(types, positions, cluster_labels):
            f.write(f"{atom_file} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {label}\n")
    
    print(f"{combined_filename} generated")
   

# Ajout d'un forward pass pour récupérer la couche latente

def get_latent_representation(autoencoder, descriptor):
    """
    Extracts the latent representation of the input descriptor using the autoencoder.
    """
    # Forward pass to get reconstruction and latent representation
    _, latent = autoencoder(descriptor)
    return latent