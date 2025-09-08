# autoencoder_utils.py

import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import ovito.io  # Ensure that you have the correct ovito module installed
import pandas as pd 

   
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
    #positions_ase_defect.symbols = ["Ni" if symbol == "Au" else symbol for symbol in positions_ase_defect.get_chemical_symbols()]
    positions_ase_defect.symbols = ["Ni" if symbol != "Ni" else symbol for symbol in positions_ase_defect.get_chemical_symbols()]

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

def load_rawdata_single(sim_file):

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
    positions_ase.symbols = [
    "Fe" if symbol == "Na" else "Ni" if symbol == "Au" else symbol
    for symbol in positions_ase.get_chemical_symbols()
    ]


    print("ASE type Atoms object generated")
    print("Atoms type have benn changed from Au to Ni")
    print("Atoms type have benn changed from Na to Fe")

        # Return the relevant data
    return {
        "positions": positions,
        "positions_ase": positions_ase,
        "particle_types": particle_types,
        "type_definitions": type_definitions,
    }


    # Load the descriptors for one structure only
def load_data_single(desc_file):
    # Replace these lines with actual data loading
    descriptor = np.load(desc_file)
# Example defective descriptors
    return torch.tensor(descriptor, dtype=torch.float32)
   
# ---------------------------
# Define a function to load your XYZ file into a DataFrame.
def load_xyz_with_recon(filename):
    """
    Load an XYZ file where:
      - The first line is the number of atoms.
      - The second line contains lattice and property information.
      - The subsequent lines contain:
            Species x y z ReconError
    Returns a pandas DataFrame with columns: 'Species', 'x', 'y', 'z', 'ReconError'.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    # First line: number of atoms (we can store or ignore it)
    num_atoms = int(lines[0].strip())
    # Second line: lattice and property info (skip for now)
    data_lines = lines[2:]
    # Split each line by whitespace
    rows = [line.strip().split() for line in data_lines if line.strip()]
    # Create DataFrame; adjust the column names if needed
    df = pd.DataFrame(rows, columns=['Species', 'x', 'y', 'z', 'ReconError'])
    # Convert numeric columns to float
    for col in ['x', 'y', 'z', 'ReconError']:
        df[col] = pd.to_numeric(df[col])
    return df