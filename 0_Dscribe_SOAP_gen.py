import sys
import numpy as np
from ovito.io import import_file
from ovito.io.ase import ovito_to_ase
from dscribe.descriptors import SOAP  # Ensure you have dscribe installed
#from multiprocessing import Pool
import time
import os
import glob
#import sparse

# Start time
start_time = time.time()

# Main function
if __name__ == "__main__":
    # Ensure the script receives the file as an argument
    if len(sys.argv) != 2:
        print("Usage: python <script_name>.py <filename>")
        sys.exit(1)

    # Get the file name from the command-line arguments
    sim_file = sys.argv[1]

    # Load the simulation file
    pipeline = import_file(sim_file)
    data = pipeline.compute()

    try:
        positions_ase = ovito_to_ase(data)
        
        # Replace atom types
       # positions_ase.symbols = [
       #     "Ni" if symbol == "Au" else symbol 
       #     for symbol in positions_ase.get_chemical_symbols()
       # ]

        positions_ase.symbols = [
            "Ni" if symbol != "Ni" else symbol 
            for symbol in positions_ase.get_chemical_symbols()
        ]

    except Exception as e:
        print(f"Error during ASE conversion or atomic replacement: {e}")
        sys.exit(1)

    
    print("Computing descriptors ...")

# FCC

    r_cut = 4.7  # Cutoff radius for SOAP descriptor 
    n_max = 4  # Maximum radial basis functions
    l_max = 4  # Maximum angular momentum
    sigma = 0.25  # Width of Gaussian smearing


    print(f"Parameters: r_cut={r_cut}, n_max={n_max}, l_max={l_max}, sigma={sigma}")

    # Create the SOAP descriptor
    soap = SOAP(
    species=["Ni"],
    periodic=True,
    r_cut=r_cut,
    n_max=n_max,
    l_max=l_max,
    sigma=sigma,
    #sparse=True
    )

    descriptors_matrix = soap.create(positions_ase, n_jobs=1)

    print("Descriptor matrix computed.")
    print(descriptors_matrix.shape)

    # Save descriptor matrix to a file
    
    fname = "SOAP_" + sim_file + '_'
    # Define the directory to check (use "." for the current directory)
    directory = "."

    # Search for all .pkl files in the directory
    #npy_files = glob.glob(os.path.join(directory, fname+".npy"))
    npy_files = glob.glob(f"{fname}*.npy")

    # Check if any .pkl files are found
    if npy_files:
        for file in npy_files:
            os.remove(file)
            print(f"Deleted: {file}")

    else:
        print("No .npy files found.")
    
    np.save(fname, descriptors_matrix)
    print("Descriptor matrix saved.")

    # Save the parameters into a text file
    params_file = fname + "_params.txt"

    with open(params_file, "w") as f:
        f.write(f"File name: {sim_file}\n")
        f.write(f"r_cut: {r_cut}\n")
        f.write(f"n_max: {n_max}\n")
        f.write(f"l_max: {l_max}\n")
        f.write(f"sigma: {sigma}\n")
        f.write(f"Number of descriptors: {descriptors_matrix.shape[1]}\n")

    # End time
    end_time = time.time()

    # Elapsed time in seconds
    elapsed_time = end_time - start_time
    print(f"SOAP descriptors completed in {elapsed_time:.2f} seconds.")
