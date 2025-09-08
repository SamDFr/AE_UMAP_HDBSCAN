# %%
import data_loading.load_data as load_data
import defect_detection.detect_default as detect_default
import defect_detection.write_xyz as write_xyz
import torch
import numpy as np
import joblib
import random
import os
import glob
import argparse
import sys

# %%
## This is the HPC version of the code

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

# %%
# Load the trained Autoencoder model

autoencoder = torch.load('./training/ae/autoencoder_model.pth', map_location=torch.device('cpu'))


# %%
# Define the base data directory
data_dir = os.path.join(os.getcwd(), 'run/data')

# Automatically list all subdirectories in the data directory
subdirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

if not subdirs:
    raise FileNotFoundError("No subdirectories found in the 'data' directory.")

# Process each subdirectory
for subdir in subdirs:
    print(f"Processing directory: {subdir}")

    # Find the simulation file (first file ending with '.gz') in the current subdirectory
    gz_files = glob.glob(os.path.join(subdir, "*.gz"))
    if gz_files:
        sim_file_defect = gz_files[0]  # selects the first found .gz file
        print("Selected sim file:", sim_file_defect)
    else:
        print(f"No .gz file found in directory {subdir}, skipping.")
        continue  # assumes this code is inside a loop over directories

    # Load simulation data from the .gz file
    data = load_data.load_rawdata_single(sim_file_defect)
    positions_defect = data["positions"]
    positions_ase_defect = data["positions_ase"]
    particle_type_defect = data["particle_types"]
    type_definitions_defect = data["type_definitions"]

    # Find the descriptor file (first file ending with '.npy') in the current subdirectory
    npy_files = glob.glob(os.path.join(subdir, "*.npy"))
    if npy_files:
        desc_file_defect = npy_files[0]  # selects the first found .npy file
        print("Selected desc file:", desc_file_defect)
    else:
        print(f"No .npy file found in directory {subdir}, skipping.")
        continue  # assumes this code is inside a loop over directories

    # Load SOAP descriptors from the .npy file
    descriptors = load_data.load_data_single(desc_file_defect)

    # Optionally standardize the descriptors
    norma = True
    if norma:
        scaler = joblib.load('./training/ae/standard_scaler.pkl')
        # Clone, detach, and convert the tensor to torch.float32
        descriptors_defect_noscale = descriptors.clone().detach().to(torch.float32)
        descriptor_np = descriptors_defect_noscale.cpu().numpy()
        descriptors_scaled = scaler.transform(descriptor_np)
        descriptors_defect = torch.tensor(descriptors_scaled, dtype=torch.float32)
    else:
        descriptors_defect = descriptors.clone().detach().to(torch.float32)


    # Compute the reconstruction error for all the atoms in the structure 

    reconstruction_error = detect_default.compute_recon_error(autoencoder, descriptors_defect, type='ae')

    # Detect defects

    # Parse command-line arguments for thresholds
    parser = argparse.ArgumentParser(description="Process error threshold values for defect detection.")
    parser.add_argument("thresholds", type=str,
                        help="Comma-separated list of error threshold values (e.g., '0,1,2,3')")
    args = parser.parse_args()

    # Try converting the thresholds into a list of integers; exit if conversion fails
    try:
        thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
        if not thresholds:
            raise ValueError
    except ValueError:
        sys.exit("Error: Error thresholds must be a comma-separated list of float (e.g., '0,1,2,3').")

    # Convert the input string into a list of integers
    print("Using thresholds:", thresholds)
    
    cell_size = positions_ase_defect.get_cell()

    directory_xyz, directory_npy_reconerror, directory_npy_desc = subdir+'/defect_results/AE_xyz_files/' , subdir+'/defect_results/AE_npy_files/recon_error/', subdir+'/defect_results/AE_npy_files/desc/'
    directories = [directory_xyz, directory_npy_reconerror, directory_npy_desc]
    for i in directories:
        os.makedirs(i, exist_ok=True)
        if os.path.exists(i):
            for filename in os.listdir(i):
                file_path = os.path.join(i, filename)
                # Check if it is a file before removing it
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            print(f"Deleted files in: {i}")

    for threshold in thresholds:

        defect_indices = detect_default.defect_indices_atoms(reconstruction_error,threshold)
        reconstruction_error_def = reconstruction_error[defect_indices]

        # Assuming defect_indices and positions_defect are available after running the defect detection
        print(f"Detected {len(defect_indices)} defects out of {len(descriptors_defect)} atoms for MSE threshold of {threshold}.")

        if threshold == 0: # if the error = 0, all the atoms are considered
            file_name_descriptor_matrix_defect = directory_npy_desc+'detected_defects_AE_allatoms_desc.npy'
            file_name_reconerror = directory_npy_reconerror+'detected_defects_AE_allatoms_.npy'
            file_name_xyz = directory_xyz+'detected_defects_AE_allatoms_.xyz'
        else: 
            file_name_descriptor_matrix_defect = directory_npy_desc+'detected_defects_AE_'+str(threshold)+'_desc.npy'
            file_name_reconerror = directory_npy_reconerror+'detected_defects_AE_'+str(threshold)+'_.npy'
            file_name_xyz = directory_xyz+'detected_defects_AE_'+str(threshold)+'_.xyz'

        write_xyz.write_xyz(
            file_name_xyz, 
            positions_defect[defect_indices], 
            [particle_type_defect.name] * len(positions_defect[defect_indices]), 
            "Ni",
            cell_size, 
            reconstruction_error_def
        )
        
        # Create a 2-column matrix: first column for the defect indices, second for the reconstruction errors.
        defect_data = np.column_stack((defect_indices, reconstruction_error_def))
        
        # Save the matrix
        np.save(file_name_reconerror, defect_data)
        print(f"{file_name_reconerror} generated")
        print()
        
        # Save desc 
        if threshold != 0: #this avoid writing again the large .npy file 
            np.save(file_name_descriptor_matrix_defect, descriptors_defect_noscale[defect_indices])
            print(f"{file_name_descriptor_matrix_defect} generated")
            print()

            # Continue processing for this subdirectory as needed...
    print(f"Finished processing directory: {subdir}\n")


