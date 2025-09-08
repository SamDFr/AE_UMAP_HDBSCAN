# %%
import autoencoder_utils.training_prep_ae as training_prep_ae
import autoencoder_utils.load_ae_model as load_ae_model
import data_loading.load_data as load_data
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import ovito 
import matplotlib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import csv
import os
import joblib
import random
from autoencoder_utils.optunasearch_ae import run_optuna_search
import glob

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
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)
matplotlib.use('Agg')  # Définir le backend en mode headless
mplstyle.use('fast') #The fast style set simplification and chunking parameters to reasonable settings to speed up plotting large amounts of data

# %%
#Training descriptor matrix

# Define the base data directory
curr_dir = os.getcwd()
data_dir = os.path.join(curr_dir, './run/data/dataset/AE_training')

if not data_dir:
    raise FileNotFoundError("No './run/data/dataset/AE_training/' found.")

npy_files = glob.glob(os.path.join(data_dir, "*.npy"))
if npy_files:
    desc_file = npy_files[0]  # selects the first found .npy file
    print("Selected desc file:", desc_file)
else:
    raise FileNotFoundError(f"No .npy file found in directory {data_dir}")

#desc_file = "./desc_files/SOAP_subset_37044Ni.cif_.npy"
descriptor = load_data.load_data_single(desc_file)

norma = True
if norma:
    # Standardize the descriptors
    scaler = StandardScaler()
    
    # Instead of using torch.tensor(), use clone().detach() to copy the existing tensor
    descriptor_no_scale = descriptor.clone().detach().to(torch.float32)
    
    # Convert the tensor to a numpy array (if it's on GPU, move it to CPU first)
    descriptor_np = descriptor_no_scale.cpu().numpy()
    
    # Fit the scaler and transform the data
    descriptor_scaled = scaler.fit_transform(descriptor_np)
    
    # Convert the scaled numpy array back to a PyTorch tensor
    descriptor = torch.tensor(descriptor_scaled, dtype=torch.float32)
    
    # Save the fitted scaler to a file
    scaler_file = './training/ae/standard_scaler.pkl'
    if os.path.exists(scaler_file):
        os.remove(scaler_file)
        print(f"{scaler_file} already exists and has been deleted.")
    joblib.dump(scaler, scaler_file)
    print(f"Scaler saved to {scaler_file}")
else:
    descriptor = descriptor.clone().detach().to(torch.float32)

# %%
optunasearch = False
if optunasearch:
    val = 0.2
    test = 0.1
    study = run_optuna_search(descriptor, val_size=val, test_size=test, n_trials=30)
    best_params = study.best_trial.params
    print(best_params)

    # Par exemple, si vous voulez ré-entraîner votre réseau final avec ces paramètres :
    input_dim = descriptor.shape[1]
    hidden_dim_1 = best_params['hidden_ratio_1']
    hidden_dim_2 = best_params['hidden_ratio_2']
    latent_dim = best_params['latent_ratio']
    batch_size   = best_params['batch_size']
    learning_rate = best_params['lr']
    num_epochs   = best_params['num_epochs']

    # Split the data
    train_data, val_data, test_data = training_prep_ae.train_val_test_split(descriptor, val_size=val, test_size=test)

    # DataLoader for training, validation, and testing
    train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data), batch_size=batch_size, shuffle=True)

else:
    # Hyperparameters
    input_dim = descriptor.shape[1]
    hidden_dim_1 = int(input_dim // 1.3) #int(input_dim // 1.5)
    hidden_dim_2 = int(hidden_dim_1 // 2)
    latent_dim = int(input_dim // 5) #int(input_dim // 5)
    batch_size = 256 #256
    num_epochs = 80 #150
    learning_rate = 0.00027362954959066834 #1e-4

    # Split the data

    val = 0.2
    test = 0.1
    train_data, val_data, test_data = training_prep_ae.train_val_test_split(descriptor, val_size=val, test_size=test)

    # DataLoader for training, validation, and testing
    train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data), batch_size=batch_size, shuffle=True)


# %%
# Autoencoder model
autoencoder = load_ae_model.Autoencoder(input_dim, hidden_dim_1, hidden_dim_2, latent_dim)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

# Train the autoencoder with validation
train_losses, val_losses = training_prep_ae.train_autoencoder(autoencoder, train_loader, val_loader, num_epochs, criterion, optimizer)

# Save the trained model's state dictionary to a file
model_save_path = './training/ae/autoencoder_model.pth'
if os.path.exists(model_save_path):
    os.remove(model_save_path)
    print(f"{model_save_path} already exists and has been deleted.")
torch.save(autoencoder, model_save_path)
print(f"Autoencoder model saved to {model_save_path}")

# %%
# After training, test the autoencoder on the test set
training_prep_ae.test_autoencoder(autoencoder, test_loader, criterion)

# %%
#Save errors into files

# Open the file in write mode and create a csv writer object
csv_filename = "./training/ae/losses.csv"

# Check if the file already exists and delete it if it does
if os.path.exists(csv_filename):
    os.remove(csv_filename)
    print(f"{csv_filename} already exists and has been deleted.")


with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write the header row
    writer.writerow(["training loss", "validation loss"])
    
    # Write the data rows by iterating through both lists simultaneously
    for train_loss, val_loss in zip(train_losses, val_losses):
        writer.writerow([train_loss, val_loss])

print(f"Data saved to {csv_filename}")


# %%
# Plotting the loss curves after training
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', linewidth=2)
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', linewidth=2)
plt.xlabel('Epochs',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Training and Validation Loss', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)

plot_file_name = './training/ae/training_errors.pdf'
if os.path.exists(plot_file_name):
    os.remove(plot_file_name)
    print(f"{plot_file_name} already exists and has been deleted.")

plt.savefig(plot_file_name, format='pdf', bbox_inches='tight')


# ------------------------------------------------------------
# ► Save hyper-parameters and run metadata for reproducibility
# ------------------------------------------------------------
hyperparams = {
    "descriptor_file":    desc_file,
    "seed":               seed,
    "input_dim":          input_dim,
    "hidden_dim_1":       hidden_dim_1,
    "hidden_dim_2":       hidden_dim_2,
    "latent_dim":         latent_dim,
    "batch_size":         batch_size,
    "num_epochs":         num_epochs,
    "learning_rate":      learning_rate,
    "val_size":           val,
    "test_size":          test,
    "normalization":      norma,
    "optuna_search":      optunasearch,
}

# Where to put the file
hp_path = "./training/ae/hyperparameters.txt"

# Delete any old copy so you never append accidentally
if os.path.exists(hp_path):
    os.remove(hp_path)
    print(f"{hp_path} already exists and has been deleted.")

# Write key: value pairs line-by-line
with open(hp_path, "w") as f:
    for k, v in hyperparams.items():
        f.write(f"{k}: {v}\n")

print(f"Hyper-parameters saved to {hp_path}")
