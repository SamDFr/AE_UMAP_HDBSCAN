# optunasearch.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import optuna
from optuna.samplers import TPESampler  # ou autre sampler si besoin
# from optuna.pruners import MedianPruner  # si vous voulez pruner les trials pas prometteurs

# Importez vos fonctions et classes
# Pour cet exemple, j'écris directement quelques import fictifs
from autoencoder_utils.load_ae_model import Autoencoder
from autoencoder_utils.training_prep_ae import train_val_test_split, train_autoencoder  # déjà définis dans votre code


def objective(trial, data, val_size, test_size):
    """
    Fonction objective pour Optuna.
    trial   : objet Trial d'Optuna
    data    : votre Tensor/array de descripteurs (ex. 'descriptor' dans le notebook)
    val_size: taille du split validation
    test_size: taille du split test
    """

    # == 1) Suggérer des hyperparamètres ==
    # Exemples:
    # - ratio pour hidden_dim_1
    hidden_ratio_1 = trial.suggest_float("hidden_ratio_1", 1.2, 3.0, step=0.2)
    # ratio pour hidden_dim_2 
    hidden_ratio_2 = trial.suggest_float("hidden_ratio_2", 1.2, 3.0, step=0.2)
    # - ratio pour latent_dim
    latent_ratio = trial.suggest_float("latent_ratio", 1.2, 3.0, step=0.2)
    # - batch_size
    batch_size   = trial.suggest_int("batch_size", 64, 512, step=128)
    # - learning_rate
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    # - epochs (optionnel, ou vous pouvez le fixer)
    num_epochs   = trial.suggest_int("num_epochs", 50, 250, step=50)

    # == 2) Construire le modèle avec ces hyperparamètres ==
    input_dim = data.shape[1]
    hidden_dim_1 = int(input_dim // hidden_ratio_1)
    hidden_dim_2 = int(hidden_dim_1 // hidden_ratio_2)
    latent_dim = int(hidden_ratio_2 // latent_ratio)

    autoencoder = Autoencoder(input_dim, hidden_dim_1, hidden_dim_2, latent_dim)

    # == 3) Préparer vos DataLoaders ==
    train_data, val_data, _ = train_val_test_split(data, val_size=val_size, test_size=test_size)
    train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(val_data), batch_size=batch_size, shuffle=True)

    # == 4) Définir la loss et l'optimizer ==
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    # == 5) Entraîner le modèle ==
    train_losses, val_losses = train_autoencoder(
        autoencoder, train_loader, val_loader,
        num_epochs, criterion, optimizer
    )

    # On récupère la dernière loss de validation pour juger la performance
    #final_val_loss = train_losses[-1]
    final_val_loss = val_losses[-1]

    # == 6) Retourner la métrique à minimiser (ici la validation loss) ==
    return final_val_loss


def run_optuna_search(data, val_size=0.2, test_size=0.1, n_trials=20):
    """
    Lance l'optimisation Optuna et renvoie le meilleur résultat.
    """
    # On peut configurer un sampler, un pruner, etc.
    sampler = TPESampler()  
    # pruner = MedianPruner(n_startup_trials=5)
    
    # Création de l'étude.
    study = optuna.create_study(
        direction="minimize",  # car on veut minimiser la val_loss
        sampler=sampler,
        # pruner=pruner  # si vous utilisez un pruner
    )

    # Lancer l’optimisation
    study.optimize(lambda trial: objective(trial, data, val_size, test_size), n_trials=n_trials)

    # Afficher ou renvoyer les meilleurs hyperparamètres
    print("\n===== Résultats de l'optimisation =====")
    print(f"Nombre d'essais: {len(study.trials)}")
    print(f"Meilleure Loss de validation: {study.best_value}")
    print(f"Meilleurs hyperparamètres: {study.best_trial.params}")

    return study
