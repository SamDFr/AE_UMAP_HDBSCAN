# autoencoder_utils.py

import torch.nn as nn

# Define the Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, latent_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),  # Input to hidden_dim
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),  # Hidden_dim to hidden_dim // 2
            nn.ReLU()
        )
        self.fc_latent = nn.Linear(hidden_dim_2, latent_dim)  # Hidden_dim // 2 to latent_dim
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_2),  # Latent_dim to hidden_dim // 2
            nn.ReLU(),
            nn.Linear(hidden_dim_2, hidden_dim_1),  # Hidden_dim // 2 to hidden_dim
            nn.ReLU(),
            nn.Linear(hidden_dim_1, input_dim)  # Hidden_dim to input_dim
        )

    def forward(self, x):
        # Encoder
        hidden = self.encoder(x)  # Pass through the encoder
        latent = self.fc_latent(hidden)  # Compress to latent space

        # Decoder
        reconstructed = self.decoder(latent)  # Reconstruct the input from latent space

        return reconstructed, latent  # Return reconstructed data and latent representation
        

def get_latent_representation(autoencoder, descriptor):
    """
    Extracts the latent representation of the input descriptor using the autoencoder.
    """
    # Forward pass to get reconstruction and latent representation
    _, latent = autoencoder(descriptor)
    return latent


