# autoencoder_utils.py

import os
import torch
from sklearn.model_selection import train_test_split

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
        if (epoch == 0) or ((epoch + 1) % 10 == 0):
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
