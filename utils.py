import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def calculate_mmd2_exact(X_full, X_coreset, coreset_weights, kernel_gamma):
    if len(X_coreset) == 0 or len(X_full) == 0:
        return np.inf

    X_full_torch = torch.from_numpy(X_full).float()
    X_coreset_torch = torch.from_numpy(X_coreset).float()
    w_torch = torch.from_numpy(coreset_weights).float()

    n = X_full_torch.shape[0]
    m = X_coreset_torch.shape[0]

    # RBF kernel function
    def rbf_kernel(X1, X2, gamma):
        dist_sq = torch.cdist(X1, X2)**2
        return torch.exp(-gamma * dist_sq)

    K_full_full = rbf_kernel(X_full_torch, X_full_torch, kernel_gamma)
    K_coreset_coreset = rbf_kernel(X_coreset_torch, X_coreset_torch, kernel_gamma)
    K_full_coreset = rbf_kernel(X_full_torch, X_coreset_torch, kernel_gamma)

    term1 = torch.mean(K_full_full) # E_{x,x' ~ P}[k(x, x')]
    term2 = w_torch @ K_coreset_coreset @ w_torch # E_{y,y' ~ Q}[k(y, y')]
    term3 = -2 * torch.mean(K_full_coreset @ w_torch) # -2 * E_{x ~ P, y ~ Q}[k(x, y)]

    mmd2 = term1 + term2 + term3
    return max(0, mmd2.item()) # MMD^2 should be non-negative


def calculate_mmd2_approx(X_full, X_coreset, coreset_weights, rbf_sampler):
    if len(X_coreset) == 0 or len(X_full) == 0:
        return np.inf

    # Transform datasets to RFF feature space
    z_X = rbf_sampler.transform(X_full)  # Shape: (n, n_components)
    z_Y = rbf_sampler.transform(X_coreset)  # Shape: (m, n_components)

    # Compute mean of z_X (uniform weights: 1/n)
    mean_z_X = np.mean(z_X, axis=0)  # Shape: (n_components,)

    # Compute weighted mean of z_Y
    mean_z_Y = np.dot(coreset_weights, z_Y)  # Shape: (n_components,)

    # Compute MMD^2 as squared Euclidean norm of the difference
    mmd2 = np.sum((mean_z_X - mean_z_Y)**2)

    return max(0, mmd2)


# Function to train the Autoencoder (if not pre-trained)
def train_autoencoder(model, train_loader, epochs=20, learning_rate=1e-3, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.to(device)
    model.train()
    
    print("Training Autoencoder...")
    for epoch in range(epochs):
        total_loss = 0
        for data in train_loader:
            img, _ = data
            img = img.view(img.size(0), -1).to(device) # Flatten image
            
            # Forward pass
            output = model(img)
            loss = criterion(output, img)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * img.size(0)
        
        avg_loss = total_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    print("Autoencoder training complete.")
