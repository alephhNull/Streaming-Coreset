import numpy as np
import torch

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