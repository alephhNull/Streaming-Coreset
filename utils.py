import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import ot


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


def calculate_wass_distance(X_full, X_coreset, coreset_weights, p=1):
    
    X_full = np.asarray(X_full, dtype=float)
    X_coreset = np.asarray(X_coreset, dtype=float)
    coreset_weights = np.asarray(coreset_weights, dtype=float)

    n_full = X_full.shape[0]
    n_core = X_coreset.shape[0]

    # 1) Build the weight vectors:
    #    - full dataset: uniform weights summing to 1
    a = np.ones(n_full) / n_full
    #    - coreset: given weights (must sum to 1)
    b = coreset_weights / np.sum(coreset_weights)

    # 2) Compute cost matrix M_{i,j} = ||x_i - y_j||^p
    #    Here we use the Euclidean norm to the power p.
    M = ot.dist(X_full, X_coreset, metric='euclidean')**p

    # 3) Solve the optimal transport problem
    #    emd2 returns the *value* of the minimal transport cost,
    #    i.e. W_p^p. So we take the p-th root to get W_p.
    Wp_p = ot.emd2(a, b, M)
    Wp = Wp_p**(1.0 / p)

    return Wp
