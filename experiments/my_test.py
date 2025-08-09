import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import torch
import torch.optim as optim

def weighted_mmd_rbf(C, w, X_full, gamma=1.0):
    """
    Calculates the squared Maximum Mean Discrepancy (MMD) between a weighted coreset (C, w)
    and a full dataset X_full.

    Args:
        C (np.ndarray): The coreset data points.
        w (np.ndarray): The weights corresponding to the coreset points.
        X_full (np.ndarray): The full dataset.
        gamma (float): The gamma parameter for the RBF kernel.

    Returns:
        float: The calculated squared MMD.
    """
    # Kernel matrix for the coreset points
    K_CC = rbf_kernel(C, C, gamma=gamma)
    # Kernel matrix between coreset and full dataset
    K_CX = rbf_kernel(C, X_full, gamma=gamma)
    # Kernel matrix for the full dataset (only need the mean)
    K_XX_mean = np.mean(rbf_kernel(X_full, X_full, gamma=gamma))

    # Term 1: <μ_C, μ_C>_k = w^T * K_CC * w
    term1 = w.T @ K_CC @ w

    # Term 2: -2 * <μ_C, μ_π>_k = -2 * w^T * mean(K_CX, axis=1)
    term2 = -2 * np.dot(w, np.mean(K_CX, axis=1))

    # Term 3: <μ_π, μ_π>_k
    term3 = K_XX_mean

    mmd2 = term1 + term2 + term3
    # MMD can sometimes be slightly negative due to floating point errors, so we clip at 0.
    return max(0, mmd2)


def weighted_kernel_herding(X_train, m, gamma=1.0):
    """
    Selects a coreset of size m from the training data using Weighted Kernel Herding (WKH)
    and returns the indices of the selected points.

    Args:
        X_train (np.ndarray): The full training dataset.
        m (int): The desired size of the coreset.
        gamma (float): The gamma parameter for the RBF kernel.

    Returns:
        tuple: A tuple containing the indices of the selected coreset points and their corresponding weights.
    """
    n_samples = X_train.shape[0]
    selected_indices = []
    mu_pi_embedding = np.mean(rbf_kernel(X_train, X_train, gamma=gamma), axis=0)
    current_embedding = np.zeros(n_samples)

    for k in range(m):
        search_values = current_embedding - mu_pi_embedding
        search_values[selected_indices] = np.inf
        best_x_idx = np.argmin(search_values)
        selected_indices.append(best_x_idx)

        coreset_so_far_indices = np.array(selected_indices)
        K = rbf_kernel(X_train[coreset_so_far_indices], X_train[coreset_so_far_indices], gamma=gamma)
        z = np.mean(rbf_kernel(X_train, X_train[coreset_so_far_indices], gamma=gamma), axis=0)
        weights = np.linalg.pinv(K) @ z
        current_embedding = weights @ rbf_kernel(X_train[coreset_so_far_indices], X_train, gamma=gamma)

    coreset_indices = np.array(selected_indices)
    K = rbf_kernel(X_train[coreset_indices], X_train[coreset_indices], gamma=gamma)
    z = np.mean(rbf_kernel(X_train, X_train[coreset_indices], gamma=gamma), axis=0)
    final_weights = np.linalg.pinv(K) @ z
    
    return coreset_indices, final_weights


def sequential_bayesian_quadrature(X_train, m, gamma=1.0):
    """
    Selects a coreset of size m from the training data using Sequential Bayesian Quadrature (SBQ)
    and returns the indices of the selected points.

    Args:
        X_train (np.ndarray): The full training dataset.
        m (int): The desired size of the coreset.
        gamma (float): The gamma parameter for the RBF kernel.

    Returns:
        tuple: A tuple containing the indices of the selected coreset points and their corresponding weights.
    """
    n_samples = X_train.shape[0]
    selected_indices = []
    var_pi = np.mean(rbf_kernel(X_train, X_train, gamma=gamma))
    
    for _ in range(m):
        min_variance = float('inf')
        best_x_idx = -1
        candidate_indices = np.setdiff1d(np.arange(n_samples), selected_indices)
        
        for i in candidate_indices:
            temp_indices = selected_indices + [i]
            K = rbf_kernel(X_train[temp_indices], X_train[temp_indices], gamma=gamma)
            K_inv = np.linalg.pinv(K)
            z = np.mean(rbf_kernel(X_train, X_train[temp_indices], gamma=gamma), axis=0)
            variance = var_pi - z.T @ K_inv @ z
            
            if variance < min_variance:
                min_variance = variance
                best_x_idx = i
                
        selected_indices.append(best_x_idx)

    coreset_indices = np.array(selected_indices)
    K = rbf_kernel(X_train[coreset_indices], X_train[coreset_indices], gamma=gamma)
    K_inv = np.linalg.pinv(K)
    z = np.mean(rbf_kernel(X_train, X_train[coreset_indices], gamma=gamma), axis=0)
    weights = K_inv @ z
    
    return coreset_indices, weights


def random_selection_coreset(X_train, m):
    """
    Selects a random coreset of size m and assigns uniform weights.

    Args:
        X_train (np.ndarray): The full training dataset.
        m (int): The desired size of the coreset.

    Returns:
        tuple: A tuple containing the indices of the random coreset and uniform weights.
    """
    n_samples = X_train.shape[0]
    indices = np.random.choice(n_samples, m, replace=False)
    weights = np.full(m, 1/m)
    return indices, weights


def gradient_descent_coreset(X_train, m, gamma=1.0, epochs=20, lambda_reg=1e-3, lr=0.01):
    """
    Selects a coreset using gradient descent to optimize weights for all training points.

    Args:
        X_train (np.ndarray): The full training dataset.
        m (int): The desired size of the coreset.
        gamma (float): The gamma parameter for the RBF kernel.
        epochs (int): Number of optimization epochs.
        lambda_reg (float): Regularization strength for the L0 surrogate.
        lr (float): Learning rate for the Adam optimizer.

    Returns:
        tuple: A tuple containing the indices of the selected coreset points and their final, renormalized weights.
    """
    n_samples = X_train.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pre-compute the full kernel matrix and move to device
    K = torch.from_numpy(rbf_kernel(X_train, X_train, gamma=gamma)).float().to(device)
    K_mean_row = torch.mean(K, dim=1)
    K_mean_all = torch.mean(K)

    # Initialize trainable parameters theta, such that ReLU(theta) starts at 0.1
    theta = torch.full((n_samples,), 0.1, device=device, requires_grad=True)
    optimizer = optim.Adam([theta], lr=lr)
    
    # L0 surrogate hyperparameter
    epsilon = 1e-6

    print("Optimizing coreset weights via Gradient Descent...")
    for epoch in range(epochs):
        # Enforce non-negativity
        w = torch.relu(theta)
        # Enforce sum-to-one constraint
        w_normalized = w / (torch.sum(w) + epsilon)
        
        # MMD^2 term
        mmd_sq = w_normalized @ K @ w_normalized - 2 * w_normalized @ K_mean_row + K_mean_all
        
        # L0 surrogate sparsity term
        l0_surrogate = torch.sum(torch.abs(w))
        
        # Total loss
        loss = mmd_sq + lambda_reg * l0_surrogate
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Final weights after optimization
    final_weights = torch.relu(theta).detach().cpu().numpy()
    num_nonzero = np.sum(final_weights > 1e-9)
    print("Number of non-zero weights after optimization:", num_nonzero, 'All points:', len(X_train))
    
    # Select top m indices based on final weights
    coreset_indices = np.argsort(final_weights)[-m:]
    
    # Get the weights for the coreset and re-normalize them to sum to 1
    coreset_weights = final_weights[coreset_indices]
    coreset_weights_normalized = coreset_weights / np.sum(coreset_weights)
    
    return coreset_indices, coreset_weights_normalized


def compute_mu_pi_rff_with_sampler(X, D=1000, gamma=1.0, random_state=42):
    """
    Compute RFF mean-embedding mu_pi for dataset X using sklearn RBFSampler.
    Returns mu_pi (D,), the fitted RBFSampler object, and the full Phi matrix.
    """
    sampler = RBFSampler(gamma=gamma, n_components=D, random_state=random_state)
    Phi = sampler.fit_transform(X)   # shape (n_samples, D)
    mu_pi = np.mean(Phi, axis=0)
    return mu_pi, sampler, Phi


def weighted_kernel_herding_rff(mu_pi, X_candidate, sampler, m):
    """
    Fully-corrective weighted kernel herding in RFF space using a provided RBFSampler.
    
    Args:
      mu_pi (np.ndarray): The (D,) Euclidean mean embedding of the dataset in RFF space.
      X_candidate (np.ndarray): The (n_candidates, d) candidate points to pick from (original input space).
      sampler (RBFSampler): A fitted sklearn RBFSampler used to compute mu_pi (ensures consistent mapping).
      m (int): The desired size of the coreset.
    
    Returns:
      tuple: A tuple containing:
        - np.ndarray: An array of shape (m,) with the indices into X_candidate.
        - np.ndarray: An array of shape (m,) with the final weights for the coreset points.
    """
    n_candidates, d_orig = X_candidate.shape
    D = mu_pi.shape[0] # Dimension of the RFF space

    # 1. Pre-compute the RFF features for all candidate points.
    # This maps the data from its original space to the explicit RFF feature space.
    X_candidate_rff = sampler.transform(X_candidate)

    selected_indices = []
    # The coreset's mean embedding starts at zero.
    current_embedding = np.zeros(D)

    for k in range(m):
        # 2. Find the next best point.
        # We want to find the point x that maximizes the inner product with the current residual:
        # argmax_x <mu_pi - mu_coreset, phi(x)>
        # This is equivalent to maximizing (mu_pi - current_embedding)^T @ phi(x)
        residual = mu_pi - current_embedding
        search_values = X_candidate_rff @ residual

        # Mask out already selected points so we don't pick them again.
        search_values[selected_indices] = -np.inf
        
        best_x_idx = np.argmax(search_values)
        selected_indices.append(best_x_idx)

        # 3. Fully-corrective step: Re-calculate weights for the entire current coreset.
        # This solves a linear least-squares problem to find weights 'w' that minimize
        # || mu_pi - sum(w_i * phi(x_i)) ||^2, where x_i are the selected coreset points.
        
        # Get the RFF features of the selected points so far.
        coreset_rff = X_candidate_rff[selected_indices] # Shape: (k+1, D)
        
        # The solution is w = (K_rff)^-1 @ z_rff
        # where K_rff is the Gram matrix of coreset points in RFF space.
        K_rff = coreset_rff @ coreset_rff.T
        
        # And z_rff is the vector of inner products between coreset points and the target mean embedding.
        z_rff = coreset_rff @ mu_pi
        
        # Solve the system using the pseudo-inverse for stability.
        weights = np.linalg.pinv(K_rff) @ z_rff
        
        # Update the coreset's mean embedding using the new optimal weights.
        current_embedding = weights @ coreset_rff

    # The final weights are computed during the last iteration of the loop.
    final_weights = weights
    
    return np.array(selected_indices), final_weights

# 1. Load and preprocess the MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist["data"], mnist["target"].astype(np.uint8)

np.random.seed(1421380)
subsample_indices = np.random.choice(X.shape[0], 2500, replace=False)
X_sub, y_sub = X[subsample_indices], y[subsample_indices]

scaler = StandardScaler()
X_sub_scaled = scaler.fit_transform(X_sub)

X_train, X_test, y_train, y_test = train_test_split(X_sub_scaled, y_sub, test_size=0.2, random_state=42)
print(f"Dataset prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples.")

# 2. Define coreset size and kernel parameter
m = 50
gamma = 0.001

# # 3. Coreset selection
# print(f"\nSelecting coresets of size {m}...")
# gd_indices, gd_weights = gradient_descent_coreset(X_train, m=m, gamma=gamma, lambda_reg=1e-4, lr=0.1, epochs=200)
# print(gd_weights)
print("Running Weighted Kernel Herding (WKH)...")
wkh_indices, wkh_weights = weighted_kernel_herding(X_train, m, gamma=gamma)
# print("Running Sequential Bayesian Quadrature (SBQ)...")
# sbq_indices, sbq_weights = sequential_bayesian_quadrature(X_train, m, gamma=gamma)
print("Running Random Selection...")
random_indices, random_weights = random_selection_coreset(X_train, m)
print("Coreset selection complete.")


D = 1000
print("Running Weighted Kernel Herding (WKH) with RFF...")
mu_pi, sampler, Phi_train = compute_mu_pi_rff_with_sampler(X_train, D=D, gamma=gamma, random_state=42)
selected_idx, weights = weighted_kernel_herding_rff(mu_pi, X_train.copy(), sampler, m=m)


# 4. Calculate Weighted MMD
print("\n--- Weighted MMD (Coreset vs. Full Train Set) ---")
# mmd_gd = weighted_mmd_rbf(X_train[gd_indices], gd_weights, X_train, gamma=gamma)
mmd_wkh = weighted_mmd_rbf(X_train[wkh_indices], wkh_weights, X_train, gamma=gamma)
# mmd_sbq = weighted_mmd_rbf(X_train[sbq_indices], sbq_weights, X_train, gamma=gamma)
mmd_random = weighted_mmd_rbf(X_train[random_indices], random_weights, X_train, gamma=gamma)
print(f"Random Selection MMD: {mmd_random:.4f}")
# print(f"GD Coreset MMD:       {mmd_gd:.4f}")
print(f"WKH MMD:              {mmd_wkh:.4f}")
# print(f"SBQ MMD:              {mmd_sbq:.4f}")
mmd_rff = weighted_mmd_rbf(X_train[selected_idx], weights, X_train, gamma=gamma)
print(f"RFF MMD:              {mmd_rff:.4f}")

# 5. Compare classification accuracy
print("\n--- Classification Accuracy on Test Set ---")
model_full = LogisticRegression(max_iter=1000, random_state=42)
model_full.fit(X_train, y_train)
accuracy_full = accuracy_score(y_test, model_full.predict(X_test))
print(f"Full Training Set ({X_train.shape[0]} samples): {accuracy_full:.4f}")

model_random = LogisticRegression(max_iter=1000, random_state=42)
model_random.fit(X_train[random_indices], y_train[random_indices])
accuracy_random = accuracy_score(y_test, model_random.predict(X_test))
print(f"Random Coreset    ({m} samples):     {accuracy_random:.4f}")

# model_gd = LogisticRegression(max_iter=1000, random_state=42)
# model_gd.fit(X_train[gd_indices], y_train[gd_indices])
# accuracy_gd = accuracy_score(y_test, model_gd.predict(X_test))
# print(f"GD Coreset        ({m} samples):     {accuracy_gd:.4f}")

model_wkh = LogisticRegression(max_iter=1000, random_state=42)
model_wkh.fit(X_train[wkh_indices], y_train[wkh_indices])
accuracy_wkh = accuracy_score(y_test, model_wkh.predict(X_test))
print(f"WKH Coreset       ({m} samples):     {accuracy_wkh:.4f}")

# model_sbq = LogisticRegression(max_iter=1000, random_state=42)
# model_sbq.fit(X_train[sbq_indices], y_train[sbq_indices])
# accuracy_sbq = accuracy_score(y_test, model_sbq.predict(X_test))
# print(f"SBQ Coreset       ({m} samples):     {accuracy_sbq:.4f}")

model_rff = LogisticRegression(max_iter=1000, random_state=42)
model_rff.fit(X_train[selected_idx], y_train[selected_idx])
accuracy_rff = accuracy_score(y_test, model_rff.predict(X_test))
print(f"RFF Coreset       ({m} samples):     {accuracy_rff:.4f}")
