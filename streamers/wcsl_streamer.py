from typing import Tuple, List
import numpy as np
import torch
import torch.optim as optim
from streamers.abstract_streamer import AbstractStreamingCoreset
from torch.autograd import grad

class WCSLStreamer(AbstractStreamingCoreset):
    """
    Implements the Wasserstein Coreset via Sinkhorn Loss (WCSL) for streaming data.
    """

    def __init__(self, target_coreset_size: int, buffer_capacity: int, batch_size: int, random_seed: int,
                 lambda_reg: float = 1.0, learning_rate: float = 0.01, n_iterations: int = 100, device: str = 'cuda'):
        """
        Initializes the WCSLStreamer.

        Args:
            target_coreset_size (int): The desired size of the coreset (m).
            buffer_capacity (int): The maximum number of data points to hold in the buffer.
            batch_size (int): The number of data points in each incoming batch.
            lambda_reg (float): The regularization parameter for the Sinkhorn loss.
            learning_rate (float): The learning rate for the coreset point optimization.
            n_iterations (int): The number of iterations for the WCSL algorithm.
            device (str): The device to run the computations on ('cpu' or 'cuda').
        """
        self.target_coreset_size = target_coreset_size
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.device = device

        self.buffer = torch.empty((0, 0), dtype=torch.float32, device=self.device)
        self.provenance_buffer = []
        self.is_processing = False
        
        self.final_coreset_indices = None
        self.final_weights = None
        self.final_provenance = None

        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

    def process_batch(self, X_batch_np: np.ndarray, y_batch_np: np.ndarray, batch_idx: int) -> None:
        if self.is_processing:
            print(f"Skipping batch {batch_idx} as a coreset selection is already in progress.")
            return

        X_batch = torch.from_numpy(X_batch_np).to(self.device)
        
        if self.buffer.numel() == 0:
            self.buffer = torch.empty((0, X_batch.shape[1]), dtype=torch.float32, device=self.device)

        self.buffer = torch.cat([self.buffer, X_batch], dim=0)
        self.provenance_buffer.extend([(batch_idx, i) for i in range(X_batch.shape[0])])

        if self.buffer.shape[0] > self.buffer_capacity:
            self.is_processing = True
            print(f"Buffer capacity exceeded. Running coreset selection on {self.buffer.shape[0]} points.")
            self._run_coreset_selection()
            self.is_processing = False

    def _run_coreset_selection(self):
        """
        Runs the WCSL algorithm to select a coreset from the buffer.
        """
        # Algorithm 1: Wasserstein Coreset via Sinkhorn Loss (WCSL)
        
        # 1. Initialize coreset by random sampling
        initial_indices = np.random.choice(self.buffer.shape[0], self.target_coreset_size, replace=False)
        coreset_points = self.buffer[initial_indices].clone().detach().requires_grad_(True)
        
        optimizer = optim.Adam([coreset_points], lr=self.learning_rate)

        # 2. Iteratively update coreset points
        for _ in range(self.n_iterations):
            optimizer.zero_grad()
            
            # Resample a mini-batch from the buffer
            mini_batch_indices = np.random.choice(self.buffer.shape[0], self.target_coreset_size, replace=False)
            mini_batch = self.buffer[mini_batch_indices]
            
            # Compute distance matrix
            cost_matrix = torch.cdist(coreset_points, mini_batch, p=2)**2
            
            # Compute Sinkhorn loss and its gradient with respect to the cost matrix
            _, grad_M = self._compute_sinkhorn_loss_and_grad(cost_matrix)
            
            # Backpropagate to get the gradient with respect to coreset points
            cost_matrix.backward(grad_M)
            
            optimizer.step()

        # Select the points from the buffer closest to the optimized coreset points
        final_dist_matrix = torch.cdist(coreset_points.detach(), self.buffer, p=2)**2
        final_indices = torch.argmin(final_dist_matrix, dim=1).cpu().numpy()
        
        # Update buffer and provenance
        self.buffer = self.buffer[final_indices]
        self.provenance_buffer = [self.provenance_buffer[i] for i in final_indices]

    def _compute_sinkhorn_loss_and_grad(self, M: torch.Tensor, max_iter: int = 10):
        """
        Computes the Sinkhorn loss S_lambda and its analytic gradient w.r.t. the cost matrix M
        using the dual L-BFGS solver and Theorem 4.1 from 'Wasserstein Coreset via Sinkhorn Loss'.

        Args:
            M (torch.Tensor): Cost matrix of shape (n, m).
            lambda_reg (float): The regularization parameter lambda (> 0).
            max_iter (int): Maximum L-BFGS iterations for the dual problem.

        Returns:
            loss (torch.Tensor): Scalar Sinkhorn loss.
            grad_M (torch.Tensor): Analytic gradient, same shape as M.
        """
        device = M.device
        n, m = M.shape

        # Uniform marginals
        a = torch.full((n,), 1.0 / n, device=device)
        b = torch.full((m,), 1.0 / m, device=device)
        a_inv = 1.0 / a

        # --------------------
        # Helper: stable logsumexp
        # --------------------
        def logsumexp(z: torch.Tensor, dim: int):
            m_val, _ = z.max(dim=dim, keepdim=True)
            return (m_val + (z - m_val).exp().sum(dim=dim, keepdim=True).log()).squeeze(dim)

        # --------------------
        # Dual variables -> alpha
        # --------------------
        lambda_reg = float(self.lambda_reg)       # <— make scalar
        assert M.dim() == 2, f"M must be 2-D, got {M.shape}"
        n, m = M.shape
        # … define a, b, a_inv …

        def alpha_from_beta(beta: torch.Tensor):
            beta = beta.view(-1)                  # (m,)
            # now beta[None,:] is (1,m), M is (n,m) → result is (n,m)
            z = lambda_reg * (beta.unsqueeze(0) - M)
            # logsumexp over columns gives (n,), as desired
            lse = logsumexp(z, dim=1)            # (n,)
            return (torch.log(a).unsqueeze(1).squeeze(1) - lse) / lambda_reg  # (n,)

        # --------------------
        # Objective f(β) to minimize
        # --------------------
        def f_beta(beta: torch.Tensor):
            alpha = alpha_from_beta(beta)
            return -(alpha @ a + beta @ b - 1.0 / self.lambda_reg)

        # Initialize β (m-1 free variables, last fixed to 0)
        beta_free = torch.zeros(m - 1, device=device, requires_grad=True)

        # L-BFGS optimizer
        optimizer = torch.optim.LBFGS(
            [beta_free],
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-9
        )

        # Closure for L-BFGS
        def closure():
            optimizer.zero_grad()
            beta = torch.cat([beta_free, torch.tensor([0.0], device=device)])
            loss_dual = f_beta(beta)
            # ensure scalar
            loss_scalar = loss_dual if loss_dual.dim() == 0 else loss_dual.sum()
            loss_scalar.backward(retain_graph=True)
            return loss_scalar

        optimizer.step(closure)

        # After L-BFGS: recover β*, α*, and the optimal plan T*
        with torch.no_grad():
            # recover beta (shape should be (m,))
            beta = torch.cat([beta_free, torch.tensor([0.0], device=device)])

            # make sure alpha is [n,] and beta is [m,]
            alpha = alpha_from_beta(beta)          # (n,)
            beta = beta.view(1, -1)                # (1, m)

            # now alpha[:,None] is (n,1), beta is (1,m) → result is (n,m)
            T = torch.exp(self.lambda_reg * (alpha.view(-1, 1) + beta - M))

            loss = torch.sum(T * M)

        # --------------------
        # Analytic gradient (Theorem 4.1)
        # ∇_M S = T + λ (s_u ⊕ s_v - M) ⊙ T
        # --------------------
        # Split off last column for reduced variables
        T_tilde = T[:, :-1]          # (n, m-1)
        b_tilde = b[:-1]             # (m-1,)

        # μ_r and μ̃_c
        mu_r = (M * T).sum(dim=1)                # (n,)
        mu_c_tilde = (M[:, :-1] * T_tilde).sum(dim=0)  # (m-1,)

        # Build D = diag(b̃) - T̃^T diag(a^{-1}) T̃
        D = torch.diag(b_tilde) - T_tilde.T @ (a_inv.unsqueeze(1) * T_tilde)  # (m-1, m-1)
        eps = 1e-8
        D_reg = D + eps * torch.eye(D.shape[0], device=D.device)

        # Solve for s̃_v: D s̃_v = μ̃_c - T̃^T (a^{-1} ⊙ μ_r)
        rhs = mu_c_tilde - T_tilde.T @ (a_inv * mu_r)
        s_tilde_v = torch.linalg.solve(D_reg, rhs)  # (m-1,)

        # Compute s_u and full s_v
        s_u = a_inv * (mu_r - T_tilde @ s_tilde_v)     # (n,)
        s_v = torch.cat([s_tilde_v, torch.zeros(1, device=device)])  # (m,)

        # Broadcast-sum (s_u ⊕ s_v - M)
        su_sv_minus_M = s_u.unsqueeze(1) + s_v.unsqueeze(0) - M

        # Final gradient
        grad_M = T + self.lambda_reg * su_sv_minus_M * T

        return loss, grad_M

    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        if not self.is_processing and self.buffer.shape[0] > self.target_coreset_size:
            print("Running final coreset selection on remaining buffer.")
            self._run_coreset_selection()

        self.final_provenance = self.provenance_buffer
        self.final_weights = np.ones(len(self.final_provenance)) / len(self.final_provenance)
        self.final_coreset_indices = np.array([p[0] * self.batch_size + p[1] for p in self.final_provenance])
        
        return self.final_coreset_indices, self.final_weights, self.final_provenance

    def print_coreset_provenance(self) -> None:
        if self.final_provenance is None:
            self.get_final_coreset()
            
        print("\nCoreset Provenance:")
        for i, (batch_idx, local_idx) in enumerate(self.final_provenance):
            print(f"  Point {i}: Batch {batch_idx}, Index in Batch {local_idx}, Weight: {self.final_weights[i]:.4f}")
