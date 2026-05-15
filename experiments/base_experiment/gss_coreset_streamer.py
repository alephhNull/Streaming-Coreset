import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Tuple, List
import torch.nn as nn
import torch.optim as optim

class AbstractStreamingCoreset(ABC):
    @abstractmethod
    def process_batch(self, X_batch_np: np.ndarray, y_batch_np: np.ndarray, batch_idx: int) -> None:
        pass
    @abstractmethod
    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        pass
    @abstractmethod
    def print_coreset_provenance(self) -> None:
        pass


class GSSStreamer(AbstractStreamingCoreset):
    """
    Gradient-based Sample Selection (GSS) - Greedy Variant.
    
    Includes an active training loop for the surrogate model during the stream
    to ensure the parameter gradients reflect the true, evolving loss landscape.
    """
    def __init__(
        self,
        buffer_capacity: int,
        surrogate_model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        num_random_samples: int = 10,
        rehearsal_iters: int = 1
    ):
        self.buffer_capacity = buffer_capacity
        self.surrogate_model = surrogate_model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.device = device
        
        # GSS Hyperparameters
        self.num_random_samples = num_random_samples
        self.rehearsal_iters = rehearsal_iters
        
        # Buffer State
        self.buffer_X: List[torch.Tensor] = []
        self.buffer_y: List[torch.Tensor] = []
        self.buffer_scores: List[float] = []
        self.buffer_provenance: List[Tuple[int, int]] = []

    def _get_parameter_gradients(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes flattened parameter gradients for a single sample."""
        self.surrogate_model.zero_grad()
        out = self.surrogate_model(x.unsqueeze(0))
        loss = self.criterion(out, y.unsqueeze(0))
        loss.backward()
        
        grads = []
        for param in self.surrogate_model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        return torch.cat(grads)

    def process_batch(self, X_batch_np: np.ndarray, y_batch_np: np.ndarray, batch_idx: int) -> None:
        X_batch = torch.tensor(X_batch_np, dtype=torch.float32).to(self.device)
        y_batch = torch.tensor(y_batch_np, dtype=torch.long).to(self.device)
        
        # 1. GSS Buffer Selection (Sample-by-sample)
        self.surrogate_model.eval() # Ensure deterministic evaluation for scoring
        for local_idx in range(len(X_batch)):
            x_i = X_batch[local_idx]
            y_i = y_batch[local_idx]
            
            g = self._get_parameter_gradients(x_i, y_i)
            g_norm = torch.norm(g)
            
            if g_norm < 1e-8:
                continue
                
            c = 1.0 # Default score
            
            if len(self.buffer_X) > 0:
                n_samples = min(self.num_random_samples, len(self.buffer_X))
                rand_indices = np.random.choice(len(self.buffer_X), n_samples, replace=False)
                max_cosine_sim = -1.0
                
                for idx in rand_indices:
                    G_i = self._get_parameter_gradients(self.buffer_X[idx], self.buffer_y[idx])
                    G_i_norm = torch.norm(G_i)
                    if G_i_norm > 1e-8:
                        cos_sim = torch.dot(g, G_i) / (g_norm * G_i_norm)
                        max_cosine_sim = max(max_cosine_sim, cos_sim.item())
                
                c = max_cosine_sim + 1.0 
            
            if len(self.buffer_X) < self.buffer_capacity:
                self.buffer_X.append(x_i.detach().clone())
                self.buffer_y.append(y_i.detach().clone())
                self.buffer_scores.append(c)
                self.buffer_provenance.append((batch_idx, local_idx))
            else:
                if c < 1.0: 
                    scores_tensor = torch.tensor(self.buffer_scores, dtype=torch.float32)
                    sum_scores = torch.sum(scores_tensor)
                    
                    P_i = np.ones(len(self.buffer_X)) / len(self.buffer_X) if sum_scores == 0 else (scores_tensor / sum_scores).numpy()
                    
                    candidate_idx = np.random.choice(len(self.buffer_X), p=P_i)
                    C_i = self.buffer_scores[candidate_idx]
                    
                    denominator = C_i + c
                    p_replace = C_i / denominator if denominator > 1e-8 else 0.0
                    
                    if np.random.uniform(0, 1) < p_replace:
                        self.buffer_X[candidate_idx] = x_i.detach().clone()
                        self.buffer_y[candidate_idx] = y_i.detach().clone()
                        self.buffer_scores[candidate_idx] = c
                        self.buffer_provenance[candidate_idx] = (batch_idx, local_idx)

        # 2. Surrogate Model Training (The Missing Piece)
        self._train_surrogate(X_batch, y_batch)

    def _train_surrogate(self, X_batch: torch.Tensor, y_batch: torch.Tensor) -> None:
        """Trains the surrogate on the incoming batch and rehearses from the buffer."""
        self.surrogate_model.train()
        
        for _ in range(self.rehearsal_iters):
            # Step A: Train on new batch
            self.optimizer.zero_grad()
            loss = self.criterion(self.surrogate_model(X_batch), y_batch)
            loss.backward()
            self.optimizer.step()
            
            # Step B: Rehearse on current buffer to prevent forgetting in the surrogate
            if len(self.buffer_X) > 0:
                buffer_batch_size = min(len(X_batch), len(self.buffer_X))
                rand_indices = np.random.choice(len(self.buffer_X), buffer_batch_size, replace=False)
                
                buf_X_batch = torch.stack([self.buffer_X[i] for i in rand_indices])
                buf_y_batch = torch.stack([self.buffer_y[i] for i in rand_indices])
                
                self.optimizer.zero_grad()
                loss_buf = self.criterion(self.surrogate_model(buf_X_batch), buf_y_batch)
                loss_buf.backward()
                self.optimizer.step()

    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        if not self.buffer_X:
            return np.array([]), np.array([]), []
            
        X_core = torch.stack(self.buffer_X).cpu().numpy()
        y_core = torch.stack(self.buffer_y).cpu().numpy()
        return X_core, y_core, self.buffer_provenance

    def print_coreset_provenance(self) -> None:
        print(f"--- GSS Coreset Provenance (Capacity {self.buffer_capacity}) ---")
        for i, (batch_idx, local_idx) in enumerate(self.buffer_provenance):
            print(f"Slot {i:03d} -> Source Batch {batch_idx:03d}, Local Index: {local_idx:03d}, Score: {self.buffer_scores[i]:.3f}")