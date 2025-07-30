import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from streamers.abstract_streamer import AbstractStreamingCoreset

class SimpleNet(nn.Module):
    """
    Two-hidden-layer MLP with 100 units each, as per GSS paper for MNIST.
    """
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class SimpleNet(nn.Module):
    """
    Two-hidden-layer MLP with 100 units each, as per GSS paper for MNIST.
    """
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class GSSStreamer(AbstractStreamingCoreset):
    """
    GSS variant with a maximally expensive buffer update process.
    """
    def __init__(self,
                 coreset_size: int,
                 buffer_capacity: int,
                 input_dim: int,
                 num_classes: int,
                 device: torch.device,
                 batch_size: int,
                 random_seed: int = None,
                 n_samples_for_score: int = 10):
        self.coreset_size = coreset_size
        self.buffer_capacity = buffer_capacity
        self.n_samples_for_score = n_samples_for_score
        self.model = SimpleNet(input_dim, num_classes).to(device)
        self.batch_size = batch_size
        self.device = device
        
        self.buffer = None
        self.labels = []
        self.provenance = []
        self.scores = []

        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

    def _get_individual_gradient(self, X: np.ndarray, y: np.ndarray) -> torch.Tensor:
        """
        Compute gradient for a SINGLE sample.
        """
        self.model.zero_grad()
        inputs = torch.from_numpy(X).to(self.device).float()
        labels = torch.from_numpy(y).to(self.device).long()
        
        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        
        grads = []
        for p in self.model.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
        
        return torch.cat(grads)

    def process_batch(self, X_batch_np: np.ndarray, y_batch_np: np.ndarray, batch_idx: int) -> None:
        B, D = X_batch_np.shape
        if self.buffer is None:
            self.buffer = np.zeros((0, D), dtype=float)

        # Process new samples one by one
        for i in range(B):
            x = X_batch_np[i:i+1]
            y = y_batch_np[i:i+1]
            
            # Gradient for the new sample (1 backprop)
            g_new = self._get_individual_gradient(x, y)

            max_cos = -1.0
            if len(self.scores) > 0:
                num_buffer_samples = len(self.labels)
                n_samples = min(self.n_samples_for_score, num_buffer_samples)
                idxs = np.random.choice(num_buffer_samples, n_samples, replace=False)
                
                # --- START OF MAXIMALLY EXPENSIVE OPERATION ---
                # For each sampled buffer point, calculate its individual gradient.
                # This involves n_samples_for_score separate backpropagations.
                buffer_grads = []
                for buf_idx in idxs:
                    x_buf = self.buffer[buf_idx:buf_idx+1]
                    y_buf = np.array([self.labels[buf_idx]])
                    g_buf = self._get_individual_gradient(x_buf, y_buf)
                    buffer_grads.append(g_buf)
                
                if buffer_grads:
                    g_buf_stack = torch.stack(buffer_grads)
                    cos_similarities = F.cosine_similarity(g_new.unsqueeze(0), g_buf_stack)
                    max_cos = cos_similarities.max().item()
                # --- END OF MAXIMALLY EXPENSIVE OPERATION ---

            c_new = max_cos + 1.0

            if len(self.scores) < self.buffer_capacity:
                self.buffer = np.vstack([self.buffer, x])
                self.labels.append(int(y[0]))
                self.provenance.append((batch_idx, i))
                self.scores.append(c_new)
            else:
                scores_arr = np.array(self.scores)
                probs = scores_arr / scores_arr.sum()
                cand_idx = np.random.choice(len(self.scores), p=probs)
                c_i = self.scores[cand_idx]
                
                if np.random.rand() < c_i / (c_i + c_new):
                    self.buffer[cand_idx] = x
                    self.labels[cand_idx] = int(y[0])
                    self.provenance[cand_idx] = (batch_idx, i)
                    self.scores[cand_idx] = c_new

    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        num_buffer_samples = self.buffer.shape[0]
        if num_buffer_samples > self.coreset_size:
            indices = np.random.choice(num_buffer_samples, self.coreset_size, replace=False)
        else:
            indices = np.arange(num_buffer_samples)

        final_buffer = self.buffer[indices]
        final_provenance = [self.provenance[i] for i in indices]
        weights = np.ones(len(indices)) / len(indices)
        flat_indices = np.array([b * self.batch_size + j for b, j in final_provenance])
        
        return flat_indices, weights, final_provenance
    
    
    def print_coreset_provenance(self) -> None:
        """Prints the provenance of the final coreset samples."""
        flat_indices, weights, provenance_list = self.get_final_coreset()
        print("Coreset Provenance:")
        for i, (batch_idx, local_idx) in enumerate(provenance_list):
            print(f"- Point {i}: GlobalIdx={flat_indices[i]}, Batch={batch_idx}, LocalIdx={local_idx}, Weight={weights[i]:.4f}")