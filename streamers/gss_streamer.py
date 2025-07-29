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


class GSSStreamer(AbstractStreamingCoreset):
    """
    Greedy variant of Gradient-based Sample Selection (GSS) for streaming coreset selection.
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

        # Initialize empty buffer of shape (0, D) after seeing first batch
        self.buffer = None
        self.labels = []             # true labels
        self.provenance = []         # (batch_idx, local_idx)
        self.scores = []             # c_i = max_cos_sim + 1

        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

    def _get_gradients(self, X: np.ndarray, y: np.ndarray) -> torch.Tensor:
        """
        Compute gradient feature vector for a mini-batch using true labels.
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
        # init buffer on first call
        if self.buffer is None:
            self.buffer = np.zeros((0, D), dtype=float)

        # append new batch one sample at a time
        for i in range(B):
            x = X_batch_np[i:i+1]
            y = y_batch_np[i:i+1]

            # compute c_new: max_cos + 1
            if len(self.scores) > 0:
                idxs = np.random.choice(len(self.scores),
                                        min(self.n_samples_for_score, len(self.scores)),
                                        replace=False)
                X_buf = self.buffer[idxs]
                y_buf = np.array(self.labels)[idxs]
                g_buf = self._get_gradients(X_buf, y_buf)
                g_new = self._get_gradients(x, y)

                if g_buf.dim() == 1:
                    g_buf = g_buf.unsqueeze(0)
                cos = F.cosine_similarity(g_new.unsqueeze(0), g_buf)
                max_cos = cos.max().item()
            else:
                max_cos = -1.0
                g_new = self._get_gradients(x, y)

            c_new = max_cos + 1.0

            if len(self.scores) < self.buffer_capacity:
                # fill buffer
                self.buffer = np.vstack([self.buffer, x])
                self.labels.append(int(y[0]))
                self.provenance.append((batch_idx, i))
                self.scores.append(c_new)
            else:
                # consider replacement only if c_new < 1 (max_cos < 0)
                if c_new < 1.0:
                    scores_arr = np.array(self.scores)
                    probs = scores_arr / scores_arr.sum()
                    cand = np.random.choice(len(self.scores), p=probs)

                    c_i = self.scores[cand]
                    if np.random.rand() < c_i / (c_i + c_new):
                        self.buffer[cand] = x
                        self.labels[cand] = int(y[0])
                        self.provenance[cand] = (batch_idx, i)
                        self.scores[cand] = c_new

    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        N = self.buffer.shape[0]
        if N > self.coreset_size:
            idx = np.random.choice(N, self.coreset_size, replace=False)
        else:
            idx = np.arange(N)

        final_buf = self.buffer[idx]
        final_prov = [self.provenance[i] for i in idx]
        weights = np.ones(len(idx)) / len(idx)
        flat = np.array([b * self.batch_size + j for b, j in final_prov])
        return flat, weights, final_prov

    def print_coreset_provenance(self) -> None:
        flat, w, prov = self.get_final_coreset()
        print("Coreset Provenance:")
        for i, (b, j) in enumerate(prov):
            print(f"- Point {i}: GlobalIdx={flat[i]}, Batch={b}, LocalIdx={j}, Weight={w[i]:.4f}")

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

# if __name__ == "__main__":
#     # Example usage
#     INPUT_DIM = 10
#     NUM_CLASSES = 2
#     CORESET_SIZE = 20
#     BUFFER_CAPACITY = 100
#     N_SAMPLES = 10
#     BATCH_SIZE = 10
#     NUM_BATCHES = 25

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = SimpleNet(INPUT_DIM, NUM_CLASSES).to(device)
#     gss = GSSGreedyStreamer(CORESET_SIZE,
#                              BUFFER_CAPACITY,
#                              N_SAMPLES,
#                              model,
#                              BATCH_SIZE,
#                              random_seed=42)

#     for batch_idx in range(NUM_BATCHES):
#         X = np.random.rand(BATCH_SIZE, INPUT_DIM).astype(float)
#         y = np.random.randint(0, NUM_CLASSES, size=(BATCH_SIZE,))
#         gss.process_batch(X, y, batch_idx)

#     gss.print_coreset_provenance()
