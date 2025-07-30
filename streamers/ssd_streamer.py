import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Dict
from abc import ABC, abstractmethod

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

class SummarizingModelGeneric(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        logits = self.classifier(h)
        return logits, h

class SSDStreamerGeneric(AbstractStreamingCoreset):
    def __init__(
        self,
        buffer_capacity: int,
        target_coreset_size: int,
        num_classes: int,
        feature_dim: int,
        random_seed: int,
        summarizing_lr: float = 1e-2,
        gamma: float = 1.0,
        summarizing_interval: int = 6,
        inner_lr: float = 0.1,
        synth_steps: int = 5
    ):
        self.buffer_capacity = buffer_capacity
        self.target_coreset_size = target_coreset_size
        self.num_classes = num_classes
        self.random_seed=random_seed
        self.gamma = gamma
        self.interval = summarizing_interval
        self.inner_lr = inner_lr
        self.synth_steps = synth_steps

        self.k_synth_per_class = buffer_capacity // num_classes
        self.num_synth_slots = self.k_synth_per_class * num_classes
        self.num_real_slots = buffer_capacity - self.num_synth_slots

        self.synth_memory: Dict[int, List[torch.Tensor]] = {c: [] for c in range(num_classes)}
        self.synth_provenance: Dict[int, List[Tuple[int,int]]] = {c: [] for c in range(num_classes)}

        self.real_memory = torch.empty((0, feature_dim), dtype=torch.float32)
        self.real_labels: List[int] = []
        self.real_provenance: List[Tuple[int,int]] = []

        self.model = SummarizingModelGeneric(feature_dim, num_classes)
        self.optimizer = optim.SGD(self.model.parameters(), lr=summarizing_lr, momentum=0.9)
        self.loss_fn = nn.CrossEntropyLoss()

        self.total_seen = 0
        self.batches_seen = 0

        if random_seed:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

    def _update_summarizing_model(self, X: torch.Tensor, y: torch.Tensor):
        self.model.train()
        self.optimizer.zero_grad()
        logits, _ = self.model(X)
        loss = self.loss_fn(logits, y)
        if len(self.real_memory) > 0:
            m = min(len(self.real_memory), X.size(0))
            idx = np.random.choice(len(self.real_memory), m, replace=False)
            mem_X = self.real_memory[idx]
            mem_y = torch.LongTensor([self.real_labels[i] for i in idx])
            mem_logits, _ = self.model(mem_X)
            loss = loss + self.loss_fn(mem_logits, mem_y)
        loss.backward()
        self.optimizer.step()

    def _summarize_batch(self, X: torch.Tensor, y: torch.Tensor):
        self._update_summarizing_model(X, y)
        self.model.eval()

        for c in torch.unique(y).tolist():
            c = int(c)
            real_feats = X[y == c]
            if real_feats.size(0) == 0:
                continue

            # multiple inner steps per synth
            for step in range(self.synth_steps):
                for slot_i in range(len(self.synth_memory[c])):
                    synth = self.synth_memory[c][slot_i]
                    synth_var = synth.clone().detach().requires_grad_(True)

                    real_logits, _ = self.model(real_feats)
                    real_loss = self.loss_fn(real_logits, torch.full((len(real_feats),), c, dtype=torch.long))
                    real_grads = torch.autograd.grad(real_loss, self.model.parameters(), retain_graph=True)
                    real_vec = torch.cat([g.flatten() for g in real_grads])

                    synth_logits, _ = self.model(synth_var.unsqueeze(0))
                    synth_loss = self.loss_fn(synth_logits, torch.tensor([c]))
                    synth_grads = torch.autograd.grad(synth_loss, self.model.parameters(), create_graph=True)
                    synth_vec = torch.cat([g.flatten() for g in synth_grads])

                    grad_loss = torch.norm(real_vec - synth_vec, p=2)

                    anchors = [self.synth_memory[c2][0] for c2 in range(self.num_classes)
                               if c2 != c and self.synth_memory[c2]]
                    rel_loss = torch.tensor(0.0, device=X.device)
                    if anchors:
                        anchor_batch = torch.stack(anchors)
                        _, real_h = self.model(real_feats)
                        _, synth_h = self.model(synth_var.unsqueeze(0))
                        _, anchor_h = self.model(anchor_batch)
                        real_dist = torch.norm(real_h.mean(0) - anchor_h, dim=1).mean()
                        synth_dist = torch.norm(synth_h.squeeze(0) - anchor_h, dim=1).mean()
                        rel_loss = torch.norm(real_dist - synth_dist, p=2)

                    total_loss = grad_loss + self.gamma * rel_loss
                    synth_grad = torch.autograd.grad(total_loss, synth_var)[0]
                    updated = (synth_var - self.inner_lr * synth_grad).detach()
                    self.synth_memory[c][slot_i] = updated

    def process_batch(self, X_np: np.ndarray, y_np: np.ndarray, batch_idx: int):
        X = torch.from_numpy(X_np).float()
        y = torch.from_numpy(y_np).long()
        B = X.size(0)

        for i in range(B):
            self.total_seen += 1
            xi = X[i].unsqueeze(0)
            yi = int(y[i].item())

            if len(self.synth_memory[yi]) < self.k_synth_per_class:
                self.synth_memory[yi].append(xi.squeeze(0).detach())
                self.synth_provenance[yi].append((batch_idx, i))
                continue

            if len(self.real_memory) < self.num_real_slots:
                self.real_memory = torch.cat([self.real_memory, xi], dim=0)
                self.real_labels.append(yi)
                self.real_provenance.append((batch_idx, i))
            else:
                j = np.random.randint(0, self.total_seen)
                if j < self.num_real_slots:
                    self.real_memory[j] = xi
                    self.real_labels[j] = yi
                    self.real_provenance[j] = (batch_idx, i)

        if (self.batches_seen + 1) % self.interval == 0:
            self._summarize_batch(X, y)
        self.batches_seen += 1

    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        all_samples, all_prov = [], []
        for c in range(self.num_classes):
            for i, synth in enumerate(self.synth_memory[c]):
                all_samples.append(synth.cpu().numpy())
                all_prov.append(self.synth_provenance[c][i])
        for i in range(len(self.real_memory)):
            all_samples.append(self.real_memory[i].cpu().numpy())
            all_prov.append(self.real_provenance[i])

        M = len(all_samples)
        if M == 0:
            return np.array([]), np.array([]), []

        if M > self.target_coreset_size:
            keep = np.random.choice(M, self.target_coreset_size, replace=False)
            all_samples = [all_samples[i] for i in keep]
            all_prov = [all_prov[i] for i in keep]

        weights = np.ones(len(all_samples)) / len(all_samples)
        indices = np.arange(len(all_samples))
        return indices, weights, all_prov

    def print_coreset_provenance(self) -> None:
        idxs, weights, prov = self.get_final_coreset()
        print("---- Coreset Provenance ----")
        for i, ((b, l), w) in enumerate(zip(prov, weights)):
            typ = 'Synth' if i < self.num_synth_slots else 'Real'
            print(f"{i:03d} | {typ:7s} | batch={b:03d}, idx={l:03d} | w={w:.4f}")
        print("-----------------------------")
