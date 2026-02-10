import numpy as np
import math
import matplotlib.pyplot as plt
from collections import Counter
from typing import List
import random

import sys 
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from streamers.mirror_descent_streamer import MirrorDescentHerdingStreamer
from streamers.caratheodory_streamer import CaratheoderyStreamingCoreset
from streamers.amh_streamer import AdaptiveKernelHerdingStreamer
from streamers.rff_wkh_streamer import WKHStreamingCoreset
from streamers.kernel_herding_streamer import KernelHerdingStreamer
from dataloaders import load_dataset

# Data loading (torchvision)
try:
    import torch
    from torchvision import datasets, transforms
except Exception as e:
    raise ImportError("torch/torchvision required to load MNIST. Install them or provide the data manually. " + str(e))

from sklearn.kernel_approximation import RBFSampler

# ----------------- Experiment parameters -----------------
SEED = 0
np.random.seed(SEED)
random.seed(SEED)

NUM_PER_CLASS = 250                # 250 * 10 = 2500 total
TOTAL = NUM_PER_CLASS * 10
BATCH_SIZE = 50                    # number of images per streaming batch
N_CLASSES = 10

# Buffer / coreset settings (the user suggested M=16)
M = 40                             # final coreset size and buffer capacity for this experiment
BUFFER_CAPACITY = M
CORESET_SIZE = M

# RFF sampler params
RFF_DIM = 1024                      # number of RFF components (adjustable)
RFF_GAMMA = 0.001                   # RBF kernel gamma

# Removal schedule: remove few items per iterative reopt to see gradual behavior
REMOVAL_BATCH_SIZE = 1

# checkpoints to save visualizations (indices of batches)
SAVE_EVERY = 1                      # save snapshot after every batch (set >1 to reduce outputs)
SNAPSHOT_DIR = "snapshots_mnist_stream"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)



# ----------------- Utility functions -----------------

def sample_stratified_mnist_subset_embedded(num_per_class=NUM_PER_CLASS, seed=SEED):
    # download if necessary
    train_loader, val_loader, X_train, X_val, y_train, y_val = load_dataset('mnist', 50000, 50000, seed, embedding='resnet18', embed_dim=None, device='cpu')
    # mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    # X_all = mnist.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
    # y_all = mnist.targets.numpy().astype(int)

    X_all = X_train
    y_all = y_train

    idxs = []
    rng = np.random.RandomState(seed)
    per_class = {
        0: NUM_PER_CLASS,
        1: NUM_PER_CLASS * 2,
        2: NUM_PER_CLASS,
        3: NUM_PER_CLASS * 3,
        4: NUM_PER_CLASS,
        5: NUM_PER_CLASS * 5,
        6: NUM_PER_CLASS,
        7: NUM_PER_CLASS* 3,
        8: NUM_PER_CLASS * 2,
        9: NUM_PER_CLASS * 1,
    }
    for c in range(N_CLASSES):
        class_idx = np.where(y_all == c)[0]
        chosen = rng.choice(class_idx, size=per_class[c], replace=False)
        idxs.append(chosen)

    idxs = np.concatenate(idxs)
    X = X_all[idxs]
    y = y_all[idxs]

    # order by label so stream is non-i.i.d.: all 0s, then all 1s, ...
    order = np.argsort(y)
    # order = np.random.choice(order, len(order), replace=False)
    X = X[order]
    y = y[order]
    return X, y


def sample_stratified_mnist_subset(num_per_class=NUM_PER_CLASS, seed=SEED):
    # download if necessary
    mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    X_all = mnist.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
    y_all = mnist.targets.numpy().astype(int)

    idxs = []
    rng = np.random.RandomState(seed)
    for c in range(N_CLASSES):
        class_idx = np.where(y_all == c)[0]
        chosen = rng.choice(class_idx, size=num_per_class, replace=False)
        idxs.append(chosen)

    idxs = np.concatenate(idxs)
    X = X_all[idxs]
    y = y_all[idxs]

    # order by label so stream is non-i.i.d.: all 0s, then all 1s, ...
    order = np.argsort(y)
    X = X[order]
    y = y[order]
    return X, y


def make_batches(X: np.ndarray, y: np.ndarray, batch_size: int) -> List[tuple]:
    n = X.shape[0]
    batches = []
    for i in range(0, n, batch_size):
        batches.append((X[i : i + batch_size], y[i : i + batch_size], i // batch_size))
    return batches


def visualize_buffer_images(buffer_X: np.ndarray, buffer_y: np.ndarray, provenance: List[tuple], save_path: str, title: str = ""):
    # show up to M images in a grid, annotate with label and provenance
    k = buffer_X.shape[0]
    cols = min(8, k)
    rows = math.ceil(k / cols) if k > 0 else 1
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8))
    axs = np.array(axs).reshape(-1)
    for i in range(rows * cols):
        ax = axs[i]
        ax.axis("off")
        if i < k:
            img = buffer_X[i].reshape(28, 28)
            ax.imshow(img, cmap="gray", interpolation="nearest")
            label = int(buffer_y[i])
            prov = provenance[i]
            ax.set_title(f"{label}\nB{prov[0]}:i{prov[1]}", fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def class_balance_stats(labels: np.ndarray):
    c = Counter(int(x) for x in labels)
    counts = np.array([c[i] for i in range(N_CLASSES)], dtype=int)
    return counts


def l1_distance_to_uniform(counts: np.ndarray):
    target = np.ones_like(counts, dtype=float) * (counts.sum() / float(len(counts)))
    return np.sum(np.abs(counts - target)) / counts.sum()


def kl_divergence_to_uniform(counts: np.ndarray, eps=1e-12):
    p = (counts.astype(float) + eps) / (counts.sum() + eps * len(counts))
    q = np.ones_like(p) / len(p)
    return np.sum(p * np.log(p / q))

# ----------------- Main experiment -----------------

def main():
    print("Preparing MNIST subset and batches...")
    X, y = sample_stratified_mnist_subset_embedded(NUM_PER_CLASS, SEED)
    batches = make_batches(X, y, BATCH_SIZE)
    print(f"Total items: {X.shape[0]}, total batches: {len(batches)}")

    # RFF sampler
    sampler = RBFSampler(gamma=RFF_GAMMA, n_components=RFF_DIM, random_state=SEED)
    # fit the sampler to get random_weights_ shape; RBFSampler.fit accepts data
    sampler.fit(X)

    # Instantiate streamer
    streamer = MirrorDescentHerdingStreamer(
        coreset_size=CORESET_SIZE,
        buffer_capacity=BUFFER_CAPACITY,
        sampler=sampler,
        batch_size=BATCH_SIZE,
        md_iterations=100,
        eta=10,
        verbose=False,
        removal_batch_size=REMOVAL_BATCH_SIZE,
    )

    # streamer = AdaptiveKernelHerdingStreamer(
    #     coreset_size=CORESET_SIZE,
    #     buffer_capacity=BUFFER_CAPACITY,
    #     sampler=sampler,
    #     batch_size=BATCH_SIZE,
    # )

    # keep snapshots and metrics
    snapshots = []
    batch_logs = []

    # We'll also record buffer distributions after certain important events
    # 1) after finishing label 0 batches (end of first label)
    # 2) midpoint of label 1 batches
    batches_per_label = NUM_PER_CLASS // BATCH_SIZE
    mid_of_label1_idx = batches_per_label + batches_per_label // 2  # 0-indexed

    for (Xb, yb, bidx) in batches:
        streamer.process_batch(Xb, yb, bidx)

        # capture current buffer info
        buf_X = streamer.buffer_X.copy()
        buf_y = streamer.buffer_y.copy()
        prov = list(streamer.buffer_provenance)

        counts = class_balance_stats(buf_y)
        batch_logs.append({"batch_idx": bidx, "buffer_size": len(buf_y), "counts": counts.copy()})

        # Print a compact status line
        print(f"Batch {bidx:02d} | buffer_size={len(buf_y):2d} | counts (per-class nonzero): {[(i,c) for i,c in enumerate(counts) if c>0]}")

        # # Save snapshot images periodically
        # if (bidx % SAVE_EVERY == 0) or (bidx in [batches_per_label - 1, mid_of_label1_idx]):
        #     title = f"Batch {bidx} | buffer_size={len(buf_y)}"
        #     fname = os.path.join(SNAPSHOT_DIR, f"snapshot_batch_{bidx:03d}.png")
        #     visualize_buffer_images(buf_X, buf_y, prov, fname, title=title)
        #     snapshots.append(fname)

        # Special check at midpoint of label-1 batches (user's diagnostic):
        if bidx == mid_of_label1_idx:
            # expected: ~3/4 label 0, ~1/4 label 1 among the buffer
            c0 = counts[0]
            c1 = counts[1]
            total = counts.sum()
            print("--- Diagnostic check at midpoint of label-1 stream ---")
            print(f"Buffer counts (label 0, label 1, total) = ({c0}, {c1}, {total})")
            if total > 0:
                print(f"Fractions -> label0: {c0/total:.2f}, label1: {c1/total:.2f}")
                if c0 >= 0.7 * total and c1 >= 0.2 * total:
                    print("OK: buffer contains a majority of label-0 but also a healthy share of label-1 (≈ expected)")
                elif c0 == total:
                    print("WARNING: buffer is dominated exclusively by label-0 — algorithm may be failing to adapt")
                else:
                    print("Notice: buffer composition deviates from the expected 3/4:1/4 pattern; investigate weights/removal schedule")

    # After stream, request final coreset
    flat_idx, final_weights, provenance = streamer.get_final_coreset()
    final_labels = np.array([p[0] for p in provenance])  # careful: provenance is (batch, id); we want labels
    # NOTE: buffer_y after finalize contains labels; but to be safe, get streamer.buffer_y
    final_labels = streamer.buffer_y.copy()
    final_counts = class_balance_stats(final_labels)

    print("\n=== Final coreset summary ===")
    print(f"Final coreset size = {len(final_labels)}")
    print(f"Per-class counts: {final_counts}")
    print(f"L1 distance to uniform (0..1): {l1_distance_to_uniform(final_counts):.4f}")
    print(f"KL divergence to uniform: {kl_divergence_to_uniform(final_counts):.4f}")

    # # Save final visualization
    # visualize_buffer_images(streamer.buffer_X, streamer.buffer_y, streamer.buffer_provenance,
    #                         os.path.join(SNAPSHOT_DIR, "final_coreset.png"), title="Final Coreset")

    print(f"Snapshots saved to: {SNAPSHOT_DIR}")
    print("Experiment finished.")

if __name__ == "__main__":
    main()
