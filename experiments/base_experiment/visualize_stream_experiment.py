import os
import sys
import numpy as np
from collections import deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

# Ensure project root is in sys.path for internal imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

try:
    import torchvision
except ImportError:
    print("Error: torchvision is required to fetch raw images for the visualization.")
    sys.exit(1)

from base_experiment import BaseExperimentConfig, MetricsMode, load_embedded_train_split, resolve_n_classes, resolve_stream_length
from stream_builders import logits_to_per_class_counts, interleaved_chunk_indices
from streaming_coreset import StreamingCoreset
from reservoir_rff_baseline import ReservoirRFFBaseline
from sklearn.kernel_approximation import RBFSampler

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


class SlidingWindowBaseline:
    """FIFO baseline that keeps the most recent M stream timesteps."""

    def __init__(self, M: int):
        self.M = int(M)
        self.buffer_idx = deque(maxlen=self.M)

    def process_batch(self, _X_batch, _y_batch, batch_idx: int):
        self.buffer_idx.append(int(batch_idx))

    def get_indices_weights(self):
        if len(self.buffer_idx) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        idx = np.asarray(list(self.buffer_idx), dtype=np.int64)
        w = np.full(idx.shape[0], 1.0 / float(idx.shape[0]), dtype=np.float64)
        return idx, w
    

def load_stratified_stream_with_images(cfg: BaseExperimentConfig):
    """
    Hooks into the original stream building logic but extracts raw images mapped 
    to the exact same indices used to build the embedded stream.
    """
    n_classes = resolve_n_classes(cfg)
    logits = dict(cfg.label_logits) if cfg.label_logits else {c: 1.0 for c in range(n_classes)}
    L = resolve_stream_length(cfg)
    per_class = logits_to_per_class_counts(logits, n_classes, L)

    print("[data] Loading embeddings to establish indices...")
    X_all, y_all = load_embedded_train_split(
        cfg.dataset_name, cfg.seed, subset_size=cfg.data_subset_size, device=cfg.embed_device
    )

    print("[data] Fetching raw images for visualization mapping...")
    raw_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    images = raw_dataset.data
    targets = np.array(raw_dataset.targets)
    
    # Verify alignment between dataloaders subset and raw PyTorch dataset
    if np.array_equal(y_all[:10], targets[:10]):
        images_all = images[:len(y_all)]
    else:
        print("[data] Warning: Direct index alignment failed. Falling back to class-representative mapping.")
        class_imgs = {c: images[targets == c] for c in range(10)}
        class_counters = {c: 0 for c in range(10)}
        matched = []
        for y_val in y_all:
            matched.append(class_imgs[y_val][class_counters[y_val] % len(class_imgs[y_val])])
            class_counters[y_val] += 1
        images_all = np.array(matched)

    # Mimic stratified_sample_stream to trace indices
    rng = np.random.RandomState(cfg.seed)
    idxs = []
    for c in range(n_classes):
        n_want = per_class[c]
        if n_want == 0:
            continue
        class_idx = np.where(y_all == c)[0]
        chosen = rng.choice(class_idx, size=n_want, replace=False)
        idxs.append(chosen)
    idxs_flat = np.concatenate(idxs)
    idx_order = interleaved_chunk_indices(per_class, n_classes, cfg.num_splits)

    # These are the original indices in the exact order of the stream
    final_stream_indices = idxs_flat[idx_order]
    
    X_stream = X_all[final_stream_indices]
    y_stream = y_all[final_stream_indices]
    images_stream = images_all[final_stream_indices]

    return X_stream, y_stream, images_stream

def get_indices_from_streamer(streamer):
    """Safely extracts the final timesteps kept by the sampler."""
    if hasattr(streamer, 'get_final_coreset'):
        indices, _, _ = streamer.get_final_coreset()
        return indices
    if hasattr(streamer, 'get_indices_weights'):
        indices, _ = streamer.get_indices_weights()
        return indices
    # Fallback to provenance tuples: p[0] is batch_idx (timestep)
    return [p[0] for p in streamer.buffer_provenance]

def plot_stream_experiment(
    y_stream,
    images_stream,
    sw_indices,
    res_indices,
    our_indices,
    M,
    filename="stream_visualization.pdf",
):
    """Generates a 4-row comparison layout."""
    fig = plt.figure(figsize=(20, 7))
    
    gs = gridspec.GridSpec(4, 1, height_ratios=[1.5, 2.2, 2.2, 2.2], hspace=0.2)
    cmap = plt.get_cmap('tab10')

    # ==========================
    # ROW 1: Stream Decomposition
    # ==========================
    ax_stream = fig.add_subplot(gs[0])
    blocks = []
    current_label = y_stream[0]
    start_t = 0
    for t in range(1, len(y_stream)):
        if y_stream[t] != current_label:
            blocks.append((current_label, start_t, t))
            current_label = y_stream[t]
            start_t = t
    blocks.append((current_label, start_t, len(y_stream)))

    for (label, s_t, e_t) in blocks:
        color = cmap(label % 10)
        width = e_t - s_t
        rect = Rectangle((s_t, 0), width, 1, facecolor=color, edgecolor='white', linewidth=1.5)
        ax_stream.add_patch(rect)
        
        class_name = CIFAR10_CLASSES[label]
        text = f"{class_name}\n({width})"
        # Only add text if the block has enough horizontal width to fit it
        if width >= max(2, len(y_stream)*0.015):
            ax_stream.text(s_t + width/2, 0.5, text, ha='center', va='center',
                           color='white', fontweight='bold', fontsize=10)

    ax_stream.set_xlim(0, len(y_stream))
    ax_stream.set_ylim(0, 1)
    ax_stream.set_yticks([])
    ax_stream.set_xlabel("Stream Steps (t)", fontsize=11, fontweight='bold')
    ax_stream.set_title("Stream Decomposition", fontsize=14, fontweight='bold')

    # ==========================
    # ROW 2: Sliding Window
    # ==========================
    gs_row2 = gridspec.GridSpecFromSubplotSpec(1, M, subplot_spec=gs[1], wspace=0.1)
    for i, t in enumerate(sw_indices[:M]):
        ax_img = fig.add_subplot(gs_row2[0, i])
        ax_img.imshow(images_stream[t])
        ax_img.axis('off')
        
        # Bottom color ribbon
        color = cmap(y_stream[t] % 10)
        ribbon = Rectangle((0, -0.15), 1, 0.15, transform=ax_img.transAxes,
                           facecolor=color, clip_on=False)
        ax_img.add_patch(ribbon)
        
        if i == 0:
            ax_img.text(-0.15, 0.5, "Sliding\nWindow", transform=ax_img.transAxes,
                        ha='right', va='center', fontsize=12, fontweight='bold')

    # ==========================
    # ROW 3: Reservoir Sampling
    # ==========================
    gs_row3 = gridspec.GridSpecFromSubplotSpec(1, M, subplot_spec=gs[2], wspace=0.1)
    for i, t in enumerate(res_indices[:M]):
        ax_img = fig.add_subplot(gs_row3[0, i])
        ax_img.imshow(images_stream[t])
        ax_img.axis('off')
        
        # Bottom color ribbon
        color = cmap(y_stream[t] % 10)
        ribbon = Rectangle((0, -0.15), 1, 0.15, transform=ax_img.transAxes,
                           facecolor=color, clip_on=False)
        ax_img.add_patch(ribbon)
        
        if i == 0:
            ax_img.text(-0.15, 0.5, "Reservoir\nSampling", transform=ax_img.transAxes,
                        ha='right', va='center', fontsize=12, fontweight='bold')

    # ==========================
    # ROW 4: Streaming Coreset
    # ==========================
    gs_row4 = gridspec.GridSpecFromSubplotSpec(1, M, subplot_spec=gs[3], wspace=0.1)
    for i, t in enumerate(our_indices[:M]):
        ax_img = fig.add_subplot(gs_row4[0, i])
        ax_img.imshow(images_stream[t])
        ax_img.axis('off')
        
        # Bottom color ribbon
        color = cmap(y_stream[t] % 10)
        ribbon = Rectangle((0, -0.15), 1, 0.15, transform=ax_img.transAxes,
                           facecolor=color, clip_on=False)
        ax_img.add_patch(ribbon)
        
        if i == 0:
            ax_img.text(-0.15, 0.5, "Our Method", transform=ax_img.transAxes,
                        ha='right', va='center', fontsize=12, fontweight='bold')

    # Ensure format="pdf" is passed to guarantee the vector format output
    plt.savefig(filename, bbox_inches='tight', format="pdf", dpi=300)
    print(f"[plot] Saved visualization to: {filename}")
    plt.close(fig)

def main():
    # Desired weights corresponding to labels (0 to 6)
    ws = [3, 1, 2, 4, 1, 3, 2]
    logits = {i: float(ws[i]) for i in range(len(ws))}
    
    cfg = BaseExperimentConfig(
        dataset_name="cifar10",
        label_logits=logits,
        stream_length=1000,
        num_splits=1,
        metrics=MetricsMode.BOTH,
        seed=42,
        output_dir=os.path.join(_PROJECT_ROOT, "snapshots_base_experiment", "visualized_stream"),
    )
    os.makedirs(cfg.output_dir, exist_ok=True)

    X_stream, y_stream, images_stream = load_stratified_stream_with_images(cfg)

    M, D, gamma = 16, 1024, 0.001
    np.random.seed(cfg.seed)
    sampler_rff = RBFSampler(gamma=gamma, n_components=D, random_state=cfg.seed + 12345)
    sampler_rff.fit(X_stream[: min(10, len(X_stream))])

    streamers = {
        "our_coreset": StreamingCoreset(M, D, sampler_rff, batch_size=1, K_iter=100),
        "sliding_window": SlidingWindowBaseline(M),
        "reservoir": ReservoirRFFBaseline(M, D, sampler_rff, batch_size=1),
    }

    print("[experiment] Simulating stream over baselines...")
    for t in range(len(X_stream)):
        streamers["our_coreset"].process_batch(X_stream[t : t + 1], y_stream[t : t + 1], batch_idx=t)
        streamers["sliding_window"].process_batch(X_stream[t : t + 1], y_stream[t : t + 1], batch_idx=t)
        streamers["reservoir"].process_batch(X_stream[t : t + 1], y_stream[t : t + 1], batch_idx=t)

    # Fetch resulting indices and sort them chronologically by stream timestep
    our_indices = sorted(get_indices_from_streamer(streamers["our_coreset"]))
    sw_indices = sorted(get_indices_from_streamer(streamers["sliding_window"]))
    res_indices = sorted(get_indices_from_streamer(streamers["reservoir"]))

    print("[plot] Generating requested layout...")
    # Updated output name to .pdf
    plot_path = os.path.join(cfg.output_dir, "coreset_stream_decomposition.pdf")
    plot_stream_experiment(y_stream, images_stream, sw_indices, res_indices, our_indices, M, filename=plot_path)

if __name__ == "__main__":
    main()