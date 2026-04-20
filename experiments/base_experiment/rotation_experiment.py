import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is in sys.path for internal imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

try:
    import torch
    import torchvision
    import torchvision.transforms.functional as TF
    import torchvision.models as models
    import torchvision.transforms as T
    from PIL import Image
except ImportError:
    print("Error: torch, torchvision, and Pillow are required for this experiment.")
    sys.exit(1)

from streaming_coreset import StreamingCoreset
from sklearn.kernel_approximation import RBFSampler

def get_indices_from_streamer(streamer):
    """Safely extracts the final timesteps (batch indices) kept by the sampler."""
    if hasattr(streamer, 'get_final_coreset'):
        indices, _, _ = streamer.get_final_coreset()
        return indices
    # Fallback to provenance tuples: p[0] is batch_idx (timestep)
    return [p[0] for p in streamer.buffer_provenance]

def generate_embedded_rotation_stream(device="cpu"):
    """
    Fetches an MNIST digit, generates 360 rotated frames, and 
    embeds them using ResNet18 to match the previous experiments.
    """
    print("[data] Fetching MNIST dataset...")
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    
    # Pick a '3' because it is highly asymmetric
    target_idx = (dataset.targets == 3).nonzero(as_tuple=True)[0][0].item()
    base_img_tensor = dataset.data[target_idx].numpy()
    pil_img = Image.fromarray(base_img_tensor)
    
    print("[data] Generating 360 degrees of rotation...")
    stream_images = []
    pil_frames = []
    for angle in range(360):
        # Rotate creates the animation frame. Fill background with 0 (black).
        rot_img = TF.rotate(pil_img, float(angle), fill=0)
        pil_frames.append(rot_img)
        stream_images.append(np.array(rot_img))
        
    stream_images = np.array(stream_images)
    y_stream = np.zeros(360)  # Dummy labels
    
    print("[data] Embedding rotation frames via ResNet18...")
    # Load pretrained ResNet18 and remove classification head to get 512-D features
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = torch.nn.Identity()
    resnet = resnet.to(device)
    resnet.eval()

    # Standard ImageNet transforms, adapting 1-channel MNIST to 3-channel
    transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1)), # 1 channel to 3 channels
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    X_stream = []
    with torch.no_grad():
        for img in pil_frames:
            tensor = transform(img).unsqueeze(0).to(device)
            feat = resnet(tensor).squeeze(0).cpu().numpy()
            X_stream.append(feat)
            
    X_stream = np.array(X_stream, dtype=np.float64)
    print(f"[data] Generated embedded stream of shape: {X_stream.shape}")
    
    return X_stream, y_stream, stream_images

def main():
    output_dir = os.path.join(_PROJECT_ROOT, "snapshots_base_experiment", "visualized_rotation")
    os.makedirs(output_dir, exist_ok=True)

    # Automatically use GPU if available for faster ResNet embedding
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_stream, y_stream, images_stream = generate_embedded_rotation_stream(device=device)

    # RFF Setup (using standard ResNet gamma)
    D = 1024
    gamma = 0.001 
    seed = 42
    
    print(f"[setup] Fitting RBFSampler (D={D}, gamma={gamma}) on ResNet embeddings...")
    np.random.seed(seed)
    sampler_rff = RBFSampler(gamma=gamma, n_components=D, random_state=seed)
    sampler_rff.fit(X_stream[:10]) 

    M_values = [3, 4, 6]
    results = {}

    for M in M_values:
        print(f"[experiment] Running StreamingCoreset for M={M}...")
        coreset = StreamingCoreset(M, D, sampler_rff, batch_size=1, K_iter=100)
        
        # Stream the 360 feature vectors
        for t in range(360):
            coreset.process_batch(X_stream[t : t + 1], y_stream[t : t + 1], batch_idx=t)
            
        # Extract indices and sort them chronologically (by angle)
        results[M] = sorted(get_indices_from_streamer(coreset))

    print("[plot] Generating visualization...")
    fig, axes = plt.subplots(len(M_values), max(M_values), figsize=(14, 8))
    fig.suptitle("Coreset Buffer Contents for Rotating MNIST Digit (ResNet Latent Space)", fontsize=16, fontweight='bold')
    
    # Turn off all axes by default
    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')

    # Plot the corresponding raw images based on the indices chosen in latent space
    for row_idx, M in enumerate(M_values):
        indices = results[M]
        
        # Add a row label to the first axis of the row
        axes[row_idx, 0].text(-0.3, 0.5, f"M = {M}", transform=axes[row_idx, 0].transAxes,
                              fontsize=14, fontweight='bold', va='center', ha='right')
        
        for col_idx, angle in enumerate(indices):
            ax = axes[row_idx, col_idx]
            ax.imshow(images_stream[angle], cmap='gray')
            ax.set_title(f"Angle: {angle}°", fontsize=12)
            ax.axis('on')
            ax.set_xticks([])
            ax.set_yticks([])
            
            for spine in ax.spines.values():
                spine.set_edgecolor('gray')
                spine.set_linewidth(0.5)

    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    
    plot_path = os.path.join(output_dir, "embedded_rotation_experiment.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"[plot] Saved visualization to: {plot_path}")
    plt.close(fig)

if __name__ == "__main__":
    main()