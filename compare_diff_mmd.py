import argparse
import numpy as np
from sklearn.kernel_approximation import RBFSampler
import matplotlib.pyplot as plt
from mmdplusstreamer import OnlineMMDPlusStreamer
from dataloaders import load_adult_data
from utils import calculate_mmd2_exact, calculate_mmd2_approx

def generate_coresets(config, d_size_large):
    """Generate coresets using OnlineMMDPlus with a large d_size."""
    np.random.seed(config['random_seed'])
    X_train, _, _, _ = load_adult_data(config['dataset_subset_size'])
    rbf_large = RBFSampler(gamma=config['kernel_gamma'], n_components=d_size_large, random_state=42)
    rbf_large.fit(X_train)
    
    coresets = {}
    for m in config['coreset_sizes']:
        streamer = OnlineMMDPlusStreamer(
            m_coreset_size=m, n_rff_components=d_size_large,
            buffer_capacity=config['buffer_capacity'], n_epochs_online=config['n_epochs_online'],
            lr_online=config['lr_online'], lambda_log_online=config['lambda_log_online'],
            random_seed=42
        )
        streamer.set_rbf_sampler(rbf_large)
        for i in range(int(np.ceil(X_train.shape[0] / config['batch_size']))):
            batch = X_train[i * config['batch_size']:(i + 1) * config['batch_size']]
            if batch.shape[0]: streamer.process_batch(batch)
        idxs = streamer.print_coreset_provenance(config['batch_size'])
        _, w = streamer.get_final_coreset()
        Xc = X_train[idxs]
        coresets[m] = (Xc, w)
    return coresets, X_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coreset_sizes", nargs="+", type=int, default=list(range(10, 31, 5)))
    parser.add_argument("--dataset_subset_size", type=int, default=2500)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--kernel_gamma", type=float, default=0.1)
    parser.add_argument("--buffer_capacity", type=int, default=150)
    parser.add_argument("--random_seed", type=int, default=23874291)
    parser.add_argument("--n_epochs_online", type=int, default=20)
    parser.add_argument("--lr_online", type=float, default=0.1)
    parser.add_argument("--lambda_log_online", type=float, default=5e-5)
    args = parser.parse_args()
    config = vars(args)

    # Step 1: Generate coresets with a large d_size
    d_size_large = 200
    coresets, X_train = generate_coresets(config, d_size_large)

    # Step 2: Compute exact MMD for each coreset
    exact_mmds = {}
    for m, (Xc, w) in coresets.items():
        exact_mmds[m] = calculate_mmd2_exact(X_train, Xc, w, config['kernel_gamma'])

    # Step 3: Compute approximated MMD for each d_size
    D_sizes = [20, 50, 100, 200]
    approx_mmds = {d_size: {} for d_size in D_sizes}
    for d_size in D_sizes:
        rbf_d = RBFSampler(gamma=config['kernel_gamma'], n_components=d_size, random_state=42)
        rbf_d.fit(X_train)
        for m, (Xc, w) in coresets.items():
            approx_mmds[d_size][m] = calculate_mmd2_approx(X_train, Xc, w, rbf_d)

    # Step 4: Plot |approximated MMD - exact MMD| vs d_size for selected m
    selected_m = [10, 20, 30]  # Representative coreset sizes
    plt.figure(figsize=(10, 6))
    for m in selected_m:
        if m in coresets:
            errors = [abs(approx_mmds[d_size][m] - exact_mmds[m]) for d_size in D_sizes]
            plt.plot(D_sizes, errors, label=f'm={m}', marker='o')
    plt.xlabel('RFF Components (d_size)')
    plt.ylabel('|Approximated MMD - Exact MMD|')
    plt.title('MMD Approximation Error vs RFF Components')
    plt.legend()
    plt.grid()
    plt.savefig('mmd_error_vs_d_size.png')
    plt.show()