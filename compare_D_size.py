import argparse
import numpy as np
from sklearn.kernel_approximation import RBFSampler
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

from streamers.bcsstreamer import BilevelCoresetSelector
from streamers.ocsstreamer import OCSStreamer
from streamers.reservoirstreamer import ReservoirSamplerBatchStreamer
from streamers.mmdplusstreamer import OnlineMMDPlusStreamer
from dataloaders import load_adult_data, load_electricity_tiny
from utils import calculate_mmd2_exact, calculate_mmd2_approx
from downstream_tasks import train_classifier

def run_experiment(config):
    # Unpack config
    ds_size = config['dataset_subset_size']
    batch_size = config['batch_size']
    n_rff = config['n_rff_components']
    gamma = config['kernel_gamma']
    seed = config['random_seed']
    benchmarks = config['benchmarks']
    core_sizes = config['coreset_sizes']
    buffer_cap = config['buffer_capacity']
    online_epochs = config['n_epochs_online']
    lr_online = config['lr_online']
    lambda_online = config['lambda_log_online']
    reservoir_trials = config['reservoir_trials']
    ocs_tau = config['ocs_tau']
    bcsr_outer = config['bcsr_outer_loops']
    bcsr_inner = config['bcsr_inner_loops']
    bcsr_lr_outer = config['bcsr_lr_outer']
    bcsr_lr_inner = config['bcsr_lr_inner']
    bcsr_lambda = config['bcsr_lambda']

    print("Loading data...")
    np.random.seed(seed)
    X_train, X_val, y_train, y_val = load_adult_data(ds_size)
    n_total = X_train.shape[0]
    num_batches = int(np.ceil(n_total / batch_size))
    print(f"Data: {n_total} points, batch_size={batch_size}, batches={num_batches}")

    rbf = RBFSampler(gamma=gamma, n_components=n_rff, random_state=42)
    rbf.fit(X_train)

    acc_whole = train_classifier(X_train, X_val, y_train, y_val)
    print(f"Baseline (whole dataset) accuracy: {acc_whole:.4f}")

    # initialize results structure
    results = {'coreset_size': core_sizes}
    for bm in benchmarks:
        results[f"MMD_Exact"] = []
        results[f"MMD_Approx"] = []

    for m in core_sizes:
        print(f"\n== Coreset size: {m}")
        
        # OnlineMMDPlus
        if 'OnlineMMDPlus' in benchmarks:
            streamer = OnlineMMDPlusStreamer(
                m_coreset_size=m, n_rff_components=n_rff,
                buffer_capacity=buffer_cap, n_epochs_online=online_epochs,
                lr_online=lr_online, lambda_log_online=lambda_online,
                random_seed=42
            )
            streamer.set_rbf_sampler(rbf)
            for i in range(num_batches):
                batch = X_train[i*batch_size:(i+1)*batch_size]
                if batch.shape[0]: streamer.process_batch(batch)
            idxs = streamer.print_coreset_provenance(batch_size)
            _, w = streamer.get_final_coreset()
            Xc, yc = X_train[idxs], y_train[idxs]
            # acc = train_classifier(Xc, X_val, yc, y_val)
            mmd_exact = calculate_mmd2_exact(X_train, Xc, w, gamma)
            mmd_approx = calculate_mmd2_approx(X_train, Xc, w, rbf)
            results['MMD_Exact'].append(mmd_exact)
            results['MMD_Approx'].append(mmd_approx)
            print(f"OnlineMMDPlus -, mmd:{mmd_exact:.6f}, approx_mmd:{mmd_approx:.6f}")

    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="+", default=['OnlineMMDPlus'],
                        help="Which methods to run")
    parser.add_argument("--coreset_sizes", nargs="+", type=int, default=list(range(10, 31, 2)))
    parser.add_argument("--dataset_subset_size", type=int, default=2500)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--n_rff_components", type=int, default=100)
    parser.add_argument("--kernel_gamma", type=float, default=0.1)
    parser.add_argument("--buffer_capacity", type=int, default=150)
    parser.add_argument("--random_seed", type=int, default=23874291)
    parser.add_argument("--n_epochs_online", type=int, default=20)
    parser.add_argument("--lr_online", type=float, default=0.1)
    parser.add_argument("--lambda_log_online", type=float, default=5e-5)
    parser.add_argument("--reservoir_trials", type=int, default=10)
    parser.add_argument("--ocs_tau", type=float, default=1.0)
    parser.add_argument("--bcsr_outer_loops", type=int, default=5)
    parser.add_argument("--bcsr_inner_loops", type=int, default=5)
    parser.add_argument("--bcsr_lr_outer", type=float, default=0.01)
    parser.add_argument("--bcsr_lr_inner", type=float, default=0.001)
    parser.add_argument("--bcsr_lambda", type=float, default=0.001)

    args = parser.parse_args()
    config = vars(args)
    D_sizes = [20, 50, 100, 200, 1000]
    approx_mmds = {}
    exact_mmds = {}
    for d_size in D_sizes:
        config['n_rff_components'] = d_size
        results = run_experiment(config)
        approx_mmds[d_size] = results['MMD_Approx']
        exact_mmds[d_size] = results['MMD_Exact']

    # Save results to a file
    with open('mmd_results.pkl', 'wb') as f:
        pickle.dump({'approx_mmds': approx_mmds, 'exact_mmds': exact_mmds}, f)
    

    # Set a professional style with Seaborn
    sns.set_style("whitegrid", {
        "grid.color": "#e0e0e0",
        "grid.linestyle": "-",
        "axes.edgecolor": "#333333",
        "axes.facecolor": "white",
        "font.family": "Times New Roman"
    })


    # Create figure with a larger size for better readability
    plt.figure(figsize=(8, 5), dpi=100)

    # Color palette for a professional look
    colors = sns.color_palette("deep", n_colors=len(approx_mmds))

    # Plot approximate MMD lines
    for idx, (d_size, mmds) in enumerate(approx_mmds.items()):
        plt.plot(config['coreset_sizes'], mmds, marker='o', markersize=6, linewidth=2, 
                 color=colors[idx], label=f'Approx MMD (d={d_size})')

    # Plot exact MMD line with a distinct style
    plt.plot(config['coreset_sizes'], exact_mmds[1000], linestyle='--', linewidth=2.5, 
             color='black', label='Exact MMD')

    # Customize axes
    plt.xlabel('Coreset Size', fontsize=14, fontweight='bold')
    plt.ylabel('MMD²', fontsize=14, fontweight='bold')
    plt.title('MMD² vs Coreset Size for Different RFF Dimensions', 
              fontsize=16, fontweight='bold', pad=15)

    # Customize legend
    plt.legend(fontsize=12, loc='best', frameon=True, 
              framealpha=0.9, edgecolor='#333333')

    # Customize ticks
    plt.tick_params(axis='both', which='major', labelsize=12, 
                    direction='in', length=5, color='#333333')

    # Adjust layout to prevent clipping
    plt.tight_layout()

    # Save and show the plot
    plt.savefig('mmd_vs_coreset_size.png', dpi=300, bbox_inches='tight')
    plt.show()