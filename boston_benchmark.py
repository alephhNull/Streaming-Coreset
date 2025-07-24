import argparse
import numpy as np
from sklearn.kernel_approximation import RBFSampler
import matplotlib.pyplot as plt

from streamers.reservoirstreamer import ReservoirSamplerBatchStreamer
from streamers.mmdplusstreamer import OnlineMMDPlusStreamer
from dataloaders import load_boston
from utils import calculate_mmd2_exact
from downstream_tasks import train_regressor

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

    print("Loading data...")
    np.random.seed(seed)
    X_train, X_val, y_train, y_val = load_boston()
    n_total = X_train.shape[0]
    num_batches = int(np.ceil(n_total / batch_size))
    print(f"Data: {n_total} points, batch_size={batch_size}, batches={num_batches}")

    rbf = RBFSampler(gamma=gamma, n_components=n_rff, random_state=42)
    rbf.fit(X_train)

    rmse_whole = train_regressor(X_train, X_val, y_train, y_val)
    print(f"Baseline (whole dataset) RMSE: {rmse_whole:.4f}")

    # Initialize results structure
    results = {'coreset_size': core_sizes}
    for bm in benchmarks:
        results[f"{bm}_RMSE"] = []
        results[f"{bm}_MMD"] = []

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
            rmse = train_regressor(Xc, X_val, yc, y_val)
            mmd = calculate_mmd2_exact(X_train, Xc, w, gamma)
            results['OnlineMMDPlus_RMSE'].append(rmse)
            results['OnlineMMDPlus_MMD'].append(mmd)
            print(f"OnlineMMDPlus -> rmse:{rmse:.4f}, mmd:{mmd:.6f}")

        # Reservoir Sampling
        if 'Reservoir' in benchmarks:
            rmses, mmds = [], []
            for t in range(reservoir_trials):
                r = ReservoirSamplerBatchStreamer(m, 42 + t)
                for i in range(num_batches):
                    start = i*batch_size
                    size = min(batch_size, n_total - start)
                    r.process_batch(start, size)
                Xc, yc, w = r.get_final_coreset_details(X_train, y_train)
                rmses.append(train_regressor(Xc, X_val, yc, y_val))
                mmds.append(calculate_mmd2_exact(X_train, Xc, w, gamma))
            results['Reservoir_RMSE'].append(np.mean(rmses))
            results['Reservoir_MMD'].append(np.mean(mmds))
            print(f"Reservoir -> rmse:{np.mean(rmses):.4f}, mmd:{np.mean(mmds):.6f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    ax1, ax2 = axes
    for bm in benchmarks:
        ax1.plot(core_sizes, results[f"{bm}_RMSE"], label=bm)
        ax2.plot(core_sizes, results[f"{bm}_MMD"], label=bm)
    ax1.axhline(rmse_whole, color='black', linestyle='--', label='Whole Dataset')
    ax1.set(xlabel='Coreset Size', ylabel='Val. RMSE', title='RMSE vs Coreset')
    ax2.set(xlabel='Coreset Size', ylabel='Exact MMD²', title='MMD vs Coreset', yscale='log')
    ax1.legend(); ax2.legend()
    ax1.grid(True); ax2.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="+", default=['OnlineMMDPlus', 'Reservoir'],
                        help="Which methods to run")
    parser.add_argument("--coreset_sizes", nargs="+", type=int, default=list(range(20, 30, 2)))
    parser.add_argument("--dataset_subset_size", type=int, default=506)  # Boston has 506 samples
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--n_rff_components", type=int, default=100)
    parser.add_argument("--kernel_gamma", type=float, default=0.1)  
    parser.add_argument("--buffer_capacity", type=int, default=150)
    parser.add_argument("--random_seed", type=int, default=2093)
    parser.add_argument("--n_epochs_online", type=int, default=20)
    parser.add_argument("--lr_online", type=float, default=0.1)
    parser.add_argument("--lambda_log_online", type=float, default=5e-5)
    parser.add_argument("--reservoir_trials", type=int, default=10)

    args = parser.parse_args()
    config = vars(args)

    print("Using config:", config)
    run_experiment(config)