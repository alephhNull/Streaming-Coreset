import argparse
import numpy as np
from sklearn.kernel_approximation import RBFSampler
import matplotlib.pyplot as plt
from bcsstreamer import BilevelCoresetSelector
from ocsstreamer import OCSStreamer
from reservoirstreamer import ReservoirSamplerBatchStreamer
from mmdplusstreamer import OnlineMMDPlusStreamer
from dataloaders import load_cifar10_encoded
from utils import calculate_mmd2_exact
from downstream_tasks import train_classifier

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.decomposition import PCA
import numpy as np
import torch.nn as nn




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
    X_train, X_val, y_train, y_val = load_cifar10_encoded(ds_size, 20)  # Modified
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
        results[f"{bm}_Acc"] = []
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
            acc = train_classifier(Xc, X_val, yc, y_val)
            mmd = calculate_mmd2_exact(X_train, Xc, w, gamma)
            results['OnlineMMDPlus_Acc'].append(acc)
            results['OnlineMMDPlus_MMD'].append(mmd)
            print(f"OnlineMMDPlus -> acc:{acc:.4f}, mmd:{mmd:.6f}")

        # Reservoir Sampling
        if 'Reservoir' in benchmarks:
            accs, mmds = [], []
            for t in range(reservoir_trials):
                r = ReservoirSamplerBatchStreamer(m, 42 + t)
                for i in range(num_batches):
                    start = i*batch_size
                    size = min(batch_size, n_total - start)
                    r.process_batch(start, size)
                Xc, yc, w = r.get_final_coreset_details(X_train, y_train)
                accs.append(train_classifier(Xc, X_val, yc, y_val))
                mmds.append(calculate_mmd2_exact(X_train, Xc, w, gamma))
            results['Reservoir_Acc'].append(np.mean(accs))
            results['Reservoir_MMD'].append(np.mean(mmds))
            print(f"Reservoir -> acc:{np.mean(accs):.4f}, mmd:{np.mean(mmds):.6f}")

        # OCS Streamer
        if 'OCS' in benchmarks:
            ocs = OCSStreamer(m_coreset_size=m, batch_size=batch_size, tau=ocs_tau, random_seed=42)
            for i in range(num_batches):
                Xb = X_train[i*batch_size:(i+1)*batch_size]
                yb = y_train[i*batch_size:(i+1)*batch_size]
                idxb = np.arange(i*batch_size, min((i+1)*batch_size, n_total))
                if Xb.shape[0]:
                    ocs.process_batch(Xb, yb, idxb, X_train, y_train)
            Xc, yc, w = ocs.get_final_coreset_details(X_train, y_train)
            acc = train_classifier(Xc, X_val, yc, y_val)
            mmd = calculate_mmd2_exact(X_train, Xc, w, gamma)
            results['OCS_Acc'].append(acc)
            results['OCS_MMD'].append(mmd)
            print(f"OCS -> acc:{acc:.4f}, mmd:{mmd:.6f}")

        # BCSR
        if 'BCSR' in benchmarks:
            bcsr = BilevelCoresetSelector(
                input_dim=X_train.shape[1], m_coreset_size=m,
                outer_loops=bcsr_outer, inner_loops=bcsr_inner,
                lr_outer=bcsr_lr_outer, lr_inner=bcsr_lr_inner,
                lambda_reg=bcsr_lambda, random_seed=42
            )
            bcsr.set_full_data(X_train, y_train)
            for i in range(num_batches):
                Xb = X_train[i*batch_size:(i+1)*batch_size]
                yb = y_train[i*batch_size:(i+1)*batch_size]
                if Xb.shape[0]: bcsr.process_batch(Xb, yb)
            idxs, w = bcsr.get_final_coreset()
            if idxs.size:
                Xc, yc = X_train[idxs], y_train[idxs]
                acc = train_classifier(Xc, X_val, yc, y_val)
                mmd = calculate_mmd2_exact(X_train, Xc, w, gamma)
            else:
                acc, mmd = 0.0, float('inf')
            results['BCSR_Acc'].append(acc)
            results['BCSR_MMD'].append(mmd)
            print(f"BCSR -> acc:{acc:.4f}, mmd:{mmd:.6f}")

    # Plot
    fig, axes = plt.subplots(1,2, figsize=(18,6))
    ax1, ax2 = axes
    for bm in benchmarks:
        ax1.plot(core_sizes, results[f"{bm}_Acc"], label=bm)
        ax2.plot(core_sizes, results[f"{bm}_MMD"], label=bm)
    ax1.axhline(acc_whole, color='black', linestyle='--', label='Whole Dataset')
    ax1.set(xlabel='Coreset Size', ylabel='Val. Acc', title='Accuracy vs Coreset')
    ax2.set(xlabel='Coreset Size', ylabel='Exact MMD²', title='MMD vs Coreset', yscale='log')
    ax1.legend(); ax2.legend()
    ax1.grid(True); ax2.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="+", default=['OnlineMMDPlus','Reservoir'],
                        help="Which methods to run")
    parser.add_argument("--coreset_sizes", nargs="+", type=int, default=list(range(50, 101, 10)))
    parser.add_argument("--dataset_subset_size", type=int, default=2500)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--n_rff_components", type=int, default=1000)
    parser.add_argument("--kernel_gamma", type=float, default=0.01)
    parser.add_argument("--buffer_capacity", type=int, default=150)
    parser.add_argument("--random_seed", type=int, default=2342)
    parser.add_argument("--n_epochs_online", type=int, default=5)
    parser.add_argument("--lr_online", type=float, default=0.1)
    parser.add_argument("--lambda_log_online", type=float, default=1e-5)
    parser.add_argument("--reservoir_trials", type=int, default=10)
    parser.add_argument("--ocs_tau", type=float, default=1.0)
    parser.add_argument("--bcsr_outer_loops", type=int, default=5)
    parser.add_argument("--bcsr_inner_loops", type=int, default=5)
    parser.add_argument("--bcsr_lr_outer", type=float, default=0.01)
    parser.add_argument("--bcsr_lr_inner", type=float, default=0.001)
    parser.add_argument("--bcsr_lambda", type=float, default=0.001)

    args = parser.parse_args()
    config = vars(args)

#     config = {
#     "benchmarks": ["OnlineMMDPlus", "OCS"],  # only these
#     "coreset_sizes": [30, 60, 90],
#     "dataset_subset_size": 5000,
#     "batch_size": 100,
#     "n_rff_components": 200,
#     "kernel_gamma": 0.05,
#     "buffer_capacity": 200,
#     "random_seed": 12345,
#     "n_epochs_online": 30,
#     "lr_online": 0.05,
#     "lambda_log_online": 1e-4,
#     "reservoir_trials": 5,
#     "ocs_tau": 0.5,
#     "bcsr_outer_loops": 3,
#     "bcsr_inner_loops": 10,
#     "bcsr_lr_outer": 0.02,
#     "bcsr_lr_inner": 0.002,
#     "bcsr_lambda": 0.0005,
# }


    # Example config printing
    print("Using config:", config)

    run_experiment(config)
