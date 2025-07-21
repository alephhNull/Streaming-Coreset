import numpy as np
from sklearn.kernel_approximation import RBFSampler
from reservoirstreamer import ReservoirSamplerBatchStreamer
from mmdplusstreamer import OnlineMMDPlusStreamer
from dataloaders import load_dataset
from utils import calculate_mmd2_exact
from downstream_tasks import train_classifier, train_nn_classifier
import torch


def run_onlinemmdplus(train_loader, X_train, y_train, gamma, n_rff, seed, coreset_size, buffer_cap, online_epochs, lr_online, lambda_online, device):
    rbf = RBFSampler(gamma=gamma, n_components=n_rff, random_state=seed)
    rbf.fit(X_train)

    streamer = OnlineMMDPlusStreamer(
                m_coreset_size=coreset_size, n_rff_components=n_rff,
                buffer_capacity=buffer_cap, n_epochs_online=online_epochs,
                lr_online=lr_online, lambda_log_online=lambda_online,
                random_seed=seed, device=device
            )
    
    streamer.set_rbf_sampler(rbf)

    for batch_x, _ in train_loader:
        batch_np = batch_x.cpu().numpy()
        if batch_np.shape[0]:
            streamer.process_batch(batch_np)

    idxs = streamer.print_coreset_provenance(train_loader.batch_size)
    _, w = streamer.get_final_coreset()
    Xc, yc = X_train[idxs], y_train[idxs]

    return Xc, yc, w

def run_reservoir(train_loader, X_train, y_train, coreset_size, seed):
    r = ReservoirSamplerBatchStreamer(coreset_size, seed)
    start = 0

    for batch_x, _ in train_loader:
        size = len(batch_x)
        r.process_batch(start, size)
        start += size

    Xc, yc, w = r.get_final_coreset_details(X_train, y_train)
    return Xc, yc, w



def run_single_experiment(config):
    dataset_name = config['dataset']
    ds_size = config['dataset_subset_size']
    seed = config['random_seed']
    embedding = config['embedding']
    embed_dim = config['embed_dim']
    batch_size = config['batch_size']

    n_rff = config['n_rff_components']
    gamma = config['kernel_gamma']
    buffer_cap = config['buffer_capacity']
    online_epochs = config['n_epochs_online']
    lr_online = config['lr_online']
    lambda_online = config['lambda_log_online']

    benchmarks = config['benchmarks']
    core_size = config['coreset_size']
    
    reservoir_trials = config['reservoir_trials']


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, X_train, X_val, y_train, y_val = load_dataset(dataset_name, ds_size, batch_size, seed, embedding, embed_dim, device)

    acc_whole = train_classifier(X_train, X_val, y_train, y_val)
    print(f"Baseline (whole dataset) accuracy: {acc_whole:.4f}")

    print(f"\n== Coreset size: {core_size}")

    for bm in benchmarks:
        if bm =='OnlineMMDPlus':
            Xc, yc, w = run_onlinemmdplus(train_loader, X_train, y_train, gamma, n_rff, seed, core_size,
                                          buffer_cap, online_epochs, lr_online, lambda_online, device)
            acc_final = train_classifier(Xc, X_val, yc, y_val)
            mmd_final = calculate_mmd2_exact(X_train, Xc, w, gamma)
            print(f"OnlineMMDPlus -> acc:{acc_final:.4f}, mmd:{mmd_final:.6f}")

        if bm == 'Reservoir':
            accs = []
            mmds = []
            for t in range(reservoir_trials):
                Xc, yc, w = run_reservoir(train_loader, X_train, y_train, core_size, seed+t)
                acc_final = train_classifier(Xc, X_val, yc, y_val)
                mmd_final = calculate_mmd2_exact(X_train, Xc, w, gamma)
                accs.append(acc_final)
                mmds.append(mmd_final)
            print(f"Reservoir -> acc:{np.mean(accs):.4f}, mmd:{np.mean(mmds):.6f}")



if __name__ == "__main__":

    config = {
    "dataset": "cifar10",
    "embedding": 'resnet18',
    "embed_dim": 50,
    "benchmarks": ["OnlineMMDPlus", "Reservoir"],  # only these
    "coreset_size": 150,
    "dataset_subset_size": 10000,
    "batch_size": 50,
    "n_rff_components": 1000,
    "kernel_gamma": 0.001,
    "buffer_capacity": 150,
    "random_seed": 2710,
    "n_epochs_online": 20,
    "lr_online": 0.1,
    "lambda_log_online": 1e-7,
    "reservoir_trials": 10,
}


    # Example config printing
    print("Using config:", config)

    run_single_experiment(config)