import argparse
import numpy as np
from sklearn.kernel_approximation import RBFSampler
import matplotlib.pyplot as plt
from torch import device
from bcsstreamer import BilevelCoresetSelector
from ocsstreamer import OCSStreamer
from reservoirstreamer import ReservoirSamplerBatchStreamer
from mmdplusstreamer import OnlineMMDPlusStreamer
from dataloaders import load_dataset
from utils import calculate_mmd2_exact
from downstream_tasks import train_classifier
from streaming_utils import stream_simulator_gen
import time
import torch


def run_streaming_algorithm(streamer, train_loader, X_train, y_train, arrival_interval_ms, verbose=False):
    ## METRICS TRACKING: Initialize timers and counters
    batch_processing_times = []
    stream_start_time = time.monotonic()
    final_stats = {}

    # Use the new high-velocity simulator instead of the raw loader
    stream_simulator = stream_simulator_gen(train_loader, arrival_interval_ms)

    for batch_idx, (batch_x, _), stats in stream_simulator:
        batch_np = batch_x.cpu().numpy()
        
        batch_start_time = time.monotonic()
        if batch_np.shape[0]:
            # ✅ CORRECTED: Pass the global batch_idx to the streamer
            streamer.process_batch(batch_np, batch_idx)
        batch_end_time = time.monotonic()
        
        batch_processing_times.append(batch_end_time - batch_start_time)
        final_stats = stats # Keep track of the latest stats

    stream_end_time = time.monotonic()
    total_stream_time = stream_end_time - stream_start_time
    
    ## METRICS TRACKING: Calculate new performance metrics
    processed_count = final_stats.get('processed', 0)
    dropped_count = final_stats.get('dropped', 0)
    total_count = processed_count + dropped_count

    # Avg time for the batches that were ACTUALLY processed
    avg_batch_time = np.mean(batch_processing_times) if batch_processing_times else 0
    
    # Throughput based on what was successfully processed over the total time
    effective_throughput = (processed_count * train_loader.batch_size) / total_stream_time if total_stream_time > 0 else 0

    # Data loss due to inability to keep up with stream velocity
    velocity_data_loss_pct = (dropped_count / total_count) * 100 if total_count > 0 else 0
    
    metrics = {
        'avg_batch_time_ms': 1000*avg_batch_time,
        'effective_throughput_pps': effective_throughput,
        'velocity_data_loss_pct': velocity_data_loss_pct,
        'batches_processed': processed_count,
        'batches_dropped': dropped_count,
    }

    idxs, w, _ = streamer.get_final_coreset()
    Xc, yc = X_train[idxs], y_train[idxs]

    if verbose:
        streamer.print_coreset_provenance()

    return Xc, yc, w, metrics


def run_onlinemmdplus(train_loader, X_train, y_train, gamma, n_rff, seed, coreset_size, 
                      buffer_cap, online_epochs, lr_online, lambda_online, device, 
                      arrival_interval_ms): ## METRICS TRACKING: New parameter
    
    rbf = RBFSampler(gamma=gamma, n_components=n_rff, random_state=seed)
    rbf.fit(X_train)

    streamer = OnlineMMDPlusStreamer(
        m_coreset_size=coreset_size, batch_size=train_loader.batch_size, n_rff_components=n_rff,
        buffer_capacity=buffer_cap, n_epochs_online=online_epochs,
        lr_online=lr_online, lambda_log_online=lambda_online,
        random_seed=seed, device=device
    )
    
    streamer.set_rbf_sampler(rbf)

    Xc, yc, w, metrics = run_streaming_algorithm(streamer, train_loader, X_train, y_train, arrival_interval_ms, True)

    return Xc, yc, w, metrics


def run_reservoir(train_loader, X_train, y_train, coreset_size, seed, arrival_interval_ms):
    r = ReservoirSamplerBatchStreamer(coreset_size, seed)
    
    # run_streaming_algorithm will handle the batch iteration and data accumulation
    Xc, yc, w, metrics = run_streaming_algorithm(r, train_loader, X_train, y_train, arrival_interval_ms)
    
    return Xc, yc, w, metrics



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
    arrival_interval_ms = config.get('arrival_interval', None)

    reservoir_trials = config['reservoir_trials']


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, X_train, X_val, y_train, y_val = load_dataset(dataset_name, ds_size, batch_size, seed, embedding, embed_dim, device)

    acc_whole, auc_whole, f1_whole = train_classifier(X_train, X_val, y_train, y_val)
    print(f"Baseline (whole dataset) accuracy: {acc_whole:.4f}, auc: {auc_whole:.4f}, f1: {f1_whole:.4f}")

    print(f"\n== Coreset size: {core_size}")

    for bm in benchmarks:
        if bm =='OnlineMMDPlus':
            Xc, yc, w, metrics = run_onlinemmdplus(train_loader, X_train, y_train, gamma, n_rff, seed, core_size,
                                          buffer_cap, online_epochs, lr_online, lambda_online, device, arrival_interval_ms)
            acc_final, auc_final, f1_final = train_classifier(Xc, X_val, yc, y_val)
            mmd_final = calculate_mmd2_exact(X_train, Xc, w, gamma)
            print(f"\nOnlineMMDPlus -> acc:{acc_final:.4f}, mmd:{mmd_final:.6f}, auc:{auc_final:.4f}, f1:{f1_final:.4f}")
            print(f"  └─ Stream Perf: Batches Processed: {metrics['batches_processed']}, Dropped: {metrics['batches_dropped']}")
            print(f"  └─ Metrics: Avg Proc Time: {metrics['avg_batch_time_ms']:.2f}ms | "
                  f"Throughput: {int(metrics['effective_throughput_pps'])} pts/sec | "
                  f"Data Loss (Velocity): {metrics['velocity_data_loss_pct']:.2f}%")

        if bm == 'Reservoir':
            accs = []
            mmds = []
            aucs = []
            f1s = []
            for t in range(reservoir_trials):
                Xc, yc, w, metrics = run_reservoir(train_loader, X_train, y_train, core_size, seed+t, arrival_interval_ms)
                acc_final, auc_final, f1_final = train_classifier(Xc, X_val, yc, y_val)
                mmd_final = calculate_mmd2_exact(X_train, Xc, w, gamma)
                accs.append(acc_final)
                mmds.append(mmd_final)
                aucs.append(auc_final)
                f1s.append(f1_final)
            print(f"Reservoir -> acc:{np.mean(accs):.4f}, mmd:{np.mean(mmds):.6f}, auc:{np.mean(aucs):.4f}, f1:{np.mean(f1s):.4f}")
