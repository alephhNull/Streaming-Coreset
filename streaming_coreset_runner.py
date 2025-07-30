import argparse
import numpy as np
from sklearn.kernel_approximation import RBFSampler
import matplotlib.pyplot as plt
from torch import device
from streamers.bcsstreamer import BilevelCoresetSelector
from streamers.mmd_critic_streamer import MMDCriticStreamer
from streamers.ocsstreamer import OCSStreamer
from streamers.reservoirstreamer import ReservoirSamplerBatchStreamer
from streamers.co2_streamer import CO2Streamer
from streamers.mmdplusstreamer import OnlineMMDPlusStreamer
from streamers.wcsl_streamer import WCSLStreamer
from streamers.camel_streamer import CAMELStreamer
from streamers.freesel_streamer import FreeSelStreamer
from streamers.gss_streamer import GSSStreamer
from streamers.ssd_streamer import SSDStreamerGeneric

from dataloaders import load_dataset
from utils import calculate_mmd2_exact, calculate_wass_distance
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

    for batch_idx, (batch_x, batch_y), stats in stream_simulator:
        batch_np = batch_x.cpu().numpy()
        batch_y_np = batch_y.cpu().numpy()

        batch_start_time = time.monotonic()
        if batch_np.shape[0]:
            # ✅ CORRECTED: Pass the global batch_idx to the streamer
            streamer.process_batch(batch_np, batch_y_np, batch_idx)
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
    r = ReservoirSamplerBatchStreamer(coreset_size, train_loader.batch_size, seed)
    
    # run_streaming_algorithm will handle the batch iteration and data accumulation
    Xc, yc, w, metrics = run_streaming_algorithm(r, train_loader, X_train, y_train, arrival_interval_ms)
    
    return Xc, yc, w, metrics


def run_co2(train_loader, X_train, y_train, coreset_size, buffer_capacity, seed, arrival_interval_ms):
    SINKHORN_REG = 2 * X_train.shape[1]

    streamer = CO2Streamer(
        wanted_coreset_size=coreset_size,
        buffer_capacity=buffer_capacity,
        batch_size=train_loader.batch_size,
        random_seed=seed,
        reg=SINKHORN_REG
    )
    
    # run_streaming_algorithm will handle the batch iteration and data accumulation
    Xc, yc, w, metrics = run_streaming_algorithm(streamer, train_loader, X_train, y_train, arrival_interval_ms)
    
    return Xc, yc, w, metrics


def run_wcsl(train_loader, X_train, y_train, coreset_size, buffer_capacity, seed, arrival_interval_ms):
    SINKHORN_REG = 2 * X_train.shape[1]
    
    wcsl_streamer = WCSLStreamer(target_coreset_size=coreset_size,
                                 lambda_reg=SINKHORN_REG,
                                 buffer_capacity=buffer_capacity,
                                 batch_size=train_loader.batch_size,
                                 random_seed=seed)

    # run_streaming_algorithm will handle the batch iteration and data accumulation
    Xc, yc, w, metrics = run_streaming_algorithm(wcsl_streamer, train_loader, X_train, y_train, arrival_interval_ms)

    return Xc, yc, w, metrics

def run_mmd_critic(train_loader, X_train, y_train, coreset_size, buffer_capacity, gamma, seed, arrival_interval_ms):
    mmd_critic_streamer = MMDCriticStreamer(
        target_coreset_size=coreset_size,
        buffer_capacity=buffer_capacity,
        batch_size=train_loader.batch_size,
        random_seed=seed,
        gamma=gamma
    )

    # run_streaming_algorithm will handle the batch iteration and data accumulation
    Xc, yc, w, metrics = run_streaming_algorithm(mmd_critic_streamer, train_loader, X_train, y_train, arrival_interval_ms)

    return Xc, yc, w, metrics

def run_camel(train_loader, X_train, y_train, coreset_size, buffer_capacity, seed, arrival_interval_ms):
    camel_streamer = CAMELStreamer(
        buffer_capacity=buffer_capacity,
        coreset_size=coreset_size,
        batch_size=train_loader.batch_size,
        random_seed=seed
    )

    # run_streaming_algorithm will handle the batch iteration and data accumulation
    Xc, yc, w, metrics = run_streaming_algorithm(camel_streamer, train_loader, X_train, y_train, arrival_interval_ms)

    return Xc, yc, w, metrics

def run_freesel(train_loader, X_train, y_train, coreset_size, buffer_capacity, seed, arrival_interval_ms):
    freesel_streamer = FreeSelStreamer(
        buffer_capacity=buffer_capacity,
        coreset_size=coreset_size,
        batch_size=train_loader.batch_size,
        random_seed=seed
    )

    # run_streaming_algorithm will handle the batch iteration and data accumulation
    Xc, yc, w, metrics = run_streaming_algorithm(freesel_streamer, train_loader, X_train, y_train, arrival_interval_ms)

    return Xc, yc, w, metrics

def run_gss(train_loader, X_train, y_train, n_classes, coreset_size, buffer_capacity, seed, arrival_interval_ms):


    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gss_streamer = GSSStreamer(coreset_size, buffer_capacity, X_train.shape[1], n_classes, device,
                                train_loader.batch_size, random_seed=seed)


    # run_streaming_algorithm will handle the batch iteration and data accumulation
    Xc, yc, w, metrics = run_streaming_algorithm(gss_streamer, train_loader, X_train, y_train, arrival_interval_ms)

    return Xc, yc, w, metrics


def run_ssd(train_loader, X_train, y_train, n_classes, coreset_size, buffer_capacity, seed, arrival_interval_ms):
    ssd_streamer = SSDStreamerGeneric(
        buffer_capacity=buffer_capacity,
        target_coreset_size=coreset_size,
        num_classes=n_classes,
        random_seed=seed,
        feature_dim=X_train.shape[1]
    )

    # run_streaming_algorithm will handle the batch iteration and data accumulation
    Xc, yc, w, metrics = run_streaming_algorithm(ssd_streamer, train_loader, X_train, y_train, arrival_interval_ms, verbose=True)

    return Xc, yc, w, metrics

def run_single_experiment(config):
    dataset_name = config['dataset']
    ds_size = config['dataset_subset_size']
    seed = config['random_seed']
    embedding = config['embedding']
    embed_dim = config['embed_dim']
    batch_size = config['batch_size']
    n_classes = config.get('n_classes', 2)

    n_rff = config['n_rff_components']
    gamma = config['kernel_gamma']
    buffer_cap = config['buffer_capacity']
    online_epochs = config['n_epochs_online']
    lr_online = config['lr_online']
    lambda_online = config['lambda_log_online']

    benchmarks = config['benchmarks']
    core_size = config['coreset_size']
    arrival_interval_ms = config.get('arrival_interval', None)

    n_mmd_trials = config.get('mmd_trials', 1)
    n_co2_trials = config.get('co2_trials', 1)
    n_wcsl_trials = config.get('wcsl_trials', 1)
    n_mmd_critic_trials = config.get('mmd_critic_trials', 1)
    n_camel_trials = config.get('camel_trials', 1)
    n_freesel_trials = config.get('freesel_trials', 1)
    n_gss_trials = config.get('gss_trials', 1)
    n_ssd_trials = config.get('ssd_trials', 1)
    n_reservoir_trials = config.get('reservoir_trials', 10)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    experiment_result = {bm: {} for bm in benchmarks}

    train_loader, val_loader, X_train, X_val, y_train, y_val = load_dataset(dataset_name, ds_size, batch_size, seed, embedding, embed_dim, device)

    acc_whole, auc_whole, f1_whole = train_classifier(X_train, X_val, y_train, y_val)

    experiment_result['whole_data'] = {
        'accuracy': acc_whole,
        'auc': auc_whole,
        'f1': f1_whole
    }

    print(f"\n== Coreset size: {core_size}")

    for bm in benchmarks:
        experiment_result[bm] = []
        if bm =='OnlineMMDPlus':
            for t in range(n_mmd_trials): 
                Xc, yc, w, metrics = run_onlinemmdplus(train_loader, X_train, y_train, gamma, n_rff, seed+t, core_size,
                                              buffer_cap, online_epochs, lr_online, lambda_online, device, arrival_interval_ms)
                acc_final, auc_final, f1_final = train_classifier(Xc, X_val, yc, y_val)
                mmd_final = calculate_mmd2_exact(X_train, Xc, w, gamma)
                W1_final = calculate_wass_distance(X_train, Xc, w)
                experiment_result[bm].append({
                    'trial': t,
                    'accuracy': acc_final,
                    'auc': auc_final,
                    'f1': f1_final,
                    'mmd': mmd_final,
                    'W1': W1_final,
                    'streaming_metrics': metrics
                })

        if bm == 'Reservoir':
            for t in range(n_reservoir_trials):
                Xc, yc, w, metrics = run_reservoir(train_loader, X_train, y_train, core_size, seed+t, arrival_interval_ms)
                acc_final, auc_final, f1_final = train_classifier(Xc, X_val, yc, y_val)
                mmd_final = calculate_mmd2_exact(X_train, Xc, w, gamma)
                W1_final = calculate_wass_distance(X_train, Xc, w)
                experiment_result[bm].append({
                    'trial': t,
                    'accuracy': acc_final,
                    'auc': auc_final,
                    'f1': f1_final,
                    'mmd': mmd_final,
                    'W1': W1_final,
                    'streaming_metrics': metrics
                })
            
        if bm == 'CO2':
            for t in range(n_co2_trials):
                Xc, yc, w, metrics = run_co2(train_loader, X_train, y_train, core_size, buffer_cap, seed+t, arrival_interval_ms)
                acc_final, auc_final, f1_final = train_classifier(Xc, X_val, yc, y_val)
                mmd_final = calculate_mmd2_exact(X_train, Xc, w, gamma)
                W1_final = calculate_wass_distance(X_train, Xc, w)
                experiment_result[bm].append({
                    'trial': t,
                    'accuracy': acc_final,
                    'auc': auc_final,
                    'f1': f1_final,
                    'mmd': mmd_final,
                    'W1': W1_final,
                    'streaming_metrics': metrics
                })

        if bm == 'WCSL':
            for t in range(n_wcsl_trials):
                Xc, yc, w, metrics = run_wcsl(train_loader, X_train, y_train, core_size, buffer_cap, seed+t, arrival_interval_ms)
                acc_final, auc_final, f1_final = train_classifier(Xc, X_val, yc, y_val)
                mmd_final = calculate_mmd2_exact(X_train, Xc, w, gamma)
                W1_final = calculate_wass_distance(X_train, Xc, w)
                experiment_result[bm].append({
                    'trial': t,
                    'accuracy': acc_final,
                    'auc': auc_final,
                    'f1': f1_final,
                    'mmd': mmd_final,
                    'W1': W1_final,
                    'streaming_metrics': metrics
                })

        if bm == 'MMD_Critic':
            for t in range(n_mmd_critic_trials):
                Xc, yc, w, metrics = run_mmd_critic(train_loader, X_train, y_train, core_size, buffer_cap, gamma, seed+t, arrival_interval_ms)
                acc_final, auc_final, f1_final = train_classifier(Xc, X_val, yc, y_val)
                mmd_final = calculate_mmd2_exact(X_train, Xc, w, gamma)
                W1_final = calculate_wass_distance(X_train, Xc, w)
                experiment_result[bm].append({
                    'trial': t,
                    'accuracy': acc_final,
                    'auc': auc_final,
                    'f1': f1_final,
                    'mmd': mmd_final,
                    'W1': W1_final,
                    'streaming_metrics': metrics
                })

        if bm == 'CAMEL':
            for t in range(n_camel_trials):
                Xc, yc, w, metrics = run_camel(train_loader, X_train, y_train, core_size, buffer_cap, seed+t, arrival_interval_ms)
                acc_final, auc_final, f1_final = train_classifier(Xc, X_val, yc, y_val)
                mmd_final = calculate_mmd2_exact(X_train, Xc, w, gamma)
                W1_final = calculate_wass_distance(X_train, Xc, w)
                experiment_result[bm].append({
                    'trial': t,
                    'accuracy': acc_final,
                    'auc': auc_final,
                    'f1': f1_final,
                    'mmd': mmd_final,
                    'W1': W1_final,
                    'streaming_metrics': metrics
                })

        if bm == 'FreeSel':
            for t in range(n_freesel_trials):
                Xc, yc, w, metrics = run_freesel(train_loader, X_train, y_train, core_size, buffer_cap, seed+t, arrival_interval_ms)
                acc_final, auc_final, f1_final = train_classifier(Xc, X_val, yc, y_val)
                mmd_final = calculate_mmd2_exact(X_train, Xc, w, gamma)
                W1_final = calculate_wass_distance(X_train, Xc, w)
                experiment_result[bm].append({
                    'trial': t,
                    'accuracy': acc_final,
                    'auc': auc_final,
                    'f1': f1_final,
                    'mmd': mmd_final,
                    'W1': W1_final,
                    'streaming_metrics': metrics
                })
            
        if bm == 'GSS':
            for t in range(n_gss_trials):
                Xc, yc, w, metrics = run_gss(train_loader, X_train, y_train, n_classes, core_size, buffer_cap, seed+t, arrival_interval_ms)
                acc_final, auc_final, f1_final = train_classifier(Xc, X_val, yc, y_val)
                mmd_final = calculate_mmd2_exact(X_train, Xc, w, gamma)
                W1_final = calculate_wass_distance(X_train, Xc, w)
                experiment_result[bm].append({
                    'trial': t,
                    'accuracy': acc_final,
                    'auc': auc_final,
                    'f1': f1_final,
                    'mmd': mmd_final,
                    'W1': W1_final,
                    'streaming_metrics': metrics
                })

        if bm == 'SSD':
            for t in range(n_ssd_trials):
                Xc, yc, w, metrics = run_ssd(train_loader, X_train, y_train, n_classes, core_size, buffer_cap, seed+t, arrival_interval_ms)
                acc_final, auc_final, f1_final = train_classifier(Xc, X_val, yc, y_val)
                mmd_final = calculate_mmd2_exact(X_train, Xc, w, gamma)
                W1_final = calculate_wass_distance(X_train, Xc, w)
                experiment_result[bm].append({
                    'trial': t,
                    'accuracy': acc_final,
                    'auc': auc_final,
                    'f1': f1_final,
                    'mmd': mmd_final,
                    'W1': W1_final,
                    'streaming_metrics': metrics
                })

    return experiment_result