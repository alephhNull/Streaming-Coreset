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
from streamers.supersampling_streamer import SupersamplingCoreset
from streamers.wcsl_streamer import WCSLStreamer
from streamers.camel_streamer import CAMELStreamer
from streamers.freesel_streamer import FreeSelStreamer
from streamers.gss_streamer import GSSStreamer
from streamers.ssd_streamer import SSDStreamer
from streamers.rff_wkh_streamer import WKHStreamingCoreset

from dataloaders import load_dataset
from utils import calculate_mmd2_exact, calculate_wass_distance
from downstream_tasks import *
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


def run_wkh(train_loader, X_train, y_train, coreset_size, buffer_capacity, n_rff, gamma, seed, arrival_interval_ms):
    sampler=RBFSampler(gamma=gamma, n_components=n_rff, random_state=seed)
    sampler.fit(X_train)

    wkh_streamer = WKHStreamingCoreset(
        coreset_size=coreset_size,
        buffer_capacity=buffer_capacity,
        sampler=sampler,
        batch_size=train_loader.batch_size
    )

    # run_streaming_algorithm will handle the batch iteration and data accumulation
    Xc, yc, w, metrics = run_streaming_algorithm(wkh_streamer, train_loader, X_train, y_train, arrival_interval_ms)

    return Xc, yc, w, metrics

def run_super_sampling(train_loader, X_train, y_train, coreset_size, buffer_capacity, n_rff, seed, arrival_interval_ms):
    super_sampling_streamer = SupersamplingCoreset(
        target_coreset_size=coreset_size,
        buffer_size=buffer_capacity,
        batch_size=train_loader.batch_size,
        input_dim=X_train.shape[1],
        num_features=n_rff,
        gamma=1.0,
        seed=seed
    )

    # run_streaming_algorithm will handle the batch iteration and data accumulation
    Xc, yc, w, metrics = run_streaming_algorithm(super_sampling_streamer, train_loader, X_train, y_train, arrival_interval_ms)

    return Xc, yc, w, metrics

def run_ssd(train_loader, X_train, y_train, n_classes, coreset_size, buffer_capacity, seed, arrival_interval_ms):
    ssd_streamer = SSDStreamer(
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
    # Unpack config and parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load data
    train_loader, val_loader, X_train, X_val, y_train, y_val = load_dataset(
        config['dataset'], config['dataset_subset_size'], config['batch_size'],
        config['random_seed'], config['embedding'], config['embed_dim'], device
    )

    # Downstream tasks and metrics
    tasks = config['tasks']            # e.g. ['logistic_regression', 'SVM', 'KNN']
    metrics = config['metrics']        # e.g. ['accuracy', 'f1', 'auc']
    dist_metrics = config.get('dist_metrics', [])

    # Trials per benchmark
    benchmarks = config['benchmarks']
    trials = {bm: config.get(f"{bm.lower()}_trials", 1) for bm in benchmarks}

    # Prepare experiment_result
    experiment_result = {}
    # Baseline on whole data

    task_funcs = {
        'logistic_regression': train_logistic_regression,
        'SVM': train_svm_classifier,
        'KNN': train_knn_classifier,
        'NaiveBayes': train_naive_bayes_classifier,
        'RandomForest': train_random_forest_classifier,
        'XGBoost': train_xgboost_classifier,
        'linear_regression': train_linear_regression,
        'RF_regression': train_random_forest_regression,
        'XGB_regression': train_xgboost_regression,
        'KMeans': lambda Xtr, Xv, ytr, yv, w=None: train_kmeans_clustering(Xtr, ytr),
        'DBSCAN': lambda Xtr, Xv, ytr, yv, w=None: train_dbscan_clustering(Xtr, ytr),
        'Agglomerative': lambda Xtr, Xv, ytr, yv, w=None: train_agglomerative_clustering(Xtr, ytr),
    }
    # Map dist metric names to functions
    dist_funcs = {
        'MMD': calculate_mmd2_exact,
        '1-Wasserstein': calculate_wass_distance,
    }

    experiment_result['whole_data'] = {}
    for task in tasks:
        func = task_funcs[task]
        res = func(X_train, X_val, y_train, y_val)
        experiment_result['whole_data'][task] = res

    # Streaming benchmarks
    for bm in benchmarks:
        experiment_result[bm] = {task: [] for task in tasks}
        experiment_result[bm]['streaming_metrics'] = []

        for t in range(trials[bm]):
            # Choose the coreset method
            if bm == 'OnlineMMDPlus':
                Xc, yc, w, stream_meta = run_onlinemmdplus(
                    train_loader, X_train, y_train,
                    config['kernel_gamma'], config['n_rff_components'],
                    config['random_seed'] + t, config['coreset_size'],
                    config['buffer_capacity'], config['n_epochs_online'],
                    config['lr_online'], config['lambda_log_online'],
                    device, config.get('arrival_interval')
                )
            elif bm == 'Reservoir':
                Xc, yc, w, stream_meta = run_reservoir(
                    train_loader, X_train, y_train,
                    config['coreset_size'], config['random_seed'] + t,
                    config.get('arrival_interval')
                )
            elif bm == 'CO2':
                Xc, yc, w, stream_meta = run_co2(
                    train_loader, X_train, y_train,
                    config['coreset_size'], config['buffer_capacity'],
                    config['random_seed'] + t, config.get('arrival_interval')
                )
            elif bm == 'WCSL':
                Xc, yc, w, stream_meta = run_wcsl(
                    train_loader, X_train, y_train,
                    config['coreset_size'], config['buffer_capacity'],
                    config['random_seed'] + t, config.get('arrival_interval')
                )
            elif bm == 'MMD_Critic':
                Xc, yc, w, stream_meta = run_mmd_critic(
                    train_loader, X_train, y_train,
                    config['coreset_size'], config['buffer_capacity'],
                    config['kernel_gamma'], config['random_seed'] + t,
                    config.get('arrival_interval')
                )
            elif bm == 'CAMEL':
                Xc, yc, w, stream_meta = run_camel(
                    train_loader, X_train, y_train,
                    config['coreset_size'], config['buffer_capacity'],
                    config['random_seed'] + t, config.get('arrival_interval')
                )
            elif bm == 'FreeSel':
                Xc, yc, w, stream_meta = run_freesel(
                    train_loader, X_train, y_train,
                    config['coreset_size'], config['buffer_capacity'],
                    config['random_seed'] + t, config.get('arrival_interval')
                )
            elif bm == 'GSS':
                Xc, yc, w, stream_meta = run_gss(
                    train_loader, X_train, y_train,
                    config.get('n_classes', 2), config['coreset_size'],
                    config['buffer_capacity'], config['random_seed'] + t,
                    config.get('arrival_interval')
                )
            elif bm == 'SSD':
                Xc, yc, w, stream_meta = run_ssd(
                    train_loader, X_train, y_train,
                    config.get('n_classes', 2), config['coreset_size'],
                    config['buffer_capacity'], config['random_seed'] + t,
                    config.get('arrival_interval')
                )
            elif bm == 'WKH':
                Xc, yc, w, stream_meta = run_wkh(
                    train_loader, X_train, y_train,
                    config['coreset_size'], config['buffer_capacity'],
                    config['n_rff_components'], config['kernel_gamma'],
                    config['random_seed'] + t, config.get('arrival_interval')
                )
            elif bm == 'SuperSampling':
                Xc, yc, w, stream_meta = run_super_sampling(
                    train_loader, X_train, y_train,
                    config['coreset_size'], config['buffer_capacity'],
                    config['n_rff_components'], config['random_seed'] + t,
                    config.get('arrival_interval')
                )
            else:
                raise ValueError(f"Unknown benchmark: {bm}")
            
            # Assert coreset size
            
            assert Xc.shape[0] == config['coreset_size'], f"Coreset Xc shape {Xc.shape[0]} != {config['coreset_size']}"
            assert w.shape[0] == config['coreset_size'], f"Coreset weights shape {w.shape[0]} != {config['coreset_size']}"
            # Compute distribution metrics
            dist_vals = {}
            for dm in dist_metrics:
                if dm == 'MMD':
                    dist_vals['MMD'] = calculate_mmd2_exact(X_train, Xc, w, config['kernel_gamma'])
                elif dm == '1-Wasserstein':
                    dist_vals['1-Wasserstein'] = calculate_wass_distance(X_train, Xc, w)


            # Evaluate downstream tasks
            for task in tasks:
                res = task_funcs[task](Xc, X_val, yc, y_val, w)
                entry = {'trial': t, **res}
                if dist_vals:
                    entry['dist'] = dist_vals
                experiment_result[bm][task].append(entry)

            experiment_result[bm]['streaming_metrics'].append(stream_meta)

    return experiment_result