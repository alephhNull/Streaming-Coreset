from tabulate import tabulate
import numpy as np
import math


# Helper to round or stringify
def _fmt(x, digits=4):
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return 'nan' if math.isnan(x) else round(x, digits)
    return str(x)


def print_experiment_summary(config: dict,
                             result: dict,
                             config_keys=None):
    

    if config_keys is None:
        config_keys = [
            "dataset", "embedding", "embed_dim", "coreset_size", "dataset_subset_size",
            "batch_size", "n_rff_components", "kernel_gamma", "buffer_capacity", 
            "n_epochs_online", "lambda_log_online", "arrival_interval"
        ]

    config_table = [[k, config[k]] for k in config_keys if k in config]
    print("\n=== Experiment Configuration ===")
    print(tabulate(config_table, headers=["Parameter", "Value"], tablefmt="grid"))
    
    tasks      = config['tasks']
    down_ms    = config.get('metrics', [])             # e.g. ['f1']
    stream_ms  = config.get('streaming_metrics', [])   # e.g. ['avg_batch_time_ms']
    dist_ms    = config.get('dist_metrics', [])        # e.g. ['MMD']

    # Build headers:
    headers = ["Benchmark"]
    # downstream metrics per task
    for task in tasks:
        for m in down_ms:
            headers.append(f"{task}:{m}")
    # dist metrics (global)
    for dm in dist_ms:
        headers.append(dm)
    # streaming metrics (global)
    for sm in stream_ms:
        headers.append(sm)

    rows = []
    # 1) whole_data baseline
    row = ["whole_data"]
    for task in tasks:
        wd = result['whole_data'][task]
        for m in down_ms:
            row.append(_fmt(wd.get(m, '')))
    # baseline has no dist or streaming
    row += ['-' for _ in (dist_ms + stream_ms)]
    rows.append(row)

    # 2) each streaming benchmark
    for bm in config['benchmarks']:
        row = [bm]
        # downstream per task
        for task in tasks:
            entries = result[bm][task]
            for m in down_ms:
                vals = [e.get(m, np.nan) for e in entries]
                mean = np.nanmean(vals) if vals else np.nan
                row.append(_fmt(mean))
        # dist metrics (pick from any task; they're identical across tasks)
        # we use the first task
        first_entries = result[bm][tasks[0]]
        for dm in dist_ms:
            vals = [e.get('dist', {}).get(dm, np.nan) for e in first_entries]
            mean = np.nanmean(vals) if vals else np.nan
            row.append(_fmt(mean, digits=6))
        # streaming metrics (global)
        sm_list = result[bm]['streaming_metrics']
        for sm in stream_ms:
            vals = [s.get(sm, np.nan) for s in sm_list]
            mean = np.nanmean(vals) if vals else np.nan
            row.append(_fmt(mean))
        rows.append(row)

    print("\n=== Experiment Summary ===")
    print(tabulate(rows, headers=headers, tablefmt="github"))
