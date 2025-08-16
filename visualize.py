from tabulate import tabulate
import numpy as np
import math
import matplotlib.pyplot as plt


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
            entries = result[bm]  # Now result[bm] is a list of trials
            for m in down_ms:
                vals = [e.get(task, {}).get(m, np.nan) for e in entries]
                mean = np.nanmean(vals) if vals else np.nan
                row.append(_fmt(mean))
        # dist metrics (pick from any trial; they're identical across trials)
        # we use the first trial
        first_trial = result[bm][0] if result[bm] else {}
        for dm in dist_ms:
            vals = [e.get('dist', {}).get(dm, np.nan) for e in result[bm]]
            mean = np.nanmean(vals) if vals else np.nan
            row.append(_fmt(mean, digits=6))
        # streaming metrics (global)
        for sm in stream_ms:
            vals = [e.get('streaming_metrics', {}).get(sm, np.nan) for e in result[bm]]
            mean = np.nanmean(vals) if vals else np.nan
            row.append(_fmt(mean))
        rows.append(row)

    print("\n=== Experiment Summary ===")
    print(tabulate(rows, headers=headers, tablefmt="github"))


def _collect_dist_curves(config_result_pairs, metric_name: str):
    """
    Build {benchmark: {coreset_size: mean_metric_over_trials}} from one or more runs.

    Parameters
    ----------
    config_result_pairs : list[tuple[dict, dict]] | tuple[dict, dict]
        Either a list of (config, result) pairs (for multiple coreset sizes),
        or a single (config, result) pair (plots one point).
    metric_name : str
        One of the names in config['dist_metrics'], e.g. 'MMD' or '1-Wasserstein'.

    Returns
    -------
    curves : dict[str, dict[int, float]]
        Mapping benchmark -> {coreset_size -> mean_value_over_trials}.
    """

    # Allow passing a single (config, result)
    if isinstance(config_result_pairs, tuple) and len(config_result_pairs) == 2:
        config_result_pairs = [config_result_pairs]

    curves_raw = {}  # {bm: {size: [means_per_run]}}
    for config, result in config_result_pairs:
        size = int(config['coreset_size'])
        tasks = config['tasks']
        benchmarks = config['benchmarks']

        # We follow your table logic: distances are identical across trials,
        # so we use any trial.
        for bm in benchmarks:
            entries = result[bm]  # list over trials
            # Mean over trials for this run/size:
            vals = [e.get('dist', {}).get(metric_name, np.nan) for e in entries]
            run_mean = np.nanmean(vals) if len(vals) > 0 else np.nan

            curves_raw.setdefault(bm, {}).setdefault(size, []).append(run_mean)

    # Collapse across runs for the same size (e.g., if you repeated the whole run for that size)
    curves = {}
    for bm, by_size in curves_raw.items():
        curves[bm] = {sz: float(np.nanmean(vs)) for sz, vs in by_size.items()}

    return curves


def _plot_curves(curves, metric_name: str, title: str = None, save_path: str = None):
    """
    Plot helper for curves produced by _collect_dist_curves.
    """
    # Use a professional style
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(7, 5))
    colormap = plt.cm.get_cmap("tab10")  # good discrete colormap

    for i, (bm, size_to_val) in enumerate(curves.items()):
        if not size_to_val:
            continue
        sizes = sorted(size_to_val.keys())
        ys = [size_to_val[s] for s in sizes]
        ys = np.log(ys)  # keep your log transform

        ax.plot(
            sizes, ys,
            marker='o',
            label=bm,
            color=colormap(i % colormap.N),
            linewidth=2,
            markersize=6
        )

    ax.set_xlabel("Coreset Size", fontsize=13)
    ax.set_ylabel(metric_name, fontsize=13)
    ax.set_title(title or f"{metric_name} vs Coreset Size", fontsize=14, pad=15)

    # Legend outside the plot
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=False,
        fontsize=11
    )

    # Make tick labels bigger
    ax.tick_params(axis='both', which='major', labelsize=11)

    plt.tight_layout()
    fig.savefig(save_path, format="pdf")
    plt.show()




def plot_mmd_vs_coreset_size(config_result_pairs):
    """
    Plot MMD vs coreset size.

    Parameters
    ----------
    config_result_pairs : list[tuple[dict, dict]] | tuple[dict, dict]
        - Pass a list of (config, result) pairs gathered from multiple runs with
          different coreset sizes to get a curve.
        - Passing a single (config, result) will produce a single point.
    """
    curves = _collect_dist_curves(config_result_pairs, metric_name='MMD')
    dataset_name  = config_result_pairs[0][0]['dataset']
    _plot_curves(curves, metric_name='MMD', title=f'{dataset_name} MMD vs Coreset Size', save_path=f'results/{dataset_name}_mmd_vs_coreset_size.pdf')
    

def plot_wasserstein_vs_coreset_size(config_result_pairs):
    """
    Plot 1-Wasserstein distance vs coreset size.

    Parameters
    ----------
    config_result_pairs : list[tuple[dict, dict]] | tuple[dict, dict]
        - Pass a list of (config, result) pairs gathered from multiple runs with
          different coreset sizes to get a curve.
        - Passing a single (config, result) will produce a single point.
    """
    curves = _collect_dist_curves(config_result_pairs, metric_name='1-Wasserstein')
    dataset_name  = config_result_pairs[0][0]['dataset']
    _plot_curves(curves, metric_name='1-Wasserstein', title=f'{dataset_name} 1-Wasserstein vs Coreset Size', save_path=f'results/{dataset_name}_1-wasserstein_vs_coreset_size.pdf')
