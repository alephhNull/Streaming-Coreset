import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# make sure we can import run_single_experiment and print_experiment_summary
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from streaming_coreset_runner import run_single_experiment
from visualize import print_experiment_summary

if __name__ == "__main__":

    # base config (you provided)
    base_config = {
        "dataset": "cifar10",
        "embedding": 'resnet18',
        "embed_dim": None,
        "benchmarks": ["MDH", "WKH", "Reservoir"],  # only these
        "coreset_size": 200,   # will overwrite below
        "dataset_subset_size": 2500,
        "batch_size": 50,
        "n_rff_components": 1000,
        "kernel_gamma": 0.001,
        "buffer_capacity": 200,  # will overwrite below
        "random_seed": 1421380,
        "n_epochs_online": 20,
        "lr_online": 0.1,
        "lambda_log_online": 1e-7,
        "reservoir_trials": 5,
        "n_classes": 10,
        "tasks": ['logistic_regression'],
        "dist_metrics": ['MMD'],
        "metrics": ['acc'],
        "streaming_metrics": ['avg_batch_time_ms'],
        "wkh_trials": 1,
        "onlinemmdplus_trials": 1,
        "mdh_trials": 1,
        "supersampling_trials": 1,
        "md_iterations": 100,
        "md_eta": 10
    }

    coreset_sizes = list(range(50, 501, 50))  # 50,100,...,500
    benchmarks = base_config["benchmarks"]

    # labels for plots
    label_map = {
        "MDH": "Mirror Descent Herding",
        "WKH": "Weighted Kernel Herding",
        "Reservoir": "Reservior sampling"  # spelling per your request
    }

    # storage
    avg_time_by_bench = {b: [] for b in benchmarks}
    mmd_by_bench = {b: [] for b in benchmarks}

    # run all experiments
    for size in coreset_sizes:
        config = dict(base_config)  # copy
        config["coreset_size"] = size
        config["buffer_capacity"] = size

        print(f"\n[RUN] coreset_size={size}, buffer_capacity={size}")
        result = run_single_experiment(config)

        # optional: save each result
        with open(f"mnist_{size}_experiment_result.json", "w") as f:
            json.dump(result, f, indent=2)

        # extract metrics
        for b in benchmarks:
            try:
                avg_time = float(result[b][0]['streaming_metrics']['avg_batch_time_ms'])
            except Exception:
                avg_time = np.nan
            try:
                mmd = float(result[b][0]['dist']['MMD'])
            except Exception:
                mmd = np.nan

            avg_time_by_bench[b].append(avg_time)
            mmd_by_bench[b].append(mmd)

        # you can also print summaries per run
        # print_experiment_summary(config, result)

    # ---------------- plotting ----------------
    xs = np.array(coreset_sizes)
    styles = {
        "MDH": ("-o", 7),
        "WKH": ("-s", 6),
        "Reservoir": ("-^", 6)
    }

    # avg_batch_time_ms plot
    plt.figure(figsize=(9,5))
    for b in benchmarks:
        ys = np.array(avg_time_by_bench[b], dtype=float)
        plt.plot(xs, ys, styles[b][0], label=label_map[b], markersize=styles[b][1])
    plt.xlabel("Coreset size")
    plt.ylabel("avg_batch_time_ms")
    plt.title("Average batch time (ms) vs coreset size")
    plt.xticks(xs, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("mnist_coreset_avg_batch_time_ms.png", dpi=200)
    plt.show()

    # MMD plot
    plt.figure(figsize=(9,5))
    for b in benchmarks:
        ys = np.array(mmd_by_bench[b], dtype=float)
        plt.plot(xs, ys, styles[b][0], label=label_map[b], markersize=styles[b][1])
    plt.xlabel("Coreset size")
    plt.ylabel("MMD")
    plt.title("MMD vs coreset size")
    plt.xticks(xs, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("mnist_coreset_mmd.png", dpi=200)
    plt.show()

    print("\n[SAVED] Plots generated successfully.")
