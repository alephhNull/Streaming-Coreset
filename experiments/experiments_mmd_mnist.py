import sys 
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from streaming_coreset_runner import run_single_experiment
from visualize import print_experiment_summary, plot_mmd_vs_coreset_size, plot_wasserstein_vs_coreset_size
import json



if __name__ == "__main__":

    m_s = [30, 50, 70, 100, 150, 200, 300]
    buffer_capacities = [30, 50, 70, 100, 150, 200, 300]
    lambda_logs = [5e-6, 2e-6, 1e-6, 5e-7, 1e-7, 8e-8, 1e-8]

    config_base = {
    "dataset": "mnist",
    "embedding": 'resnet18',
    "embed_dim": 50,
    "benchmarks": ["OnlineMMDPlus", "CO2", "CAMEL", "FreeSel", "GSS", "SSD", "SuperSampling", "MMD_Critic", "WCSL", "Reservoir"],  # only these
    "dataset_subset_size": 2500,
    "batch_size": 50,
    "n_rff_components": 1000,
    "kernel_gamma": 0.001,
    "buffer_capacity": 150,
    "random_seed": 1421380,
    "n_epochs_online": 20,
    "lr_online": 0.1,
    "reservoir_trials": 5,
    "n_classes": 10,
    "dist_metrics": ['MMD', '1-Wasserstein'],
    "metrics": ['acc'],
    "streaming_metrics": [],
    "tasks": ["logistic_regression"],
    "onlinemmdplus_trials": 5,
    "supersampling_trials": 5,
}
    

    config_result_pairs = []

    for coreset_size, buffer_capacity, lambda_log in zip(m_s, buffer_capacities, lambda_logs):
        config = config_base.copy()
        config["coreset_size"] = coreset_size
        config["buffer_capacity"] = buffer_capacity
        config["lambda_log_online"] = lambda_log
        result = run_single_experiment(config)
        config_result_pairs.append((config, result))


    with open("results/mnist_dist_metrics_vs_coreset_size.json", "w") as f:
            json.dump(config_result_pairs, f, indent=2)

    # with open("mnist_dist_metrics_vs_coreset_size.json", "r") as f:
    #     config_result_pairs = json.load(f)

    plot_mmd_vs_coreset_size(config_result_pairs)
    plot_wasserstein_vs_coreset_size(config_result_pairs)





