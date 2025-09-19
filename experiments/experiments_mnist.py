import sys 
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from streaming_coreset_runner import run_single_experiment
from visualize import print_experiment_summary
import json



if __name__ == "__main__":

    config = {
    "dataset": "mnist",
    "embedding": 'resnet18',
    "embed_dim": 50,
    "benchmarks": ["MDH", "WKH", "Reservoir"],  # only these
    "coreset_size": 150,
    "dataset_subset_size": 2500,
    "batch_size": 50,
    "n_rff_components": 1000,
    "kernel_gamma": 0.001,
    "buffer_capacity": 150,
    "random_seed": 1421380,
    "n_epochs_online": 20,
    "lr_online": 0.1,
    "lambda_log_online": 1e-7,
    "reservoir_trials": 5,
    "n_classes": 10,
    "tasks": ['logistic_regression'],
    "dist_metrics": ['MMD', '1-Wasserstein'],
    "metrics": ['acc'],
    "streaming_metrics": ['avg_batch_time_ms'],
    "wkh_trials": 1,
    "onlinemmdplus_trials": 1,
    "mdh_trials": 1,
    "supersampling_trials": 1,
    "md_iterations": 100,
    "md_eta": 1
}



    result = run_single_experiment(config)
    with open("mnist_150_experiment_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print_experiment_summary(config, result)