import sys 
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from streaming_coreset_runner import run_single_experiment
from visualize import print_experiment_summary


if __name__ == "__main__":

    config = {
    "dataset": "cifar10",
    "embedding": 'resnet18',
    "embed_dim": 100,
    "benchmarks": ["WKH", "OnlineMMDPlus", "Reservoir"],  # only these
    "coreset_size": 150,
    "dataset_subset_size": 2500,
    "batch_size": 50,
    "n_rff_components": 1000,
    "kernel_gamma": 0.001,
    "buffer_capacity": 150,
    "random_seed": 42123,
    "n_epochs_online": 20,
    "lr_online": 0.1,
    "lambda_log_online": 1e-7,
    "reservoir_trials": 10,
    "n_classes": 10,
    "tasks": ['logistic_regression', 'RandomForest'],
    "dist_metrics": ['MMD', '1-Wasserstein'],
    "metrics": ['acc', 'f1'],
    "streaming_metrics": ['avg_batch_time_ms'],
    "wkh_trials": 10,
    "onlinemmdplus_trials": 10
}


    result = run_single_experiment(config)
    print_experiment_summary(config, result)