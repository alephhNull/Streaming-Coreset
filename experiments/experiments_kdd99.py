import sys 
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from streaming_coreset_runner import run_single_experiment
from visualize import print_experiment_summary


if __name__ == "__main__":

    config = {
    "dataset": "kdd99",
    "embedding": None,
    "embed_dim": None,
    "benchmarks": ["OnlineMMDPlus", "Reservoir"],  # only these
    "coreset_size": 20,
    "dataset_subset_size": 10000,
    "batch_size": 50,
    "n_rff_components": 1000,
    "kernel_gamma": 0.1,
    "buffer_capacity": 150,
    "random_seed": 918282,
    "n_epochs_online": 20,
    "lr_online": 0.1,
    "lambda_log_online": 5e-5,
    "reservoir_trials": 10,
    "n_classes": 2
}


    result = run_single_experiment(config)
    print_experiment_summary(config, result)