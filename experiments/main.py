import sys 
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from streaming_coreset_runner import run_single_experiment
from visualize import print_experiment_summary

if __name__ == "__main__":
    configs = [
        {
            "dataset": "adult",
            "embedding": None,
            "embed_dim": None,
            "benchmarks": ["OnlineMMDPlus", "CO2", "Reservoir"],  # only these
            "coreset_size": 30,
            "dataset_subset_size": 2500,
            "batch_size": 50,
            "n_rff_components": 1000,
            "kernel_gamma": 0.1,
            "buffer_capacity": 150,
            "random_seed": 10921,
            "n_epochs_online": 20,
            "lr_online": 0.1,
            "lambda_log_online": 5e-5,
            "reservoir_trials": 10,
            "co2_trials": 5
        },
        {
    "dataset": "cifar10",
    "embedding": 'resnet18',
    "embed_dim": 50,
    "benchmarks": ["OnlineMMDPlus", "CO2", "Reservoir"],  # only these
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
    "co2_trials": 5
},
{
    "dataset": "covtype",
    "embedding": None,
    "embed_dim": 10,
    "benchmarks": ["OnlineMMDPlus", "CO2","Reservoir"],  # only these
    "coreset_size": 30,
    "dataset_subset_size": 10000,
    "batch_size": 50,
    "n_rff_components": 1000,
    "kernel_gamma": 0.001,
    "buffer_capacity": 150,
    "random_seed": 9921911,
    "n_epochs_online": 20,
    "lr_online": 0.1,
    "lambda_log_online": 1e-7,
    "reservoir_trials": 10,
    "co2_trials": 5
},
{
    "dataset": "electricity",
    "embedding": None,
    "embed_dim": None,
    "benchmarks": ["OnlineMMDPlus", "CO2", "Reservoir"],  # only these
    "coreset_size": 30,
    "dataset_subset_size": 10000,
    "batch_size": 50,
    "n_rff_components": 1000,
    "kernel_gamma": 0.1,
    "buffer_capacity": 150,
    "random_seed": 918282,
    "n_epochs_online": 20,
    "lr_online": 0.1,
    "lambda_log_online": 1e-5,
    "reservoir_trials": 10,
    "co2_trials": 5
},
{
    "dataset": "fashion_mnist",
    "embedding": 'resnet18',
    "embed_dim": 50,
    "benchmarks": ["OnlineMMDPlus", "CO2", "Reservoir"],  # only these
    "coreset_size": 150,
    "dataset_subset_size": 2500,
    "batch_size": 50,
    "n_rff_components": 1000,
    "kernel_gamma": 0.001,
    "buffer_capacity": 150,
    "random_seed": 29222,
    "n_epochs_online": 20,
    "lr_online": 0.1,
    "lambda_log_online": 1e-7,
    "reservoir_trials": 10,
    "co2_trials": 5
},
{
    "dataset": "kdd99",
    "embedding": None,
    "embed_dim": None,
    "benchmarks": ["OnlineMMDPlus", "CO2", "Reservoir"],  # only these
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
    "co2_trials": 5
},
{
    "dataset": "mnist",
    "embedding": 'resnet18',
    "embed_dim": 50,
    "benchmarks": ["OnlineMMDPlus", "CO2", "Reservoir"],  # only these
    "coreset_size": 150,
    "dataset_subset_size": 2500,
    "batch_size": 50,
    "n_rff_components": 1000,
    "kernel_gamma": 0.001,
    "buffer_capacity": 150,
    "random_seed": 29222,
    "n_epochs_online": 20,
    "lr_online": 0.1,
    "lambda_log_online": 1e-7,
    "reservoir_trials": 10,
    "co2_trials": 5
},
{
    "dataset": "svhn",
    "embedding": 'resnet18',
    "embed_dim": 50,
    "benchmarks": ["OnlineMMDPlus", "CO2", "Reservoir"],  # only these
    "coreset_size": 150,
    "dataset_subset_size": 2500,
    "batch_size": 50,
    "n_rff_components": 1000,
    "kernel_gamma": 0.001,
    "buffer_capacity": 150,
    "random_seed": 29222,
    "n_epochs_online": 20,
    "lr_online": 0.1,
    "lambda_log_online": 1e-7,
    "reservoir_trials": 10,
    "co2_trials": 5
}
    ]



    config_keys = ["dataset", "embedding", "embed_dim", "coreset_size", "dataset_subset_size", "buffer_capacity"]

    results = []
    for config in configs:
        result = run_single_experiment(config)
        results.append(result)

    for config, result in zip(configs, results):
        print_experiment_summary(config, result, config_keys)
