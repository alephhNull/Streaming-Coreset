from experiments import run_single_experiment



if __name__ == "__main__":

    config = {
    "dataset": "cifar10",
    "embedding": 'resnet18',
    "embed_dim": 50,
    "benchmarks": ["OnlineMMDPlus", "Reservoir"],  # only these
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
}


    # Example config printing
    print("Using config:", config)

    run_single_experiment(config)