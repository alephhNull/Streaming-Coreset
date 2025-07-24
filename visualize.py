from tabulate import tabulate
import numpy as np

def print_experiment_summary(config: dict, result: dict, config_keys=None):
    # === Print Important Configs ===
    if config_keys is None:
        config_keys = [
            "dataset", "embedding", "embed_dim", "coreset_size", "dataset_subset_size",
            "batch_size", "n_rff_components", "kernel_gamma", "buffer_capacity", 
            "n_epochs_online", "lambda_log_online", "arrival_interval"
        ]

    config_table = [[k, config[k]] for k in config_keys if k in config]
    print("\n=== Experiment Configuration ===")
    print(tabulate(config_table, headers=["Parameter", "Value"], tablefmt="grid"))

    # === Print Unified Summary Table (Accuracy, AUC, F1, MMD, Streaming Metrics) ===
    summary_table = []

    for bm, results in result.items():
        if bm == 'whole_data':
            summary_table.append([
                bm,
                round(results['accuracy'], 4),
                round(results['auc'], 4) if not np.isnan(results['auc']) else 'nan',
                round(results['f1'], 4),
                '-', '-', '-', '-', '-', '-'
            ])
        else:
            accs = [r['accuracy'] for r in results]
            aucs = [r['auc'] for r in results if not np.isnan(r['auc'])]
            f1s = [r['f1'] for r in results]
            mmds = [r['mmd'] for r in results]

            # Get streaming metric keys from the first trial
            metric_keys = results[0]['streaming_metrics'].keys()

            avg_metrics = {
                key: np.mean([r['streaming_metrics'][key] for r in results])
                for key in metric_keys
            }

            summary_table.append([
                bm,
                round(np.mean(accs), 4),
                round(np.mean(aucs), 4) if aucs else 'nan',
                round(np.mean(f1s), 4),
                round(np.mean(mmds), 6),
                round(avg_metrics.get('avg_batch_time_ms', 0.0), 2),
                round(avg_metrics.get('effective_throughput_pps', 0.0), 2),
                round(avg_metrics.get('velocity_data_loss_pct', 0.0), 2),
                int(avg_metrics.get('batches_processed', 0)),
                int(avg_metrics.get('batches_dropped', 0))
            ])

    headers = [
        "Benchmark", "Accuracy", "AUC", "F1", "MMD",
        "Avg Time (ms)", "Throughput (pps)", "Loss (%)", "Processed", "Dropped"
    ]

    print("\n=== Mean Results per Benchmark (including streaming metrics) ===")
    print(tabulate(summary_table, headers=headers, tablefmt="grid"))
