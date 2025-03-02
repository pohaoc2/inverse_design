import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler
from inverse_design.analyze.analyze_aggregated_results import DEFAULT_METRICS
from inverse_design.analyze.analyze_utils import calculate_metrics_statistics


def get_error_percent(values, target_value):
    """
    Calculate error percentage relative to target value.
    
    Args:
        values (np.array): Array of metric values
        target_value (float): Target value for the metric
    
    Returns:
        np.array: Error percentage values
    """
    return (values - target_value) / abs(target_value) * 100


def plot_mse_vs_samples(posterior_df_dict, prior_df, target_metrics, default_metrics, metrics_list, save_path=None):
    """
    Plot error percentage vs number of samples for each metric and mean error.
    Metrics are normalized to [0, 1] range before calculating error percentage.
    X-axis uses log2 scale for better interpretability.
    """
    
    n_plots = len(metrics_list) + 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4*n_plots))
    
    # Convert sample sizes to log2 for plotting
    sample_sizes = sorted(posterior_df_dict.keys())
    log2_samples = np.log2(sample_sizes)
    
    all_errors_percent = []
    # Plot for each individual metric
    for idx, metric in enumerate(metrics_list):
        errors_percent = []
        target = target_metrics[metric]
        min_max_scaler = MinMaxScaler()
        min_max_scaler.fit(posterior_df_dict[sample_sizes[-1]][metric]["metric"].reshape(-1, 1))
        normalized_target = min_max_scaler.transform(np.array([target]).reshape(-1, 1))[0][0]

        for n in sample_sizes:
            posterior_df = posterior_df_dict[n]
            normalized_values = min_max_scaler.transform(posterior_df[metric]["mode"].reshape(-1, 1))[0][0]
            error_percent = get_error_percent(normalized_values, normalized_target)
            errors_percent.append(error_percent)
        all_errors_percent.append(errors_percent)
        
        # Calculate prior error percentage
        normalized_prior = min_max_scaler.transform(np.array([prior_df[metric]["mode"]]).reshape(-1, 1))[0][0]
        prior_error_percent = get_error_percent(normalized_prior, normalized_target)
        
        # Calculate default error percentage
        default_mean = default_metrics[metric]['mean']
        normalized_default = min_max_scaler.transform(np.array([default_mean]).reshape(-1, 1))[0][0]
        default_error_percent = get_error_percent(normalized_default, normalized_target)
        
        # Plot using log2 scale
        best_error_percent_idx = np.argmin([abs(x) for x in errors_percent])
        best_error_percent = errors_percent[best_error_percent_idx]
        axes[idx].plot(log2_samples, errors_percent, '-o', label=f'Posterior (best error percent = {best_error_percent:.2f}%)')
        axes[idx].axhline(y=prior_error_percent, color='gray', linestyle='--', 
                         label=f'Prior error percent: {prior_error_percent:.2f}%')
        axes[idx].axhline(y=default_error_percent, color='green', linestyle='--', 
                         label=f'Default error percent: {default_error_percent:.2f}%')
        
        # Set x-ticks to show powers of 2
        axes[idx].set_xticks(log2_samples)
        axes[idx].set_xticklabels([f'2^{int(x)}' for x in log2_samples])
        
        axes[idx].set_xlabel('Number of Samples')
        axes[idx].set_ylabel('Error Percentage')
        axes[idx].set_title(f'Error Percentage vs Samples for {metric}')
        axes[idx].grid(True)
        axes[idx].legend()
    
    # Plot mean error percentage across all metrics
    mean_errors = np.mean(all_errors_percent, axis=0)
    axes[-1].plot(log2_samples, mean_errors, '-o', label='Posterior')
    axes[-1].axhline(y=np.mean([prior_error_percent, default_error_percent]), color='gray', linestyle='--', 
                     label=f'Average Reference error percent: {np.mean([prior_error_percent, default_error_percent]):.2f}%')
    
    # Set x-ticks for mean error plot
    axes[-1].set_xticks(log2_samples)
    axes[-1].set_xticklabels([f'2^{int(x)}' for x in log2_samples])
    
    axes[-1].set_xlabel('Number of Samples')
    axes[-1].set_ylabel('Mean Error Percentage')
    axes[-1].set_title('Mean Error Percentage vs Samples (All Metrics)')
    axes[-1].grid(True)
    axes[-1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def main():
    parameter_base_folder = "ARCADE_OUTPUT/STEM_CELL_META_SIGNAL_HETEROGENEITY_POSTERIOR"

    metrics_names = ["symmetry", "cycle_length", "act"]
    target_metrics = {"symmetry": 0.8, "cycle_length": 30.0, "act": 0.6}
    prior_metrics_file = (
        "ARCADE_OUTPUT/STEM_CELL_META_SIGNAL_HETEROGENEITY/final_metrics.csv"
    )
    prior_metrics_df = pd.read_csv(prior_metrics_file)
    prior_metrics_statistics = calculate_metrics_statistics(prior_metrics_df, metrics_names)
    default_metrics = {metric: DEFAULT_METRICS[metric] for metric in target_metrics}

    sample_sizes = [2**i for i in range(4, 9)]
    posterior_metrics_dict = {}
    for n in sample_sizes:
        posterior_df = pd.read_csv(f"{parameter_base_folder}/n{n}/final_metrics.csv")
        posterior_metrics_dict[n] = calculate_metrics_statistics(posterior_df, metrics_names)

    save_file = f"{parameter_base_folder}/mse_vs_samples.png"
    plot_mse_vs_samples(posterior_metrics_dict, prior_metrics_statistics, target_metrics, default_metrics, metrics_names, save_file)

if __name__ == "__main__":
    main()
