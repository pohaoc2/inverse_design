import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler
from inverse_design.analyze.analyze_aggregated_results import DEFAULT_METRICS
from inverse_design.analyze.analyze_utils import calculate_metrics_statistics


def calculate_normalized_error(metric_values, target_value, loss_function="absolute", scaler=None):
    """
    Calculate normalized error percentages for a set of metric values.
    
    Args:
        metric_values (np.array): Array of metric values to evaluate
        target_value (float): Target value for the metric
        scaler (MinMaxScaler, optional): Pre-fitted scaler. If None, creates new one.
    
    Returns:
        tuple: (normalized_error_percent, scaler)
    """
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(metric_values.reshape(-1, 1))
    
    normalized_values = scaler.transform(metric_values.reshape(-1, 1))
    normalized_target = scaler.transform(np.array([target_value]).reshape(-1, 1))[0][0]
    if loss_function == "absolute":
        error = calculate_absolute_error(normalized_values, normalized_target)
    elif loss_function == "squared":
        error = calculate_squared_error(normalized_values, normalized_target)
    
    return error, scaler

def calculate_squared_error(values, target_value):
    """
    Calculate squared error (distance) between values and target value.
    
    Args:
        values (np.array): Array of metric values
        target_value (float): Target value for the metric

    Returns:
        np.array: Squared error values
    """
    return (values - target_value) ** 2

def calculate_absolute_error(values, target_value):
    """
    Calculate absolute error (distance) between values and target value.
    
    Args:
        values (np.array): Array of metric values
        target_value (float): Target value for the metric
    
    Returns:
        np.array: Absolute error values
    """
    return np.abs(values - target_value)

def calculate_metric_errors(posterior_df_dict, prior_df, default_metrics, metric, target, sample_sizes, loss_function="absolute"):
    """
    Calculate error percentages for a metric across different sample sizes and reference values.
    
    Args:
        posterior_df_dict (dict): Dictionary of posterior dataframes for different sample sizes
        prior_df (dict): Prior distribution metrics
        default_metrics (dict): Default metrics values
        metric (str): Name of the metric to analyze
        target (float): Target value for the metric
        sample_sizes (list): List of sample sizes to analyze
        loss_function (str): Loss function to use

    Returns:
        tuple: (errors_percent, prior_error_percent, default_error_percent)
    """
    # Fit scaler on full dataset
    metric_values = posterior_df_dict[sample_sizes[-1]][metric]["metric"]

    _, scaler = calculate_normalized_error(metric_values, target, loss_function)
    
    # Calculate errors for each sample size
    posterior_errors = []
    for n in sample_sizes:
        posterior_values = posterior_df_dict[n][metric]["mode"]
        posterior_error, _ = calculate_normalized_error(
            np.array([posterior_values]), target, loss_function, scaler
        )
        posterior_errors.append(posterior_error[0][0])
    
    # Calculate reference error percentages
    prior_error, _ = calculate_normalized_error(
        np.array([prior_df[metric]["mode"]]), target, loss_function, scaler
    )
    prior_error = prior_error[0][0]
    
    default_error, _ = calculate_normalized_error(
        np.array([default_metrics[metric]['mean']]), target, loss_function, scaler
    )
    default_error = default_error[0][0]
    
    error_dict = {
        "posterior_errors": posterior_errors,
        "prior_errors": prior_error,
        "default_errors": default_error
    }

    return error_dict


def plot_error_analysis(posterior_df_dict, prior_df, target_metrics, default_metrics, metrics_list, 
                       x_values, x_label, loss_function="absolute", use_log2_scale=False, save_path=None):
    """
    Generic plotting function for error analysis vs a given parameter (samples or acceptance percentage).
    
    Args:
        posterior_df_dict (dict): Dictionary of posterior dataframes
        prior_df (dict): Prior distribution metrics
        target_metrics (dict): Target values for each metric
        default_metrics (dict): Default metrics values
        metrics_list (list): List of metrics to analyze
        x_values (list): Values for x-axis
        x_label (str): Label for x-axis
        loss_function (str): Loss function to use
        use_log2_scale (bool): Whether to use log2 scale for x-axis
        save_path (str, optional): Path to save the plot
    """
    n_plots = len(metrics_list) + 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4*n_plots))
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    plot_x = np.log2(x_values) if use_log2_scale else x_values
    all_errors_percent = {"posterior_errors": [], "prior_errors": [], "default_errors": []}
    
    # Plot for each individual metric
    for idx, metric in enumerate(metrics_list):
        target = target_metrics[metric]
        error_dict = calculate_metric_errors(
            posterior_df_dict, prior_df, default_metrics, metric, target, x_values, loss_function
        )
        all_errors_percent["posterior_errors"].append(error_dict["posterior_errors"])
        all_errors_percent["prior_errors"].append(error_dict["prior_errors"])
        all_errors_percent["default_errors"].append(error_dict["default_errors"])
        
        best_absolute_error = np.min(error_dict["posterior_errors"])
        
        axes[idx].plot(plot_x, error_dict["posterior_errors"], '-o', 
                      label=f'Posterior (lowest error = {best_absolute_error:.2f})')
        axes[idx].axhline(y=error_dict["prior_errors"], color='gray', linestyle='--', 
                         label=f'Prior error: {error_dict["prior_errors"]:.2f}')
        axes[idx].axhline(y=error_dict["default_errors"], color='green', linestyle='--', 
                         label=f'Default error: {error_dict["default_errors"]:.2f}')
        
        if use_log2_scale:
            axes[idx].set_xticks(plot_x)
            axes[idx].set_xticklabels([f'2^{int(x)}' for x in plot_x])
        
        axes[idx].set_xlabel(x_label)
        axes[idx].set_ylabel('Error')
        axes[idx].set_title(f'{loss_function} Error vs {x_label} for {metric} (target = {target})')
        axes[idx].grid(True)
        axes[idx].legend()
    
    total_errors = np.sum(all_errors_percent["posterior_errors"], axis=0)
    axes[-1].plot(plot_x, total_errors, '-o', label='Posterior')
    axes[-1].axhline(y=np.sum(all_errors_percent["prior_errors"]), color='gray', linestyle='--', 
                     label=f'Total prior error: {np.sum(all_errors_percent["prior_errors"]):.2f}')
    axes[-1].axhline(y=np.sum(all_errors_percent["default_errors"]), color='green', linestyle='--', 
                     label=f'Total default error: {np.sum(all_errors_percent["default_errors"]):.2f}')
    
    if use_log2_scale:
        axes[-1].set_xticks(plot_x)
        axes[-1].set_xticklabels([f'2^{int(x)}' for x in plot_x])
    
    axes[-1].set_xlabel(x_label)
    axes[-1].set_ylabel('Total Error')
    axes[-1].set_title(f'Total {loss_function} Error vs {x_label} (All Metrics)')
    axes[-1].grid(True)
    axes[-1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_error_vs_samples(posterior_df_dict, prior_df, target_metrics, default_metrics, metrics_list, loss_function="absolute", save_path=None):
    """Wrapper function for plotting absolute error vs number of samples."""
    sample_sizes = sorted(posterior_df_dict.keys())
    plot_error_analysis(
        posterior_df_dict, prior_df, target_metrics, default_metrics, metrics_list,
        sample_sizes, 'Number of Samples', loss_function, use_log2_scale=True, save_path=save_path
    )


def plot_error_vs_acceptance(posterior_df_dict, prior_df, target_metrics, default_metrics, metrics_list, loss_function="absolute", save_path=None):
    """Wrapper function for plotting absolute error vs acceptance percentage."""
    acceptance_percentages = sorted(posterior_df_dict.keys())
    plot_error_analysis(
        posterior_df_dict, prior_df, target_metrics, default_metrics, metrics_list,
        acceptance_percentages, 'Acceptance Percentage', loss_function, use_log2_scale=False, save_path=save_path
    )


def main():
    parameter_base_folder = "ARCADE_OUTPUT/STEM_CELL/MS_POSTERIOR_N512/MS_POSTERIOR"

    metrics_names = ["symmetry", "cycle_length", "act"]
    target_metrics = {"symmetry": 0.8, "cycle_length": 30.0, "act": 0.6}
    prior_metrics_file = (
        "ARCADE_OUTPUT/STEM_CELL/MS_PRIOR_N512/final_metrics.csv"
    )
    prior_metrics_df = pd.read_csv(prior_metrics_file)
    prior_metrics_statistics = calculate_metrics_statistics(prior_metrics_df, metrics_names)
    default_metrics = {metric: DEFAULT_METRICS[metric] for metric in target_metrics}
    
    loss_function = "absolute"

    if 1:
        parameter_base_folder = "ARCADE_OUTPUT/STEM_CELL/MS_POSTERIOR_N512/MS_POSTERIOR_10P"
        sample_sizes = [2**i for i in range(4, 9)]
        posterior_metrics_dict = {}
        for n in sample_sizes:
            posterior_df = pd.read_csv(f"{parameter_base_folder}/n{n}/final_metrics.csv")
            posterior_metrics_dict[n] = calculate_metrics_statistics(posterior_df, metrics_names)
            
        save_file = f"{parameter_base_folder}/error_vs_samples.png"
        plot_error_vs_samples(posterior_metrics_dict, prior_metrics_statistics, target_metrics, default_metrics, metrics_names, loss_function, save_file)
    if 1:
        parameter_base_folder = "ARCADE_OUTPUT/STEM_CELL/MS_POSTERIOR_N512"
        acceptance_percentages = [5, 10, 15, 20]
        posterior_metrics_dict = {}
        for acceptance_percentage in acceptance_percentages:
            posterior_df = pd.read_csv(f"{parameter_base_folder}/MS_POSTERIOR_{acceptance_percentage}P/n32/final_metrics.csv")
            posterior_metrics_dict[acceptance_percentage] = calculate_metrics_statistics(posterior_df, metrics_names)
        
        save_file = f"{parameter_base_folder}/error_vs_acceptance.png"
        plot_error_vs_acceptance(
            posterior_metrics_dict, 
            prior_metrics_statistics, 
            target_metrics, 
            default_metrics, 
            metrics_names, 
            loss_function,
            save_file
        )

if __name__ == "__main__":
    main()
