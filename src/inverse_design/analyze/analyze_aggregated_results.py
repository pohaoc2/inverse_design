import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
import seaborn as sns
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from inverse_design.analyze.analyze_utils import collect_parameter_data, analyze_metric_percentiles
from inverse_design.analyze.parameter_config import PARAMETER_LIST
from inverse_design.analyze.metrics_config import DEFAULT_METRICS


def plot_top_bottom_parameter_distributions(
    all_param_df,
    parameter_list,
    parameter_base_folder,
    percentile: float = 10,
    save_file: str = None,
):
    """
    Plot parameter distributions for top and bottom percentile cases.
    """
    n_params = len(parameter_list)
    plots_per_row = 4
    n_rows = (n_params + plots_per_row - 1) // plots_per_row  # Ceiling division
    n_cols = min(n_params, plots_per_row)
    
    # Create figure with appropriate size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    
    # Make axes 2D if it's 1D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, param in enumerate(parameter_list):
        # Calculate row and column indices
        row_idx = idx // plots_per_row
        col_idx = idx % plots_per_row
        ax = axes[row_idx, col_idx]

        # Plot density distributions for each group
        for label in ["top", "bottom"]:
            group_data = all_param_df[all_param_df["percentile_label"] == label]
            color = "blue" if label == "top" else "red"
            sns.kdeplot(
                data=group_data[param],
                ax=ax,
                label=f"{label.capitalize()} {percentile}%",
                color=color,
            )

        ax.set_title(f"{param} Distribution")
        ax.set_xlabel(param)
        ax.set_ylabel("Density")
        ax.legend()
    
    # Hide empty subplots if any
    for idx in range(len(parameter_list), n_rows * n_cols):
        row_idx = idx // plots_per_row
        col_idx = idx % plots_per_row
        axes[row_idx, col_idx].set_visible(False)

    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file)
    else:
        plt.show()
    plt.close()
    # Print summary statistics
    print("\nParameter Statistics:")
    print("=" * 80)
    for label in ["top", "bottom"]:
        group_data = all_param_df[all_param_df["percentile_label"] == label]
        print(f"\n{label.capitalize()} {percentile}% - Mean values:")
        print(group_data[parameter_list].mean())


def plot_pca_parameters(
    all_param_df,
    parameter_list,
    parameter_base_folder,
    metrics_name: str,
    percentile: float = 10,
    save_file: str = None,
):
    """
    Perform PCA on parameters and visualize top and bottom percentile cases in 2D.
    """
    # Prepare data for PCA
    X = all_param_df[parameter_list]
    labels = (all_param_df["percentile_label"] == "top").astype(int)

    # Standardize the features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)

    # Create plot
    plt.figure(figsize=(10, 8))

    # Plot points for each group
    for label, color in zip(["top", "bottom"], ["blue", "red"]):
        mask = all_param_df["percentile_label"] == label
        plt.scatter(
            data_pca[mask, 0],
            data_pca[mask, 1],
            c=color,
            label=f"{label.capitalize()} {percentile}%",
            alpha=0.6,
        )

    # Add parameter vectors
    for i, param in enumerate(parameter_list):
        plt.arrow(
            0,
            0,
            pca.components_[0, i] * 3,
            pca.components_[1, i] * 3,
            color="black",
            alpha=0.5,
        )
        plt.text(
            pca.components_[0, i] * 3.2,
            pca.components_[1, i] * 3.2,
            param,
            color="black",
            ha="center",
            va="center",
        )

    # Add plot details
    plt.xlabel(f"First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.title(f"Top and Bottom {percentile}% Cases for {metrics_name} PCA")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    plt.axvline(x=0, color="k", linestyle="--", alpha=0.3)

    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file)
    else:
        plt.show()
    plt.close()
    # Print analysis results
    print("\nPCA Explained Variance Ratios:")
    print("=" * 80)
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {ratio:.1%}")

    loadings_df = pd.DataFrame(
        pca.components_,
        index=[f"PC{i+1}" for i in range(len(pca.components_))],
        columns=parameter_list,
    )
    print("\nParameter contributions:")
    print(loadings_df)


def get_kde_bounds(kde_plot, percentile: float = 0.01):
    """
    Get the x-axis bounds that contain (1 - 2*percentile)% of the KDE distribution.
    
    Args:
        kde_plot: The KDE plot from seaborn
        percentile: The percentile to exclude from each tail (default: 0.01 for 1%)
    
    Returns:
        tuple: (x_min, x_max) containing the bounds
    """
    # Get the x and y values from the KDE
    line = kde_plot.lines[0]
    x_vals = line.get_xdata()
    y_vals = line.get_ydata()
    
    # Calculate cumulative distribution and find percentiles
    cum_dist = np.cumsum(y_vals) / np.sum(y_vals)
    x_min = x_vals[np.argmax(cum_dist >= percentile)]
    x_max = x_vals[np.argmax(cum_dist >= (1 - percentile))]
    
    return x_min, x_max

def _remove_outliers(data, iqr_multiplier):
    if len(data) > 0:
        # Calculate IQR and bounds
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        # Remove outliers
        data = data[(data >= lower_bound) & (data <= upper_bound)]
    return data



def plot_distributions(prior_df, posterior_dfs, target_metrics, metrics_list, default_metrics=None, save_path=None, plot_type='kde', remove_outliers=False, iqr_multiplier=1.5):
    """
    Plot distributions of metrics with prior, multiple posteriors, target, and default values.
    
    Args:
        prior_df (pd.DataFrame): DataFrame containing prior samples
        posterior_dfs (list[pd.DataFrame]): List of DataFrames containing posterior samples
        target_metrics (dict): Dictionary of target metric values
        metrics_list (list): List of metrics to plot
        default_metrics (dict): Dictionary containing mean and std for default metrics
            Format: {'metric_name': {'mean': value, 'std': value}}
        save_path (str, optional): Path to save the figure
        plot_type (str): Type of plot to generate. Options: 'kde' for KDE plots, 'bar' for bar plots, 'violin' for violin plots
        remove_outliers (bool): Whether to remove outliers using IQR method
        iqr_multiplier (float): Multiplier for IQR to determine outlier bounds (default: 1.5)
    """
    # Convert single DataFrame to list for backwards compatibility
    if isinstance(posterior_dfs, pd.DataFrame):
        posterior_dfs = [posterior_dfs]
        
    # Define colors for multiple posteriors
    posterior_colors = ['blue', 'purple', 'cyan', 'magenta', 'orange', 'green', 'red', 'brown', 'pink', 'gray']
    posterior_labels = [f'Posterior {i+1}' for i in range(len(posterior_dfs))]
    posterior_labels = ["Cellular and environmental parameters"]
    
    fig, axes = plt.subplots(len(metrics_list), 1, figsize=(10, 4*len(metrics_list)))
    if len(metrics_list) == 1:
        axes = [axes]
    x_mins = [0.0, 0, 0, 0]
    x_maxs = [1.0, 40, 150, 500]
    for idx, metric in enumerate(metrics_list):
        ax = axes[idx]
        # Filter out NaN and infinite values
        valid_prior = prior_df[metric].replace([np.inf, -np.inf], np.nan).dropna()
        if remove_outliers and len(valid_prior) > 0:
            valid_prior = _remove_outliers(valid_prior, iqr_multiplier)
            print(f"Removed {len(prior_df[metric]) - len(valid_prior)} outliers for {metric} in prior")
        if len(valid_prior) == 0:
            print(f"Warning: No valid data points for {metric} in prior")
            continue
        valid_posteriors = []
        for posterior_df in posterior_dfs:
            valid_posterior = posterior_df[metric].replace([np.inf, -np.inf], np.nan).dropna()
            if remove_outliers and len(valid_posterior) > 0:
                valid_posterior = _remove_outliers(valid_posterior, iqr_multiplier)
                valid_posteriors.append(valid_posterior)
                print(f"Removed {len(posterior_df[metric]) - len(valid_posterior)} outliers for {metric} in posterior")
            else:
                valid_posteriors.append(valid_posterior)

        if plot_type == 'kde':
            # Plot prior distribution as KDE
            prior_kde = sns.kdeplot(data=valid_prior, ax=ax, label="Prior", color="gray")
            ax.set_ylabel("Prior Density", color="gray")
            ax.tick_params(axis="y", labelcolor="gray")
            ax2 = ax.twinx()
            
            # Plot each posterior distribution as KDE
            posterior_kdes = []
            x_bounds = [prior_kde.get_lines()[0].get_xdata()[[0, -1]]]  # Start with prior bounds
            for posterior_df, color, label in zip(valid_posteriors, posterior_colors, posterior_labels):
                
                posterior_kde = sns.kdeplot(
                    data=posterior_df, 
                    ax=ax2, 
                    label=label, 
                    color=color
                )
                lines = posterior_kde.get_lines()
                mode_value = lines[-1].get_xdata()[np.argmax(lines[-1].get_ydata())]
                posterior_kdes.append(posterior_kde)
                x_bounds.append(posterior_kde.get_lines()[0].get_xdata()[[0, -1]])
                ax2.axvline(x=mode_value, color=color, linestyle='--', alpha=0.5, label=f"{label} (mode: {mode_value:.3f})")
            
            # Calculate overall x-axis bounds
            x_min = min(target_metrics[metric], min(bound[0] for bound in x_bounds))
            x_max = max(target_metrics[metric], max(bound[1] for bound in x_bounds))
            
            ax2.set_ylabel("Posterior Density", color=posterior_colors[0])
            ax2.tick_params(axis="y", labelcolor=posterior_colors[0])
            
        else:
            prior_hist = sns.histplot(data=valid_prior, ax=ax, label="Environmental Only", color="gray", alpha=0.5, bins=20)
            ax.set_ylabel("Prior Count", color="gray")
            ax.tick_params(axis="y", labelcolor="gray")
            x_bounds = [prior_hist.get_xlim()]
            if 0:
                ax2 = ax.twinx()
                for posterior_df, color, label in zip(valid_posteriors, posterior_colors, posterior_labels):
                    posterior_hist = sns.histplot(
                        data=posterior_df, 
                        ax=ax2, 
                        label=label, 
                        color=color,
                        alpha=0.5,
                        bins=20
                    )
                    # Calculate mode (bin with highest count)
                    counts, bins = np.histogram(posterior_df, bins=20)  # Use fixed number of bins
                    mode_value = bins[np.argmax(counts)]
                    x_bounds.append(posterior_hist.get_xlim())
                    ax2.axvline(x=mode_value, color=color, linestyle='--', alpha=0.5, label=f"{label} (mode: {mode_value:.3f})")
                ax2.set_ylabel("Posterior Count", color=posterior_colors[0])
                ax2.tick_params(axis="y", labelcolor=posterior_colors[0])
            # Calculate overall x-axis bounds
            x_min = min(target_metrics[metric], min(bound[0] for bound in x_bounds))
            x_max = max(target_metrics[metric], max(bound[1] for bound in x_bounds))
            
        # Plot target line
        if 0:
            ax.axvline(
                x=target_metrics[metric],
                color="red",
                linestyle="--",
                label=f"Target ({target_metrics[metric]:.3f})",
            )
        
        if default_metrics and metric in default_metrics:
            mean = default_metrics[metric]['mean']
            std = default_metrics[metric]['std']
            ax.axvline(
                x=mean,
                color="green",
                linestyle="-",
                label=f"Default ({mean:.3f} ± {std:.3f})",
            )
            # Add striped region for mean ± std
            ax.fill_between(
                [mean - std, mean + std],
                ax.get_ylim()[0],
                ax.get_ylim()[1],
                color="green",
                alpha=0.2,
                hatch='///',  # diagonal stripes
                label="Default ±1σ"
            )
            # Add edges to make the region more visible
            ax.axvline(x=mean - std, color='green', linestyle=':', alpha=0.5)
            ax.axvline(x=mean + std, color='green', linestyle=':', alpha=0.5)
            x_min = min(x_min, mean - std)
            x_max = max(x_max, mean + std)

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        if 0:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(
                lines1 + lines2, 
                labels1 + labels2,
                bbox_to_anchor=(1.15, 1),
                loc='upper left'
            )

        ax.set_title(f"{metric} Distribution")
        ax.set_xlabel(metric)
        padding = 0.1 * (x_max - x_min)
        #ax.set_xlim(x_min - padding, x_max + padding)
        x_min, x_max = x_mins[idx], x_maxs[idx]
        ax.set_xlim(x_min, x_max)
        ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_cell_states_histogram(csv_file, save_file=None):
    """
    Plot histogram of cell states from simulation metrics.

    Args:
        csv_file: Path to simulation_metrics.csv
        parameter_base_folder: Base folder for saving plots
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Extract cell states from the last column, handling empty/invalid entries
    states_data = []
    for entry in df.iloc[:, -1]:
        try:
            if pd.isna(entry) or entry == "":
                continue
            states_data.append(eval(str(entry)))  # Convert to string first and evaluate
        except (ValueError, SyntaxError):
            continue

    # Initialize counters for each state
    state_counts = {
        "UNDEFINED": [],
        "QUIESCENT": [],
        "MIGRATORY": [],
        "PROLIFERATIVE": [],
        "APOPTOTIC": [],
        "NECROTIC": [],
        "SENESCENT": [],
    }
    # Calculate average counts for each state across seeds
    for states_list in states_data:
        for state, count in states_list.items():
            state_counts[state].append(count)

    # Create figure
    plt.figure(figsize=(12, 6))

    # Plot histogram for each non-zero state
    non_zero_states = {
        state: counts
        for state, counts in state_counts.items()
        if any(count > 0 for count in counts)
    }

    if len(non_zero_states) > 0:
        plt.hist(non_zero_states.values(), label=non_zero_states.keys(), bins=30, alpha=0.7)
        plt.xlabel("Average Cell Count")
        plt.ylabel("Frequency")
        plt.title("Distribution of Cell States Across Simulations")
        plt.legend()
    else:
        plt.text(
            0.5,
            0.5,
            "No non-zero cell states found",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )

    if save_file is not None:
        plt.savefig(save_file)
    else:
        plt.show()

    # Print summary statistics
    print("\nCell State Statistics:")
    print("=" * 50)
    for state, counts in state_counts.items():
        if any(count > 0 for count in counts):
            print(f"{state}:")
            print(f"  Mean: {np.mean(counts):.2f}")
            print(f"  Std:  {np.std(counts):.2f}")
            print(f"  Min:  {np.min(counts):.2f}")
            print(f"  Max:  {np.max(counts):.2f}")


def plot_metric_pairplot(
    metrics_df: pd.DataFrame,
    metrics_list: List[str],
    percentile_labels: Optional[pd.Series] = None,
    save_file: Optional[str] = None,
):
    """
    Create a pairplot showing relationships between different metrics.
    
    Args:
        metrics_df: DataFrame containing the metrics
        metrics_list: List of metric names to include
        percentile_labels: Optional series of labels for coloring points (e.g., 'top'/'bottom')
        save_file: Optional path to save the plot
    """
    # Create plot DataFrame with only the metrics we want
    plot_df = metrics_df[metrics_list].copy()
    
    # Add percentile labels if provided
    if percentile_labels is not None:
        plot_df['Performance'] = percentile_labels
        hue = 'Performance'
        palette = {'low': 'blue', 'high': 'red', "None": 'gray'}
    else:
        hue = None
        palette = None
    
    # Create pairplot
    g = sns.pairplot(
        data=plot_df,
        hue=hue,
        palette=palette,
        diag_kind='kde',  # Use KDE plots on diagonal
        plot_kws={'alpha': 0.6},  # Add some transparency to points
        diag_kws={'fill': True},  # Fill the KDE plots
    )
    
    # Add correlation coefficients to the upper triangle
    for i, var1 in enumerate(metrics_list):
        for j, var2 in enumerate(metrics_list):
            if i < j:  # Upper triangle only
                ax = g.axes[i, j]
                corr = plot_df[var1].corr(plot_df[var2])
                ax.text(0.9, 0.9, f'corr = {corr:.2f}',
                       horizontalalignment='center',
                       verticalalignment='center',
                       transform=ax.transAxes,
                       fontsize=10)
    
    # Adjust layout and title
    g.fig.suptitle('Pairwise Relationships Between Metrics', y=1.02)
    plt.tight_layout()
    
    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_histogram_comparison(
    dfs,
    labels,
    metric,
    color='k',
    save_path=None,
    target_metric=None,
    remove_outliers=False,
    iqr_multiplier=1.5):
    """
    Create horizontal histogram plots for a single metric, with each subplot showing a different distribution.
    
    Args:
        dfs (list[pd.DataFrame]): List of DataFrames containing multiple samples
        metric (str): The metric to plot
        save_path (str, optional): Path to save the figure
        remove_outliers (bool): Whether to remove outliers using IQR method
        iqr_multiplier (float): Multiplier for IQR to determine outlier bounds (default: 1.5)
    """
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
    
    n_distributions = len(dfs)
    fig, axes = plt.subplots(n_distributions, 1, figsize=(6, 2*n_distributions))
    if n_distributions == 1:
        axes = [axes]
    
    valid_dfs = []
    for df in dfs:
        valid_df = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
        if remove_outliers and len(valid_df) > 0:
            valid_df = _remove_outliers(valid_df, iqr_multiplier)
            valid_dfs.append(valid_df)
            print(f"Removed {len(df[metric]) - len(valid_df)} outliers for {metric} in df")
        else:
            valid_dfs.append(valid_df)
    
    # Calculate overall x-axis bounds
    x_min = valid_dfs[0].min()
    x_max = valid_dfs[0].max()
    for valid_df in valid_dfs:
        x_min = min(x_min, valid_df.min())
        x_max = max(x_max, valid_df.max())
    
    padding = 0.1 * (x_max - x_min)
    x_min -= padding
    x_max += padding

    base_rgb = to_rgb(color)
    colors = []
    for i in range(n_distributions):
        darken_factor = 0.15 * i
        adjusted_rgb = tuple((1 - darken_factor) * c for c in base_rgb)
        colors.append(to_hex(adjusted_rgb))

    for i, (df, label, plot_color) in enumerate(zip(valid_dfs, labels, colors)):
        ax = axes[i]
        sns.histplot(data=df, ax=ax, color=plot_color, alpha=0.5, bins=20)
        ax.set_title(label)
        if i == len(valid_dfs) - 1:
            ax.set_xlabel(metric)
        else:
            ax.set_xlabel('')
        ax.set_xlim(x_min, x_max)
        ax.grid(axis='y', linestyle='--')
        if i > 0:
            ax.sharex(axes[0])
        if target_metric:
            ax.axvline(x=target_metric, color='red', linestyle='--', label=f'Target {metric} = {target_metric}')
        ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Specify your parameters
    parameter_base_folder = "ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid"
    input_folder = parameter_base_folder + "/iter_0/inputs"
    csv_file = f"{parameter_base_folder}/iter_0/final_metrics.csv"

    metrics_name = "n_cells"
    metrics_df = pd.read_csv(csv_file)
    if 1:
        colors = ['purple', 'blue', 'magenta', 'orange', 'green', 'red', 'brown', 'pink']
        #posterior_metrics_files = [f"{parameter_base_folder}/accepted_metrics_{i}p.csv" for i in range(20, 4, -5)]
        posterior_1 = csv_file
        posterior_2 = f"{parameter_base_folder}/iter_1/final_metrics.csv"
        posterior_3 = f"{parameter_base_folder}/iter_2/final_metrics.csv"
        posterior_4 = f"{parameter_base_folder}/iter_3/final_metrics.csv"
        posterior_5 = f"{parameter_base_folder}/iter_4/final_metrics.csv"
        posterior_metrics_files = [posterior_1, posterior_2, posterior_3, posterior_4, posterior_5]
        posterior_metrics_dfs = [pd.read_csv(posterior_metrics_file) for posterior_metrics_file in posterior_metrics_files]
        env_only_file = (
            "ARCADE_OUTPUT/STEM_CELL/DENSITY_SOURCE/grid/final_metrics.csv"
        )
        cellular_only_file = (
            "ARCADE_OUTPUT/STEM_CELL/MS_ALL/MS_PRIOR_N1024/final_metrics.csv"
        )
        both_file = csv_file
        target_metrics = {
            "symmetry": 0.75,
            #"cycle_length": 30.0,
            "doub_time": 32,
            #"n_cells": 100,
            "act_ratio": 0.7,
            #"vol": 2000,
            #"activity": 0.5,
            #"colony_diameter": 300,
        }
        default_metrics = {metric: DEFAULT_METRICS[metric] for metric in target_metrics}

        df_paths = [cellular_only_file, env_only_file, both_file]
        dfs = [pd.read_csv(df) for df in df_paths]
        dfs = posterior_metrics_dfs
        labels = ["Cellular only", "Environmental only", "Cellular and environmental parameters"]
        labels = ["iter_0", "iter_1", "iter_2", "iter_3", "iter_4"]
        for i, (metric_name, metric_value) in enumerate(target_metrics.items()):
            save_file = f"{parameter_base_folder}/metric_distributions_bar_{metric_name}_removed_outliers.png"
            plot_histogram_comparison(
                dfs,
                labels,
                metric_name,
                color=colors[i],
                save_path=save_file,
                target_metric=metric_value,
                remove_outliers=True,
            )
            save_file = f"{parameter_base_folder}/metric_distributions_bar_{metric_name}.png"
            plot_histogram_comparison(
                dfs,
                labels,
                metric_name,
                color=colors[i],
                save_path=save_file,
                target_metric=metric_value,
                remove_outliers=False,
            )

        if 0:
            plot_type = 'violin'
            save_file = f"{parameter_base_folder}/metric_distributions_{plot_type}.png"
            plot_distributions(
                pd.read_csv(prior_metrics_file),
                posterior_metrics_dfs,
                target_metrics,
                list(target_metrics.keys()),
                default_metrics,
                save_file,
                plot_type=plot_type,
                remove_outliers=True,
            )
    percentile = 10
    top_n_input_file, bottom_n_input_file, labeled_metrics_df = analyze_metric_percentiles(
        csv_file, metrics_name, percentile, verbose=False
    )

    # Create labeled parameter DataFrame
    input_files = list(top_n_input_file) + list(bottom_n_input_file)
    input_files = [f"{input_folder}/{input_file}" for input_file in input_files]
    labels = ["top"] * len(top_n_input_file) + ["bottom"] * len(bottom_n_input_file)
    analyzed_param_df = collect_parameter_data(
        input_files, PARAMETER_LIST
    )
    analyzed_param_df = analyzed_param_df.sort_values(
        "input_folder",
        key=lambda x: x.str.split("_").str[1].astype(int)
    )

    # add a column to metrics_df with the percentile label
    # if the metric['input_folder'] is in the top_n_input_file, assign label "top", otherwise "bottom"
    metrics_df['percentile_label'] = metrics_df['input_folder'].apply(
        lambda x: "high" if x in top_n_input_file else "low" if x in bottom_n_input_file else "None"
    )
    analyzed_param_df['percentile_label'] = analyzed_param_df['input_folder'].apply(
        lambda x: "high" if x in top_n_input_file else "low" if x in bottom_n_input_file else "None"
    )

    if 0:
        save_file = f"{parameter_base_folder}/metric_pairplot_{metrics_name}.png"
        remove_metrics = ['input_folder', 'percentile_label', 'age', 'age_std', 'states', 'colony_growth_r', 'colony_growth', "doub_time",]
        metrics_list = [col for col in metrics_df.columns if col not in remove_metrics and not col.endswith('_std')]
        #metrics_list = [col for col in metrics_df.columns if col not in remove_metrics and col.endswith('_std')]
        plot_metric_pairplot(
            metrics_df,
            metrics_list,
            metrics_df['percentile_label'],
            save_file
        )

    if 0:
        save_file = f"{parameter_base_folder}/parameter_distributions_{metrics_name}.png"
        plot_top_bottom_parameter_distributions(
            analyzed_param_df, PARAMETER_LIST, parameter_base_folder, percentile, save_file
        )
    if 0:
        save_file = f"{parameter_base_folder}/pca_parameters_{metrics_name}.png"
        plot_pca_parameters(
            analyzed_param_df,
        PARAMETER_LIST,
        parameter_base_folder,
        metrics_name,
        percentile,
        save_file,
    )

    if 0:
        plot_cell_states_histogram(
            csv_file,
            f"{parameter_base_folder}/cell_states_histogram.png",
        )
