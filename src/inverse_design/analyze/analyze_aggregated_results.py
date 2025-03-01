import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from inverse_design.analyze.analyze_utils import collect_parameter_data, analyze_metric_percentiles
from inverse_design.analyze.parameter_config import PARAMETER_LIST


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


def plot_metric_distributions(
    posterior_metrics_file,
    prior_metrics_file,
    target_metrics,
    save_file: Optional[str] = None,
):
    """
    Plot density distributions of metrics from posterior and/or prior, with optional target values.

    Args:
        posterior_metrics_file: Path to simulation_metrics.csv (posterior)
        prior_metrics_file: Path to completed_doubling.csv (prior)
        target_metrics: Dictionary of target values for each metric
        save_file: Path to save the plot, optional
    """
    # Explicitly type the variables
    posterior_df = pd.read_csv(posterior_metrics_file)
    prior_df = pd.read_csv(prior_metrics_file)
    metrics_list = list(target_metrics.keys())

    # Create subplot for each metric
    fig, axes = plt.subplots(1, len(metrics_list), figsize=(15, 5))
    if len(metrics_list) == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics_list):
        ax = axes[idx]
        prior_kde = sns.kdeplot(data=prior_df[metric], ax=ax, label="Prior", color="gray")
        ax.set_ylabel("Prior Density", color="gray")
        ax.tick_params(axis="y", labelcolor="gray")
        ax2 = ax.twinx()
        posterior_kde = sns.kdeplot(data=posterior_df[metric], ax=ax2, label="Posterior", color="blue")
        ax2.set_ylabel("Posterior Density", color="blue")
        ax2.tick_params(axis="y", labelcolor="blue")

        prior_x_min, prior_x_max = get_kde_bounds(prior_kde, percentile=0.01)
        posterior_x_min, posterior_x_max = get_kde_bounds(posterior_kde, percentile=0.01)
        x_min = min(prior_x_min, posterior_x_min)
        x_max = max(prior_x_max, posterior_x_max)
            
        padding = 0.1 * (x_max - x_min)
        ax.set_xlim(x_min - padding, x_max + padding)

        ax.axvline(
            x=target_metrics[metric],
            color="red",
            linestyle="--",
            label=f"Target ({target_metrics[metric]})",
        )
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2)

        ax.set_title(f"{metric} Distribution")
        ax.set_xlabel(metric)

    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file)
    else:
        plt.show()
    plt.close()


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


if __name__ == "__main__":
    # Specify your parameters
    parameter_base_folder = "ARCADE_OUTPUT/SENSITIVITY/symmetry"
    input_folder = parameter_base_folder + "/inputs"
    csv_file = f"{parameter_base_folder}/final_metrics.csv"

    metrics_name = "n_cells"
    metrics_df = pd.read_csv(csv_file)
    if 1:
        posterior_metrics_file = csv_file
        prior_metrics_file = (
            "ARCADE_OUTPUT/STEM_CELL_META_SIGNAL_HETEROGENEITY/final_metrics.csv"
        )
        target_metrics = {
            "symmetry": 0.8,
            "cycle_length": 30.0,
            "act": 0.6,
        }

        save_file = f"{parameter_base_folder}/metric_distributions.png"
        plot_metric_distributions(
            posterior_metrics_file=posterior_metrics_file,
            prior_metrics_file=prior_metrics_file,
            target_metrics=target_metrics,
            save_file=save_file,
        )

    percentile = 10
    top_n_input_file, bottom_n_input_file, labeled_metrics_df = analyze_metric_percentiles(
        csv_file, metrics_name, percentile, verbose=True
    )

    # Create labeled parameter DataFrame
    input_files = list(top_n_input_file) + list(bottom_n_input_file)

    labels = ["top"] * len(top_n_input_file) + ["bottom"] * len(bottom_n_input_file)
    analyzed_param_df = collect_parameter_data(
        input_files, input_folder, PARAMETER_LIST, labels
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

    if 1:
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

    if 1:
        save_file = f"{parameter_base_folder}/parameter_distributions_{metrics_name}.png"
        plot_top_bottom_parameter_distributions(
            analyzed_param_df, PARAMETER_LIST, parameter_base_folder, percentile, save_file
        )
    if 1:
        save_file = f"{parameter_base_folder}/pca_parameters_{metrics_name}.png"
        plot_pca_parameters(
            analyzed_param_df,
        PARAMETER_LIST,
        parameter_base_folder,
        metrics_name,
        percentile,
        save_file,
    )

    if 1:
        plot_cell_states_histogram(
            csv_file,
            f"{parameter_base_folder}/cell_states_histogram.png",
        )
