import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def analyze_metric_percentiles(
    csv_file_path, metrics_name: str, percentile: float = 10, verbose: bool = True
):
    """
    Analyze rows where metrics fall into the bottom and top percentiles of their distributions.

    Args:
        csv_file_path: Path to the aggregated results CSV file
        metrics_name: Name of the metric to analyze
        percentile: Percentile value between 0 and 50 (default: 10)
        verbose: Whether to print detailed analysis (default: True)

    Returns:
        tuple: (top_n_input_file, bottom_n_input_file, labeled_df)

    Raises:
        ValueError: If percentile is not between 0 and 50
    """
    if not 0 < percentile <= 50:
        raise ValueError("Percentile must be between 0 and 50")

    df = pd.read_csv(csv_file_path)

    # Calculate lower and upper percentiles
    lower_bound = df[metrics_name].quantile(percentile / 100)
    upper_bound = df[metrics_name].quantile(1 - percentile / 100)

    # Create a copy of the dataframe and add labels
    all_data = df.copy()
    all_data["percentile_label"] = "not_assigned"
    all_data.loc[df[metrics_name] <= lower_bound, "percentile_label"] = "bottom"
    all_data.loc[df[metrics_name] >= upper_bound, "percentile_label"] = "top"

    # Get the original outputs for backward compatibility
    bottom_n = df[df[metrics_name] <= lower_bound]
    top_n = df[df[metrics_name] >= upper_bound]
    top_n_input_file = top_n["input_folder"].unique()
    bottom_n_input_file = bottom_n["input_folder"].unique()

    if verbose:
        print(f"\nAnalysis for {metrics_name}")
        print("=" * 80)
        print(f"\nBottom {percentile}% (≤ {lower_bound:.3f}):")
        print(bottom_n)
        print(f"\nTop {percentile}% (≥ {upper_bound:.3f}):")
        print(top_n)
        print("\nLabel distribution:")
        print(all_data["percentile_label"].value_counts())

    return top_n_input_file, bottom_n_input_file, all_data


def get_parameters_from_json(input_folder, parameter_list):
    """Extract parameters from json file."""
    # Find the base configuration file by looking for a .json file without CELL or LOCATION in the name
    json_files = list(Path(input_folder).glob("*.json"))
    config_file = next(f for f in json_files if "CELL" not in f.name and "LOCATION" not in f.name)

    with open(config_file, "r") as f:
        data = json.load(f)

    params = {}
    param_mapping = {
        "CELL_VOLUME_SIGMA": ("populations", "cancerous", "CELL_VOLUME_SIGMA"),
        "APOPTOSIS_AGE_SIGMA": ("populations", "cancerous", "APOPTOSIS_AGE_SIGMA"),
        "NECROTIC_FRACTION": ("populations", "cancerous", "NECROTIC_FRACTION"),
        "ACCURACY": ("populations", "cancerous", "ACCURACY"),
        "AFFINITY": ("populations", "cancerous", "AFFINITY"),
        "COMPRESSION_TOLERANCE": ("populations", "cancerous", "COMPRESSION_TOLERANCE"),
    }

    for param in parameter_list:
        path = param_mapping[param]
        value = data
        for key in path:
            value = value[key]

        params[param] = value

    return params


def collect_parameter_data(input_files, parameter_base_folder, parameter_list, labels=None):
    """
    Collect parameter data from JSON files for a list of input folders.

    Args:
        input_files: List of input folder names
        parameter_base_folder: Base folder containing parameter files
        parameter_list: List of parameters to extract
        labels: Optional list of labels corresponding to input_files

    Returns:
        pandas.DataFrame: DataFrame containing the parameter values, labels, and folder names
    """
    params_list = []

    for i, input_folder in enumerate(input_files):
        full_path = Path(parameter_base_folder) / input_folder
        params = get_parameters_from_json(full_path, parameter_list)
        # Add folder name to params
        params["input_folder"] = input_folder
        if labels is not None:
            params["percentile_label"] = labels[i]
        params_list.append(params)

    return pd.DataFrame(params_list)


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
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))

    for idx, param in enumerate(parameter_list):
        ax = axes[idx]

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
    plt.title(f"PCA of Parameters for Top and Bottom {percentile}% Cases")
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


def plot_doubling_time_relationship(labeled_df, save_file=None):
    """
    Plot the relationship between doubling time and its standard deviation.

    Args:
        labeled_df: DataFrame containing labeled data
        save_file: Path to save the plot
    """
    # Create figure
    plt.figure(figsize=(10, 8))

    # Define plot settings for each group
    plot_settings = {
        "top": {"color": "blue", "marker": "o"},
        "bottom": {"color": "red", "marker": "s"},
    }

    # Plot data and fit lines for each group
    for group, settings in plot_settings.items():
        if group == "top":
            plot_settings[group]["corr"] = 0
            continue
        group_data = labeled_df[labeled_df["percentile_label"] == group]

        # Scatter plot
        scatter = plt.scatter(
            group_data["doub_time"],
            group_data["doub_time_std"],
            alpha=0.6,
            c=group_data["colony_g_rate"],
            cmap="viridis",
            marker=settings["marker"],
            label=f"{group.capitalize()} percentile",
        )

        # Fit line
        z = np.polyfit(group_data["doub_time"], group_data["doub_time_std"], 1)
        p = np.poly1d(z)
        plt.plot(
            group_data["doub_time"],
            p(group_data["doub_time"]),
            f"--",
            color=settings["color"],
            alpha=0.8,
            label=f"{group.capitalize()} fit (slope: {z[0]:.2f})",
        )

        # Calculate correlation coefficient
        corr = group_data["doub_time"].corr(group_data["doub_time_std"])
        plot_settings[group]["corr"] = corr

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Colony Growth Rate", rotation=270, labelpad=15)

    # Add labels and title
    plt.xlabel("Doubling Time")
    plt.ylabel("Doubling Time Std")
    plt.title(
        "Relationship between Doubling Time and its Standard Deviation\n"
        + f'Correlation coefficients: Top={plot_settings["top"]["corr"]:.2f}, '
        + f'Bottom={plot_settings["bottom"]["corr"]:.2f}'
    )
    plt.legend()
    if save_file is not None:
        plt.savefig(save_file)
    else:
        plt.show()
    plt.close()


def plot_metric_distributions(
    posterior_metrics_file: Optional[str] = None,
    prior_metrics_file: Optional[str] = None,
    target_metrics: Optional[dict] = None,
    save_file: Optional[str] = None,
):
    """
    Plot density distributions of metrics from posterior and/or prior, with optional target values.

    Args:
        posterior_metrics_file: Path to simulation_metrics.csv (posterior), optional
        prior_metrics_file: Path to completed_doubling.csv (prior), optional
        target_metrics: Dictionary of target values for each metric, optional
        save_file: Path to save the plot, optional
    """
    # Explicitly type the variables
    posterior_df: Optional[pd.DataFrame] = (
        pd.read_csv(posterior_metrics_file) if posterior_metrics_file else None
    )
    prior_df: Optional[pd.DataFrame] = (
        pd.read_csv(prior_metrics_file) if prior_metrics_file else None
    )

    # Determine metrics to plot
    if target_metrics:
        metrics_list = list(target_metrics.keys())
    elif posterior_df is not None:
        # Only include numeric columns
        metrics_list = posterior_df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    elif prior_df is not None:
        # Only include numeric columns
        metrics_list = prior_df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    else:
        raise ValueError(
            "At least one of posterior_metrics_file, prior_metrics_file, or target_metrics must be provided"
        )

    # Create subplot for each metric
    fig, axes = plt.subplots(1, len(metrics_list), figsize=(15, 5))
    if len(metrics_list) == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics_list):
        ax = axes[idx]

        # Plot prior distribution if available
        if (
            prior_df is not None
            and metric in prior_df.columns
            and pd.api.types.is_numeric_dtype(prior_df[metric])
        ):
            sns.kdeplot(data=prior_df[metric], ax=ax, label="Prior", color="gray")
            ax.set_ylabel("Prior Density", color="gray")
            ax.tick_params(axis="y", labelcolor="gray")

            # If both distributions are present, use twin y-axis for posterior
            if (
                posterior_df is not None
                and metric in posterior_df.columns
                and pd.api.types.is_numeric_dtype(posterior_df[metric])
            ):
                ax2 = ax.twinx()
                sns.kdeplot(data=posterior_df[metric], ax=ax2, label="Posterior", color="blue")
                ax2.set_ylabel("Posterior Density", color="blue")
                ax2.tick_params(axis="y", labelcolor="blue")

                # Combine legends
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2)
            else:
                ax.legend()
        # If only posterior is available, plot it normally
        elif (
            posterior_df is not None
            and metric in posterior_df.columns
            and pd.api.types.is_numeric_dtype(posterior_df[metric])
        ):
            sns.kdeplot(data=posterior_df[metric], ax=ax, label="Posterior", color="blue")
            ax.set_ylabel("Density")
            ax.legend()

        # Add target value if available
        if target_metrics and metric in target_metrics:
            ax.axvline(
                x=target_metrics[metric],
                color="red",
                linestyle="--",
                label=f"Target ({target_metrics[metric]})",
            )
            # Update legends to include target line
            if prior_df is not None and posterior_df is not None:
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2)
            else:
                ax.legend()

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
        # Sum up counts across all seeds
        seed_sums = {state: 0 for state in state_counts.keys()}
        for seed_data in states_list:
            for state, count in seed_data.items():
                seed_sums[state] += count

        # Calculate average across seeds
        n_seeds = len(states_list)
        for state in state_counts:
            state_counts[state].append(seed_sums[state] / n_seeds)

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


if __name__ == "__main__":
    # Specify your parameters
    csv_file = "ARCADE_OUTPUT/STEM_CELL/STEM_CELL_VARY_VOLUME_POSTERIOR/simulation_metrics.csv"
    metrics_name = "doub_time_std"
    parameter_list = [
        "CELL_VOLUME_SIGMA",
        # "NECROTIC_FRACTION",
        # "APOPTOSIS_AGE_SIGMA",
        "ACCURACY",
        "AFFINITY",
        "COMPRESSION_TOLERANCE",
    ]
    parameter_base_folder = "ARCADE_OUTPUT/STEM_CELL/STEM_CELL_VARY_VOLUME_POSTERIOR"
    percentile = 30
    top_n_input_file, bottom_n_input_file, labeled_df = analyze_metric_percentiles(
        csv_file, metrics_name, percentile, verbose=True
    )

    # Create labeled parameter DataFrame
    input_files = list(top_n_input_file) + list(bottom_n_input_file)
    labels = ["top"] * len(top_n_input_file) + ["bottom"] * len(bottom_n_input_file)
    analyzed_param_df = collect_parameter_data(
        input_files, parameter_base_folder, parameter_list, labels
    )
    analyzed_param_df = analyzed_param_df.sort_values(
        "input_folder",
        key=lambda x: x.str.extract("input_(\d+)").iloc[:, 0].astype(int),
    )

    # Run analyses with the combined DataFrame
    save_file = f"{parameter_base_folder}/parameter_distributions.png"
    plot_top_bottom_parameter_distributions(
        analyzed_param_df, parameter_list, parameter_base_folder, percentile, save_file
    )
    save_file = f"{parameter_base_folder}/pca_parameters.png"
    plot_pca_parameters(
        analyzed_param_df,
        parameter_list,
        parameter_base_folder,
        metrics_name,
        percentile,
        save_file,
    )

    # Plot doubling time relationship
    # plot_doubling_time_relationship(labeled_df, parameter_base_folder)
    plot_cell_states_histogram(
        f"{parameter_base_folder}/simulation_metrics.csv",
        f"{parameter_base_folder}/cell_states_histogram.png",
    )
    if 1:
        posterior_metrics_file = (
            "ARCADE_OUTPUT/STEM_CELL/STEM_CELL_VARY_VOLUME_POSTERIOR/simulation_metrics.csv"
        )
        prior_metrics_file = (
            "ARCADE_OUTPUT/STEM_CELL/STEM_CELL_VARY_VOLUME_PRIOR/simulation_metrics.csv"
        )
        target_metrics = {
            "doub_time": 35.0,
            "doub_time_std": 10.0,
            "act_t2": 0.6,
            # "colony_g_rate": 0.8
        }
        save_file = f"{parameter_base_folder}/metric_distributions.png"
        plot_metric_distributions(
            posterior_metrics_file=posterior_metrics_file,
            prior_metrics_file=prior_metrics_file,
            target_metrics=target_metrics,
            save_file=save_file,
        )
