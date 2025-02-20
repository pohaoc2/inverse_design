import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



def analyze_metric_percentiles(csv_file_path, metrics_name: str, percentile: float = 10, verbose: bool = True):
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
    lower_bound = df[metrics_name].quantile(percentile/100)
    upper_bound = df[metrics_name].quantile(1 - percentile/100)
    
    # Create a copy of the dataframe and add labels
    all_data = df.copy()
    all_data['percentile_label'] = 'not_assigned'
    all_data.loc[df[metrics_name] <= lower_bound, 'percentile_label'] = 'bottom'
    all_data.loc[df[metrics_name] >= upper_bound, 'percentile_label'] = 'top'
    
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
        print(all_data['percentile_label'].value_counts())

    return top_n_input_file, bottom_n_input_file, all_data

def get_parameters_from_json(input_folder, parameter_list):
    """Extract parameters from json file."""
    json_file = Path(input_folder) / "uniform_kde_aggressive.json"
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    params = {}
    param_mapping = {
        "CELL_VOLUME_SIGMA": ("populations", "cancerous", "CELL_VOLUME_SIGMA"),
        "APOPTOSIS_AGE_SIGMA": ("populations", "cancerous", "APOPTOSIS_AGE_SIGMA"),
        "NECROTIC_FRACTION": ("populations", "cancerous", "NECROTIC_FRACTION"),
        "ACCURACY": ("populations", "cancerous", "ACCURACY"),
        "AFFINITY": ("populations", "cancerous", "AFFINITY"),
        "COMPRESSION_TOLERANCE": ("populations", "cancerous", "COMPRESSION_TOLERANCE")
    }
    
    for param in parameter_list:
        path = param_mapping[param]
        value = data
        for key in path:
            value = value[key]
        params[param] = value
    
    return params

def collect_parameter_data(input_files, parameter_base_folder, parameter_list):
    """
    Collect parameter data from JSON files for a list of input folders.
    
    Args:
        input_files: List of input folder names
        parameter_base_folder: Base folder containing parameter files
        parameter_list: List of parameters to extract
    
    Returns:
        pandas.DataFrame: DataFrame containing the parameter values
    """
    params_list = []
    
    for input_folder in input_files:
        full_path = Path(parameter_base_folder) / input_folder
        params = get_parameters_from_json(full_path, parameter_list)
        params_list.append(params)
    
    return pd.DataFrame(params_list)

def analyze_and_plot_parameters(top_df, bottom_df, parameter_list, parameter_base_folder, percentile: float = 10):
    """
    Analyze and plot parameter distributions for top and bottom percentile cases.
    """
    
    n_params = len(parameter_list)
    fig, axes = plt.subplots(1, n_params, figsize=(4*n_params, 4))
    
    for idx, param in enumerate(parameter_list):
        ax = axes[idx]
        
        # Plot density distributions
        sns.kdeplot(data=top_df[param], ax=ax, label=f'Top {percentile}%', color='blue')
        sns.kdeplot(data=bottom_df[param], ax=ax, label=f'Bottom {percentile}%', color='red')
        
        ax.set_title(f'{param} Distribution')
        ax.set_xlabel(param)
        ax.set_ylabel('Density')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{parameter_base_folder}/parameter_distributions.png')
    plt.show()
    
    # Print summary statistics
    print("\nParameter Statistics:")
    print("=" * 80)
    print(f"\nTop {percentile}% - Mean values:")
    print(top_df.mean())
    print(f"\nBottom {percentile}% - Mean values:")
    print(bottom_df.mean())

def analyze_pca_parameters(top_df, bottom_df, parameter_list, parameter_base_folder, metrics_name: str, percentile: float = 10):
    """
    Perform PCA on parameters and visualize top and bottom percentile cases in 2D.
    
    Args:
        csv_file_path: Path to the aggregated results CSV file
        parameter_list: List of parameters to analyze
        parameter_base_folder: Base folder containing parameter files
        metrics_names: List of metrics used to determine top/bottom percentile
        percentile: Percentile value between 0 and 50 (default: 10)
    """
    
    
    # Combine data for PCA
    all_data = pd.concat([top_df, bottom_df])
    
    # Create labels (1 for top 10%, 0 for bottom 10%)
    labels = np.array([1] * len(top_df) + [0] * len(bottom_df))
    
    # Standardize the features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(all_data)
    
    # Perform PCA
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot points
    plt.scatter(data_pca[labels==1, 0], data_pca[labels==1, 1], 
               c='blue', label=f'Top {percentile}%', alpha=0.6)
    plt.scatter(data_pca[labels==0, 0], data_pca[labels==0, 1], 
               c='red', label=f'Bottom {percentile}%', alpha=0.6)
    
    # Add parameter vectors
    for i, param in enumerate(parameter_list):
        plt.arrow(0, 0,
                 pca.components_[0, i] * 3,
                 pca.components_[1, i] * 3,
                 color='black', alpha=0.5)
        plt.text(pca.components_[0, i] * 3.2,
                pca.components_[1, i] * 3.2,
                param,
                color='black',
                ha='center',
                va='center')
    
    # Add plot details
    plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title(f'PCA of Parameters for Top and Bottom {percentile}% Cases')
    plt.legend()
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add origin lines
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{parameter_base_folder}/pca_parameters.png')
    plt.show()
    
    # Print explained variance ratios
    print("\nPCA Explained Variance Ratios:")
    print("=" * 80)
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {ratio:.1%}")

    # Create a DataFrame to make it easier to read
    loadings_df = pd.DataFrame(pca.components_, 
                            index=[f"PC{i+1}" for i in range(len(pca.components_))], 
                            columns=parameter_list)

    # Show the contribution of each parameter in PC1
    print("Contribution of each parameter to PC1:")
    print(loadings_df.loc["PC1"])
    print(loadings_df.loc["PC2"])

def plot_doubling_time_relationship(labeled_df, parameter_base_folder):
    """
    Plot the relationship between doubling time and its standard deviation.
    
    Args:
        labeled_df: DataFrame containing labeled data
        parameter_base_folder: Base folder for saving plots
    """
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Define plot settings for each group
    plot_settings = {
        'top': {'color': 'blue', 'marker': 'o'},
        'bottom': {'color': 'red', 'marker': 's'}
    }
    
    # Plot data and fit lines for each group
    for group, settings in plot_settings.items():
        if group == 'top':
            plot_settings[group]['corr'] = 0
            continue
        group_data = labeled_df[labeled_df['percentile_label'] == group]
        
        # Scatter plot
        scatter = plt.scatter(group_data['doub_time'], 
                            group_data['doub_time_std'], 
                            alpha=0.6, 
                            c=group_data['colony_g_rate'], 
                            cmap='viridis',
                            marker=settings['marker'],
                            label=f'{group.capitalize()} percentile')
        
        # Fit line
        z = np.polyfit(group_data['doub_time'], group_data['doub_time_std'], 1)
        p = np.poly1d(z)
        plt.plot(group_data['doub_time'], 
                p(group_data['doub_time']), 
                f"--",
                color=settings['color'],
                alpha=0.8, 
                label=f'{group.capitalize()} fit (slope: {z[0]:.2f})')
        
        # Calculate correlation coefficient
        corr = group_data['doub_time'].corr(group_data['doub_time_std'])
        plot_settings[group]['corr'] = corr
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Colony Growth Rate', rotation=270, labelpad=15)
    
    # Add labels and title
    plt.xlabel('Doubling Time')
    plt.ylabel('Doubling Time Std')
    plt.title('Relationship between Doubling Time and its Standard Deviation\n' + 
             f'Correlation coefficients: Top={plot_settings["top"]["corr"]:.2f}, ' + 
             f'Bottom={plot_settings["bottom"]["corr"]:.2f}')
    plt.legend()
    plt.savefig(f'{parameter_base_folder}/doubling_time_relationship.png')
def plot_metric_distributions(posterior_metrics_file, prior_metrics_file, target_metrics, save_file=None):
    """
    Plot density distributions of metrics from posterior and prior, with target values.
    
    Args:
        posterior_metrics_file: Path to simulation_metrics.csv (posterior)
        prior_metrics_file: Path to completed_doubling.csv (prior)
        target_metrics: Dictionary of target values for each metric
    """
    # Load data
    posterior_df = pd.read_csv(posterior_metrics_file)
    prior_df = pd.read_csv(prior_metrics_file)
    
    metrics_list = list(target_metrics.keys())
    
    # Create subplot for each metric
    fig, axes = plt.subplots(1, len(metrics_list), figsize=(15, 5))
    
    for idx, metric in enumerate(metrics_list):
        # Plot posterior distribution
        sns.kdeplot(data=posterior_df[metric], ax=axes[idx], label='Posterior', color='blue')
        
        # Plot prior distribution
        sns.kdeplot(data=prior_df[metric], ax=axes[idx], label='Prior', color='gray')
        
        # Add vertical line for target value
        axes[idx].axvline(x=target_metrics[metric], color='red', linestyle='--', 
                         label=f'Target ({target_metrics[metric]})')
        
        axes[idx].set_title(f'{metric} Distribution')
        axes[idx].set_xlabel(metric)
        axes[idx].set_ylabel('Density')
        axes[idx].legend()
    
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file)
    else:
        plt.show()


if __name__ == "__main__":
    # Specify your parameters
    csv_file = "ARCADE_OUTPUT/STEM_CELL/simulation_metrics.csv"
    metrics_name = "doub_time_std"
    parameter_list = ["CELL_VOLUME_SIGMA", "NECROTIC_FRACTION", 
                     "ACCURACY", "AFFINITY", "COMPRESSION_TOLERANCE"]
    parameter_base_folder = "ARCADE_OUTPUT/STEM_CELL"
    percentile = 30
    top_n_input_file, bottom_n_input_file, labeled_df = analyze_metric_percentiles(csv_file, metrics_name, percentile, verbose=False)
    top_param_df = collect_parameter_data(top_n_input_file, parameter_base_folder, parameter_list)
    bottom_param_df = collect_parameter_data(bottom_n_input_file, parameter_base_folder, parameter_list)
    # Run analyses
    analyze_and_plot_parameters(top_param_df, bottom_param_df, parameter_list, parameter_base_folder, percentile)
    analyze_pca_parameters(top_param_df, bottom_param_df, parameter_list, parameter_base_folder, metrics_name, percentile)
    
    # Plot doubling time relationship
    plot_doubling_time_relationship(labeled_df, parameter_base_folder)

    posterior_metrics_file = "ARCADE_OUTPUT/STEM_CELL/simulation_metrics.csv"
    prior_metrics_file = "prior_metrics_formatted.csv"
    target_metrics = {
        "doub_time": 50,
        "doub_time_std": 0.0,
        "act_t2": 0.5,
        #"colony_g_rate": 0.8
    }
    save_file = "stem_cell_metric_distributions.png"
    plot_metric_distributions(posterior_metrics_file, prior_metrics_file, target_metrics, save_file)